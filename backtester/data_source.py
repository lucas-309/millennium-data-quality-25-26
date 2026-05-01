from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
import os
import pickle
import time

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

class DataSource(ABC):
    """Interface for fetching historical market data."""
    
    @abstractmethod
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for given tickers and date range."""
        pass

class PickleDataSource(DataSource):
    """Implementation of DataSource that reads from a local pickle file."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cache file not found at {file_path}. Please run cache_sp500_data.py first.")
        
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        result = pd.DataFrame()
        
        # Ensure dates are timestamps for comparison
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        for ticker in tickers:
            if ticker in self.data:
                ticker_df = self.data[ticker]
                # Filter by date
                mask = (ticker_df.index >= start_ts) & (ticker_df.index <= end_ts)
                filtered_data = ticker_df.loc[mask, 'Adj Close']
                result[ticker] = filtered_data
            else:
                print(f"Warning: Ticker {ticker} not found in cache.")
        
        return result


class WhartonDataSource(DataSource):
    """
    Implementation of DataSource using Wharton WRDS data from Parquet/Excel.
    
    Expected columns:
    - tic: Ticker Symbol
    - datadate: Data Date
    - prccd: Price Close Daily
    - ajexdi: Adjustment Factor (cumulative by ex-date)
    - cshtrd: Trading Volume Daily
    - trfd: Daily Total Return Factor
    - divd: Cash Dividends - Daily
    
    Optional: merge earnings surprise events (event dates + SUE) onto each daily row via
    backward ``merge_asof`` so each trading day carries the latest known announcement on or
    before that date (see ``include_surprise``).
    """
    
    def __init__(
        self,
        file_path: Optional[str] = None,
        use_trfd: bool = False,
        cache_parquet: bool = True,
        include_surprise: bool = True,
        surprise_path: Optional[str] = None,
        surprise_measure: str = "EPS",
    ):
        self.file_path = file_path
        self.use_trfd = use_trfd
        self.cache_parquet = cache_parquet
        self.include_surprise = include_surprise
        self.surprise_path = surprise_path
        self.surprise_measure = surprise_measure
        self.data = None
        self.resolved_path: Optional[Path] = None
        self._load_data()
    
    def _load_data(self):
        resolved = self._resolve_source_path()
        self.resolved_path = resolved

        # Load file
        if str(resolved).endswith('.xlsx') or str(resolved).endswith('.xls'):
            self.data = pd.read_excel(resolved)
            self._cache_to_parquet(resolved)
        elif str(resolved).endswith('.csv'):
            self.data = pd.read_csv(resolved, low_memory=False)
            self._cache_to_parquet(resolved)
        elif str(resolved).endswith('.parquet'):
            self.data = pd.read_parquet(resolved)
        else:
            raise TypeError("Data file format is not supported, please use .csv, .xlsx, or .parquet")
        
        # Ensure required columns exist
        required_cols = ['tic', 'datadate']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column
        self.data['datadate'] = pd.to_datetime(self.data['datadate'])
        
        # Calculate split-adjusted close first
        if 'prccd' in self.data.columns and 'ajexdi' in self.data.columns:
            # FIX: prccd / ajexdi to correctly split-adjust
            self.data['Split Adj Close'] = self.data['prccd'] / self.data['ajexdi']
        else:
            if not self.use_trfd:
                raise ValueError("prccd and ajexdi columns required when use_trfd=False")
        
        # Calculate fully adjusted close price using Total Return Factor
        if self.use_trfd:
            if 'trfd' not in self.data.columns:
                raise ValueError("trfd column not found but use_trfd=True")
            
            # Fill NA trfd with 1.0 (preserves neutral return for days with missing data)
            self.data['trfd'] = self.data['trfd'].fillna(1.0)
            
            # Filter out NaNs to ensure we anchor to a valid price
            valid_mask = self.data['Split Adj Close'].notna()
            valid_data = self.data[valid_mask]
            
            if not valid_data.empty:
                # Find the last valid row per ticker to anchor the prices backwards
                last_valid = valid_data.groupby('tic').tail(1).set_index('tic')
                
                # Map the final anchor trfd back to every row of that ticker
                mapped_last_trfd = self.data['tic'].map(last_valid['trfd'])
                
                # Formula: Adj Close_t = Split Adj Close_t * (trfd_t / trfd_Anchor_T)
                # This perfectly mimics Yahoo Finance: today's price matches raw Split Close, 
                # but historical prices are deflated backward to account for cumulative dividend yields.
                self.data['Adj Close'] = self.data['Split Adj Close'] * (self.data['trfd'] / mapped_last_trfd)
            else:
                self.data['Adj Close'] = self.data['Split Adj Close']
                
            # Fill any remaining NaNs with the standard Split Adj Close
            self.data['Adj Close'] = self.data['Adj Close'].fillna(self.data['Split Adj Close'])
        else:
            self.data['Adj Close'] = self.data['Split Adj Close']
            
        # Add volume if available
        if 'cshtrd' in self.data.columns:
            self.data['Volume'] = self.data['cshtrd']
            
        # Expose cash dividends independently if not using fully adjusted trfd
        if not self.use_trfd and 'divd' in self.data.columns:
            self.data['Dividend'] = self.data['divd'].fillna(0.0)
        
        print(
            f"WhartonDataSource loaded {len(self.data)} records for "
            f"{self.data['tic'].nunique()} tickers from {self.resolved_path}"
        )
        self._merge_surprise_if_needed()

    def _normalize_tic_for_surprise(self, s: pd.Series) -> pd.Series:
        t = s.astype(str).str.strip().str.upper()
        alias = {
            "BRK-B": "BRK.B",
            "BF-B": "BF.B",
            "FI": "FISV",
            "PARA": "PARAA",
            "-": "USD",
        }
        t = t.replace(alias)
        return t.str.replace("-", ".", regex=False)

    def _merge_surprise_if_needed(self) -> None:
        if not self.include_surprise:
            return

        module_dir = Path(__file__).resolve().parent
        if self.surprise_path:
            path = Path(self.surprise_path)
            if not path.is_absolute():
                path = module_dir / path
        else:
            # Prefer quarterly surprise file; keep legacy fallback for compatibility.
            preferred = module_dir / "SUE_quarterly.csv"
            legacy = module_dir / "surprise earning.csv"
            path = preferred if preferred.exists() else legacy

        if not path.exists():
            print(f"Warning: surprise file not found at {path}, skipping surprise merge.")
            return

        try:
            surp = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"Warning: could not read surprise file {path}: {exc}")
            return

        if "TICKER" not in surp.columns or "anndats" not in surp.columns:
            print("Warning: surprise file missing TICKER or anndats; skipping merge.")
            return

        if self.surprise_measure and "MEASURE" in surp.columns:
            surp = surp[surp["MEASURE"].astype(str).str.upper() == str(self.surprise_measure).upper()].copy()
        if surp.empty:
            print(f"Warning: no surprise rows after MEASURE filter ({self.surprise_measure}); skipping merge.")
            return

        surp["tic"] = self._normalize_tic_for_surprise(surp["TICKER"])
        surp["surprise_ann_date"] = pd.to_datetime(surp["anndats"], errors="coerce")
        surp = surp.dropna(subset=["tic", "surprise_ann_date"])
        surp = surp.sort_values(["tic", "surprise_ann_date"])

        keep_cols = ["tic", "surprise_ann_date"]
        for c in ("suescore", "actual", "surpmean", "surpstdev"):
            if c in surp.columns:
                keep_cols.append(c)
        surp = surp[keep_cols].drop_duplicates(subset=["tic", "surprise_ann_date"], keep="last")
        right = surp.rename(columns={"surprise_ann_date": "surprise_ann_date_right"})

        left = self.data.copy()
        left["tic"] = left["tic"].astype(str).str.strip().str.upper()

        # merge_asof with by= requires strictly sorted blocks; a global sort can still fail
        # pandas' check on some versions, so merge per ticker.
        merged_chunks: List[pd.DataFrame] = []
        value_cols = [c for c in right.columns if c not in ("tic", "surprise_ann_date_right")]
        for tic_key, l_chunk in left.groupby("tic", sort=False):
            l_chunk = l_chunk.sort_values("datadate")
            r_chunk = right.loc[right["tic"] == tic_key, ["surprise_ann_date_right"] + value_cols].sort_values(
                "surprise_ann_date_right"
            )
            if r_chunk.empty:
                empty = l_chunk.copy()
                empty["surprise_ann_date"] = pd.NaT
                for c in value_cols:
                    empty[c] = np.nan
                merged_chunks.append(empty)
                continue
            m = pd.merge_asof(
                l_chunk,
                r_chunk,
                left_on="datadate",
                right_on="surprise_ann_date_right",
                direction="backward",
                allow_exact_matches=True,
            )
            m = m.rename(columns={"surprise_ann_date_right": "surprise_ann_date"})
            merged_chunks.append(m)

        merged = pd.concat(merged_chunks, ignore_index=True)

        if "suescore" in merged.columns:
            merged["last_sue"] = merged["suescore"]
        else:
            merged["last_sue"] = np.nan

        ann = merged["surprise_ann_date"]
        dd = merged["datadate"]
        merged["days_since_announcement"] = (dd.dt.normalize() - ann.dt.normalize()).dt.days
        merged["is_event_day"] = (dd.dt.normalize() == ann.dt.normalize()).astype(np.int8)

        self.data = merged
        n_with = merged["last_sue"].notna().sum()
        print(f"Merged surprise (measure={self.surprise_measure}): {n_with} / {len(merged)} rows with last_sue.")

    def _resolve_source_path(self) -> Path:
        """
        Resolve the data source path with sensible defaults.
        Priority:
        1) Explicit file_path if provided.
        2) WhartonDataSource4.parquet
        3) WhartonDataSource4.csv
        """
        module_dir = Path(__file__).resolve().parent

        if self.file_path:
            candidate = Path(self.file_path)
            if not candidate.is_absolute():
                candidate = module_dir / candidate
            if not candidate.exists():
                raise FileNotFoundError(f"Data file not found at {candidate}")
            return candidate

        default_parquet = module_dir / "WhartonDataSource4.parquet"
        default_csv = module_dir / "WhartonDataSource4.csv"

        if default_parquet.exists():
            return default_parquet
        if default_csv.exists():
            return default_csv

        raise FileNotFoundError(
            "No default Wharton source found. Expected one of: "
            f"{default_parquet} or {default_csv}"
        )

    def _cache_to_parquet(self, source_path: Path) -> None:
        """Cache non-parquet source files to parquet for faster future loads."""
        if not self.cache_parquet:
            return

        parquet_path = source_path.with_suffix(".parquet")
        if parquet_path.exists():
            return

        try:
            self.data.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
            print(f"Cached parquet snapshot to {parquet_path}")
        except Exception as exc:
            # Non-fatal: caller can still proceed with in-memory dataframe.
            print(f"Warning: failed to cache parquet to {parquet_path}: {exc}")
    
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        mask = (
            (self.data['datadate'] >= start_ts) & 
            (self.data['datadate'] <= end_ts) & 
            (self.data['tic'].isin(tickers))
        )
        filtered = self.data.loc[mask, ['tic', 'datadate', 'Adj Close']].copy()
        if filtered.empty:
            return pd.DataFrame()
            
        result = filtered.pivot(index='datadate', columns='tic', values='Adj Close').sort_index()
        return result
        
    def get_historical_data_with_volume(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        result = {}
        for ticker in tickers:
            mask = (
                (self.data['datadate'] >= start_ts) & 
                (self.data['datadate'] <= end_ts) & 
                (self.data['tic'] == ticker)
            )
            cols = ['datadate', 'Adj Close']
            if 'Volume' in self.data.columns:
                cols.append('Volume')
            if 'Dividend' in self.data.columns:
                cols.append('Dividend')
                
            ticker_data = self.data.loc[mask, cols].copy()
            if ticker_data.empty:
                continue
            
            ticker_data = ticker_data.set_index('datadate').sort_index()
            result[ticker] = ticker_data
            
        return result

    def get_historical_data_with_factors(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        factor_cols: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return per-ticker timeseries including price and optional factor columns.

        By default, this exposes surprise-related factors when available:
        - last_sue
        - days_since_announcement
        - is_event_day
        """
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        if factor_cols is None:
            factor_cols = ["last_sue", "days_since_announcement", "is_event_day"]

        base_cols = ["datadate", "Adj Close"]
        selected_factors = [c for c in factor_cols if c in self.data.columns]
        selected_cols = base_cols + selected_factors

        result: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            mask = (
                (self.data["datadate"] >= start_ts)
                & (self.data["datadate"] <= end_ts)
                & (self.data["tic"] == ticker)
            )
            ticker_data = self.data.loc[mask, selected_cols].copy()
            if ticker_data.empty:
                continue
            ticker_data = ticker_data.set_index("datadate").sort_index()
            result[ticker] = ticker_data

        return result

# TODO: refactor implementations into sep. files, e.g. yahoo_finance_data_source.py
class YahooFinanceDataSource(DataSource):
    """Implementation of DataSource using Yahoo Finance. Queries historical price data, as well as compares weighted portfolios to SPY ETF."""

    def __init__(self, batch_size: int = 25, pause_seconds: float = 1.0, max_retries: int = 2):
        self.batch_size = max(batch_size, 1)
        self.pause_seconds = max(pause_seconds, 0.0)
        self.max_retries = max(max_retries, 0)

    def _extract_adjusted_close(self, downloaded: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        if downloaded.empty:
            return pd.DataFrame()

        adjusted_close = downloaded.get("Adj Close", pd.DataFrame())
        if isinstance(adjusted_close, pd.Series):
            column_name = tickers[0] if tickers else adjusted_close.name or "Adj Close"
            return adjusted_close.to_frame(name=column_name)
        return adjusted_close

    def _download_batch(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                downloaded = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    threads=False,
                    progress=False,
                )
                adjusted_close = self._extract_adjusted_close(downloaded, tickers)
                if not adjusted_close.empty:
                    return adjusted_close
            except Exception as exc:
                last_exception = exc

            if attempt < self.max_retries:
                time.sleep(self.pause_seconds * (attempt + 1))

        if len(tickers) == 1:
            if last_exception is not None:
                print(f"Warning: failed to download {tickers[0]}: {last_exception}")
            return pd.DataFrame()

        frames = []
        for ticker in tickers:
            try:
                downloaded = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    threads=False,
                    progress=False,
                )
                adjusted_close = self._extract_adjusted_close(downloaded, [ticker])
                if not adjusted_close.empty:
                    frames.append(adjusted_close)
            except Exception as exc:
                print(f"Warning: failed to download {ticker}: {exc}")
            time.sleep(max(self.pause_seconds / 2.0, 0.0))

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        return combined.loc[:, ~combined.columns.duplicated()]
    
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        requested = [ticker for ticker in tickers if ticker]
        if not requested:
            return pd.DataFrame()

        if len(requested) <= self.batch_size:
            return self._download_batch(requested, start_date, end_date)

        frames = []
        for start in range(0, len(requested), self.batch_size):
            batch = requested[start:start + self.batch_size]
            frame = self._download_batch(batch, start_date, end_date)
            if not frame.empty:
                frames.append(frame)
            time.sleep(self.pause_seconds)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined.sort_index().sort_index(axis=1)
    
    def get_historical_data_with_volume(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch historical price and volume data for given tickers and date range, organized by ticker. Function is DEPRECATED (remove)"""
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        result = {}
        for ticker in tickers:
            ticker_data = pd.DataFrame()
            ticker_data['Adj Close'] = data['Adj Close'][ticker] if isinstance(data['Adj Close'], pd.DataFrame) else data['Adj Close']
            ticker_data['Volume'] = data['Volume'][ticker] if isinstance(data['Volume'], pd.DataFrame) else data['Volume']
            result[ticker] = ticker_data
        
        return result
    
    def read_spy_holdings(self, file_path: str) -> pd.DataFrame:
        """
        Read SPY ETF holdings from State Street dailies .xlsx file.
        Expected format: Excel file with header rows followed by data rows
        containing 'Ticker' and 'Weight' columns
        """
        try:
            raw_df = pd.read_excel(file_path, header=None)
            ticker_row = None
            for i, row in raw_df.iterrows():
                if 'Ticker' in row.values:
                    ticker_row = i
                    break
            if ticker_row is None:
                print("Error: Could not find 'Ticker' column in the file")
                return pd.DataFrame()
            
            df = pd.read_excel(file_path, header=ticker_row)
            last_valid_row = df[df['Ticker'].notna()].index.max()
            df = df.loc[:last_valid_row].copy()
            df['Weight'] = df['Weight'] / 100.0
            df = df[['Ticker', 'Weight']]
            return df
        except Exception as e:
            print(f"Error reading SPY holdings file: {e}")
            return pd.DataFrame()
    
    def calculate_weighted_portfolio(self, holdings_df: pd.DataFrame, price_data: pd.DataFrame) -> pd.Series:
        """
        Calculate the weighted portfolio value from constituent stocks.
        
        Parameters:
        - holdings_df: DataFrame with 'Ticker' and 'Weight' columns
        - price_data: DataFrame with tickers as columns and dates as index
        
        Returns:
        - Series with weighted portfolio values
        """
        if price_data.empty:
            print("Warning: Price data is empty")
            return pd.Series()
        valid_tickers = [ticker for ticker in holdings_df['Ticker'] if ticker in price_data.columns]
        
        if not valid_tickers:
            print("Warning: No valid tickers found in both holdings and price data")
            return pd.Series(0.0, index=price_data.index)

        if len(price_data) <= 1:
            print("Warning: Insufficient price data points for normalization")
            return pd.Series(0.0, index=price_data.index)
        
        weights = holdings_df.loc[holdings_df['Ticker'].isin(valid_tickers), ['Ticker', 'Weight']]
        total_weight = weights['Weight'].sum()
        if total_weight == 0:
            print("Warning: Total weight of valid tickers is zero")
            return pd.Series(0.0, index=price_data.index)
            
        weights['NormalizedWeight'] = weights['Weight'] / total_weight
        
        weighted_portfolio = pd.Series(0.0, index=price_data.index)
        for ticker in valid_tickers:
            ticker_prices = price_data[ticker]
            # Handle missing values
            if ticker_prices.isna().any():
                print(f"Warning: NaN values found for {ticker}, filling forward")
                ticker_prices = ticker_prices.ffill().bfill()
                
            first_valid_price = ticker_prices.iloc[0]
            if pd.isna(first_valid_price) or first_valid_price == 0:
                print(f"Warning: Invalid first price for {ticker}, skipping")
                continue
                
            normalized_prices = ticker_prices / first_valid_price
            ticker_weight = weights.loc[weights['Ticker'] == ticker, 'NormalizedWeight'].values[0]
            weighted_portfolio += normalized_prices * ticker_weight
        
        return weighted_portfolio

    def verify_spy_vs_constituents(self, spy_data: pd.Series, weighted_portfolio: pd.Series, threshold: float = 0.0001) -> Dict[str, Any]:
        """
        Verify if SPY returns match the weighted constituents returns within the threshold.
        
        Returns:
        - Dictionary with verification results
        """
        if spy_data.empty or weighted_portfolio.empty:
            return {
                'within_threshold': False,
                'max_difference': float('inf'),
                'mean_difference': float('inf'),
                'spy_returns': pd.Series(),
                'portfolio_returns': pd.Series(),
                'spearman_coeff': 0.0,
                'error': 'Empty data provided'
            }
        
        aligned_data = pd.concat([spy_data, weighted_portfolio], axis=1)
        aligned_data.columns = ['SPY', 'Portfolio']
        aligned_data = aligned_data.dropna()
        
        if aligned_data.empty:
            return {
                'within_threshold': False,
                'max_difference': float('inf'),
                'mean_difference': float('inf'),
                'spy_returns': pd.Series(),
                'portfolio_returns': pd.Series(),
                'spearman_coeff': 0.0,
                'error': 'No overlapping dates between SPY and weighted portfolio'
            }
        
        norm_spy = aligned_data['SPY'] / aligned_data['SPY'].iloc[0]
        norm_portfolio = aligned_data['Portfolio'] / aligned_data['Portfolio'].iloc[0]
        
        spy_returns = norm_spy.pct_change().dropna()
        portfolio_returns = norm_portfolio.pct_change().dropna()
        
        combined_returns = pd.concat([spy_returns, portfolio_returns], axis=1)
        combined_returns.columns = ['SPY', 'Portfolio']
        combined_returns = combined_returns.dropna()
        
        diff = (combined_returns['SPY'] - combined_returns['Portfolio']).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        within_threshold = (diff <= threshold).all()
        
        pearson_coeff = combined_returns.corr(method='pearson').iloc[0, 1]
        spearman_coeff = combined_returns.corr(method='spearman').iloc[0, 1]

        worst_days = diff.nlargest(5)
        
        return {
            'within_threshold': within_threshold,
            'max_difference': max_diff,
            'mean_difference': mean_diff,
            'spy_returns': spy_returns,
            'portfolio_returns': portfolio_returns,
            'pearson_coeff': pearson_coeff,
            'spearman_coeff': spearman_coeff,
            'worst_days': worst_days,
            'data_quality': {
                'spy_nan_count': spy_data.isna().sum(),
                'portfolio_nan_count': weighted_portfolio.isna().sum()
            }
        }


class WhartonResearchDataSource(DataSource):
    """Research-oriented Wharton loader used by the SignalStrategy pipeline.

    The older ``WhartonDataSource`` (above) is the interactive / earnings-surprise
    flavour kept for ``main.py``. This class focuses on the columns that
    ``load_wharton_research_dataset`` needs — raw-panel access, event dates,
    and a split-adjusted total-return reference price — and is used throughout
    the research / simulation stack.
    """

    DEFAULT_EVENT_COLUMNS = ("anncdate", "recorddate", "paydate", "divdpaydate")

    # Columns the research loader actually reads. Reading the full 76-column
    # parquet decompresses to ~9GB RSS; projecting to this whitelist drops
    # the read peak to <1GB. _load_data falls back to the full read for any
    # columns that aren't in the parquet (older snapshots).
    _REQUIRED_COLUMNS: tuple[str, ...] = (
        "tic", "datadate", "prccd", "ajexdi", "trfd",
        "cshtrd", "divd", "cshoc", "eps",
        "conm", "gsector", "gind", "sic", "naics", "exchg", "fic",
        "anncdate", "recorddate", "paydate", "divdpaydate",
    )

    def __init__(self, file_path: str, use_total_return: bool = True):
        self.file_path = file_path
        self.use_total_return = use_total_return
        self.data = self._load_data(file_path)
        self._prepare_derived_fields()

    def _load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        if file_path.endswith(".parquet"):
            # Project to the columns the research loader reads. The parquet
            # has ~76 columns but only ~20 are consumed downstream; reading
            # the rest costs gigabytes of RAM during decode for nothing.
            try:
                import pyarrow.parquet as _pq
                schema_cols = set(_pq.read_schema(file_path).names)
                wanted = [c for c in self._REQUIRED_COLUMNS if c in schema_cols]
                data = pd.read_parquet(file_path, columns=wanted) if wanted else pd.read_parquet(file_path)
            except Exception:
                # Fall back to a full read if pyarrow inspection fails — keeps
                # this loader robust to unusual parquet layouts.
                data = pd.read_parquet(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            data = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        else:
            raise TypeError("Unsupported Wharton file format. Expected parquet, xlsx, xls, or csv.")

        if "datadate" not in data.columns or "tic" not in data.columns:
            raise ValueError("Wharton data must contain at least 'tic' and 'datadate' columns.")

        data = data.copy()
        data["tic"] = data["tic"].astype(str).str.strip()
        data["datadate"] = pd.to_datetime(data["datadate"])

        for column in self.DEFAULT_EVENT_COLUMNS:
            if column in data.columns:
                data[column] = pd.to_datetime(data[column], errors="coerce")

        data = data.sort_values(["datadate", "tic"]).reset_index(drop=True)
        data = data.drop_duplicates(subset=["tic", "datadate"], keep="last")
        return data

    def _prepare_derived_fields(self) -> None:
        if "prccd" not in self.data.columns:
            raise ValueError("Wharton data must contain the 'prccd' close-price column.")

        if "ajexdi" in self.data.columns:
            adjustment = self.data["ajexdi"].replace(0, np.nan)
        else:
            adjustment = pd.Series(1.0, index=self.data.index)

        # WRDS / Compustat documents adjusted close as PRCCD / AJEXDI.
        self.data["split_adjusted_close"] = self.data["prccd"] / adjustment

        total_return_factor = self.data["trfd"].fillna(1.0) if "trfd" in self.data.columns else 1.0
        # This reference price preserves the daily total-return ratio when pct_change is applied.
        self.data["total_return_reference"] = self.data["split_adjusted_close"] * total_return_factor

        if "cshtrd" in self.data.columns:
            self.data["Volume"] = self.data["cshtrd"]

        if "divd" in self.data.columns:
            self.data["Cash Dividend"] = self.data["divd"].fillna(0.0)

        if "cshoc" in self.data.columns:
            self.data["Market Cap"] = self.data["prccd"] * self.data["cshoc"]

    def list_tickers(self) -> List[str]:
        return sorted(self.data["tic"].dropna().unique().tolist())

    def get_available_columns(self) -> List[str]:
        return self.data.columns.tolist()

    def get_raw_data(
        self,
        tickers: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        frame = self.data

        if tickers:
            requested = {ticker.upper() for ticker in tickers}
            frame = frame[frame["tic"].isin(requested)]

        if start_date is not None:
            frame = frame[frame["datadate"] >= pd.Timestamp(start_date)]

        if end_date is not None:
            frame = frame[frame["datadate"] <= pd.Timestamp(end_date)]

        if columns:
            required_columns = ["tic", "datadate"]
            selected_columns = list(dict.fromkeys(required_columns + list(columns)))
            existing_columns = [column for column in selected_columns if column in frame.columns]
            frame = frame[existing_columns]

        return frame.copy()

    def get_feature_panel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        value_column: str,
    ) -> pd.DataFrame:
        filtered = self.get_raw_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            columns=[value_column],
        )
        if filtered.empty or value_column not in filtered.columns:
            return pd.DataFrame()

        panel = filtered.pivot(index="datadate", columns="tic", values=value_column)
        return panel.sort_index().sort_index(axis=1)

    def get_event_dates(
        self,
        tickers: Optional[Sequence[str]] = None,
        event_columns: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        requested_event_columns = event_columns or self.DEFAULT_EVENT_COLUMNS
        available_event_columns = [column for column in requested_event_columns if column in self.data.columns]
        if not available_event_columns:
            return pd.DataFrame(columns=["ticker", "event_type", "event_date"])

        filtered = self.get_raw_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            columns=available_event_columns,
        )

        events = []
        for column in available_event_columns:
            subset = filtered[["tic", column]].dropna().copy()
            if subset.empty:
                continue
            subset["event_type"] = column
            subset = subset.rename(columns={"tic": "ticker", column: "event_date"})
            events.append(subset[["ticker", "event_type", "event_date"]].drop_duplicates())

        if not events:
            return pd.DataFrame(columns=["ticker", "event_type", "event_date"])

        event_frame = pd.concat(events, ignore_index=True)
        event_frame["ticker"] = event_frame["ticker"].astype(str).str.strip()
        event_frame["event_date"] = pd.to_datetime(event_frame["event_date"])
        return event_frame.sort_values(["event_type", "event_date", "ticker"]).reset_index(drop=True)

    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        price_column = "total_return_reference" if self.use_total_return else "split_adjusted_close"
        return self.get_feature_panel(tickers, start_date, end_date, price_column)
