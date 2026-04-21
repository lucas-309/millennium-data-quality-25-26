from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
import os
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
    
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, threads=True) # newest update replaces "Close" with "Adj Close" if set auto_adjust = True
        return data['Adj Close']
    
    def get_historical_data_with_volume(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch historical price and volume data for given tickers and date range, organized by ticker. Function is DEPRECATED (remove)"""
        data = yf.download(tickers, start=start_date, end=end_date)
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