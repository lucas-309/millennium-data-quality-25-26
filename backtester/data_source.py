from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
import os
import pickle
import time

import numpy as np
import pandas as pd
import yfinance as yf

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


class WhartonDataSource(DataSource):
    """Data source backed by locally stored Wharton / Compustat daily data."""

    DEFAULT_EVENT_COLUMNS = ("anncdate", "recorddate", "paydate", "divdpaydate")

    def __init__(self, file_path: str, use_total_return: bool = True):
        self.file_path = file_path
        self.use_total_return = use_total_return
        self.data = self._load_data(file_path)
        self._prepare_derived_fields()

    def _load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        if file_path.endswith(".parquet"):
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
