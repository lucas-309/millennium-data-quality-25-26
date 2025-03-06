from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from typing import List, Optional, Dict, Any
import numpy as np

class DataSource(ABC):
    """Interface for fetching historical market data."""
    
    @abstractmethod
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data for given tickers and date range."""
        pass

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