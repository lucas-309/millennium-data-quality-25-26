import sys
import os
import pandas as pd
import numpy as np

# Ensure we can import from the backtester folder
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from backtester.data_source import WhartonDataSource, YahooFinanceDataSource


def test_wharton_vs_yfinance_bulk():
    print("Initializing WhartonDataSource...")
    try:
        wharton_ds = WhartonDataSource('backtester/WhartonDataSource.parquet', use_trfd=True)
    except Exception as e:
        print(f"Error loading Wharton data: {e}")
        return

    print("Initializing YahooFinanceDataSource...")
    yf_ds = YahooFinanceDataSource()

    # Let's dynamically pull the first 30 tickers available in the Wharton dataset
    available_tickers = wharton_ds.data['tic'].unique().tolist()
    tickers = available_tickers[:30]
    
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    print(f"\nFetching historical data for {len(tickers)} tickers ({start_date} to {end_date})...")
    
    # Get Data from both
    w_data = wharton_ds.get_historical_data(tickers, start_date, end_date)
    y_data = yf_ds.get_historical_data(tickers, start_date, end_date)
    
    if w_data.empty or y_data.empty:
        print("Failed to fetch data from one of the sources.")
        return

    # Find common tickers that successfully downloaded from both sources
    common_tickers = list(set(w_data.columns).intersection(set(y_data.columns)))
    print(f"\nSuccessfully aligned {len(common_tickers)} tickers from both sources.")
    
    w_data = w_data[common_tickers]
    y_data = y_data[common_tickers]

    # Strip timezones from both indices to ensure Pandas perfectly aligns the dates
    w_data.index = pd.to_datetime(w_data.index).tz_localize(None)
    y_data.index = pd.to_datetime(y_data.index).tz_localize(None)

    # Calculate returns (keep NaNs so one broken ticker doesn't wipe the dataframe)
    w_ret = w_data.pct_change()
    y_ret = y_data.pct_change()

    # Intersect dates to ensure we are comparing exact same trading days
    common_dates = w_ret.index.intersection(y_ret.index)
    w_ret = w_ret.loc[common_dates]
    y_ret = y_ret.loc[common_dates]

    # Calculate Absolute Differences in Return Percentage (%)
    abs_diff = (w_ret - y_ret).abs() * 100
    
    # Calculate means
    mean_diff_per_ticker = abs_diff.mean().sort_values(ascending=False)
    overall_mean_diff = mean_diff_per_ticker.mean()

    print("\n--- Mean Absolute Difference in Daily Returns by Ticker (%) ---")
    print(mean_diff_per_ticker.map(lambda x: f"{x:.5f}%").to_string())

    print("\n=======================================================")
    print(f"OVERALL AVERAGE DIFFERENCE ACROSS ENTIRE SET: {overall_mean_diff:.5f}%")
    print("=======================================================")
    
    print("\nConclusion: If the average difference is extremely close to 0%, the Wharton methodology is highly robust and fully mirrors Yahoo Finance.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    test_wharton_vs_yfinance_bulk()
