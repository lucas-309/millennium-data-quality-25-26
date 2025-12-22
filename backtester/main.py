import pandas as pd
import os
import sys

# Add parent directory to path if running as a script to support absolute imports
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtester.data_source import YahooFinanceDataSource, PickleDataSource
from backtester.order_generator import MeanReversionOrderGenerator
# from backtester.momentum_strategy import MomentumOrderGenerator
from backtester.backtest_engine import EquityBacktestEngine
from backtester.metrics import ExtendedMetrics

# TODO: refactor into python notebooks, this is a MEAN REV demo of the backtester as a .py file
def main():
    """
    Example of using the backtester to backtest a mean reversion strategy on a portfolio of equities.
    """
    # Check for cached data first
    cache_file = 'sp500_data.pkl'
    used_cache = False
    if os.path.exists(cache_file):
        print(f"Found cache file: {cache_file}")
        try:
            data_source = PickleDataSource(cache_file)
            used_cache = True
        except Exception as e:
            print(f"Error loading cache: {e}. Falling back to Yahoo Finance API.")
            data_source = YahooFinanceDataSource()
    else:
        print("Cache file not found, using Yahoo Finance API")
        data_source = YahooFinanceDataSource()

    tickers = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2011-01-01"
    end_date = "2024-01-01"
    
    data = data_source.get_historical_data(tickers, start_date, end_date)

    # Fallback logic if cache returned empty/insufficient data
    if (data.empty or len(data) < 10) and used_cache:
        print("Cached data was empty or insufficient. Falling back to Yahoo Finance API.")
        data_source = YahooFinanceDataSource()
        data = data_source.get_historical_data(tickers, start_date, end_date)

    if data.empty:
        print("Error: No data could be fetched. Exiting.")
        return

    order_generator = MeanReversionOrderGenerator()
    backtest_engine = EquityBacktestEngine(initial_cash=100000)
    metrics_calculator = ExtendedMetrics()

    orders = order_generator.generate_orders(data)
    if not orders:
        print("Warning: No orders were generated. Check strategy parameters or data.")

    backtest_results = backtest_engine.run_backtest(orders, data)
    portfolio_values = backtest_results["portfolio_values"]["Portfolio Value"]
    
    if portfolio_values.empty:
         print("Error: Portfolio values empty. Cannot calculate metrics.")
         return

    returns = portfolio_values.pct_change().dropna()

    # Use the SAME valid data source for benchmark
    benchmark_data = data_source.get_historical_data(["SPY"], start_date, end_date)
    if benchmark_data.empty:
         print("Warning: Could not fetch benchmark data.")
         benchmark_returns = None
    else:
         benchmark_returns = benchmark_data["SPY"].pct_change().dropna()

    metrics = metrics_calculator.calculate(portfolio_values, returns, benchmark_returns)
    # Note: all values are annualized and assume 252 trading days in a year
    # Note: all returns are in fractional format. For example, 0.01 is 1% return
    print("###\nBacktest Metrics:")
    for metric in metrics.keys():
        print(f" -> {metric}: {metrics[metric]:.2f}")
    
    metrics_calculator.plot_returns(returns, title="Mean Reversion Strategy vs S&P 500")

if __name__ == "__main__":
    main()