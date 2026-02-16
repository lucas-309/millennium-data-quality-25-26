import pandas as pd
import os
import sys

# Add parent directory to path if running as a script to support absolute imports
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtester.data_source import YahooFinanceDataSource, PickleDataSource
from strategies.mean_reversion import MeanReversionOrderGenerator
from strategies.momentum_strategy import MomentumOrderGenerator
from strategies.pairs_trading import PairsTradingOrderGenerator
from backtester.backtesters.equity_backtest import EquityBacktestEngine
from backtester.metrics import ExtendedMetrics

STRATEGIES = {
    "1": "Mean Reversion",
    "2": "Momentum",
    "3": "Pairs Trading",
}


def get_data_source():
    cache_file = 'sp500_data.pkl'
    if os.path.exists(cache_file):
        print(f"Found cache file: {cache_file}")
        try:
            return PickleDataSource(cache_file), True
        except Exception as e:
            print(f"Error loading cache: {e}. Falling back to Yahoo Finance API.")
    else:
        print("Cache file not found, using Yahoo Finance API")
    return YahooFinanceDataSource(), False


def prompt_tickers(default):
    raw = input(f"Enter tickers comma-separated (default: {','.join(default)}): ").strip()
    if raw:
        return [t.strip().upper() for t in raw.split(",")]
    return default


def prompt_dates():
    start = input("Start date YYYY-MM-DD (default: 2011-01-01): ").strip() or "2011-01-01"
    end = input("End date YYYY-MM-DD (default: 2025-01-01): ").strip() or "2025-01-01"
    return start, end


def build_strategy(choice, tickers):
    if choice == "1":
        print(">> Mean Reversion (100-day rolling window)")
        return MeanReversionOrderGenerator(), tickers

    elif choice == "2":
        window = input("Lookback window days (default: 125): ").strip()
        threshold = input("Threshold (default: 0.02): ").strip()
        window = int(window) if window else 125
        threshold = float(threshold) if threshold else 0.02
        print(f">> Momentum (window={window}, threshold={threshold})")
        return MomentumOrderGenerator(window_days=window, threshold=threshold), tickers

    elif choice == "3":
        print("Enter pairs as: TICKER1/TICKER2, TICKER3/TICKER4")
        raw = input("Pairs (default: KO/PEP, V/MA): ").strip()
        if raw:
            pairs = []
            for p in raw.split(","):
                a, b = p.strip().split("/")
                pairs.append((a.strip().upper(), b.strip().upper()))
        else:
            pairs = [("KO", "PEP"), ("V", "MA")]

        # Ensure all pair tickers are in the ticker list
        all_tickers = list(tickers)
        for a, b in pairs:
            if a not in all_tickers:
                all_tickers.append(a)
            if b not in all_tickers:
                all_tickers.append(b)

        lookback = input("Lookback window (default: 60): ").strip()
        lookback = int(lookback) if lookback else 60

        print(f">> Pairs Trading (pairs={pairs}, lookback={lookback})")
        return PairsTradingOrderGenerator(
            pairs=pairs, lookback_window=lookback,
        ), all_tickers

    else:
        print("Invalid choice.")
        sys.exit(1)


def main():
    print("=== Strategy Backtester ===\n")
    for key, name in STRATEGIES.items():
        print(f"  {key}. {name}")

    choice = input("\nSelect a strategy [1-3]: ").strip()
    if choice not in STRATEGIES:
        print("Invalid selection.")
        return

    default_tickers = ["NVDA"]
    tickers = prompt_tickers(default_tickers)
    start_date, end_date = prompt_dates()

    order_generator, tickers = build_strategy(choice, tickers)

    # Fetch data
    data_source, used_cache = get_data_source()
    data = data_source.get_historical_data(tickers, start_date, end_date)

    if (data.empty or len(data) < 10) and used_cache:
        print("Cached data was empty or insufficient. Falling back to Yahoo Finance API.")
        data_source = YahooFinanceDataSource()
        data = data_source.get_historical_data(tickers, start_date, end_date)

    if data.empty:
        print("Error: No data could be fetched. Exiting.")
        return

    print(f"\nData loaded: {len(data)} rows, tickers: {list(data.columns)}")
    print(data.head())

    # Run backtest
    backtest_engine = EquityBacktestEngine(initial_cash=100000)
    orders = order_generator.generate_orders(data)

    if not orders:
        print("Warning: No orders were generated. Check strategy parameters or data.")

    backtest_results = backtest_engine.run_backtest(orders, data)
    portfolio_values = backtest_results["portfolio_values"]["Portfolio Value"]
    daily_holdings_and_cash = backtest_results["daily_holdings_and_cash"]

    if portfolio_values.empty:
        print("Error: Portfolio values empty. Cannot calculate metrics.")
        return

    returns = portfolio_values.pct_change().dropna()

    # Benchmark
    benchmark_data = data_source.get_historical_data(["SPY"], start_date, end_date)
    if benchmark_data.empty:
        print("Warning: Could not fetch benchmark data.")
        benchmark_returns = None
    else:
        benchmark_returns = benchmark_data["SPY"].pct_change().dropna()

    # Monthly holdings snapshot
    print(f"\n### Monthly Holdings ({STRATEGIES[choice]}):")
    monthly_dates = daily_holdings_and_cash.resample('ME').last()
    for date, row in monthly_dates.iterrows():
        holdings = {col: int(row[col]) for col in row.index if col != 'Cash' and row[col] != 0}
        cash = row.get('Cash', 0)
        if holdings:
            holdings_str = ", ".join(f"{t}: {q}" for t, q in holdings.items())
            print(f"  {date.strftime('%Y-%m')}  |  Cash: ${cash:,.0f}  |  {holdings_str}")
        else:
            print(f"  {date.strftime('%Y-%m')}  |  Cash: ${cash:,.0f}  |  (no holdings)")

    # Metrics
    metrics_calculator = ExtendedMetrics()
    metrics = metrics_calculator.calculate(portfolio_values, returns, benchmark_returns, data, daily_holdings_and_cash)

    print(f"\n### Backtest Metrics ({STRATEGIES[choice]}):")
    for metric, value in metrics.items():
        print(f"  -> {metric}: {value:.2f}")

    metrics_calculator.plot_returns(returns, benchmark_returns=benchmark_returns, title=f"{STRATEGIES[choice]} vs S&P 500")


if __name__ == "__main__":
    main()
