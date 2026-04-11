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
from strategies.betting_against_beta import BettingAgainstBetaOrderGenerator
from backtester.backtesters.equity_backtest import EquityBacktestEngine
from backtester.metrics import ExtendedMetrics

STRATEGIES = {
    "1": "Mean Reversion",
    "2": "Momentum",
    "3": "Pairs Trading",
    "4": "Betting Against Beta",
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
        lookback = input("Lookback window days (default: 100): ").strip()
        entry_z = input("Entry z-score threshold (default: -1.5): ").strip()
        lookback = int(lookback) if lookback else 100
        entry_z = float(entry_z) if entry_z else -1.5
        print(f">> Mean Reversion (lookback={lookback}, entry_z={entry_z})")
        return MeanReversionOrderGenerator(lookback=lookback, entry_zscore=entry_z), tickers

    elif choice == "2":
        lookback = input("Lookback window days (default: 252): ").strip()
        trailing_stop = input("Trailing stop % (default: 0.10): ").strip()
        lookback = int(lookback) if lookback else 252
        trailing_stop = float(trailing_stop) if trailing_stop else 0.10
        print(f">> Momentum (lookback={lookback}, trailing_stop={trailing_stop})")
        return MomentumOrderGenerator(lookback_days=lookback, trailing_stop_pct=trailing_stop), tickers

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
        return PairsTradingOrderGenerator(pairs=pairs, lookback_window=lookback), all_tickers

    elif choice == "4":
        lookback = input("Beta lookback period (default: 252): ").strip()
        lookback = int(lookback) if lookback else 252
        print(f">> Betting Against Beta (lookback={lookback})")

        # BAB needs SPY + many tickers for cross-sectional beta ranking
        all_tickers = list(tickers)
        if "SPY" not in all_tickers:
            all_tickers.append("SPY")
        print("  Note: BAB works best with many tickers (e.g., full S&P 500 via cache).")

        return BettingAgainstBetaOrderGenerator(lookback_period=lookback), all_tickers

    else:
        print("Invalid choice.")
        sys.exit(1)


def main():
    print("=== Strategy Backtester ===\n")
    for key, name in STRATEGIES.items():
        print(f"  {key}. {name}")

    choice = input("\nSelect a strategy [1-4]: ").strip()
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

    # Engine stats
    print(f"\n### Execution Summary:")
    print(f"  Trades executed: {backtest_results['trade_count']}")
    print(f"  Total commissions: ${backtest_results['total_commissions']:,.2f}")
    print(f"  Total slippage cost: ${backtest_results['total_slippage_cost']:,.2f}")
    order_log = backtest_results["order_log"]
    if not order_log.empty:
        rejected = order_log[order_log["status"] == "REJECTED"]
        if len(rejected) > 0:
            print(f"  Orders rejected: {len(rejected)}")
            for reason, count in rejected["reason"].value_counts().items():
                print(f"    - {reason}: {count}")

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
        if isinstance(value, float) and not (pd.isna(value) or abs(value) == float('inf')):
            print(f"  -> {metric}: {value:.4f}")
        else:
            print(f"  -> {metric}: {value}")

    metrics_calculator.plot_returns(returns, benchmark_returns=benchmark_returns, title=f"{STRATEGIES[choice]} vs S&P 500")


if __name__ == "__main__":
    main()
