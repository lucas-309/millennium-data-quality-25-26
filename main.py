import pandas as pd
import os
import sys

# Add parent directory to path if running as a script to support absolute imports
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtester.data_source import YahooFinanceDataSource, PickleDataSource, WhartonDataSource
from strategies.mean_reversion import MeanReversionOrderGenerator
from strategies.momentum_strategy import MomentumOrderGenerator
from strategies.pairs_trading import PairsTradingOrderGenerator
from strategies.moving_avg import MovingAverageOrderGenerator
from backtester.backtest_engine import EquityBacktestEngine
from backtester.metrics import ExtendedMetrics

STRATEGIES = {
    "1": "Mean Reversion",
    "2": "Momentum",
    "3": "Pairs Trading",
    "4": "Moving Average",
}

DATA_SOURCES = {
    "auto": "Auto (cache -> Yahoo fallback)",
    "yahoo": "Yahoo Finance",
    "wharton": "WhartonDataSource",
}


def normalize_ticker_for_source(ticker: str, source_name: str) -> str:
    """Normalize ticker format for the selected source."""
    t = ticker.strip().upper()
    if not t:
        return t

    if source_name == "wharton":
        # Wharton/Compustat style
        alias = {
            "BRK-B": "BRK.B",
            "BF-B": "BF.B",
            "FI": "FISV",
            "PARA": "PARAA",
            "-": "USD",
        }
        t = alias.get(t, t)
        t = t.replace("-", ".")
    else:
        # Yahoo style
        alias = {
            "BRK.B": "BRK-B",
            "BF.B": "BF-B",
        }
        t = alias.get(t, t)

    return t


def normalize_tickers_for_source(tickers, source_name: str):
    return [normalize_ticker_for_source(t, source_name) for t in tickers]


def get_full_universe_tickers(data_source, source_name: str):
    """
    Build ticker universe directly from the selected source.
    - wharton: all unique tic values in loaded Wharton dataset
    - yahoo/auto with cache: all tickers in cache dict
    - yahoo without cache: static SPY holdings snapshot
    """
    if source_name == "wharton":
        if hasattr(data_source, "data") and isinstance(data_source.data, pd.DataFrame) and "tic" in data_source.data.columns:
            t = data_source.data["tic"].dropna().astype(str).str.strip().str.upper()
            tickers = sorted(set([x for x in t.tolist() if x]))
            print(f"Using full Wharton universe: {len(tickers)} tickers.")
            return tickers
        print("Warning: could not extract full Wharton universe, falling back to NVDA.")
        return ["NVDA"]

    if isinstance(data_source, PickleDataSource) and isinstance(data_source.data, dict):
        tickers = sorted(set([str(k).strip().upper() for k in data_source.data.keys() if str(k).strip()]))
        print(f"Using full cached universe: {len(tickers)} tickers.")
        return tickers

    if isinstance(data_source, YahooFinanceDataSource):
        holdings_path = os.path.join("backtester", "holdings-daily-us-en-spy.xlsx")
        holdings_df = data_source.read_spy_holdings(holdings_path)
        if not holdings_df.empty and "Ticker" in holdings_df.columns:
            tickers = sorted(
                set(
                    holdings_df["Ticker"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .tolist()
                )
            )
            print(f"Using full Yahoo-backed holdings universe: {len(tickers)} tickers.")
            return tickers

    print("Warning: could not extract full universe for selected source, falling back to NVDA.")
    return ["NVDA"]


def get_data_source(source_choice: str):
    source_choice = source_choice.lower().strip()

    if source_choice == "wharton":
        print("Using WhartonDataSource (WhartonDataSource4 default).")
        return WhartonDataSource(use_trfd=True, include_surprise=True), False, "wharton"

    if source_choice == "yahoo":
        print("Using Yahoo Finance API.")
        return YahooFinanceDataSource(), False, "yahoo"

    # auto mode (legacy behavior)
    cache_file = 'sp500_data.pkl'
    if os.path.exists(cache_file):
        print(f"Found cache file: {cache_file}")
        try:
            return PickleDataSource(cache_file), True, "yahoo"
        except Exception as e:
            print(f"Error loading cache: {e}. Falling back to Yahoo Finance API.")
    else:
        print("Cache file not found, using Yahoo Finance API")
    return YahooFinanceDataSource(), False, "yahoo"


def prompt_tickers(default, data_source=None, source_name: str = "yahoo"):
    raw = input(
        f"Enter tickers comma-separated (default: {','.join(default)}). "
        f"Use 'full' for full source universe: "
    ).strip()
    if raw:
        if raw.lower() == "full":
            return get_full_universe_tickers(data_source, source_name)
        return [t.strip().upper() for t in raw.split(",") if t.strip()]
    return default


def prompt_dates():
    start = input("Start date YYYY-MM-DD (default: 2011-01-01): ").strip() or "2011-01-01"
    end = input("End date YYYY-MM-DD (default: 2025-01-01): ").strip() or "2025-01-01"
    return start, end


def prompt_data_source():
    print("\nAvailable data sources:")
    for key, name in DATA_SOURCES.items():
        print(f"  - {key}: {name}")
    choice = input("Select data source [auto/yahoo/wharton] (default: auto): ").strip().lower() or "auto"
    if choice not in DATA_SOURCES:
        print(f"Invalid data source '{choice}'. Using 'auto'.")
        return "auto"
    return choice


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

    elif choice == "4":
        print(">> Moving Average (5-day vs. 20-day rolling windows)")
        return MovingAverageOrderGenerator(), tickers

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

    source_choice = prompt_data_source()
    data_source, used_cache, source_name = get_data_source(source_choice)

    default_tickers = ["NVDA"]
    tickers = prompt_tickers(default_tickers, data_source=data_source, source_name=source_name)
    start_date, end_date = prompt_dates()

    order_generator, tickers = build_strategy(choice, tickers)
    # Strategy may expand universe (e.g., pairs). Normalize after full ticker list is known.

    # Fetch data
    tickers = normalize_tickers_for_source(tickers, source_name)
    data = data_source.get_historical_data(tickers, start_date, end_date)

    if (data.empty or len(data) < 10) and used_cache:
        print("Cached data was empty or insufficient. Falling back to Yahoo Finance API.")
        data_source = YahooFinanceDataSource()
        source_name = "yahoo"
        tickers = normalize_tickers_for_source(tickers, source_name)
        data = data_source.get_historical_data(tickers, start_date, end_date)

    if data.empty:
        print("Error: No data could be fetched. Exiting.")
        return

    print(f"\nData loaded: {len(data)} rows, tickers: {list(data.columns)}")
    print(data.head())

    # Run backtest
    backtest_engine = EquityBacktestEngine(initial_cash=100000, commission_rate=0.001)
    orders = order_generator.generate_orders(data)

    if not orders:
        print("Warning: No orders were generated. Check strategy parameters or data.")

    backtest_results = backtest_engine.run_backtest(orders, data)
    portfolio_values = backtest_results["portfolio_values"]["Portfolio Value"]
    daily_holdings_and_cash = backtest_results["daily_holdings_and_cash"]
    total_transaction_costs = backtest_results["total_transaction_costs"]

    if portfolio_values.empty:
        print("Error: Portfolio values empty. Cannot calculate metrics.")
        return

    returns = portfolio_values.pct_change().dropna()

    # Benchmark
    benchmark_ticker = normalize_ticker_for_source("SPY", source_name)
    benchmark_data = data_source.get_historical_data([benchmark_ticker], start_date, end_date)
    if benchmark_data.empty:
        print("Warning: Could not fetch benchmark data.")
        benchmark_returns = None
    else:
        benchmark_col = benchmark_ticker if benchmark_ticker in benchmark_data.columns else benchmark_data.columns[0]
        benchmark_returns = benchmark_data[benchmark_col].pct_change().dropna()

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
    print(f"  -> Total Transaction Costs: ${total_transaction_costs:,.2f}")
    for metric, value in metrics.items():
        print(f"  -> {metric}: {value:.2f}")

    metrics_calculator.plot_returns(returns, benchmark_returns=benchmark_returns, title=f"{STRATEGIES[choice]} vs S&P 500")


if __name__ == "__main__":
    main()
