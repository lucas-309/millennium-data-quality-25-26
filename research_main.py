from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.research_backtester import BacktestConfig, run_strategy_suite
from backtester.research_data import load_wharton_research_dataset
from backtester.research_reports import (
    export_research_outputs,
    generate_data_quality_report,
    validate_spy_reconstruction,
)
from backtester.data_source import WhartonDataSource
from backtester.stress_test import regime_report, worst_drawdowns
from backtester.monte_carlo import bootstrap_many_metrics
from backtester.tearsheet import generate_tearsheet
from backtester.multi_strategy import combine_strategy_returns
from strategies.research_strategies import build_default_strategy_suite


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the research-grade strategy backtester.")
    parser.add_argument("--data-file", default="backtester/WhartonDataSource.parquet", help="Path to the Wharton daily data file.")
    parser.add_argument("--start-date", default="2010-01-01", help="Backtest start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default="2025-12-31", help="Backtest end date in YYYY-MM-DD format.")
    parser.add_argument("--construction-method", default="equal_weight",
                        choices=["equal_weight", "inverse_vol", "mean_variance"])
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--signal-lag", type=int, default=1)
    parser.add_argument(
        "--strategy-allocation-method",
        default="equal_weight",
        choices=["equal_weight", "inverse_vol"],
    )
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--output-dir", default="research_outputs")
    parser.add_argument("--skip-spy-validation", action="store_true")


    # === NEW: Stress testing ===
    parser.add_argument("--stress-test", action="store_true",
                        help="Replay backtest over historical crisis regimes.")

    # === NEW: Monte Carlo bootstrap ===
    parser.add_argument("--monte-carlo", type=int, default=0, metavar="N",
                        help="Bootstrap N iterations to compute CIs on metrics. 0 to skip.")

    # === NEW: Tearsheets ===
    parser.add_argument("--tearsheet", action="store_true",
                        help="Generate a pyfolio-style PNG tearsheet per strategy.")

    # === NEW: Multi-strategy combination ===
    parser.add_argument("--combine-method", default="none",
                        choices=["none", "equal", "inverse_vol", "risk_parity", "hrp"],
                        help="Combine individual strategy returns into a 'Millennium book'.")
    parser.add_argument("--combine-vol-target", type=float, default=None,
                        help="Annualized vol target for combined portfolio.")
    return parser


def _print_strategy_summary(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    for key in (
        "Cumulative Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Downside Volatility",
        "Average Turnover",
        "Annualized Transaction Cost Drag",
        "Information Ratio",
    ):
        value = metrics.get(key)
        if value is not None:
            print(f"  {key}: {value:.4f}")


# ---------------------------------------------------------------------------
# Prop-shop analysis hooks
# ---------------------------------------------------------------------------
def run_stress_tests(results: dict, output_dir: Path) -> None:
    print("\n=== Stress Test Report ===")
    stress_dir = output_dir / "stress_tests"
    stress_dir.mkdir(parents=True, exist_ok=True)

    for name, result in results.items():
        report = regime_report(result.net_returns)
        report.to_csv(stress_dir / f"{name}_regimes.csv", index=False)

        drawdowns = worst_drawdowns(result.net_returns, top_n=5)
        drawdowns.to_csv(stress_dir / f"{name}_drawdowns.csv", index=False)

        # Print compact summary
        print(f"\n  [{name}]")
        bad_regimes = report.nsmallest(3, "cumulative_return")[["regime", "cumulative_return", "max_drawdown"]]
        for _, row in bad_regimes.iterrows():
            print(f"    {row['regime']}: ret={row['cumulative_return']:.2%}, max_dd={row['max_drawdown']:.2%}")


def run_monte_carlo(results: dict, n_iterations: int, output_dir: Path) -> None:
    print(f"\n=== Monte Carlo Bootstrap ({n_iterations} iterations) ===")
    mc_dir = output_dir / "monte_carlo"
    mc_dir.mkdir(parents=True, exist_ok=True)

    def sharpe(r):
        std = r.std(ddof=0)
        return r.mean() / std * np.sqrt(252) if std > 0 else 0.0

    def max_dd(r):
        cum = (1 + r).cumprod()
        return (cum / cum.cummax() - 1).min()

    def ann_return(r):
        cum = (1 + r).cumprod()
        n = len(r)
        return (cum.iloc[-1] ** (252 / max(n, 1))) - 1 if n > 0 else 0.0

    for name, result in results.items():
        df = bootstrap_many_metrics(
            result.net_returns,
            metrics={"sharpe": sharpe, "max_dd": max_dd, "ann_return": ann_return},
            n_iterations=n_iterations,
            seed=42,
        )
        df.to_csv(mc_dir / f"{name}_bootstrap.csv")
        print(f"\n  [{name}]")
        for metric, row in df.iterrows():
            print(f"    {metric}: {row['point']:.4f} (95% CI [{row['lower']:.4f}, {row['upper']:.4f}])")


def run_tearsheets(results: dict, benchmark: pd.Series, output_dir: Path) -> None:
    print("\n=== Generating Tearsheets ===")
    ts_dir = output_dir / "tearsheets"
    ts_dir.mkdir(parents=True, exist_ok=True)

    for name, result in results.items():
        output_path = ts_dir / f"{name}_tearsheet.png"
        try:
            generate_tearsheet(
                result.net_returns,
                benchmark=benchmark,
                strategy_name=name,
                output_path=output_path,
            )
            print(f"  Saved: {output_path}")
        except Exception as exc:
            print(f"  Failed for {name}: {exc}")


def run_combination(results: dict, method: str, vol_target, output_dir: Path) -> pd.Series:
    print(f"\n=== Multi-Strategy Combination ({method}) ===")
    strategy_returns = {name: r.net_returns for name, r in results.items()}
    combined = combine_strategy_returns(
        strategy_returns,
        method=method,
        lookback=252,
        rebalance_freq="ME",
        vol_target_annual=vol_target,
    )
    combined.to_csv(output_dir / f"combined_{method}_returns.csv", header=True)

    # Print summary
    std = combined.std(ddof=0)
    if std > 0 and len(combined) > 0:
        sharpe = combined.mean() / std * np.sqrt(252)
        ann_ret = (1 + combined).prod() ** (252 / len(combined)) - 1
        cum = (1 + combined).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        print(f"  Sharpe: {sharpe:.4f}")
        print(f"  Annualized Return: {ann_ret:.4f}")
        print(f"  Max Drawdown: {dd:.4f}")
    return combined


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data_file = Path(args.data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Research data file not found: {data_file}")

    wharton_source = WhartonDataSource(file_path=str(data_file), use_total_return=True)
    dataset = load_wharton_research_dataset(
        file_path=str(data_file),
        start_date=args.start_date,
        end_date=args.end_date,
    )

    strategies = build_default_strategy_suite()

    config = BacktestConfig(
        construction_method=args.construction_method,
        transaction_cost_bps=args.transaction_cost_bps,
        signal_lag=args.signal_lag,
        strategy_allocation_method=args.strategy_allocation_method,
        long_only=args.long_only,
    )

    results, signal_correlation = run_strategy_suite(dataset, strategies, config)
    data_quality_report = generate_data_quality_report(
        wharton_source,
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=dataset.tickers,
    )
    spy_validation = {}
    if not args.skip_spy_validation:
        holdings_file = data_file.parent / "holdings-daily-us-en-spy.xlsx"
        if holdings_file.exists():
            spy_validation = validate_spy_reconstruction(
                holdings_file=str(holdings_file),
                start_date=args.start_date,
                end_date=args.end_date,
            )

    output_dir = Path(args.output_dir)
    export_research_outputs(
        dataset=dataset,
        results=results,
        signal_correlation=signal_correlation,
        output_dir=args.output_dir,
        data_quality_report=data_quality_report,
        spy_validation=spy_validation,
    )

    if not signal_correlation.empty:
        print("Signal Correlation Matrix")
        print(signal_correlation.round(3).to_string())

    if data_quality_report:
        print("\nData Quality Summary")
        for key, value in data_quality_report.items():
            print(f"  {key}: {value}")

    if spy_validation:
        print("\nSPY Reconstruction Validation")
        for key, value in spy_validation.items():
            print(f"  {key}: {value}")

    for name, result in results.items():
        _print_strategy_summary(name, result.metrics)
        if result.lag_table is not None:
            print("  Lag Sharpe Snapshot:")
            print(result.lag_table[["sharpe_ratio", "avg_spearman_ic"]].round(3).to_string())

    # === Optional extended analyses ===
    if args.stress_test:
        run_stress_tests(results, output_dir)

    if args.monte_carlo > 0:
        run_monte_carlo(results, args.monte_carlo, output_dir)

    if args.tearsheet:
        run_tearsheets(results, dataset.benchmark_returns, output_dir)

    if args.combine_method != "none":
        run_combination(results, args.combine_method, args.combine_vol_target, output_dir)


if __name__ == "__main__":
    main()
