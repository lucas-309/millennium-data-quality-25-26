"""Multi-model ensemble — robust portfolio via expanding-window model blending.

Eliminates the lookahead bias in run_book.py by:
  1. Running 6 combination models in parallel with expanding windows
  2. Equal-weight blending at the meta level (no picking winners)
  3. Conservative regime tilts using only backward-looking features
  4. Validated with deflated Sharpe and PBO

Models:
  1. Equal Weight
  2. Inverse Volatility (expanding window)
  3. Risk Parity (expanding window)
  4. HRP (expanding window)
  5. Minimum Variance (expanding window)
  6. Maximum Diversification (expanding window)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from backtester.research_data import load_wharton_research_dataset
from backtester.research_backtester import BacktestConfig
from backtester.tearsheet import generate_tearsheet
from backtester.stress_test import regime_report, worst_drawdowns
from backtester.ensemble import (
    DEFAULT_MODELS,
    EnsembleConfig,
    run_expanding_window_ensemble,
)
from run_book import (
    vol_managed_returns,
    trend_filtered_returns,
    run_selection_sleeve,
    annualize,
)
from strategies.research_strategies import (
    CrossSectionalMomentumStrategy,
    LowVolatilityStrategy,
    SmallCapTiltStrategy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-model expanding-window ensemble")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--output-dir", default="ensemble_results")
    p.add_argument("--target-vol", type=float, default=0.16)
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--sma-window", type=int, default=200)
    p.add_argument("--rf-annual", type=float, default=0.04)
    p.add_argument("--selection-top-pct", type=float, default=0.20)
    p.add_argument("--selection-max-pos", type=float, default=0.08)
    p.add_argument("--burn-in-years", type=float, default=2.0)
    p.add_argument("--no-regime-tilt", action="store_true")
    p.add_argument("--tcost-bps", type=float, default=10.0)
    return p.parse_args()


def build_sleeves(dataset, args) -> dict[str, pd.Series]:
    """Build the same 5 sleeves as run_book.py."""
    benchmark = dataset.benchmark_returns
    bench_index = (1 + benchmark.fillna(0)).cumprod()

    sleeves = {}

    # Timing sleeve 1: vol-managed
    vm_ret, _ = vol_managed_returns(
        benchmark, target_vol=args.target_vol, max_leverage=args.max_leverage,
    )
    sleeves["Vol-Managed Equity"] = vm_ret

    # Timing sleeve 2: trend-filtered vol-managed
    tf_ret, _ = trend_filtered_returns(
        bench_index, vm_ret, risk_free_annual=args.rf_annual, sma_window=args.sma_window,
    )
    sleeves["Trend-Filtered Equity"] = tf_ret

    # Selection sleeves
    sel_config = BacktestConfig(
        rebalance_frequency="ME",
        signal_lag=1,
        transaction_cost_bps=10.0,
        long_quantile=args.selection_top_pct,
        short_quantile=0.0,
        leverage=1.0,
        max_position_weight=args.selection_max_pos,
        construction_method="inverse_vol",
        long_only=True,
        min_names=20,
    )

    selection_strategies = [
        CrossSectionalMomentumStrategy(lookback_days=126, skip_days=21),
        LowVolatilityStrategy(window=126),
        SmallCapTiltStrategy(),
    ]
    for strat in selection_strategies:
        label = f"{strat.name} + Trend"
        try:
            raw = run_selection_sleeve(strat, dataset, sel_config)
            ret, _ = trend_filtered_returns(
                bench_index, raw,
                risk_free_annual=args.rf_annual, sma_window=args.sma_window,
            )
            sleeves[label] = ret
        except Exception as exc:
            print(f"  FAILED {label}: {exc}")

    return sleeves


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tearsheet_dir = output_dir / "tearsheets"
    tearsheet_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading Wharton {args.start} -> {args.end}...")
    dataset = load_wharton_research_dataset(
        file_path="backtester/WhartonDataSource.parquet",
        start_date=args.start, end_date=args.end,
    )
    print(f"  {len(dataset.tickers)} tickers, {len(dataset.prices)} trading days")

    # Build sleeves
    print("\nBuilding sleeves...")
    sleeves = build_sleeves(dataset, args)
    sleeve_frame = pd.DataFrame(sleeves)
    print(f"  {len(sleeves)} sleeves built")

    # Sleeve correlation
    corr = sleeve_frame.corr()
    print(f"\n{'='*60}\n  Sleeve Correlation Matrix\n{'='*60}")
    print(corr.round(3).to_string())
    corr.to_csv(output_dir / "sleeve_correlation.csv")

    # Benchmark
    benchmark = dataset.benchmark_returns
    bench_index = (1 + benchmark.fillna(0)).cumprod()
    bench_m = annualize(benchmark)

    # Run ensemble
    print(f"\n{'='*60}\n  Running Expanding-Window Ensemble (6 models)\n{'='*60}")
    config = EnsembleConfig(
        burn_in_days=int(args.burn_in_years * 252),
        rebalance_freq="ME",
        transaction_cost_bps=args.tcost_bps,
        regime_tilt_enabled=not args.no_regime_tilt,
        regime_tilt_max=0.20,
    )
    result = run_expanding_window_ensemble(
        sleeve_returns=sleeve_frame,
        benchmark_returns=benchmark,
        benchmark_level=bench_index,
        models=DEFAULT_MODELS,
        config=config,
    )

    # Report ALL individual model results
    print(f"\n{'='*60}\n  Individual Model Results (ALL reported, no cherry-picking)\n{'='*60}")
    print(f"  {'Model':<25s} {'Sharpe':>8s} {'AnnRet':>9s} {'Vol':>8s} {'MaxDD':>9s}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*8} {'-'*9}")

    model_rows = []
    for name, ret_series in result.model_returns.items():
        m = annualize(ret_series)
        m["strategy"] = name
        model_rows.append(m)
        print(f"  {name:<25s} {m.get('Sharpe Ratio',0):>8.3f} "
              f"{m.get('Annualized Return',0):>+8.2%} "
              f"{m.get('Annualized Volatility',0):>7.2%} "
              f"{m.get('Max Drawdown',0):>8.2%}")

    # Ensemble result
    ens_m = annualize(result.ensemble_returns)
    ens_m["strategy"] = "ENSEMBLE (meta-blend)"
    model_rows.append(ens_m)

    print(f"  {'---':->25s} {'---':->8s} {'---':->9s} {'---':->8s} {'---':->9s}")
    print(f"  {'ENSEMBLE (meta-blend)':<25s} {ens_m.get('Sharpe Ratio',0):>8.3f} "
          f"{ens_m.get('Annualized Return',0):>+8.2%} "
          f"{ens_m.get('Annualized Volatility',0):>7.2%} "
          f"{ens_m.get('Max Drawdown',0):>8.2%}")

    print(f"\n  Sortino: {ens_m.get('Sortino Ratio', 0):.3f}")
    print(f"  Calmar:  {ens_m.get('Calmar Ratio', 0):.3f}")
    print(f"  Win Rate: {ens_m.get('Win Rate', 0):.1%}")

    # Validation
    print(f"\n{'='*60}\n  Overfitting Validation\n{'='*60}")
    print(f"  Deflated Sharpe p-value: {result.deflated_sharpe_pvalue:.4f}  "
          f"(n_trials={result.n_trials}, want > 0.50)")
    print(f"  PBO Score:               {result.pbo_score:.4f}  "
          f"(want < 0.50)")
    if result.deflated_sharpe_pvalue > 0.5:
        print("  -> Deflated Sharpe: PASS (signal likely genuine)")
    else:
        print("  -> Deflated Sharpe: CAUTION (may be noise)")
    if result.pbo_score < 0.5:
        print("  -> PBO: PASS (not overfit)")
    else:
        print("  -> PBO: CAUTION (possible overfitting)")

    # Regime history
    if not result.regime_history.empty:
        rh = result.regime_history.set_index("date")
        high_vol_pct = (rh["vol_regime"] > 1.2).mean()
        downtrend_pct = (rh["trend_state"] < 0.5).mean()
        print(f"\n  Regime stats: {high_vol_pct:.0%} high-vol rebalances, "
              f"{downtrend_pct:.0%} downtrend rebalances")

    # Comparison vs old run_book.py approach
    print(f"\n{'='*60}\n  vs. run_book.py (single-model HRP)\n{'='*60}")
    print(f"  run_book.py (HRP, fixed a priori): uses single combination method")
    print(f"  run_ensemble.py (meta-blend):      {ens_m.get('Sharpe Ratio',0):.3f} (6 models, no selection)")

    # Stress test
    print(f"\n{'='*60}\n  Stress Test\n{'='*60}")
    stress = regime_report(result.ensemble_returns)
    stress.to_csv(output_dir / "stress_regimes.csv", index=False)
    bad = stress.nsmallest(5, "cumulative_return")[["regime", "cumulative_return", "max_drawdown", "sharpe"]]
    for _, row in bad.iterrows():
        print(f"  {row['regime']}: ret={row['cumulative_return']:+.2%}  "
              f"dd={row['max_drawdown']:.2%}  sharpe={row['sharpe']:.2f}")

    dd_table = worst_drawdowns(result.ensemble_returns, top_n=3)
    print(f"\n  Top 3 drawdowns:")
    for _, row in dd_table.iterrows():
        recovery = row["recovery"] if pd.notna(row["recovery"]) else "(open)"
        rec_str = recovery.date() if hasattr(recovery, "date") else recovery
        print(f"    {row['start'].date()} -> {row['trough'].date()} -> {rec_str}: "
              f"depth={row['depth']:.2%}")

    # Tearsheets
    try:
        generate_tearsheet(
            result.ensemble_returns, benchmark=benchmark,
            strategy_name="Multi-Model Ensemble",
            output_path=tearsheet_dir / "Ensemble.png",
        )
        for name, ret in result.model_returns.items():
            safe_name = name.replace(" ", "_").replace("-", "_")
            generate_tearsheet(
                ret, benchmark=benchmark, strategy_name=name,
                output_path=tearsheet_dir / f"{safe_name}.png",
            )
    except Exception as exc:
        print(f"  tearsheet error: {exc}")

    # Master plot
    fig, ax = plt.subplots(figsize=(14, 8))
    palette = plt.get_cmap("tab10")

    # Plot each model
    for i, (name, ret) in enumerate(result.model_returns.items()):
        cum = (1 + ret.fillna(0)).cumprod() - 1
        ax.plot(cum.index, cum.values, label=name, linewidth=1.0,
                color=palette(i % 10), alpha=0.5)

    # Plot ensemble (thick)
    ens_cum = (1 + result.ensemble_returns.fillna(0)).cumprod() - 1
    ax.plot(ens_cum.index, ens_cum.values, label="ENSEMBLE", linewidth=3,
            color="black", alpha=1.0)

    # Benchmark
    bench_cum = (1 + benchmark.fillna(0)).cumprod() - 1
    ax.plot(bench_cum.index, bench_cum.values, label="Equal-Weight Benchmark",
            linewidth=2, linestyle="--", color="grey", alpha=0.8)

    ax.set_title("Multi-Model Ensemble vs Individual Models",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary CSV
    bench_row = {**bench_m, "strategy": "Equal-Weight Benchmark"}
    all_rows = [bench_row] + model_rows
    summary = pd.DataFrame(all_rows)
    cols = ["strategy", "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
            "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Win Rate"]
    summary = summary[[c for c in cols if c in summary.columns]]
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Validation CSV
    val_df = pd.DataFrame([{
        "metric": "Deflated Sharpe p-value",
        "value": result.deflated_sharpe_pvalue,
        "threshold": 0.50,
        "pass": result.deflated_sharpe_pvalue > 0.50,
    }, {
        "metric": "PBO Score",
        "value": result.pbo_score,
        "threshold": 0.50,
        "pass": result.pbo_score < 0.50,
    }, {
        "metric": "n_trials",
        "value": result.n_trials,
        "threshold": None,
        "pass": True,
    }])
    val_df.to_csv(output_dir / "validation.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  Results saved to {output_dir}/")
    print(f"{'='*60}")

    return ens_m


if __name__ == "__main__":
    main()
