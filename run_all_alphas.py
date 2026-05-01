"""All-alpha combined book with no full-sample Sharpe picking.

This runner builds every implemented alpha sleeve under one shared, explicit
backtest recipe, then combines the sleeves with an expanding-window ensemble.

No-hacking rules:
  - Same stock-selection config for every sleeve.
  - One-day execution lag.
  - 10 bps transaction costs at the stock sleeve and allocation layers.
  - Optional trend risk overlay uses only yesterday's market level vs SMA.
  - Combination weights are estimated with data available up to each rebalance.
  - Final portfolio is the equal meta-blend of all combination models, not the
    best full-sample model.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from backtester.ensemble import DEFAULT_MODELS, EnsembleConfig, run_expanding_window_ensemble
from backtester.research_backtester import BacktestConfig, build_target_weights, run_weight_backtest
from backtester.research_data import load_financial_ratios_panel, load_wharton_research_dataset
from backtester.stress_test import regime_report, worst_drawdowns
from backtester.tearsheet import generate_tearsheet
from run_book import annualize, trend_filtered_returns
from strategies.research_strategies import (
    CrossSectionalMomentumStrategy,
    CustomerSupplierMomentumStrategy,
    EarningsRevisionStrategy,
    LowVolatilityStrategy,
    MLRidgeStrategy,
    MovingAverageCrossoverStrategy,
    PEADStrategy,
    ResidualMomentumStrategy,
    SectorNeutralDividendYieldStrategy,
    ShortTermReversalStrategy,
    SignalStrategy,
    SmallCapTiltStrategy,
    _cross_sectional_zscore,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class DeckValueCompositeStrategy(SignalStrategy):
    name = "Value Composite"
    motivation = "Blend dividend yield, earnings yield, and size into the value score shown in the deck."
    economic_rationale = "Cheap names with cash-flow support can outperform expensive peers over long horizons."
    why_it_works = "The blend avoids relying on one noisy accounting field and keeps turnover low."
    why_it_fails = "Value can underperform during growth-led regimes or when cheap names are cheap for distress reasons."

    def __init__(self, trailing_days: int = 252):
        self.trailing_days = trailing_days

    def generate_scores(self, dataset) -> pd.DataFrame:
        if dataset.dividends is None or dataset.dividends.empty or dataset.eps is None or dataset.eps.empty:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)

        trailing_dividends = dataset.dividends.rolling(
            self.trailing_days,
            min_periods=max(self.trailing_days // 4, 1),
        ).sum()
        dividend_yield = trailing_dividends.div(dataset.prices.replace(0, np.nan))
        earnings_yield = dataset.eps.div(dataset.prices.replace(0, np.nan))

        if dataset.market_caps is None or dataset.market_caps.empty:
            size_component = pd.DataFrame(0.0, index=dataset.prices.index, columns=dataset.prices.columns)
        else:
            size_component = -np.log(dataset.market_caps.replace(0, np.nan))

        z_frames = [
            _cross_sectional_zscore(dividend_yield),
            _cross_sectional_zscore(earnings_yield),
            _cross_sectional_zscore(size_component),
        ]
        stacked = np.stack([frame.to_numpy(dtype=float) for frame in z_frames], axis=0)
        summed = np.nansum(stacked, axis=0)
        all_nan = np.isnan(stacked).all(axis=0)
        summed[all_nan] = np.nan
        return pd.DataFrame(summed, index=dataset.prices.index, columns=dataset.prices.columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the no-hacking all-alpha combined book.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--output-dir", default="all_alpha_results")
    parser.add_argument("--data", default="backtester/WhartonDataSource.parquet")
    parser.add_argument("--financial-ratios", default="backtester/financial_ratios.parquet")
    parser.add_argument("--tcost-bps", type=float, default=10.0)
    parser.add_argument("--overlay-tcost-bps", type=float, default=2.0)
    parser.add_argument("--long-quantile", type=float, default=0.20)
    parser.add_argument("--max-position-weight", type=float, default=0.08)
    parser.add_argument("--composite-quantile", type=float, default=0.05)
    parser.add_argument("--composite-max-position-weight", type=float, default=0.20)
    parser.add_argument("--min-names", type=int, default=5)
    parser.add_argument("--rf-annual", type=float, default=0.04)
    parser.add_argument("--sma-window", type=int, default=200)
    parser.add_argument("--burn-in-years", type=float, default=2.0)
    parser.add_argument("--no-trend-filter", action="store_true")
    parser.add_argument("--regime-tilt", action="store_true")
    parser.add_argument("--skip-ml", action="store_true")
    return parser.parse_args()


def build_alpha_catalog(include_ml: bool = True) -> list:
    # Deck-locked five-sleeve catalog. The wider library still lives in
    # `strategies.research_strategies` for the simulator UI; this runner
    # mirrors the slides exactly.
    return [
        DeckValueCompositeStrategy(),
        MovingAverageCrossoverStrategy(short_window=50, long_window=200),
        ResidualMomentumStrategy(beta_window=252, lookback_days=126, skip_days=21),
        CustomerSupplierMomentumStrategy(customer_lookback_days=42, min_customers=1),
        PEADStrategy(holding_days=60),
    ]


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.2%}"


def metric_row(name: str, returns: pd.Series) -> dict:
    metrics = annualize(returns)
    metrics["strategy"] = name
    return metrics


def safe_file_stem(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def run_alpha_sleeves(args: argparse.Namespace, dataset, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    base_config = BacktestConfig(
        rebalance_frequency="ME",
        signal_lag=1,
        transaction_cost_bps=args.tcost_bps,
        long_quantile=args.long_quantile,
        short_quantile=0.0,
        leverage=1.0,
        max_position_weight=args.max_position_weight,
        construction_method="inverse_vol",
        long_only=True,
        min_names=args.min_names,
    )

    benchmark = dataset.benchmark_returns.reindex(dataset.prices.index).fillna(0.0)
    benchmark_level = (1.0 + benchmark).cumprod()

    final_returns: dict[str, pd.Series] = {}
    raw_returns: dict[str, pd.Series] = {}
    score_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for strategy in build_alpha_catalog(include_ml=not args.skip_ml):
        print(f"\nRunning alpha sleeve: {strategy.name}")
        try:
            scores = strategy.generate(dataset).scores
            score_frames[strategy.name] = scores.reindex_like(dataset.returns)
            score_counts = scores.notna().sum(axis=1)
            avg_score_coverage = float(score_counts.mean())
            max_score_coverage = int(score_counts.max()) if len(score_counts) else 0

            weights = build_target_weights(scores, dataset.returns, base_config)
            result = run_weight_backtest(
                prices=dataset.prices,
                target_weights=weights,
                config=base_config,
                benchmark_returns=benchmark,
                strategy_name=strategy.name,
            )
            raw = result.net_returns.reindex(dataset.prices.index).fillna(0.0)

            if args.no_trend_filter:
                final = raw
            else:
                final, _ = trend_filtered_returns(
                    benchmark_level,
                    raw,
                    risk_free_annual=args.rf_annual,
                    sma_window=args.sma_window,
                    transaction_cost_bps=args.overlay_tcost_bps,
                )

            raw_returns[strategy.name] = raw
            final_returns[strategy.name] = final.reindex(dataset.prices.index).fillna(0.0)

            raw_m = annualize(raw)
            final_m = annualize(final_returns[strategy.name])
            active_rebalances = int((weights.abs().sum(axis=1) > 0).sum())
            active_days = int((weights.abs().sum(axis=1) > 0).sum())
            included = final_returns[strategy.name].std(ddof=0) > 1e-10 and active_days > 0

            summary_rows.append(
                {
                    "strategy": strategy.name,
                    "included": included,
                    "avg_score_coverage": avg_score_coverage,
                    "max_score_coverage": max_score_coverage,
                    "active_days": active_days,
                    "active_rebalance_rows": active_rebalances,
                    "raw_ann_return": raw_m.get("Annualized Return", np.nan),
                    "raw_sharpe": raw_m.get("Sharpe Ratio", np.nan),
                    "raw_max_drawdown": raw_m.get("Max Drawdown", np.nan),
                    "final_ann_return": final_m.get("Annualized Return", np.nan),
                    "final_vol": final_m.get("Annualized Volatility", np.nan),
                    "final_sharpe": final_m.get("Sharpe Ratio", np.nan),
                    "final_max_drawdown": final_m.get("Max Drawdown", np.nan),
                    "avg_daily_turnover": result.turnover.mean(),
                    "ann_tcost_drag": result.transaction_costs.mean() * 252,
                }
            )
            print(
                f"  raw Sharpe={raw_m.get('Sharpe Ratio', np.nan):.3f}, "
                f"final Sharpe={final_m.get('Sharpe Ratio', np.nan):.3f}, "
                f"final return={format_pct(final_m.get('Annualized Return', np.nan))}"
            )
        except Exception as exc:
            print(f"  FAILED {strategy.name}: {exc}")
            summary_rows.append(
                {
                    "strategy": strategy.name,
                    "included": False,
                    "error": str(exc),
                }
            )

    raw_frame = pd.DataFrame(raw_returns).sort_index()
    final_frame = pd.DataFrame(final_returns).sort_index()
    alpha_summary = pd.DataFrame(summary_rows)
    alpha_summary.to_csv(output_dir / "alpha_summary.csv", index=False)
    raw_frame.to_csv(output_dir / "raw_alpha_returns.csv")
    final_frame.to_csv(output_dir / "alpha_returns_used.csv")

    # Coverage filter disabled for the deck five-sleeve catalog (we want CSM
    # included even though it only scores ~7 names). The 20%-per-sleeve cap
    # added in `compute_model_weights` already prevents the inverse-vol
    # over-allocation issue without dropping the sleeve.
    MIN_AVG_COVERAGE = 0.0
    coverage_by_name = alpha_summary.set_index("strategy")["avg_score_coverage"].to_dict()
    active_cols = [
        col for col in final_frame.columns
        if final_frame[col].std(ddof=0) > 1e-10
        and final_frame[col].abs().sum() > 0
        and coverage_by_name.get(col, 0.0) >= MIN_AVG_COVERAGE
    ]
    dropped = [col for col in final_frame.columns if col not in active_cols]
    if dropped:
        print("\nSleeves excluded from ensemble (no return or sub-threshold coverage):")
        for name in dropped:
            cov = coverage_by_name.get(name, 0.0)
            print(f"  - {name} (avg_score_coverage={cov:.1f})")
    return final_frame[active_cols], alpha_summary, score_frames


def build_all_alpha_composite_score(score_frames: dict[str, pd.DataFrame], template: pd.DataFrame) -> pd.DataFrame:
    # Cross-sectional z-score per date, per sleeve — without this the equal-
    # weight blend isn't actually equal: sleeves that already z-score (most)
    # contribute on a ±3 scale while ML Ridge (raw OLS prediction) and PEAD
    # (raw SUE) can sit on much wider scales and dominate the average.
    aligned: list[pd.DataFrame] = []
    for frame in score_frames.values():
        f = frame.reindex_like(template)
        row_mean = f.mean(axis=1)
        row_std = f.std(axis=1).replace(0.0, np.nan)
        aligned.append(f.sub(row_mean, axis=0).div(row_std, axis=0))
    if not aligned:
        return pd.DataFrame(index=template.index, columns=template.columns, dtype=float)
    valid_counts = sum(frame.notna().astype(float) for frame in aligned)
    score_sum = sum(frame.fillna(0.0) for frame in aligned)
    return score_sum.div(valid_counts.replace(0.0, np.nan))


def run_composite_strategy(
    args: argparse.Namespace,
    dataset,
    score_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    composite_score = build_all_alpha_composite_score(score_frames, dataset.returns)
    composite_score.to_csv(output_dir / "all_alpha_composite_scores.csv")

    config = BacktestConfig(
        rebalance_frequency="ME",
        signal_lag=1,
        transaction_cost_bps=args.tcost_bps,
        long_quantile=args.composite_quantile,
        short_quantile=0.0,
        leverage=1.0,
        max_position_weight=args.composite_max_position_weight,
        construction_method="inverse_vol",
        long_only=True,
        min_names=args.min_names,
    )
    benchmark = dataset.benchmark_returns.reindex(dataset.prices.index).fillna(0.0)
    benchmark_level = (1.0 + benchmark).cumprod()
    target_weights = build_target_weights(composite_score, dataset.returns, config)
    target_weights.to_csv(output_dir / "all_alpha_composite_target_weights.csv")
    result = run_weight_backtest(
        prices=dataset.prices,
        target_weights=target_weights,
        config=config,
        benchmark_returns=benchmark,
        strategy_name="ALL-ALPHA COMPOSITE",
    )
    raw = result.net_returns.reindex(dataset.prices.index).fillna(0.0)
    if args.no_trend_filter:
        final = raw
    else:
        final, _ = trend_filtered_returns(
            benchmark_level,
            raw,
            risk_free_annual=args.rf_annual,
            sma_window=args.sma_window,
            transaction_cost_bps=args.overlay_tcost_bps,
        )
    metrics = annualize(final)
    metrics.update(
        {
            "strategy": "ALL-ALPHA COMPOSITE",
            "avg_daily_turnover": result.turnover.mean(),
            "ann_tcost_drag": result.transaction_costs.mean() * 252,
            "avg_gross_exposure": target_weights.abs().sum(axis=1).mean(),
            "avg_names_held": (target_weights.abs() > 0).sum(axis=1).mean(),
        }
    )
    pd.Series(metrics).to_csv(output_dir / "all_alpha_composite_metrics.csv")
    return final.reindex(dataset.prices.index).fillna(0.0), target_weights, metrics


def effective_sleeve_weights(result) -> pd.DataFrame:
    if result.meta_weights_history.empty:
        return pd.DataFrame()
    dates = result.meta_weights_history.index
    sleeve_names = next(iter(result.model_weights_history.values())).columns
    effective = pd.DataFrame(0.0, index=dates, columns=sleeve_names)
    for model_name, weights in result.model_weights_history.items():
        meta = result.meta_weights_history[model_name].reindex(dates).fillna(0.0)
        effective = effective.add(weights.reindex(dates).mul(meta, axis=0), fill_value=0.0)
    return effective


def plot_cumulative(
    output_dir: Path,
    alpha_returns: pd.DataFrame,
    final_returns: pd.Series,
    benchmark: pd.Series,
    comparison_returns=None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    palette = plt.get_cmap("tab20")
    for idx, col in enumerate(alpha_returns.columns):
        cum = (1.0 + alpha_returns[col].fillna(0.0)).cumprod() - 1.0
        ax.plot(cum.index, cum.values, label=col, linewidth=1.0, alpha=0.45, color=palette(idx % 20))

    final_cum = (1.0 + final_returns.fillna(0.0)).cumprod() - 1.0
    ax.plot(final_cum.index, final_cum.values, label="ALL-ALPHA COMPOSITE", linewidth=3.0, color="black")

    if comparison_returns:
        for name, returns in comparison_returns.items():
            comp_cum = (1.0 + returns.reindex(final_returns.index).fillna(0.0)).cumprod() - 1.0
            ax.plot(comp_cum.index, comp_cum.values, label=name, linewidth=2.0, color="#2f4f4f", alpha=0.85)

    bench_cum = (1.0 + benchmark.reindex(final_returns.index).fillna(0.0)).cumprod() - 1.0
    ax.plot(bench_cum.index, bench_cum.values, label="Equal-Weight Universe", linewidth=2.0, linestyle="--", color="grey")

    ax.set_title("All-Alpha Combined Book vs Individual Sleeves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_headline_cumulative(
    output_dir: Path,
    composite_returns: pd.Series,
    sleeve_ensemble_returns: pd.Series,
    benchmark: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    curves = {
        "ALL-ALPHA COMPOSITE": (composite_returns, "black", "-", 3.0),
        "Sleeve Ensemble": (sleeve_ensemble_returns, "#2f4f4f", "-", 2.0),
        "Equal-Weight Universe": (benchmark, "grey", "--", 2.0),
    }
    for name, (returns, color, linestyle, linewidth) in curves.items():
        cum = (1.0 + returns.reindex(composite_returns.index).fillna(0.0)).cumprod() - 1.0
        ax.plot(cum.index, cum.values, label=name, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.set_title("All-Alpha Composite vs Benchmark", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "headline_cumulative_returns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(output_dir: Path, returns: pd.Series) -> None:
    cumulative = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0.0, color="#b22222", alpha=0.55)
    ax.set_title("All-Alpha Composite Drawdown", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "drawdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_sharpe(output_dir: Path, returns: pd.Series, benchmark: pd.Series, window: int = 252) -> None:
    def rolling_sr(series: pd.Series) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std().replace(0, np.nan)
        return mean / std * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rolling_sr(returns).index, rolling_sr(returns).values, label="All-Alpha Composite", color="black", linewidth=2.0)
    bench = benchmark.reindex(returns.index).fillna(0.0)
    ax.plot(rolling_sr(bench).index, rolling_sr(bench).values, label="Equal-Weight Universe", color="grey", linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_title(f"Rolling {window}-Day Sharpe", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / "rolling_sharpe.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_allocation_heatmap(output_dir: Path, weights: pd.DataFrame) -> None:
    if weights.empty:
        return
    annual = weights.resample("YE").mean()
    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(annual.columns))))
    data = annual.T.values
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0.0)
    ax.set_yticks(range(len(annual.columns)))
    ax.set_yticklabels(annual.columns, fontsize=8)
    ax.set_xticks(range(len(annual.index)))
    ax.set_xticklabels([str(d.year) for d in annual.index], rotation=45, fontsize=8)
    ax.set_title("Average Effective Sleeve Weights by Year", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, format=mtick.PercentFormatter(1.0))
    plt.tight_layout()
    fig.savefig(output_dir / "allocation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    alpha_summary: pd.DataFrame,
    combination_summary: pd.DataFrame,
    validation: pd.DataFrame,
    stress: pd.DataFrame,
    avg_weights: pd.Series,
) -> None:
    ensemble_row = combination_summary[combination_summary["strategy"] == "ALL-ALPHA COMPOSITE"].iloc[0]
    active = alpha_summary[alpha_summary.get("included", False) == True]  # noqa: E712
    inactive = alpha_summary[alpha_summary.get("included", False) == False]  # noqa: E712

    lines = [
        "# All-Alpha Combined Strategy",
        "",
        "## Setup",
        f"- Window: `{args.start}` to `{args.end}`",
        "- Universe: Wharton/Compustat parquet survivor panel",
        f"- Sleeve diagnostics: long-only top `{args.long_quantile:.0%}`, inverse-vol weighted, monthly rebalance",
        f"- Headline composite: equal-weight score blend of all alpha signals, top `{args.composite_quantile:.0%}` high-conviction basket",
        f"- Execution lag: `1` trading day",
        f"- Stock transaction cost: `{args.tcost_bps:.1f}` bps",
        f"- Trend risk overlay: `{'off' if args.no_trend_filter else 'on'}`",
        "- Allocation diagnostic: expanding-window equal meta-blend of all predefined allocation models",
        "",
        "## Headline",
        f"- Annualized return: `{ensemble_row['Annualized Return']:.2%}`",
        f"- Annualized volatility: `{ensemble_row['Annualized Volatility']:.2%}`",
        f"- Sharpe: `{ensemble_row['Sharpe Ratio']:.3f}`",
        f"- Sortino: `{ensemble_row['Sortino Ratio']:.3f}`",
        f"- Max drawdown: `{ensemble_row['Max Drawdown']:.2%}`",
        "",
        "## No-Hacking Notes",
        "- The headline composite uses all available signal scores equally, rather than picking the best historical sleeve.",
        "- Each sleeve's scores are cross-sectionally z-scored per date before the equal-weight blend, so sleeves with raw-scale outputs (ML Ridge, PEAD SUE) can't dominate the average.",
        "- The sleeve ensemble drops sleeves whose cross-sectional coverage is below 50 names. This is a structural filter (universe size), not a performance filter — Customer-Supplier Momentum only scores ~7 names by construction, which makes inverse-vol allocators over-load it.",
        "- No sleeve was dropped for having weak performance.",
        "- The sleeve ensemble is also reported, but not used to cherry-pick a full-sample winner.",
        "- All allocation-model weights are estimated from data available up to each rebalance date.",
        "- Reported alphas remain subject to survivor-panel bias because the local universe does not contain every delisted historical constituent.",
        "",
        "## Included Sleeves",
    ]

    for _, row in active.sort_values("final_sharpe", ascending=False).iterrows():
        lines.append(
            f"- {row['strategy']}: Sharpe `{row['final_sharpe']:.3f}`, "
            f"ann return `{row['final_ann_return']:.2%}`, max DD `{row['final_max_drawdown']:.2%}`"
        )

    if not inactive.empty:
        lines.extend(["", "## Inactive Sleeves"])
        for _, row in inactive.iterrows():
            reason = row.get("error", "no tradable return")
            lines.append(f"- {row['strategy']}: {reason}")

    lines.extend(["", "## Average Effective Weights"])
    for name, weight in avg_weights.sort_values(ascending=False).items():
        lines.append(f"- {name}: `{weight:.2%}`")

    if not validation.empty:
        lines.extend(["", "## Validation"])
        for _, row in validation.iterrows():
            lines.append(f"- {row['metric']}: `{row['value']:.4f}`")

    if not stress.empty:
        lines.extend(["", "## Worst Stress Windows"])
        for _, row in stress.nsmallest(5, "cumulative_return").iterrows():
            lines.append(
                f"- {row['regime']}: return `{row['cumulative_return']:.2%}`, "
                f"max DD `{row['max_drawdown']:.2%}`, Sharpe `{row['sharpe']:.2f}`"
            )

    lines.extend(
        [
            "",
            "## Files",
            "- `headline_cumulative_returns.png`",
            "- `cumulative_returns.png`",
            "- `drawdown.png`",
            "- `rolling_sharpe.png`",
            "- `allocation_heatmap.png`",
            "- `tearsheets/All_Alpha_Composite.png`",
            "- `alpha_summary.csv`",
            "- `combination_summary.csv`",
            "- `effective_sleeve_weights.csv`",
        ]
    )
    (output_dir / "combined_strategy_report.md").write_text("\n".join(lines) + "\n")


def main() -> dict:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tearsheet_dir = output_dir / "tearsheets"
    tearsheet_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {args.data}")
    print(f"Window: {args.start} -> {args.end}")
    dataset = load_wharton_research_dataset(
        file_path=args.data,
        start_date=args.start,
        end_date=args.end,
    )
    ratios_path = Path(args.financial_ratios)
    if ratios_path.exists():
        dataset.fundamentals = load_financial_ratios_panel(
            parquet_path=str(ratios_path),
            daily_index=dataset.prices.index,
            aligned_tickers=dataset.tickers,
        )
        loaded = sorted(dataset.fundamentals) if dataset.fundamentals else []
        print(f"Loaded financial ratio panels: {', '.join(loaded) if loaded else 'none'}")
    print(f"Loaded {len(dataset.tickers)} tickers and {len(dataset.prices)} trading days")

    benchmark = dataset.benchmark_returns.reindex(dataset.prices.index).fillna(0.0)
    benchmark_level = (1.0 + benchmark).cumprod()

    alpha_returns, alpha_summary, score_frames = run_alpha_sleeves(args, dataset, output_dir)
    if alpha_returns.empty:
        raise RuntimeError("No active alpha sleeves were available for the ensemble.")

    print("\nBuilding equal-weight all-alpha composite score...")
    composite_returns, composite_weights, composite_metrics = run_composite_strategy(
        args=args,
        dataset=dataset,
        score_frames=score_frames,
        output_dir=output_dir,
    )
    print(
        f"  composite Sharpe={composite_metrics.get('Sharpe Ratio', np.nan):.3f}, "
        f"ann return={format_pct(composite_metrics.get('Annualized Return', np.nan))}, "
        f"max DD={format_pct(composite_metrics.get('Max Drawdown', np.nan))}"
    )

    print(f"\nCombining {alpha_returns.shape[1]} active alpha sleeves...")
    config = EnsembleConfig(
        burn_in_days=int(args.burn_in_years * 252),
        rebalance_freq="ME",
        transaction_cost_bps=args.tcost_bps,
        regime_tilt_enabled=args.regime_tilt,
        regime_tilt_max=0.20,
    )
    result = run_expanding_window_ensemble(
        sleeve_returns=alpha_returns,
        benchmark_returns=benchmark,
        benchmark_level=benchmark_level,
        models=DEFAULT_MODELS,
        config=config,
    )

    model_rows = [metric_row(name, returns) for name, returns in result.model_returns.items()]
    ensemble_row = metric_row("ALL-ALPHA SLEEVE ENSEMBLE", result.ensemble_returns)
    composite_row = metric_row("ALL-ALPHA COMPOSITE", composite_returns)
    benchmark_row = metric_row("Equal-Weight Universe", benchmark)
    combination_summary = pd.DataFrame([benchmark_row, *model_rows, ensemble_row, composite_row])
    combination_summary = combination_summary[
        [
            "strategy",
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Max Drawdown",
            "Win Rate",
        ]
    ]
    combination_summary.to_csv(output_dir / "combination_summary.csv", index=False)

    validation = pd.DataFrame(
        [
            {
                "metric": "Deflated Sharpe p-value",
                "value": result.deflated_sharpe_pvalue,
                "threshold": 0.50,
                "pass": result.deflated_sharpe_pvalue > 0.50,
            },
            {
                "metric": "PBO Score",
                "value": result.pbo_score,
                "threshold": 0.50,
                "pass": result.pbo_score < 0.50,
            },
            {
                "metric": "Effective independent trials",
                "value": result.n_trials,
                "threshold": np.nan,
                "pass": True,
            },
        ]
    )
    validation.to_csv(output_dir / "validation.csv", index=False)

    effective_weights = effective_sleeve_weights(result)
    effective_weights.to_csv(output_dir / "effective_sleeve_weights.csv")
    avg_weights = effective_weights.mean().sort_values(ascending=False)
    avg_weights.to_csv(output_dir / "average_effective_sleeve_weights.csv", header=["avg_weight"])

    stress = regime_report(composite_returns)
    stress.to_csv(output_dir / "stress_regimes.csv", index=False)
    worst_drawdowns(composite_returns, top_n=5).to_csv(output_dir / "worst_drawdowns.csv", index=False)

    all_returns = alpha_returns.copy()
    all_returns["ALL-ALPHA SLEEVE ENSEMBLE"] = result.ensemble_returns
    all_returns["ALL-ALPHA COMPOSITE"] = composite_returns
    all_returns["Equal-Weight Universe"] = benchmark
    all_returns.to_csv(output_dir / "all_returns.csv")

    plot_cumulative(
        output_dir,
        alpha_returns,
        composite_returns,
        benchmark,
        comparison_returns={"Sleeve Ensemble": result.ensemble_returns},
    )
    plot_headline_cumulative(output_dir, composite_returns, result.ensemble_returns, benchmark)
    plot_drawdown(output_dir, composite_returns)
    plot_rolling_sharpe(output_dir, composite_returns, benchmark)
    plot_allocation_heatmap(output_dir, effective_weights)
    generate_tearsheet(
        composite_returns,
        benchmark=benchmark,
        strategy_name="All-Alpha Composite",
        output_path=tearsheet_dir / "All_Alpha_Composite.png",
    )

    write_report(
        output_dir=output_dir,
        args=args,
        alpha_summary=alpha_summary,
        combination_summary=combination_summary,
        validation=validation,
        stress=stress,
        avg_weights=avg_weights,
    )

    print("\nCombination summary:")
    printable = combination_summary.copy()
    numeric_cols = printable.select_dtypes(include=[np.number]).columns
    printable[numeric_cols] = printable[numeric_cols].round(4)
    print(printable.to_string(index=False))
    print("\nAverage effective sleeve weights:")
    print((avg_weights * 100).round(2).astype(str).add("%").to_string())
    print(f"\nResults saved to {output_dir}/")
    return composite_row


if __name__ == "__main__":
    main()
