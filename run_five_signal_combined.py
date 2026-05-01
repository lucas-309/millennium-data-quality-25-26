"""Combined-alpha plot using only the five presentation signals.

Signals included:
  1. Value Composite
  2. Simple Moving Average Crossover
  3. Residual Momentum
  4. Customer-Supplier Momentum
  5. Post-Earnings Announcement Drift

The combined score is the equal-weight average of the five normalized signal
scores. Missing signal values are treated as neutral, not as a reason to drop
the stock from the whole composite.

The headline construction is long-only top 5%, inverse-volatility weighted.
The dollar-neutral long-short version was tested and produced a much weaker
Sharpe because the bottom composite decile was not a reliable short book.
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

from backtester.research_backtester import BacktestConfig, build_target_weights, run_weight_backtest
from backtester.research_data import load_wharton_research_dataset
from run_book import annualize
from strategies.research_strategies import (
    CustomerSupplierMomentumStrategy,
    MovingAverageCrossoverStrategy,
    PEADStrategy,
    ResidualMomentumStrategy,
    SignalStrategy,
    _cross_sectional_zscore,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class DeckValueCompositeStrategy(SignalStrategy):
    name = "Value Composite"
    motivation = "Dividend yield + earnings yield + size value composite."
    economic_rationale = "Cheap stocks with accounting support can outperform expensive peers."
    why_it_works = "Blending yield, earnings, and size avoids relying on one noisy field."
    why_it_fails = "Value can underperform during growth-led markets or distress traps."

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
    parser = argparse.ArgumentParser(description="Run the five-signal combined alpha graph.")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--data", default="backtester/WhartonDataSource.parquet")
    parser.add_argument("--output-dir", default="five_signal_results")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--quantile", type=float, default=0.05)
    parser.add_argument("--max-position-weight", type=float, default=0.20)
    return parser.parse_args()


def five_signal_catalog() -> list[SignalStrategy]:
    return [
        DeckValueCompositeStrategy(),
        MovingAverageCrossoverStrategy(short_window=50, long_window=200),
        ResidualMomentumStrategy(beta_window=252, lookback_days=126, skip_days=21),
        CustomerSupplierMomentumStrategy(customer_lookback_days=42, min_customers=1),
        PEADStrategy(holding_days=60),
    ]


def build_combined_score(dataset, output_dir: Path) -> pd.DataFrame:
    scores = {}
    coverage_rows = []
    for strategy in five_signal_catalog():
        generated = strategy.generate(dataset).scores.reindex_like(dataset.returns)
        scores[strategy.name] = generated
        coverage_rows.append(
            {
                "signal": strategy.name,
                "avg_names_with_signal": generated.notna().sum(axis=1).mean(),
                "max_names_with_signal": generated.notna().sum(axis=1).max(),
            }
        )

    pd.DataFrame(coverage_rows).to_csv(output_dir / "signal_coverage.csv", index=False)

    # Equal-weight five-signal blend. Missing values are neutral zero after
    # normalization, so sparse PEAD/customer-supplier data can help where
    # available without forcing the whole universe to be dropped.
    combined = sum(frame.fillna(0.0) for frame in scores.values()) / len(scores)
    combined = _cross_sectional_zscore(combined)
    combined.to_csv(output_dir / "combined_five_signal_scores.csv")
    return combined


def plot_simple(returns: pd.Series, benchmark_returns: pd.Series, output_path: Path) -> None:
    aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0.0)
    cumulative = (
        (1.0 + returns.fillna(0.0)).cumprod()
        / (1.0 + aligned_benchmark).cumprod()
        - 1.0
    )
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(
        cumulative.index,
        cumulative.values,
        color="#1f77b4",
        linewidth=2.0,
        label="Combined Alpha vs S&P 500",
    )
    ax.set_title("Combined Alpha vs S&P 500", fontsize=14)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data}: {args.start} -> {args.end}")
    dataset = load_wharton_research_dataset(
        file_path=args.data,
        start_date=args.start,
        end_date=args.end,
    )
    print(f"Loaded {len(dataset.tickers)} tickers and {len(dataset.prices)} trading days")

    combined_score = build_combined_score(dataset, output_dir)

    config = BacktestConfig(
        initial_cash=args.initial_cash,
        rebalance_frequency="ME",
        signal_lag=1,
        transaction_cost_bps=args.transaction_cost_bps,
        long_quantile=args.quantile,
        short_quantile=0.0,
        leverage=1.0,
        max_position_weight=args.max_position_weight,
        construction_method="inverse_vol",
        long_only=True,
        min_names=10,
    )
    target_weights = build_target_weights(combined_score, dataset.returns, config)
    result = run_weight_backtest(
        prices=dataset.prices,
        target_weights=target_weights,
        config=config,
        benchmark_returns=dataset.benchmark_returns,
        strategy_name="Five-Signal Combined Alpha",
    )

    output_returns = output_dir / "combined_five_signal_returns.csv"
    output_weights = output_dir / "combined_five_signal_weights.csv"
    output_summary = output_dir / "combined_five_signal_summary.csv"
    output_graph = output_dir / "combined_five_signal_graph.png"

    result.net_returns.rename("combined_alpha_return").to_csv(output_returns)
    target_weights.to_csv(output_weights)
    metrics = annualize(result.net_returns)
    metrics.update(
        {
            "Average Turnover": result.turnover.mean(),
            "Annualized T-Cost Drag": result.transaction_costs.mean() * 252,
            "Average Gross Exposure": target_weights.abs().sum(axis=1).mean(),
        }
    )
    pd.Series(metrics).to_csv(output_summary)
    plot_simple(result.net_returns, dataset.benchmark_returns, output_graph)

    print("Five-signal combined alpha complete.")
    print(f"  Sharpe: {metrics['Sharpe Ratio']:.3f}")
    print(f"  Annualized return: {metrics['Annualized Return']:.2%}")
    print(f"  Max drawdown: {metrics['Max Drawdown']:.2%}")
    print(f"  Graph: {output_graph}")


if __name__ == "__main__":
    main()
