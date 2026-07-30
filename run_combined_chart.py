"""Simple Combined Alpha vs Benchmark vs Risk-Free chart.

Honest setup: long-only top 10% basket, inverse-vol weights with 10%
position cap, monthly rebalance, 10 bps transaction costs, 1-day execution
lag, NO trend filter overlay.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


OUT = Path("slide_assets/combined_strategy")
OUT.mkdir(parents=True, exist_ok=True)
RFR = 0.045
WINDOW_START = "2015-01-01"


def main() -> None:
    composite = pd.read_csv("all_alpha_results/composite_honest.csv", index_col=0, parse_dates=True)["return"]
    all_ret = pd.read_csv("all_alpha_results/all_returns.csv", index_col=0, parse_dates=True)
    bench = all_ret["Equal-Weight Universe"].fillna(0.0)

    composite = composite.loc[WINDOW_START:].fillna(0.0)
    bench = bench.loc[WINDOW_START:].reindex(composite.index).fillna(0.0)

    cum_strat = (1 + composite).cumprod() - 1
    cum_bench = (1 + bench).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(cum_strat.index, cum_strat.values, color="#1f77b4", linewidth=1.7,
            label="Combined Alpha")
    ax.plot(cum_bench.index, cum_bench.values, color="#ff7f0e", linewidth=1.4,
            linestyle="--", label="Benchmark (Equal-Weight Universe)")

    ax.set_title("Combined Alpha vs Benchmark", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)

    plt.tight_layout()
    out = OUT / "combined_alpha.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Print metrics for the deck text
    excess = composite - RFR / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    ann = (1 + composite).prod() ** (252 / len(composite)) - 1
    vol = composite.std() * np.sqrt(252)
    cum = (1 + composite).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    bann = (1 + bench).prod() ** (252 / len(bench)) - 1
    bvol = bench.std() * np.sqrt(252)
    bsharpe = (bench - RFR / 252).mean() / (bench - RFR / 252).std() * np.sqrt(252)
    bdd = ((1 + bench).cumprod() / (1 + bench).cumprod().cummax() - 1).min()

    print(f"saved: {out}")
    print(f"\nCombined Alpha ({WINDOW_START}+):")
    print(f"  ann return        : {ann:.2%}")
    print(f"  ann volatility    : {vol:.2%}")
    print(f"  Sharpe (RFR=4.5%) : {sharpe:.2f}")
    print(f"  max drawdown      : {dd:.2%}")
    print(f"  cumulative return : {(cum.iloc[-1] - 1):.0%}")
    print(f"\nBenchmark:")
    print(f"  ann return        : {bann:.2%}")
    print(f"  ann volatility    : {bvol:.2%}")
    print(f"  Sharpe (RFR=4.5%) : {bsharpe:.2f}")
    print(f"  max drawdown      : {bdd:.2%}")


if __name__ == "__main__":
    main()
