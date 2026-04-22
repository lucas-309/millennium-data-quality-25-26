"""Pyfolio-style comprehensive tearsheet.

Generates a multi-panel performance report covering returns, drawdowns,
rolling statistics, monthly heatmaps, and distribution diagnostics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


def _format_pct(ax):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def plot_cumulative_returns(ax, returns: pd.Series, benchmark: Optional[pd.Series] = None):
    cum = (1 + returns.fillna(0)).cumprod() - 1
    ax.plot(cum.index, cum.values, label="Strategy", linewidth=2, color="#1f77b4")
    if benchmark is not None:
        bench_cum = (1 + benchmark.reindex(returns.index).fillna(0)).cumprod() - 1
        ax.plot(bench_cum.index, bench_cum.values,
                label="Benchmark", linewidth=2, linestyle="--", color="#ff7f0e")
    ax.set_title("Cumulative Returns", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    _format_pct(ax)


def plot_drawdown(ax, returns: pd.Series):
    cum = (1 + returns.fillna(0)).cumprod()
    dd = (cum / cum.cummax() - 1)
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
    ax.set_title("Drawdown", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    _format_pct(ax)


def plot_rolling_sharpe(ax, returns: pd.Series, window: int = 126, risk_free: float = 0.0):
    daily_rf = risk_free / 252
    excess = returns - daily_rf
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)
    ax.plot(sharpe.index, sharpe.values, color="#2ca02c", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Rolling {window}-Day Sharpe", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_rolling_beta(ax, returns: pd.Series, benchmark: pd.Series, window: int = 126):
    if benchmark is None:
        ax.set_visible(False)
        return
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) == 0:
        return
    aligned.columns = ["strategy", "benchmark"]
    cov = aligned["strategy"].rolling(window).cov(aligned["benchmark"])
    var = aligned["benchmark"].rolling(window).var().replace(0, np.nan)
    beta = cov / var
    ax.plot(beta.index, beta.values, color="#9467bd", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Rolling {window}-Day Beta", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_monthly_heatmap(ax, returns: pd.Series):
    monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    if monthly.empty:
        ax.set_visible(False)
        return
    frame = monthly.to_frame("ret")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    pivot = frame.pivot_table(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_title("Monthly Returns Heatmap", fontsize=12)
    plt.colorbar(im, ax=ax, format=mtick.PercentFormatter(1.0))


def plot_return_distribution(ax, returns: pd.Series):
    clean = returns.dropna()
    if clean.empty:
        ax.set_visible(False)
        return
    ax.hist(clean.values, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black")
    mean = clean.mean()
    std = clean.std()
    ax.axvline(mean, color="black", linestyle="--", linewidth=1, label=f"Mean={mean:.4f}")
    ax.axvline(mean - 2 * std, color="red", linestyle=":", linewidth=1, label=f"-2σ={mean - 2 * std:.4f}")
    ax.axvline(mean + 2 * std, color="red", linestyle=":", linewidth=1)
    ax.set_title("Daily Return Distribution", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)


def generate_tearsheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    strategy_name: str = "Strategy",
    output_path: Optional[Path] = None,
    risk_free: float = 0.04,
) -> plt.Figure:
    """Generate a 6-panel tearsheet and optionally save to disk.

    Returns the matplotlib Figure.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Tearsheet — {strategy_name}", fontsize=16, fontweight="bold")

    plot_cumulative_returns(axes[0, 0], returns, benchmark)
    plot_drawdown(axes[0, 1], returns)
    plot_rolling_sharpe(axes[1, 0], returns, window=126, risk_free=risk_free)
    plot_rolling_beta(axes[1, 1], returns, benchmark, window=126)
    plot_monthly_heatmap(axes[2, 0], returns)
    plot_return_distribution(axes[2, 1], returns)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
