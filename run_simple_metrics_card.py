"""Build a slide-ready 'simple metrics' card matching the
Metrics.calculate(...) signature used elsewhere in the repo:
    Avg Daily Return, Cumulative Return, Log Return,
    Volatility, Sharpe Ratio, Max Drawdown
with risk_free_rate = 0.045 (4.5%).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("all_alpha_results")
OUT = Path("slide_assets/combined_strategy")
OUT.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE = 0.045

COLOR_NAVY = "#0a1f44"
COLOR_TEAL = "#2f8f7f"
COLOR_DOWN = "#b22222"
COLOR_GREY = "#9aa0a6"


def calculate(portfolio_values: pd.Series, returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> dict:
    metrics = {}
    metrics["Avg Daily Return"] = float(returns.mean())
    metrics["Cumulative Return"] = float((1 + returns).prod() - 1)
    metrics["Log Return"] = float(np.log(1 + returns).mean())
    metrics["Volatility"] = float(returns.std() * np.sqrt(252))
    excess = returns - risk_free_rate / 252
    metrics["Sharpe Ratio"] = float(excess.mean() / excess.std() * np.sqrt(252))
    running_max = portfolio_values.cummax()
    drawdown = portfolio_values / running_max - 1
    metrics["Max Drawdown"] = float(drawdown.min())
    return metrics


def render_card(returns_df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
    })

    columns = ["ALL-ALPHA COMPOSITE", "ALL-ALPHA SLEEVE ENSEMBLE", "Equal-Weight Universe"]
    display_names = ["Combined Strategy", "Sleeve Ensemble", "Benchmark"]
    series_metrics = []
    for col in columns:
        r = returns_df[col].fillna(0.0)
        pv = (1 + r).cumprod()
        series_metrics.append(calculate(pv, r))

    fig = plt.figure(figsize=(13, 6.5), facecolor="white")
    fig.suptitle("Combined Strategy — Performance Metrics",
                 fontsize=22, fontweight="bold", y=0.96, color=COLOR_NAVY)
    fig.text(0.5, 0.91,
             f"2000-01-01 to 2025-01-01  ·  167-ticker Wharton survivor panel  ·  10 bps tcost  ·  risk-free rate = {RISK_FREE_RATE:.1%}",
             ha="center", fontsize=10, color="#555")

    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.30,
                          left=0.06, right=0.96, top=0.84, bottom=0.10)

    composite = series_metrics[0]
    benchmark = series_metrics[2]

    # Hero row — three big numbers from the composite
    def big_panel(ax, title, value, sub, color=COLOR_NAVY):
        ax.axis("off")
        ax.text(0.5, 0.78, title, ha="center", fontsize=12, color="#666", transform=ax.transAxes)
        ax.text(0.5, 0.38, value, ha="center", fontsize=34, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.06, sub, ha="center", fontsize=10, color="#777", transform=ax.transAxes)

    big_panel(fig.add_subplot(gs[0, 0]), "Sharpe Ratio",
              f"{composite['Sharpe Ratio']:.2f}",
              f"benchmark {benchmark['Sharpe Ratio']:.2f}")
    big_panel(fig.add_subplot(gs[0, 1]), "Cumulative Return",
              f"{composite['Cumulative Return']:.0%}",
              f"benchmark {benchmark['Cumulative Return']:.0%}")
    big_panel(fig.add_subplot(gs[0, 2]), "Max Drawdown",
              f"{composite['Max Drawdown']:.1%}",
              f"benchmark {benchmark['Max Drawdown']:.1%}",
              color=COLOR_DOWN)

    # Comparison table — exactly the 6 metrics from the screenshot
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    rows_order = [
        ("Avg Daily Return", "{:.4%}"),
        ("Cumulative Return", "{:.2%}"),
        ("Log Return", "{:.4%}"),
        ("Volatility", "{:.2%}"),
        ("Sharpe Ratio", "{:.2f}"),
        ("Max Drawdown", "{:.2%}"),
    ]
    rows = []
    for metric, fmt in rows_order:
        rows.append([metric] + [fmt.format(m[metric]) for m in series_metrics])
    col_labels = ["Metric", *display_names]
    table = ax_table.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center", colWidths=[0.30, 0.235, 0.235, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.7)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor(COLOR_NAVY)
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f5f6f8")
            table[(i, j)].set_edgecolor("#ddd")
            if j == 0:
                table[(i, j)].set_text_props(fontweight="bold")

    fig.savefig(OUT / "00_metrics_card.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Print to stdout for the deck blurb
    print(f"\nSimple metrics  (risk-free rate = {RISK_FREE_RATE:.1%})")
    print("=" * 72)
    for name, m in zip(display_names, series_metrics):
        print(f"\n{name}")
        for metric, fmt in rows_order:
            print(f"  {metric:<22}: {fmt.format(m[metric])}")


def main() -> None:
    returns_df = pd.read_csv(RESULTS / "all_returns.csv", index_col=0, parse_dates=True)
    render_card(returns_df)


if __name__ == "__main__":
    main()
