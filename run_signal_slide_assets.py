"""Generate slide-ready assets from the all-alpha combined run.

Reads the CSVs/PNGs in `all_alpha_results/` and produces high-DPI, deck-friendly
charts plus a one-page metrics card in `slide_assets/combined_strategy/`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


RESULTS = Path("all_alpha_results")
OUT = Path("slide_assets/combined_strategy")
OUT.mkdir(parents=True, exist_ok=True)

# Deck palette — high-contrast, color-blind friendly
COLOR_COMPOSITE = "#0a1f44"   # near-black navy
COLOR_ENSEMBLE = "#2f8f7f"    # teal
COLOR_BENCH = "#9aa0a6"       # grey
COLOR_DOWN = "#b22222"        # firebrick (drawdowns)
COLOR_UP = "#1f6f3f"          # forest (positive bars)
COLOR_ACCENT = "#d97706"      # amber (highlights)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_returns() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "all_returns.csv", index_col=0, parse_dates=True)
    return df


def load_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "combination_summary.csv")


def load_alpha_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "alpha_summary.csv")


def load_validation() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "validation.csv")


def load_stress() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "stress_regimes.csv")


def load_weights() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "average_effective_sleeve_weights.csv", index_col=0)


# ---------------------------------------------------------------------------
# Hero chart: composite vs ensemble vs benchmark
# ---------------------------------------------------------------------------
def chart_hero(returns: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    curves = [
        ("ALL-ALPHA COMPOSITE", COLOR_COMPOSITE, 3.0, "-"),
        ("ALL-ALPHA SLEEVE ENSEMBLE", COLOR_ENSEMBLE, 2.4, "-"),
        ("Equal-Weight Universe", COLOR_BENCH, 2.0, "--"),
    ]
    for name, color, lw, ls in curves:
        if name not in returns.columns:
            continue
        cum = (1.0 + returns[name].fillna(0.0)).cumprod() - 1.0
        ax.plot(cum.index, cum.values, label=name.title().replace("All-Alpha", "All-Alpha"),
                color=color, linewidth=lw, linestyle=ls)

    ax.set_title("Combined Strategy — 25-Year Cumulative Performance", pad=14)
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    ax.set_facecolor("#fafafa")
    plt.tight_layout()
    fig.savefig(OUT / "01_hero_cumulative.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------------
def chart_drawdown(returns: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    curves = [
        ("ALL-ALPHA COMPOSITE", COLOR_COMPOSITE, 0.6),
        ("ALL-ALPHA SLEEVE ENSEMBLE", COLOR_ENSEMBLE, 0.45),
        ("Equal-Weight Universe", COLOR_BENCH, 0.3),
    ]
    for name, color, alpha in curves:
        if name not in returns.columns:
            continue
        cum = (1.0 + returns[name].fillna(0.0)).cumprod()
        dd = cum / cum.cummax() - 1.0
        ax.fill_between(dd.index, dd.values, 0.0, color=color, alpha=alpha,
                        label=name.title().replace("All-Alpha", "All-Alpha"))

    ax.set_title("Drawdown Profile", pad=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", frameon=False)
    ax.set_facecolor("#fafafa")
    plt.tight_layout()
    fig.savefig(OUT / "02_drawdown.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------
def chart_rolling_sharpe(returns: pd.DataFrame, window: int = 252) -> None:
    def rolling(s: pd.Series) -> pd.Series:
        m = s.rolling(window).mean()
        sd = s.rolling(window).std().replace(0, np.nan)
        return m / sd * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    if "ALL-ALPHA COMPOSITE" in returns.columns:
        rs = rolling(returns["ALL-ALPHA COMPOSITE"])
        ax.plot(rs.index, rs.values, color=COLOR_COMPOSITE, linewidth=2.2, label="Composite")
    if "ALL-ALPHA SLEEVE ENSEMBLE" in returns.columns:
        rs = rolling(returns["ALL-ALPHA SLEEVE ENSEMBLE"])
        ax.plot(rs.index, rs.values, color=COLOR_ENSEMBLE, linewidth=1.8, label="Sleeve Ensemble")
    if "Equal-Weight Universe" in returns.columns:
        rs = rolling(returns["Equal-Weight Universe"])
        ax.plot(rs.index, rs.values, color=COLOR_BENCH, linewidth=1.6, linestyle="--",
                label="Equal-Weight Universe")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axhline(1, color=COLOR_ACCENT, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.text(returns.index[-1], 1.05, "Sharpe = 1.0", color=COLOR_ACCENT,
            fontsize=9, ha="right", va="bottom")

    ax.set_title("Rolling 1-Year Sharpe Ratio", pad=14)
    ax.set_ylabel("Sharpe")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=False)
    ax.set_facecolor("#fafafa")
    plt.tight_layout()
    fig.savefig(OUT / "03_rolling_sharpe.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sleeve Sharpe contribution
# ---------------------------------------------------------------------------
def chart_sleeve_sharpes(alpha_summary: pd.DataFrame) -> None:
    df = alpha_summary[alpha_summary["included"] == True].copy()  # noqa: E712
    df = df.sort_values("final_sharpe", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(df["strategy"], df["final_sharpe"], color=COLOR_ENSEMBLE, alpha=0.85)
    for bar, val in zip(bars, df["final_sharpe"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10)

    ax.axvline(1.0, color=COLOR_ACCENT, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_title("Individual Sleeve Sharpe Ratios", pad=14)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_xlim(0, max(df["final_sharpe"].max() * 1.15, 1.2))
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_facecolor("#fafafa")
    plt.tight_layout()
    fig.savefig(OUT / "04_sleeve_sharpes.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Effective allocation (donut)
# ---------------------------------------------------------------------------
def chart_allocation(weights: pd.DataFrame) -> None:
    w = weights.iloc[:, 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = plt.get_cmap("viridis")
    colors = [palette(i / max(len(w) - 1, 1)) for i in range(len(w))]
    wedges, _ = ax.pie(
        w.values, labels=None, colors=colors, startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    legend_labels = [f"{name}  —  {val:.1%}" for name, val in w.items()]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=10)
    ax.set_title("Average Effective Sleeve Weights", pad=14)
    plt.tight_layout()
    fig.savefig(OUT / "05_allocation_donut.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stress regime bars
# ---------------------------------------------------------------------------
def chart_stress(stress: pd.DataFrame) -> None:
    df = stress.sort_values("cumulative_return")
    labels = df["regime"].str.replace("_", " ")
    vals = df["cumulative_return"].values
    colors = [COLOR_DOWN if v < 0 else COLOR_UP for v in vals]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, vals, color=colors, alpha=0.85)
    span = max(abs(min(vals)), abs(max(vals)))
    pad = span * 0.04
    for bar, v in zip(bars, vals):
        x = v - pad if v < 0 else v + pad
        ha = "right" if v < 0 else "left"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{v:+.1%}",
                va="center", ha=ha, fontsize=10, color="#222")

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlim(min(vals) - span * 0.18, max(vals) + span * 0.12)
    ax.set_title("Composite Performance Across Named Stress Regimes", pad=14)
    ax.set_xlabel("Cumulative Return")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_facecolor("#fafafa")
    plt.tight_layout()
    fig.savefig(OUT / "06_stress_regimes.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# One-page metrics card
# ---------------------------------------------------------------------------
def metrics_card(summary: pd.DataFrame, validation: pd.DataFrame) -> None:
    composite = summary[summary["strategy"] == "ALL-ALPHA COMPOSITE"].iloc[0]
    ensemble = summary[summary["strategy"] == "ALL-ALPHA SLEEVE ENSEMBLE"].iloc[0]
    bench = summary[summary["strategy"] == "Equal-Weight Universe"].iloc[0]

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35,
                          left=0.06, right=0.96, top=0.88, bottom=0.10)

    fig.suptitle("All-Alpha Combined Strategy — Summary Card",
                 fontsize=20, fontweight="bold", y=0.96, color=COLOR_COMPOSITE)
    fig.text(0.5, 0.91, "25-year backtest, 2000-01-01 to 2025-01-01  ·  167-ticker survivor panel  ·  10 bps tcost  ·  1-day execution lag",
             ha="center", fontsize=10, color="#555")

    # Big number panels: composite headline
    def _panel(ax, title, value, sub, color=COLOR_COMPOSITE):
        ax.axis("off")
        ax.text(0.5, 0.78, title, ha="center", fontsize=11, color="#666", transform=ax.transAxes)
        ax.text(0.5, 0.40, value, ha="center", fontsize=30, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.10, sub, ha="center", fontsize=9, color="#777", transform=ax.transAxes)

    _panel(fig.add_subplot(gs[0, 0]), "Sharpe Ratio",
           f"{composite['Sharpe Ratio']:.2f}",
           f"benchmark {bench['Sharpe Ratio']:.2f}")
    _panel(fig.add_subplot(gs[0, 1]), "Annualized Return",
           f"{composite['Annualized Return']:.1%}",
           f"benchmark {bench['Annualized Return']:.1%}")
    _panel(fig.add_subplot(gs[0, 2]), "Max Drawdown",
           f"{composite['Max Drawdown']:.1%}",
           f"benchmark {bench['Max Drawdown']:.1%}", color=COLOR_DOWN)

    # Composite vs Ensemble side-by-side table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    metrics = [
        ("Annualized Return", "Annualized Return", "{:.2%}"),
        ("Annualized Volatility", "Annualized Volatility", "{:.2%}"),
        ("Sharpe Ratio", "Sharpe Ratio", "{:.2f}"),
        ("Sortino Ratio", "Sortino Ratio", "{:.2f}"),
        ("Calmar Ratio", "Calmar Ratio", "{:.2f}"),
        ("Max Drawdown", "Max Drawdown", "{:.2%}"),
        ("Win Rate", "Win Rate", "{:.1%}"),
    ]
    rows = [(label, fmt.format(composite[col]), fmt.format(ensemble[col]), fmt.format(bench[col]))
            for label, col, fmt in metrics]
    cell_text = rows
    col_labels = ["Metric", "All-Alpha Composite", "Sleeve Ensemble", "Benchmark"]
    table = ax_table.table(
        cellText=cell_text, colLabels=col_labels,
        loc="center", cellLoc="center", colWidths=[0.30, 0.235, 0.235, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.55)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#0a1f44")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f5f6f8")
            table[(i, j)].set_edgecolor("#ddd")

    # Validation row
    ax_val = fig.add_subplot(gs[2, :])
    ax_val.axis("off")
    val_rows = []
    for _, row in validation.iterrows():
        try:
            val_str = f"{float(row['value']):.2f}"
        except (ValueError, TypeError):
            val_str = str(row["value"])
        passed = "PASS" if row["pass"] else "FAIL"
        threshold = "" if pd.isna(row.get("threshold")) else f"vs {row['threshold']}"
        val_rows.append((row["metric"], val_str, threshold, passed))

    title_y = 0.85
    ax_val.text(0.5, title_y, "Statistical Validation", ha="center",
                fontsize=13, fontweight="bold", color=COLOR_COMPOSITE,
                transform=ax_val.transAxes)
    for i, (metric, val, thr, status) in enumerate(val_rows):
        x = 0.05 + i * 0.32
        color = COLOR_UP if status == "PASS" else COLOR_DOWN
        ax_val.text(x, 0.55, metric, fontsize=10, color="#555", transform=ax_val.transAxes)
        ax_val.text(x, 0.30, f"{val}  {thr}", fontsize=12, fontweight="bold",
                    color=COLOR_COMPOSITE, transform=ax_val.transAxes)
        ax_val.text(x, 0.05, status, fontsize=10, fontweight="bold", color=color,
                    transform=ax_val.transAxes)

    fig.savefig(OUT / "00_metrics_card.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# How-it-works architecture flow (boxes & arrows)
# ---------------------------------------------------------------------------
def chart_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, text, color=COLOR_COMPOSITE, fontsize=11, text_color="white"):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white",
                             linewidth=2, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.6))

    ax.text(6.5, 6.2, "How the Combined Strategy Works", ha="center",
            fontsize=18, fontweight="bold", color=COLOR_COMPOSITE)

    # Layer 1: 11 sleeves
    sleeves = [
        "Small-Cap Tilt", "Value Composite", "EPS Revision",
        "Sector-Neutral\nDiv Yield", "Cross-Sectional\nMomentum", "Short-Term\nReversal",
        "Residual\nMomentum", "Low Volatility", "PEAD",
        "MA Crossover", "ML Ridge",
    ]
    n = len(sleeves)
    cell_w, cell_h = 1.10, 0.85
    total_w = n * cell_w + (n - 1) * 0.05
    x0 = (13 - total_w) / 2
    y0 = 4.5
    for i, s in enumerate(sleeves):
        box(x0 + i * (cell_w + 0.05), y0, cell_w, cell_h, s,
            color="#2f8f7f", fontsize=8.5)
    ax.text(13 / 2, y0 + cell_h + 0.30, "11 Independent Alpha Sleeves",
            ha="center", fontsize=12, fontweight="bold", color="#333")
    ax.text(13 / 2, y0 - 0.30, "each sleeve cross-sectionally z-scores its signal per date",
            ha="center", fontsize=10, fontstyle="italic", color="#666")

    # Arrows down to two paths
    arrow(13 / 2, y0 - 0.45, 4, 3.4)
    arrow(13 / 2, y0 - 0.45, 9, 3.4)

    # Path A — score blend
    box(1.0, 2.2, 6.0, 1.2,
        "Path A: Score Blend\nz-score each sleeve » average across sleeves\n» pick top 5% basket » backtest as one portfolio",
        color=COLOR_COMPOSITE, fontsize=10)
    # Path B — ensemble
    box(7.0, 2.2, 5.0, 1.2,
        "Path B: Sleeve Ensemble\neach sleeve runs as own backtest » 6 portfolio\nmodels (Eq Wt, Inv Vol, RP, HRP, MinVar, MaxDiv)\nblend daily returns",
        color="#2f8f7f", fontsize=9.5)

    # Arrows down to outcomes
    arrow(4.0, 2.15, 4.0, 1.55)
    arrow(9.5, 2.15, 9.5, 1.55)

    # Outcomes
    box(1.0, 0.6, 6.0, 0.95,
        "ALL-ALPHA COMPOSITE\nSharpe 1.00 · Return 14.5% · MaxDD −29%",
        color=COLOR_COMPOSITE, fontsize=11)
    box(7.0, 0.6, 5.0, 0.95,
        "SLEEVE ENSEMBLE\nSharpe 0.92 · Return 9.2% · MaxDD −22%",
        color="#2f8f7f", fontsize=11)

    fig.savefig(OUT / "07_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    returns = load_returns()
    summary = load_summary()
    alpha_summary = load_alpha_summary()
    validation = load_validation()
    stress = load_stress()
    weights = load_weights()

    print("Generating slide assets in", OUT)
    metrics_card(summary, validation)
    chart_hero(returns)
    chart_drawdown(returns)
    chart_rolling_sharpe(returns)
    chart_sleeve_sharpes(alpha_summary)
    chart_allocation(weights)
    chart_stress(stress)
    chart_architecture()

    print("\nSlide assets:")
    for f in sorted(OUT.glob("*.png")):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
