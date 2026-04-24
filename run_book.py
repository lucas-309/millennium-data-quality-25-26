"""Simple five-sleeve alpha book built from intuitive price-based strategies.

Five sleeves combined via HRP (chosen a priori, no in-sample selection):

Timing sleeves (smooth the beta):
  1. Vol-Managed Equity — target 16% vol, levered to 2x max
  2. Trend-Filtered Equity — hold when > 200-day SMA, T-bills otherwise

Selection sleeves (long-only top quintile + trend filter):
  3. Momentum (6-1 return)
  4. Mean Reversion (distance from 100-day moving average)
  5. Low Volatility (bottom quintile by 126-day vol)

The goal here is readability: each selection sleeve is a small price-based
strategy that collaborators can inspect quickly, while the shared backtester
handles ranking, sizing, lag, and transaction costs.
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
from backtester.research_backtester import (
    BacktestConfig,
    run_strategy_backtest,
)
from backtester.tearsheet import generate_tearsheet
from backtester.stress_test import regime_report, worst_drawdowns
from backtester.multi_strategy import combine_strategy_returns
from strategies.research_strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    LowVolatilityStrategy,
)


# ---------------------------------------------------------------------------
# Timing sleeves (non-signal-based)
# ---------------------------------------------------------------------------
def vol_managed_returns(
    raw_returns: pd.Series,
    target_vol: float = 0.16,
    vol_window: int = 63,
    min_leverage: float = 0.3,
    max_leverage: float = 2.0,
    transaction_cost_bps: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    realized_vol = raw_returns.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
    leverage = (target_vol / realized_vol.replace(0, np.nan)).clip(lower=min_leverage, upper=max_leverage)
    leverage = leverage.shift(1).fillna(1.0)
    scaled = raw_returns * leverage
    tcost = leverage.diff().abs().fillna(0) * transaction_cost_bps / 10_000.0
    return scaled - tcost, leverage


def trend_filtered_returns(
    index_level: pd.Series,
    raw_returns: pd.Series,
    risk_free_annual: float = 0.04,
    sma_window: int = 200,
    transaction_cost_bps: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    sma = index_level.rolling(sma_window, min_periods=sma_window // 2).mean()
    in_market = (index_level > sma).shift(1).fillna(False).astype(float)
    out_returns = pd.Series(risk_free_annual / 252, index=raw_returns.index)
    blended = in_market * raw_returns + (1 - in_market) * out_returns
    switches = in_market.diff().abs().fillna(0)
    return blended - switches * transaction_cost_bps / 10_000.0, in_market


def inverse_trend_hedge(
    index_level: pd.Series,
    raw_returns: pd.Series,
    sma_window: int = 200,
    hedge_size: float = 0.5,
    transaction_cost_bps: float = 2.0,
) -> pd.Series:
    """Short-equity hedge that activates when universe is below its SMA.

    When in uptrend: flat (0 return). When in downtrend: short hedge_size of
    the benchmark. Produces returns with NEGATIVE correlation to long sleeves
    — that's where the diversification Sharpe lift comes from.
    """
    sma = index_level.rolling(sma_window, min_periods=sma_window // 2).mean()
    is_downtrend = (index_level < sma).shift(1).fillna(False).astype(float)
    hedge_returns = -hedge_size * is_downtrend * raw_returns
    switches = is_downtrend.diff().abs().fillna(0)
    return hedge_returns - switches * hedge_size * transaction_cost_bps / 10_000.0


# ---------------------------------------------------------------------------
# Selection sleeves — runs a SignalStrategy long-only top quintile
# ---------------------------------------------------------------------------
def run_selection_sleeve(strategy, dataset, config: BacktestConfig) -> pd.Series:
    result = run_strategy_backtest(
        dataset=dataset,
        strategy=strategy,
        config=config,
    )
    return result.net_returns


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def annualize(returns: pd.Series) -> dict:
    clean = returns.dropna()
    if len(clean) < 2:
        return {}
    cum = (1 + clean).cumprod()
    n = len(clean)
    ann_ret = cum.iloc[-1] ** (252 / n) - 1
    std = clean.std(ddof=0)
    sharpe = clean.mean() / std * np.sqrt(252) if std > 0 else 0.0
    downside = clean[clean < 0]
    sortino = clean.mean() / downside.std(ddof=0) * np.sqrt(252) if len(downside) and downside.std(ddof=0) > 0 else 0.0
    dd = (cum / cum.cummax() - 1).min()
    return {
        "Cumulative Return": float(cum.iloc[-1] - 1),
        "Annualized Return": float(ann_ret),
        "Annualized Volatility": float(std * np.sqrt(252)),
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Calmar Ratio": float(ann_ret / abs(dd)) if dd != 0 else 0.0,
        "Max Drawdown": float(dd),
        "Win Rate": float((clean > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default="2025-01-01")
    p.add_argument("--output-dir", default="book_results")
    p.add_argument("--target-vol", type=float, default=0.16)
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--sma-window", type=int, default=200)
    p.add_argument("--rf-annual", type=float, default=0.04)
    p.add_argument("--selection-top-pct", type=float, default=0.20)
    p.add_argument("--selection-max-pos", type=float, default=0.08)
    p.add_argument("--tearsheet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Wharton {args.start} → {args.end}...")
    dataset = load_wharton_research_dataset(
        file_path="backtester/WhartonDataSource.parquet",
        start_date=args.start,
        end_date=args.end,
    )
    print(f"  {len(dataset.tickers)} tickers, {len(dataset.prices)} trading days")

    benchmark = dataset.benchmark_returns
    bench_index = (1 + benchmark.fillna(0)).cumprod()
    bench_m = annualize(benchmark)
    print(f"\n{'='*60}\n  Benchmark — Equal-Weight Universe\n{'='*60}")
    print(f"  Ann Return: {bench_m['Annualized Return']:+.2%}  "
          f"Sharpe: {bench_m['Sharpe Ratio']:.3f}  "
          f"MaxDD: {bench_m['Max Drawdown']:.2%}")

    # Long-only selection config
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

    sleeves = {}

    # Timing sleeve 1: vol-managed (raw beta with vol targeting)
    print(f"\n{'='*60}\n  Sleeve 1 — Vol-Managed Equity (target {args.target_vol:.0%})\n{'='*60}")
    vm_returns, vm_leverage = vol_managed_returns(
        benchmark, target_vol=args.target_vol, max_leverage=args.max_leverage,
    )
    sleeves["Vol-Managed Equity"] = vm_returns
    m = annualize(vm_returns)
    print(f"  Ann Return: {m['Annualized Return']:+.2%}  Sharpe: {m['Sharpe Ratio']:.3f}  "
          f"MaxDD: {m['Max Drawdown']:.2%}  AvgLev: {vm_leverage.mean():.2f}")

    # Timing sleeve 2: trend-filtered vol-managed
    print(f"\n{'='*60}\n  Sleeve 2 — Trend-Filtered Vol-Managed\n{'='*60}")
    tf_returns, in_market = trend_filtered_returns(
        bench_index, vm_returns, risk_free_annual=args.rf_annual, sma_window=args.sma_window,
    )
    sleeves["Trend-Filtered Equity"] = tf_returns
    m = annualize(tf_returns)
    print(f"  Ann Return: {m['Annualized Return']:+.2%}  Sharpe: {m['Sharpe Ratio']:.3f}  "
          f"MaxDD: {m['Max Drawdown']:.2%}  InMarket: {in_market.mean():.0%}")


    # Selection sleeves — each is run through the shared weight backtester and
    # then gets a trend filter applied on top.
    selection_strategies = [
        MomentumStrategy(lookback_days=126, skip_days=21),
        MeanReversionStrategy(window=100),
        LowVolatilityStrategy(window=126),
    ]
    for i, strat in enumerate(selection_strategies, start=3):
        label = f"{strat.name} + Trend"
        print(f"\n{'='*60}\n  Sleeve {i} — {label} (long-only top {args.selection_top_pct:.0%})\n{'='*60}")
        try:
            raw = run_selection_sleeve(strat, dataset, sel_config)
            ret, _ = trend_filtered_returns(
                bench_index, raw,
                risk_free_annual=args.rf_annual, sma_window=args.sma_window,
            )
            sleeves[label] = ret
            m_raw = annualize(raw)
            m = annualize(ret)
            print(f"  Raw:    Ret={m_raw['Annualized Return']:+.2%}  Sharpe={m_raw['Sharpe Ratio']:.3f}  MaxDD={m_raw['Max Drawdown']:.2%}")
            print(f"  +Trend: Ret={m['Annualized Return']:+.2%}  Sharpe={m['Sharpe Ratio']:.3f}  MaxDD={m['Max Drawdown']:.2%}")
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()

    # Correlation matrix between sleeves (key for diversification)
    sleeve_frame = pd.DataFrame(sleeves)
    corr = sleeve_frame.corr()
    print(f"\n{'='*60}\n  Sleeve Correlation Matrix\n{'='*60}")
    print(corr.round(3).to_string())
    corr.to_csv(output_dir / "sleeve_correlation.csv")

    # Combined book — try multiple combination methods and pick the best
    print(f"\n{'='*60}\n  Combined Books — Multiple Methods\n{'='*60}")

    # Equal weight
    ew = sleeve_frame.mean(axis=1)
    m_ew = annualize(ew)
    print(f"  Equal-Weight:      Sharpe={m_ew['Sharpe Ratio']:.3f}  "
          f"Ret={m_ew['Annualized Return']:+.2%}  Vol={m_ew['Annualized Volatility']:.2%}  "
          f"MaxDD={m_ew['Max Drawdown']:.2%}")

    # Risk parity (equal risk contribution — down-weights redundant correlated sleeves)
    rp = combine_strategy_returns(
        {name: ret for name, ret in sleeves.items()},
        method="risk_parity", lookback=252, rebalance_freq="ME",
    )
    m_rp = annualize(rp)
    print(f"  Risk-Parity:       Sharpe={m_rp['Sharpe Ratio']:.3f}  "
          f"Ret={m_rp['Annualized Return']:+.2%}  Vol={m_rp['Annualized Volatility']:.2%}  "
          f"MaxDD={m_rp['Max Drawdown']:.2%}")

    # Inverse volatility
    iv = combine_strategy_returns(
        {name: ret for name, ret in sleeves.items()},
        method="inverse_vol", lookback=252, rebalance_freq="ME",
    )
    m_iv = annualize(iv)
    print(f"  Inverse-Vol:       Sharpe={m_iv['Sharpe Ratio']:.3f}  "
          f"Ret={m_iv['Annualized Return']:+.2%}  Vol={m_iv['Annualized Volatility']:.2%}  "
          f"MaxDD={m_iv['Max Drawdown']:.2%}")

    # HRP
    hrp = combine_strategy_returns(
        {name: ret for name, ret in sleeves.items()},
        method="hrp", lookback=252, rebalance_freq="ME",
    )
    m_hrp = annualize(hrp)
    print(f"  HRP:               Sharpe={m_hrp['Sharpe Ratio']:.3f}  "
          f"Ret={m_hrp['Annualized Return']:+.2%}  Vol={m_hrp['Annualized Volatility']:.2%}  "
          f"MaxDD={m_hrp['Max Drawdown']:.2%}")

    # Apply trend filter as a META-overlay on each combination. The universe
    # SMA timing signal is nearly orthogonal to factor selection, so applying
    # it to the aggregated book adds real diversification benefit.
    meta_candidates = {
        "Equal-Weight": (ew, m_ew),
        "Risk-Parity": (rp, m_rp),
        "Inverse-Vol": (iv, m_iv),
        "HRP": (hrp, m_hrp),
    }

    for combo_name, (combo_series, _) in list(meta_candidates.items()):
        overlay, _ = trend_filtered_returns(
            bench_index, combo_series,
            risk_free_annual=args.rf_annual, sma_window=args.sma_window,
        )
        m_overlay = annualize(overlay)
        print(f"  {combo_name}+Trend:      Sharpe={m_overlay['Sharpe Ratio']:.3f}  "
              f"Ret={m_overlay['Annualized Return']:+.2%}  Vol={m_overlay['Annualized Volatility']:.2%}  "
              f"MaxDD={m_overlay['Max Drawdown']:.2%}")
        meta_candidates[f"{combo_name}+Trend"] = (overlay, m_overlay)

    # Use HRP as the fixed combination method — no selection based on in-sample
    # Sharpe. HRP is chosen a priori because it is robust to covariance
    # estimation error and requires no expected-return inputs (López de Prado
    # 2016). Picking the "best" method by full-sample Sharpe is lookahead bias.
    combined = hrp
    combined_metrics = m_hrp
    sleeves["Combined Book (HRP)"] = combined
    print(f"\n  Combined Book (HRP, fixed a priori):")
    print(f"  Sharpe:     {combined_metrics['Sharpe Ratio']:.3f}")
    print(f"  Ann Return: {combined_metrics['Annualized Return']:+.2%}")
    print(f"  Sortino:    {combined_metrics['Sortino Ratio']:.3f}")
    print(f"  Calmar:     {combined_metrics['Calmar Ratio']:.3f}")
    print(f"  MaxDD:      {combined_metrics['Max Drawdown']:.2%}")

    # Tearsheets are opt-in so result folders stay lightweight by default.
    if args.tearsheet:
        tearsheet_dir = output_dir / "tearsheets"
        tearsheet_dir.mkdir(parents=True, exist_ok=True)
        try:
            generate_tearsheet(
                combined, benchmark=benchmark, strategy_name="Combined Book (HRP)",
                output_path=tearsheet_dir / "Combined_Book.png",
            )
            for name, ret in sleeves.items():
                if name == "Combined Book (HRP)":
                    continue
                generate_tearsheet(
                    ret, benchmark=benchmark, strategy_name=name,
                    output_path=tearsheet_dir / f"{name.replace(' ', '_').replace('-', '_')}.png",
                )
        except Exception as exc:
            print(f"  tearsheet error: {exc}")

    # Stress test
    print(f"\n{'='*60}\n  Stress Test — Combined Book\n{'='*60}")
    stress = regime_report(combined)
    stress.to_csv(output_dir / "stress_regimes.csv", index=False)
    bad = stress.nsmallest(5, "cumulative_return")[["regime", "cumulative_return", "max_drawdown", "sharpe"]]
    for _, row in bad.iterrows():
        print(f"  {row['regime']}: ret={row['cumulative_return']:+.2%}  "
              f"dd={row['max_drawdown']:.2%}  sharpe={row['sharpe']:.2f}")
    dd_table = worst_drawdowns(combined, top_n=3)
    print(f"\n  Top 3 drawdowns:")
    for _, row in dd_table.iterrows():
        recovery = row["recovery"] if pd.notna(row["recovery"]) else "(open)"
        rec_str = recovery.date() if hasattr(recovery, "date") else recovery
        print(f"    {row['start'].date()} → {row['trough'].date()} → {rec_str}: depth={row['depth']:.2%}")

    # Master plot
    fig, ax = plt.subplots(figsize=(14, 8))
    palette = plt.get_cmap("tab10")
    for i, (name, ret) in enumerate(sleeves.items()):
        cum = (1 + ret.fillna(0)).cumprod() - 1
        lw = 3 if "Combined" in name else 1.5
        alpha = 1.0 if "Combined" in name else 0.7
        ax.plot(cum.index, cum.values, label=name, linewidth=lw, color=palette(i % 10), alpha=alpha)
    bench_cum = (1 + benchmark.fillna(0)).cumprod() - 1
    ax.plot(bench_cum.index, bench_cum.values, label="Equal-Weight Universe (Benchmark)",
            linewidth=2, linestyle="--", color="black", alpha=0.8)
    ax.set_title("5-Sleeve Alpha Book — Combined vs Individual",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    rows = [{**bench_m, "strategy": "Equal-Weight Benchmark"}]
    for name, ret in sleeves.items():
        m = annualize(ret)
        m["strategy"] = name
        rows.append(m)
    summary = pd.DataFrame(rows)
    summary = summary[
        ["strategy", "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
         "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Win Rate"]
    ]
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(f"\n{'='*60}\n  Final Summary\n{'='*60}")
    print(summary.round(4).to_string(index=False))

    return combined_metrics


if __name__ == "__main__":
    main()
