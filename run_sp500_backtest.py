"""Run the full strategy suite on the yfinance-backed S&P 500 universe.

Mirror of the Wharton-backed pipeline, but sourcing prices from the per-ticker
yfinance cache. Writes:
  sp500_results/summary.csv        — per-strategy Sharpe / return / vol / DD
  sp500_results/factor_attrib.csv  — FF3+MOM regression per strategy
  sp500_results/<Strategy>.csv     — daily gross/net returns per strategy
  sp500_results/report.md          — deck-ready summary

Requires the cache to be populated first:
  python run_sp500_fetch.py --start 2005-01-01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import research_backtester as rbt
from backtester.attribution import factor_attribution
from backtester.factor_proxies import build_ff3_mom_factors
from backtester.sp500_universe import load_sp500_universe
from backtester.yfinance_cache import list_cached_tickers
from backtester.yfinance_loader import load_yfinance_research_dataset
from strategies.research_strategies import build_default_strategy_suite

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "sp500_results"


def _summary_row(name: str, result: rbt.BacktestResult) -> dict:
    m = result.metrics
    audit = result.survivorship_audit or {}
    return {
        "strategy": name,
        "sharpe": m.get("Sharpe Ratio"),
        "sortino": m.get("Sortino Ratio"),
        "ann_return": m.get("Annualized Return"),
        "ann_vol": m.get("Annualized Volatility"),
        "max_drawdown": m.get("Max Drawdown"),
        "ann_turnover": m.get("Annualized Turnover"),
        "ann_tcost_drag": m.get("Annualized Transaction Cost Drag"),
        "universe_size": audit.get("universe_size"),
        "inception_biased_count": audit.get("inception_biased_count"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01",
                    help="window start (default 2010 to cover most SP500 names)")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--min-history", type=int, default=252)
    ap.add_argument("--long-quantile", type=float, default=0.2)
    ap.add_argument("--tcost-bps", type=float, default=10.0)
    ap.add_argument("--tag", default="",
                    help="suffix appended to output files (e.g. _q30 for a variant)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    cached = list_cached_tickers()
    if not cached:
        raise SystemExit("yfinance cache is empty — run `python run_sp500_fetch.py` first")
    print(f"cached tickers: {len(cached)}")

    snap = load_sp500_universe()
    dataset = load_yfinance_research_dataset(
        start_date=args.start, end_date=args.end,
        tickers=cached, min_history=args.min_history, sp500_frame=snap.frame,
    )
    print(f"dataset: {dataset.prices.shape[1]} tickers × "
          f"{dataset.prices.shape[0]} trading days "
          f"({dataset.prices.index.min().date()} → {dataset.prices.index.max().date()})")

    config = rbt.BacktestConfig(
        rebalance_frequency="ME",
        signal_lag=1,
        transaction_cost_bps=args.tcost_bps,
        long_quantile=args.long_quantile,
        short_quantile=0.0,
        leverage=1.0,
        max_position_weight=0.05,
        construction_method="inverse_vol",
        long_only=True,
    )

    factors = build_ff3_mom_factors(dataset)

    summary_rows = []
    attrib_rows = []
    for strategy in build_default_strategy_suite():
        strat_out = strategy.generate(dataset)
        cfg = rbt.merge_backtest_config(config, strat_out.backtest_overrides)
        weights = rbt.build_target_weights(strat_out.scores, dataset.returns, cfg)
        result = rbt.run_weight_backtest(
            prices=dataset.prices, target_weights=weights, config=cfg,
            benchmark_returns=dataset.benchmark_returns, strategy_name=strat_out.name,
        )
        summary_rows.append(_summary_row(strat_out.name, result))

        report = factor_attribution(result.net_returns, factors)
        if report:
            row = {
                "strategy": strat_out.name,
                "alpha_annual": report["alpha_annualized"],
                "alpha_tstat": report["alpha_tstat"],
                "r_squared": report["r_squared"],
                "n_obs": report["n_obs"],
            }
            for factor_name, payload in report["factors"].items():
                row[f"beta_{factor_name}"] = payload["beta"]
                row[f"tstat_{factor_name}"] = payload["tstat"]
                row[f"contrib_{factor_name}"] = payload["annualized_contribution"]
            attrib_rows.append(row)

        daily = pd.DataFrame({
            "gross_return": result.gross_returns,
            "net_return": result.net_returns,
            "turnover": result.turnover,
            "tcost": result.transaction_costs,
        })
        daily.to_csv(OUT_DIR / f"{strat_out.name.replace(' ', '_')}{args.tag}.csv")
        print(f"  {strat_out.name:32s}  Sharpe {result.metrics.get('Sharpe Ratio'):+.3f}  "
              f"AnnRet {result.metrics.get('Annualized Return'):+.3%}")

    summary = pd.DataFrame(summary_rows).set_index("strategy")
    attrib = pd.DataFrame(attrib_rows).set_index("strategy") if attrib_rows else pd.DataFrame()

    summary.to_csv(OUT_DIR / f"summary{args.tag}.csv")
    if not attrib.empty:
        attrib.to_csv(OUT_DIR / f"factor_attrib{args.tag}.csv")

    report_path = OUT_DIR / f"report{args.tag}.md"
    report_path.write_text(_format_report(summary, attrib, args, dataset))
    print(f"\nwrote {report_path}")
    print(summary.round(3).to_string())


def _df_to_markdown(df: pd.DataFrame) -> str:
    df = df.reset_index()
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("—")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_report(summary: pd.DataFrame, attrib: pd.DataFrame, args, dataset) -> str:
    parts = [
        f"# S&P 500 yfinance backtest",
        "",
        f"- Window: {args.start} → {args.end}",
        f"- Universe: {dataset.prices.shape[1]} tickers (of {len(load_sp500_universe().tickers)} "
        f"current S&P 500 members)",
        f"- Rebalance: monthly, inverse-vol, long-only, top {int(args.long_quantile*100)}% by score",
        f"- Transaction cost: {args.tcost_bps} bps per side",
        "",
        "## Performance summary",
        "",
        _df_to_markdown(summary.round(3)),
        "",
    ]
    if not attrib.empty:
        parts.extend([
            "## Factor attribution (FF3 + MOM proxies)",
            "",
            _df_to_markdown(attrib.round(3)),
            "",
        ])
    parts.extend([
        "## Notes & caveats",
        "",
        "- Universe is today's S&P 500 constituents (scraped from Wikipedia). This is",
        "  survivorship-biased — names that were in the index during the backtest but",
        "  are now delisted or removed are absent.",
        "- Market cap is a **current-shares proxy** (today's sharesOutstanding × daily",
        "  Adj Close). It tracks price but does not reflect buybacks or issuance.",
        "- EPS is the trailing-twelve-month scalar from yfinance.info, broadcast as a",
        "  constant panel. **This effectively disables the EPS Revision strategy** —",
        "  its signal is the change in EPS over time, and a constant panel has zero",
        "  change. Run on the Wharton pull to get a working EPS Revision backtest.",
        "- Returns use Adj Close (dividend-reinvested) so total return is captured.",
    ])
    return "\n".join(parts)


if __name__ == "__main__":
    main()
