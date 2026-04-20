"""Per-strategy FF3+MOM factor attribution report.

Runs the default strategy suite through the backtester, regresses each
strategy's net return on [MKT, SMB, HML, MOM] proxies built from the same
universe, and writes a table of alphas, betas, and t-stats.

Usage:
  python run_factor_attribution.py --start 2005-01-01 --end 2024-12-31 \
                                   --out factor_attribution.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import research_backtester as rbt
from backtester.attribution import factor_attribution
from backtester.factor_proxies import build_ff3_mom_factors
from backtester.research_data import load_wharton_research_dataset
from strategies.research_strategies import build_default_strategy_suite

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "backtester" / "WhartonDataSource.parquet"


def _run_one_strategy(strategy, dataset, config):
    strat_out = strategy.generate(dataset)
    cfg = rbt.merge_backtest_config(config, strat_out.backtest_overrides)
    weights = rbt.build_target_weights(strat_out.scores, dataset.returns, cfg)
    result = rbt.run_weight_backtest(
        prices=dataset.prices,
        target_weights=weights,
        config=cfg,
        benchmark_returns=dataset.benchmark_returns,
        strategy_name=strat_out.name,
    )
    return strat_out.name, result.net_returns


def build_attribution_table(start: str, end: str) -> pd.DataFrame:
    dataset = load_wharton_research_dataset(
        file_path=str(DATA_PATH), start_date=start, end_date=end,
    )
    factors = build_ff3_mom_factors(dataset)
    config = rbt.BacktestConfig(
        rebalance_frequency="ME",
        signal_lag=1,
        transaction_cost_bps=10.0,
        long_quantile=0.2,
        short_quantile=0.0,
        leverage=1.0,
        max_position_weight=0.08,
        construction_method="inverse_vol",
        long_only=True,
    )

    rows = []
    for strategy in build_default_strategy_suite():
        name, net_returns = _run_one_strategy(strategy, dataset, config)
        report = factor_attribution(net_returns, factors)
        if not report:
            continue
        row = {
            "strategy": name,
            "n_obs": report["n_obs"],
            "alpha_annual": report["alpha_annualized"],
            "alpha_tstat": report["alpha_tstat"],
            "r_squared": report["r_squared"],
        }
        for factor_name, payload in report["factors"].items():
            row[f"beta_{factor_name}"] = payload["beta"]
            row[f"tstat_{factor_name}"] = payload["tstat"]
            row[f"contrib_{factor_name}"] = payload["annualized_contribution"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("strategy")


def _format_markdown(table: pd.DataFrame) -> str:
    cols = ["alpha_annual", "alpha_tstat", "beta_MKT", "tstat_MKT",
            "beta_SMB", "tstat_SMB", "beta_HML", "tstat_HML",
            "beta_MOM", "tstat_MOM", "r_squared", "n_obs"]
    header = "| strategy | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [header, sep]
    for name, row in table.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, np.nan)
            if c.startswith("alpha_annual") or c.startswith("beta_") or c == "r_squared":
                vals.append(f"{v:+.3f}")
            elif c.startswith("tstat") or c == "alpha_tstat":
                vals.append(f"{v:+.2f}")
            else:
                vals.append(f"{int(v)}")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default="factor_attribution.csv")
    ap.add_argument("--markdown", default="factor_attribution.md")
    args = ap.parse_args()

    table = build_attribution_table(args.start, args.end)
    out_path = REPO_ROOT / args.out
    table.to_csv(out_path)
    md_path = REPO_ROOT / args.markdown
    md_path.write_text(
        f"# Factor attribution: FF3 + MOM (proxies)\n\n"
        f"Window: {args.start} → {args.end}\n\n"
        f"{_format_markdown(table)}\n"
    )
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
