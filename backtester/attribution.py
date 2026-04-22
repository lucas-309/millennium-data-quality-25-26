"""Performance attribution: factor-based and Brinson sector attribution.

Factor attribution via OLS regression decomposes strategy returns into factor
betas and residual alpha. Brinson attribution decomposes active returns into
sector allocation and security selection effects.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def factor_attribution(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Dict[str, object]:
    """OLS regression: strategy ~ factors. Decompose total return into factor
    contributions + residual alpha.

    Returns a dict with betas, alpha, t-stats, R², and contribution breakdown.
    """
    daily_rf = risk_free_rate / 252
    excess = (strategy_returns - daily_rf).dropna()
    aligned = pd.concat([excess, factor_returns], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return {}

    y = aligned.iloc[:, 0].values
    factor_names = list(aligned.columns[1:])
    X = np.column_stack([np.ones(len(aligned)), aligned.iloc[:, 1:].values])

    try:
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {}

    if not np.all(np.isfinite(coefs)):
        return {}

    with np.errstate(all="ignore"):
        y_hat = X @ coefs
        residual = y - y_hat
        ss_res = float((residual ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 and np.isfinite(ss_res) else 0.0

    # Standard errors
    n = len(y)
    k = X.shape[1]
    sigma2 = ss_res / max(n - k, 1)
    cov_coefs = np.linalg.pinv(X.T @ X) * sigma2
    with np.errstate(all="ignore"):
        std_errors = np.sqrt(np.diag(cov_coefs))
        t_stats = coefs / np.where(std_errors > 0, std_errors, np.nan)

    # Factor contribution to total return
    factor_contribs = {}
    for i, name in enumerate(factor_names):
        # Total return contribution = beta * mean factor return, annualized
        factor_mean = aligned[name].mean()
        factor_contribs[name] = {
            "beta": float(coefs[i + 1]),
            "tstat": float(t_stats[i + 1]),
            "annualized_contribution": float(coefs[i + 1] * factor_mean * 252),
        }

    return {
        "alpha_daily": float(coefs[0]),
        "alpha_annualized": float(coefs[0] * 252),
        "alpha_tstat": float(t_stats[0]),
        "r_squared": float(r_squared),
        "n_obs": int(n),
        "factors": factor_contribs,
    }


def brinson_attribution(
    portfolio_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_weights: pd.Series,
    benchmark_returns: pd.Series,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """Brinson-Fachler sector attribution.

    Decomposes active return into:
    - Allocation effect: (w_p - w_b) * (r_b_sector - r_b_total)
    - Selection effect: w_b * (r_p - r_b_sector)
    - Interaction effect: (w_p - w_b) * (r_p - r_b_sector)

    Returns a per-sector attribution DataFrame.
    """
    df = pd.DataFrame({
        "port_weight": portfolio_weights,
        "bench_weight": benchmark_weights,
        "port_return": portfolio_returns,
        "bench_return": benchmark_returns,
    }).fillna(0)
    df["sector"] = df.index.map(sector_map).fillna("OTHER")

    sector_agg = df.groupby("sector").apply(
        lambda g: pd.Series({
            "port_weight": g["port_weight"].sum(),
            "bench_weight": g["bench_weight"].sum(),
            "port_return": (g["port_weight"] * g["port_return"]).sum() / g["port_weight"].sum() if g["port_weight"].sum() > 0 else 0,
            "bench_return": (g["bench_weight"] * g["bench_return"]).sum() / g["bench_weight"].sum() if g["bench_weight"].sum() > 0 else 0,
        }),
        include_groups=False,
    )

    total_bench_return = (df["bench_weight"] * df["bench_return"]).sum()

    sector_agg["allocation"] = (
        (sector_agg["port_weight"] - sector_agg["bench_weight"])
        * (sector_agg["bench_return"] - total_bench_return)
    )
    sector_agg["selection"] = sector_agg["bench_weight"] * (
        sector_agg["port_return"] - sector_agg["bench_return"]
    )
    sector_agg["interaction"] = (
        (sector_agg["port_weight"] - sector_agg["bench_weight"])
        * (sector_agg["port_return"] - sector_agg["bench_return"])
    )
    sector_agg["total_active"] = (
        sector_agg["allocation"] + sector_agg["selection"] + sector_agg["interaction"]
    )
    return sector_agg


def return_contribution_ranking(
    portfolio_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank individual names by contribution to total portfolio return.

    Contribution_i = sum over time of (weight_i(t) * return_i(t+1)).
    """
    lagged_weights = portfolio_weights.shift(1).fillna(0)
    contribs = (lagged_weights * asset_returns).sum()
    contribs = contribs.sort_values(ascending=False)

    top = contribs.head(top_n).to_frame("contribution")
    top["rank_type"] = "contributor"

    bottom = contribs.tail(top_n).to_frame("contribution")
    bottom["rank_type"] = "detractor"

    return pd.concat([top, bottom])
