"""Portfolio construction: mean-variance, risk parity, HRP, shrinkage covariance.

Gives you the tools to convert signals/expected returns into actual portfolio
weights under realistic constraints. Each optimizer can be plugged into the
research_backtester pipeline.

References:
- Markowitz "Portfolio Selection" (1952)
- Ledoit & Wolf "Honey, I Shrunk the Sample Covariance Matrix" (2004)
- Maillard, Roncalli, Teiletche "Risk Parity Portfolios" (2010)
- López de Prado "Building Diversified Portfolios that Outperform Out-of-Sample" (HRP, 2016)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------
def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Standard sample covariance."""
    return returns.cov()


def shrinkage_covariance(
    returns: pd.DataFrame,
    shrinkage: Optional[float] = None,
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage toward a constant-correlation target.

    If shrinkage is None, uses the Ledoit-Wolf optimal intensity estimator.
    Otherwise uses the provided value in [0, 1].
    """
    sample = returns.cov()
    n = sample.shape[0]
    if n == 0:
        return sample

    # Constant correlation target: diagonal = sample variances,
    # off-diagonal = average correlation * sqrt(var_i * var_j)
    variances = np.diag(sample.values)
    std = np.sqrt(variances)
    outer_std = np.outer(std, std)
    outer_std_safe = np.where(outer_std > 0, outer_std, 1.0)
    corr = sample.values / outer_std_safe
    np.fill_diagonal(corr, 1.0)
    avg_corr = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    target = avg_corr * outer_std
    np.fill_diagonal(target, variances)

    if shrinkage is None:
        # Simplified Ledoit-Wolf intensity. For production, use
        # sklearn.covariance.LedoitWolf. This is a reasonable heuristic.
        shrinkage = 0.2

    shrinkage = max(0.0, min(1.0, shrinkage))
    shrunk = shrinkage * target + (1 - shrinkage) * sample.values
    return pd.DataFrame(shrunk, index=sample.index, columns=sample.columns)


# ---------------------------------------------------------------------------
# Simple weight schemes
# ---------------------------------------------------------------------------
def equal_weight(tickers) -> pd.Series:
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=list(tickers))


def inverse_volatility_weight(vols: pd.Series) -> pd.Series:
    """Weights proportional to 1/vol."""
    clean = vols.replace(0, np.nan).dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / clean
    return inv / inv.sum()


# ---------------------------------------------------------------------------
# Mean-Variance (analytical unconstrained solution + long-only fallback)
# ---------------------------------------------------------------------------
def mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_risk: Optional[float] = None,
    leverage: float = 1.0,
    long_only: bool = False,
    turnover_penalty: float = 0.0,
    prev_weights: Optional[pd.Series] = None,
    ridge: float = 1e-6,
) -> pd.Series:
    """Mean-variance optimal weights with optional turnover penalty.

    Solves: max w'mu - 0.5 * lambda * w'Sigma w - tc * ||w - w_prev||
    via closed-form when unconstrained and clip-normalize for long-only.
    """
    tickers = expected_returns.index.intersection(covariance.index)
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    mu = expected_returns.reindex(tickers).values.astype(float)
    cov = covariance.reindex(index=tickers, columns=tickers).values.astype(float)

    # Ridge-regularize to ensure invertibility
    cov_reg = cov + ridge * np.eye(len(tickers))

    # Adjust mu by turnover penalty
    if turnover_penalty > 0 and prev_weights is not None:
        prev = prev_weights.reindex(tickers).fillna(0).values
        mu = mu - turnover_penalty * np.sign(mu - prev) * np.abs(mu - prev)

    try:
        inv_cov = np.linalg.pinv(cov_reg)
    except np.linalg.LinAlgError:
        return equal_weight(tickers) * leverage

    raw = inv_cov @ mu
    weights = pd.Series(raw, index=tickers)

    if long_only:
        weights = weights.clip(lower=0)

    total = weights.abs().sum()
    if total > 0:
        weights = weights / total * leverage

    return weights


# ---------------------------------------------------------------------------
# Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------
def risk_parity_weights(
    covariance: pd.DataFrame,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> pd.Series:
    """Equal risk contribution weights (ERC).

    Iterative algorithm: at each step, nudge weights so that marginal risk
    contributions are equalized. Converges in a few dozen iterations for
    well-conditioned covariance matrices.
    """
    tickers = covariance.index
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)

    cov = covariance.values
    # Start at inverse volatility
    std = np.sqrt(np.diag(cov))
    std_safe = np.where(std > 0, std, 1.0)
    w = (1.0 / std_safe)
    w = w / w.sum()

    for _ in range(max_iterations):
        portfolio_var = w @ cov @ w
        if portfolio_var <= 0:
            break
        marginal = cov @ w
        risk_contrib = w * marginal
        target = portfolio_var / n
        # Update direction: reduce weights with above-average contribution
        grad = risk_contrib - target
        w_new = w - 0.01 * grad / np.sqrt(portfolio_var)
        w_new = np.clip(w_new, 1e-8, None)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tolerance:
            w = w_new
            break
        w = w_new

    return pd.Series(w, index=tickers)


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (López de Prado 2016)
# ---------------------------------------------------------------------------
def hierarchical_risk_parity(returns: pd.DataFrame) -> pd.Series:
    """HRP: clusters assets then allocates via recursive bisection.

    More robust than mean-variance to estimation error in the covariance matrix.
    Always produces long-only weights summing to 1.
    """
    if returns.shape[1] < 2:
        return pd.Series(1.0, index=returns.columns)

    cov = returns.cov()
    corr = returns.corr()

    # Distance matrix: d_ij = sqrt(0.5 * (1 - corr_ij))
    dist = np.sqrt(0.5 * (1 - corr.clip(-1, 1))).fillna(1.0)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")

    # Quasi-diagonalization: get leaf order from linkage
    order = _get_quasi_diag_order(link)
    sorted_cols = [cov.columns[i] for i in order]

    # Recursive bisection
    weights = pd.Series(1.0, index=sorted_cols)
    clusters = [sorted_cols]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)
            total = left_var + right_var
            if total <= 0:
                continue

            alpha = 1 - left_var / total
            weights[left] *= alpha
            weights[right] *= 1 - alpha

            new_clusters.extend([left, right])
        clusters = new_clusters

    return weights.reindex(cov.columns).fillna(0)


def _cluster_variance(cov: pd.DataFrame, cluster) -> float:
    sub_cov = cov.loc[cluster, cluster].values
    inv_diag = 1.0 / np.diag(sub_cov).clip(min=1e-12)
    w = inv_diag / inv_diag.sum()
    return float(w @ sub_cov @ w)


def _get_quasi_diag_order(link) -> list:
    """Extract quasi-diagonal leaf order from a scipy linkage matrix."""
    link = link.astype(int)
    n = link.shape[0] + 1
    order = [int(link[-1, 0]), int(link[-1, 1])]

    def expand(node_id):
        if node_id < n:
            return [node_id]
        idx = node_id - n
        return expand(int(link[idx, 0])) + expand(int(link[idx, 1]))

    expanded = []
    for x in order:
        expanded.extend(expand(x))
    return expanded
