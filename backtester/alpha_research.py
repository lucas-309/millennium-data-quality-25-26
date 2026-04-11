"""Alpha research framework: IC, quantile returns, decay, orthogonalization.

These are the tools you need to evaluate whether a signal is actually predictive
before dumping it into a backtest. Without IC analysis you're flying blind.

Key references:
- Grinold & Kahn, "Active Portfolio Management" (2000)
- López de Prado, "Advances in Financial Machine Learning" (2018)
- Qian, Hua, Sorensen, "Quantitative Equity Portfolio Management" (2007)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------
def information_coefficient(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.Series:
    """Compute cross-sectional IC per date.

    IC is the correlation between signal scores and realized forward returns,
    computed across the cross-section of assets on each date. A consistently
    positive IC means the signal ranks assets correctly.

    Args:
        scores: Signal scores, shape (T, N).
        forward_returns: Realized forward returns over the prediction horizon.
        method: "pearson" for raw IC or "spearman" for Rank IC.

    Returns:
        IC time series (one value per date).
    """
    common_dates = scores.index.intersection(forward_returns.index)
    ics = {}
    for date in common_dates:
        pair = pd.concat([scores.loc[date], forward_returns.loc[date]], axis=1).dropna()
        if len(pair) < 3:
            ics[date] = np.nan
            continue
        ics[date] = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    return pd.Series(ics, dtype=float).sort_index()


def ic_summary(ic_series: pd.Series) -> dict:
    """Key IC stats: mean, std, IR (IC/std), hit rate."""
    clean = ic_series.dropna()
    if clean.empty:
        return {"mean_ic": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "hit_rate": np.nan}
    mean = clean.mean()
    std = clean.std(ddof=0)
    return {
        "mean_ic": mean,
        "ic_std": std,
        "ic_ir": mean / std * np.sqrt(252) if std > 0 else np.nan,
        "hit_rate": (clean > 0).mean(),
        "n_obs": len(clean),
    }


def ic_decay(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10, 21, 63, 126, 252),
    method: str = "spearman",
) -> pd.DataFrame:
    """IC as a function of forward horizon.

    A strong alpha has high IC at short horizons that decays over time.
    If IC is flat, the signal has no short-term edge.
    """
    rows = []
    for h in horizons:
        forward = returns.shift(-h).rolling(h, min_periods=max(1, h // 2)).sum()
        ic = information_coefficient(scores, forward, method=method)
        summary = ic_summary(ic)
        summary["horizon_days"] = h
        rows.append(summary)
    return pd.DataFrame(rows).set_index("horizon_days")


def signal_half_life(scores: pd.DataFrame) -> float:
    """Estimate signal half-life via AR(1) fit on the average autocorrelation.

    Half-life = -ln(2) / ln(phi) where phi is the AR(1) coefficient.
    A signal with a 20-day half-life decays 50% over 20 days.
    """
    demeaned = scores.sub(scores.mean(axis=1), axis=0)
    stacked = demeaned.stack()
    lag1 = demeaned.shift(1).stack()
    pair = pd.concat([stacked, lag1], axis=1, join="inner").dropna()
    if len(pair) < 10:
        return np.nan
    phi = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    if not (0 < phi < 1):
        return np.nan
    return -np.log(2) / np.log(phi)


# ---------------------------------------------------------------------------
# Quantile analysis
# ---------------------------------------------------------------------------
def quantile_returns(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute mean forward return per quantile per date.

    A monotonic relationship between quantile and return is the gold standard
    evidence that a signal has predictive power.

    Returns:
        DataFrame indexed by date with columns Q1..Qn (Q1 = lowest score).
    """
    common_dates = scores.index.intersection(forward_returns.index)
    rows = {}
    for date in common_dates:
        scores_row = scores.loc[date].dropna()
        returns_row = forward_returns.loc[date].reindex(scores_row.index).dropna()
        if len(returns_row) < n_quantiles:
            continue
        try:
            quantile_labels = pd.qcut(
                scores_row.reindex(returns_row.index),
                q=n_quantiles,
                labels=[f"Q{i+1}" for i in range(n_quantiles)],
                duplicates="drop",
            )
        except ValueError:
            continue
        grouped = returns_row.groupby(quantile_labels, observed=True).mean()
        rows[date] = grouped
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.sort_index()


def quantile_spread(qr: pd.DataFrame) -> pd.Series:
    """Top-quantile minus bottom-quantile return spread."""
    if qr.empty:
        return pd.Series(dtype=float)
    q_cols = sorted(qr.columns, key=lambda x: int(str(x).lstrip("Q")))
    return (qr[q_cols[-1]] - qr[q_cols[0]]).rename("spread")


def quantile_monotonicity(qr: pd.DataFrame) -> float:
    """How monotonic is the quantile→return relationship?

    Returns the Spearman correlation between quantile rank and mean return.
    +1 = perfectly monotonic (top quantile highest return, etc.).
    """
    if qr.empty:
        return np.nan
    mean_by_quantile = qr.mean()
    ranks = pd.Series(range(len(mean_by_quantile)), index=mean_by_quantile.index)
    return mean_by_quantile.corr(ranks, method="spearman")


# ---------------------------------------------------------------------------
# Signal turnover and persistence
# ---------------------------------------------------------------------------
def signal_turnover(
    scores: pd.DataFrame,
    top_quantile: float = 0.2,
) -> pd.Series:
    """Fraction of top-quantile names that change from day to day.

    High turnover → strategy will incur high transaction costs.
    """
    if scores.empty:
        return pd.Series(dtype=float)

    top_sets = []
    top_n_per_date = []
    for date in scores.index:
        row = scores.loc[date].dropna()
        if row.empty:
            top_sets.append(set())
            top_n_per_date.append(0)
            continue
        n_top = max(1, int(len(row) * top_quantile))
        top = set(row.nlargest(n_top).index)
        top_sets.append(top)
        top_n_per_date.append(n_top)

    turnover = [np.nan]
    for i in range(1, len(top_sets)):
        prev, curr = top_sets[i - 1], top_sets[i]
        if not prev or not curr:
            turnover.append(np.nan)
            continue
        churn = len(curr.symmetric_difference(prev)) / 2
        turnover.append(churn / max(len(curr), 1))
    return pd.Series(turnover, index=scores.index)


# ---------------------------------------------------------------------------
# Orthogonalization
# ---------------------------------------------------------------------------
def orthogonalize(
    target: pd.DataFrame,
    against: List[pd.DataFrame],
) -> pd.DataFrame:
    """Residualize `target` against other signals cross-sectionally per date.

    For each date, run OLS: target_scores ~ other_signals. The residuals are
    the portion of the target signal that is not explained by the others.
    This lets you isolate pure alpha from already-captured factors.
    """
    if not against:
        return target.copy()

    residual = pd.DataFrame(index=target.index, columns=target.columns, dtype=float)

    for date in target.index:
        y = target.loc[date].dropna()
        if y.empty:
            continue
        # Build regressor matrix
        X_parts = []
        for other in against:
            if date not in other.index:
                continue
            X_parts.append(other.loc[date].reindex(y.index))
        if not X_parts:
            residual.loc[date, y.index] = y.values
            continue
        X = pd.concat(X_parts, axis=1).dropna()
        y_aligned = y.reindex(X.index)
        if len(y_aligned) < 3:
            residual.loc[date, y.index] = y.values
            continue

        X_mat = np.column_stack([np.ones(len(X)), X.values])
        y_vec = y_aligned.values
        try:
            coefs, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
            y_hat = X_mat @ coefs
            resid = y_vec - y_hat
            residual.loc[date, X.index] = resid
        except np.linalg.LinAlgError:
            residual.loc[date, y.index] = y.values

    return residual


# ---------------------------------------------------------------------------
# Signal combination
# ---------------------------------------------------------------------------
def ic_weighted_combination(
    signals: dict,
    returns: pd.DataFrame,
    lookback: int = 63,
) -> pd.DataFrame:
    """Combine multiple signals weighted by their rolling IC.

    Signals with higher recent IC get larger weights in the combined score.
    """
    # Compute IC for each signal
    ics = {name: information_coefficient(s, returns.shift(-1), method="spearman")
           for name, s in signals.items()}

    ic_frame = pd.DataFrame(ics)
    rolling_ic = ic_frame.rolling(lookback, min_periods=lookback // 4).mean()

    # Normalize weights so positive ICs sum to 1
    weights = rolling_ic.clip(lower=0.0)
    weight_sum = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(weight_sum, axis=0).fillna(0)

    combined = None
    for name, sig in signals.items():
        w = weights[name]
        weighted = sig.mul(w.reindex(sig.index), axis=0)
        combined = weighted if combined is None else combined.add(weighted, fill_value=0)
    return combined
