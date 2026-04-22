"""Multi-strategy combination — build a Millennium-style book of uncorrelated alphas.

The multi-manager shop approach: each strategy is an independent pod with its
own risk budget. Combine their net returns via risk parity, equal weight, or
HRP to build a diversified book where each pod contributes equal risk.

This is where you actually get the "sum of uncorrelated alphas" effect that
makes multi-manager funds work.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .portfolio_optimizer import (
    equal_weight,
    inverse_volatility_weight,
    risk_parity_weights,
    hierarchical_risk_parity,
    shrinkage_covariance,
)


def combine_strategy_returns(
    strategy_returns: Dict[str, pd.Series],
    method: str = "risk_parity",
    lookback: int = 252,
    rebalance_freq: str = "ME",
    vol_target_annual: Optional[float] = None,
) -> pd.Series:
    """Combine multiple strategy return series into a single portfolio return.

    Args:
        strategy_returns: Name -> daily return series.
        method: One of "equal", "inverse_vol", "risk_parity", "hrp".
        lookback: Rolling window for covariance estimation.
        rebalance_freq: Pandas freq string for rebalancing dates.
        vol_target_annual: If provided, scale the combined return to hit this
                           annualized vol target.

    Returns:
        Combined portfolio return series.
    """
    if not strategy_returns:
        return pd.Series(dtype=float)

    returns_frame = pd.DataFrame(strategy_returns).sort_index()
    returns_frame = returns_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # pd.date_range on "ME"/"QE"/etc. yields calendar period-ends, many of
    # which fall on weekends/holidays and are NOT in returns_frame.index.
    # We therefore track the next rebalance boundary and trigger on the first
    # trading day that crosses it.
    rebalance_boundaries = pd.date_range(
        start=returns_frame.index.min(),
        end=returns_frame.index.max(),
        freq=rebalance_freq,
    )
    boundaries_iter = iter(rebalance_boundaries)
    next_boundary = next(boundaries_iter, None)
    # Skip boundaries that precede the first trading day (so the very first
    # trading day doesn't double-count as a rebalance crossing).
    while next_boundary is not None and next_boundary < returns_frame.index[0]:
        next_boundary = next(boundaries_iter, None)

    weights_history = pd.DataFrame(
        0.0, index=returns_frame.index, columns=returns_frame.columns
    )

    current_weights = None
    for date in returns_frame.index:
        crossed_boundary = next_boundary is not None and date >= next_boundary
        if current_weights is None or crossed_boundary:
            history = returns_frame.loc[:date].tail(lookback)
            if len(history) < max(lookback // 4, 20):
                current_weights = equal_weight(returns_frame.columns)
            else:
                current_weights = _compute_weights(history, method)
            # Advance past every boundary we've already crossed so we don't
            # re-rebalance on the next day too.
            while next_boundary is not None and next_boundary <= date:
                next_boundary = next(boundaries_iter, None)
        weights_history.loc[date] = current_weights

    combined = (returns_frame * weights_history).sum(axis=1)

    # Optional vol targeting
    if vol_target_annual is not None and vol_target_annual > 0:
        realized_vol = combined.rolling(63, min_periods=20).std() * np.sqrt(252)
        scalar = (vol_target_annual / realized_vol.replace(0, np.nan)).clip(upper=3.0).fillna(1.0)
        combined = combined * scalar

    return combined.fillna(0.0)


def _compute_weights(returns_window: pd.DataFrame, method: str) -> pd.Series:
    if method == "equal":
        return equal_weight(returns_window.columns)

    if method == "inverse_vol":
        vols = returns_window.std()
        return inverse_volatility_weight(vols)

    cov = shrinkage_covariance(returns_window)

    if method == "risk_parity":
        return risk_parity_weights(cov)

    if method == "hrp":
        return hierarchical_risk_parity(returns_window)

    raise ValueError(f"Unknown combination method: {method}")
