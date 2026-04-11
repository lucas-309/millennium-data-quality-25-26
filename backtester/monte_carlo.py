"""Monte Carlo robustness: stationary block bootstrap for return CIs.

Block bootstrap preserves the autocorrelation structure of returns (which
naive iid bootstrap destroys). Use this to compute confidence intervals on
Sharpe, max drawdown, CAGR, etc.

Reference: Politis & Romano "The stationary bootstrap" (1994).
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd


def stationary_block_bootstrap_sample(
    returns: pd.Series,
    avg_block_length: int = 10,
    rng: np.random.Generator = None,
) -> pd.Series:
    """Generate a single stationary block bootstrap sample.

    Uses geometric block lengths with mean `avg_block_length`. This preserves
    autocorrelation up to roughly the block length.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(returns)
    if n == 0:
        return returns.copy()

    p = 1.0 / avg_block_length
    values = returns.values
    output = np.empty(n)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        length = rng.geometric(p)
        for j in range(length):
            if i >= n:
                break
            output[i] = values[(start + j) % n]
            i += 1

    return pd.Series(output, index=returns.index)


def bootstrap_confidence_interval(
    returns: pd.Series,
    metric_fn: Callable[[pd.Series], float],
    alpha: float = 0.05,
    n_iterations: int = 1000,
    avg_block_length: int = 10,
    seed: int = None,
) -> Tuple[float, float, float]:
    """Bootstrap (point, lower, upper) CI for a metric.

    Args:
        returns: Historical return series.
        metric_fn: Function computing a single metric from a return series.
        alpha: Significance level (0.05 → 95% CI).
        n_iterations: Number of bootstrap samples.
        avg_block_length: Expected block length in the bootstrap.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    point = metric_fn(returns)
    samples = np.empty(n_iterations)

    for i in range(n_iterations):
        resample = stationary_block_bootstrap_sample(returns, avg_block_length, rng)
        samples[i] = metric_fn(resample)

    lower = np.quantile(samples, alpha / 2)
    upper = np.quantile(samples, 1 - alpha / 2)
    return float(point), float(lower), float(upper)


def bootstrap_many_metrics(
    returns: pd.Series,
    metrics: dict,  # name -> Callable
    alpha: float = 0.05,
    n_iterations: int = 500,
    avg_block_length: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute bootstrap CIs for multiple metrics in one pass."""
    rng = np.random.default_rng(seed)
    point_values = {name: fn(returns) for name, fn in metrics.items()}
    samples = {name: np.empty(n_iterations) for name in metrics}

    for i in range(n_iterations):
        resample = stationary_block_bootstrap_sample(returns, avg_block_length, rng)
        for name, fn in metrics.items():
            samples[name][i] = fn(resample)

    rows = []
    for name, fn in metrics.items():
        s = samples[name]
        rows.append({
            "metric": name,
            "point": point_values[name],
            "lower": float(np.quantile(s, alpha / 2)),
            "upper": float(np.quantile(s, 1 - alpha / 2)),
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
        })
    return pd.DataFrame(rows).set_index("metric")
