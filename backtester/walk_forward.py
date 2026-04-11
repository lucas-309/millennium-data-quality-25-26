"""Walk-forward validation, purged K-fold, deflated Sharpe, probability of overfitting.

Every quant strategy needs out-of-sample validation. In-sample Sharpe ratios
are meaningless without multiple-testing adjustment and OOS verification.

References:
- López de Prado, "Advances in Financial Machine Learning" Ch 7 (Purged CV)
- Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014)
- Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest Overfitting" (2017)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Purged K-Fold Cross Validation (AFML, López de Prado)
# ---------------------------------------------------------------------------
def purged_kfold_splits(
    index: pd.DatetimeIndex,
    n_splits: int = 5,
    embargo_days: int = 10,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Purged K-fold splits with embargo period to prevent leakage.

    For each test fold, remove train samples that fall within embargo_days of
    the test period. Otherwise leakage from autocorrelated features or labels
    will inflate CV scores.

    Returns:
        List of (train_index, test_index) tuples.
    """
    n = len(index)
    fold_size = n // n_splits
    splits = []

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n
        test_idx = index[test_start:test_end]

        # Embargo: drop train points within embargo_days of test fold
        train_mask = np.ones(n, dtype=bool)
        embargo_start = max(0, test_start - embargo_days)
        embargo_end = min(n, test_end + embargo_days)
        train_mask[embargo_start:embargo_end] = False
        train_idx = index[train_mask]

        splits.append((train_idx, test_idx))

    return splits


def combinatorial_purged_cv(
    index: pd.DatetimeIndex,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_days: int = 10,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Combinatorial purged CV: all combinations of n_test_splits folds as test.

    Produces many more train/test splits than K-fold, letting you compute PBO
    with higher fidelity.
    """
    n = len(index)
    fold_size = n // n_splits
    fold_indices = []
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else n
        fold_indices.append((start, end))

    combinations = list(itertools.combinations(range(n_splits), n_test_splits))
    splits = []
    for test_folds in combinations:
        test_mask = np.zeros(n, dtype=bool)
        for fold_i in test_folds:
            start, end = fold_indices[fold_i]
            test_mask[start:end] = True

        train_mask = ~test_mask
        # Apply embargo
        for fold_i in test_folds:
            start, end = fold_indices[fold_i]
            embargo_start = max(0, start - embargo_days)
            embargo_end = min(n, end + embargo_days)
            train_mask[embargo_start:embargo_end] = False

        splits.append((index[train_mask], index[test_mask]))

    return splits


# ---------------------------------------------------------------------------
# Walk-forward execution
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    param_set: dict
    train_score: float
    test_score: float
    train_period: Tuple
    test_period: Tuple


def walk_forward_backtest(
    param_grid: Dict[str, list],
    train_fn: Callable[[dict, pd.DatetimeIndex], float],
    test_fn: Callable[[dict, pd.DatetimeIndex], float],
    index: pd.DatetimeIndex,
    n_splits: int = 5,
    embargo_days: int = 10,
) -> pd.DataFrame:
    """Grid search with purged K-fold CV.

    For each parameter combination, evaluate on each CV fold. Return a long
    DataFrame with rows = (fold, param_set, train_score, test_score).
    """
    # Enumerate all param combinations
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    param_dicts = [dict(zip(keys, combo)) for combo in combos]

    splits = purged_kfold_splits(index, n_splits=n_splits, embargo_days=embargo_days)

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        for params in param_dicts:
            train_score = train_fn(params, train_idx)
            test_score = test_fn(params, test_idx)
            rows.append({
                "fold": fold_i,
                **params,
                "train_score": train_score,
                "test_score": test_score,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & López de Prado 2014)
# ---------------------------------------------------------------------------
def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute probability that an observed Sharpe exceeds a noise benchmark
    after adjusting for multiple-testing bias.

    Returns a p-value-like probability in [0, 1]. Values near 1 suggest the
    Sharpe is genuinely high; values near 0 suggest it could be noise.
    """
    if n_trials <= 0 or n_observations <= 1:
        return np.nan

    # Expected maximum Sharpe from n independent trials of noise
    euler_mascheroni = 0.5772
    expected_max = (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # Variance of the Sharpe estimator
    variance = (1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2) / (n_observations - 1)
    if variance <= 0:
        return np.nan

    z = (sharpe - expected_max) / np.sqrt(variance)
    return stats.norm.cdf(z)


def annualized_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    excess = clean - risk_free_rate / trading_days
    std = excess.std(ddof=0)
    if std == 0:
        return np.nan
    return excess.mean() / std * np.sqrt(trading_days)


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (Bailey et al. 2017)
# ---------------------------------------------------------------------------
def probability_of_backtest_overfitting(cv_metrics: pd.DataFrame) -> float:
    """PBO: probability that the best in-sample strategy underperforms median OOS.

    Args:
        cv_metrics: DataFrame where rows are trials (parameter sets) and columns
                    are CV splits' OOS metrics.

    Returns:
        PBO in [0, 1]. PBO > 0.5 means your best IS strategy is worse than
        median OOS — textbook overfitting.
    """
    if cv_metrics.empty or cv_metrics.shape[1] < 2:
        return np.nan

    n_splits = cv_metrics.shape[1]
    # Use half for IS, half for OOS
    half = n_splits // 2
    split_combinations = list(itertools.combinations(range(n_splits), half))

    if len(split_combinations) == 0:
        return np.nan

    overfit_count = 0
    total = 0
    for is_splits in split_combinations:
        oos_splits = [j for j in range(n_splits) if j not in is_splits]
        is_scores = cv_metrics.iloc[:, list(is_splits)].mean(axis=1)
        oos_scores = cv_metrics.iloc[:, oos_splits].mean(axis=1)

        if is_scores.empty or oos_scores.empty:
            continue

        best_is_idx = is_scores.idxmax()
        oos_of_best = oos_scores.loc[best_is_idx]
        oos_median = oos_scores.median()

        if oos_of_best < oos_median:
            overfit_count += 1
        total += 1

    return overfit_count / total if total > 0 else np.nan
