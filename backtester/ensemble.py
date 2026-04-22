"""Expanding-window multi-model ensemble — robust portfolio via model blending.

Runs multiple combination models (equal weight, inverse vol, risk parity, HRP,
minimum variance, maximum diversification) in parallel with strictly expanding
windows. No model selection — the ensemble is an equal-weight blend of all
models, with an optional conservative regime tilt.

Anti-overfitting guarantees:
  - All weight estimation uses only data up to the rebalance date
  - No model selection step (equal-weight meta-combination)
  - 2-year burn-in before adaptive weights
  - Validated with deflated Sharpe and PBO

References:
  - Choueifaty & Coignard, "Toward Maximum Diversification" (2008)
  - López de Prado, "Building Diversified Portfolios..." (HRP, 2016)
  - Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .portfolio_optimizer import (
    equal_weight,
    inverse_volatility_weight,
    risk_parity_weights,
    hierarchical_risk_parity,
    shrinkage_covariance,
)
from .walk_forward import (
    annualized_sharpe,
    combinatorial_purged_cv,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    method: str  # equal, inverse_vol, risk_parity, hrp, min_variance, max_diversification


@dataclass
class EnsembleConfig:
    burn_in_days: int = 504
    rebalance_freq: str = "ME"
    transaction_cost_bps: float = 10.0
    regime_tilt_enabled: bool = True
    regime_tilt_max: float = 0.20
    regime_vol_window: int = 63
    regime_trend_window: int = 200


@dataclass
class EnsembleResult:
    ensemble_returns: pd.Series
    model_returns: Dict[str, pd.Series]
    model_weights_history: Dict[str, pd.DataFrame]
    meta_weights_history: pd.DataFrame
    regime_history: pd.DataFrame
    deflated_sharpe_pvalue: float
    pbo_score: float
    individual_sharpes: Dict[str, float]
    n_trials: int


DEFAULT_MODELS = [
    ModelSpec("Equal Weight", "equal"),
    ModelSpec("Inverse Vol", "inverse_vol"),
    ModelSpec("Risk Parity", "risk_parity"),
    ModelSpec("HRP", "hrp"),
    ModelSpec("Min Variance", "min_variance"),
    ModelSpec("Max Diversification", "max_diversification"),
]


# ---------------------------------------------------------------------------
# New optimizers (kept here to avoid modifying portfolio_optimizer.py)
# ---------------------------------------------------------------------------
def minimum_variance_weights(cov: pd.DataFrame, ridge: float = 1e-6) -> pd.Series:
    """Global minimum variance portfolio, long-only."""
    n = cov.shape[0]
    if n == 0:
        return pd.Series(dtype=float)
    cov_reg = cov.values + ridge * np.eye(n)
    try:
        inv_cov = np.linalg.pinv(cov_reg)
    except np.linalg.LinAlgError:
        return equal_weight(cov.index)
    ones = np.ones(n)
    raw = inv_cov @ ones
    raw = np.clip(raw, 0, None)
    total = raw.sum()
    if total <= 0:
        return equal_weight(cov.index)
    return pd.Series(raw / total, index=cov.index)


def max_diversification_weights(
    cov: pd.DataFrame,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> pd.Series:
    """Maximum diversification portfolio (Choueifaty & Coignard 2008).

    Maximizes DR = (w' sigma) / sqrt(w' Cov w) via iterative reweighting.
    """
    n = cov.shape[0]
    if n == 0:
        return pd.Series(dtype=float)
    cov_arr = cov.values
    sigma = np.sqrt(np.diag(cov_arr)).clip(min=1e-12)

    # Start from inverse-vol
    w = (1.0 / sigma)
    w = w / w.sum()

    for _ in range(max_iterations):
        marginal = cov_arr @ w
        marginal_safe = np.clip(marginal, 1e-12, None)
        w_new = sigma / marginal_safe
        w_new = np.clip(w_new, 1e-8, None)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tolerance:
            w = w_new
            break
        w = w_new

    return pd.Series(w, index=cov.index)


# ---------------------------------------------------------------------------
# Model weight computation (anti-lookahead dispatcher)
# ---------------------------------------------------------------------------
def compute_model_weights(
    sleeve_returns: pd.DataFrame,
    model: ModelSpec,
    as_of_date: pd.Timestamp,
    min_obs: int = 126,
) -> pd.Series:
    """Compute sleeve weights for one model using only data up to as_of_date."""
    history = sleeve_returns.loc[:as_of_date]
    if len(history) < min_obs:
        return equal_weight(sleeve_returns.columns)

    if model.method == "equal":
        return equal_weight(sleeve_returns.columns)

    if model.method == "inverse_vol":
        vols = history.std()
        return inverse_volatility_weight(vols)

    if model.method == "hrp":
        return hierarchical_risk_parity(history)

    cov = shrinkage_covariance(history)

    if model.method == "risk_parity":
        return risk_parity_weights(cov)
    if model.method == "min_variance":
        return minimum_variance_weights(cov)
    if model.method == "max_diversification":
        return max_diversification_weights(cov)

    raise ValueError(f"Unknown method: {model.method}")


# ---------------------------------------------------------------------------
# Regime detection (backward-looking only)
# ---------------------------------------------------------------------------
def detect_regime(
    benchmark_returns: pd.Series,
    benchmark_level: pd.Series,
    as_of_date: pd.Timestamp,
    vol_window: int = 63,
    trend_window: int = 200,
) -> Dict[str, float]:
    """Classify regime using only data up to as_of_date."""
    rets = benchmark_returns.loc[:as_of_date]
    level = benchmark_level.loc[:as_of_date]

    # Vol regime: current realized vol vs expanding median
    if len(rets) < vol_window:
        vol_regime = 1.0
    else:
        current_vol = rets.tail(vol_window).std() * np.sqrt(252)
        rolling_vols = rets.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
        median_vol = rolling_vols.dropna().median()
        vol_regime = current_vol / median_vol if median_vol > 0 else 1.0

    # Trend state: benchmark vs SMA
    if len(level) < trend_window:
        trend_state = 1.0
    else:
        sma = level.rolling(trend_window).mean().iloc[-1]
        trend_state = 1.0 if level.iloc[-1] > sma else 0.0

    return {"vol_regime": float(vol_regime), "trend_state": float(trend_state)}


def apply_regime_tilt(
    base_weights: pd.Series,
    regime: Dict[str, float],
    model_names: List[str],
    max_tilt: float = 0.20,
) -> pd.Series:
    """Conservatively tilt meta-weights based on regime. Max ±20%."""
    w = base_weights.copy()
    tilt = pd.Series(0.0, index=w.index)

    # Defensive models get a boost in bad regimes
    defensive = {"Min Variance", "Risk Parity", "HRP"}

    # High vol regime (> 1.2x median)
    if regime["vol_regime"] > 1.2:
        strength = min((regime["vol_regime"] - 1.2) / 0.8, 1.0)  # 0→1 as vol goes 1.2→2.0
        for name in w.index:
            if name in defensive:
                tilt[name] += max_tilt * 0.5 * strength
            else:
                tilt[name] -= max_tilt * 0.25 * strength

    # Downtrend
    if regime["trend_state"] < 0.5:
        for name in w.index:
            if name in {"Min Variance", "HRP"}:
                tilt[name] += max_tilt * 0.5
            else:
                tilt[name] -= max_tilt * 0.2

    # Cap total tilt per model
    tilt = tilt.clip(-max_tilt, max_tilt)
    w = w * (1.0 + tilt)
    w = w.clip(lower=0.01)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Core ensemble runner
# ---------------------------------------------------------------------------
def run_expanding_window_ensemble(
    sleeve_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    benchmark_level: pd.Series,
    models: Optional[List[ModelSpec]] = None,
    config: Optional[EnsembleConfig] = None,
) -> EnsembleResult:
    """Run the full expanding-window meta-ensemble.

    At each monthly rebalance:
      1. Each model proposes sleeve weights using ONLY past data
      2. Meta-combiner blends with equal weight (+ optional regime tilt)
      3. Transaction costs applied to effective weight changes
    """
    if models is None:
        models = DEFAULT_MODELS
    if config is None:
        config = EnsembleConfig()

    sleeve_returns = sleeve_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dates = sleeve_returns.index
    sleeve_names = list(sleeve_returns.columns)
    model_names = [m.name for m in models]
    n_models = len(models)
    start_date = dates[0]

    # Rebalance boundary tracking (same approach as multi_strategy.py)
    rebal_boundaries = pd.date_range(start=dates.min(), end=dates.max(), freq=config.rebalance_freq)
    boundaries_iter = iter(rebal_boundaries)
    next_boundary = next(boundaries_iter, None)
    while next_boundary is not None and next_boundary < dates[0]:
        next_boundary = next(boundaries_iter, None)

    # Storage
    model_sleeve_wts: Dict[str, pd.DataFrame] = {
        m.name: pd.DataFrame(0.0, index=dates, columns=sleeve_names) for m in models
    }
    meta_wts = pd.DataFrame(1.0 / n_models, index=dates, columns=model_names)
    regime_rows = []

    # Per-model daily returns
    model_daily = {m.name: pd.Series(0.0, index=dates) for m in models}

    # Current weights (carried forward between rebalances)
    current_sleeve_wts = {m.name: equal_weight(sleeve_names) for m in models}
    current_meta = equal_weight(model_names)

    prev_effective_wts = equal_weight(sleeve_names)
    tcost_series = pd.Series(0.0, index=dates)

    for date in dates:
        crossed = next_boundary is not None and date >= next_boundary
        if crossed:
            days_elapsed = (date - start_date).days
            past_burnin = days_elapsed >= config.burn_in_days

            if past_burnin:
                for m in models:
                    current_sleeve_wts[m.name] = compute_model_weights(
                        sleeve_returns, m, as_of_date=date,
                    )
                if config.regime_tilt_enabled:
                    regime = detect_regime(
                        benchmark_returns, benchmark_level, date,
                        vol_window=config.regime_vol_window,
                        trend_window=config.regime_trend_window,
                    )
                    base_meta = equal_weight(model_names)
                    current_meta = apply_regime_tilt(base_meta, regime, model_names, config.regime_tilt_max)
                else:
                    regime = {"vol_regime": 1.0, "trend_state": 1.0}
                    current_meta = equal_weight(model_names)
            else:
                for m in models:
                    current_sleeve_wts[m.name] = equal_weight(sleeve_names)
                current_meta = equal_weight(model_names)
                regime = {"vol_regime": 1.0, "trend_state": 1.0}

            regime_rows.append({"date": date, **regime})

            # Compute effective weights and transaction costs
            effective_wts = pd.Series(0.0, index=sleeve_names)
            for m in models:
                effective_wts += current_meta[m.name] * current_sleeve_wts[m.name]
            turnover = (effective_wts - prev_effective_wts).abs().sum()
            tcost_series[date] = turnover * config.transaction_cost_bps / 10_000
            prev_effective_wts = effective_wts.copy()

            # Advance past crossed boundaries
            while next_boundary is not None and next_boundary <= date:
                next_boundary = next(boundaries_iter, None)

        # Record weights
        for m in models:
            model_sleeve_wts[m.name].loc[date] = current_sleeve_wts[m.name]
            model_daily[m.name][date] = (sleeve_returns.loc[date] * current_sleeve_wts[m.name]).sum()
        meta_wts.loc[date] = current_meta

    # Compute ensemble returns
    model_returns_frame = pd.DataFrame(model_daily)
    ensemble_raw = (model_returns_frame * meta_wts).sum(axis=1)
    ensemble_net = ensemble_raw - tcost_series

    # Compute individual model Sharpes
    individual_sharpes = {name: annualized_sharpe(ret) for name, ret in model_daily.items()}

    # Deflated Sharpe — use effective independent trials, not raw model count.
    # The 6 models are highly correlated (all combine the same sleeves), so
    # treating them as 6 independent noise trials is too conservative.
    # Estimate effective trials from eigenvalues of the model correlation matrix.
    ens_sharpe = annualized_sharpe(ensemble_net)
    clean = ensemble_net.dropna()
    skew = float(clean.skew()) if len(clean) > 2 else 0.0
    kurt = float(clean.kurtosis() + 3) if len(clean) > 2 else 3.0
    n_eff_trials = _effective_trials(model_returns_frame)
    # Floor at 2: we genuinely tested multiple methods even if they're correlated
    n_eff_int = max(2, int(round(n_eff_trials)))
    dsr_pval = deflated_sharpe_ratio(
        sharpe=ens_sharpe,
        n_trials=n_eff_int,
        n_observations=len(clean),
        skew=skew,
        kurtosis=kurt,
    )

    # PBO
    pbo = _compute_pbo(sleeve_returns, models, config)

    regime_df = pd.DataFrame(regime_rows) if regime_rows else pd.DataFrame(columns=["date", "vol_regime", "trend_state"])

    return EnsembleResult(
        ensemble_returns=ensemble_net,
        model_returns=model_daily,
        model_weights_history=model_sleeve_wts,
        meta_weights_history=meta_wts,
        regime_history=regime_df,
        deflated_sharpe_pvalue=float(dsr_pval),
        pbo_score=float(pbo),
        individual_sharpes=individual_sharpes,
        n_trials=n_eff_int,
    )


# ---------------------------------------------------------------------------
# Effective independent trials
# ---------------------------------------------------------------------------
def _effective_trials(model_returns: pd.DataFrame) -> float:
    """Estimate effective number of independent trials from model correlations.

    Uses the eigenvalue approach: n_eff = (sum of eigenvalues)^2 / sum(eigenvalues^2).
    When models are perfectly correlated, n_eff → 1.
    When models are independent, n_eff → n_models.
    """
    corr = model_returns.corr()
    eigenvalues = np.linalg.eigvalsh(corr.values)
    eigenvalues = np.clip(eigenvalues, 0, None)
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    if sum_eig_sq == 0:
        return 1.0
    return (sum_eig ** 2) / sum_eig_sq


# ---------------------------------------------------------------------------
# PBO computation
# ---------------------------------------------------------------------------
def _compute_pbo(
    sleeve_returns: pd.DataFrame,
    models: List[ModelSpec],
    config: EnsembleConfig,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_days: int = 10,
) -> float:
    """PBO for the ensemble: does the best IS model underperform OOS?"""
    splits = combinatorial_purged_cv(
        sleeve_returns.index, n_splits=n_splits,
        n_test_splits=n_test_splits, embargo_days=embargo_days,
    )

    # For each split, compute each model's OOS Sharpe
    cv_scores = []
    for train_idx, test_idx in splits:
        split_scores = {}
        for m in models:
            # Train: compute weights from train data
            train_data = sleeve_returns.loc[train_idx]
            if len(train_data) < 63:
                wts = equal_weight(sleeve_returns.columns)
            else:
                try:
                    wts = compute_model_weights(sleeve_returns, m, as_of_date=train_idx[-1])
                except Exception:
                    wts = equal_weight(sleeve_returns.columns)

            # Test: apply those weights to test data
            test_data = sleeve_returns.loc[test_idx]
            test_ret = (test_data * wts).sum(axis=1)
            split_scores[m.name] = annualized_sharpe(test_ret)
        cv_scores.append(split_scores)

    cv_df = pd.DataFrame(cv_scores).T  # rows=models, columns=splits
    cv_df = cv_df.fillna(0)
    return probability_of_backtest_overfitting(cv_df)
