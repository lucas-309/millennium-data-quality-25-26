"""Factor models: rolling CAPM, residualization, sector/beta neutralization.

Used to strip systematic factor exposures from signals so you're trading pure
alpha, not rebranded market beta. Essential for market-neutral pods.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def rolling_market_model(
    returns: pd.DataFrame,
    market: pd.Series,
    window: int = 252,
    min_periods: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling CAPM regression per ticker: r_i = alpha_i + beta_i * r_market.

    Returns two DataFrames: (alpha, beta), each shape (T, N).
    """
    if min_periods is None:
        min_periods = max(window // 2, 20)

    aligned_market = market.reindex(returns.index)

    cov = returns.rolling(window, min_periods=min_periods).cov(aligned_market)
    var = aligned_market.rolling(window, min_periods=min_periods).var().replace(0, np.nan)
    beta = cov.div(var, axis=0)

    stock_mean = returns.rolling(window, min_periods=min_periods).mean()
    market_mean = aligned_market.rolling(window, min_periods=min_periods).mean()
    alpha = stock_mean.sub(beta.mul(market_mean, axis=0), axis=0)

    return alpha, beta


def residualize_returns(
    returns: pd.DataFrame,
    market: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Strip the market-model component from returns; keep only residuals.

    residual_i = r_i - (alpha_i + beta_i * r_market)

    Used for residual momentum / idiosyncratic reversal strategies.
    """
    alpha, beta = rolling_market_model(returns, market, window=window)
    aligned_market = market.reindex(returns.index)
    predicted = alpha.add(beta.mul(aligned_market, axis=0), axis=0)
    return returns.sub(predicted).replace([np.inf, -np.inf], np.nan)


def sector_neutralize(
    scores: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """Cross-sectionally demean scores within each sector per date.

    After neutralization, each sector has zero mean signal, so the portfolio
    has no sector bet. Residual alpha is within-sector dispersion.
    """
    if not sector_map:
        return scores.copy()

    # Build sector -> ticker list mapping
    sector_groups: Dict[str, list] = {}
    for ticker, sector in sector_map.items():
        if ticker in scores.columns:
            sector_groups.setdefault(sector, []).append(ticker)

    neutral = scores.copy()
    for _sector, tickers in sector_groups.items():
        if len(tickers) < 2:
            continue
        sub = scores[tickers]
        sector_mean = sub.mean(axis=1)
        neutral[tickers] = sub.sub(sector_mean, axis=0)

    return neutral


def beta_neutralize(
    scores: pd.DataFrame,
    betas: pd.DataFrame,
) -> pd.DataFrame:
    """Residualize scores against per-stock rolling betas cross-sectionally.

    After this, the signal has zero correlation with beta across the universe,
    so longs and shorts have offsetting beta exposure.
    """
    common_cols = scores.columns.intersection(betas.columns)
    neutral = scores.copy()

    for date in scores.index:
        if date not in betas.index:
            continue
        y = scores.loc[date, common_cols].dropna()
        x = betas.loc[date, common_cols].reindex(y.index).dropna()
        if len(x) < 3:
            continue
        y_aligned = y.reindex(x.index)
        x_values = x.values.astype(float)
        y_values = y_aligned.values.astype(float)
        centered_x = x_values - x_values.mean()
        centered_y = y_values - y_values.mean()
        denominator = centered_x @ centered_x
        if denominator <= 0:
            continue
        try:
            beta_coef = float((centered_x @ centered_y) / denominator)
            residual = centered_y - beta_coef * centered_x
            residual -= residual.mean()
            residual[np.abs(residual) < 1e-12] = 0.0
            neutral.loc[date, x.index] = residual
        except np.linalg.LinAlgError:
            continue

    return neutral


def factor_exposure_report(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free: float = 0.0,
) -> Dict[str, float]:
    """OLS: strategy_returns ~ factors. Report alpha, betas, R², t-stats.

    Standard Fama-French style factor attribution.
    """
    excess = strategy_returns - risk_free / 252
    pair = pd.concat([excess, factor_returns], axis=1, join="inner").dropna()
    if len(pair) < 10:
        return {}

    y = pair.iloc[:, 0].values
    X = np.column_stack([np.ones(len(pair)), pair.iloc[:, 1:].values])
    factor_names = list(pair.columns[1:])

    try:
        coefs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {}

    y_hat = X @ coefs
    resid = y - y_hat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors via (X'X)^-1 * sigma^2
    n = len(y)
    k = X.shape[1]
    sigma2 = ss_res / max(n - k, 1)
    try:
        cov = np.linalg.pinv(X.T @ X) * sigma2
        std_errors = np.sqrt(np.diag(cov))
        t_stats = coefs / np.where(std_errors > 0, std_errors, np.nan)
    except np.linalg.LinAlgError:
        t_stats = np.full(k, np.nan)

    report = {
        "alpha_annual": coefs[0] * 252,
        "alpha_tstat": t_stats[0],
        "r_squared": r_squared,
        "n_obs": n,
    }
    for i, name in enumerate(factor_names):
        report[f"beta_{name}"] = coefs[i + 1]
        report[f"tstat_{name}"] = t_stats[i + 1]
    return report


def fama_french_3_factor(
    market_returns: pd.Series,
    size_returns: Optional[pd.Series] = None,
    value_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build a Fama-French 3-factor DataFrame (MKT, SMB, HML).

    If size_returns or value_returns not provided, uses zeros (reduces to CAPM).
    """
    frame = pd.DataFrame({"MKT": market_returns})
    frame["SMB"] = size_returns if size_returns is not None else 0.0
    frame["HML"] = value_returns if value_returns is not None else 0.0
    return frame
