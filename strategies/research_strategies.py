from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtester.research_data import ResearchDataset


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    means = frame.mean(axis=1, skipna=True)
    stds = frame.std(axis=1, skipna=True).replace(0, np.nan)
    return frame.sub(means, axis=0).div(stds, axis=0)


@dataclass
class StrategyOutput:
    name: str
    scores: pd.DataFrame
    motivation: str = ""
    economic_rationale: str = ""
    why_it_works: str = ""
    why_it_fails: str = ""
    uses_event_data: bool = False
    event_type: Optional[str] = None
    backtest_overrides: dict[str, Any] = field(default_factory=dict)
    combine_in_portfolio: bool = True


class SignalStrategy(ABC):
    """Simple collaborator-facing strategy contract.

    Strategy authors only need to implement `generate_scores(dataset)` and
    return a DataFrame aligned with `dataset.prices`. Higher scores mean
    stronger long candidates. The backtester handles normalization, ranking,
    sizing, lag, turnover, and transaction costs.
    """

    name: str = ""
    motivation: str = ""
    economic_rationale: str = ""
    why_it_works: str = ""
    why_it_fails: str = ""
    uses_event_data: bool = False
    event_type: Optional[str] = None
    backtest_overrides: dict[str, Any] = {}
    combine_in_portfolio: bool = True

    def generate(self, dataset: ResearchDataset) -> StrategyOutput:
        raw_scores = self.generate_scores(dataset)
        scores = self.normalize_scores(raw_scores, dataset).replace([np.inf, -np.inf], np.nan)
        return StrategyOutput(
            name=self.name,
            scores=scores,
            motivation=self.motivation,
            economic_rationale=self.economic_rationale,
            why_it_works=self.why_it_works,
            why_it_fails=self.why_it_fails,
            uses_event_data=self.uses_event_data,
            event_type=self.event_type,
            backtest_overrides=dict(self.backtest_overrides),
            combine_in_portfolio=self.combine_in_portfolio,
        )

    def normalize_scores(self, raw_scores: pd.DataFrame, dataset: ResearchDataset) -> pd.DataFrame:
        return _cross_sectional_zscore(raw_scores)

    @abstractmethod
    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        raise NotImplementedError


class MeanReversionStrategy(SignalStrategy):
    name = "Mean Reversion"
    motivation = "Buy names trading below their rolling average and fade names trading above it."
    economic_rationale = "Large moves away from a recent price anchor often mean-revert once the shock passes."
    why_it_works = "The signal is easy to explain: below-average names score higher, above-average names score lower."
    why_it_fails = "Strong trends can stay extended for longer than the moving-average anchor suggests."

    def __init__(self, window: int = 100):
        self.window = window

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        prices = dataset.prices
        rolling_mean = prices.rolling(self.window, min_periods=max(self.window // 2, 2)).mean()
        return -(prices.div(rolling_mean.replace(0, np.nan)) - 1.0)


class MomentumStrategy(SignalStrategy):
    name = "Momentum"
    motivation = "Own recent winners and avoid recent losers."
    economic_rationale = "Prices often keep drifting in the same direction because information gets incorporated gradually."
    why_it_works = "It is a direct price-based signal that collaborators can implement and debug with a simple return calculation."
    why_it_fails = "Momentum can unwind hard during sharp reversals."

    def __init__(self, lookback_days: int = 126, skip_days: int = 21):
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        prices = dataset.prices
        return prices.shift(self.skip_days).div(prices.shift(self.lookback_days)) - 1.0


class CrossSectionalMomentumStrategy(MomentumStrategy):
    name = "Cross-Sectional Momentum"


class ShortTermReversalStrategy(SignalStrategy):
    name = "Short-Term Reversal"
    motivation = "Buy recent losers and fade recent winners over a short horizon."
    economic_rationale = "Very short-term moves can overshoot on news, liquidity, and crowding."
    why_it_works = "The implementation is just the negative of recent returns, so it is easy to extend or debug."
    why_it_fails = "Turnover is high and sustained trends can overpower the bounce."

    def __init__(self, lookback_days: int = 5):
        self.lookback_days = lookback_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        return -(dataset.prices.div(dataset.prices.shift(self.lookback_days)) - 1.0)


class ShortTermReversalStrategy(SignalStrategy):
    name = "Short-Term Reversal"
    motivation = "Short-term losers tend to bounce back over the following days."
    economic_rationale = "Overreaction to news and liquidity shocks creates transient price dislocations that mean-revert."
    why_it_works = "Daily/weekly reversal is one of the most persistent short-horizon anomalies in equities."
    why_it_fails = "Strongly trending markets suppress reversal; execution cost eats the signal at high turnover."

    def __init__(self, lookback_days: int = 5):
        self.lookback_days = lookback_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        # Negate recent return — biggest recent losers get the highest score
        # (i.e. long candidates), biggest recent winners get the lowest.
        return -(dataset.prices.div(dataset.prices.shift(self.lookback_days)) - 1.0)


class LowVolatilityStrategy(SignalStrategy):
    name = "Low Volatility"
    motivation = "Hold the calmer names and de-emphasize the noisiest ones."
    economic_rationale = "Lower-volatility stocks have historically delivered better risk-adjusted returns than their beta would suggest."
    why_it_works = "Rolling volatility is easy for collaborators to inspect and reason about."
    why_it_fails = "Speculative rallies can punish defensive portfolios."

    def __init__(self, window: int = 126):
        self.window = window

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        volatility = dataset.returns.rolling(self.window, min_periods=max(self.window // 2, 2)).std()
        return -volatility


class MLRidgeStrategy(SignalStrategy):
    name = "ML Ridge (rolling OLS)"
    motivation = "Let a rolling ridge regression blend momentum, short-term reversal, volatility, and market beta rather than hand-picking weights."
    economic_rationale = "Each input carries an independent risk premium. A data-driven linear combination lets their relative payoffs rotate with regime instead of relying on static weights."
    why_it_works = "Two-year rolling fit averages out single-period noise while still tracking coefficient drift. Ridge stabilises coefficients when momentum and reversal are correlated at the sampling horizon."
    why_it_fails = "Structural breaks (2008, 2020) briefly invalidate the pre-break fit; when all features decorrelate from forward returns, coefficients collapse toward zero."

    def __init__(
        self,
        training_lookback_days: int = 504,
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        reversal_lookback: int = 5,
        vol_lookback: int = 60,
        beta_lookback: int = 60,
        forward_horizon: int = 5,
        refit_frequency: str = "ME",
        ridge_lambda: float = 1.0,
        holdout_start: Optional[str] = None,
    ):
        self.training_lookback_days = training_lookback_days
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.reversal_lookback = reversal_lookback
        self.vol_lookback = vol_lookback
        self.beta_lookback = beta_lookback
        self.forward_horizon = forward_horizon
        self.refit_frequency = refit_frequency
        self.ridge_lambda = ridge_lambda
        self.holdout_start = holdout_start  # optional: refit only up to here for a hard hold-out

    def _build_features(self, dataset: ResearchDataset) -> dict[str, pd.DataFrame]:
        prices = dataset.prices
        returns = dataset.returns
        benchmark = dataset.benchmark_returns.reindex(returns.index).fillna(0.0)

        span = max(self.momentum_lookback - self.momentum_skip, 1)
        mom = prices.pct_change(span, fill_method=None).shift(self.momentum_skip)
        rev = -prices.pct_change(self.reversal_lookback, fill_method=None)
        vol = returns.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        cov = returns.rolling(self.beta_lookback, min_periods=self.beta_lookback // 2).cov(benchmark)
        var = benchmark.rolling(self.beta_lookback, min_periods=self.beta_lookback // 2).var().replace(0, np.nan)
        beta = cov.div(var, axis=0)

        return {
            "mom": _cross_sectional_zscore(mom),
            "rev": _cross_sectional_zscore(rev),
            "vol": _cross_sectional_zscore(vol),
            "beta": _cross_sectional_zscore(beta),
        }

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        from backtester.research_backtester import _get_rebalance_dates

        returns = dataset.returns
        features = self._build_features(dataset)
        feature_names = list(features.keys())
        n_features = len(feature_names)

        panel = np.stack(
            [features[k].reindex_like(returns).values for k in feature_names],
            axis=-1,
        )

        log_r = np.log1p(returns.fillna(0.0).values)
        T, N = log_r.shape
        cum = np.concatenate([np.zeros((1, N)), np.cumsum(log_r, axis=0)], axis=0)
        fwd_log = np.full((T, N), np.nan)
        H = self.forward_horizon
        if T > H:
            fwd_log[: T - H] = cum[H + 1 : T + 1] - cum[1 : T - H + 1]
        fwd_returns = np.expm1(fwd_log)

        rebalance_dates = _get_rebalance_dates(returns.index, self.refit_frequency)
        scores_out = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
        positions = {d: i for i, d in enumerate(returns.index)}
        holdout_ts = pd.Timestamp(self.holdout_start) if self.holdout_start else None

        for date in rebalance_dates:
            pos = positions.get(date)
            if pos is None:
                continue
            train_end_pos = pos - H - 1
            if holdout_ts is not None and date >= holdout_ts:
                holdout_pos = positions.get(holdout_ts)
                if holdout_pos is not None:
                    train_end_pos = min(train_end_pos, holdout_pos - H - 1)
            train_start_pos = max(train_end_pos - self.training_lookback_days + 1, 0)
            if train_end_pos - train_start_pos < 100:
                continue

            X_slice = panel[train_start_pos : train_end_pos + 1].reshape(-1, n_features)
            y_slice = fwd_returns[train_start_pos : train_end_pos + 1].reshape(-1)
            mask = np.all(np.isfinite(X_slice), axis=1) & np.isfinite(y_slice)
            if mask.sum() < 100:
                continue
            X = X_slice[mask]
            y = y_slice[mask]

            XtX = X.T @ X + self.ridge_lambda * np.eye(n_features)
            try:
                coefs = np.linalg.solve(XtX, X.T @ y)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coefs)):
                continue

            X_today = panel[pos]
            valid = np.all(np.isfinite(X_today), axis=1)
            X_masked = np.where(valid[:, None], X_today, 0.0)
            with np.errstate(all="ignore"):
                preds = X_masked @ coefs
            scores_out.iloc[pos] = np.where(valid, preds, np.nan)

        return scores_out.ffill()


def build_default_strategy_suite() -> list[SignalStrategy]:
    return [
        MeanReversionStrategy(),
        MomentumStrategy(),
        LowVolatilityStrategy(),
        MLRidgeStrategy(),
    ]
