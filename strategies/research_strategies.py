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


def build_default_strategy_suite() -> list[SignalStrategy]:
    return [
        MeanReversionStrategy(),
        MomentumStrategy(),
        LowVolatilityStrategy(),
    ]
