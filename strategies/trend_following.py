"""Trend following with vol targeting.

Classic CTA approach applied to equities: long assets with positive recent
trend, short assets with negative trend, sized inversely to volatility so each
position contributes equal risk.

This is the signal that has historically made CTAs profitable over decades.
Works well when combined with mean-reversion strategies (negative correlation
during crashes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.research_data import ResearchDataset
from strategies.research_strategies import SignalStrategy


class TrendFollowingStrategy(SignalStrategy):
    name = "Trend Following"
    motivation = "Capture persistent price trends across the cross-section."
    economic_rationale = "Investors underreact to fundamental changes and herd into trends, creating drift that persists before corrections."
    why_it_works = "Trend signals have long-horizon persistence and negatively correlate with mean-reversion strategies."
    why_it_fails = "Trend strategies get whipsawed in ranging markets and hit during sharp reversals."

    def __init__(
        self,
        lookback_days: int = 252,
        vol_window: int = 63,
        min_periods: int = 100,
    ):
        self.lookback_days = lookback_days
        self.vol_window = vol_window
        self.min_periods = min_periods

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        prices = dataset.prices
        returns = dataset.returns

        # Trend signal: cumulative return over lookback
        trend = prices.div(prices.shift(self.lookback_days)) - 1.0

        # Volatility-adjusted: divide by rolling vol so each name has equal risk weight
        vol = returns.rolling(self.vol_window, min_periods=self.min_periods).std()
        vol_adjusted = trend.div(vol.replace(0, np.nan))

        return vol_adjusted.replace([np.inf, -np.inf], np.nan)
