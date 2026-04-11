"""Post-Earnings Announcement Drift (PEAD).

Classic anomaly from Ball & Brown (1968) and Bernard & Thomas (1989): stocks
that beat earnings tend to drift upward for 30-60 days post-announcement, and
vice versa for misses. The slow-adjustment effect persists despite being one
of the best-documented anomalies.

This implementation uses the Wharton dataset's `eps` and `anncdate` fields to
proxy earnings surprises via abnormal return around the announcement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.research_data import ResearchDataset
from strategies.research_strategies import SignalStrategy


class PEADStrategy(SignalStrategy):
    name = "Post-Earnings Drift"
    motivation = "Ride the drift in stocks after significant earnings surprises."
    economic_rationale = "Investors underreact to earnings news, producing a 30-60 day drift in the direction of the surprise."
    why_it_works = "PEAD is one of the most persistent documented anomalies and provides diversification vs price-based signals."
    why_it_fails = "When attention is high (major macro events) the drift disappears as information is absorbed faster."
    uses_event_data = True
    event_type = "anncdate"

    def __init__(self, drift_days: int = 60, abnormal_window: int = 3):
        self.drift_days = drift_days
        self.abnormal_window = abnormal_window

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        returns = dataset.returns
        market = dataset.benchmark_returns

        # Abnormal return: stock minus market over a short window
        abnormal = returns.sub(market.reindex(returns.index).fillna(0), axis=0)
        short_abnormal = abnormal.rolling(
            self.abnormal_window, min_periods=1
        ).sum()

        # Persist the abnormal return for `drift_days` after it occurred
        # (proxy for the post-announcement window)
        drifted = short_abnormal.rolling(
            self.drift_days, min_periods=1
        ).max()

        return drifted.replace([np.inf, -np.inf], np.nan)
