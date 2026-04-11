"""Quality factor (Novy-Marx gross profitability proxy).

Classic Novy-Marx factor is Gross Profit / Total Assets. When fundamental data
isn't available, we use a price-based proxy: stocks with high trailing return
stability and low drawdowns ("quality of returns"). This is a reasonable stand-in
when you don't have balance sheet data.

Reference: Novy-Marx "The Other Side of Value" (2013).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.research_data import ResearchDataset
from strategies.research_strategies import SignalStrategy


class QualityStrategy(SignalStrategy):
    name = "Quality"
    motivation = "Prefer stocks with consistent, high-quality return profiles."
    economic_rationale = "High-quality businesses produce steadier cash flows and earnings, which get underpriced relative to junk."
    why_it_works = "Quality has strong long-run premium and low drawdowns during crises."
    why_it_fails = "Can underperform in speculative junk rallies and rapid recovery regimes."

    def __init__(
        self,
        lookback_days: int = 252,
        vol_weight: float = 0.4,
        drawdown_weight: float = 0.3,
        return_weight: float = 0.3,
    ):
        self.lookback_days = lookback_days
        self.vol_weight = vol_weight
        self.drawdown_weight = drawdown_weight
        self.return_weight = return_weight

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        returns = dataset.returns
        prices = dataset.prices

        # Cross-sectionally z-score each component so the weighted sum is balanced
        def zscore_cs(frame):
            mu = frame.mean(axis=1, skipna=True)
            sigma = frame.std(axis=1, skipna=True).replace(0, np.nan)
            return frame.sub(mu, axis=0).div(sigma, axis=0)

        # Component 1: inverse rolling volatility (lower vol = higher quality)
        vol = returns.rolling(self.lookback_days, min_periods=self.lookback_days // 2).std()
        inv_vol_score = zscore_cs(-vol)

        # Component 2: max drawdown (less negative = higher quality)
        rolling_max = prices.rolling(self.lookback_days, min_periods=self.lookback_days // 2).max()
        drawdown = prices.div(rolling_max) - 1.0
        # max_drawdown is the *most negative* point in the rolling drawdown series
        max_drawdown = drawdown.rolling(self.lookback_days, min_periods=1).min()
        inv_dd_score = zscore_cs(max_drawdown)  # less-negative = higher = better

        # Component 3: Sharpe-like ratio (return / vol)
        mean_ret = returns.rolling(self.lookback_days, min_periods=self.lookback_days // 2).mean()
        sharpe_like = zscore_cs(mean_ret.div(vol.replace(0, np.nan)))

        combined = (
            self.vol_weight * inv_vol_score +
            self.drawdown_weight * inv_dd_score +
            self.return_weight * sharpe_like
        )

        return combined.replace([np.inf, -np.inf], np.nan)
