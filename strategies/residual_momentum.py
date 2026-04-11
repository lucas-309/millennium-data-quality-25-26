"""Residual momentum (factor-neutral momentum).

Classic 12-1 momentum is dominated by market beta: high-beta stocks exhibit more
momentum. Residual momentum strips the market component first, leaving pure
stock-specific momentum. This is the preferred momentum signal in
market-neutral books.

Reference: Blitz, Huij, Martens "Residual Momentum" (2011).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.factor_models import residualize_returns
from backtester.research_data import ResearchDataset
from strategies.research_strategies import SignalStrategy


class ResidualMomentumStrategy(SignalStrategy):
    name = "Residual Momentum"
    motivation = "Trade stock-specific momentum after removing market beta exposure."
    economic_rationale = "Stripping the market factor isolates trends driven by company-specific information rather than systematic risk."
    why_it_works = "Residual momentum exhibits more persistence than raw momentum and is less prone to market-wide reversals."
    why_it_fails = "Factor regression error introduces estimation noise; residualization may strip too much in high-beta regimes."

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        beta_window: int = 252,
    ):
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.beta_window = beta_window

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        returns = dataset.returns
        market = dataset.benchmark_returns

        residuals = residualize_returns(returns, market, window=self.beta_window)

        # 12-1 momentum on residuals: cumulative residual return from t-lookback to t-skip
        cum_resid = residuals.rolling(self.lookback_days, min_periods=self.lookback_days // 2).sum()
        skip_resid = residuals.rolling(self.skip_days, min_periods=self.skip_days // 2).sum()
        score = cum_resid.shift(self.skip_days) - skip_resid.shift(self.skip_days)

        return score.replace([np.inf, -np.inf], np.nan)
