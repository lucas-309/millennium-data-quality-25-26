"""Idiosyncratic short-term reversal.

Short-term reversal trades names that just moved sharply, betting they'll fade.
Raw reversal is contaminated by market beta. Residualizing against the market
first isolates the idiosyncratic shock, which is much more likely to revert.

Reference: Da, Liu, Schaumburg "A Closer Look at the Short-Term Return Reversal" (2014).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.factor_models import residualize_returns
from backtester.research_data import ResearchDataset
from strategies.research_strategies import SignalStrategy


class IdiosyncraticReversalStrategy(SignalStrategy):
    name = "Idiosyncratic Reversal"
    motivation = "Fade short-term idiosyncratic price shocks after removing market beta."
    economic_rationale = "Liquidity-driven overreactions mean-revert once market-wide moves are stripped out."
    why_it_works = "Residualized reversal has a stronger IC than raw reversal and less beta contamination."
    why_it_fails = "Structural news (earnings, guidance) can produce genuine trends that shouldn't mean-revert."

    def __init__(self, lookback_days: int = 5, beta_window: int = 252):
        self.lookback_days = lookback_days
        self.beta_window = beta_window

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        returns = dataset.returns
        market = dataset.benchmark_returns

        residuals = residualize_returns(returns, market, window=self.beta_window)
        recent_residual = residuals.rolling(
            self.lookback_days, min_periods=max(self.lookback_days // 2, 1)
        ).sum()

        # Short high residual returns, long low residual returns
        return (-recent_residual).replace([np.inf, -np.inf], np.nan)
