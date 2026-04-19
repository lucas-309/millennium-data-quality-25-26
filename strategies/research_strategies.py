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


def _sector_neutral_zscore(frame: pd.DataFrame, metadata: Optional[pd.DataFrame], group_column: str = "gsector") -> pd.DataFrame:
    if metadata is None or metadata.empty or group_column not in metadata.columns:
        return _cross_sectional_zscore(frame)

    groups = metadata.dropna(subset=[group_column]).groupby(group_column).groups
    neutralized = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    covered_tickers: set[str] = set()

    for _, tickers in groups.items():
        group_tickers = [ticker for ticker in tickers if ticker in frame.columns]
        if not group_tickers:
            continue
        neutralized[group_tickers] = _cross_sectional_zscore(frame[group_tickers])
        covered_tickers.update(group_tickers)

    remaining = [ticker for ticker in frame.columns if ticker not in covered_tickers]
    if remaining:
        neutralized[remaining] = _cross_sectional_zscore(frame[remaining])

    return neutralized


@dataclass
class StrategyOutput:
    name: str
    scores: pd.DataFrame
    motivation: str
    economic_rationale: str
    why_it_works: str
    why_it_fails: str
    uses_event_data: bool = False
    event_type: Optional[str] = None
    backtest_overrides: dict[str, Any] = field(default_factory=dict)
    combine_in_portfolio: bool = True


class SignalStrategy(ABC):
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


class SmallCapTiltStrategy(SignalStrategy):
    name = "Small-Cap Tilt"
    motivation = "Own smaller-cap names and short larger-cap peers within the same large-cap universe."
    economic_rationale = "Smaller names often carry a persistent risk premium and receive less broad institutional attention."
    why_it_works = "Market-cap ranks move slowly, so the signal keeps turnover low while monetizing the size spread."
    why_it_fails = "Size tends to lag in risk-off environments or when mega-cap leadership dominates index returns."
    backtest_overrides = {
        "rebalance_frequency": "ME",
        "construction_method": "equal_weight",
        "long_quantile": 0.1,
        "short_quantile": 0.1,
    }

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        if dataset.market_caps is None or dataset.market_caps.empty:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)
        return -np.log(dataset.market_caps.replace(0, np.nan))


class ValueCompositeStrategy(SignalStrategy):
    name = "Value Composite"
    motivation = "Blend shareholder payout and earnings support into a slow-moving cross-sectional value score."
    economic_rationale = "Stocks with stronger cash yield and earnings support can trade at persistent discounts to peers."
    why_it_works = "Dividend yield, earnings yield, and a modest size tilt create a broader valuation signal than any single field alone."
    why_it_fails = "Value can stay cheap for long periods and becomes crowded when macro leadership rotates sharply."
    backtest_overrides = {
        "rebalance_frequency": "ME",
        "construction_method": "equal_weight",
        "long_quantile": 0.1,
        "short_quantile": 0.1,
    }
    combine_in_portfolio = False

    def __init__(self, trailing_days: int = 252):
        self.trailing_days = trailing_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        if dataset.dividends is None or dataset.dividends.empty or dataset.eps is None or dataset.eps.empty:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)

        trailing_dividends = dataset.dividends.rolling(
            self.trailing_days,
            min_periods=max(self.trailing_days // 4, 1),
        ).sum()
        dividend_yield = trailing_dividends.div(dataset.prices.replace(0, np.nan))
        earnings_yield = dataset.eps.div(dataset.prices.replace(0, np.nan))

        if dataset.market_caps is None or dataset.market_caps.empty:
            size_component = pd.DataFrame(0.0, index=dataset.prices.index, columns=dataset.prices.columns)
        else:
            size_component = -np.log(dataset.market_caps.replace(0, np.nan))

        return (
            _cross_sectional_zscore(dividend_yield)
            + _cross_sectional_zscore(earnings_yield)
            + _cross_sectional_zscore(size_component)
        )


class EarningsRevisionStrategy(SignalStrategy):
    name = "EPS Revision"
    motivation = "Rank stocks by improving versus deteriorating earnings expectations."
    economic_rationale = "Analysts and the market often underreact to gradual changes in the earnings outlook."
    why_it_works = "EPS revisions update less frequently than price, so the signal can capture post-announcement drift with manageable turnover."
    why_it_fails = "Revision signals break when estimates lag reality or when macro shocks overwhelm company-specific fundamentals."
    uses_event_data = True
    event_type = "anncdate"
    backtest_overrides = {
        "rebalance_frequency": "ME",
        "construction_method": "equal_weight",
        "long_quantile": 0.1,
        "short_quantile": 0.1,
    }

    def __init__(self, lookback_days: int = 63):
        self.lookback_days = lookback_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        if dataset.eps is None or dataset.eps.empty:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)
        baseline = dataset.eps.shift(self.lookback_days).replace(0, np.nan)
        return dataset.eps.div(baseline) - 1.0


class SectorNeutralDividendYieldStrategy(SignalStrategy):
    name = "Sector-Neutral Dividend Yield"
    motivation = "Own stronger dividend-yield names while neutralizing simple sector composition effects."
    economic_rationale = "Dividend policy can signal cash-flow durability, but yield comparisons are cleaner inside sectors than across them."
    why_it_works = "Sector neutralization removes the largest structural industry skews and keeps the signal focused on within-sector payout support."
    why_it_fails = "Dividend yield can still become a value trap when payouts lag a deteriorating earnings outlook."
    uses_event_data = True
    event_type = "divdpaydate"
    backtest_overrides = {
        "rebalance_frequency": "ME",
        "construction_method": "equal_weight",
        "long_quantile": 0.1,
        "short_quantile": 0.1,
    }

    def __init__(self, trailing_days: int = 252):
        self.trailing_days = trailing_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        if dataset.dividends is None or dataset.dividends.empty:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)

        trailing_dividends = dataset.dividends.rolling(
            self.trailing_days,
            min_periods=max(self.trailing_days // 4, 1),
        ).sum()
        return trailing_dividends.div(dataset.prices.replace(0, np.nan))

    def normalize_scores(self, raw_scores: pd.DataFrame, dataset: ResearchDataset) -> pd.DataFrame:
        return _sector_neutral_zscore(raw_scores, dataset.metadata)


class CrossSectionalMomentumStrategy(SignalStrategy):
    name = "Cross-Sectional Momentum"
    motivation = "Rank stocks by medium-term return, long the top names."
    economic_rationale = "Medium-term winners continue to outperform due to slow information diffusion and underreaction to fundamentals."
    why_it_works = "12-1 momentum is the most persistent cross-sectional anomaly in equities across decades and markets."
    why_it_fails = "Crashes during sharp reversals and regime changes (e.g. March 2009)."

    def __init__(self, lookback_days: int = 126, skip_days: int = 21):
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        prices = dataset.prices
        return prices.shift(self.skip_days).div(prices.shift(self.lookback_days)) - 1.0


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
    motivation = "Hold the lowest-volatility names, exploiting the low-vol anomaly."
    economic_rationale = "Leverage constraints and lottery preference cause low-vol stocks to be chronically underpriced relative to their risk-adjusted returns."
    why_it_works = "Low-vol long-only portfolios have matched or beaten broad equity returns with significantly smaller drawdowns across many decades."
    why_it_fails = "Underperforms speculative rallies and rotation toward high-beta growth names."

    def __init__(self, window: int = 126):
        self.window = window

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        vol = dataset.returns.rolling(self.window, min_periods=self.window // 2).std()
        return -vol  # higher score = lower vol = more desirable


def build_default_strategy_suite() -> list[SignalStrategy]:
    return [
        SmallCapTiltStrategy(),
        ValueCompositeStrategy(),
        EarningsRevisionStrategy(),
        SectorNeutralDividendYieldStrategy(),
        CrossSectionalMomentumStrategy(),
        LowVolatilityStrategy(),
    ]
