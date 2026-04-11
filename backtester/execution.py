"""Realistic execution cost models: market impact, borrow costs, ADV participation.

These models augment the basic EquityBacktestEngine with prop-shop grade frictions:
- Market impact: temporary + permanent price impact from crossing the book
- Borrow costs: daily financing charge on short positions
- ADV participation: hard cap on order size as % of average daily volume

Reference: Almgren-Chriss (2000), Kissell-Glantz (2003), Obizhaeva-Wang (2013).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


class MarketImpactModel(ABC):
    """Abstract market impact model. Returns cost in basis points."""

    @abstractmethod
    def impact_bps(
        self,
        order_notional: float,
        adv_notional: float,
        volatility: float = 0.02,
    ) -> float:
        """Compute total impact in basis points for a given order size.

        Args:
            order_notional: Order size in dollars.
            adv_notional: Average daily volume in dollars.
            volatility: Daily volatility of the asset (decimal, e.g., 0.02 for 2%).
        """


@dataclass
class LinearImpact(MarketImpactModel):
    """Linear impact: cost = lambda * (order / ADV).

    Simple but effective. Appropriate for small-to-medium sized orders.
    """
    lambda_bps: float = 10.0  # bps per 100% of ADV

    def impact_bps(self, order_notional, adv_notional, volatility=0.02):
        if adv_notional <= 0:
            return 0.0
        participation = abs(order_notional) / adv_notional
        return self.lambda_bps * participation


@dataclass
class SquareRootImpact(MarketImpactModel):
    """Almgren-Chriss square-root impact model.

    cost_bps = k * sigma * sqrt(Q / V) * 10000

    where Q/V is participation (order / ADV) and sigma is daily vol.
    This is the industry standard for capacity modeling.
    """
    k: float = 1.0  # market-specific constant (0.5-2.0 typical)

    def impact_bps(self, order_notional, adv_notional, volatility=0.02):
        if adv_notional <= 0:
            return 0.0
        participation = abs(order_notional) / adv_notional
        # Convert sigma (decimal) to basis points via 10000
        return self.k * volatility * np.sqrt(participation) * 10000


@dataclass
class PowerLawImpact(MarketImpactModel):
    """Generalized power-law impact: cost = c * sigma * (Q/V)^alpha * 10000.

    Alpha < 1 captures sublinear impact (most real markets). Alpha = 0.5 reduces
    to square-root. Alpha = 1.0 reduces to linear.
    """
    c: float = 1.0
    alpha: float = 0.6

    def impact_bps(self, order_notional, adv_notional, volatility=0.02):
        if adv_notional <= 0:
            return 0.0
        participation = abs(order_notional) / adv_notional
        return self.c * volatility * (participation ** self.alpha) * 10000


@dataclass
class BorrowCostModel:
    """Daily financing cost on short positions.

    Standard GC (general collateral) rate ~25-50bps annualized. Hard-to-borrow
    names can be 500bps-5000bps. Specific `hard_to_borrow` tickers apply their
    own rate.
    """
    default_annual_rate: float = 0.005  # 50 bps general collateral default
    hard_to_borrow: Dict[str, float] = field(default_factory=dict)
    trading_days: int = 252

    def daily_cost(self, ticker: str, short_notional: float) -> float:
        """Cost in dollars for holding `short_notional` short for one day."""
        if short_notional <= 0:
            return 0.0
        rate = self.hard_to_borrow.get(ticker, self.default_annual_rate)
        return short_notional * rate / self.trading_days


@dataclass
class ExecutionConfig:
    """Bundles all execution parameters into one config object."""
    impact_model: Optional[MarketImpactModel] = None
    borrow_model: Optional[BorrowCostModel] = None
    max_adv_participation: float = 0.10  # 10% of ADV max per day
    spread_bps: float = 2.0  # half-spread crossing cost
    min_adv_dollars: float = 0.0  # skip trades on names below this ADV floor


def estimate_adv(
    volume_series,
    price_series,
    window: int = 20,
):
    """Compute rolling average daily volume in dollars.

    Args:
        volume_series: Share volume time series.
        price_series: Price time series.

    Returns:
        Rolling ADV in dollar notional.
    """
    dollar_volume = volume_series * price_series
    return dollar_volume.rolling(window=window, min_periods=1).mean()


def cap_order_by_adv(
    requested_quantity: int,
    price: float,
    adv_dollars: float,
    max_participation: float,
) -> int:
    """Cap order quantity at max_participation * ADV.

    Returns the capped share quantity (never increases, only reduces).
    """
    if adv_dollars <= 0 or price <= 0:
        return 0
    max_dollars = adv_dollars * max_participation
    max_shares = int(max_dollars // price)
    return max(0, min(requested_quantity, max_shares))
