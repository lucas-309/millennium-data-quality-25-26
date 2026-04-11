"""Pre-trade risk management: limits, exposure checks, drawdown kill-switch.

Models the risk layer of a multi-manager pod shop: every order is checked
against hard limits before execution, and the pod is frozen if drawdown
exceeds a threshold (just like a Millennium pod being cut).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class RiskDecision(Enum):
    ALLOW = "ALLOW"
    REDUCE = "REDUCE"
    REJECT = "REJECT"


@dataclass
class RiskLimits:
    """Hard risk limits applied before every order execution."""
    max_gross_leverage: float = 2.0           # gross notional / equity
    max_net_leverage: float = 1.0             # net notional / equity
    max_position_pct: float = 0.05            # any single name
    max_sector_pct: float = 0.25              # per sector exposure
    max_abs_beta: float = 0.30                # |portfolio beta|
    max_drawdown_stop: float = 0.15           # 15% drawdown kill switch
    vol_target_annual: Optional[float] = None # scale positions to hit target
    sector_map: Optional[Dict[str, str]] = None
    beta_map: Optional[Dict[str, float]] = None


@dataclass
class RiskState:
    """Mutable state tracking portfolio risk over time."""
    equity_high: float = 0.0
    current_equity: float = 0.0
    kill_switch_active: bool = False
    rejected_orders: List[dict] = field(default_factory=list)

    @property
    def drawdown(self) -> float:
        if self.equity_high <= 0:
            return 0.0
        return (self.current_equity / self.equity_high) - 1.0


class RiskManager:
    """Pre-trade risk checks. Call `check_order` before executing.

    Also call `update_equity` once per day with the latest portfolio value to
    maintain the drawdown kill-switch.
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.state = RiskState()

    # ------------------------------------------------------------------
    # Equity / drawdown tracking
    # ------------------------------------------------------------------
    def update_equity(self, current_equity: float, date=None) -> None:
        self.state.current_equity = current_equity
        self.state.equity_high = max(self.state.equity_high, current_equity)

        dd = self.state.drawdown
        if dd <= -self.limits.max_drawdown_stop and not self.state.kill_switch_active:
            self.state.kill_switch_active = True

    def reset_kill_switch(self) -> None:
        """Re-arm after manual review (e.g., pod allowed to trade again)."""
        self.state.kill_switch_active = False
        self.state.equity_high = self.state.current_equity

    # ------------------------------------------------------------------
    # Pre-trade check
    # ------------------------------------------------------------------
    def check_order(
        self,
        ticker: str,
        side: str,             # "BUY" or "SELL"
        quantity: int,
        price: float,
        current_holdings: Dict[str, int],
        current_cash: float,
        price_lookup: Optional[Dict[str, float]] = None,
    ) -> Tuple[RiskDecision, int, str]:
        """Check if an order passes risk limits.

        Returns (decision, approved_quantity, reason). Decision is ALLOW
        (approved_quantity == requested), REDUCE (quantity capped), or REJECT.
        """
        if self.state.kill_switch_active:
            # Only allow closes (reducing exposure), block new opens.
            if not self._is_closing_order(ticker, side, current_holdings):
                return RiskDecision.REJECT, 0, "drawdown_kill_switch"

        # Compute portfolio state
        equity = self._compute_equity(current_cash, current_holdings, price_lookup, ticker, price)
        if equity <= 0:
            return RiskDecision.REJECT, 0, "zero_equity"

        # Per-position limit
        current_pos = current_holdings.get(ticker, 0)
        delta = quantity if side == "BUY" else -quantity
        new_pos = current_pos + delta
        new_pos_value = abs(new_pos) * price
        current_pos_value = abs(current_pos) * price
        max_value = equity * self.limits.max_position_pct

        # Only block/reduce when the order actually INCREASES absolute exposure
        # beyond the cap. Orders that reduce |position| (de-risking) are always
        # allowed even if the position is still above the cap afterward — you
        # need to be able to work down an oversized name in multiple clips.
        if new_pos_value > max_value and abs(new_pos) > abs(current_pos):
            max_shares = int(max_value // price)
            # Max quantity such that resulting position fits under the cap in
            # the direction we're trading. Handles crossing zero correctly.
            if delta > 0:  # BUY
                allowed = max(0, max_shares - current_pos)
            else:  # SELL
                allowed = max(0, current_pos + max_shares)

            if allowed < quantity:
                if allowed <= 0:
                    return RiskDecision.REJECT, 0, "position_limit"
                return RiskDecision.REDUCE, allowed, "position_limit_capped"

        # Gross leverage check
        gross_after = self._gross_exposure_after(current_holdings, ticker, delta, price_lookup, price)
        if gross_after / equity > self.limits.max_gross_leverage:
            return RiskDecision.REJECT, 0, "gross_leverage_limit"

        # Net leverage check
        net_after = self._net_exposure_after(current_holdings, ticker, delta, price_lookup, price)
        if abs(net_after) / equity > self.limits.max_net_leverage:
            return RiskDecision.REJECT, 0, "net_leverage_limit"

        # Sector exposure check
        if self.limits.sector_map and ticker in self.limits.sector_map:
            sector = self.limits.sector_map[ticker]
            sector_exp = self._sector_exposure(current_holdings, sector, price_lookup)
            sector_exp += delta * price
            if abs(sector_exp) / equity > self.limits.max_sector_pct:
                return RiskDecision.REJECT, 0, "sector_limit"

        # Beta exposure check
        if self.limits.beta_map:
            portfolio_beta = self._portfolio_beta(current_holdings, price_lookup, equity)
            ticker_beta = self.limits.beta_map.get(ticker, 1.0)
            new_beta = portfolio_beta + (delta * price / equity) * ticker_beta
            if abs(new_beta) > self.limits.max_abs_beta:
                return RiskDecision.REJECT, 0, "beta_limit"

        return RiskDecision.ALLOW, quantity, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_closing_order(ticker: str, side: str, holdings: Dict[str, int]) -> bool:
        current = holdings.get(ticker, 0)
        if side == "BUY":
            return current < 0  # covering a short
        return current > 0  # selling a long

    @staticmethod
    def _compute_equity(
        cash: float,
        holdings: Dict[str, int],
        price_lookup: Optional[Dict[str, float]],
        current_ticker: str,
        current_price: float,
    ) -> float:
        equity = cash
        for t, qty in holdings.items():
            if price_lookup and t in price_lookup:
                equity += qty * price_lookup[t]
            elif t == current_ticker:
                equity += qty * current_price
        return equity

    @staticmethod
    def _gross_exposure_after(
        holdings: Dict[str, int],
        ticker: str,
        delta: int,
        price_lookup: Optional[Dict[str, float]],
        current_price: float,
    ) -> float:
        gross = 0.0
        for t, qty in holdings.items():
            px = (price_lookup or {}).get(t, current_price if t == ticker else 0.0)
            if t == ticker:
                gross += abs(qty + delta) * px
            else:
                gross += abs(qty) * px
        if ticker not in holdings:
            gross += abs(delta) * current_price
        return gross

    @staticmethod
    def _net_exposure_after(
        holdings: Dict[str, int],
        ticker: str,
        delta: int,
        price_lookup: Optional[Dict[str, float]],
        current_price: float,
    ) -> float:
        net = 0.0
        for t, qty in holdings.items():
            px = (price_lookup or {}).get(t, current_price if t == ticker else 0.0)
            if t == ticker:
                net += (qty + delta) * px
            else:
                net += qty * px
        if ticker not in holdings:
            net += delta * current_price
        return net

    def _sector_exposure(
        self,
        holdings: Dict[str, int],
        sector: str,
        price_lookup: Optional[Dict[str, float]],
    ) -> float:
        if not self.limits.sector_map:
            return 0.0
        exposure = 0.0
        for t, qty in holdings.items():
            if self.limits.sector_map.get(t) == sector:
                px = (price_lookup or {}).get(t, 0.0)
                exposure += qty * px
        return exposure

    def _portfolio_beta(
        self,
        holdings: Dict[str, int],
        price_lookup: Optional[Dict[str, float]],
        equity: float,
    ) -> float:
        if not self.limits.beta_map or equity <= 0:
            return 0.0
        beta = 0.0
        for t, qty in holdings.items():
            px = (price_lookup or {}).get(t, 0.0)
            weight = qty * px / equity
            ticker_beta = self.limits.beta_map.get(t, 1.0)
            beta += weight * ticker_beta
        return beta


def vol_target_scale(
    returns: pd.Series,
    target_annual_vol: float,
    lookback: int = 63,
    trading_days: int = 252,
) -> pd.Series:
    """Compute vol-targeting scalar to apply to positions.

    Returns a scalar time series: positions * scalar → target volatility.
    """
    realized_vol = returns.rolling(lookback).std() * np.sqrt(trading_days)
    scalar = target_annual_vol / realized_vol.replace(0, np.nan)
    return scalar.clip(upper=3.0).fillna(1.0)  # cap leverage at 3x
