import pandas as pd
from typing import List, Dict, Any

from .order_generator import OrderGenerator


class MomentumOrderGenerator(OrderGenerator):
    """Momentum strategy with trailing stop and momentum decay exit.

    Enters when price is near its rolling high (breakout), exits via trailing
    stop-loss or momentum decay (no new high in N days).
    """

    def __init__(
        self,
        lookback_days: int = 252,
        entry_threshold: float = 0.02,
        trailing_stop_pct: float = 0.10,
        allocation_per_trade: float = 0.15,
        max_positions: int = 4,
        momentum_decay_days: int = 20,
    ):
        self.lookback_days = lookback_days
        self.entry_threshold = entry_threshold
        self.trailing_stop_pct = trailing_stop_pct
        self.allocation_per_trade = allocation_per_trade
        self.max_positions = max_positions
        self.momentum_decay_days = momentum_decay_days

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []
        prices = data.sort_index()
        tickers = list(prices.columns)
        rolling_high = prices.rolling(window=self.lookback_days).max().shift(1)
        momentum_strength = prices.div(prices.shift(self.lookback_days)) - 1.0

        # ticker -> {peak, days_since_new_high}
        position_state: Dict[str, dict] = {ticker: None for ticker in tickers}

        for date in prices.index:
            day_prices = prices.loc[date]
            day_highs = rolling_high.loc[date]

            # Process exits first.
            for ticker in tickers:
                state = position_state[ticker]
                price = day_prices.get(ticker)
                high = day_highs.get(ticker)
                if state is None or pd.isna(price) or pd.isna(high):
                    continue

                if price > state["peak"]:
                    state["peak"] = price
                    state["days_since_new_high"] = 0
                else:
                    state["days_since_new_high"] += 1

                should_exit = (
                    price <= state["peak"] * (1 - self.trailing_stop_pct)
                    or state["days_since_new_high"] >= self.momentum_decay_days
                )

                if should_exit:
                    orders.append({
                        "date": date,
                        "type": "SELL",
                        "ticker": ticker,
                        "quantity": 1.0,
                    })
                    position_state[ticker] = None

            open_positions = sum(state is not None for state in position_state.values())
            available_slots = max(self.max_positions - open_positions, 0)
            if available_slots == 0:
                continue

            entry_mask = (day_prices >= day_highs * (1 - self.entry_threshold)) & day_prices.notna() & day_highs.notna()
            candidates = momentum_strength.loc[date][entry_mask].dropna().sort_values(ascending=False)

            for ticker, _ in candidates.items():
                if available_slots == 0:
                    break
                if position_state[ticker] is not None:
                    continue

                orders.append({
                    "date": date,
                    "type": "BUY",
                    "ticker": ticker,
                    "quantity": float(self.allocation_per_trade),
                })
                position_state[ticker] = {
                    "peak": float(day_prices[ticker]),
                    "days_since_new_high": 0,
                }
                available_slots -= 1

        orders.sort(key=lambda o: o["date"])
        return orders
