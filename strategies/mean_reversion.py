import pandas as pd
from typing import List, Dict, Any

from .order_generator import OrderGenerator


class MeanReversionOrderGenerator(OrderGenerator):
    """Long-only mean reversion with position management.

    The old version emitted a BUY or SELL every single trading day, which
    meant it accumulated an unbounded short position during any sustained
    uptrend (e.g. NVDA 2020-2024 blew up to a 163k-share short, -180x
    cumulative return).

    This version only trades on state transitions:
      - currently flat and price < SMA  -> go long ``position_size`` of
        portfolio value (default 50%)
      - currently long and price >= SMA -> liquidate
    Never shorts. Fires one order per crossover instead of one per day.
    """

    def __init__(self, window: int = 100, position_size: float = 0.5):
        self.window = window
        self.position_size = position_size

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []
        sma = data.rolling(window=self.window).mean()

        for ticker in data.columns:
            is_long = False
            for date in data.index:
                price = data.at[date, ticker]
                avg = sma.at[date, ticker]
                if pd.isna(avg) or pd.isna(price):
                    continue

                if price < avg and not is_long:
                    orders.append({
                        "date": date, "type": "BUY", "ticker": ticker,
                        "quantity": self.position_size,
                    })
                    is_long = True
                elif price >= avg and is_long:
                    orders.append({
                        "date": date, "type": "SELL", "ticker": ticker,
                        "quantity": 1.0,
                    })
                    is_long = False

        return orders