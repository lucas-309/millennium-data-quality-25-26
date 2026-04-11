import pandas as pd
import numpy as np
from typing import List, Dict, Any

from .order_generator import OrderGenerator


class MeanReversionOrderGenerator(OrderGenerator):
    """Z-score mean reversion strategy with position limits and stop-loss.

    Buys when price drops significantly below its rolling mean (negative z-score),
    sells when price reverts to the mean or drops further (stop-loss).
    """

    def __init__(
        self,
        lookback: int = 100,
        entry_zscore: float = -1.5,
        exit_zscore: float = 0.0,
        stop_loss_zscore: float = -3.0,
        allocation_per_trade: float = 0.10,
        max_positions: int = 5,
    ):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.allocation_per_trade = allocation_per_trade
        self.max_positions = max_positions

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []
        prices = data.sort_index()
        rolling_mean = prices.rolling(window=self.lookback).mean()
        rolling_std = prices.rolling(window=self.lookback).std().replace(0, np.nan)
        zscores = (prices - rolling_mean) / rolling_std

        in_position: Dict[str, bool] = {ticker: False for ticker in prices.columns}

        for date in prices.index:
            row = zscores.loc[date]

            # Exit before entering new names so freed slots are reusable on the same day.
            for ticker in prices.columns:
                z = row.get(ticker)
                if not in_position[ticker] or pd.isna(z):
                    continue

                if z >= self.exit_zscore or z <= self.stop_loss_zscore:
                    orders.append({
                        "date": date,
                        "type": "SELL",
                        "ticker": ticker,
                        "quantity": 1.0,
                    })
                    in_position[ticker] = False

            open_positions = sum(in_position.values())
            available_slots = max(self.max_positions - open_positions, 0)
            if available_slots == 0:
                continue

            candidates = row[(row <= self.entry_zscore) & row.notna()]
            candidates = candidates.sort_values()

            for ticker, _ in candidates.items():
                if available_slots == 0:
                    break
                if in_position[ticker]:
                    continue

                orders.append({
                    "date": date,
                    "type": "BUY",
                    "ticker": ticker,
                    "quantity": float(self.allocation_per_trade),
                })
                in_position[ticker] = True
                available_slots -= 1

        orders.sort(key=lambda o: o["date"])
        return orders
