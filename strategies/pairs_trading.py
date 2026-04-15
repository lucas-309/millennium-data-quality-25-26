import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from .order_generator import OrderGenerator


class PairsTradingOrderGenerator(OrderGenerator):
    """Statistical arbitrage pairs trading strategy.

    Identifies correlated stock pairs and trades the spread when it deviates
    from its historical mean. Long the underperformer, short the overperformer,
    close both when the spread reverts.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        lookback_window: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.0,
        allocation_per_pair: float = 0.50,
    ):
        self.pairs = pairs
        self.lookback_window = lookback_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.allocation_per_pair = allocation_per_pair

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []

        for ticker_a, ticker_b in self.pairs:
            if ticker_a not in data.columns or ticker_b not in data.columns:
                continue

            price_a = data[ticker_a]
            price_b = data[ticker_b]

            # Require strictly positive prices for a valid log-ratio calculation
            valid_prices = (price_a > 0) & (price_b > 0)
            if not valid_prices.any():
                continue

            price_a = price_a[valid_prices]
            price_b = price_b[valid_prices]

            # Log price ratio
            ratio = np.log(price_a / price_b)

            rolling_mean = ratio.rolling(window=self.lookback_window).mean()
            rolling_std = ratio.rolling(window=self.lookback_window).std().replace(0, np.nan)

            zscore = (ratio - rolling_mean) / rolling_std
            zscore = zscore.replace([np.inf, -np.inf], np.nan)

            # Track position state for this pair:
            # None = flat, "long_b" = long B / short A, "long_a" = long A / short B
            position = None

            for date in data.index:
                if date not in zscore.index:
                    continue
                z = zscore.loc[date]
                if z is None or pd.isna(z):
                    continue

                if position is None:
                    if z > self.entry_zscore:
                        # ticker_a expensive, ticker_b cheap
                        # → long ticker_b, short ticker_a
                        orders.append({
                            "date": date,
                            "type": "BUY",
                            "ticker": ticker_b,
                            "quantity": self.allocation_per_pair,
                        })
                        orders.append({
                            "date": date,
                            "type": "SELL",
                            "ticker": ticker_a,
                            "quantity": self.allocation_per_pair,
                        })
                        position = "long_b"
                    elif z < -self.entry_zscore:
                        # ticker_b expensive, ticker_a cheap
                        # → long ticker_a, short ticker_b
                        orders.append({
                            "date": date,
                            "type": "BUY",
                            "ticker": ticker_a,
                            "quantity": self.allocation_per_pair,
                        })
                        orders.append({
                            "date": date,
                            "type": "SELL",
                            "ticker": ticker_b,
                            "quantity": self.allocation_per_pair,
                        })
                        position = "long_a"
                else:
                    # Close both legs when spread reverts
                    if position == "long_b" and z <= self.exit_zscore:
                        orders.append({
                            "date": date,
                            "type": "SELL",
                            "ticker": ticker_b,
                            "quantity": 1.0,
                        })
                        orders.append({
                            "date": date,
                            "type": "BUY",
                            "ticker": ticker_a,
                            "quantity": 1.0,
                        })
                        position = None
                    elif position == "long_a" and z >= self.exit_zscore:
                        orders.append({
                            "date": date,
                            "type": "SELL",
                            "ticker": ticker_a,
                            "quantity": 1.0,
                        })
                        orders.append({
                            "date": date,
                            "type": "BUY",
                            "ticker": ticker_b,
                            "quantity": 1.0,
                        })
                        position = None

        orders.sort(key=lambda o: (o["date"], o["ticker"], o["type"]))
        return orders
