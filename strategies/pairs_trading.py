import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from .order_generator import OrderGenerator


class PairsTradingOrderGenerator(OrderGenerator):
    """Statistical arbitrage pairs trading with cointegration checks and stop-loss.

    Trades the log-price-ratio spread when it deviates from its rolling mean.
    Includes correlation gate, exit buffer to prevent whipsaw, and emergency
    stop-loss for spread blowouts.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        lookback_window: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: float = 4.0,
        allocation_per_leg: float = 0.10,
        min_correlation: float = 0.7,
    ):
        self.pairs = pairs
        self.lookback_window = lookback_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.allocation_per_leg = allocation_per_leg
        self.min_correlation = min_correlation

    def _check_pair_quality(self, price_a: pd.Series, price_b: pd.Series, end_idx: int) -> bool:
        """Check if a pair has sufficient correlation over the lookback window."""
        start_idx = max(0, end_idx - self.lookback_window)
        window_a = price_a.iloc[start_idx:end_idx]
        window_b = price_b.iloc[start_idx:end_idx]

        if len(window_a) < self.lookback_window // 2:
            return False

        corr = window_a.corr(window_b)
        return not pd.isna(corr) and abs(corr) >= self.min_correlation

    def _compute_spread(self, price_a: pd.Series, price_b: pd.Series) -> pd.Series:
        log_a = np.log(price_a.replace(0, np.nan))
        log_b = np.log(price_b.replace(0, np.nan))
        rolling_cov = log_a.rolling(self.lookback_window).cov(log_b)
        rolling_var = log_b.rolling(self.lookback_window).var().replace(0, np.nan)
        hedge_ratio = (rolling_cov / rolling_var).clip(lower=-5.0, upper=5.0)
        hedge_ratio = hedge_ratio.replace([np.inf, -np.inf], np.nan).ffill()
        return log_a - hedge_ratio * log_b

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []

        for ticker_a, ticker_b in self.pairs:
            if ticker_a not in data.columns or ticker_b not in data.columns:
                continue

            price_a = data[ticker_a]
            price_b = data[ticker_b]

            spread = self._compute_spread(price_a, price_b)
            rolling_mean = spread.rolling(window=self.lookback_window).mean()
            rolling_std = spread.rolling(window=self.lookback_window).std().replace(0, np.nan)
            zscore = (spread - rolling_mean) / rolling_std

            # Track position state for this specific pair
            # None = flat, "long_b" = long B / short A, "long_a" = long A / short B
            position = None

            for i, date in enumerate(data.index):
                z = zscore.get(date)
                if z is None or pd.isna(z):
                    continue

                if position is None:
                    # Check pair quality before entering
                    if not self._check_pair_quality(price_a, price_b, i):
                        continue

                    if z > self.entry_zscore:
                        # A expensive relative to B -> short A, long B
                        orders.append({"date": date, "type": "BUY", "ticker": ticker_b,
                                       "quantity": self.allocation_per_leg})
                        orders.append({"date": date, "type": "SELL", "ticker": ticker_a,
                                       "quantity": self.allocation_per_leg})
                        position = "long_b"

                    elif z < -self.entry_zscore:
                        # B expensive relative to A -> short B, long A
                        orders.append({"date": date, "type": "BUY", "ticker": ticker_a,
                                       "quantity": self.allocation_per_leg})
                        orders.append({"date": date, "type": "SELL", "ticker": ticker_b,
                                       "quantity": self.allocation_per_leg})
                        position = "long_a"
                else:
                    should_exit = False

                    # Normal exit: spread reverted past the exit buffer
                    if position == "long_b" and z <= self.exit_zscore:
                        should_exit = True
                    elif position == "long_a" and z >= -self.exit_zscore:
                        should_exit = True

                    # Stop-loss: spread blew out
                    if position == "long_b" and z > self.stop_loss_zscore:
                        should_exit = True
                    elif position == "long_a" and z < -self.stop_loss_zscore:
                        should_exit = True

                    if should_exit:
                        # Exit: fully unwind both legs. The engine interprets
                        # float quantity <= 1.0 as a fraction of the existing
                        # position, so 1.0 = close 100% of holdings.
                        if position == "long_b":
                            orders.append({"date": date, "type": "SELL", "ticker": ticker_b,
                                           "quantity": 1.0})
                            orders.append({"date": date, "type": "BUY", "ticker": ticker_a,
                                           "quantity": 1.0})
                        elif position == "long_a":
                            orders.append({"date": date, "type": "SELL", "ticker": ticker_a,
                                           "quantity": 1.0})
                            orders.append({"date": date, "type": "BUY", "ticker": ticker_b,
                                           "quantity": 1.0})
                        position = None

        orders.sort(key=lambda o: o["date"])
        return orders
