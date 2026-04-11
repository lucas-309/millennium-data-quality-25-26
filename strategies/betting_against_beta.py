import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

from .order_generator import OrderGenerator

logger = logging.getLogger(__name__)


class BettingAgainstBetaOrderGenerator(OrderGenerator):
    """Betting Against Beta (BAB) strategy.

    Longs low-beta stocks and shorts high-beta stocks, rebalanced periodically.
    Requires SPY as a column in the data for beta calculation.
    """

    def __init__(
        self,
        lookback_period: int = 252,
        rebalance_frequency: str = 'ME',
        allocation_long: float = 0.40,
        allocation_short: float = 0.40,
        decile_fraction: float = 0.10,
    ):
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.allocation_long = allocation_long
        self.allocation_short = allocation_short
        self.decile_fraction = decile_fraction

    def _calculate_betas(self, returns: pd.DataFrame, date, tickers: List[str]) -> Dict[str, float]:
        """Calculate betas for all tickers relative to SPY up to the given date."""
        spy_returns = returns['SPY']
        betas = {}

        for ticker in tickers:
            if ticker == 'SPY':
                continue

            combined = pd.concat([returns[ticker], spy_returns], axis=1, join='inner').loc[:date]
            combined = combined.iloc[-self.lookback_period:]

            if len(combined) < self.lookback_period // 2:
                continue

            stock_ret = combined.iloc[:, 0]
            market_ret = combined.iloc[:, 1]

            cov = stock_ret.cov(market_ret)
            var = market_ret.var()
            if var > 0:
                betas[ticker] = cov / var

        return betas

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate BAB orders. Data must be a DataFrame with SPY as one of the columns."""
        if 'SPY' not in data.columns:
            raise ValueError("SPY must be a column in the data for beta calculation.")

        returns = data.pct_change().dropna()
        tickers = [c for c in data.columns if c != 'SPY']

        if len(returns) <= self.lookback_period:
            return []

        start_date = returns.index[self.lookback_period]
        end_date = returns.index[-1]
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_frequency)

        all_orders = []
        current_longs = set()
        current_shorts = set()

        for date in rebalance_dates:
            if date not in data.index:
                # Find nearest prior trading day
                valid_dates = data.index[data.index <= date]
                if len(valid_dates) == 0:
                    continue
                date = valid_dates[-1]

            # Close all existing positions before rebalancing
            for ticker in current_longs:
                all_orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": 1.0})
            for ticker in current_shorts:
                all_orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": 1.0})
            current_longs.clear()
            current_shorts.clear()

            # Calculate betas
            betas = self._calculate_betas(returns, date, tickers)
            if len(betas) < 20:
                continue

            beta_series = pd.Series(betas).dropna().sort_values()
            n = len(beta_series)
            decile_size = max(int(n * self.decile_fraction), 1)

            low_beta_tickers = beta_series.head(decile_size).index.tolist()
            high_beta_tickers = beta_series.tail(decile_size).index.tolist()

            # Long low-beta, short high-beta with equal allocation per stock
            long_alloc = self.allocation_long / len(low_beta_tickers)
            short_alloc = self.allocation_short / len(high_beta_tickers)

            for ticker in low_beta_tickers:
                all_orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": long_alloc})
                current_longs.add(ticker)
                logger.debug("BAB BUY %s on %s (low beta)", ticker, date)

            for ticker in high_beta_tickers:
                all_orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": short_alloc})
                current_shorts.add(ticker)
                logger.debug("BAB SELL %s on %s (high beta)", ticker, date)

        all_orders.sort(key=lambda o: o["date"])
        return all_orders
