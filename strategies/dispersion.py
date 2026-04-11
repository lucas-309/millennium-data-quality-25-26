"""Earnings dispersion strategy.

Requires per-ticker DataFrames with a 'dispersion' column (e.g., std(analyst EPS) /
|mean(analyst EPS)|). This data is NOT provided by the standard data sources in this
project -- you must supply it externally (e.g., from IBES, Bloomberg, or similar).
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

from .order_generator import OrderGenerator

logger = logging.getLogger(__name__)


class DispersionStrategy(OrderGenerator):
    """High-minus-low earnings dispersion strategy.

    Longs stocks with high analyst forecast dispersion, shorts stocks with low
    dispersion. Rebalances on a fixed calendar frequency.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Mapping from ticker to DataFrame with columns 'Adj Close' and 'dispersion'.
    """

    def __init__(
        self,
        rebalance_frequency: str = 'ME',
        allocation_long: float = 0.40,
        allocation_short: float = 0.40,
        top_fraction: float = 0.10,
        bottom_fraction: float = 0.10,
    ):
        self.rebalance_frequency = rebalance_frequency
        self.allocation_long = allocation_long
        self.allocation_short = allocation_short
        self.top_fraction = top_fraction
        self.bottom_fraction = bottom_fraction

    def _get_rebalance_dates(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        if not all_dates:
            return pd.DatetimeIndex([])
        return pd.date_range(start=min(all_dates), end=max(all_dates), freq=self.rebalance_frequency)

    def generate_orders(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate dispersion-based orders.

        Args:
            data: Dict mapping ticker -> DataFrame with 'Adj Close' and 'dispersion' columns.

        Raises:
            ValueError: If no tickers have a 'dispersion' column.
        """
        # Validate data
        valid_tickers = [t for t, df in data.items()
                         if 'dispersion' in df.columns and 'Adj Close' in df.columns]
        if not valid_tickers:
            raise ValueError("No tickers have both 'Adj Close' and 'dispersion' columns.")

        all_orders: List[Dict[str, Any]] = []
        rebalance_dates = self._get_rebalance_dates(data)
        current_longs = set()
        current_shorts = set()

        for date in rebalance_dates:
            # Close existing positions
            for ticker in current_longs:
                all_orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": 1.0})
            for ticker in current_shorts:
                all_orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": 1.0})
            current_longs.clear()
            current_shorts.clear()

            # Build cross-section
            dispersions = {}
            prices = {}
            for ticker in valid_tickers:
                df = data[ticker]
                sub = df.loc[:date]
                if sub.empty:
                    continue
                disp_val = sub['dispersion'].iloc[-1]
                price_val = sub['Adj Close'].iloc[-1]
                if pd.isna(disp_val) or pd.isna(price_val) or price_val <= 0:
                    continue
                dispersions[ticker] = disp_val
                prices[ticker] = float(price_val)

            if len(dispersions) < 10:
                continue

            disp_series = pd.Series(dispersions).sort_values()
            n = len(disp_series)
            low_size = max(int(n * self.bottom_fraction), 1)
            high_size = max(int(n * self.top_fraction), 1)

            low_tickers = disp_series.head(low_size).index.tolist()
            high_tickers = disp_series.tail(high_size).index.tolist()

            # Long high dispersion
            long_alloc = self.allocation_long / len(high_tickers)
            for ticker in high_tickers:
                all_orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": long_alloc})
                current_longs.add(ticker)
                logger.debug("[%s] DISP BUY %s", date.date(), ticker)

            # Short low dispersion
            short_alloc = self.allocation_short / len(low_tickers)
            for ticker in low_tickers:
                all_orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": short_alloc})
                current_shorts.add(ticker)
                logger.debug("[%s] DISP SELL %s", date.date(), ticker)

        return all_orders
