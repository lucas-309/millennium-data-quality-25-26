import pandas as pd
import numpy as np
from typing import List, Dict, Any

from .order_generator import OrderGenerator

# uses fixed-sized sharing: there's a small-cap tilt, update to vol-weighted sizing or dollar-neutrality

# make sure we're using pre-announcement consensus snapshot (avoid lookahead bias)

# t-cost is important !!! (test with t-cost)

# 60-day holding overlaps acrouss announcements so single name 
# can be in the book multiple times - backtester needs to handle properly

class PEADOrderGenerator(OrderGenerator):
    """
    Post-Earnings Announcement Drift strategy

    On each earnings announcement date, rank the universe by Standardized
    Unexpected Earnings (SUE). Go long the top decile, short the bottom
    decile, entering at the close of the first tradable session after the
    announcement and holding for `holding_days` trading days

    Parameters
    ----------
    events : pd.DataFrame
        Capital IQ event-study data. Expected columns:
          ['date', 'ticker', 'event_type', 'actual_eps',
           'consensus_eps', 'eps_stdev', 'announcement_time']
        `announcement_time` is 'bmo' (before market open) or 'amc'
        (after market close). `date` is the calendar announcement date.

    holding_days : int - number of trading days to hold the position (default 60)
    quantity : int - shares per leg. fixed-share sizing 
    top_decile : float - fraction of the universe on each side (default 0.10)

    min_universe : int - minimum number of announcements on a given date to run the
        cross-sectional rank. Below this, skip the date — a 3-stock
        decile is noise, not signal.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        holding_days: int = 60,
        quantity: int = 100,
        top_decile: float = 0.10,
        min_universe: int = 20,
    ):
        self.events = events.copy()
        self.events['date'] = pd.to_datetime(self.events['date'])
        self.holding_days = holding_days
        self.quantity = quantity
        self.top_decile = top_decile
        self.min_universe = min_universe

    # this variable is now input!
    # change when Boss loads sue
    def _compute_sue(self, events: pd.DataFrame) -> pd.DataFrame:
        """Standardized Unexpected Earnings = (actual - consensus) / stdev."""
        events = events.copy()
        # guard against zero/near-zero stdev which would blow SUE up.
        valid = events['eps_stdev'] > 1e-6
        events['sue'] = np.where(
            valid,
            (events['actual_eps'] - events['consensus_eps']) / events['eps_stdev'],
            np.nan,
        )
        return events.dropna(subset=['sue'])

    def _entry_date(self, ann_date: pd.Timestamp, when: str,
                    trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
        """
        Map an announcement to the first tradable close.
          - 'bmo' (before open): trade at the close of the same day.
          - 'amc' (after close) or unknown: trade at the next session's close.
        Returns None if no session is available (e.g. end of sample).
        """
        if when == 'bmo':
            # same-day close if it's a trading day, else next session.
            candidates = trading_days[trading_days >= ann_date]
        else:
            # next trading day strictly after the announcement date.
            candidates = trading_days[trading_days > ann_date]
        return candidates[0] if len(candidates) else None

    def _exit_date(self, entry: pd.Timestamp,
                   trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
        """Exit `holding_days` trading days after entry."""
        idx = trading_days.get_indexer([entry])[0]
        if idx < 0:
            return None
        exit_idx = idx + self.holding_days
        if exit_idx >= len(trading_days):
            return None
        return trading_days[exit_idx]

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []
        trading_days = pd.DatetimeIndex(sorted(data.index.unique()))
        price_universe = set(data.columns)

        # restrict to earnings events only and compute SUE up front.
        earnings = self.events[self.events['event_type'] == 'earnings']
        earnings = self._compute_sue(earnings)

        # cross-sectional ranking is per announcement date.
        for ann_date, group in earnings.groupby('date'):
            # only rank names we can actually trade (must have price data)
            group = group[group['ticker'].isin(price_universe)]
            if len(group) < self.min_universe:
                continue

            # decile cutoffs on this date's cross-section
            hi = group['sue'].quantile(1 - self.top_decile)
            lo = group['sue'].quantile(self.top_decile)

            longs = group[group['sue'] >= hi]
            shorts = group[group['sue'] <= lo]

            for _, ev in pd.concat([
                longs.assign(side='BUY'),
                shorts.assign(side='SELL'),
            ]).iterrows():
                entry = self._entry_date(
                    ann_date, ev.get('announcement_time', 'amc'), trading_days
                )
                if entry is None:
                    continue
                exit_ = self._exit_date(entry, trading_days)
                if exit_ is None:
                    continue

                # confirm the ticker has a price on both entry and exit (guards against delistings mid-trade).
                if pd.isna(data.at[entry, ev['ticker']]) or \
                   pd.isna(data.at[exit_, ev['ticker']]):
                    continue

                # open
                orders.append({
                    "date": entry,
                    "type": ev['side'],
                    "ticker": ev['ticker'],
                    "quantity": self.quantity,
                })
                # close
                orders.append({
                    "date": exit_,
                    "type": "SELL" if ev['side'] == "BUY" else "BUY",
                    "ticker": ev['ticker'],
                    "quantity": self.quantity,
                })

        orders.sort(key=lambda o: (o['date'], o['ticker']))
        return orders