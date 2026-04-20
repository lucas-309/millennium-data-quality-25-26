"""Daily FF3 + Momentum factor proxies built from the research universe.

We don't ship Ken French's factor library, so we construct proxies inside the
same universe the strategies trade. Each factor is a long-short portfolio with
equal-weight legs, quantile breakpoints refreshed at rebalance frequency, held
intra-rebalance with signal_lag=1 so no look-ahead.

  MKT = equal-weight universe return (risk-free ≈ 0 over our window;
        Sharpe convention is unadjusted per repo-wide practice)
  SMB = bottom-30% market cap − top-30% market cap
  HML = top-30% E/P         − bottom-30% E/P      (E/P is our book-to-market
                                                    stand-in, since Compustat
                                                    book value isn't in the
                                                    parquet)
  MOM = top-30% 12-1m return − bottom-30% 12-1m return (Jegadeesh-Titman,
                                                         skip last 21 days)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .research_backtester import _get_rebalance_dates
from .research_data import ResearchDataset


def _long_short_from_score(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    quantile: float,
    rebalance_frequency: str,
) -> pd.Series:
    rebalance_dates = _get_rebalance_dates(returns.index, rebalance_frequency)
    aligned = scores.reindex(returns.index).ffill()

    long_w = pd.DataFrame(0.0, index=rebalance_dates, columns=returns.columns, dtype=float)
    short_w = pd.DataFrame(0.0, index=rebalance_dates, columns=returns.columns, dtype=float)

    for date in rebalance_dates:
        row = aligned.loc[date].dropna()
        if len(row) < 10:
            continue
        n = max(int(len(row) * quantile), 2)
        longs = row.nlargest(n).index
        shorts = row.nsmallest(n).index
        long_w.loc[date, longs] = 1.0 / len(longs)
        short_w.loc[date, shorts] = 1.0 / len(shorts)

    long_w = long_w.reindex(returns.index).ffill().fillna(0.0)
    short_w = short_w.reindex(returns.index).ffill().fillna(0.0)

    held_long = long_w.shift(1).fillna(0.0)
    held_short = short_w.shift(1).fillna(0.0)
    long_ret = (held_long * returns).sum(axis=1)
    short_ret = (held_short * returns).sum(axis=1)
    return (long_ret - short_ret).rename("longshort")


def build_ff3_mom_factors(
    dataset: ResearchDataset,
    rebalance_frequency: str = "ME",
    quantile: float = 0.3,
    momentum_lookback: int = 252,
    momentum_skip: int = 21,
) -> pd.DataFrame:
    returns = dataset.returns
    frame = pd.DataFrame(index=returns.index)
    frame["MKT"] = dataset.benchmark_returns.reindex(returns.index).fillna(0.0)

    if dataset.market_caps is not None and not dataset.market_caps.empty:
        size_score = -np.log(dataset.market_caps.replace(0, np.nan))
        frame["SMB"] = _long_short_from_score(size_score, returns, quantile, rebalance_frequency)
    else:
        frame["SMB"] = 0.0

    if dataset.eps is not None and not dataset.eps.empty:
        ep_score = dataset.eps.div(dataset.prices.replace(0, np.nan))
        frame["HML"] = _long_short_from_score(ep_score, returns, quantile, rebalance_frequency)
    else:
        frame["HML"] = 0.0

    span = max(momentum_lookback - momentum_skip, 1)
    mom_score = dataset.prices.pct_change(span, fill_method=None).shift(momentum_skip)
    frame["MOM"] = _long_short_from_score(mom_score, returns, quantile, rebalance_frequency)

    return frame.fillna(0.0)
