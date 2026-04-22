"""Compose a ResearchDataset from the yfinance per-ticker cache.

Mirrors the shape returned by ``load_wharton_research_dataset`` so the
existing backtester and strategy stack run unchanged. Where yfinance can't
match Compustat (e.g. true historical EPS panels, point-in-time shares
outstanding) the fields are approximated and the degradations are noted
in the module docstring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .research_data import ResearchDataset
from .yfinance_cache import list_cached_tickers, load_cached

# Wikipedia GICS sector names → numeric GICS sector codes (match the
# Compustat convention the simulator frontend already renders). We include the
# common yfinance.info synonyms ("Technology" vs "Information Technology",
# "Financial Services" vs "Financials", etc.) so both data sources map cleanly.
GICS_CODE_BY_NAME = {
    "Energy": "10",
    "Materials": "15",
    "Basic Materials": "15",
    "Industrials": "20",
    "Consumer Discretionary": "25",
    "Consumer Cyclical": "25",
    "Consumer Staples": "30",
    "Consumer Defensive": "30",
    "Health Care": "35",
    "Healthcare": "35",
    "Financials": "40",
    "Financial Services": "40",
    "Information Technology": "45",
    "Technology": "45",
    "Communication Services": "50",
    "Communications": "50",
    "Utilities": "55",
    "Real Estate": "60",
}


def _slice_price_panel(
    per_ticker: dict,
    field: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for ticker, payload in per_ticker.items():
        hist = payload.get("history")
        if hist is None or hist.empty or field not in hist.columns:
            continue
        series = hist[field].loc[(hist.index >= start) & (hist.index <= end)]
        if series.empty:
            continue
        cols[ticker] = series
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()


def _build_dividends_panel(
    per_ticker: dict,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for ticker, payload in per_ticker.items():
        div = payload.get("dividends")
        if div is None or len(div) == 0:
            continue
        s = div.reindex(index).fillna(0.0)
        cols[ticker] = s
    if not cols:
        return pd.DataFrame(0.0, index=index, columns=[])
    return pd.concat(cols, axis=1).fillna(0.0).sort_index()


def _build_metadata(
    per_ticker: dict,
    sp500_frame: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows = {}
    sp500_lookup: dict[str, pd.Series] = {}
    if sp500_frame is not None and not sp500_frame.empty:
        sp500_lookup = {r["ticker"]: r for _, r in sp500_frame.iterrows()}
    for ticker, payload in per_ticker.items():
        info = payload.get("info") or {}
        # Prefer Wikipedia's GICS sector (exact names) over yfinance info
        # (which uses retail-friendly synonyms).
        wiki_row = sp500_lookup.get(ticker)
        sector_name = (wiki_row.get("sector") if wiki_row is not None else None) or info.get("sector")
        subind = (wiki_row.get("subindustry") if wiki_row is not None else None) or info.get("industry")
        name = info.get("longName") or info.get("shortName") or (
            wiki_row.get("security") if wiki_row is not None else ""
        )
        gsector = GICS_CODE_BY_NAME.get(sector_name, "") if sector_name else ""
        rows[ticker] = {
            "conm": name,
            "gsector": gsector,
            "gind": subind or "",
            "sic": info.get("sic", ""),
            "naics": info.get("naics", ""),
            "fic": info.get("country", ""),
            "exchg": info.get("exchange", ""),
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "ticker"
    return frame.sort_index()


def _build_events(per_ticker: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for ticker, payload in per_ticker.items():
        div = payload.get("dividends")
        if div is None or len(div) == 0:
            continue
        mask = (div.index >= start) & (div.index <= end)
        for dt, amount in div.loc[mask].items():
            rows.append({"ticker": ticker, "event_date": dt, "event_type": "dividend", "amount": float(amount)})
        splits = payload.get("splits")
        if splits is not None and len(splits) > 0:
            mask_s = (splits.index >= start) & (splits.index <= end)
            for dt, ratio in splits.loc[mask_s].items():
                rows.append({"ticker": ticker, "event_date": dt, "event_type": "split", "amount": float(ratio)})
    if not rows:
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "amount"])
    return pd.DataFrame(rows)


def load_yfinance_research_dataset(
    start_date: str,
    end_date: str,
    tickers: Optional[Iterable[str]] = None,
    min_history: int = 252,
    sp500_frame: Optional[pd.DataFrame] = None,
) -> ResearchDataset:
    """Build a ResearchDataset from the yfinance cache.

    Parameters
    ----------
    start_date, end_date : ISO date strings
    tickers : iterable of symbols, or None to load every cached ticker
    min_history : drop tickers with fewer than this many trading days in the window
    sp500_frame : optional dataframe with ticker → sector mapping (used to fill
        metadata when yfinance.info is missing). Pass the frame from
        backtester.sp500_universe.load_sp500_universe().
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if tickers is None:
        tickers = list_cached_tickers()
    per_ticker: dict[str, dict] = {}
    for t in tickers:
        payload = load_cached(t)
        if payload is None:
            continue
        per_ticker[t] = payload
    if not per_ticker:
        raise ValueError("yfinance cache is empty; run run_sp500_fetch.py first")

    # Use Adj Close as the total-return reference (yfinance auto-reinvests
    # dividends into Adj Close when auto_adjust=False + actions=True are set).
    prices = _slice_price_panel(per_ticker, "Adj Close", start_ts, end_ts)
    if prices.empty:
        raise ValueError(f"no price data in the window {start_date} → {end_date}")

    prices = prices.dropna(axis=1, how="all").sort_index().sort_index(axis=1)
    if min_history > 0:
        keep = prices.count() >= min_history
        prices = prices.loc[:, keep]
    if prices.empty:
        raise ValueError(f"no tickers have >= {min_history} observations in the window")

    aligned = prices.columns.tolist()
    per_ticker = {t: per_ticker[t] for t in aligned}

    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_returns = returns.mean(axis=1, skipna=True).fillna(0.0).rename("equal_weight_market")

    volumes = _slice_price_panel(per_ticker, "Volume", start_ts, end_ts).reindex(columns=aligned)
    dividends = _build_dividends_panel(per_ticker, prices.index).reindex(columns=aligned).fillna(0.0)

    # EPS: trailing EPS from info as a constant panel — thin but consistent.
    # Honest limitation relative to Compustat's quarterly EPS history.
    eps_scalar = {}
    for ticker, payload in per_ticker.items():
        info = payload.get("info") or {}
        val = info.get("trailingEps")
        if isinstance(val, (int, float)) and np.isfinite(val):
            eps_scalar[ticker] = float(val)
    eps = pd.DataFrame(index=prices.index, columns=aligned, dtype=float)
    for t, v in eps_scalar.items():
        if t in eps.columns:
            eps[t] = v

    # Market cap: sharesOutstanding × daily adjusted close (current-shares proxy).
    shares = {}
    for ticker, payload in per_ticker.items():
        info = payload.get("info") or {}
        val = info.get("sharesOutstanding")
        if isinstance(val, (int, float)) and np.isfinite(val) and val > 0:
            shares[ticker] = float(val)
    shares_series = pd.Series(shares)
    market_caps = prices.mul(shares_series.reindex(prices.columns).fillna(0.0), axis=1)
    market_caps = market_caps.replace(0.0, np.nan)

    metadata = _build_metadata(per_ticker, sp500_frame).reindex(aligned)
    events = _build_events(per_ticker, start_ts, end_ts)

    return ResearchDataset(
        prices=prices,
        returns=returns,
        benchmark_returns=benchmark_returns,
        volumes=volumes,
        dividends=dividends,
        eps=eps,
        market_caps=market_caps,
        metadata=metadata,
        events=events,
    )
