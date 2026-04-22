"""SPY benchmark + risk-free rate helpers, pulled via yfinance and cached.

For use with ``ExtendedMetrics.calculate`` when benchmark-relative performance
(Sharpe with real rf, alpha, beta, information ratio) is needed. Results are
cached per date range under ``data/`` so repeated runs don't re-hit the network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = _REPO_ROOT / "data"


def _cache_path(series_key: str, start: str, end: str, cache_dir: Path) -> Path:
    safe = f"{series_key}_{start.replace('-', '')}_{end.replace('-', '')}.parquet"
    return cache_dir / safe


def _download_close(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(
        ticker, start=start, end=end,
        auto_adjust=False, progress=False, threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} ({start}..{end})")

    # Newer yfinance versions return a MultiIndex even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ("Adj Close", "Close"):
        if col in df.columns:
            return df[col].dropna().astype(float).rename(ticker)
    raise RuntimeError(f"No Adj Close / Close column in yfinance response for {ticker}")


def _load_cached_series(
    ticker: str, series_key: str, start: str, end: str, cache_dir: Path
) -> pd.Series:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(series_key, start, end, cache_dir)
    if path.exists():
        frame = pd.read_parquet(path)
        return frame.iloc[:, 0]
    series = _download_close(ticker, start, end)
    series.to_frame(series_key).to_parquet(path)
    return series


def load_spy_returns(
    start: str, end: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> pd.Series:
    """Daily SPY returns (pct_change of Adj Close), cached per date range."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    prices = _load_cached_series("SPY", "spy", start, end, cache_dir)
    return prices.pct_change().dropna().rename("spy_returns")


def load_risk_free_rate(
    start: str, end: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> pd.Series:
    """Annualized risk-free rate as a decimal (0.05 == 5%) per trading day.

    Source: Yahoo ``^IRX`` (13-week T-bill discount yield). Yahoo quotes it as
    a percent (e.g. 5.12 means 5.12%), so we divide by 100 here. Returns a
    Series indexed by date — divide by ``trading_days`` if you need the
    per-day excess-return subtraction.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    yields_pct = _load_cached_series("^IRX", "irx", start, end, cache_dir)
    return (yields_pct / 100.0).rename("risk_free_rate")


def align_to_index(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Reindex to ``target_index`` and forward-fill small gaps (holidays, etc)."""
    return series.reindex(target_index).ffill()


def load_benchmark_bundle(
    start: str, end: str,
    price_index: Optional[pd.DatetimeIndex] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> dict:
    """Pull SPY returns and the risk-free rate, optionally aligned to a panel.

    Parameters
    ----------
    start, end : str
        YYYY-MM-DD bounds passed through to yfinance.
    price_index : DatetimeIndex, optional
        If provided, both series are reindexed (and forward-filled) to match.
        Pass ``prices.index`` from the price panel you backtest on.
    cache_dir : path, optional
        Defaults to ``<repo>/data/``.

    Returns
    -------
    dict with keys:
        spy_returns      — pd.Series of daily SPY returns
        risk_free_annual — pd.Series of annualized rf yields (decimal)
        risk_free_daily  — pd.Series of rf yields divided by 252
        risk_free_mean   — float, mean annualized rf (scalar for ExtendedMetrics)
    """
    spy = load_spy_returns(start, end, cache_dir)
    rf_annual = load_risk_free_rate(start, end, cache_dir)

    if price_index is not None:
        spy = align_to_index(spy, price_index)
        rf_annual = align_to_index(rf_annual, price_index)

    rf_clean = rf_annual.dropna()
    rf_mean = float(rf_clean.mean()) if len(rf_clean) else 0.0

    return {
        "spy_returns": spy,
        "risk_free_annual": rf_annual,
        "risk_free_daily": rf_annual / 252.0,
        "risk_free_mean": rf_mean,
    }
