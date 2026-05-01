from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .data_source import WhartonResearchDataSource

DEFAULT_EVENT_COLUMNS = ("anncdate", "recorddate", "paydate", "divdpaydate")
DEFAULT_METADATA_COLUMNS = (
    "conm",
    "gvkey",
    "gsector",
    "gind",
    "gsubind",
    "sic",
    "naics",
    "fic",
    "exchg",
)
DEFAULT_FINANCIAL_RATIOS_PATH = Path(__file__).resolve().parent / "financial_ratios.parquet"


@dataclass
class ResearchDataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    benchmark_returns: pd.Series
    volumes: Optional[pd.DataFrame] = None
    dividends: Optional[pd.DataFrame] = None
    eps: Optional[pd.DataFrame] = None
    market_caps: Optional[pd.DataFrame] = None
    metadata: Optional[pd.DataFrame] = None
    events: Optional[pd.DataFrame] = None
    # Wharton WRDS Financial Ratios — monthly cross-section of the
    # Fama-French value sleeve inputs (B/M, E/P, CF/P, D/P), pre-aligned to
    # the daily price index by forward-fill so a strategy can read them as
    # `dataset.fundamentals["bm"]` without doing the resample itself.
    fundamentals: Optional[dict[str, pd.DataFrame]] = None

    @property
    def tickers(self) -> list[str]:
        return self.prices.columns.tolist()


def _clean_panel(panel: pd.DataFrame, min_history: int = 252) -> pd.DataFrame:
    if panel.empty:
        return panel

    panel = panel.sort_index().sort_index(axis=1)
    panel = panel.dropna(axis=1, how="all")
    if min_history > 0:
        valid_columns = panel.count() >= min_history
        panel = panel.loc[:, valid_columns]
    return panel


def _panel_from_raw(raw: pd.DataFrame, value_column: str, fill_value: Optional[float] = None) -> pd.DataFrame:
    if raw.empty or value_column not in raw.columns:
        return pd.DataFrame()

    panel = raw.pivot(index="datadate", columns="tic", values=value_column).sort_index().sort_index(axis=1)
    if fill_value is not None:
        panel = panel.fillna(fill_value)
    return panel


def _build_metadata(raw: pd.DataFrame, metadata_columns: Sequence[str]) -> pd.DataFrame:
    available_columns = [column for column in metadata_columns if column in raw.columns]
    if not available_columns:
        return pd.DataFrame()

    ordered = raw.sort_values(["tic", "datadate"])
    latest = ordered.groupby("tic")[available_columns].last()
    latest.index.name = "ticker"
    return latest.sort_index()


def load_financial_ratios_panel(
    parquet_path: str,
    daily_index: pd.DatetimeIndex,
    aligned_tickers: Sequence[str],
) -> Optional[dict[str, pd.DataFrame]]:
    """Read the pre-built financial-ratios parquet and align each ratio to
    the price panel's daily index.

    The parquet is monthly (one row per public_date, MultiIndex columns
    `(ratio, ticker)`). We forward-fill into the daily trading calendar with
    a one-month execution lag — Wharton's `public_date` is the *release*
    date, but we conservatively wait an extra month so a strategy never
    trades on a ratio that wasn't fully disclosed by EOD.
    """
    panel = pd.read_parquet(parquet_path)
    if panel.empty:
        return None

    panel = panel.sort_index().sort_index(axis=1)
    # Conservative one-month publication lag: a public_date of 2024-03-31
    # only becomes tradeable on 2024-04-30. Without this shift a strategy
    # would peek at end-of-month fundamentals on the same trading day.
    panel.index = panel.index + pd.offsets.MonthEnd(1)

    universe = pd.Index(aligned_tickers)
    fundamentals: dict[str, pd.DataFrame] = {}
    for ratio in panel.columns.get_level_values(0).unique():
        # Slice one ratio at a time, restrict to the tickers that survived
        # the price-panel cleanup, then ffill onto daily.
        ratio_panel = panel[ratio]
        ratio_panel = ratio_panel.reindex(columns=universe)
        # Reindex to the union of monthly + daily so ffill works across
        # both grids, then restrict back to daily trading days.
        joined_index = ratio_panel.index.union(daily_index).sort_values()
        ratio_daily = ratio_panel.reindex(joined_index).ffill().reindex(daily_index)
        fundamentals[ratio] = ratio_daily
    return fundamentals


def load_wharton_research_dataset(
    file_path: str,
    start_date: str,
    end_date: str,
    tickers: Optional[Iterable[str]] = None,
    use_total_return: bool = True,
    min_history: int = 252,
    event_columns: Sequence[str] = DEFAULT_EVENT_COLUMNS,
    financial_ratios_path: Optional[str | Path] = DEFAULT_FINANCIAL_RATIOS_PATH,
) -> ResearchDataset:
    # Memory note: the WhartonResearchDataSource holds the full long-form
    # parquet in self.data (~1-2GB after derived columns). We extract the
    # wide-form panels we need, then release the long-form view explicitly
    # before returning so the caller's working set is bounded by the wide
    # panels (~250MB), not the long-form raw frame.
    import gc as _gc
    source = WhartonResearchDataSource(file_path=file_path, use_total_return=use_total_return)
    requested_tickers = sorted({ticker.upper() for ticker in tickers}) if tickers else source.list_tickers()

    raw = source.get_raw_data(
        tickers=requested_tickers,
        start_date=start_date,
        end_date=end_date,
        columns=[
            "split_adjusted_close",
            "total_return_reference",
            "Volume",
            "Cash Dividend",
            "eps",
            "Market Cap",
            *DEFAULT_METADATA_COLUMNS,
            *event_columns,
        ],
    )

    prices = source.get_historical_data(requested_tickers, start_date, end_date)
    prices = _clean_panel(prices, min_history=min_history)
    if prices.empty:
        raise ValueError("No usable Wharton price history was found for the requested date range.")

    aligned_tickers = prices.columns.tolist()
    raw = raw[raw["tic"].isin(aligned_tickers)].copy()

    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    benchmark_returns = returns.mean(axis=1, skipna=True).fillna(0.0).rename("equal_weight_market")

    volumes = _clean_panel(_panel_from_raw(raw, "Volume"), min_history=0).reindex(columns=aligned_tickers)
    dividends = _clean_panel(_panel_from_raw(raw, "Cash Dividend", fill_value=0.0), min_history=0).reindex(columns=aligned_tickers)
    eps = _clean_panel(_panel_from_raw(raw, "eps"), min_history=0).reindex(columns=aligned_tickers)
    market_caps = _clean_panel(_panel_from_raw(raw, "Market Cap"), min_history=0).reindex(columns=aligned_tickers)
    metadata = _build_metadata(raw, DEFAULT_METADATA_COLUMNS).reindex(aligned_tickers)
    events = source.get_event_dates(
        tickers=aligned_tickers,
        event_columns=event_columns,
        start_date=start_date,
        end_date=end_date,
    )
    fundamentals = None
    if financial_ratios_path is not None:
        ratios_path = Path(financial_ratios_path)
        if ratios_path.exists():
            fundamentals = load_financial_ratios_panel(
                parquet_path=str(ratios_path),
                daily_index=prices.index,
                aligned_tickers=aligned_tickers,
            )

    # Drop the heavy long-form references before returning. Without this
    # the parent caller's gc.collect() can't reach them — pandas pins
    # numpy buffers via internal cache structures.
    try:
        source.data = None  # type: ignore[assignment]
    except Exception:
        pass
    del source, raw
    _gc.collect()

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
        fundamentals=fundamentals,
    )
