from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .data_source import WhartonDataSource

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


def load_wharton_research_dataset(
    file_path: str,
    start_date: str,
    end_date: str,
    tickers: Optional[Iterable[str]] = None,
    use_total_return: bool = True,
    min_history: int = 252,
    event_columns: Sequence[str] = DEFAULT_EVENT_COLUMNS,
) -> ResearchDataset:
    source = WhartonDataSource(file_path=file_path, use_total_return=use_total_return)
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
