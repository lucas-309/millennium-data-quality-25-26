"""Customer-Supplier Momentum strategy (Cohen & Frazzini 2008).

Long suppliers whose disclosed major customers had strong recent returns; short
suppliers whose customers had weak returns. Exploits slow information diffusion
along the supply chain: customer-level news takes time to be priced into
suppliers' equity.

Expected input to ``generate_orders``:
    A wide-format ``pd.DataFrame`` with a ``DatetimeIndex`` (daily) and ticker
    strings as columns, containing adjusted close prices. Matches the output of
    ``WhartonDataSource.get_historical_data``.

Expected CapIQ relationship file (CSV or Excel):
    - One row per (supplier, customer, fiscal_year) disclosure.
    - Columns (case-insensitive, aliases tolerated):
        * supplier ticker — one of: supplier_ticker, supplier, supplier_tic,
          src_tic, src_ticker, supplier_symbol, supplier_gvkey
        * customer ticker — one of: customer_ticker, customer_tic,
          customer_symbol, cust_tic, cust_ticker, customer_gvkey
          (or a customer_name column + ``customer_name_to_ticker`` crosswalk)
        * fiscal_year — one of: fiscal_year, fyear, year, srcyear, srcdate
        * revenue_fraction (optional) — one of: revenue_fraction, rev_frac,
          salecs_pct, cust_pct, revenue_pct, salecs_frac, salecs
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .order_generator import OrderGenerator


class CustomerSupplierMomentumOrderGenerator(OrderGenerator):
    _SUPPLIER_ALIASES: Sequence[str] = (
        "supplier_ticker", "supplier", "supplier_tic", "src_tic",
        "src_ticker", "supplier_symbol", "supplier_gvkey",
        "stic",  # WRDS Compustat Segment: supplier ticker
    )
    _CUSTOMER_TICKER_ALIASES: Sequence[str] = (
        "customer_ticker", "customer_tic", "customer_symbol",
        "cust_tic", "cust_ticker", "customer_gvkey",
        "ctic",    # WRDS Compustat Segment: customer ticker
        "cgvkey",  # WRDS fallback when ctic is missing
    )
    _CUSTOMER_NAME_ALIASES: Sequence[str] = (
        "customer_name", "cnms", "custname", "cust_name", "customer",
    )
    _YEAR_ALIASES: Sequence[str] = (
        "fiscal_year", "fyear", "year", "srcyear", "srcdate",
    )
    _REVENUE_ALIASES: Sequence[str] = (
        "revenue_fraction", "rev_frac", "salecs_pct", "cust_pct",
        "revenue_pct", "salecs_frac", "salecs",
    )

    # Compustat Segment `ctype` values that represent actual company customers.
    # Other values (GEOSEG, MARKETS, GOVERNMENT, etc.) are not tradeable entities.
    _DEFAULT_CTYPE_FILTER = frozenset({"COMPANY"})

    def __init__(
        self,
        capiq_path: str,
        rebalance_frequency: str = "ME",
        signal_lag_months: int = 0,
        top_fraction: float = 0.2,
        bottom_fraction: float = 0.2,
        long_notional_fraction: float = 0.5,
        short_notional_fraction: float = 0.5,
        min_customers_required: int = 1,
        weight_by_revenue: bool = True,
        customer_name_to_ticker: Optional[Dict[str, str]] = None,
        min_universe: int = 10,
        ctype_filter: Optional[Sequence[str]] = None,
        verbose: bool = False,
    ):
        self.capiq_path = capiq_path
        self.rebalance_frequency = rebalance_frequency
        self.signal_lag_months = signal_lag_months
        self.top_fraction = top_fraction
        self.bottom_fraction = bottom_fraction
        self.long_notional_fraction = long_notional_fraction
        self.short_notional_fraction = short_notional_fraction
        self.min_customers_required = min_customers_required
        self.weight_by_revenue = weight_by_revenue
        self.customer_name_to_ticker = customer_name_to_ticker or {}
        self.min_universe = min_universe
        self.ctype_filter = (
            frozenset(s.upper() for s in ctype_filter)
            if ctype_filter is not None
            else self._DEFAULT_CTYPE_FILTER
        )
        self.verbose = verbose

        self._relationships_by_year = self._load_relationships(capiq_path)
        self._available_years = sorted(self._relationships_by_year.keys())

    # --- CapIQ loader -------------------------------------------------------

    def _load_relationships(self, path: str) -> Dict[int, Dict[str, Dict[str, float]]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CapIQ relationship file not found at {path}")

        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, low_memory=False)

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = self._normalize_relationship_columns(df)

        # Optional ctype filter (WRDS Compustat Segment uses 'COMPANY' for real
        # customers, plus 'GEOSEG'/'MARKETS'/etc. for non-company disclosures).
        if "ctype" in df.columns and self.ctype_filter:
            df = df[df["ctype"].astype(str).str.strip().str.upper().isin(self.ctype_filter)]

        df["supplier_ticker"] = self._clean_ticker(df["supplier_ticker"])
        df["customer_ticker"] = self._clean_ticker(df["customer_ticker"])
        df = df[df["supplier_ticker"].ne("") & df["supplier_ticker"].ne("NAN")]
        df = df[df["customer_ticker"].ne("") & df["customer_ticker"].ne("NAN")]
        df = df[df["customer_ticker"].str.match(r"^[A-Z0-9.]{1,8}$", na=False)]
        # Drop self-relationships — a ticker cannot be a momentum signal for itself.
        df = df[df["supplier_ticker"] != df["customer_ticker"]]

        df["fiscal_year"] = self._coerce_to_year(df["fiscal_year"])
        df = df.dropna(subset=["fiscal_year"])
        df["fiscal_year"] = df["fiscal_year"].astype(int)

        if self.weight_by_revenue and "revenue_fraction" in df.columns:
            df["revenue_fraction"] = pd.to_numeric(df["revenue_fraction"], errors="coerce").abs()
        else:
            df["revenue_fraction"] = 1.0

        # Sort so that duplicate (supplier, customer, year) triples prefer the row
        # with the richest salecs data; then dedupe keeping that row.
        df = df.sort_values("revenue_fraction", na_position="last").drop_duplicates(
            subset=["supplier_ticker", "customer_ticker", "fiscal_year"], keep="first"
        )

        # salecs is in dollars (millions), not fractions. Normalizing by supplier-year
        # total converts to proportional weights — but only if ALL customers in that
        # supplier-year have salecs reported. If any are missing, fall back to equal
        # weights for the whole supplier-year so one reported $M value doesn't crowd
        # out the NaN ones.
        df["weight"] = self._compute_weights(df)
        df = df.dropna(subset=["weight"])

        result: Dict[int, Dict[str, Dict[str, float]]] = {}
        for (year, supplier), group in df.groupby(["fiscal_year", "supplier_ticker"]):
            result.setdefault(int(year), {})[supplier] = dict(
                zip(group["customer_ticker"], group["weight"])
            )

        if self.verbose:
            n_links = sum(len(cust_map) for ymap in result.values() for cust_map in ymap.values())
            n_suppliers = len({s for ymap in result.values() for s in ymap})
            print(
                f"CustomerSupplierMomentum: loaded {n_links} supplier-customer links across "
                f"{len(result)} fiscal years, {n_suppliers} unique suppliers."
            )
        return result

    def _normalize_relationship_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        def _pick(aliases: Sequence[str]) -> Optional[str]:
            for alias in aliases:
                if alias in df.columns:
                    return alias
            return None

        supplier_col = _pick(self._SUPPLIER_ALIASES)
        customer_ticker_col = _pick(self._CUSTOMER_TICKER_ALIASES)
        customer_name_col = _pick(self._CUSTOMER_NAME_ALIASES)
        year_col = _pick(self._YEAR_ALIASES)
        revenue_col = _pick(self._REVENUE_ALIASES)

        if supplier_col is None or year_col is None:
            raise ValueError(
                "CapIQ file must contain a supplier ticker column and a fiscal year column. "
                f"Recognized supplier aliases: {list(self._SUPPLIER_ALIASES)}. "
                f"Recognized year aliases: {list(self._YEAR_ALIASES)}."
            )

        rename_map = {supplier_col: "supplier_ticker", year_col: "fiscal_year"}

        if customer_ticker_col is not None:
            rename_map[customer_ticker_col] = "customer_ticker"
            df = df.rename(columns=rename_map)
        elif customer_name_col is not None and self.customer_name_to_ticker:
            df = df.rename(columns=rename_map)
            crosswalk = {
                k.strip().upper(): v.strip().upper()
                for k, v in self.customer_name_to_ticker.items()
            }
            df["customer_ticker"] = (
                df[customer_name_col].astype(str).str.strip().str.upper().map(crosswalk)
            )
        else:
            raise ValueError(
                "CapIQ file has no recognized customer_ticker column. Provide one of "
                f"{list(self._CUSTOMER_TICKER_ALIASES)}, or pass a customer_name_to_ticker "
                "crosswalk if only customer names are available."
            )

        if revenue_col is not None and revenue_col != "revenue_fraction":
            df = df.rename(columns={revenue_col: "revenue_fraction"})

        return df

    @staticmethod
    def _clean_ticker(series: pd.Series) -> pd.Series:
        # Upper-case, strip whitespace, and normalize WRDS-style class tickers
        # ('BRK-B') to the Wharton parquet convention ('BRK.B').
        return (
            series.astype(str).str.strip().str.upper().str.replace("-", ".", regex=False)
        )

    @staticmethod
    def _compute_weights(df: pd.DataFrame) -> pd.Series:
        weights = pd.Series(np.nan, index=df.index, dtype=float)
        for (_, _), group in df.groupby(["fiscal_year", "supplier_ticker"], sort=False):
            revs = group["revenue_fraction"]
            if revs.notna().all() and revs.sum() > 0:
                weights.loc[group.index] = revs / revs.sum()
            else:
                # Any missing salecs in the group → equal-weight the whole group.
                n = len(group)
                if n > 0:
                    weights.loc[group.index] = 1.0 / n
        return weights

    @staticmethod
    def _coerce_to_year(series: pd.Series) -> pd.Series:
        # Handles three common encodings: integer year (2018), four-digit string
        # ('2018'), and datetime-like strings ('2018-12-31'). Non-parseable rows
        # become NaN and are dropped upstream.
        numeric = pd.to_numeric(series, errors="coerce")
        parsed_dates = pd.to_datetime(series, errors="coerce")
        years_from_dates = parsed_dates.dt.year
        return numeric.where(numeric.notna(), years_from_dates)

    # --- signal computation ------------------------------------------------

    def _get_applicable_year(self, rebalance_date: pd.Timestamp) -> Optional[int]:
        # Conservative no-look-ahead: use the most recent fiscal year whose
        # 10-K would definitively have been filed before rebalance_date.
        target = rebalance_date.year - 1
        candidates = [y for y in self._available_years if y <= target]
        return max(candidates) if candidates else None

    def _compute_signals(
        self,
        monthly_returns: pd.DataFrame,
        signal_month_end: pd.Timestamp,
        year_map: Dict[str, Dict[str, float]],
    ) -> pd.Series:
        if signal_month_end not in monthly_returns.index:
            return pd.Series(dtype=float)

        customer_returns = monthly_returns.loc[signal_month_end]
        signals: Dict[str, float] = {}

        for supplier, customer_weights in year_map.items():
            if supplier not in monthly_returns.columns:
                continue
            valid = {
                c: w for c, w in customer_weights.items()
                if c in customer_returns.index and pd.notna(customer_returns[c])
            }
            if len(valid) < self.min_customers_required:
                continue
            total = sum(valid.values())
            if total <= 0:
                continue
            signals[supplier] = sum(
                (w / total) * customer_returns[c] for c, w in valid.items()
            )

        return pd.Series(signals, dtype=float)

    def _signal_month_end(
        self, monthly_index: pd.DatetimeIndex, rebalance_date: pd.Timestamp
    ) -> Optional[pd.Timestamp]:
        locs = monthly_index.get_indexer([rebalance_date])
        if locs[0] < 0:
            return None
        pos = int(locs[0]) - self.signal_lag_months
        if pos < 0 or pos >= len(monthly_index):
            return None
        return monthly_index[pos]

    # --- order construction ------------------------------------------------

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        if data.empty:
            return []

        data = data.sort_index()
        # Resample to month-end but keep the actual last trading day of each bin
        # as the date label. Otherwise rebalance dates land on weekends/holidays
        # that never appear in the engine's trading-day iteration.
        month_grouper = data.groupby(pd.Grouper(freq=self.rebalance_frequency))
        last_trading_day_per_bin = month_grouper.apply(
            lambda g: g.index[-1] if len(g) else pd.NaT
        ).dropna()
        monthly_prices = data.loc[last_trading_day_per_bin.values]
        monthly_returns = monthly_prices.pct_change(fill_method=None)
        rebalance_dates = monthly_returns.index

        orders: List[Dict[str, Any]] = []
        previous_positions: Dict[str, str] = {}

        # Precompute the "next" rebalance date for each current rebalance, used to
        # require clean price coverage through the holding period (the engine NaNs
        # out portfolio value if a held ticker goes missing mid-period).
        next_rebalance_map = dict(zip(rebalance_dates[:-1], rebalance_dates[1:]))
        last_data_date = data.index[-1]

        for rebalance_date in rebalance_dates:
            applicable_year = self._get_applicable_year(rebalance_date)
            if applicable_year is None:
                continue

            signal_month = self._signal_month_end(rebalance_dates, rebalance_date)
            if signal_month is None:
                continue

            signals = self._compute_signals(
                monthly_returns, signal_month, self._relationships_by_year[applicable_year]
            ).dropna()
            if len(signals) < self.min_universe:
                continue

            holding_end = next_rebalance_map.get(rebalance_date, last_data_date)
            tradeable_today = self._tradeable_on(data, rebalance_date)
            tradeable_forward = self._tradeable_through(data, rebalance_date, holding_end)
            # Only open positions we can hold cleanly to next rebalance;
            # close decisions are made against today's price only.
            signals = signals[signals.index.isin(tradeable_forward)]
            if len(signals) < self.min_universe:
                continue

            n_long = max(int(len(signals) * self.top_fraction), 1)
            n_short = max(int(len(signals) * self.bottom_fraction), 1)
            sorted_signals = signals.sort_values()
            short_tickers = sorted_signals.head(n_short).index.tolist()
            long_tickers = sorted_signals.tail(n_long).index.tolist()

            # Close every previous position (that is tradeable today) before
            # opening the new book. Each rebalance yields exactly the target
            # portfolio shape; turnover cost accepted as-is. If a previously-
            # held ticker is untradeable today (NaN/zero price, e.g. delisted
            # or a data gap), keep it on the books and retry on the next
            # rebalance rather than crashing the engine.
            carried_over: Dict[str, str] = {}
            for ticker, side in previous_positions.items():
                if ticker not in tradeable_today:
                    # Shouldn't normally happen: positions are only opened when
                    # _tradeable_through guarantees coverage through this date.
                    # Carrying over when the ticker has gone NaN today will crash
                    # the engine on portfolio valuation, so the caller's price
                    # panel has a quality issue we can't paper over here.
                    carried_over[ticker] = side
                    continue
                if side == "LONG":
                    orders.append({
                        "date": rebalance_date, "type": "SELL",
                        "ticker": ticker, "quantity": 1.0,
                    })
                else:
                    orders.append({
                        "date": rebalance_date, "type": "BUY",
                        "ticker": ticker, "quantity": 1.0,
                    })

            long_frac = self.long_notional_fraction / len(long_tickers) if long_tickers else 0.0
            short_frac = self.short_notional_fraction / len(short_tickers) if short_tickers else 0.0

            for ticker in long_tickers:
                orders.append({
                    "date": rebalance_date, "type": "BUY",
                    "ticker": ticker, "quantity": long_frac,
                })
            for ticker in short_tickers:
                orders.append({
                    "date": rebalance_date, "type": "SELL",
                    "ticker": ticker, "quantity": short_frac,
                })

            previous_positions = {t: "LONG" for t in long_tickers}
            previous_positions.update({t: "SHORT" for t in short_tickers})
            previous_positions.update(carried_over)

        # Stable sort by date preserves the close-before-open ordering within each date,
        # which matters because the engine reads order.cash at start-of-day and won't
        # refresh it mid-day — closes must execute before opens draw on cash.
        orders.sort(key=lambda o: o["date"])
        return orders

    @staticmethod
    def _tradeable_on(data: pd.DataFrame, rebalance_date: pd.Timestamp) -> set:
        if rebalance_date not in data.index:
            return set()
        row = data.loc[rebalance_date]
        return {t for t, p in row.items() if pd.notna(p) and p > 0}

    @staticmethod
    def _tradeable_through(
        data: pd.DataFrame, from_date: pd.Timestamp, to_date: pd.Timestamp
    ) -> set:
        # A ticker is tradeable for this rebalance period only if it has a valid
        # positive price on every data date from the rebalance day through the
        # next rebalance day. Weaker checks (e.g. only the rebalance day itself)
        # let the engine crash when an intermediate NaN kills portfolio valuation.
        window = data.loc[from_date:to_date]
        if window.empty:
            return set()
        valid = window.notna().all(axis=0) & (window > 0).all(axis=0)
        return set(window.columns[valid])
