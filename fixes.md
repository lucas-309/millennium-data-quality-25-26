# Fixes / Improvements (Ideas)

## Data source verification (SPY vs constituents)
- Replace the current “normalized-price weighted sum” replication with a **self-financing replication**:
  - Convert constituent weights into **share counts** on the effective holdings date (or on the first common trading day).
  - Let weights **drift** with relative returns, and optionally **rebalance** at a configurable frequency (daily/weekly/monthly).
- Model **index/ETF realities** explicitly so “FAIL” is interpretable:
  - Handle **constituent changes over time** (adds/removes) rather than using a single static holdings snapshot.
  - Decide whether you’re validating against **SPY ETF** (fees, tracking error, distributions) vs **S&P 500 index** (different target).
- Improve missing-data handling:
  - Avoid dropping tickers then renormalizing silently; emit a report of removed tickers and weight lost.
  - Align calendars carefully (trading days, corporate action dates, timezone-naive indices).
- Upgrade diagnostics from a single threshold to a richer report:
  - Per-day diff summary + top-N worst days already exists; add **attribution** (which constituents drive tracking error).
  - Store verification results to a CSV/JSON artifact for reproducibility.

## Wharton / WRDS integration
- Make `WhartonDataSource` adjustment modes explicit and testable:
  - `use_trfd=True` should be treated as “total-return-like adjusted series”; `use_trfd=False` as “split-adjusted + explicit dividends”.
  - Add sanity checks for each ticker: monotonic `trfd` expectations (if applicable), non-zero `ajexdi`, missingness rates.
- Separate WRDS extraction from ingestion:
  - Standardize a **required schema** for WRDS exports (columns + dtypes), and validate on load.
  - Write a merge/update utility that appends new tickers/dates into the Parquet dataset and deduplicates by `(tic, datadate)`.

## Backtester engine / portfolio accounting
- Add a “realistic accounting” mode:
  - Transaction costs, slippage, borrow costs for shorts, and dividend cashflows (especially when `use_trfd=False`).
  - Robust handling for “insufficient cash” in dynamic sizing (partial fills vs skip).
- Unify duplicate engines:
  - There are two equity engines (`backtester/backtest_engine.py` and `backtester/backtesters/equity_backtest.py`); keep one canonical.

## Structure / maintainability
- Split `backtester/data_source.py` into separate files (Yahoo / Pickle / Wharton) and keep shared verification helpers separate.
- Add CLI scripts for repeatable workflows:
  - “cache data”, “verify SPY tracking”, “parse constituent changes”, “merge WRDS extracts”.

## Tests
- Add unit tests for:
  - `WhartonDataSource` adjustment math (split-adjusted and `trfd`-anchored series).
  - Constituent-changes parsing (schema + tickers extracted).
  - SPY replication invariants (no look-ahead, share-count accounting, rebalancing behavior).

