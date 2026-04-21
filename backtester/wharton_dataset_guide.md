# Wharton Dataset Quick Guide

This is a practical, human-written guide for using the Wharton dataset in this repo.

## What this dataset is

`WhartonDataSource4.parquet` is a daily security-level dataset (mostly U.S. equities and ETF rows like `SPY`) used by `WhartonDataSource` in `backtester/data_source.py`.

Each row is one `(tic, datadate)` observation.

---

## How to load it in code

```python
from data_source import WhartonDataSource

# default source: WhartonDataSource4.parquet (or .csv fallback)
ds = WhartonDataSource(
    use_trfd=True,           # dividend-aware adjusted close
    include_surprise=True,   # merge SUE/events from "SUE_quarterly.csv" (default)
)
```

### Main methods

- `get_historical_data(tickers, start_date, end_date)`  
  Returns wide price dataframe (index=date, columns=ticker, values=`Adj Close`).

- `get_historical_data_with_volume(tickers, start_date, end_date)`  
  Returns per-ticker dataframes with `Adj Close`, and if available `Volume`/`Dividend`.

---

## Column groups

You will see three types of columns:

1. **Raw WRDS/Wharton columns** (from the source file)
2. **Derived columns** (computed in `WhartonDataSource`)
3. **Surprise/event columns** (optional, from `SUE_quarterly.csv`)

---

## 1) Raw WRDS/Wharton columns (from source file)

### Identity and security keys

- `tic`: ticker symbol in source format (Compustat/Wharton style)
- `gvkey`: firm identifier (stable company key)
- `iid`: issue/security identifier
- `cusip`: CUSIP identifier
- `cik`: SEC CIK

### Date and status

- `datadate`: market date for row
- `secstat`: security status
- `costat`: company status
- `dldte`: delisting date (if any)
- `dlrsn`: delisting reason code (if any)

### Price, return, and distribution fields

- `prccd`: close price (raw)
- `prchd`: high price
- `prcld`: low price
- `prcod`: open price
- `prcstd`: price status code
- `ajexdi`: split adjustment factor
- `trfd`: total return factor (used for dividend-aware adjustment)
- `div`: dividend indicator/value field
- `divd`: cash dividend amount
- `divdpaydate`, `divdpaydateind`: dividend pay date metadata
- `divsp`, `divsppaydate`: special dividend fields
- `dvi`: dividend-related flag/indicator
- `dvrated`: dividend rate field
- `recorddate`: record date
- `paydate`, `paydateind`: pay date metadata

### Volume and shares

- `cshtrd`: trading volume
- `cshoc`: shares outstanding

### Company description and classification

- `conm`: company name
- `conml`: long company name
- `busdesc`: business description
- `naics`: NAICS code
- `sic`: SIC code
- `ggroup`, `gind`, `gsector`, `gsubind`: GICS hierarchy fields

### Exchange / geography / domicile

- `exchg`: exchange code
- `fic`: country incorporation code
- `loc`: location country code
- `state`, `city`, `county`: location fields
- `incorp`: incorporation state/country
- `adrrc`: address country code

### Corporate action / capitalization fields

- `anncdate`: announcement date field
- `capgn`, `capgnpaydate`: capital gain distribution fields
- `cheqv`, `cheqvpaydate`: cash equivalent/value fields

### Contact and profile metadata

- `add1`, `add2`, `add3`, `add4`, `addzip`: address fields
- `phone`, `fax`, `weburl`: contact fields
- `ein`: employer tax id
- `fyrc`: fiscal year-end month
- `ipodate`: IPO date

### Misc source fields

- `tpci`, `idbflag`, `prican`, `prirow`, `priusa`
- `spcindcd`, `spcseccd`, `spcsrc`
- `stko`, `curcdd`, `curcddv`
- `eps`, `epsmo`

> Note: Some fields are sparsely populated and may be mostly null depending on ticker/date.

---

## 2) Derived columns added by `WhartonDataSource`

- `Split Adj Close`  
  Computed as `prccd / ajexdi`.

- `Adj Close`  
  Main working price used by backtests:
  - if `use_trfd=False`: equals `Split Adj Close`
  - if `use_trfd=True`: `Split Adj Close * (trfd / trfd_anchor)` to better match total-return behavior.

- `Volume`  
  Copied from `cshtrd` if available.

- `Dividend`  
  Added only when `use_trfd=False` and `divd` exists; filled with `0.0` where missing.

---

## 3) Surprise/event columns (when `include_surprise=True`)

These are merged from `SUE_quarterly.csv` by ticker + date using backward `merge_asof`:
each daily row gets the latest known event on or before that day.

- `surprise_ann_date`: matched announcement date (`anndats`)
- `suescore`: standardized unexpected earnings score from event file
- `actual`: announced actual value
- `surpmean`: consensus mean estimate
- `surpstdev`: consensus estimate std dev
- `last_sue`: convenience copy of `suescore`
- `days_since_announcement`: days since latest matched event
- `is_event_day`: `1` when `datadate == surprise_ann_date`, else `0`

---

## Common usage patterns

### A) Price-only backtest (faster)

```python
ds = WhartonDataSource(use_trfd=True, include_surprise=False)
px = ds.get_historical_data(["SPY", "AAPL", "MSFT"], "2024-01-01", "2026-01-01")
```

### B) Price + surprise factor research

```python
ds = WhartonDataSource(use_trfd=True, include_surprise=True, surprise_measure="EPS")
df = ds.data  # long panel with surprise columns included
```

### C) Price + factor accessor (per ticker)

Use this when you want clean per-ticker frames with both price and surprise features:

```python
ds = WhartonDataSource(use_trfd=True, include_surprise=True, surprise_measure="EPS")

factor_panel = ds.get_historical_data_with_factors(
    ["AAPL", "SPY"],
    "2024-01-01",
    "2024-12-31",
)

# factor_panel["AAPL"] columns:
#   Adj Close, last_sue, days_since_announcement, is_event_day
```

You can also choose custom factors:

```python
factor_panel = ds.get_historical_data_with_factors(
    ["AAPL"],
    "2024-01-01",
    "2024-12-31",
    factor_cols=["last_sue", "is_event_day"],
)
```

### D) Using Wharton from `main.py`

`main.py` now supports selecting data source from CLI:
- `auto` (cache -> Yahoo fallback)
- `yahoo`
- `wharton`

When `wharton` is selected, it initializes:

```python
WhartonDataSource(use_trfd=True, include_surprise=True)
```

Ticker normalization is applied automatically in `main.py` so common naming mismatches are handled per source.

---

## Practical caveats

- Ticker naming is not always Yahoo-style (`BRK.B` vs `BRK-B`).
- Some names are partial/missing in specific windows.
- Surprise merge depends on event ticker quality and date alignment.
- For strongest historical replication, prefer `gvkey`/`iid` joins over ticker-only workflows.
