# Wharton/WRDS Query Checklist (To Reconcile SPY Tracking)

Use this checklist to rebuild a point-in-time constituent panel that tracks SPY more closely than ticker-only pulls.

## Goal

Build a daily panel with:
- Correct S&P 500 membership by date
- Correct ticker/security identity through renames/spinoffs
- Daily prices/adjustment fields from a consistent source
- Daily constituent weights (or best available approximation)

## 1) Pull Point-in-Time S&P 500 Membership

Primary requirement: membership start/end dates per security.

- **Need fields**:
  - `gvkey` (or equivalent permanent firm identifier)
  - `iid` (security identifier, if available)
  - membership `from` date
  - membership `thru` date
  - index identifier (S&P 500)

If your WRDS entitlement has a dedicated index-constituent history table, use that directly.  
If not, use Compustat index constituent history tables and filter to S&P 500.

## 2) Pull Security Identifier History (Ticker/CUSIP Through Time)

Ticker alone is not stable. You need a mapping table with effective dates.

- **Need fields**:
  - `gvkey`
  - `iid`
  - `tic` (historical ticker)
  - `cusip` (historical CUSIP)
  - security-name field
  - identifier effective start/end dates

Use this to map names like `FI/FISV`, `PARA/VIAB/PARAA`, class-share format changes, etc.

## 3) Pull Daily Prices + Adjustment Inputs

From your Wharton/Compustat daily security file (same source for SPY and constituents).

- **Need fields**:
  - `gvkey`, `iid`, `datadate`
  - `prccd`
  - `ajexdi`
  - `trfd`
  - `cshtrd`
  - `tic` (for diagnostics only, not primary key)
  - optional: `secstat`, `exchg`, `fic`

Keep your existing adjusted-close construction (`prccd/ajexdi` plus `trfd`) consistent for all securities.

## 4) Pull Point-in-Time Weights

Best case: index constituent weights by date.

- **Need fields**:
  - index date
  - `gvkey`/`iid` (or mappable key)
  - constituent weight

If true historical daily weights are unavailable:
- use monthly/quarterly constituent weights, then forward-fill between rebalances;
- avoid a single static holdings file for long windows.

## 5) Join Order (Recommended)

1. Membership history (date-filtered to backtest window)  
2. Join identifier history (`gvkey` + `iid` + effective dates)  
3. Join daily prices (`gvkey` + `iid` + `datadate`)  
4. Join weights by date/security  
5. Build final daily panel: one row per (`date`, `security`)

## 6) Validation Checks (Must Pass)

- No duplicate security rows per (`date`, `gvkey`, `iid`)
- SPY present for full test window
- Membership count roughly matches expected index size by date
- Missing-price rate by date < threshold (track and report)
- Top-weight names present on worst tracking days
- Weight sum per date ~= 1.0 (after expected normalization)

## 7) Minimum Practical Query Set

If you only run three extra pulls, do these:

1. **Index constituent history** (with start/end dates)  
2. **Security identifier history** (`gvkey/iid` to ticker/CUSIP through time)  
3. **Point-in-time constituent weights** (daily if possible, else lower frequency)

This is the minimal set to fix most Wharton-vs-Yahoo discrepancy sources.

## 8) Implementation Notes For This Repo

- Keep `YahooFinanceDataSource.calculate_weighted_portfolio` as the baseline verifier.
- Build a pre-processing step that creates a canonical daily file:
  - `date`, `gvkey`, `iid`, `tic`, `adj_close`, `weight`, `in_index`
- Then feed only `tic`/`adj_close` pivots into the existing verification functions.
- Keep alias report (`wharton_alias_report.csv`) as a diagnostic artifact, not the primary mapping engine.
