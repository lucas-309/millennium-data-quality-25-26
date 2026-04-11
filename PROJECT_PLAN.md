# Project Reset Plan

## Goal
Rebuild the project around a research-grade workflow that can support the full presentation:

- clean local data inputs
- transparent backtester mechanics
- 5 materially different signals
- lag studies
- event studies
- signal combination
- portfolio construction comparisons
- SPY / SPX validation

## Presentation Structure

### 1. Data Background

Price data used
- Wharton / Compustat daily security data in `backtester/WhartonDataSource.parquet`
- Core fields: `prccd`, `ajexdi`, `trfd`, `cshtrd`, `cshoc`
- Optional benchmark / validation source: Yahoo Finance or cached local pickle for `SPY` plus SPY constituents

Non-price data used
- `divd` / `div` for dividend cash flow and yield features
- `anncdate`, `recorddate`, `paydate`, `divdpaydate` for event studies
- `eps` for future non-price extensions
- `gsector`, `gind`, `gsubind`, `sic`, `naics` for sector-aware diagnostics and portfolio constraints

### 2. Literature Reviewed

Core papers to cover in the deck
- Jegadeesh and Titman (1993): medium-term momentum
- Jegadeesh (1990) and Lehmann (1990): short-term reversal / overreaction
- Ang, Hodrick, Xing, and Zhang (2006): low-volatility / low-risk anomaly framing
- Frazzini and Pedersen (2014): Betting Against Beta
- Gatev, Goetzmann, and Rouwenhorst (2006): pairs / relative-value framing if kept as benchmark strategy

Optional extensions if time permits
- Lee and Swaminathan (2000): momentum interaction with trading activity
- Dividend announcement / dividend yield literature for event-based framing

### 3. Data Quality Checks

Required checks to show explicitly
- date range and universe coverage
- duplicate `ticker x date` rows
- missing-value counts by field
- split-adjusted price construction
- total-return construction when `TRFD` is used
- zero / negative price checks
- volume sanity checks
- dividend sparsity and event-date availability
- ticker mapping between Wharton and SPY holdings file

Corporate actions / dividends / splits
- use split-adjusted prices from `PRCCD / AJEXDI`
- use total-return reference price from `(PRCCD / AJEXDI) * TRFD` when evaluating total returns
- compare split-adjusted and total-return behavior on known dividend payers
- verify dividend event dates line up with non-zero dividend observations

### 4. Backtester Architecture

Data portion
- `backtester/data_source.py`
  - `WhartonDataSource`
  - `YahooFinanceDataSource`
  - `PickleDataSource`
- `backtester/research_data.py`
  - loads prices, returns, volume, dividends, metadata, events

Portfolio input
- strategy outputs are daily cross-sectional score panels
- portfolio construction converts scores into weights
- supported weighting methods:
  - equal weight
  - inverse volatility / `1/sigma`
  - mean-variance

Backtest returns from the portfolio minus t-cost
- `backtester/research_backtester.py`
- daily portfolio return = lagged weights x next-day asset returns
- t-cost drag = daily turnover x cost in bps

Lags
- default execution lag = 1 trading day
- lag table supports `-3` through `+5`
- interpretation for the deck:
  - `lag = 0`: `sig(t)` with `r(t+1)`
  - `lag = 1`: `sig(t-1)` with `r(t+1)`
  - negative lags are look-ahead diagnostics and should only be used as a sanity check

Performance metrics
- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- downside volatility
- Sortino ratio
- max drawdown
- average turnover
- annualized turnover
- average t-cost drag
- annualized t-cost drag
- benchmark correlation and information ratio when benchmark returns exist

### 5. Backtester Validation

SPX / SPY reconstruction plan
- use `backtester/holdings-daily-us-en-spy.xlsx`
- rebuild a weighted constituent portfolio
- compare daily returns vs `SPY`
- report:
  - mean absolute return difference
  - max absolute return difference
  - Pearson correlation
  - Spearman correlation
  - worst mismatch days

Important caveat
- if only one holdings file snapshot is available, validation is still useful as a point-in-time reconstruction check, not a full historical replication

### 6. Strategy Slate

The repo now supports these primary strategies in `strategies/research_strategies.py`

1. Small-Cap Tilt
- Signal: negative log market cap
- Motivation: own smaller names and short larger peers
- Why it works: persistent size spread with low turnover
- Why it fails: mega-cap leadership can swamp the factor

2. Value Composite
- Signal: dividend yield + earnings yield + size z-score blend
- Motivation: capture slow-moving cross-sectional cheapness
- Why it works: combines cash-yield and earnings support instead of relying on one valuation field
- Why it fails: cheap names can stay cheap or become macro crowding trades

3. EPS Revision
- Signal: change in EPS versus 63 trading days prior
- Motivation: capture post-announcement drift from improving fundamentals
- Why it works: earnings information diffuses more slowly than pure price action
- Why it fails: estimate updates can lag reality in fast macro shocks

4. Sector-Neutral Dividend Yield
- Signal: trailing 12-month dividend yield, z-scored within sector
- Motivation: isolate payout support without loading on obvious sector bets
- Why it works: within-sector yield comparisons are cleaner than naive cross-sector yield spreads
- Why it fails: high yield can still be a distress trap

### 7. Event Study

Base event study to include
- event type: `anncdate`
- window: `-30` to `+30`
- y-axis: cumulative residual return
- residual return:
  - preferred: stock return minus `SPY`
  - fallback when `SPY` is unavailable in local Wharton data: stock return minus equal-weight universe return

What to plot
- median
- 10th percentile
- 25th percentile
- 75th percentile
- 90th percentile
- event count

### 8. Signal Combination

Required deck content
- pairwise correlation matrix of signal returns
- combined portfolio backtest

Recommended combination order
1. backtest each standalone signal with its own research config
2. keep only positive standalone strategies for the final portfolio blend
3. correlation-screen the strategies to avoid redundant exposures
4. combine the selected strategy portfolios with equal weight or inverse-vol scaling

### 9. Portfolio Construction

Baseline
- monthly rebalance
- one-day execution lag
- long-short deciles
- gross leverage = 1.0
- position cap = 5%

Compare at least two methods
- equal weight on selected names
- inverse volatility `1/sigma`

Optional extension
- mean-variance using rolling covariance and signal scores as expected-return proxy

### 10. Execution Order

1. Validate the data layer
- run missingness checks
- verify adjusted price logic
- inspect event coverage

2. Validate the backtester
- run synthetic unit tests
- run SPY reconstruction
- sanity check turnover and t-cost drag

3. Run single-signal tests
- same rebalance frequency
- same transaction cost assumptions
- same lag convention

4. Run lag tables
- report `SR(-3)` through `SR(5)` for each signal

5. Run event study
- start with dividend announcement dates

6. Run signal combination
- compare combined vs standalone results

7. Build the deck
- one slide per section in the required structure

## Repo Additions For This Reset

New code added
- `backtester/research_data.py`
- `backtester/research_backtester.py`
- `strategies/research_strategies.py`
- `research_main.py`
- `unit_tests/test_research_framework.py`

Core change
- `backtester/data_source.py` now includes `WhartonDataSource`

## Recommended Headline Result Order

Use this ordering in the presentation unless the live numbers say otherwise
- Momentum
- BAB
- Low Volatility
- Volume Shock Reversal
- Dividend Yield
- Combined Multi-Signal

This gives one trend signal, one contrarian signal, one low-risk signal, one flow signal, one non-price / event-linked signal, and one ensemble result.
