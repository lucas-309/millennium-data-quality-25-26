# Presentation Notes

## 1. Data Background

Price data used
- Local Wharton / Compustat daily file: `backtester/WhartonDataSource.parquet`
- Split-adjusted close: `PRCCD / AJEXDI`
- Total-return reference: `(PRCCD / AJEXDI) * TRFD`

Non-price data used
- `divd` for trailing dividend yield
- `eps` for earnings-yield and revision signals
- `Market Cap` for size
- `anncdate` and `divdpaydate` for event studies
- `gsector` for sector-neutral dividend ranking

## 2. Literature Reviewed

- Jegadeesh and Titman (1993): momentum
- Jegadeesh (1990) / Lehmann (1990): short-term reversal
- Frazzini and Pedersen (2014): Betting Against Beta
- Dividend / payout anomaly literature
- Post-earnings-announcement drift literature for revisions

## 3. Data Quality Checks

From `research_outputs_v3/data_quality_report.csv`:
- Rows: `532,393`
- Tickers: `165`
- Date range: `2013-01-02` to `2025-12-10`
- Duplicate `ticker x date`: `0`
- Non-positive split-adjusted prices: `0`
- Non-positive total-return references: `0`
- Split-adjustment events: `30,953`
- Dividend events: `7,671`

Important missingness:
- `trfd`: `13,024`
- `divd`: sparse by construction
- `eps`: `4,864`

## 4. Backtester Architecture

- Data loading: `backtester/data_source.py`, `backtester/research_data.py`
- Portfolio construction: `backtester/research_backtester.py`
- Strategy definitions: `strategies/research_strategies.py`
- Research runner: `research_main.py`

Portfolio mechanics
- Input: daily cross-sectional signal scores
- Construction: long-short deciles, equal-weight by default
- Execution lag: `1` day
- Transaction costs: `10` bps
- Performance: return, vol, Sharpe, Sortino, drawdown, turnover, t-cost drag

## 5. Backtester Validation

- SPY holdings file snapshot: `27-Feb-2025`
- Validation must therefore be interpreted as a point-in-time reconstruction, not a full-history replication
- The Yahoo-based validation path was hardened with batching, retries, and single-name fallback
- A clean point-in-time rerun is still subject to Yahoo throttling; use the exported validation artifact only when the external pull succeeds

## 6. Strategies Tested

### Small-Cap Tilt
- Signal: `-log(market_cap)`
- Economic rationale: persistent size spread
- 2013-01-01 to 2025-12-10 results:
  - Annualized return: `4.19%`
  - Volatility: `9.14%`
  - Sharpe: `0.494`

### Value Composite
- Signal: `dividend_yield + earnings_yield + size`
- Economic rationale: slow-moving cross-sectional cheapness
- Results:
  - Annualized return: `1.96%`
  - Volatility: `7.56%`
  - Sharpe: `0.294`

### EPS Revision
- Signal: `EPS(t) / EPS(t-63) - 1`
- Economic rationale: post-earnings-announcement drift
- Results:
  - Annualized return: `1.05%`
  - Volatility: `5.54%`
  - Sharpe: `0.216`
- Event study: `anncdate`, window `[-30, +30]`

### Sector-Neutral Dividend Yield
- Signal: trailing 12-month dividend yield, ranked within sector
- Economic rationale: payout support without naive sector bet
- Results:
  - Annualized return: `1.07%`
  - Volatility: `5.45%`
  - Sharpe: `0.222`
- Event study: `divdpaydate`, window `[-30, +30]`

## 7. Signal Combination

Signal correlation highlights from `research_outputs_v3/signal_correlation.csv`:
- Small-Cap Tilt vs Value Composite: `0.682`
- Small-Cap Tilt vs EPS Revision: `-0.253`
- Small-Cap Tilt vs Sector-Neutral Dividend Yield: `0.036`

Final combination logic
- keep only positive standalone strategies
- drop redundant signals with correlation above the threshold
- blend the surviving strategy portfolios equally

Selected combined portfolio
- `Small-Cap Tilt`
- `EPS Revision`
- `Sector-Neutral Dividend Yield`

## 8. Final Portfolio Result

From `research_outputs_v3/strategy_summary.csv`:
- Annualized return: `2.39%`
- Annualized volatility: `3.59%`
- Sharpe: `0.675`
- Sortino: `1.116`
- Max drawdown: `-10.68%`
- Average turnover: `0.0239`
- Annualized t-cost drag: `0.60%`

## 9. Recommended Story For The Deck

- The original anomaly set was too generic for this universe
- The research reset kept the backtester but replaced the strategy slate with signals that were actually positive net of costs
- The best outcome came from combining low-correlation standalone portfolios, not averaging every raw signal together
- The final headline portfolio is a diversified blend of size, EPS revision, and sector-neutral dividend yield
