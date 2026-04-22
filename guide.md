# Project Guide

## Project Structure

```
millennium-data-quality-25-26/
├── run_book.py                           # THE deliverable — 5-sleeve alpha book
├── research_main.py                      # Research pipeline CLI entry point
├── PROJECT_PLAN.md                       # Research workflow / presentation plan
├── PRESENTATION_NOTES.md                 # Notes for the research deck
├── SESSION_NOTES.md                      # This session's arc, mistakes, improvements
│
├── backtester/
│   ├── WhartonDataSource.parquet         # Local 165-ticker SP500 daily data (2000-2025)
│   ├── data_source.py                    # WhartonDataSource, YahooFinanceDataSource, PickleDataSource
│   ├── research_data.py                  # ResearchDataset loader (prices, returns, volumes, events)
│   ├── research_backtester.py            # Weight-based backtester, lag tables, event studies
│   ├── research_reports.py               # Data quality, SPY validation, CSV/plot exports
│   │
│   ├── metrics.py                        # Sharpe, Sortino, Calmar, alpha/beta, IR, win rate
│   ├── execution.py                      # Market impact models, borrow costs, ADV caps
│   ├── risk_manager.py                   # Pre-trade limits, drawdown kill-switch, vol targeting
│   │
│   ├── alpha_research.py                 # IC, quantile returns, decay, orthogonalization
│   ├── factor_models.py                  # Rolling CAPM, residualization, factor attribution
│   ├── walk_forward.py                   # Purged K-fold CV, deflated Sharpe, PBO
│   │
│   ├── portfolio_optimizer.py            # Mean-variance, risk parity, HRP, shrinkage cov
│   ├── multi_strategy.py                 # Combine strategy returns (equal/risk-parity/hrp)
│   │
│   ├── stress_test.py                    # 10 historical crisis regimes
│   ├── monte_carlo.py                    # Stationary block bootstrap CIs
│   ├── sensitivity.py                    # Parameter grid sweeps
│   │
│   ├── tearsheet.py                      # 6-panel pyfolio-style PNG report
│   └── attribution.py                    # Factor attribution, Brinson, contribution ranking
│
├── strategies/
│   └── research_strategies.py            # 6 SignalStrategy classes for the research framework
│                                         # (Small-Cap Tilt, Value Composite, Earnings Revision,
│                                         #  Sector-Neutral Dividend Yield, Cross-Sectional
│                                         #  Momentum, Low Volatility)
│
├── unit_tests/                           # 88 tests covering every module above
│   ├── test_alpha_research.py
│   ├── test_execution.py
│   ├── test_factor_models.py
│   ├── test_metrics.py
│   ├── test_portfolio_optimizer.py
│   ├── test_research_framework.py
│   ├── test_risk_manager.py
│   ├── test_stress_test.py
│   ├── test_tearsheet.py
│   └── test_walk_forward.py
│
└── book_results/                         # run_book.py output directory
    ├── cumulative_returns.png
    ├── summary.csv
    ├── sleeve_correlation.csv
    ├── stress_regimes.csv
    └── tearsheets/                       # per-sleeve tearsheet PNGs
```

## Architecture Overview

### Data layer — `backtester/data_source.py` + `backtester/research_data.py`
`WhartonDataSource` loads the local parquet, computes split-adjusted and
total-return reference prices, and exposes per-ticker features (volume,
dividends, EPS, market cap, announcement dates). `load_wharton_research_dataset`
pivots that source into aligned cross-sectional panels and returns a
`ResearchDataset` dataclass.

### Strategies — `strategies/research_strategies.py`
Each strategy is a `SignalStrategy` subclass that takes a `ResearchDataset` and
returns a cross-sectional `StrategyOutput` (z-scored per date). They are used
by `run_book.py` and `research_main.py`.

### Backtest engine — `backtester/research_backtester.py`
Weight-based backtester: `build_target_weights` turns signal scores into dated
target weights (equal/inverse_vol/mean_variance, long-short or long-only).
`run_weight_backtest` applies signal lag, rebalance frequency, transaction
costs, and produces a full `BacktestResult` with metrics, lag tables, and
event studies.

### Alpha research — `backtester/alpha_research.py` + `factor_models.py` + `walk_forward.py`
Tools to evaluate whether a signal is real: IC, Rank IC, quantile spread, decay
curves, orthogonalization. Factor neutralization via rolling CAPM and
residualization. Walk-forward validation with purged K-fold and embargo,
deflated Sharpe, probability of backtest overfitting.

### Portfolio construction — `backtester/portfolio_optimizer.py` + `multi_strategy.py`
Mean-variance with turnover penalty, Equal Risk Contribution, Hierarchical
Risk Parity, Ledoit-Wolf shrinkage covariance. `multi_strategy.combine_strategy_returns`
blends individual strategy return series via equal/inverse_vol/risk_parity/hrp.

### Reporting — `backtester/tearsheet.py` + `attribution.py` + `stress_test.py` + `monte_carlo.py`
Tearsheet generator (6-panel PNG), OLS factor attribution, Brinson sector
attribution, historical regime replays against 10 named drawdown windows
(Lehman/Flash Crash/EU Debt/China Devaluation/Brexit/Volmageddon/Q4 2018/COVID/
2022 Rate Shock/SVB), and Politis-Romano stationary block bootstrap CIs.

## Creating a New Signal Strategy

1. Open `strategies/research_strategies.py`
2. Add a new class inheriting from `SignalStrategy` (see existing examples)
3. Implement `generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame`
4. Optionally override `normalize_scores` for sector-neutral handling
5. Either include it in `build_default_strategy_suite()` for the research CLI
   or import it directly in `run_book.py` as a new sleeve

The strategy contract is simple: return a DataFrame with the same shape as
`dataset.prices` where each cell is the cross-sectional score for that ticker
on that date. Higher = more desirable.

## Running the Alpha Book

```sh
.venv/bin/python run_book.py --start 2000-01-01 --end 2025-01-01
```

Key parameters:
- `--target-vol` (default 0.16): realized vol target for the vol-managed sleeve
- `--max-leverage` (default 2.0): cap on the vol-managed sleeve's leverage
- `--sma-window` (default 200): trend filter lookback
- `--selection-top-pct` (default 0.20): top quantile held in each selection sleeve
- `--selection-max-pos` (default 0.08): max weight per name in a selection sleeve

Outputs to `--output-dir` (default `book_results/`).

## Running the Research Pipeline

```sh
.venv/bin/python research_main.py --skip-spy-validation \
  --stress-test --monte-carlo 500 --tearsheet --combine-method hrp
```

Runs the default signal suite through the research backtester and exports
strategy summaries, signal correlation matrix, lag tables, event studies,
tearsheets, stress test results, and Monte Carlo confidence intervals to
`research_outputs/`.

## Metrics

`ExtendedMetrics.calculate()` in `backtester/metrics.py` reports:
- Returns: Avg Daily Return, Cumulative Return, Annualized Return, Log Return
- Risk: Volatility, Max Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Benchmark-relative: Beta, Alpha (Jensen's), Information Ratio
- Trade quality: Win Rate, Profit Factor
- Activity: Avg Daily Turnover, Annualized Turnover

Configure the annual risk-free rate via constructor: `ExtendedMetrics(risk_free_rate=0.05)`.

## Running Tests

```sh
.venv/bin/python -m unittest discover -s unit_tests
```

88 tests, all passing. Covers alpha_research, execution, factor_models,
metrics, portfolio_optimizer, research_framework, risk_manager, stress_test,
tearsheet, and walk_forward.

## Data Sources & Wharton Integration

The framework ships two Wharton loaders side-by-side:

- `WhartonDataSource` — used by `main.py` for the interactive CLI. Accepts
  `use_trfd` and `include_surprise` / `surprise_measure` for merging earnings
  announcements (SUE, surprise) into the daily panel.
- `WhartonResearchDataSource` — used by `research_main.py`, `run_book.py`,
  `run_factor_attribution.py`, and the simulation frontend. Exposes raw-panel
  access (`get_raw_data`), event dates, and a split-adjusted total-return
  reference price tuned for the `SignalStrategy` pipeline.

### `use_trfd` on `WhartonDataSource`

Wharton natively supplies raw prices (`prccd`), a cumulative split-adjustment
factor (`ajexdi`), and a daily total-return factor (`trfd`).

**`use_trfd=True` — Fully-Adjusted Price Series (like Yahoo's `Adj Close`):**
Anchors the most recent split-adjusted price and scales retroactively using
`trfd`. Cash dividends are smoothed backward into the price action so a big
special dividend doesn't register as a phantom price crash. Best for
moving-average style technical signals.

**`use_trfd=False` — Split-only adjustment:**
Uses `prccd / ajexdi` and keeps cash distributions in a separate `Dividend`
column, so a backtester that simulates physical cash deposits for dividends
can credit them explicitly. Best for portfolio-accounting strictness.
