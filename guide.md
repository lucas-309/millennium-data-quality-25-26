# Project Guide

## Project Structure

```text
millennium-data-quality-25-26/
├── run_book.py                    # Five-sleeve book runner
├── run_ensemble.py                # Ensemble runner built on the same sleeves
├── research_main.py               # Research CLI for the default strategy suite
├── PROJECT_PLAN.md
├── PRESENTATION_NOTES.md
├── SESSION_NOTES.md
│
├── backtester/
│   ├── WhartonDataSource.parquet  # Included sample dataset
│   ├── data_source.py             # Data loaders and helpers
│   ├── research_data.py           # ResearchDataset loader
│   ├── research_backtester.py     # Weight backtester and one-strategy entry point
│   ├── research_reports.py        # CSV / chart export helpers
│   ├── metrics.py
│   ├── execution.py
│   ├── risk_manager.py
│   ├── alpha_research.py
│   ├── factor_models.py
│   ├── walk_forward.py
│   ├── portfolio_optimizer.py
│   ├── multi_strategy.py
│   ├── stress_test.py
│   ├── monte_carlo.py
│   ├── sensitivity.py
│   ├── tearsheet.py
│   └── attribution.py
│
├── strategies/
│   └── research_strategies.py     # Simple starter strategies and SignalStrategy
│
├── unit_tests/
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
└── book_results/                  # Example output directory
```

## Core Mental Model

### Data
`backtester/research_data.py` loads the parquet into a `ResearchDataset`
containing aligned price and return panels plus optional metadata and events.

### Strategies
`strategies/research_strategies.py` is intentionally small. The default starter
set is:

- `Mean Reversion`
- `Momentum`
- `Low Volatility`

There is also a simple `ShortTermReversalStrategy` used by the simulator work.

### Backtester
The main collaborator-facing function is
`backtester.research_backtester.run_strategy_backtest(...)`.

A strategy only needs to:

1. Return scores aligned to `dataset.prices`
2. Use higher scores for stronger long candidates

The backtester handles:

- cross-sectional normalization
- quantile selection
- weight construction
- lag
- turnover
- transaction costs
- lag tables
- optional event studies

## Adding a New Strategy

Open `strategies/research_strategies.py` and follow the existing classes:

1. Inherit from `SignalStrategy`
2. Implement `generate_scores(self, dataset)`
3. Return a DataFrame aligned to `dataset.prices`

Example shape:

```python
class MyStrategy(SignalStrategy):
    name = "My Strategy"

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        return dataset.prices.pct_change(20)
```

To run it directly:

```python
result = run_strategy_backtest(dataset, MyStrategy(), config)
```

To add it to the default CLI suite, include it in
`build_default_strategy_suite()`.

## Running the Book

```sh
.venv/bin/python run_book.py
```

Optional tearsheets:

```sh
.venv/bin/python run_book.py --tearsheet
```

## Running the Research CLI

```sh
.venv/bin/python research_main.py --skip-spy-validation --combine-method hrp
```

Optional heavier analysis:

```sh
.venv/bin/python research_main.py \
  --skip-spy-validation \
  --stress-test \
  --monte-carlo 500 \
  --tearsheet \
  --combine-method hrp
```

## Tests

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
