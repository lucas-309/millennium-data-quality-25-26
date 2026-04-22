# Millennium Quantitative Research Playground

A backtesting framework for long-only cross-sectional equity factor strategies
on a local Wharton / Compustat daily dataset. Includes a prop-shop grade alpha
research toolkit (IC, quantile spread, factor models, walk-forward validation,
HRP portfolio optimization, stress testing, tearsheets).

The headline deliverable is `run_book.py` — a 5-sleeve alpha book that combines
vol-managed equity, trend-filtered equity, and three long-only factor sleeves
(Momentum, Low Volatility, Small-Cap Tilt) via Hierarchical Risk Parity.

**Best result on Wharton 2000-2025 (165 SP500 names, 10bps t-cost, no lookahead):**

| Strategy | Ann Return | Sharpe | Sortino | Max DD |
|---|---|---|---|---|
| Equal-Weight Benchmark | +13.63% | 0.760 | 0.973 | -50.88% |
| Small-Cap Tilt + Trend | +14.57% | **1.095** | 1.341 | -33.92% |
| **Combined Book (HRP)** | +10.34% | **1.003** | **1.253** | **-23.80%** |

See [SESSION_NOTES.md](SESSION_NOTES.md) for the full session arc, mistakes
made, and notes for improvement.

## Installation

```sh
git clone https://github.com/lucas-309/millennium-data-quality-25-26
cd millennium-data-quality-25-26
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Data file: `backtester/WhartonDataSource.parquet` is included (19MB, 165
SP500 names, 2000-2025).

## Quick Start

### Run the alpha book

```sh
.venv/bin/python run_book.py
```

Uses the default window (2000-01-01 to 2025-01-01), 16% target vol, 200-day
trend filter, long-only top 20% selection sleeves. Outputs go to
`book_results/` including cumulative returns plots, per-sleeve tearsheets,
stress test CSV, and a summary table.

Customize:
```sh
.venv/bin/python run_book.py \
  --start 2010-01-01 --end 2025-01-01 \
  --target-vol 0.16 --max-leverage 2.0 \
  --sma-window 200 --selection-top-pct 0.20 \
  --output-dir my_book_results
```

### Run the research pipeline (multi-signal suite)

```sh
.venv/bin/python research_main.py --skip-spy-validation \
  --stress-test --monte-carlo 500 --tearsheet \
  --combine-method hrp
```

This runs the `build_default_strategy_suite()` slate (Small-Cap Tilt, Value
Composite, Earnings Revision, Sector-Neutral Dividend Yield, Cross-Sectional
Momentum, Low Volatility) through the weight-based research backtester with
lag tables, event studies, tearsheets, stress regimes, Monte Carlo CIs, and
a combined-book output.

## Architecture

See [guide.md](guide.md) for the current project structure and module index.

Research workflow detail: [PROJECT_PLAN.md](PROJECT_PLAN.md)

Presentation notes: [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md)

Session notes, mistakes, improvements: [SESSION_NOTES.md](SESSION_NOTES.md)

## Running Tests

```sh
.venv/bin/python -m unittest discover -s unit_tests
```

88 tests covering the research framework, alpha research tools, factor
models, walk-forward validation, portfolio optimizer, stress testing,
Monte Carlo, tearsheet, execution costs, and risk manager.
