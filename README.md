# Millennium Quantitative Research Playground

A local equity-research sandbox built around a simple weight-based backtester
and the included Wharton / Compustat parquet file.

The repo is set up so collaborators can add straightforward strategies without
learning a large framework first. The default starter suite is intentionally
small:

- `Mean Reversion`
- `Momentum`
- `Low Volatility`

Each strategy only needs to return a score DataFrame aligned to
`dataset.prices`. The backtester handles ranking, sizing, lag, turnover, and
transaction costs.

## Installation

```sh
git clone https://github.com/lucas-309/millennium-data-quality-25-26
cd millennium-data-quality-25-26
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The sample data file `backtester/WhartonDataSource.parquet` is already checked
in.

## Quick Start

### Run the five-sleeve book

```sh
.venv/bin/python run_book.py
```

This builds:

- `Vol-Managed Equity`
- `Trend-Filtered Equity`
- `Momentum + Trend`
- `Mean Reversion + Trend`
- `Low Volatility + Trend`

Outputs go to `book_results/`. Tearsheets are opt-in:

```sh
.venv/bin/python run_book.py --tearsheet
```

### Run the research pipeline

```sh
.venv/bin/python research_main.py --skip-spy-validation --combine-method hrp
```

The default research suite is the same simple starter set: Mean Reversion,
Momentum, and Low Volatility. Optional analysis flags such as
`--stress-test`, `--monte-carlo`, and `--tearsheet` are available when you
need them.

## Strategy Contract

See [guide.md](guide.md) for the full module map. The important part for
strategy authors lives in `strategies/research_strategies.py`:

1. Inherit from `SignalStrategy`
2. Implement `generate_scores(self, dataset)`
3. Return a DataFrame aligned to `dataset.prices`

Higher score means a stronger long candidate. You do not need to build weights
or account for transaction costs inside the strategy.

## Tests

```sh
.venv/bin/python -m unittest discover -s unit_tests
```

Supporting notes:

- [guide.md](guide.md): project structure and extension points
- [PROJECT_PLAN.md](PROJECT_PLAN.md): research workflow notes
- [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md): deck notes
- [SESSION_NOTES.md](SESSION_NOTES.md): session history and cleanup notes
