# Project Guide

## Project Structure
```
millennium-data-quality/
├── main.py                          # Entry point - run backtests here
├── backtester/
│   ├── data_source.py               # Fetch market data
│   ├── metrics.py                   # Performance metrics
│   ├── cache_sp500_data.py          # Download & cache data
│   └── backtesters/
│       ├── backtest_engine.py       # Base class
│       ├── equity_backtest_engine.py # Default engine
│       └── template_engine.py       # Duplicate to create new engines
└── strategies/
    ├── order_generator.py           # Base class
    ├── mean_reversion.py            # Default strategy
    └── template_strategy.py         # Duplicate to create new strategies
```

## Creating a New Strategy

1. Copy `strategies/template_strategy.py` to `strategies/your_strategy.py`
2. Implement the `generate_orders()` method
3. In `main.py`, change the import and instantiation to use your new strategy

## Creating a New Backtest Engine

1. Copy `backtester/backtesters/template_engine.py` to `backtester/backtesters/your_engine.py`
2. Implement the `run_backtest()` method
3. In `main.py`, change the import and instantiation to use your new engine

## Running a Backtest

Follow instructions in [README.md](README.md).

Activate environment, download & cache data, and then run main.py.
To switch strategies or engines, update the imports and object instantiations in `main.py`.

## Data Sources & Wharton Integration

The backtesting framework supports multiple data architectures, predominantly `YahooFinanceDataSource` and `WhartonDataSource`. 

### Wharton WRDS Data Configuration

When leveraging the institutional `WhartonDataSource` (via Compustat/CRSP Parquet or Excel extracts), the system must carefully handle how prices are adjusted for corporate actions to maintain data integrity. 

Unlike retail data, Wharton natively supplies raw prices (`prccd`), a cumulative split-adjustment factor (`ajexdi`), and a daily total return factor (`trfd`). To handle this, `WhartonDataSource` accepts a `use_trfd` parameter that alters the architectural mode of the output:

#### 1. `use_trfd=True` (Best for Technical Signals)
Setting this to `True` configures the engine to produce a **Fully-Adjusted Price Series** (mimicking Yahoo Finance's `Adj Close`).
* It anchors the most recent split-adjusted price and scales retroactively using the total return factor (`trfd`).
* **Why use it?** Cash dividends are mathematically smoothed backward into the price action. This guarantees that a stock paying a massive special dividend does not register as a sudden price "crash" on your charts, which would otherwise trigger false technical signals (e.g., crossing moving averages).

#### 2. `use_trfd=False` (Best for Portfolio Accounting)
Setting this to `False` triggers a classic, un-smoothed institutional simulation.
* It evaluates pricing strictly adjusted for share-splits (`prccd / ajexdi`).
* It exposes raw cash distributions natively alongside the price array in a distinct `Dividend` column. 
* **Why use it?** When your backtesting engine explicitly simulates physical cash deposits for dividends. This prevents fractional-penny distortions and ensures historical capital margin requirements are hyper-realistic.