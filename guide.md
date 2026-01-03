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