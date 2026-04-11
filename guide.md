# Project Guide

## Project Structure
```
millennium-data-quality/
├── main.py                          # Entry point - run backtests here
├── research_main.py                 # Research-grade signal suite entry point
├── backtester/
│   ├── data_source.py               # Fetch market data (Yahoo / Pickle / Wharton)
│   ├── metrics.py                   # Performance metrics
│   ├── cache_sp500_data.py          # Download & cache data
│   ├── research_data.py             # Research dataset loader (Wharton)
│   ├── research_backtester.py       # Weight-based research backtester
│   └── backtesters/
│       ├── backtest_engine.py       # Base class
│       ├── equity_backtest.py       # Default order-based engine
│       └── template_engine.py       # Duplicate to create new engines
└── strategies/
    ├── order_generator.py           # Base class for order-based strategies
    ├── mean_reversion.py            # Z-score mean reversion
    ├── momentum_strategy.py         # 52-week breakout momentum
    ├── pairs_trading.py             # Statistical arbitrage pairs
    ├── betting_against_beta.py      # BAB (long low-beta, short high-beta)
    ├── dispersion.py                # Earnings dispersion (needs external data)
    ├── research_strategies.py       # Signal-score strategies (research framework)
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

## Backtest Engine Parameters

`EquityBacktestEngine` supports realistic trading frictions and risk controls:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_cash` | required | Starting portfolio cash |
| `commission_per_share` | 0.005 | Per-share commission ($/share, Interactive-Brokers-like) |
| `commission_min` | 1.0 | Minimum commission per order |
| `slippage_bps` | 5.0 | Execution slippage in basis points (BUY pays more, SELL receives less) |
| `margin_requirement` | 1.5 | Maintenance margin multiplier for short positions |
| `max_position_pct` | 0.25 | Maximum fraction of portfolio value any single position can occupy |

The engine returns:
- `portfolio_values`: Daily portfolio value DataFrame
- `daily_holdings_and_cash`: Daily position snapshot
- `order_log`: Every attempted order with status (EXECUTED / REJECTED) and reason
- `trade_count`, `total_commissions`, `total_slippage_cost`

## Order Quantity Convention

- **Integer quantity** (e.g., `100`): absolute share count
- **Float between 0 and 1** (e.g., `0.15`): percentage-based sizing
  - For BUY: percentage of current portfolio value
  - For SELL: percentage of current holdings in that ticker
  - For opening short positions with no current holdings, percentage of portfolio value

## Strategies

| Strategy | File | Key Parameters |
|----------|------|----------------|
| Mean Reversion | `strategies/mean_reversion.py` | `lookback`, `entry_zscore`, `stop_loss_zscore`, `max_positions` |
| Momentum | `strategies/momentum_strategy.py` | `lookback_days`, `trailing_stop_pct`, `momentum_decay_days` |
| Pairs Trading | `strategies/pairs_trading.py` | `pairs`, `entry_zscore`, `exit_zscore`, `stop_loss_zscore`, `min_correlation` |
| Betting Against Beta | `strategies/betting_against_beta.py` | `lookback_period`, `rebalance_frequency`, `decile_fraction` |
| Dispersion | `strategies/dispersion.py` | Requires external analyst dispersion data |

## Metrics

`ExtendedMetrics.calculate()` reports:
- Returns: Avg Daily Return, Cumulative Return, Annualized Return, Log Return
- Risk: Volatility, Max Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Benchmark-relative: Beta, Alpha (Jensen's), Information Ratio
- Trade quality: Win Rate, Profit Factor
- Activity: Avg Daily Turnover, Annualized Turnover

Configure the annual risk-free rate with `ExtendedMetrics(risk_free_rate=0.05)`.