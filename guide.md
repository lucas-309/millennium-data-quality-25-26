# Strategy Development Guide

## Project Structure
```
backtester/
├── main.py                 # Entry point - configure and run backtests
├── data_source.py          # Fetches historical price data (Yahoo Finance, cached pickle)
├── order_generator.py      # Strategy implementations (inherit from OrderGenerator)
├── backtest_engine.py      # Executes trades and tracks portfolio value
├── metrics.py              # Calculates performance metrics (Sharpe, drawdown, etc.)
└── cache_sp500_data.py     # Downloads and caches S&P 500 data
```

## Creating a Custom Strategy

### 1. Add Strategy Class to `order_generator.py`

Inherit from `OrderGenerator` and implement `generate_orders()`:
```python
class MyStrategy(OrderGenerator):
    def __init__(self, param1=50):
        self.param1 = param1
    
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []
        # Your logic here
        # Return list of: {"date": timestamp, "type": "BUY"/"SELL", "ticker": str, "quantity": int}
        return orders
```

**Input:** `data` = DataFrame with dates as index, tickers as columns, adjusted close prices as values

**Output:** List of order dicts

### 2. Update `main.py`
```python
from backtester.order_generator import MyStrategy

# Change this line:
order_generator = MyStrategy()
```

### 3. Run
```sh
python backtester/main.py
```

## Key Metrics

- **Sharpe Ratio**: Risk-adjusted return 
- **Max Drawdown**: Worst peak-to-trough loss
- **Cumulative Return**: Total return over backtest period