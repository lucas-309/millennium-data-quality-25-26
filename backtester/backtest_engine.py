from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class BacktestEngine(ABC):
    """Interface for backtesting a trading strategy."""
    
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
    
    @abstractmethod
    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest simulation given orders and historical data."""
        pass

# TODO: clean up refactor implementations into sep. files, e.g. equity_backtest_engine.py
class EquityBacktestEngine(BacktestEngine):
    """Equities (long/short) backtest engine implementation without slippage or transaction costs."""

    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        cash = self.initial_cash
        holdings = {}
        portfolio_values = []
        all_dates = data.index.sort_values()
        order_index = 0
        num_orders = len(orders)

        for current_date in all_dates:
            while order_index < num_orders and orders[order_index]['date'] == current_date:
                order = orders[order_index]
                ticker = order["ticker"]
                quantity = order["quantity"]
                price = data.at[current_date, ticker]

                if order["type"] == "BUY":
                    cost = price * quantity
                    cash -= cost
                    holdings[ticker] = holdings.get(ticker, 0) + quantity
                elif order["type"] == "SELL":
                    proceeds = price * quantity
                    cash += proceeds
                    holdings[ticker] = holdings.get(ticker, 0) - quantity

                order_index += 1

            total_value = cash
            for h_ticker, h_quantity in holdings.items():
                price = data.at[current_date, h_ticker]
                position_value = price * h_quantity
                total_value += position_value

            portfolio_values.append((current_date, total_value))
            print(f"{current_date}: Portfolio Value - {total_value:.2f}")

        portfolio_values_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index("Date")
        return {"portfolio_values": portfolio_values_df}