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
        daily_holdings_and_cash_list = [] # New list to store daily holdings and cash
        all_dates = data.index.sort_values()
        order_index = 0
        num_orders = len(orders)

        last_month = None
        for current_date in all_dates:
            # Calculate current portfolio value at the start of the day (using today's prices) for sizing
            current_holdings_value = 0
            for h_ticker, h_quantity in holdings.items():
                if h_ticker in data.columns:
                    current_holdings_value += h_quantity * data.at[current_date, h_ticker]
            current_portfolio_value = cash + current_holdings_value

            while order_index < num_orders and orders[order_index]['date'] == current_date:
                order = orders[order_index]
                ticker = order["ticker"]
                raw_quantity = order["quantity"]
                price = data.at[current_date, ticker]
                
                quantity = 0
                
                if order["type"] == "BUY":
                    # Dynamic sizing: 0 < quantity <= 1.0 implies percentage of portfolio value
                    if isinstance(raw_quantity, float) and 0 < raw_quantity <= 1.0:
                        target_value = current_portfolio_value * raw_quantity
                        quantity = int(target_value // price)
                    else:
                        quantity = raw_quantity
                        
                    cost = price * quantity
                    if cash >= cost: # Ensure we have enough cash
                        cash -= cost
                        holdings[ticker] = holdings.get(ticker, 0) + quantity
                    else:
                        # Optional: Buy as much as possible? For now, skip or partial fill could be implemented.
                        # Implementing partial fill to utilize remaining cash if dynamic sizing slightly overshot due to gaps
                        # actually for this simple engine, if fixed size fails, we skip. 
                        # if dynamic sizing, it calculates based on PV, but checking vs cash is safest.
                        pass

                elif order["type"] == "SELL":
                    # Dynamic sizing: 0 < quantity <= 1.0 implies percentage of CURRENT HOLDINGS
                    if isinstance(raw_quantity, float) and 0 < raw_quantity <= 1.0:
                        current_holding = holdings.get(ticker, 0)
                        quantity = int(current_holding * raw_quantity)
                    else:
                        quantity = raw_quantity
                    
                    # Ensure we don't sell more than we have (unless shorting is supported, assuming long-only logic here for safety or capped at 0)
                    available = holdings.get(ticker, 0)
                    quantity = min(quantity, available)
                    
                    proceeds = price * quantity
                    cash += proceeds
                    holdings[ticker] = holdings.get(ticker, 0) - quantity

                order_index += 1

            # Recalculate Total Value after trades
            total_value = cash
            current_day_holdings = {"Date": current_date, "Cash": cash}
            for h_ticker, h_quantity in holdings.items():
                price = data.at[current_date, h_ticker]
                position_value = price * h_quantity
                total_value += position_value
                current_day_holdings[h_ticker] = h_quantity
            
            portfolio_values.append((current_date, total_value))
            # print(f"{current_date}: Portfolio Value - {total_value:.2f}") # Debug print portfolio each day

        portfolio_values_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index("Date")
        daily_holdings_and_cash_df = pd.DataFrame(daily_holdings_and_cash_list).set_index("Date").fillna(0)
        return {"portfolio_values": portfolio_values_df, "daily_holdings_and_cash": daily_holdings_and_cash_df}