from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any


class BacktestEngine(ABC):
    """Interface for backtesting a trading strategy."""
    
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
    
    @abstractmethod
    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest simulation given orders and historical data."""
        pass


class EquityBacktestEngine(BacktestEngine):
    """Equities (long/short) backtest engine implementation with configurable transaction costs."""

    def __init__(self, initial_cash: float, commission_rate: float = 0.0, flat_fee_per_trade: float = 0.0):
        super().__init__(initial_cash)
        self.commission_rate = commission_rate
        self.flat_fee_per_trade = flat_fee_per_trade

    def _calculate_transaction_cost(self, trade_value: float) -> float:
        return self.flat_fee_per_trade + self.commission_rate * abs(trade_value)

    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        cash = self.initial_cash
        total_transaction_costs = 0.0
        holdings = {}
        portfolio_values = []
        daily_holdings_and_cash_list = []
        all_dates = data.index.sort_values()
        order_index = 0
        num_orders = len(orders)

        for current_date in all_dates:
            # Calculate current portfolio value at the start of day for sizing
            current_holdings_value = 0
            for h_ticker, h_quantity in holdings.items():
                if h_ticker in data.columns:
                    current_holdings_value += h_quantity * data.at[current_date, h_ticker]
            current_portfolio_value = cash + current_holdings_value

            while order_index < num_orders and orders[order_index]["date"] == current_date:
                order = orders[order_index]
                ticker = order["ticker"]
                raw_quantity = order["quantity"]
                price = data.at[current_date, ticker]
                quantity = 0

                if order["type"] == "BUY":
                    if isinstance(raw_quantity, float) and 0 < raw_quantity <= 1.0:
                        current_holding = holdings.get(ticker, 0)
                        if current_holding < 0:
                            quantity = int(abs(current_holding) * raw_quantity)
                        else:
                            target_value = current_portfolio_value * raw_quantity
                            quantity = int(target_value // price)
                    else:
                        quantity = raw_quantity

                    trade_value = price * quantity
                    txn_cost = self._calculate_transaction_cost(trade_value)
                    total_cost = trade_value + txn_cost
                    if cash >= total_cost:
                        cash -= total_cost
                        holdings[ticker] = holdings.get(ticker, 0) + quantity
                        total_transaction_costs += txn_cost

                elif order["type"] == "SELL":
                    if isinstance(raw_quantity, float) and 0 < raw_quantity <= 1.0:
                        current_holding = holdings.get(ticker, 0)
                        if current_holding > 0:
                            quantity = int(current_holding * raw_quantity)
                        else:
                            target_value = current_portfolio_value * raw_quantity
                            quantity = int(target_value // price)
                    else:
                        quantity = raw_quantity

                    trade_value = price * quantity
                    txn_cost = self._calculate_transaction_cost(trade_value)
                    proceeds = trade_value - txn_cost
                    cash += proceeds
                    holdings[ticker] = holdings.get(ticker, 0) - quantity
                    total_transaction_costs += txn_cost

                order_index += 1

            total_value = cash
            current_day_holdings = {"Date": current_date, "Cash": cash}
            for h_ticker, h_quantity in holdings.items():
                price = data.at[current_date, h_ticker]
                total_value += price * h_quantity
                current_day_holdings[h_ticker] = h_quantity

            daily_holdings_and_cash_list.append(current_day_holdings)
            portfolio_values.append((current_date, total_value))

        portfolio_values_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index("Date")
        daily_holdings_and_cash_df = pd.DataFrame(daily_holdings_and_cash_list).set_index("Date").fillna(0)
        return {
            "portfolio_values": portfolio_values_df,
            "daily_holdings_and_cash": daily_holdings_and_cash_df,
            "total_transaction_costs": total_transaction_costs,
        }
