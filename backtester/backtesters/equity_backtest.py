import pandas as pd
import numpy as np
from typing import List, Dict, Any

from .backtest_engine import BacktestEngine


class EquityBacktestEngine(BacktestEngine):
    """Equities (long/short) backtest engine with transaction costs, slippage, and risk controls."""

    def __init__(
        self,
        initial_cash: float,
        commission_per_share: float = 0.005,
        commission_min: float = 1.0,
        slippage_bps: float = 5.0,
        margin_requirement: float = 1.5,
        max_position_pct: float = 0.25,
    ):
        super().__init__(initial_cash)
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min
        self.slippage_bps = slippage_bps
        self.margin_requirement = margin_requirement
        self.max_position_pct = max_position_pct

    def _calc_commission(self, quantity: int) -> float:
        return max(self.commission_min, self.commission_per_share * abs(quantity))

    def _calc_portfolio_value(self, cash: float, holdings: dict, data: pd.DataFrame, date) -> float:
        value = cash
        for ticker, qty in holdings.items():
            if ticker in data.columns:
                value += qty * data.at[date, ticker]
        return value

    def _calc_margin_required(self, holdings: dict, data: pd.DataFrame, date) -> float:
        """Total margin required for all short positions."""
        margin = 0.0
        for ticker, qty in holdings.items():
            if qty < 0 and ticker in data.columns:
                margin += abs(qty) * data.at[date, ticker] * self.margin_requirement
        return margin

    def _normalize_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for sequence, order in enumerate(orders):
            normalized_order = dict(order)
            normalized_order["date"] = pd.Timestamp(order["date"])
            normalized_order["_sequence"] = sequence
            normalized.append(normalized_order)
        normalized.sort(key=lambda item: (item["date"], item["_sequence"]))
        return normalized

    def _resolve_quantity(
        self,
        order_type: str,
        raw_quantity: Any,
        current_holding: int,
        current_portfolio_value: float,
        exec_price: float,
    ) -> int:
        if not isinstance(raw_quantity, float) or not (0 < raw_quantity <= 1.0):
            return int(raw_quantity)

        if order_type == "BUY":
            if current_holding < 0:
                return int(np.ceil(abs(current_holding) * raw_quantity))
            target_value = current_portfolio_value * raw_quantity
            return int(target_value // exec_price)

        if current_holding > 0:
            return int(np.ceil(current_holding * raw_quantity))

        target_value = current_portfolio_value * raw_quantity
        return int(target_value // exec_price)

    def _cap_position_size(
        self,
        current_holding: int,
        requested_quantity: int,
        exec_price: float,
        current_portfolio_value: float,
        direction: int,
    ) -> int:
        if requested_quantity <= 0 or current_portfolio_value <= 0:
            return 0

        new_position = current_holding + direction * requested_quantity
        new_position_value = abs(new_position) * exec_price
        max_allowed_value = current_portfolio_value * self.max_position_pct
        if max_allowed_value <= 0 or new_position_value <= max_allowed_value:
            return requested_quantity

        allowed_position = int(max_allowed_value // exec_price)
        if direction > 0:
            allowed_quantity = allowed_position - current_holding
        else:
            allowed_quantity = current_holding + allowed_position

        return max(0, min(requested_quantity, allowed_quantity))

    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        orders = self._normalize_orders(orders)
        cash = self.initial_cash
        holdings = {}
        portfolio_values = []
        daily_holdings_and_cash_list = []
        order_log = []
        total_commissions = 0.0
        total_slippage_cost = 0.0
        trade_count = 0

        all_dates = data.index.sort_values()
        order_index = 0
        num_orders = len(orders)

        for current_date in all_dates:
            current_portfolio_value = self._calc_portfolio_value(cash, holdings, data, current_date)

            while order_index < num_orders and orders[order_index]['date'] == current_date:
                order = orders[order_index]
                ticker = order["ticker"]
                raw_quantity = order["quantity"]
                mid_price = data.at[current_date, ticker]
                current_holding = int(holdings.get(ticker, 0))

                if order["type"] == "BUY":
                    # Slippage: pay slightly more
                    exec_price = mid_price * (1 + self.slippage_bps / 10000)
                    quantity = self._resolve_quantity(
                        order_type="BUY",
                        raw_quantity=raw_quantity,
                        current_holding=current_holding,
                        current_portfolio_value=current_portfolio_value,
                        exec_price=exec_price,
                    )

                    if quantity <= 0:
                        order_log.append({"date": current_date, "ticker": ticker, "type": "BUY",
                                          "quantity": 0, "price": exec_price, "status": "REJECTED",
                                          "reason": "zero_quantity"})
                        order_index += 1
                        continue

                    # Position concentration check
                    quantity = self._cap_position_size(
                        current_holding=current_holding,
                        requested_quantity=quantity,
                        exec_price=exec_price,
                        current_portfolio_value=current_portfolio_value,
                        direction=1,
                    )
                    if quantity <= 0:
                        order_log.append({"date": current_date, "ticker": ticker, "type": "BUY",
                                          "quantity": 0, "price": exec_price, "status": "REJECTED",
                                          "reason": "position_concentration_limit"})
                        order_index += 1
                        continue

                    commission = self._calc_commission(quantity)
                    total_cost = exec_price * quantity + commission

                    if cash >= total_cost:
                        slippage_cost = (exec_price - mid_price) * quantity
                        cash -= total_cost
                        holdings[ticker] = holdings.get(ticker, 0) + quantity
                        total_commissions += commission
                        total_slippage_cost += slippage_cost
                        trade_count += 1
                        order_log.append({"date": current_date, "ticker": ticker, "type": "BUY",
                                          "quantity": quantity, "price": exec_price, "commission": commission,
                                          "slippage": slippage_cost, "status": "EXECUTED", "reason": ""})
                    else:
                        order_log.append({"date": current_date, "ticker": ticker, "type": "BUY",
                                          "quantity": quantity, "price": exec_price, "status": "REJECTED",
                                          "reason": "insufficient_cash"})

                elif order["type"] == "SELL":
                    # Slippage: receive slightly less
                    exec_price = mid_price * (1 - self.slippage_bps / 10000)
                    quantity = self._resolve_quantity(
                        order_type="SELL",
                        raw_quantity=raw_quantity,
                        current_holding=current_holding,
                        current_portfolio_value=current_portfolio_value,
                        exec_price=exec_price,
                    )

                    if quantity <= 0:
                        order_log.append({"date": current_date, "ticker": ticker, "type": "SELL",
                                          "quantity": 0, "price": exec_price, "status": "REJECTED",
                                          "reason": "zero_quantity"})
                        order_index += 1
                        continue

                    quantity = self._cap_position_size(
                        current_holding=current_holding,
                        requested_quantity=quantity,
                        exec_price=exec_price,
                        current_portfolio_value=current_portfolio_value,
                        direction=-1,
                    )
                    if quantity <= 0:
                        order_log.append({"date": current_date, "ticker": ticker, "type": "SELL",
                                          "quantity": 0, "price": exec_price, "status": "REJECTED",
                                          "reason": "position_concentration_limit"})
                        order_index += 1
                        continue

                    # Check if this would create/increase a short position
                    new_position = current_holding - quantity
                    if new_position < 0:
                        # Margin check for short selling
                        margin_needed = abs(new_position) * mid_price * self.margin_requirement
                        existing_margin = self._calc_margin_required(holdings, data, current_date)
                        # Subtract margin for current position in this ticker if already short
                        if current_holding < 0:
                            existing_margin -= abs(current_holding) * mid_price * self.margin_requirement
                        available_for_margin = cash - existing_margin
                        if available_for_margin < margin_needed:
                            order_log.append({"date": current_date, "ticker": ticker, "type": "SELL",
                                              "quantity": quantity, "price": exec_price, "status": "REJECTED",
                                              "reason": "insufficient_margin"})
                            order_index += 1
                            continue

                    commission = self._calc_commission(quantity)
                    proceeds = exec_price * quantity - commission
                    slippage_cost = (mid_price - exec_price) * quantity

                    cash += proceeds
                    holdings[ticker] = current_holding - quantity
                    total_commissions += commission
                    total_slippage_cost += slippage_cost
                    trade_count += 1
                    order_log.append({"date": current_date, "ticker": ticker, "type": "SELL",
                                      "quantity": quantity, "price": exec_price, "commission": commission,
                                      "slippage": slippage_cost, "status": "EXECUTED", "reason": ""})

                order_index += 1

            # End-of-day portfolio value
            total_value = self._calc_portfolio_value(cash, holdings, data, current_date)

            current_day_holdings = {"Date": current_date, "Cash": cash}
            for h_ticker, h_quantity in holdings.items():
                current_day_holdings[h_ticker] = h_quantity

            daily_holdings_and_cash_list.append(current_day_holdings)
            portfolio_values.append((current_date, total_value))

        portfolio_values_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index("Date")
        daily_holdings_and_cash_df = pd.DataFrame(daily_holdings_and_cash_list).set_index("Date").fillna(0)
        order_log_df = pd.DataFrame(order_log) if order_log else pd.DataFrame(
            columns=["date", "ticker", "type", "quantity", "price", "commission", "slippage", "status", "reason"])

        return {
            "portfolio_values": portfolio_values_df,
            "daily_holdings_and_cash": daily_holdings_and_cash_df,
            "order_log": order_log_df,
            "trade_count": trade_count,
            "total_commissions": total_commissions,
            "total_slippage_cost": total_slippage_cost,
        }
