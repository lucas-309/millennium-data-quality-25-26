import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np
from backtester.backtesters.equity_backtest import EquityBacktestEngine


class TestEquityBacktestEngine(unittest.TestCase):

    def _make_data(self, prices_dict, n=5):
        dates = pd.date_range('2023-01-01', periods=n, freq='B')
        return pd.DataFrame(prices_dict, index=dates)

    def test_buy_deducts_commission(self):
        """Commissions should be deducted from cash on BUY."""
        engine = EquityBacktestEngine(initial_cash=100000, commission_per_share=0.01,
                                       commission_min=1.0, slippage_bps=0.0)
        data = self._make_data({'AAPL': [100.0] * 5})
        orders = [{"date": data.index[0], "type": "BUY", "ticker": "AAPL", "quantity": 100}]
        results = engine.run_backtest(orders, data)

        # Cost: 100 * 100 + max(1.0, 0.01*100) = 10000 + 1.0 = 10001
        expected_cash = 100000 - 10001
        pv = results['portfolio_values'].iloc[0]['Portfolio Value']
        # Portfolio = cash + holdings = (100000 - 10001) + 100 * 100 = 99999
        self.assertAlmostEqual(pv, 99999.0, places=0)
        self.assertAlmostEqual(results['total_commissions'], 1.0)

    def test_slippage_on_buy(self):
        """BUY should execute at price * (1 + slippage_bps/10000)."""
        engine = EquityBacktestEngine(initial_cash=100000, commission_per_share=0.0,
                                       commission_min=0.0, slippage_bps=100.0)  # 1% slippage
        data = self._make_data({'AAPL': [100.0] * 5})
        orders = [{"date": data.index[0], "type": "BUY", "ticker": "AAPL", "quantity": 10}]
        results = engine.run_backtest(orders, data)

        # Exec price = 100 * 1.01 = 101. Cost = 10 * 101 = 1010
        # Cash = 100000 - 1010 = 98990
        # Holdings value at mid: 10 * 100 = 1000
        # Portfolio = 98990 + 1000 = 99990
        self.assertAlmostEqual(results['total_slippage_cost'], 10.0, places=1)

    def test_slippage_on_sell(self):
        """SELL should execute at price * (1 - slippage_bps/10000)."""
        engine = EquityBacktestEngine(initial_cash=100000, commission_per_share=0.0,
                                       commission_min=0.0, slippage_bps=100.0)  # 1% slippage
        data = self._make_data({'AAPL': [100.0] * 5})
        orders = [
            {"date": data.index[0], "type": "BUY", "ticker": "AAPL", "quantity": 10},
            {"date": data.index[1], "type": "SELL", "ticker": "AAPL", "quantity": 10},
        ]
        results = engine.run_backtest(orders, data)
        # Should have slippage on both buy and sell
        self.assertGreater(results['total_slippage_cost'], 0)
        self.assertEqual(results['trade_count'], 2)

    def test_insufficient_cash_rejection(self):
        """BUY should be rejected when cash is insufficient."""
        # max_position_pct=1.0 so position-concentration cap doesn't trigger first
        engine = EquityBacktestEngine(initial_cash=500, slippage_bps=0.0, max_position_pct=1.0)
        data = self._make_data({'AAPL': [100.0] * 5})
        orders = [{"date": data.index[0], "type": "BUY", "ticker": "AAPL", "quantity": 10}]
        results = engine.run_backtest(orders, data)

        self.assertEqual(results['trade_count'], 0)
        self.assertEqual(results['order_log'].iloc[0]['status'], 'REJECTED')
        self.assertEqual(results['order_log'].iloc[0]['reason'], 'insufficient_cash')

    def test_position_concentration_limit(self):
        """Should reject BUY that would exceed max_position_pct."""
        engine = EquityBacktestEngine(initial_cash=100000, max_position_pct=0.10,
                                       slippage_bps=0.0, commission_min=0.0, commission_per_share=0.0)
        data = self._make_data({'AAPL': [100.0] * 5})
        # Try to buy 200 shares = $20,000 = 20% of portfolio (exceeds 10% limit)
        orders = [{"date": data.index[0], "type": "BUY", "ticker": "AAPL", "quantity": 200}]
        results = engine.run_backtest(orders, data)

        if results['trade_count'] > 0:
            # Should have capped the position
            executed = results['order_log'][results['order_log']['status'] == 'EXECUTED']
            if len(executed) > 0:
                qty = executed.iloc[0]['quantity']
                # Max: 10% of 100000 = 10000 / 100 = 100 shares
                self.assertLessEqual(qty, 100)

    def test_dynamic_buy_sizing(self):
        """Float quantity between 0-1 should be treated as % of portfolio."""
        engine = EquityBacktestEngine(initial_cash=100000, slippage_bps=0.0,
                                       commission_min=0.0, commission_per_share=0.0,
                                       max_position_pct=1.0)
        data = self._make_data({'AAPL': [50.0] * 5})
        # 0.20 = 20% of portfolio = $20,000 / $50 = 400 shares
        orders = [{"date": data.index[0], "type": "BUY", "ticker": "AAPL", "quantity": 0.20}]
        results = engine.run_backtest(orders, data)

        self.assertEqual(results['trade_count'], 1)
        executed = results['order_log'][results['order_log']['status'] == 'EXECUTED']
        self.assertEqual(executed.iloc[0]['quantity'], 400)

    def test_dynamic_sell_sizing(self):
        """Float SELL quantity should be % of current holdings."""
        engine = EquityBacktestEngine(initial_cash=100000, slippage_bps=0.0,
                                       commission_min=0.0, commission_per_share=0.0,
                                       max_position_pct=1.0)
        data = self._make_data({'AAPL': [50.0] * 5})
        orders = [
            {"date": data.index[0], "type": "BUY", "ticker": "AAPL", "quantity": 100},
            {"date": data.index[1], "type": "SELL", "ticker": "AAPL", "quantity": 0.50},  # sell 50%
        ]
        results = engine.run_backtest(orders, data)

        self.assertEqual(results['trade_count'], 2)
        # After selling 50% of 100 shares = 50 shares remaining
        last_holdings = results['daily_holdings_and_cash'].iloc[-1]
        self.assertEqual(last_holdings['AAPL'], 50)

    def test_dynamic_short_open_and_cover(self):
        """Float SELL should open a short and float BUY should cover it when holdings are negative."""
        engine = EquityBacktestEngine(initial_cash=100000, slippage_bps=0.0,
                                      commission_min=0.0, commission_per_share=0.0,
                                      max_position_pct=1.0)
        data = self._make_data({'AAPL': [100.0] * 5})
        orders = [
            {"date": data.index[0], "type": "SELL", "ticker": "AAPL", "quantity": 0.10},
            {"date": data.index[1], "type": "BUY", "ticker": "AAPL", "quantity": 1.0},
        ]
        results = engine.run_backtest(orders, data)

        executed = results['order_log'][results['order_log']['status'] == 'EXECUTED']
        self.assertEqual(len(executed), 2)
        self.assertEqual(executed.iloc[0]['quantity'], 100)
        self.assertEqual(executed.iloc[1]['quantity'], 100)
        self.assertEqual(results['daily_holdings_and_cash'].iloc[-1]['AAPL'], 0)

    def test_order_log_structure(self):
        """Order log should have all required columns."""
        engine = EquityBacktestEngine(initial_cash=100000)
        data = self._make_data({'AAPL': [100.0] * 5})
        orders = [{"date": data.index[0], "type": "BUY", "ticker": "AAPL", "quantity": 10}]
        results = engine.run_backtest(orders, data)

        log = results['order_log']
        required_cols = {'date', 'ticker', 'type', 'quantity', 'status'}
        self.assertTrue(required_cols.issubset(set(log.columns)))

    def test_empty_orders(self):
        """Engine should handle empty order list gracefully."""
        engine = EquityBacktestEngine(initial_cash=100000)
        data = self._make_data({'AAPL': [100.0] * 5})
        results = engine.run_backtest([], data)

        self.assertEqual(results['trade_count'], 0)
        self.assertAlmostEqual(
            results['portfolio_values'].iloc[-1]['Portfolio Value'], 100000
        )


if __name__ == '__main__':
    unittest.main()
