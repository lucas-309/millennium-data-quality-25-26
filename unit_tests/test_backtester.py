import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from backtester.data_source import YahooFinanceDataSource
from strategies.mean_reversion import MeanReversionOrderGenerator
from backtester.backtesters.equity_backtest import EquityBacktestEngine


class TestBacktesterAndOrderGenerator(unittest.TestCase):

    @patch('backtester.data_source.YahooFinanceDataSource.get_historical_data')
    def test_mean_reversion_order_generation(self, mock_get_historical_data):
        num_days = 200
        dates = pd.date_range(start='2023-01-01', periods=num_days)
        np.random.seed(0)
        # Create price series that oscillates around a mean to trigger z-score signals
        prices = 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, num_days)) + np.random.normal(0, 2, num_days)
        mock_data = pd.DataFrame({'AAPL': prices}, index=dates)
        mock_get_historical_data.return_value = mock_data

        data_source = YahooFinanceDataSource()
        data = data_source.get_historical_data(['AAPL'], '2023-01-01', dates[-1].strftime('%Y-%m-%d'))
        order_generator = MeanReversionOrderGenerator(lookback=100)
        orders = order_generator.generate_orders(data)

        # Orders should only start after warmup period
        if len(orders) > 0:
            orders_dates = [order['date'] for order in orders]
            self.assertTrue(all(order_date >= dates[99] for order_date in orders_dates))
        order_types = set(order['type'] for order in orders)
        self.assertTrue(order_types <= {'BUY', 'SELL'})
        self.assertTrue(all(order['ticker'] == 'AAPL' for order in orders))
        required_keys = {'date', 'type', 'ticker', 'quantity'}
        self.assertTrue(all(required_keys.issubset(order.keys()) for order in orders))

    def test_backtest_engine_insufficient_cash(self):
        backtest_engine = EquityBacktestEngine(initial_cash=5000, max_position_pct=10.0, slippage_bps=0.0)
        orders = [{"date": pd.Timestamp('2023-01-01'), "type": "BUY", "ticker": "AAPL", "quantity": 100}]
        data = pd.DataFrame({'AAPL': [150.0]}, index=[pd.Timestamp('2023-01-01')])

        results = backtest_engine.run_backtest(orders, data)
        portfolio_values = results['portfolio_values']
        # Order should be rejected due to insufficient cash (100 shares * $150 = $15,000 > $5,000)
        self.assertAlmostEqual(portfolio_values.iloc[0]['Portfolio Value'], 5000, places=0)
        # Check order log shows rejection
        self.assertEqual(results['order_log'].iloc[0]['status'], 'REJECTED')

    def test_backtest_engine_commission_applied(self):
        engine = EquityBacktestEngine(initial_cash=100000, commission_per_share=0.01,
                                       commission_min=1.0, slippage_bps=0.0)
        orders = [{"date": pd.Timestamp('2023-01-01'), "type": "BUY", "ticker": "AAPL", "quantity": 100}]
        data = pd.DataFrame({'AAPL': [100.0]}, index=[pd.Timestamp('2023-01-01')])

        results = engine.run_backtest(orders, data)
        # Commission: max(1.0, 0.01 * 100) = 1.0
        self.assertAlmostEqual(results['total_commissions'], 1.0)
        self.assertEqual(results['trade_count'], 1)

    @patch('backtester.data_source.YahooFinanceDataSource.get_historical_data')
    def test_order_generator_with_single_data_point(self, mock_get_historical_data):
        dates = [pd.Timestamp('2023-01-01')]
        mock_data = pd.DataFrame({'AAPL': [150.0]}, index=dates)
        mock_get_historical_data.return_value = mock_data

        data_source = YahooFinanceDataSource()
        data = data_source.get_historical_data(['AAPL'], '2023-01-01', '2023-01-01')
        order_generator = MeanReversionOrderGenerator()
        orders = order_generator.generate_orders(data)

        self.assertEqual(len(orders), 0)


if __name__ == '__main__':
    unittest.main()
