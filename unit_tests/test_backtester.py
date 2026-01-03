import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from backtester.data_source import YahooFinanceDataSource
from strategies.order_generator import MeanReversionOrderGenerator
from backtester.backtesters.backtest_engine import EquityBacktestEngine

class TestBacktesterAndOrderGenerator(unittest.TestCase):

    @patch('backtester.data_source.YahooFinanceDataSource.get_historical_data')
    def test_mean_reversion_order_generation(self, mock_get_historical_data):
        # generate 100 random mock points for mean reversion 100 day window strat
        num_days = 150
        dates = pd.date_range(start='2023-01-01', periods=num_days)
        # create a price series that fluctuates around a mean to trigger buy/sell signals
        np.random.seed(0)  #
        prices = 100 + np.cumsum(np.random.normal(0, 1, size=num_days))
        mock_data = pd.DataFrame({
            'AAPL': prices
        }, index=dates)
        mock_get_historical_data.return_value = mock_data

        data_source = YahooFinanceDataSource()
        data = data_source.get_historical_data(['AAPL'], '2023-01-01', dates[-1].strftime('%Y-%m-%d'))
        order_generator = MeanReversionOrderGenerator()
        orders = order_generator.generate_orders(data)

        self.assertGreater(len(orders), 0)
        # check that orders only start after the 100th data point (since earlier ones can't compute the rolling average)
        orders_dates = [order['date'] for order in orders]
        self.assertTrue(all(order_date >= dates[99] for order_date in orders_dates))
        order_types = set(order['type'] for order in orders)
        self.assertTrue(order_types >= {'BUY', 'SELL'})
        self.assertTrue(all(order['ticker'] == 'AAPL' for order in orders))
        required_keys = {'date', 'type', 'ticker', 'quantity'}
        self.assertTrue(all(required_keys.issubset(order.keys()) for order in orders))

    def test_backtest_engine_insufficient_cash(self):
        backtest_engine = EquityBacktestEngine(initial_cash=5000)
        orders = [{"date": '2023-01-01', "type": "BUY", "ticker": "AAPL", "quantity": 100}]
        data = pd.DataFrame({
            'AAPL': [150]
        }, index=[pd.Timestamp('2023-01-01')])

        results = backtest_engine.run_backtest(orders, data)
        portfolio_values = results['portfolio_values']
        self.assertEqual(portfolio_values.iloc[0]['Portfolio Value'], 5000)

    def test_backtest_engine_sell_without_holdings(self):
        backtest_engine = EquityBacktestEngine(initial_cash=10000)
        orders = [{"date": '2023-01-01', "type": "SELL", "ticker": "AAPL", "quantity": 50}]
        data = pd.DataFrame({
            'AAPL': [150]
        }, index=[pd.Timestamp('2023-01-01')])

        results = backtest_engine.run_backtest(orders, data)
        portfolio_values = results['portfolio_values']
        self.assertEqual(portfolio_values.iloc[0]['Portfolio Value'], 10000)

    @patch('backtester.data_source.YahooFinanceDataSource.get_historical_data')
    def test_order_generator_with_single_data_point(self, mock_get_historical_data):
        dates = [pd.Timestamp('2023-01-01')]
        mock_data = pd.DataFrame({
            'AAPL': [150]
        }, index=dates)
        mock_get_historical_data.return_value = mock_data

        data_source = YahooFinanceDataSource()
        data = data_source.get_historical_data(['AAPL'], '2023-01-01', '2023-01-01')
        order_generator = MeanReversionOrderGenerator()
        orders = order_generator.generate_orders(data)

        self.assertEqual(len(orders), 0)


if __name__ == '__main__':
    unittest.main()