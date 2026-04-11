import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np
from strategies.momentum_strategy import MomentumOrderGenerator


class TestMomentum(unittest.TestCase):

    def _make_data(self, prices, ticker='TEST'):
        dates = pd.date_range('2020-01-01', periods=len(prices), freq='B')
        return pd.DataFrame({ticker: prices}, index=dates)

    def test_no_orders_during_warmup(self):
        """No orders during the lookback warmup period."""
        gen = MomentumOrderGenerator(lookback_days=100)
        prices = np.linspace(100, 120, 50)
        data = self._make_data(prices)
        orders = gen.generate_orders(data)
        self.assertEqual(len(orders), 0)

    def test_buy_near_high(self):
        """BUY when price approaches the rolling high."""
        gen = MomentumOrderGenerator(lookback_days=50, entry_threshold=0.02)
        np.random.seed(42)
        # Gradually rising prices that stay near highs
        prices = np.linspace(100, 150, 80) + np.random.normal(0, 0.5, 80)
        data = self._make_data(prices)
        orders = gen.generate_orders(data)

        buy_orders = [o for o in orders if o['type'] == 'BUY']
        self.assertGreater(len(buy_orders), 0)

    def test_trailing_stop_exit(self):
        """SELL when price drops below trailing stop from peak."""
        gen = MomentumOrderGenerator(lookback_days=50, entry_threshold=0.02,
                                      trailing_stop_pct=0.05, momentum_decay_days=1000)
        # Rise to trigger entry, then sharp drop to trigger trailing stop
        rise = np.linspace(100, 150, 60)
        peak_and_drop = np.array([155, 160, 165, 170, 160, 150, 140])  # 17.6% drop from 170
        prices = np.concatenate([rise, peak_and_drop])
        data = self._make_data(prices)
        orders = gen.generate_orders(data)

        sell_orders = [o for o in orders if o['type'] == 'SELL']
        self.assertGreater(len(sell_orders), 0)

    def test_momentum_decay_exit(self):
        """SELL when no new high is made for momentum_decay_days."""
        gen = MomentumOrderGenerator(lookback_days=50, entry_threshold=0.02,
                                      trailing_stop_pct=0.50, momentum_decay_days=5)
        # Rise to trigger entry, then flat (no new highs for 5+ days)
        rise = np.linspace(100, 150, 60)
        flat = np.full(10, 149.0)  # stays below peak, no new high
        prices = np.concatenate([rise, flat])
        data = self._make_data(prices)
        orders = gen.generate_orders(data)

        sell_orders = [o for o in orders if o['type'] == 'SELL']
        self.assertGreater(len(sell_orders), 0)

    def test_max_positions(self):
        """Should not exceed max_positions concurrent positions."""
        gen = MomentumOrderGenerator(lookback_days=50, entry_threshold=0.05, max_positions=2)
        n = 80
        prices = np.linspace(100, 200, n)
        data = pd.DataFrame(
            {f'T{i}': prices + i * 0.1 for i in range(5)},
            index=pd.date_range('2020-01-01', periods=n, freq='B')
        )
        orders = gen.generate_orders(data)

        positions = set()
        max_open = 0
        for o in orders:
            if o['type'] == 'BUY':
                positions.add(o['ticker'])
            elif o['type'] == 'SELL':
                positions.discard(o['ticker'])
            max_open = max(max_open, len(positions))
        self.assertLessEqual(max_open, 2)


if __name__ == '__main__':
    unittest.main()
