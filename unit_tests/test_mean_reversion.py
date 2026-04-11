import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np
from strategies.mean_reversion import MeanReversionOrderGenerator


class TestMeanReversion(unittest.TestCase):

    def _make_data(self, prices, ticker='TEST'):
        dates = pd.date_range('2020-01-01', periods=len(prices), freq='B')
        return pd.DataFrame({ticker: prices}, index=dates)

    def test_no_orders_during_warmup(self):
        """No orders should be generated during the lookback warmup period."""
        gen = MeanReversionOrderGenerator(lookback=50)
        prices = np.linspace(100, 110, 40)  # fewer data points than lookback
        data = self._make_data(prices)
        orders = gen.generate_orders(data)
        self.assertEqual(len(orders), 0)

    def test_buy_signal_on_negative_zscore(self):
        """BUY when price drops well below the rolling mean."""
        gen = MeanReversionOrderGenerator(lookback=50, entry_zscore=-1.0)
        np.random.seed(42)
        # Stable prices then a sharp drop
        stable = np.full(60, 100.0) + np.random.normal(0, 0.5, 60)
        drop = np.full(10, 85.0)  # way below mean
        prices = np.concatenate([stable, drop])
        data = self._make_data(prices)
        orders = gen.generate_orders(data)

        buy_orders = [o for o in orders if o['type'] == 'BUY']
        self.assertGreater(len(buy_orders), 0)

    def test_sell_on_reversion(self):
        """SELL when price reverts to the mean after buying."""
        gen = MeanReversionOrderGenerator(lookback=50, entry_zscore=-1.5, exit_zscore=0.0)
        np.random.seed(42)
        stable = np.full(60, 100.0) + np.random.normal(0, 0.5, 60)
        drop = np.full(10, 85.0)
        recovery = np.full(10, 100.0)
        prices = np.concatenate([stable, drop, recovery])
        data = self._make_data(prices)
        orders = gen.generate_orders(data)

        sell_orders = [o for o in orders if o['type'] == 'SELL']
        self.assertGreater(len(sell_orders), 0)

    def test_stop_loss_fires(self):
        """SELL (stop-loss) when z-score drops below stop-loss threshold."""
        gen = MeanReversionOrderGenerator(lookback=50, entry_zscore=-1.0, stop_loss_zscore=-2.5)
        np.random.seed(42)
        stable = np.full(60, 100.0) + np.random.normal(0, 0.5, 60)
        mild_drop = np.full(5, 92.0)  # triggers entry
        crash = np.full(5, 70.0)  # triggers stop-loss
        prices = np.concatenate([stable, mild_drop, crash])
        data = self._make_data(prices)
        orders = gen.generate_orders(data)

        sell_orders = [o for o in orders if o['type'] == 'SELL']
        self.assertGreater(len(sell_orders), 0)

    def test_max_positions_respected(self):
        """Should not exceed max_positions concurrent positions."""
        gen = MeanReversionOrderGenerator(lookback=50, entry_zscore=-0.5, max_positions=2)
        np.random.seed(42)
        # Create 5 tickers all dropping at the same time
        n = 80
        stable = np.full(60, 100.0) + np.random.normal(0, 0.5, 60)
        drop = np.full(20, 85.0)
        prices = np.concatenate([stable, drop])
        data = pd.DataFrame(
            {f'T{i}': prices + np.random.normal(0, 0.1, n) for i in range(5)},
            index=pd.date_range('2020-01-01', periods=n, freq='B')
        )
        orders = gen.generate_orders(data)

        # Count max concurrent BUY positions
        positions = set()
        max_open = 0
        for o in orders:
            if o['type'] == 'BUY':
                positions.add(o['ticker'])
            elif o['type'] == 'SELL':
                positions.discard(o['ticker'])
            max_open = max(max_open, len(positions))
        self.assertLessEqual(max_open, 2)

    def test_uses_percentage_quantities(self):
        """All orders should use float quantities between 0 and 1."""
        gen = MeanReversionOrderGenerator(lookback=50, entry_zscore=-1.0)
        np.random.seed(42)
        stable = np.full(60, 100.0) + np.random.normal(0, 0.5, 60)
        drop = np.full(10, 85.0)
        prices = np.concatenate([stable, drop])
        data = self._make_data(prices)
        orders = gen.generate_orders(data)

        for o in orders:
            q = o['quantity']
            self.assertIsInstance(q, float)
            self.assertGreater(q, 0)
            self.assertLessEqual(q, 1.0)


if __name__ == '__main__':
    unittest.main()
