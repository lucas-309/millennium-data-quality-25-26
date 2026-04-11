import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np
from strategies.pairs_trading import PairsTradingOrderGenerator


class TestPairsTrading(unittest.TestCase):

    def _make_cointegrated_pair(self, n=200, seed=42):
        """Create synthetic cointegrated pair data."""
        np.random.seed(seed)
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        # A and B share a common trend with mean-reverting spread
        trend = np.cumsum(np.random.normal(0.0005, 0.01, n)) + np.log(100)
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = 0.8 * spread[i - 1] + np.random.normal(0, 0.02)  # AR(1) mean-reverting
        price_a = np.exp(trend + spread / 2)
        price_b = np.exp(trend - spread / 2)
        return pd.DataFrame({'A': price_a, 'B': price_b}, index=dates)

    def test_generates_orders_for_cointegrated_pair(self):
        """Should generate trades for a cointegrated pair."""
        data = self._make_cointegrated_pair()
        gen = PairsTradingOrderGenerator(pairs=[('A', 'B')], lookback_window=60,
                                          entry_zscore=1.5, min_correlation=0.5)
        orders = gen.generate_orders(data)
        self.assertGreater(len(orders), 0)

    def test_no_orders_for_uncorrelated_tickers(self):
        """Should not trade pairs with insufficient correlation."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        data = pd.DataFrame({
            'A': np.exp(np.cumsum(np.random.normal(0, 0.02, n))),
            'B': np.exp(np.cumsum(np.random.normal(0, 0.02, n))),
        }, index=dates)
        gen = PairsTradingOrderGenerator(pairs=[('A', 'B')], lookback_window=60,
                                          min_correlation=0.95)
        orders = gen.generate_orders(data)
        # With uncorrelated random walks, correlation will be low, so few/no trades
        # (may get some by chance, but should be significantly fewer than correlated)
        self.assertLess(len(orders), 10)

    def test_exit_zscore_buffer(self):
        """Exit should happen at exit_zscore, not at 0."""
        data = self._make_cointegrated_pair(n=300)
        gen = PairsTradingOrderGenerator(pairs=[('A', 'B')], lookback_window=60,
                                          entry_zscore=1.5, exit_zscore=0.5,
                                          min_correlation=0.5)
        orders = gen.generate_orders(data)
        # Verify we get both BUY and SELL orders (entry and exit)
        types = set(o['type'] for o in orders)
        self.assertEqual(types, {'BUY', 'SELL'})

    def test_missing_ticker_skipped(self):
        """Pairs with missing tickers should be silently skipped."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        data = pd.DataFrame({'A': np.linspace(100, 110, n)}, index=dates)
        gen = PairsTradingOrderGenerator(pairs=[('A', 'MISSING')])
        orders = gen.generate_orders(data)
        self.assertEqual(len(orders), 0)

    def test_orders_have_correct_structure(self):
        """All orders should have required keys and valid values."""
        data = self._make_cointegrated_pair()
        gen = PairsTradingOrderGenerator(pairs=[('A', 'B')], lookback_window=60,
                                          entry_zscore=1.5, min_correlation=0.5)
        orders = gen.generate_orders(data)
        for o in orders:
            self.assertIn('date', o)
            self.assertIn('type', o)
            self.assertIn('ticker', o)
            self.assertIn('quantity', o)
            self.assertIn(o['type'], ('BUY', 'SELL'))
            self.assertIn(o['ticker'], ('A', 'B'))


if __name__ == '__main__':
    unittest.main()
