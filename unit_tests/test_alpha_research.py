import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np

from backtester.alpha_research import (
    information_coefficient,
    ic_summary,
    ic_decay,
    signal_half_life,
    quantile_returns,
    quantile_spread,
    quantile_monotonicity,
    signal_turnover,
    orthogonalize,
    ic_weighted_combination,
)


class TestInformationCoefficient(unittest.TestCase):

    def _make_frames(self, n_dates=100, n_tickers=20, seed=42, noise=0.0):
        np.random.seed(seed)
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]

        # Perfect signal with optional noise
        scores = pd.DataFrame(
            np.random.randn(n_dates, n_tickers),
            index=dates, columns=tickers,
        )
        # Forward returns proportional to signal + noise
        returns = scores * 0.01 + np.random.randn(n_dates, n_tickers) * noise
        return scores, returns

    def test_perfect_signal_has_ic_near_one(self):
        scores, returns = self._make_frames(noise=0.0)
        ic = information_coefficient(scores, returns)
        mean_ic = ic.mean()
        self.assertGreater(mean_ic, 0.95)

    def test_noisy_signal_has_lower_ic(self):
        scores, returns = self._make_frames(noise=0.02)
        ic = information_coefficient(scores, returns)
        # Still positive but less than perfect
        self.assertLess(ic.mean(), 0.95)
        self.assertGreater(ic.mean(), 0.2)

    def test_spearman_vs_pearson(self):
        scores, returns = self._make_frames(noise=0.01)
        ic_p = information_coefficient(scores, returns, method="pearson")
        ic_s = information_coefficient(scores, returns, method="spearman")
        # Both should be similar and positive
        self.assertGreater(ic_p.mean(), 0.3)
        self.assertGreater(ic_s.mean(), 0.3)

    def test_ic_summary_stats(self):
        scores, returns = self._make_frames(noise=0.01)
        ic = information_coefficient(scores, returns)
        stats = ic_summary(ic)
        self.assertIn("mean_ic", stats)
        self.assertIn("ic_ir", stats)
        self.assertIn("hit_rate", stats)
        self.assertGreater(stats["hit_rate"], 0.5)


class TestICDecay(unittest.TestCase):

    def test_ic_decay_horizons(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        tickers = [f"T{i}" for i in range(15)]
        scores = pd.DataFrame(
            np.random.randn(300, 15), index=dates, columns=tickers,
        )
        returns = scores * 0.005 + np.random.randn(300, 15) * 0.01

        decay = ic_decay(scores, returns, horizons=(1, 5, 21))
        self.assertEqual(len(decay), 3)
        self.assertIn("mean_ic", decay.columns)


class TestQuantileReturns(unittest.TestCase):

    def test_monotonic_signal_produces_monotonic_quantiles(self):
        np.random.seed(42)
        n_dates, n_tickers = 100, 25
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]

        scores = pd.DataFrame(
            np.random.randn(n_dates, n_tickers), index=dates, columns=tickers,
        )
        returns = scores * 0.01  # perfect linear relationship

        qr = quantile_returns(scores, returns, n_quantiles=5)
        mono = quantile_monotonicity(qr)
        # Should be ~1.0 (perfect rank correlation)
        self.assertGreater(mono, 0.9)

    def test_spread_positive_for_good_signal(self):
        np.random.seed(42)
        n_dates, n_tickers = 100, 25
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]

        scores = pd.DataFrame(
            np.random.randn(n_dates, n_tickers), index=dates, columns=tickers,
        )
        returns = scores * 0.01

        qr = quantile_returns(scores, returns, n_quantiles=5)
        spread = quantile_spread(qr)
        self.assertGreater(spread.mean(), 0)


class TestSignalTurnover(unittest.TestCase):

    def test_stable_signal_has_low_turnover(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        tickers = [f"T{i}" for i in range(10)]
        # Nearly constant signal
        base = np.random.randn(10)
        scores = pd.DataFrame(
            np.tile(base, (50, 1)) + np.random.randn(50, 10) * 0.01,
            index=dates, columns=tickers,
        )
        to = signal_turnover(scores, top_quantile=0.3)
        self.assertLess(to.mean(), 0.3)

    def test_random_signal_has_high_turnover(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        tickers = [f"T{i}" for i in range(10)]
        scores = pd.DataFrame(
            np.random.randn(50, 10), index=dates, columns=tickers,
        )
        to = signal_turnover(scores, top_quantile=0.3)
        self.assertGreater(to.mean(), 0.3)


class TestOrthogonalize(unittest.TestCase):

    def test_orthogonal_to_itself_is_zero(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        tickers = [f"T{i}" for i in range(15)]
        s = pd.DataFrame(
            np.random.randn(30, 15), index=dates, columns=tickers,
        )
        residual = orthogonalize(s, [s])
        # Residual after regressing on self should be ~0
        self.assertLess(residual.abs().mean().mean(), 1e-6)

    def test_orthogonalize_against_unrelated_signal(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        tickers = [f"T{i}" for i in range(15)]
        target = pd.DataFrame(
            np.random.randn(30, 15), index=dates, columns=tickers,
        )
        other = pd.DataFrame(
            np.random.randn(30, 15), index=dates, columns=tickers,
        )
        residual = orthogonalize(target, [other])
        # Residual should still have meaningful magnitude (not all removed)
        self.assertGreater(residual.abs().mean().mean(), 0.1)


class TestICWeightedCombination(unittest.TestCase):

    def test_combines_multiple_signals(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="B")
        tickers = [f"T{i}" for i in range(15)]

        returns = pd.DataFrame(
            np.random.randn(200, 15) * 0.01, index=dates, columns=tickers,
        )
        # Signal A has positive IC
        sig_a = returns.shift(-1) + np.random.randn(200, 15) * 0.005
        # Signal B has weaker positive IC
        sig_b = returns.shift(-1) * 0.3 + np.random.randn(200, 15) * 0.02

        combined = ic_weighted_combination(
            {"A": sig_a, "B": sig_b}, returns, lookback=30,
        )
        self.assertEqual(combined.shape, sig_a.shape)


if __name__ == "__main__":
    unittest.main()
