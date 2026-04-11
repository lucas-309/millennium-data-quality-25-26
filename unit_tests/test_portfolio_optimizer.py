import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np

from backtester.portfolio_optimizer import (
    sample_covariance,
    shrinkage_covariance,
    equal_weight,
    inverse_volatility_weight,
    mean_variance_weights,
    risk_parity_weights,
    hierarchical_risk_parity,
)
from backtester.multi_strategy import combine_strategy_returns


def random_returns(n=252, n_assets=5, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    tickers = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(
        np.random.normal(0.0005, 0.01, (n, n_assets)),
        index=dates, columns=tickers,
    )


class TestCovarianceEstimators(unittest.TestCase):
    def test_sample_covariance_is_symmetric(self):
        returns = random_returns()
        cov = sample_covariance(returns)
        self.assertTrue(np.allclose(cov.values, cov.values.T))

    def test_shrinkage_covariance_shape(self):
        returns = random_returns()
        shrunk = shrinkage_covariance(returns, shrinkage=0.3)
        self.assertEqual(shrunk.shape, (5, 5))
        # Should be symmetric
        self.assertTrue(np.allclose(shrunk.values, shrunk.values.T))


class TestBasicWeights(unittest.TestCase):
    def test_equal_weight_sums_to_one(self):
        w = equal_weight(["A", "B", "C", "D"])
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertEqual(len(w), 4)

    def test_inverse_volatility_weight(self):
        vols = pd.Series([0.1, 0.2, 0.4], index=["A", "B", "C"])
        w = inverse_volatility_weight(vols)
        self.assertAlmostEqual(w.sum(), 1.0)
        # Lower vol should get higher weight
        self.assertGreater(w["A"], w["C"])


class TestMeanVariance(unittest.TestCase):
    def test_positive_expected_return_gets_positive_weight(self):
        mu = pd.Series([0.01, -0.005, 0.002], index=["A", "B", "C"])
        cov = pd.DataFrame(
            np.eye(3) * 0.01,
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        w = mean_variance_weights(mu, cov, long_only=False)
        self.assertGreater(w["A"], 0)
        self.assertLess(w["B"], 0)

    def test_long_only_enforces_nonneg(self):
        mu = pd.Series([0.01, -0.005, 0.002], index=["A", "B", "C"])
        cov = pd.DataFrame(np.eye(3) * 0.01, index=mu.index, columns=mu.index)
        w = mean_variance_weights(mu, cov, long_only=True)
        for val in w.values:
            self.assertGreaterEqual(val, 0)


class TestRiskParity(unittest.TestCase):
    def test_risk_parity_equalizes_contributions(self):
        np.random.seed(42)
        returns = random_returns(n=500, n_assets=4)
        cov = shrinkage_covariance(returns)
        w = risk_parity_weights(cov)
        # Check risk contributions are approximately equal
        portfolio_var = w.values @ cov.values @ w.values
        marginal = cov.values @ w.values
        contribs = w.values * marginal
        # Normalize by portfolio var
        rc = contribs / portfolio_var
        # Should be close to 1/n for each
        self.assertLess(rc.std(), 0.05)


class TestHRP(unittest.TestCase):
    def test_hrp_sums_to_one(self):
        returns = random_returns(n=200, n_assets=6)
        w = hierarchical_risk_parity(returns)
        self.assertAlmostEqual(w.sum(), 1.0, places=5)

    def test_hrp_nonneg(self):
        returns = random_returns()
        w = hierarchical_risk_parity(returns)
        for val in w.values:
            self.assertGreaterEqual(val, -1e-10)


class TestCombineStrategyReturns(unittest.TestCase):
    def test_equal_weight_combination(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        strat_a = pd.Series(np.random.normal(0.001, 0.01, n), index=dates)
        strat_b = pd.Series(np.random.normal(0.0005, 0.008, n), index=dates)

        combined = combine_strategy_returns(
            {"A": strat_a, "B": strat_b},
            method="equal",
            lookback=30,
        )
        self.assertEqual(len(combined), n)

    def test_risk_parity_combination(self):
        np.random.seed(42)
        n = 252
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        strat_a = pd.Series(np.random.normal(0.001, 0.005, n), index=dates)
        strat_b = pd.Series(np.random.normal(0.0005, 0.015, n), index=dates)

        combined = combine_strategy_returns(
            {"A": strat_a, "B": strat_b},
            method="risk_parity",
            lookback=100,
        )
        self.assertEqual(len(combined), n)

    def test_hrp_combination(self):
        np.random.seed(42)
        n = 252
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        strategies = {
            f"S{i}": pd.Series(np.random.normal(0, 0.01, n), index=dates)
            for i in range(4)
        }
        combined = combine_strategy_returns(
            strategies, method="hrp", lookback=100,
        )
        self.assertEqual(len(combined), n)


if __name__ == "__main__":
    unittest.main()
