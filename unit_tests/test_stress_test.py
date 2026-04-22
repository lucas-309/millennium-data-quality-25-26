import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np

from backtester.stress_test import (
    regime_performance,
    regime_report,
    worst_drawdowns,
    HISTORICAL_REGIMES,
)
from backtester.monte_carlo import (
    stationary_block_bootstrap_sample,
    bootstrap_confidence_interval,
    bootstrap_many_metrics,
)
from backtester.sensitivity import parameter_sweep, sensitivity_heatmap_data


def make_returns(n=500, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(0.0005, 0.01, n), index=dates)


class TestRegimePerformance(unittest.TestCase):
    def test_regime_performance_on_covid(self):
        # Create returns series spanning COVID
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        # Inject a sharp selloff during COVID window
        values = np.zeros(200)
        values[30:60] = -0.02
        returns = pd.Series(values, index=dates)
        result = regime_performance(returns, "2020-02-15", "2020-04-30")
        self.assertLess(result["cumulative_return"], 0)
        self.assertLess(result["max_drawdown"], 0)

    def test_regime_report_covers_all_regimes(self):
        np.random.seed(42)
        dates = pd.date_range("2005-01-01", periods=5000, freq="B")
        returns = pd.Series(np.random.normal(0.0005, 0.01, 5000), index=dates)
        report = regime_report(returns)
        self.assertEqual(len(report), len(HISTORICAL_REGIMES))
        self.assertIn("regime", report.columns)
        self.assertIn("max_drawdown", report.columns)

    def test_empty_regime_returns_nan(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        returns = pd.Series([0.001] * 100, index=dates)
        result = regime_performance(returns, "2008-01-01", "2008-12-31")
        self.assertTrue(pd.isna(result["sharpe"]))


class TestWorstDrawdowns(unittest.TestCase):
    def test_identifies_drawdown_episode(self):
        returns = pd.Series(
            [0.01, -0.02, -0.03, -0.01, 0.02, 0.03, 0.04],
            index=pd.date_range("2023-01-01", periods=7, freq="B"),
        )
        dd = worst_drawdowns(returns, top_n=3)
        self.assertGreater(len(dd), 0)
        self.assertLess(dd.iloc[0]["depth"], 0)


class TestMonteCarlo(unittest.TestCase):
    def test_bootstrap_sample_same_length(self):
        returns = make_returns()
        sample = stationary_block_bootstrap_sample(returns, avg_block_length=10)
        self.assertEqual(len(sample), len(returns))

    def test_bootstrap_ci_contains_mean(self):
        returns = make_returns(n=500, seed=123)
        point, lower, upper = bootstrap_confidence_interval(
            returns,
            metric_fn=lambda r: r.mean(),
            alpha=0.10,
            n_iterations=200,
            seed=42,
        )
        # 90% CI should contain the point estimate
        self.assertLessEqual(lower, point)
        self.assertGreaterEqual(upper, point)

    def test_bootstrap_many_metrics(self):
        returns = make_returns()
        def sharpe(r):
            return r.mean() / r.std(ddof=0) * np.sqrt(252) if r.std(ddof=0) > 0 else 0.0
        def max_dd(r):
            cum = (1 + r).cumprod()
            return (cum / cum.cummax() - 1).min()

        df = bootstrap_many_metrics(
            returns,
            metrics={"sharpe": sharpe, "max_dd": max_dd},
            n_iterations=100,
            seed=42,
        )
        self.assertIn("sharpe", df.index)
        self.assertIn("max_dd", df.index)
        self.assertIn("point", df.columns)
        self.assertIn("lower", df.columns)


class TestSensitivity(unittest.TestCase):
    def test_parameter_sweep(self):
        def metric(params):
            return params["x"] * 2 + params["y"]

        result = parameter_sweep(
            {"x": [1, 2, 3], "y": [10, 20]},
            metric,
        )
        self.assertEqual(len(result), 6)  # 3 * 2
        self.assertIn("metric", result.columns)

    def test_heatmap_pivot(self):
        def metric(params):
            return params["x"] + params["y"]

        df = parameter_sweep({"x": [1, 2], "y": [10, 20]}, metric)
        heat = sensitivity_heatmap_data(df, row_param="x", col_param="y")
        self.assertEqual(heat.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
