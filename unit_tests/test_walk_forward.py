import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np

from backtester.walk_forward import (
    purged_kfold_splits,
    combinatorial_purged_cv,
    deflated_sharpe_ratio,
    annualized_sharpe,
    probability_of_backtest_overfitting,
    walk_forward_backtest,
)


class TestPurgedKFold(unittest.TestCase):

    def test_splits_cover_full_index(self):
        index = pd.date_range("2023-01-01", periods=100, freq="B")
        splits = purged_kfold_splits(index, n_splits=5, embargo_days=0)
        self.assertEqual(len(splits), 5)

        # Each fold's test set should be approximately 20% of data
        for train, test in splits:
            self.assertGreaterEqual(len(test), 15)

    def test_embargo_creates_gap(self):
        index = pd.date_range("2023-01-01", periods=100, freq="B")
        splits = purged_kfold_splits(index, n_splits=5, embargo_days=5)
        for train, test in splits:
            # No train sample should be within 5 days of any test sample
            for t_date in test[:1]:  # spot check first test date
                for tr_date in train:
                    diff = abs((tr_date - t_date).days)
                    if diff <= 5:
                        self.fail(f"Embargo violated: train {tr_date}, test {t_date}")

    def test_no_overlap_between_train_and_test(self):
        index = pd.date_range("2023-01-01", periods=60, freq="B")
        splits = purged_kfold_splits(index, n_splits=3, embargo_days=0)
        for train, test in splits:
            overlap = set(train).intersection(set(test))
            self.assertEqual(len(overlap), 0)


class TestCombinatorialPurgedCV(unittest.TestCase):

    def test_produces_many_splits(self):
        index = pd.date_range("2023-01-01", periods=60, freq="B")
        splits = combinatorial_purged_cv(index, n_splits=6, n_test_splits=2)
        # C(6,2) = 15 combinations
        self.assertEqual(len(splits), 15)


class TestDeflatedSharpe(unittest.TestCase):

    def test_dsr_lower_with_more_trials(self):
        # Fixed Sharpe, increasing trials → DSR should decrease
        dsr_1 = deflated_sharpe_ratio(sharpe=1.5, n_trials=1, n_observations=252)
        dsr_100 = deflated_sharpe_ratio(sharpe=1.5, n_trials=100, n_observations=252)
        self.assertGreater(dsr_1, dsr_100)

    def test_dsr_bounded(self):
        dsr = deflated_sharpe_ratio(sharpe=2.0, n_trials=10, n_observations=500)
        self.assertGreaterEqual(dsr, 0.0)
        self.assertLessEqual(dsr, 1.0)


class TestAnnualizedSharpe(unittest.TestCase):

    def test_positive_returns_positive_sharpe(self):
        returns = pd.Series(np.full(252, 0.001))
        sr = annualized_sharpe(returns, risk_free_rate=0.0)
        self.assertGreater(sr, 0)


class TestProbabilityOfBacktestOverfitting(unittest.TestCase):

    def test_pbo_on_random_strategies(self):
        # With random strategies, PBO should be ~0.5
        np.random.seed(42)
        n_trials = 50
        n_splits = 8
        metrics = pd.DataFrame(
            np.random.randn(n_trials, n_splits),
            columns=[f"fold{i}" for i in range(n_splits)],
        )
        pbo = probability_of_backtest_overfitting(metrics)
        # Random strategies → PBO around 0.5 (may vary)
        self.assertGreater(pbo, 0.2)
        self.assertLess(pbo, 0.8)


class TestWalkForwardBacktest(unittest.TestCase):

    def test_walk_forward_returns_grid(self):
        index = pd.date_range("2023-01-01", periods=100, freq="B")

        def train_fn(params, idx):
            return float(params["x"]) * len(idx) / 1000

        def test_fn(params, idx):
            return float(params["x"]) * len(idx) / 1000 + 0.01

        df = walk_forward_backtest(
            param_grid={"x": [1, 2, 3]},
            train_fn=train_fn,
            test_fn=test_fn,
            index=index,
            n_splits=5,
        )
        # 3 params * 5 folds = 15 rows
        self.assertEqual(len(df), 15)
        self.assertIn("train_score", df.columns)
        self.assertIn("test_score", df.columns)


if __name__ == "__main__":
    unittest.main()
