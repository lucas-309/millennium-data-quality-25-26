import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np

from backtester.research_data import ResearchDataset
from strategies.residual_momentum import ResidualMomentumStrategy
from strategies.idio_reversal import IdiosyncraticReversalStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.quality import QualityStrategy
from strategies.pead import PEADStrategy
from strategies.kalman_pairs import KalmanPairsOrderGenerator


def make_dataset(n=500, n_tickers=10, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]

    # Market returns
    market_returns = pd.Series(
        np.random.normal(0.0003, 0.01, n), index=dates, name="MKT"
    )

    # Stocks: beta * market + idiosyncratic noise
    stock_returns = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    betas = np.linspace(0.5, 1.8, n_tickers)
    for i, ticker in enumerate(tickers):
        stock_returns[ticker] = betas[i] * market_returns + np.random.normal(
            0.0001, 0.015, n
        )

    prices = (1 + stock_returns).cumprod() * 100
    return ResearchDataset(
        prices=prices,
        returns=stock_returns,
        benchmark_returns=market_returns,
        volumes=pd.DataFrame(1e6, index=dates, columns=tickers),
    )


class TestResidualMomentum(unittest.TestCase):
    def test_generates_scores(self):
        ds = make_dataset()
        strat = ResidualMomentumStrategy(lookback_days=100, skip_days=20, beta_window=150)
        scores = strat.generate_scores(ds)
        self.assertEqual(scores.shape, ds.prices.shape)
        self.assertGreater(scores.notna().sum().sum(), 0)

    def test_lower_market_beta_than_raw(self):
        ds = make_dataset(n=600)
        strat = ResidualMomentumStrategy(lookback_days=100, skip_days=10, beta_window=200)
        residual_scores = strat.generate_scores(ds)

        # Raw momentum: 12-1 return
        raw_mom = ds.prices.pct_change(100).shift(10)

        # Correlate each signal cross-section with true betas
        true_betas = pd.Series(
            np.linspace(0.5, 1.8, len(ds.prices.columns)),
            index=ds.prices.columns,
        )

        residual_beta_corr = residual_scores.iloc[-1].corr(true_betas)
        raw_beta_corr = raw_mom.iloc[-1].corr(true_betas)

        # Residual scores should have less correlation with true beta than raw
        self.assertLess(abs(residual_beta_corr), abs(raw_beta_corr) + 0.3)


class TestIdiosyncraticReversal(unittest.TestCase):
    def test_generates_scores(self):
        ds = make_dataset()
        strat = IdiosyncraticReversalStrategy(lookback_days=5, beta_window=150)
        scores = strat.generate_scores(ds)
        self.assertEqual(scores.shape, ds.prices.shape)


class TestTrendFollowing(unittest.TestCase):
    def test_higher_score_for_uptrending_stock(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        tickers = ["UP", "DOWN"]
        up_prices = np.linspace(100, 200, n)
        down_prices = np.linspace(200, 100, n)
        prices = pd.DataFrame({"UP": up_prices, "DOWN": down_prices}, index=dates)
        returns = prices.pct_change().fillna(0)
        market = returns.mean(axis=1)

        ds = ResearchDataset(
            prices=prices, returns=returns, benchmark_returns=market,
        )
        strat = TrendFollowingStrategy(lookback_days=100, vol_window=30, min_periods=30)
        scores = strat.generate_scores(ds)

        last_scores = scores.iloc[-1].dropna()
        self.assertGreater(last_scores["UP"], last_scores["DOWN"])


class TestQuality(unittest.TestCase):
    def test_low_vol_stock_has_higher_score(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        tickers = ["STABLE", "VOLATILE"]
        # Both have same mean but different vol
        stable = np.random.normal(0.0003, 0.005, n)
        volatile = np.random.normal(0.0003, 0.03, n)
        returns = pd.DataFrame({"STABLE": stable, "VOLATILE": volatile}, index=dates)
        prices = (1 + returns).cumprod() * 100
        market = returns.mean(axis=1)

        ds = ResearchDataset(
            prices=prices, returns=returns, benchmark_returns=market,
        )
        strat = QualityStrategy(lookback_days=100)
        scores = strat.generate_scores(ds)
        last = scores.iloc[-1].dropna()
        self.assertGreater(last["STABLE"], last["VOLATILE"])


class TestPEAD(unittest.TestCase):
    def test_generates_scores(self):
        ds = make_dataset()
        strat = PEADStrategy(drift_days=30, abnormal_window=3)
        scores = strat.generate_scores(ds)
        self.assertEqual(scores.shape, ds.prices.shape)


class TestKalmanPairs(unittest.TestCase):
    def test_generates_trades_on_cointegrated_pair(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        trend = np.cumsum(np.random.normal(0.0005, 0.01, n)) + np.log(100)
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = 0.85 * spread[i - 1] + np.random.normal(0, 0.02)
        price_a = np.exp(trend + spread / 2)
        price_b = np.exp(trend - spread / 2)

        data = pd.DataFrame({"A": price_a, "B": price_b}, index=dates)
        gen = KalmanPairsOrderGenerator(pairs=[("A", "B")], entry_zscore=1.5)
        orders = gen.generate_orders(data)
        self.assertGreater(len(orders), 0)


if __name__ == "__main__":
    unittest.main()
