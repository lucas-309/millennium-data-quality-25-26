import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np

from backtester.factor_models import (
    rolling_market_model,
    residualize_returns,
    sector_neutralize,
    beta_neutralize,
    factor_exposure_report,
)


class TestRollingMarketModel(unittest.TestCase):

    def test_recovers_known_beta(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        market = pd.Series(np.random.normal(0.0005, 0.01, n), index=dates)
        # Stock with true beta 1.5 and small idiosyncratic noise
        stock_returns = 1.5 * market + np.random.normal(0, 0.005, n)
        returns = pd.DataFrame({"STOCK": stock_returns}, index=dates)

        alpha, beta = rolling_market_model(returns, market, window=150)
        # Last beta should be close to 1.5
        self.assertAlmostEqual(beta["STOCK"].iloc[-1], 1.5, places=1)

    def test_residualize_removes_market_component(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        market = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        noise = pd.Series(np.random.normal(0, 0.005, n), index=dates)
        stock = 0.8 * market + noise
        returns = pd.DataFrame({"S": stock}, index=dates)

        residual = residualize_returns(returns, market, window=150)
        # Residual should have much lower correlation with market than original
        original_corr = returns["S"].corr(market)
        residual_clean = residual["S"].dropna()
        residual_corr = residual_clean.corr(market.loc[residual_clean.index])
        self.assertGreater(original_corr, 0.5)
        self.assertLess(abs(residual_corr), 0.3)


class TestSectorNeutralize(unittest.TestCase):

    def test_sector_mean_zero_after_neutralization(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        scores = pd.DataFrame({
            "AAPL": [1.0] * 10,
            "GOOGL": [2.0] * 10,
            "JPM": [5.0] * 10,
            "BAC": [7.0] * 10,
        }, index=dates)
        sector_map = {"AAPL": "TECH", "GOOGL": "TECH", "JPM": "FIN", "BAC": "FIN"}
        neutral = sector_neutralize(scores, sector_map)

        # TECH tickers: AAPL=-0.5, GOOGL=+0.5 (around mean 1.5)
        self.assertAlmostEqual(neutral["AAPL"].iloc[0] + neutral["GOOGL"].iloc[0], 0.0)
        # FIN tickers: JPM=-1.0, BAC=+1.0 (around mean 6.0)
        self.assertAlmostEqual(neutral["JPM"].iloc[0] + neutral["BAC"].iloc[0], 0.0)


class TestBetaNeutralize(unittest.TestCase):

    def test_residualized_has_no_beta_exposure(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        tickers = [f"T{i}" for i in range(20)]

        # Betas uniformly spread
        betas = pd.DataFrame(
            np.tile(np.linspace(0.5, 2.0, 20), (30, 1)),
            index=dates, columns=tickers,
        )
        # Scores perfectly correlated with beta (should be removed)
        scores = betas.copy()

        neutral = beta_neutralize(scores, betas)

        # After neutralization, cross-sectional correlation with beta should be ~0
        for date in dates:
            residual = neutral.loc[date]
            if residual.std(ddof=0) < 1e-12:
                self.assertLess(residual.abs().max(), 1e-10)
                continue
            corr = residual.corr(betas.loc[date])
            self.assertLess(abs(corr), 0.01)


class TestFactorExposureReport(unittest.TestCase):

    def test_recovers_known_betas(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        mkt = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        smb = pd.Series(np.random.normal(0, 0.008, n), index=dates)

        # Strategy has beta 1.2 on MKT and -0.3 on SMB
        strat = 1.2 * mkt - 0.3 * smb + np.random.normal(0, 0.002, n)
        strat = pd.Series(strat, index=dates)

        factors = pd.DataFrame({"MKT": mkt, "SMB": smb})
        report = factor_exposure_report(strat, factors)

        self.assertAlmostEqual(report["beta_MKT"], 1.2, places=1)
        self.assertAlmostEqual(report["beta_SMB"], -0.3, places=1)
        self.assertGreater(report["r_squared"], 0.8)


if __name__ == "__main__":
    unittest.main()
