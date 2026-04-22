import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from backtester.tearsheet import generate_tearsheet
from backtester.attribution import (
    factor_attribution,
    brinson_attribution,
    return_contribution_ranking,
)


class TestTearsheet(unittest.TestCase):
    def test_generate_saves_file(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        returns = pd.Series(np.random.normal(0.0005, 0.01, 500), index=dates)
        benchmark = pd.Series(np.random.normal(0.0003, 0.009, 500), index=dates)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tearsheet.png"
            fig = generate_tearsheet(
                returns, benchmark, strategy_name="Test", output_path=output,
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)

    def test_generate_without_benchmark(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        returns = pd.Series(np.random.normal(0.0005, 0.01, 300), index=dates)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "no_benchmark.png"
            generate_tearsheet(returns, benchmark=None, output_path=output)
            self.assertTrue(output.exists())


class TestFactorAttribution(unittest.TestCase):
    def test_recovers_known_factor_exposures(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        mkt = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        smb = pd.Series(np.random.normal(0, 0.008, n), index=dates)

        # Strategy with known exposures and alpha
        alpha_daily = 0.0003
        strat = alpha_daily + 1.1 * mkt - 0.4 * smb + np.random.normal(0, 0.002, n)
        strat = pd.Series(strat, index=dates)

        factors = pd.DataFrame({"MKT": mkt, "SMB": smb})
        result = factor_attribution(strat, factors)

        self.assertAlmostEqual(result["factors"]["MKT"]["beta"], 1.1, places=1)
        self.assertAlmostEqual(result["factors"]["SMB"]["beta"], -0.4, places=1)
        self.assertGreater(result["r_squared"], 0.8)
        self.assertGreater(result["alpha_annualized"], 0)


class TestBrinsonAttribution(unittest.TestCase):
    def test_brinson_sector_decomposition(self):
        port_weights = pd.Series([0.3, 0.2, 0.3, 0.2], index=["AAPL", "GOOGL", "JPM", "BAC"])
        bench_weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["AAPL", "GOOGL", "JPM", "BAC"])
        port_ret = pd.Series([0.05, 0.03, 0.02, 0.01], index=["AAPL", "GOOGL", "JPM", "BAC"])
        bench_ret = pd.Series([0.04, 0.02, 0.03, 0.02], index=["AAPL", "GOOGL", "JPM", "BAC"])
        sector_map = {"AAPL": "TECH", "GOOGL": "TECH", "JPM": "FIN", "BAC": "FIN"}

        result = brinson_attribution(
            port_weights, port_ret, bench_weights, bench_ret, sector_map
        )
        self.assertIn("TECH", result.index)
        self.assertIn("FIN", result.index)
        self.assertIn("total_active", result.columns)


class TestReturnContributionRanking(unittest.TestCase):
    def test_ranks_top_contributors(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        weights = pd.DataFrame({
            "WINNER": [0.5] * 10,
            "LOSER": [0.5] * 10,
        }, index=dates)
        returns = pd.DataFrame({
            "WINNER": [0.01] * 10,
            "LOSER": [-0.005] * 10,
        }, index=dates)

        ranking = return_contribution_ranking(weights, returns, top_n=1)
        # Winner should be contributor, loser should be detractor
        self.assertGreater(len(ranking), 0)


if __name__ == "__main__":
    unittest.main()
