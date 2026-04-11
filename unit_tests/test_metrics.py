import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np
from backtester.metrics import ExtendedMetrics


class TestExtendedMetrics(unittest.TestCase):

    def _make_returns(self, values):
        """Create a returns series from daily return values."""
        dates = pd.date_range('2023-01-01', periods=len(values), freq='B')
        return pd.Series(values, index=dates)

    def _make_portfolio_values(self, returns):
        """Create portfolio values from returns."""
        values = [100000]
        for r in returns:
            values.append(values[-1] * (1 + r))
        dates = pd.date_range('2023-01-01', periods=len(values), freq='B')
        return pd.Series(values, index=dates)

    def test_positive_sharpe_for_positive_returns(self):
        """Strategy with consistent positive returns should have positive Sharpe."""
        calc = ExtendedMetrics(risk_free_rate=0.0)
        returns = self._make_returns([0.01] * 100)
        pv = self._make_portfolio_values(returns.values)
        metrics = calc.calculate(pv, returns)

        self.assertGreater(metrics['Sharpe Ratio'], 0)

    def test_sortino_higher_than_sharpe_for_no_downside(self):
        """With no negative returns, Sortino should be >= Sharpe (or inf)."""
        calc = ExtendedMetrics(risk_free_rate=0.0)
        returns = self._make_returns([0.005] * 100)
        pv = self._make_portfolio_values(returns.values)
        metrics = calc.calculate(pv, returns)

        self.assertGreaterEqual(metrics['Sortino Ratio'], metrics['Sharpe Ratio'])

    def test_max_drawdown_correct(self):
        """Max drawdown should match a known scenario."""
        calc = ExtendedMetrics()
        # Portfolio: 100, 110, 90, 95 -> drawdown from 110 to 90 = -18.18%
        pv = pd.Series([100, 110, 90, 95], index=pd.date_range('2023-01-01', periods=4, freq='B'))
        returns = pv.pct_change().dropna()
        metrics = calc.calculate(pv, returns)

        expected_dd = (90 / 110) - 1  # -0.1818...
        self.assertAlmostEqual(metrics['Max Drawdown'], expected_dd, places=3)

    def test_calmar_ratio(self):
        """Calmar = annualized return / |max drawdown|."""
        calc = ExtendedMetrics(risk_free_rate=0.0)
        returns = self._make_returns([0.001] * 252)
        pv = self._make_portfolio_values(returns.values)
        metrics = calc.calculate(pv, returns)

        if metrics['Max Drawdown'] != 0:
            expected_calmar = metrics['Annualized Return'] / abs(metrics['Max Drawdown'])
            self.assertAlmostEqual(metrics['Calmar Ratio'], expected_calmar, places=2)

    def test_win_rate(self):
        """Win rate should be fraction of positive return days."""
        calc = ExtendedMetrics()
        returns = self._make_returns([0.01, -0.005, 0.02, -0.01, 0.005])
        pv = self._make_portfolio_values(returns.values)
        metrics = calc.calculate(pv, returns)

        # 3 positive out of 5
        self.assertAlmostEqual(metrics['Win Rate'], 0.6)

    def test_profit_factor(self):
        """Profit factor = sum(positive returns) / |sum(negative returns)|."""
        calc = ExtendedMetrics()
        returns = self._make_returns([0.01, -0.005, 0.02])
        pv = self._make_portfolio_values(returns.values)
        metrics = calc.calculate(pv, returns)

        expected_pf = (0.01 + 0.02) / abs(-0.005)
        self.assertAlmostEqual(metrics['Profit Factor'], expected_pf, places=2)

    def test_beta_with_benchmark(self):
        """Beta should be close to 1 when strategy matches benchmark."""
        calc = ExtendedMetrics()
        np.random.seed(42)
        returns_vals = np.random.normal(0.001, 0.01, 100)
        returns = self._make_returns(returns_vals)
        benchmark = self._make_returns(returns_vals)  # identical
        pv = self._make_portfolio_values(returns.values)
        metrics = calc.calculate(pv, returns, benchmark)

        self.assertAlmostEqual(metrics['Beta'], 1.0, places=1)

    def test_alpha_zero_when_matching_benchmark(self):
        """Alpha should be ~0 when strategy matches benchmark exactly."""
        calc = ExtendedMetrics(risk_free_rate=0.0)
        np.random.seed(42)
        returns_vals = np.random.normal(0.001, 0.01, 100)
        returns = self._make_returns(returns_vals)
        benchmark = self._make_returns(returns_vals)
        pv = self._make_portfolio_values(returns.values)
        metrics = calc.calculate(pv, returns, benchmark)

        self.assertAlmostEqual(metrics['Alpha'], 0.0, places=2)

    def test_configurable_risk_free_rate(self):
        """Risk-free rate should affect Sharpe calculation."""
        returns = self._make_returns([0.001] * 100)
        pv = self._make_portfolio_values(returns.values)

        calc_low_rf = ExtendedMetrics(risk_free_rate=0.0)
        calc_high_rf = ExtendedMetrics(risk_free_rate=0.10)

        metrics_low = calc_low_rf.calculate(pv, returns)
        metrics_high = calc_high_rf.calculate(pv, returns)

        # Higher risk-free rate -> lower Sharpe
        self.assertGreater(metrics_low['Sharpe Ratio'], metrics_high['Sharpe Ratio'])


if __name__ == '__main__':
    unittest.main()
