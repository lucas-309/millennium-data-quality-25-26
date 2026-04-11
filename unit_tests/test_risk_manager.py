import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np

from backtester.risk_manager import RiskManager, RiskLimits, RiskDecision, vol_target_scale


class TestRiskManager(unittest.TestCase):

    def _make_manager(self, **kwargs):
        limits = RiskLimits(**kwargs)
        return RiskManager(limits)

    def test_position_limit_allows_small_order(self):
        rm = self._make_manager(max_position_pct=0.10)
        decision, qty, _ = rm.check_order(
            "AAPL", "BUY", 50, 100.0,
            current_holdings={}, current_cash=100000,
        )
        self.assertEqual(decision, RiskDecision.ALLOW)
        self.assertEqual(qty, 50)

    def test_position_limit_reduces_big_order(self):
        rm = self._make_manager(max_position_pct=0.10)
        # BUY 500 at $100 = $50,000 = 50% of equity ($100k). Limit is 10%.
        decision, qty, reason = rm.check_order(
            "AAPL", "BUY", 500, 100.0,
            current_holdings={}, current_cash=100000,
        )
        self.assertIn(decision, (RiskDecision.REDUCE, RiskDecision.REJECT))
        if decision == RiskDecision.REDUCE:
            self.assertLessEqual(qty * 100, 10000)

    def test_drawdown_kill_switch_activates(self):
        rm = self._make_manager(max_drawdown_stop=0.10)
        rm.update_equity(100000)
        rm.update_equity(110000)  # new high
        rm.update_equity(98000)   # 10.9% drawdown -> trigger
        self.assertTrue(rm.state.kill_switch_active)

    def test_kill_switch_blocks_new_opens(self):
        rm = self._make_manager(max_drawdown_stop=0.05)
        rm.update_equity(100000)
        rm.update_equity(90000)  # 10% DD
        self.assertTrue(rm.state.kill_switch_active)

        decision, _, reason = rm.check_order(
            "AAPL", "BUY", 10, 100.0,
            current_holdings={}, current_cash=90000,
        )
        self.assertEqual(decision, RiskDecision.REJECT)
        self.assertEqual(reason, "drawdown_kill_switch")

    def test_kill_switch_allows_closing_orders(self):
        rm = self._make_manager(max_drawdown_stop=0.05)
        rm.update_equity(100000)
        rm.update_equity(90000)
        # Have a long position, try to sell it (closing order)
        decision, _, _ = rm.check_order(
            "AAPL", "SELL", 10, 100.0,
            current_holdings={"AAPL": 10}, current_cash=89000,
        )
        self.assertEqual(decision, RiskDecision.ALLOW)

    def test_sector_limit(self):
        sector_map = {"AAPL": "TECH", "GOOGL": "TECH", "JPM": "FIN"}
        rm = self._make_manager(
            max_sector_pct=0.20,
            sector_map=sector_map,
            max_position_pct=1.0,  # disable position limit
        )
        # Already $15k in AAPL (15% of $100k). BUY another $10k GOOGL → 25% tech
        holdings = {"AAPL": 150}
        price_lookup = {"AAPL": 100.0, "GOOGL": 100.0}
        decision, _, reason = rm.check_order(
            "GOOGL", "BUY", 100, 100.0,
            current_holdings=holdings, current_cash=85000,
            price_lookup=price_lookup,
        )
        self.assertEqual(decision, RiskDecision.REJECT)
        self.assertEqual(reason, "sector_limit")

    def test_beta_limit(self):
        beta_map = {"SPY": 1.0, "TLT": -0.5, "AAPL": 1.5}
        rm = self._make_manager(
            max_abs_beta=0.30,
            beta_map=beta_map,
            max_position_pct=1.0,
        )
        # Long $50k AAPL beta 1.5 → portfolio beta 0.75. Exceeds 0.30.
        decision, _, reason = rm.check_order(
            "AAPL", "BUY", 500, 100.0,
            current_holdings={}, current_cash=100000,
            price_lookup={"AAPL": 100.0},
        )
        self.assertEqual(decision, RiskDecision.REJECT)
        self.assertEqual(reason, "beta_limit")

    def test_vol_target_scale(self):
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0, 0.02, 252),  # 2% daily vol ≈ 32% annual
            index=pd.date_range("2023-01-01", periods=252, freq="B"),
        )
        # Target 16% vol → scalar should be ~0.5
        scalar = vol_target_scale(returns, target_annual_vol=0.16, lookback=100)
        self.assertLess(scalar.iloc[-1], 0.8)
        self.assertGreater(scalar.iloc[-1], 0.3)


if __name__ == "__main__":
    unittest.main()
