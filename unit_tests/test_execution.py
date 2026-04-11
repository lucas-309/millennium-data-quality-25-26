import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np

from backtester.execution import (
    LinearImpact,
    SquareRootImpact,
    PowerLawImpact,
    BorrowCostModel,
    estimate_adv,
    cap_order_by_adv,
)


class TestMarketImpactModels(unittest.TestCase):

    def test_linear_impact_scales_with_participation(self):
        model = LinearImpact(lambda_bps=10.0)
        # 10% participation -> 1 bp
        bps_small = model.impact_bps(order_notional=100, adv_notional=1000)
        # 50% participation -> 5 bps
        bps_large = model.impact_bps(order_notional=500, adv_notional=1000)
        self.assertAlmostEqual(bps_small, 1.0, places=5)
        self.assertAlmostEqual(bps_large, 5.0, places=5)

    def test_square_root_impact_sublinear(self):
        model = SquareRootImpact(k=1.0)
        small = model.impact_bps(order_notional=100, adv_notional=10000, volatility=0.02)
        large = model.impact_bps(order_notional=400, adv_notional=10000, volatility=0.02)
        # 4x order size → 2x impact (sqrt scaling)
        self.assertAlmostEqual(large / small, 2.0, places=5)

    def test_power_law_impact_with_alpha(self):
        model = PowerLawImpact(c=1.0, alpha=0.6)
        bps = model.impact_bps(order_notional=1000, adv_notional=10000, volatility=0.02)
        self.assertGreater(bps, 0)

    def test_zero_adv_returns_zero_impact(self):
        for model in (LinearImpact(), SquareRootImpact(), PowerLawImpact()):
            self.assertEqual(model.impact_bps(1000, 0), 0.0)

    def test_impact_scales_with_volatility(self):
        low_vol = SquareRootImpact(k=1.0).impact_bps(100, 10000, volatility=0.01)
        high_vol = SquareRootImpact(k=1.0).impact_bps(100, 10000, volatility=0.04)
        # 4x vol → 4x impact
        self.assertAlmostEqual(high_vol / low_vol, 4.0, places=5)


class TestBorrowCost(unittest.TestCase):

    def test_default_borrow_cost(self):
        model = BorrowCostModel(default_annual_rate=0.05)  # 5%
        # $100,000 short, 1 day = 100000 * 0.05 / 252
        cost = model.daily_cost("AAPL", 100000)
        expected = 100000 * 0.05 / 252
        self.assertAlmostEqual(cost, expected, places=5)

    def test_hard_to_borrow_override(self):
        model = BorrowCostModel(
            default_annual_rate=0.005,
            hard_to_borrow={"GME": 0.50},  # 50% HTB rate
        )
        gme_cost = model.daily_cost("GME", 10000)
        aapl_cost = model.daily_cost("AAPL", 10000)
        self.assertGreater(gme_cost, aapl_cost * 50)

    def test_zero_short_notional(self):
        model = BorrowCostModel()
        self.assertEqual(model.daily_cost("AAPL", 0), 0.0)


class TestADVFunctions(unittest.TestCase):

    def test_estimate_adv_rolling(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        volume = pd.Series([1000] * 30, index=dates)
        price = pd.Series([50.0] * 30, index=dates)
        adv = estimate_adv(volume, price, window=20)
        # Expected: 1000 * 50 = 50000 per day
        self.assertAlmostEqual(adv.iloc[-1], 50000, places=0)

    def test_cap_order_by_adv(self):
        # ADV $100,000, 10% max participation = $10,000 max order
        # At price $100, that's 100 shares max
        capped = cap_order_by_adv(
            requested_quantity=500,
            price=100.0,
            adv_dollars=100000,
            max_participation=0.10,
        )
        self.assertEqual(capped, 100)

    def test_cap_order_leaves_small_orders_alone(self):
        capped = cap_order_by_adv(
            requested_quantity=50,
            price=100.0,
            adv_dollars=100000,
            max_participation=0.10,
        )
        self.assertEqual(capped, 50)

    def test_cap_order_zero_adv(self):
        capped = cap_order_by_adv(100, 50, 0, 0.1)
        self.assertEqual(capped, 0)


if __name__ == "__main__":
    unittest.main()
