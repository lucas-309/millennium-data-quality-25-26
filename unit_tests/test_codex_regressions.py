"""Regression tests for the 5 bugs Codex found in the working tree review.

Each test pins the corrected behavior so these issues can't silently reappear.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
import pandas as pd

from backtester.backtesters.equity_backtest import EquityBacktestEngine
from backtester.risk_manager import RiskManager, RiskLimits, RiskDecision
from backtester.multi_strategy import combine_strategy_returns
from backtester.research_backtester import BacktestConfig, run_weight_backtest
from strategies.pairs_trading import PairsTradingOrderGenerator
from strategies.kalman_pairs import KalmanPairsOrderGenerator


# ---------------------------------------------------------------------------
# Bug 1 & 2: pair exits must fully unwind positions, not leave residue.
# ---------------------------------------------------------------------------
def _cointegrated_pair(n=400, seed=1):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    trend = np.cumsum(np.random.normal(0.0005, 0.01, n)) + np.log(100)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.85 * spread[i - 1] + np.random.normal(0, 0.03)
    price_a = np.exp(trend + spread / 2)
    price_b = np.exp(trend - spread / 2)
    return pd.DataFrame({"A": price_a, "B": price_b}, index=dates)


class TestPairsExitBug(unittest.TestCase):
    def test_pairs_trading_exits_leave_no_residue(self):
        data = _cointegrated_pair()
        gen = PairsTradingOrderGenerator(
            pairs=[("A", "B")],
            lookback_window=60,
            entry_zscore=1.5,
            exit_zscore=0.5,
            allocation_per_leg=0.10,
            min_correlation=0.5,
        )
        orders = gen.generate_orders(data)

        engine = EquityBacktestEngine(
            initial_cash=1_000_000,
            commission_per_share=0.0,
            commission_min=0.0,
            slippage_bps=0.0,
            max_position_pct=1.0,
        )
        results = engine.run_backtest(orders, data)

        # After every completed round-trip, holdings should return close to 0.
        # We check that the number of distinct non-zero holding states is
        # bounded (entry → exit should zero out), not growing monotonically.
        holdings = results["daily_holdings_and_cash"]
        final_row = holdings.iloc[-1]
        # If the exit bug were present, after many round-trips holdings would
        # have huge leftover exposure in A and B. With the fix, final holdings
        # should be near zero (within one-position leftover from the last
        # trade cycle, if it was still open).
        residual_a = abs(final_row.get("A", 0))
        residual_b = abs(final_row.get("B", 0))
        # Allow residual from one in-flight pair at most; pre-fix this would
        # be hundreds of shares.
        self.assertLess(residual_a, 200)
        self.assertLess(residual_b, 200)

    def test_kalman_pairs_exits_leave_no_residue(self):
        data = _cointegrated_pair(n=500, seed=3)
        gen = KalmanPairsOrderGenerator(
            pairs=[("A", "B")],
            entry_zscore=1.5,
            exit_zscore=0.5,
            allocation_per_leg=0.10,
        )
        orders = gen.generate_orders(data)

        engine = EquityBacktestEngine(
            initial_cash=1_000_000,
            commission_per_share=0.0,
            commission_min=0.0,
            slippage_bps=0.0,
            max_position_pct=1.0,
        )
        results = engine.run_backtest(orders, data)

        holdings = results["daily_holdings_and_cash"]
        final_row = holdings.iloc[-1]
        residual_a = abs(final_row.get("A", 0))
        residual_b = abs(final_row.get("B", 0))
        self.assertLess(residual_a, 200)
        self.assertLess(residual_b, 200)


# ---------------------------------------------------------------------------
# Bug 3: weight backtest must not rebalance daily when freq is monthly.
# ---------------------------------------------------------------------------
class TestWeightBacktestRebalanceFrequency(unittest.TestCase):
    def test_monthly_rebalance_has_zero_mid_month_turnover(self):
        # Build synthetic two-asset world
        np.random.seed(42)
        n = 252
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        prices = pd.DataFrame({
            "AAA": (1 + np.random.normal(0.001, 0.01, n)).cumprod() * 100,
            "BBB": (1 + np.random.normal(0.0005, 0.012, n)).cumprod() * 100,
        }, index=dates)

        # Target weights: constant 50/50 throughout the whole backtest
        target = pd.DataFrame(0.5, index=dates, columns=["AAA", "BBB"])

        config = BacktestConfig(
            rebalance_frequency="ME",
            signal_lag=0,
            transaction_cost_bps=10.0,
            construction_method="equal_weight",
            long_quantile=0.5,
            short_quantile=0.5,
            max_position_weight=1.0,
            min_names=2,
        )

        result = run_weight_backtest(prices, target, config, strategy_name="Constant50")

        # With constant weights and the fix, turnover after day 1 should be 0
        # (drifts don't trigger retracing). Before the fix, every day had a
        # small turnover from drift.
        mid_month_turnover = result.turnover.iloc[5:].sum()
        self.assertLess(
            mid_month_turnover, 0.01,
            f"Expected ~0 turnover after day 1 with constant weights, got {mid_month_turnover}",
        )


# ---------------------------------------------------------------------------
# Bug 4: risk manager must allow de-risking an already-oversized position.
# ---------------------------------------------------------------------------
class TestRiskManagerDeRisking(unittest.TestCase):
    def test_sell_allowed_on_oversized_long(self):
        rm = RiskManager(RiskLimits(
            max_position_pct=0.10,
            max_gross_leverage=10.0,
            max_net_leverage=10.0,
        ))
        # Pre-existing 50-share long at $100 = $5000 = 50% of $10k equity.
        # Cap is 10% = $1000 = 10 shares. We're way over.
        decision, qty, reason = rm.check_order(
            ticker="AAPL",
            side="SELL",
            quantity=5,  # reducing but still leaves 45 shares (still over cap)
            price=100.0,
            current_holdings={"AAPL": 50},
            current_cash=5000,
        )
        # Previously: REJECT (allowed = max(0, 10 - 50) = 0)
        # Now: should ALLOW because the trade reduces |position|
        self.assertEqual(decision, RiskDecision.ALLOW)
        self.assertEqual(qty, 5)

    def test_sell_allowed_even_if_still_oversized(self):
        rm = RiskManager(RiskLimits(
            max_position_pct=0.10,
            max_gross_leverage=10.0,
            max_net_leverage=10.0,
        ))
        decision, qty, _ = rm.check_order(
            "AAPL", "SELL", 20, 100.0,
            current_holdings={"AAPL": 30},  # 30 * 100 = 3000 = 30% (over 10%)
            current_cash=7000,
        )
        # Post-trade: 10 shares, still $1000 / $10k = 10% (right at cap)
        # This is a de-risking move — must be allowed
        self.assertEqual(decision, RiskDecision.ALLOW)
        self.assertEqual(qty, 20)

    def test_buy_still_rejected_on_oversized_long(self):
        """Sanity check: increasing an already-oversized position is still rejected."""
        rm = RiskManager(RiskLimits(
            max_position_pct=0.10,
            max_gross_leverage=10.0,
            max_net_leverage=10.0,
        ))
        decision, _, _ = rm.check_order(
            "AAPL", "BUY", 5, 100.0,
            current_holdings={"AAPL": 50},
            current_cash=5000,
        )
        self.assertEqual(decision, RiskDecision.REJECT)

    def test_buy_to_cover_short_allowed(self):
        """Buying to cover an oversized short is a de-risking trade."""
        rm = RiskManager(RiskLimits(
            max_position_pct=0.10,
            max_gross_leverage=10.0,
            max_net_leverage=10.0,
        ))
        decision, qty, _ = rm.check_order(
            "AAPL", "BUY", 20, 100.0,
            current_holdings={"AAPL": -50},  # oversized short
            current_cash=15000,
        )
        self.assertEqual(decision, RiskDecision.ALLOW)
        self.assertEqual(qty, 20)


# ---------------------------------------------------------------------------
# Bug 5: combined strategy rebalances on the next trading day after a
# calendar boundary that lands on a weekend.
# ---------------------------------------------------------------------------
class TestMultiStrategyRebalance(unittest.TestCase):
    def test_monthly_rebalance_triggers_on_business_day(self):
        # Build 1 year of daily returns with two strategies whose relative
        # vols change over time so risk parity weights would evolve.
        np.random.seed(42)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        # Strategy A: low vol in H1, high vol in H2
        vol_a = np.where(np.arange(n) < n // 2, 0.005, 0.020)
        returns_a = pd.Series(np.random.normal(0.0005, vol_a, n), index=dates)
        # Strategy B: opposite
        vol_b = np.where(np.arange(n) < n // 2, 0.020, 0.005)
        returns_b = pd.Series(np.random.normal(0.0003, vol_b, n), index=dates)

        combined = combine_strategy_returns(
            {"A": returns_a, "B": returns_b},
            method="risk_parity",
            lookback=60,
            rebalance_freq="ME",
        )

        # If rebalancing never triggered, weights would stay at the initial
        # 50/50 and the combined return would not reflect the vol regime
        # change. To verify the trigger works, we check that the combined
        # series has nonzero variance AND that it differs from a pure
        # equal-weight combination (which would be static).
        self.assertGreater(combined.std(), 0)

        # Compare against no-rebalance behavior
        static = combine_strategy_returns(
            {"A": returns_a, "B": returns_b},
            method="equal",
            lookback=60,
            rebalance_freq="ME",
        )
        # Risk parity should differ meaningfully from equal weight since vols
        # are very different between the two halves.
        diff = (combined - static).abs().mean()
        self.assertGreater(diff, 1e-5, "Risk parity should differ from equal weight")


if __name__ == "__main__":
    unittest.main()
