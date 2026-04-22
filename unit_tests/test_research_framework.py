import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backtester.data_source import WhartonDataSource
from backtester.research_data import ResearchDataset
from backtester.research_backtester import (
    BacktestResult,
    BacktestConfig,
    build_event_study,
    combine_strategy_portfolios,
    build_target_weights,
    run_strategy_backtest,
    run_weight_backtest,
    select_combination_candidates,
    signal_lag_table,
)
from backtester.research_reports import generate_data_quality_report, validate_spy_reconstruction
from strategies.research_strategies import MeanReversionStrategy, build_default_strategy_suite


class TestResearchFramework(unittest.TestCase):
    def test_wharton_adjusted_price_formula(self):
        raw = pd.DataFrame(
            {
                "tic": ["AAA", "AAA"],
                "datadate": ["2024-01-02", "2024-01-03"],
                "prccd": [100.0, 110.0],
                "ajexdi": [2.0, 2.0],
                "trfd": [1.0, 1.1],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            raw.to_csv(handle.name, index=False)
            source = WhartonDataSource(handle.name, use_total_return=True)

        try:
            history = source.get_historical_data(["AAA"], "2024-01-01", "2024-01-31")
            self.assertAlmostEqual(history.loc[pd.Timestamp("2024-01-02"), "AAA"], 50.0)
            self.assertAlmostEqual(history.loc[pd.Timestamp("2024-01-03"), "AAA"], 60.5)
        finally:
            os.remove(handle.name)

    def test_weight_backtest_with_transaction_costs(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100, 102, 104, 106, 108, 110],
                "BBB": [100, 99, 98, 97, 96, 95],
            },
            index=dates,
        )
        scores = pd.DataFrame(
            {
                "AAA": [1, 1, 1, 1, 1, 1],
                "BBB": [-1, -1, -1, -1, -1, -1],
            },
            index=dates,
        )
        asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
        config = BacktestConfig(
            rebalance_frequency="D",
            signal_lag=1,
            transaction_cost_bps=10.0,
            construction_method="equal_weight",
            long_quantile=0.5,
            short_quantile=0.5,
            leverage=1.0,
            max_position_weight=0.5,
            min_names=2,
        )

        weights = build_target_weights(scores, asset_returns, config)
        result = run_weight_backtest(prices, weights, config, strategy_name="Test Strategy")

        self.assertGreater(result.portfolio_value.iloc[-1], config.initial_cash)
        self.assertGreater(result.turnover.max(), 0.0)
        self.assertGreaterEqual(result.transaction_costs.sum(), 0.0)

    def test_signal_lag_and_event_study_outputs(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        returns = pd.DataFrame(
            {
                "AAA": [0.01] * 80,
                "BBB": [-0.01] * 80,
                "CCC": [0.005] * 80,
                "DDD": [-0.005] * 80,
                "EEE": [0.002] * 80,
            },
            index=dates,
        )
        scores = pd.DataFrame(
            {
                "AAA": [2.0] * 80,
                "BBB": [-2.0] * 80,
                "CCC": [1.0] * 80,
                "DDD": [-1.0] * 80,
                "EEE": [0.0] * 80,
            },
            index=dates,
        )
        lag_table = signal_lag_table(scores, returns)
        self.assertIn(0, lag_table.index)
        self.assertIn("sharpe_ratio", lag_table.columns)

        events = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "event_type": ["anncdate", "anncdate"],
                "event_date": [dates[40], dates[42]],
            }
        )
        study = build_event_study(returns, events, benchmark_returns=returns.mean(axis=1), event_type="anncdate", window=5)
        self.assertEqual(study.index.min(), -5)
        self.assertEqual(study.index.max(), 5)
        self.assertIn("median", study.columns)

    def test_data_quality_report_fields(self):
        raw = pd.DataFrame(
            {
                "tic": ["AAA", "AAA", "BBB"],
                "datadate": ["2024-01-02", "2024-01-03", "2024-01-02"],
                "prccd": [100.0, 101.0, 50.0],
                "ajexdi": [1.0, 2.0, 1.0],
                "trfd": [1.0, 1.01, 1.0],
                "cshtrd": [1000.0, 1100.0, None],
                "divd": [0.0, 0.5, 0.0],
                "eps": [1.0, 1.0, 2.0],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            raw.to_csv(handle.name, index=False)
            source = WhartonDataSource(handle.name, use_total_return=True)

        try:
            report = generate_data_quality_report(source, "2024-01-01", "2024-01-31")
            self.assertEqual(report["ticker_count"], 2)
            self.assertEqual(report["split_adjustment_events"], 1)
            self.assertEqual(report["dividend_event_count"], 1)
        finally:
            os.remove(handle.name)

    def test_spy_validation_summary_with_mock_source(self):
        mock_source = MagicMock()
        holdings = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Weight": [0.6, 0.4]})
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        prices = pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0],
                "AAA": [50.0, 51.0, 52.0],
                "BBB": [75.0, 75.5, 76.0],
            },
            index=dates,
        )
        weighted = pd.Series([1.0, 1.01, 1.02], index=dates)
        mock_source.read_spy_holdings.return_value = holdings
        mock_source.get_historical_data.return_value = prices
        mock_source.calculate_weighted_portfolio.return_value = weighted
        mock_source.verify_spy_vs_constituents.return_value = {
            "within_threshold": True,
            "max_difference": 0.001,
            "mean_difference": 0.0005,
            "pearson_coeff": 0.99,
            "spearman_coeff": 1.0,
            "worst_days": pd.Series([0.001], index=[dates[-1]]),
        }

        summary = validate_spy_reconstruction("dummy.xlsx", "2024-01-01", "2024-01-31", data_source=mock_source)
        self.assertTrue(summary["within_threshold"])
        self.assertEqual(summary["constituent_count"], 2)
        self.assertIn("worst_days", summary)

    def test_default_strategy_suite_is_small_and_simple(self):
        names = [strategy.name for strategy in build_default_strategy_suite()]
        self.assertEqual(names, ["Mean Reversion", "Momentum", "Low Volatility"])

    def test_run_strategy_backtest_helper(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 99.0, 98.0, 100.0, 101.0, 102.0],
                "BBB": [100.0, 101.0, 102.0, 101.0, 100.0, 99.0],
            },
            index=dates,
        )
        returns = prices.pct_change(fill_method=None).fillna(0.0)
        dataset = ResearchDataset(
            prices=prices,
            returns=returns,
            benchmark_returns=returns.mean(axis=1),
        )
        config = BacktestConfig(
            rebalance_frequency="D",
            signal_lag=0,
            transaction_cost_bps=0.0,
            long_quantile=0.5,
            short_quantile=0.0,
            leverage=1.0,
            max_position_weight=0.5,
            construction_method="equal_weight",
            long_only=True,
            min_names=2,
        )

        result = run_strategy_backtest(dataset, MeanReversionStrategy(window=2), config)

        self.assertEqual(result.strategy_name, "Mean Reversion")
        self.assertIn("motivation", result.notes)
        self.assertFalse(result.target_weights.empty)

    def test_combination_candidate_selection_prefers_diversifying_positive_signals(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        empty_weights = pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"])
        empty_series = pd.Series(0.0, index=dates)
        results = {
            "Small-Cap Tilt": BacktestResult(
                strategy_name="Small-Cap Tilt",
                target_weights=empty_weights,
                gross_returns=empty_series,
                net_returns=empty_series,
                turnover=empty_series,
                transaction_costs=empty_series,
                portfolio_value=pd.Series([1_000_000.0, 1_010_000.0, 1_020_000.0], index=dates),
                metrics={"Sharpe Ratio": 0.5, "Annualized Return": 0.04},
            ),
            "Value Composite": BacktestResult(
                strategy_name="Value Composite",
                target_weights=empty_weights,
                gross_returns=empty_series,
                net_returns=empty_series,
                turnover=empty_series,
                transaction_costs=empty_series,
                portfolio_value=pd.Series([1_000_000.0, 1_006_000.0, 1_012_000.0], index=dates),
                metrics={"Sharpe Ratio": 0.3, "Annualized Return": 0.02},
            ),
            "EPS Revision": BacktestResult(
                strategy_name="EPS Revision",
                target_weights=empty_weights,
                gross_returns=empty_series,
                net_returns=empty_series,
                turnover=empty_series,
                transaction_costs=empty_series,
                portfolio_value=pd.Series([1_000_000.0, 1_004_000.0, 1_008_000.0], index=dates),
                metrics={"Sharpe Ratio": 0.25, "Annualized Return": 0.015},
            ),
            "Sector-Neutral Dividend Yield": BacktestResult(
                strategy_name="Sector-Neutral Dividend Yield",
                target_weights=empty_weights,
                gross_returns=empty_series,
                net_returns=empty_series,
                turnover=empty_series,
                transaction_costs=empty_series,
                portfolio_value=pd.Series([1_000_000.0, 1_003_000.0, 1_006_000.0], index=dates),
                metrics={"Sharpe Ratio": 0.2, "Annualized Return": 0.01},
            ),
        }
        correlation = pd.DataFrame(
            {
                "Small-Cap Tilt": [1.0, 0.7, -0.2, 0.0],
                "Value Composite": [0.7, 1.0, -0.1, 0.2],
                "EPS Revision": [-0.2, -0.1, 1.0, -0.05],
                "Sector-Neutral Dividend Yield": [0.0, 0.2, -0.05, 1.0],
            },
            index=[
                "Small-Cap Tilt",
                "Value Composite",
                "EPS Revision",
                "Sector-Neutral Dividend Yield",
            ],
        )

        selected = select_combination_candidates(
            results=results,
            signal_correlation=correlation,
            max_strategies=3,
            correlation_threshold=0.6,
        )

        self.assertEqual(
            selected,
            ["Small-Cap Tilt", "EPS Revision", "Sector-Neutral Dividend Yield"],
        )

    def test_combine_strategy_portfolios_equal_weight(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        first_weights = pd.DataFrame({"AAA": [0.2, 0.2], "BBB": [-0.2, -0.2]}, index=dates)
        second_weights = pd.DataFrame({"AAA": [0.0, 0.1], "BBB": [0.0, -0.1]}, index=dates)
        empty_series = pd.Series(0.0, index=dates)
        results = {
            "One": BacktestResult(
                strategy_name="One",
                target_weights=first_weights,
                gross_returns=empty_series,
                net_returns=empty_series,
                turnover=empty_series,
                transaction_costs=empty_series,
                portfolio_value=pd.Series([1_000_000.0, 1_001_000.0], index=dates),
                metrics={},
            ),
            "Two": BacktestResult(
                strategy_name="Two",
                target_weights=second_weights,
                gross_returns=empty_series,
                net_returns=empty_series,
                turnover=empty_series,
                transaction_costs=empty_series,
                portfolio_value=pd.Series([1_000_000.0, 1_002_000.0], index=dates),
                metrics={},
            ),
        }

        combined = combine_strategy_portfolios(results, ["One", "Two"])

        self.assertAlmostEqual(combined.loc[dates[0], "AAA"], 0.1)
        self.assertAlmostEqual(combined.loc[dates[1], "AAA"], 0.15)
        self.assertAlmostEqual(combined.loc[dates[1], "BBB"], -0.15)


if __name__ == "__main__":
    unittest.main()
