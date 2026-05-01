from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .research_data import ResearchDataset


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    rebalance_frequency: str = "ME"
    signal_lag: int = 1
    transaction_cost_bps: float = 10.0
    long_quantile: float = 0.1
    short_quantile: float = 0.1
    leverage: float = 1.0
    max_position_weight: float = 0.05
    construction_method: str = "equal_weight"
    long_only: bool = False
    volatility_lookback: int = 63
    mean_variance_lookback: int = 126
    risk_free_rate: float = 0.0
    min_names: int = 10
    strategy_allocation_method: str = "equal_weight"
    max_combined_strategies: int = 3
    combination_correlation_threshold: float = 0.6


@dataclass
class BacktestResult:
    strategy_name: str
    target_weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series
    portfolio_value: pd.Series
    metrics: Dict[str, float]
    lag_table: Optional[pd.DataFrame] = None
    event_study: Optional[pd.DataFrame] = None
    notes: Optional[Dict[str, str]] = None
    survivorship_audit: Optional[Dict[str, Any]] = None


def _get_rebalance_dates(index: pd.DatetimeIndex, frequency: Optional[str]) -> pd.DatetimeIndex:
    if frequency in (None, "", "D", "B"):
        return pd.DatetimeIndex(index)

    calendar = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(calendar.resample(frequency).last().dropna().tolist())


def _bucket_count(n_names: int, quantile: float, enabled: bool = True) -> int:
    if not enabled or n_names <= 0:
        return 0
    if float(quantile) <= 0.0:
        return 0
    return max(int(n_names * float(quantile)), 1)


def _gross_budgets(
    leverage: float,
    has_long: bool,
    has_short: bool,
    long_quantile: float = 0.0,
    short_quantile: float = 0.0,
) -> tuple[float, float]:
    """Split the gross budget between long and short legs.

    When both legs are active, allocate proportionally to the requested
    quantile sizes rather than splitting 50/50. The 50/50 split was creating
    a cliff: turning on even a tiny short_quantile (e.g. 0.05) cut the long
    leg from 100% to 50% of capital, which often gutted the long alpha
    faster than the small short leg could compensate. Proportional
    allocation degrades smoothly: with long=0.20 and short=0.05 the long
    leg keeps 80% of capital; with long=short=0.20 we converge to the
    canonical 50/50 long-short construction.
    """
    if has_long and has_short:
        long_q = max(float(long_quantile), 0.0)
        short_q = max(float(short_quantile), 0.0)
        denom = long_q + short_q
        if denom <= 0.0:
            # Both legs flagged but quantiles are zero — fall back to even.
            return leverage / 2.0, leverage / 2.0
        return leverage * (long_q / denom), leverage * (short_q / denom)
    if has_long:
        return leverage, 0.0
    if has_short:
        return 0.0, leverage
    return 0.0, 0.0


def _scale_side(weights: pd.Series, target_gross: float, max_position_weight: float, sign: float) -> pd.Series:
    if weights.empty or target_gross <= 0:
        return pd.Series(dtype=float)

    scaled = weights.abs()
    if float(scaled.sum()) == 0.0:
        scaled = pd.Series(1.0, index=weights.index, dtype=float)

    scaled = scaled / scaled.sum()
    scaled = scaled * target_gross

    if max_position_weight > 0:
        scaled = scaled.clip(upper=max_position_weight)

    if float(scaled.sum()) > 0:
        rescale = target_gross / scaled.sum()
        if max_position_weight > 0:
            rescale = min(rescale, 1.0)
        scaled = scaled * rescale

    return scaled * sign


def _mean_variance_weights(
    signals: pd.Series,
    returns_history: pd.DataFrame,
    config: BacktestConfig,
) -> pd.Series:
    if returns_history.empty:
        return pd.Series(dtype=float)

    covariance = returns_history.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if covariance.empty:
        return pd.Series(dtype=float)

    mu = signals.reindex(covariance.index).fillna(0.0).to_numpy(dtype=float)
    covariance_matrix = covariance.to_numpy(dtype=float)
    try:
        raw_weights = np.linalg.pinv(covariance_matrix) @ mu
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float)

    weights = pd.Series(raw_weights, index=covariance.index, dtype=float)
    if config.long_only:
        weights = weights.clip(lower=0.0)
    return weights


def merge_backtest_config(base_config: BacktestConfig, overrides: Optional[Dict[str, Any]] = None) -> BacktestConfig:
    if not overrides:
        return base_config
    return replace(base_config, **overrides)


def build_target_weights(
    signal_scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    scores = signal_scores.reindex(asset_returns.index).sort_index()
    volatility = asset_returns.rolling(config.volatility_lookback).std().replace(0, np.nan)
    rebalance_dates = _get_rebalance_dates(asset_returns.index, config.rebalance_frequency)
    rebalance_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=asset_returns.columns)

    for date in rebalance_dates:
        row = scores.loc[date].dropna()
        if len(row) < config.min_names:
            continue

        long_count = _bucket_count(len(row), config.long_quantile)
        short_count = _bucket_count(len(row), config.short_quantile, enabled=not config.long_only)

        long_names = row.nlargest(long_count).index if long_count > 0 else pd.Index([])
        short_names = row.nsmallest(short_count).index if short_count > 0 else pd.Index([])
        short_names = short_names.difference(long_names)
        selected_names = long_names.union(short_names)

        if len(selected_names) == 0:
            continue

        if config.construction_method == "mean_variance":
            history = asset_returns.loc[:date, selected_names].tail(config.mean_variance_lookback)
            history = history.dropna(axis=1, thresh=max(config.mean_variance_lookback // 2, 10)).fillna(0.0)
            selected_signals = row.reindex(history.columns)
            raw = _mean_variance_weights(selected_signals, history, config)
            long_signal = raw.reindex(long_names).dropna()
            short_signal = raw.reindex(short_names).dropna()

            if short_names.empty:
                long_signal = long_signal.clip(lower=0.0)
            else:
                long_signal = long_signal[long_signal > 0]

            if long_names.empty:
                short_signal = short_signal.clip(upper=0.0)
            else:
                short_signal = short_signal[short_signal < 0]
            short_signal = short_signal.abs()
        else:
            if config.construction_method == "inverse_vol":
                risk_estimate = (1.0 / volatility.loc[date].replace(0, np.nan)).reindex(selected_names)
            else:
                risk_estimate = pd.Series(1.0, index=selected_names, dtype=float)

            long_signal = risk_estimate.reindex(long_names).dropna()
            short_signal = risk_estimate.reindex(short_names).dropna()

        row_weights = pd.Series(0.0, index=asset_returns.columns, dtype=float)
        long_budget, short_budget = _gross_budgets(
            config.leverage,
            has_long=not long_signal.empty,
            has_short=(not config.long_only) and (not short_signal.empty),
            long_quantile=config.long_quantile,
            short_quantile=config.short_quantile,
        )

        row_weights.loc[long_signal.index] = _scale_side(
            long_signal,
            target_gross=long_budget,
            max_position_weight=config.max_position_weight,
            sign=1.0,
        )
        row_weights.loc[short_signal.index] = _scale_side(
            short_signal,
            target_gross=short_budget,
            max_position_weight=config.max_position_weight,
            sign=-1.0,
        )

        rebalance_weights.loc[date] = row_weights

    return rebalance_weights.reindex(asset_returns.index).ffill().fillna(0.0)


def _drift_weights(weights: pd.Series, returns_row: pd.Series) -> pd.Series:
    clean_returns = returns_row.fillna(0.0)
    gross_return = float((weights * clean_returns).sum())
    denominator = 1.0 + gross_return
    if denominator == 0.0:
        return pd.Series(0.0, index=weights.index, dtype=float)

    drifted = weights * (1.0 + clean_returns)
    return drifted / denominator


def calculate_performance_metrics(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    transaction_costs: pd.Series,
    portfolio_value: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    returns = portfolio_returns.fillna(0.0)
    if returns.empty:
        return {}

    annualization = 252
    excess_returns = returns - (risk_free_rate / annualization)
    downside = returns[returns < 0]
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1.0

    annualized_return = cumulative.iloc[-1] ** (annualization / max(len(returns), 1)) - 1.0
    annualized_vol = returns.std(ddof=0) * np.sqrt(annualization)
    downside_vol = downside.std(ddof=0) * np.sqrt(annualization) if not downside.empty else 0.0

    metrics = {
        "Avg Daily Return": returns.mean(),
        "Cumulative Return": cumulative.iloc[-1] - 1.0,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": excess_returns.mean() / returns.std(ddof=0) * np.sqrt(annualization)
        if returns.std(ddof=0) > 0
        else np.nan,
        "Downside Volatility": downside_vol,
        "Sortino Ratio": excess_returns.mean() / downside.std(ddof=0) * np.sqrt(annualization)
        if not downside.empty and downside.std(ddof=0) > 0
        else np.nan,
        "Max Drawdown": drawdown.min(),
        "Average Turnover": turnover.mean(),
        "Annualized Turnover": turnover.mean() * annualization,
        "Average Transaction Cost": transaction_costs.mean(),
        "Annualized Transaction Cost Drag": transaction_costs.mean() * annualization,
        "Ending Portfolio Value": portfolio_value.iloc[-1],
    }

    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        aligned.columns = ["strategy", "benchmark"]
        if not aligned.empty:
            diff = aligned["strategy"] - aligned["benchmark"]
            metrics["Benchmark Correlation"] = aligned.corr().iloc[0, 1]
            metrics["Tracking Error"] = diff.std(ddof=0) * np.sqrt(annualization)
            metrics["Information Ratio"] = diff.mean() / diff.std(ddof=0) * np.sqrt(annualization) if diff.std(ddof=0) > 0 else np.nan

    return metrics


def run_survivorship_audit(
    prices: pd.DataFrame,
    rebalance_frequency: Optional[str] = "ME",
    held_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Point-in-time audit: what the backtester can and cannot see.

    The input universe is *today's* survivors from the data pull. Any company
    that was delisted, acquired, or went bankrupt before the pull date is
    structurally missing, so any Sharpe / return number is biased upward.
    This function quantifies what the run covered and returns a structured
    payload for display.

    Academic anchor for the magnitude of the bias:
      * Brown, Goetzmann, Ibbotson, Ross (1992): ~1%/yr on industry indices.
      * Malkiel (1995): ~1.5%/yr on mutual funds.
      * Carhart et al. (2002): 0.57–0.92%/yr on fund panels.
    """
    if prices.empty:
        return {}

    window_start = prices.index[0]
    window_end = prices.index[-1]
    universe_size = int(prices.shape[1])

    first_valid = prices.apply(lambda col: col.first_valid_index())
    last_valid = prices.apply(lambda col: col.last_valid_index())

    inception_biased = int((first_valid > window_start).sum())
    delisted_within_window = int((last_valid < window_end).sum())

    held = set(held_columns) if held_columns is not None else set()
    traded_first = {tic: ts for tic, ts in first_valid.items() if tic in held and pd.notna(ts)}
    median_inception = pd.Series(list(traded_first.values())).median() if traded_first else pd.NaT

    active_per_day = prices.notna().sum(axis=1)
    rebalance_dates = _get_rebalance_dates(prices.index, rebalance_frequency)
    active_at_rebalance = active_per_day.reindex(rebalance_dates).dropna().astype(int)

    window_years = max((window_end - window_start).days / 365.25, 1e-6)
    bias_low = 0.5
    bias_high = 1.5

    return {
        "universe_size": universe_size,
        "window_start": str(window_start.date()),
        "window_end": str(window_end.date()),
        "window_years": round(window_years, 2),
        "active_at_start": int(active_per_day.iloc[0]),
        "active_at_end": int(active_per_day.iloc[-1]),
        "min_active_in_window": int(active_per_day.min()),
        "max_active_in_window": int(active_per_day.max()),
        "inception_biased_count": inception_biased,
        "delisted_within_window_count": delisted_within_window,
        "median_inception_for_held": str(median_inception.date()) if pd.notna(median_inception) else None,
        "active_timeseries": {
            "dates": [str(d.date()) for d in active_at_rebalance.index],
            "counts": [int(v) for v in active_at_rebalance.values],
        },
        "expected_annual_upward_bias_pct": {"low": bias_low, "high": bias_high},
        "structural_bias_note": (
            f"The {universe_size}-ticker universe is the set still trading at the parquet pull date. "
            f"Failed/acquired/delisted names are absent from the panel entirely, so realized Sharpe "
            f"and annualized return are upward-biased. Published estimates place the magnitude at "
            f"roughly {bias_low:.1f}–{bias_high:.1f}% annualized for US equity survivor-only panels."
        ),
    }


def run_weight_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig,
    benchmark_returns: Optional[pd.Series] = None,
    strategy_name: str = "Strategy",
) -> BacktestResult:
    prices = prices.sort_index().sort_index(axis=1)
    asset_returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = target_weights.reindex(asset_returns.index).ffill().fillna(0.0)
    executed_weights = weights.shift(config.signal_lag).fillna(0.0)

    turnover = pd.Series(0.0, index=executed_weights.index, dtype=float)
    transaction_costs = pd.Series(0.0, index=executed_weights.index, dtype=float)
    actual_weights = pd.DataFrame(
        0.0, index=executed_weights.index, columns=executed_weights.columns, dtype=float,
    )

    current_weights = pd.Series(0.0, index=executed_weights.columns, dtype=float)
    prev_target = None
    target_values = executed_weights.values
    column_index = executed_weights.columns
    asset_return_values = asset_returns.reindex(index=executed_weights.index, columns=column_index).fillna(0.0).values

    for i, date in enumerate(executed_weights.index):
        target = pd.Series(target_values[i], index=column_index, dtype=float)

        # Detect an actual rebalance: first day, or the target weights changed.
        # Between rebalances we simply drift — no trades, no turnover, no cost.
        is_rebalance = (
            prev_target is None
            or not np.allclose(target.values, prev_target, equal_nan=False)
        )

        if is_rebalance:
            turnover.loc[date] = float((target - current_weights).abs().sum())
            current_weights = target.copy()
        else:
            turnover.loc[date] = 0.0

        transaction_costs.loc[date] = turnover.loc[date] * config.transaction_cost_bps / 10_000.0

        # Record what we actually hold at the start of this day's return window.
        actual_weights.iloc[i] = current_weights.values

        # Drift the held weights forward one day's returns for tomorrow.
        current_weights = _drift_weights(current_weights, pd.Series(asset_return_values[i], index=column_index))
        prev_target = target.values

    gross_returns = (actual_weights * asset_returns.reindex(index=actual_weights.index, columns=column_index).fillna(0.0)).sum(axis=1).fillna(0.0)
    net_returns = gross_returns - transaction_costs
    portfolio_value = (1.0 + net_returns).cumprod() * config.initial_cash
    metrics = calculate_performance_metrics(
        portfolio_returns=net_returns,
        turnover=turnover,
        transaction_costs=transaction_costs,
        portfolio_value=portfolio_value,
        benchmark_returns=benchmark_returns,
        risk_free_rate=config.risk_free_rate,
    )

    held_tickers = actual_weights.columns[(actual_weights.abs().sum(axis=0) > 0)].tolist()
    survivorship_audit = run_survivorship_audit(
        prices=prices,
        rebalance_frequency=config.rebalance_frequency,
        held_columns=held_tickers,
    )

    return BacktestResult(
        strategy_name=strategy_name,
        target_weights=weights,
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
        transaction_costs=transaction_costs,
        portfolio_value=portfolio_value,
        metrics=metrics,
        survivorship_audit=survivorship_audit,
    )


def _daily_cross_sectional_ic(scores: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    aligned_scores = scores.reindex(forward_returns.index)
    common_mask = aligned_scores.notna() & forward_returns.notna()
    valid_counts = common_mask.sum(axis=1)

    ranked_scores = aligned_scores.where(common_mask).rank(axis=1, method="average")
    ranked_returns = forward_returns.where(common_mask).rank(axis=1, method="average")

    centered_scores = ranked_scores.sub(ranked_scores.mean(axis=1), axis=0)
    centered_returns = ranked_returns.sub(ranked_returns.mean(axis=1), axis=0)

    numerator = (centered_scores * centered_returns).sum(axis=1, skipna=True)
    denominator = (
        centered_scores.pow(2).sum(axis=1, skipna=True)
        * centered_returns.pow(2).sum(axis=1, skipna=True)
    ) ** 0.5

    correlation = numerator.div(denominator.replace(0, np.nan))
    correlation[valid_counts < 3] = np.nan
    return correlation


def signal_long_short_returns(
    signal_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    long_only: bool = False,
) -> pd.Series:
    aligned_scores = signal_scores.reindex(forward_returns.index)
    common_mask = aligned_scores.notna() & forward_returns.notna()
    valid_counts = common_mask.sum(axis=1)

    rank_percentiles = aligned_scores.where(common_mask).rank(axis=1, method="average", pct=True)
    has_long = float(long_quantile) > 0.0
    has_short = (not long_only) and float(short_quantile) > 0.0

    if has_long:
        long_mask = rank_percentiles >= (1.0 - long_quantile)
        long_leg = forward_returns.where(long_mask).mean(axis=1, skipna=True)
    else:
        long_leg = pd.Series(0.0, index=forward_returns.index, dtype=float)

    if has_short:
        short_mask = rank_percentiles <= short_quantile
        short_leg = forward_returns.where(short_mask).mean(axis=1, skipna=True)
    else:
        short_leg = pd.Series(0.0, index=forward_returns.index, dtype=float)

    if has_long or has_short:
        strategy_returns = long_leg - short_leg
    else:
        strategy_returns = pd.Series(np.nan, index=forward_returns.index, dtype=float)

    strategy_returns[valid_counts < 5] = np.nan
    return strategy_returns


def signal_lag_table(
    signal_scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    lags: Iterable[int] = range(-3, 6),
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    long_only: bool = False,
) -> pd.DataFrame:
    forward_returns = asset_returns.shift(-1)
    rows = []

    for lag in lags:
        lagged_signal = signal_scores.shift(lag)
        lag_returns = signal_long_short_returns(
            lagged_signal,
            forward_returns,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            long_only=long_only,
        ).dropna()
        ic_series = _daily_cross_sectional_ic(lagged_signal, forward_returns).dropna()

        sharpe = lag_returns.mean() / lag_returns.std(ddof=0) * np.sqrt(252) if len(lag_returns) > 1 and lag_returns.std(ddof=0) > 0 else np.nan
        rows.append(
            {
                "lag": lag,
                "sharpe_ratio": sharpe,
                "avg_forward_return": lag_returns.mean() if not lag_returns.empty else np.nan,
                "avg_spearman_ic": ic_series.mean() if not ic_series.empty else np.nan,
                "observations": len(lag_returns),
            }
        )

    return pd.DataFrame(rows).set_index("lag")


def signal_correlation_matrix(
    strategy_scores: Dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    long_only: bool = False,
) -> pd.DataFrame:
    signal_returns = {}
    forward_returns = asset_returns.shift(-1)
    for name, scores in strategy_scores.items():
        signal_returns[name] = signal_long_short_returns(
            scores,
            forward_returns,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            long_only=long_only,
        )

    if not signal_returns:
        return pd.DataFrame()

    return pd.DataFrame(signal_returns).corr()


def combine_signal_frames(strategy_scores: Dict[str, pd.DataFrame], weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    combined_sum = None
    combined_count = None

    for name, frame in strategy_scores.items():
        signal_weight = 1.0 if weights is None else weights.get(name, 0.0)
        weighted_frame = frame * signal_weight
        valid = frame.notna().astype(float) * abs(signal_weight)

        combined_sum = weighted_frame if combined_sum is None else combined_sum.add(weighted_frame, fill_value=0.0)
        combined_count = valid if combined_count is None else combined_count.add(valid, fill_value=0.0)

    if combined_sum is None or combined_count is None:
        return pd.DataFrame()

    return combined_sum.div(combined_count.replace(0, np.nan))


def estimate_combination_weights(
    results: Dict[str, BacktestResult],
    signal_correlation: pd.DataFrame,
) -> Dict[str, float]:
    if not results:
        return {}

    raw_weights: Dict[str, float] = {}
    candidate_names = [name for name in results.keys() if name != "Combined Multi-Signal"]

    for name in candidate_names:
        result = results[name]
        lag_ic = 0.0
        if result.lag_table is not None and 0 in result.lag_table.index:
            lag_ic = float(result.lag_table.loc[0, "avg_spearman_ic"])
            if np.isnan(lag_ic):
                lag_ic = 0.0

        sharpe = float(result.metrics.get("Sharpe Ratio", 0.0) or 0.0)
        if np.isnan(sharpe):
            sharpe = 0.0

        crowding_penalty = 1.0
        if not signal_correlation.empty and name in signal_correlation.index:
            peer_corr = signal_correlation.loc[name].drop(labels=[name], errors="ignore").abs()
            if not peer_corr.empty:
                crowding_penalty = max(1.0 - peer_corr.mean(), 0.1)

        score = max(lag_ic, 0.0) + max(sharpe, 0.0) / 4.0
        raw_weights[name] = max(score * crowding_penalty, 0.0)

    total = sum(raw_weights.values())
    if total <= 0:
        equal_weight = 1.0 / max(len(candidate_names), 1)
        return {name: equal_weight for name in candidate_names}

    return {name: value / total for name, value in raw_weights.items()}


def select_combination_candidates(
    results: Dict[str, BacktestResult],
    signal_correlation: pd.DataFrame,
    eligible_names: Optional[Sequence[str]] = None,
    max_strategies: int = 3,
    correlation_threshold: float = 0.6,
) -> list[str]:
    candidates = list(eligible_names) if eligible_names is not None else list(results.keys())
    ranked = []

    for name in candidates:
        result = results.get(name)
        if result is None:
            continue

        sharpe = float(result.metrics.get("Sharpe Ratio", 0.0) or 0.0)
        annualized_return = float(result.metrics.get("Annualized Return", 0.0) or 0.0)
        if np.isnan(sharpe) or np.isnan(annualized_return) or sharpe <= 0.0 or annualized_return <= 0.0:
            continue

        ranked.append((name, sharpe, annualized_return))

    ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)

    selected: list[str] = []
    for name, _, _ in ranked:
        if signal_correlation.empty or name not in signal_correlation.index:
            selected.append(name)
        else:
            is_diversifying = True
            for existing in selected:
                if existing not in signal_correlation.columns:
                    continue
                corr = signal_correlation.loc[name, existing]
                if pd.notna(corr) and abs(float(corr)) > correlation_threshold:
                    is_diversifying = False
                    break
            if is_diversifying:
                selected.append(name)

        if len(selected) >= max_strategies:
            break

    if selected:
        return selected

    return [name for name, _, _ in ranked[:max_strategies]]


def combine_strategy_portfolios(
    results: Dict[str, BacktestResult],
    selected_names: Sequence[str],
    method: str = "equal_weight",
) -> pd.DataFrame:
    valid_names = [name for name in selected_names if name in results]
    if not valid_names:
        return pd.DataFrame()

    if method == "inverse_vol":
        strategy_returns = pd.DataFrame(
            {name: results[name].net_returns for name in valid_names}
        ).fillna(0.0)
        rolling_vol = strategy_returns.rolling(126).std().shift(1)
        inverse_vol = 1.0 / rolling_vol.replace(0, np.nan)
        allocations = inverse_vol.div(inverse_vol.sum(axis=1), axis=0)
        allocations = allocations.fillna(1.0 / len(valid_names))

        combined_weights = None
        for name in valid_names:
            contribution = results[name].target_weights.mul(allocations[name], axis=0)
            combined_weights = contribution if combined_weights is None else combined_weights.add(contribution, fill_value=0.0)
        return combined_weights.fillna(0.0)

    combined_weights = results[valid_names[0]].target_weights.copy()
    for name in valid_names[1:]:
        combined_weights = combined_weights.add(results[name].target_weights, fill_value=0.0)
    return combined_weights.div(len(valid_names)).fillna(0.0)


def build_event_study(
    asset_returns: pd.DataFrame,
    events: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    event_type: Optional[str] = None,
    window: int = 44,
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()

    residual_returns = asset_returns.copy()
    if benchmark_returns is not None and not benchmark_returns.empty:
        residual_returns = residual_returns.sub(benchmark_returns.reindex(asset_returns.index).fillna(0.0), axis=0)

    event_frame = events.copy()
    if event_type is not None:
        event_frame = event_frame[event_frame["event_type"] == event_type]

    if event_frame.empty:
        return pd.DataFrame()

    relative_index = pd.Index(range(-window, window + 1), name="relative_day")
    event_paths = []

    for event in event_frame.itertuples(index=False):
        ticker = getattr(event, "ticker")
        event_date = pd.Timestamp(getattr(event, "event_date"))

        if ticker not in residual_returns.columns:
            continue

        position = residual_returns.index.searchsorted(event_date)
        if position >= len(residual_returns.index):
            continue

        start = position - window
        end = position + window
        if start < 0 or end >= len(residual_returns.index):
            continue

        window_returns = residual_returns[ticker].iloc[start : end + 1].fillna(0.0)
        cumulative = window_returns.cumsum()
        cumulative.index = relative_index
        event_paths.append(cumulative)

    if not event_paths:
        return pd.DataFrame()

    stacked = pd.concat(event_paths, axis=1)
    summary = pd.DataFrame(index=relative_index)
    summary["median"] = stacked.median(axis=1)
    summary["p10"] = stacked.quantile(0.10, axis=1)
    summary["p25"] = stacked.quantile(0.25, axis=1)
    summary["p75"] = stacked.quantile(0.75, axis=1)
    summary["p90"] = stacked.quantile(0.90, axis=1)
    summary["event_count"] = stacked.notna().sum(axis=1)
    return summary


def run_strategy_suite(
    dataset: ResearchDataset,
    strategies: Sequence[Any],
    config: BacktestConfig,
) -> tuple[Dict[str, BacktestResult], pd.DataFrame]:
    results: Dict[str, BacktestResult] = {}
    signal_frames: Dict[str, pd.DataFrame] = {}
    portfolio_candidates: list[str] = []

    for strategy in strategies:
        strategy_output = strategy.generate(dataset)
        signal_frames[strategy_output.name] = strategy_output.scores
        strategy_config = merge_backtest_config(config, strategy_output.backtest_overrides)

        target_weights = build_target_weights(strategy_output.scores, dataset.returns, strategy_config)
        result = run_weight_backtest(
            prices=dataset.prices,
            target_weights=target_weights,
            config=strategy_config,
            benchmark_returns=dataset.benchmark_returns,
            strategy_name=strategy_output.name,
        )
        result.lag_table = signal_lag_table(
            strategy_output.scores,
            dataset.returns,
            long_quantile=strategy_config.long_quantile,
            short_quantile=strategy_config.short_quantile,
            long_only=strategy_config.long_only,
        )
        if strategy_output.uses_event_data and dataset.events is not None:
            result.event_study = build_event_study(
                asset_returns=dataset.returns,
                events=dataset.events,
                benchmark_returns=dataset.benchmark_returns,
                event_type=strategy_output.event_type,
            )
        result.notes = {
            "motivation": strategy_output.motivation,
            "economic_rationale": strategy_output.economic_rationale,
            "why_it_works": strategy_output.why_it_works,
            "why_it_fails": strategy_output.why_it_fails,
            "rebalance_frequency": strategy_config.rebalance_frequency,
            "construction_method": strategy_config.construction_method,
        }
        results[strategy_output.name] = result
        if strategy_output.combine_in_portfolio:
            portfolio_candidates.append(strategy_output.name)

    correlation = signal_correlation_matrix(
        signal_frames,
        dataset.returns,
        long_quantile=config.long_quantile,
        short_quantile=config.short_quantile,
        long_only=config.long_only,
    )

    selected_candidates = select_combination_candidates(
        results=results,
        signal_correlation=correlation,
        eligible_names=portfolio_candidates,
        max_strategies=config.max_combined_strategies,
        correlation_threshold=config.combination_correlation_threshold,
    )
    combined_weights = combine_strategy_portfolios(
        results=results,
        selected_names=selected_candidates,
        method=config.strategy_allocation_method,
    )
    if not combined_weights.empty:
        combined_result = run_weight_backtest(
            prices=dataset.prices,
            target_weights=combined_weights,
            config=config,
            benchmark_returns=dataset.benchmark_returns,
            strategy_name="Combined Multi-Signal",
        )
        combined_result.notes = {
            "combination_method": f"{config.strategy_allocation_method}_portfolio_blend",
            "selected_strategies": ", ".join(selected_candidates),
        }
        results[combined_result.strategy_name] = combined_result

    return results, correlation
