"""Simulator core — wraps the backtesting engine for interactive use.

Pre-computes the expensive selection-sleeve backtests once at startup over the
full available date range, then slices and recombines cheaply per request.

Fixed selection-sleeve parameters match run_book.py (signal_lag=1, 10bps t-cost,
top-quintile long-only, inverse-vol construction, 8% max position, monthly
rebalance). User-tunable knobs are the cheap ones: date window, target vol,
max leverage, SMA trend window, risk-free rate, combination method, and which
sleeves to include.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtester.research_data import load_wharton_research_dataset
from backtester.research_backtester import (
    BacktestConfig,
    build_target_weights,
    run_weight_backtest,
)
from backtester.multi_strategy import combine_strategy_returns
from strategies.research_strategies import (
    CrossSectionalMomentumStrategy,
    LowVolatilityStrategy,
    SmallCapTiltStrategy,
)

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "backtester" / "WhartonDataSource.parquet"

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "ready": False,
    "loading": False,
    "message": "idle",
    "error": None,
    "dataset": None,
    "selection_returns": {},  # name -> daily Series (full history)
    "benchmark": None,
    "bench_index": None,
    "date_min": None,
    "date_max": None,
    "tickers": 0,
}

SELECTION_STRATEGIES = [
    ("Cross-Sectional Momentum", lambda: CrossSectionalMomentumStrategy(lookback_days=126, skip_days=21)),
    ("Low Volatility", lambda: LowVolatilityStrategy(window=126)),
    ("Small-Cap Tilt", lambda: SmallCapTiltStrategy()),
]

SELECTION_CONFIG = BacktestConfig(
    rebalance_frequency="ME",
    signal_lag=1,
    transaction_cost_bps=10.0,
    long_quantile=0.20,
    short_quantile=0.0,
    leverage=1.0,
    max_position_weight=0.08,
    construction_method="inverse_vol",
    long_only=True,
    min_names=20,
)


# ---------------------------------------------------------------------------
# Sleeve primitives (mirror run_book.py)
# ---------------------------------------------------------------------------
def vol_managed_returns(
    raw_returns: pd.Series,
    target_vol: float,
    vol_window: int = 63,
    min_leverage: float = 0.3,
    max_leverage: float = 2.0,
    transaction_cost_bps: float = 2.0,
) -> pd.Series:
    realized_vol = raw_returns.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
    leverage = (target_vol / realized_vol.replace(0, np.nan)).clip(lower=min_leverage, upper=max_leverage)
    leverage = leverage.shift(1).fillna(1.0)
    scaled = raw_returns * leverage
    tcost = leverage.diff().abs().fillna(0) * transaction_cost_bps / 10_000.0
    return scaled - tcost


def trend_filtered_returns(
    index_level: pd.Series,
    raw_returns: pd.Series,
    risk_free_annual: float,
    sma_window: int,
    transaction_cost_bps: float = 2.0,
) -> pd.Series:
    sma = index_level.rolling(sma_window, min_periods=sma_window // 2).mean()
    in_market = (index_level > sma).shift(1).fillna(False).astype(float)
    out_returns = pd.Series(risk_free_annual / 252, index=raw_returns.index)
    blended = in_market * raw_returns + (1 - in_market) * out_returns
    switches = in_market.diff().abs().fillna(0)
    return blended - switches * transaction_cost_bps / 10_000.0


# ---------------------------------------------------------------------------
# Warmup — load dataset and run selection sleeves once
# ---------------------------------------------------------------------------
def warmup(start: str = "2000-01-01", end: str = "2025-12-31") -> None:
    with _LOCK:
        if _STATE["loading"] or _STATE["ready"]:
            return
        _STATE["loading"] = True
        _STATE["message"] = "loading Wharton dataset"
        _STATE["error"] = None

    try:
        dataset = load_wharton_research_dataset(
            file_path=str(DATA_PATH), start_date=start, end_date=end,
        )
        with _LOCK:
            _STATE["message"] = f"dataset loaded ({len(dataset.tickers)} tickers); precomputing selection sleeves"

        selection_returns: Dict[str, pd.Series] = {}
        for label, builder in SELECTION_STRATEGIES:
            strat = builder()
            scores = strat.generate(dataset).scores
            weights = build_target_weights(scores, dataset.returns, SELECTION_CONFIG)
            result = run_weight_backtest(
                prices=dataset.prices,
                target_weights=weights,
                config=SELECTION_CONFIG,
                benchmark_returns=dataset.benchmark_returns,
                strategy_name=strat.name,
            )
            selection_returns[label] = result.net_returns
            with _LOCK:
                _STATE["message"] = f"precomputed {label}"

        benchmark = dataset.benchmark_returns
        bench_index = (1 + benchmark.fillna(0)).cumprod()

        with _LOCK:
            _STATE.update({
                "ready": True,
                "loading": False,
                "message": "ready",
                "dataset": dataset,
                "selection_returns": selection_returns,
                "benchmark": benchmark,
                "bench_index": bench_index,
                "date_min": str(dataset.prices.index.min().date()),
                "date_max": str(dataset.prices.index.max().date()),
                "tickers": len(dataset.tickers),
            })
    except Exception as exc:
        with _LOCK:
            _STATE["loading"] = False
            _STATE["ready"] = False
            _STATE["error"] = str(exc)
            _STATE["message"] = f"error: {exc}"
        raise


def status() -> Dict[str, Any]:
    with _LOCK:
        return {
            "ready": _STATE["ready"],
            "loading": _STATE["loading"],
            "message": _STATE["message"],
            "error": _STATE["error"],
            "date_min": _STATE["date_min"],
            "date_max": _STATE["date_max"],
            "tickers": _STATE["tickers"],
            "selection_names": list(_STATE["selection_returns"].keys()),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _annualize(returns: pd.Series, risk_free: float = 0.0) -> Dict[str, float]:
    clean = returns.dropna()
    if len(clean) < 2:
        return {}
    cum = (1 + clean).cumprod()
    n = len(clean)
    ann_ret = cum.iloc[-1] ** (252 / n) - 1
    std = clean.std(ddof=0)
    sharpe = clean.mean() / std * np.sqrt(252) if std > 0 else 0.0
    downside = clean[clean < 0]
    sortino = (
        clean.mean() / downside.std(ddof=0) * np.sqrt(252)
        if len(downside) and downside.std(ddof=0) > 0 else 0.0
    )
    dd_series = cum / cum.cummax() - 1
    dd = dd_series.min()
    return {
        "annualized_return": float(ann_ret),
        "cumulative_return": float(cum.iloc[-1] - 1),
        "annualized_volatility": float(std * np.sqrt(252)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(ann_ret / abs(dd)) if dd else 0.0,
        "max_drawdown": float(dd),
        "win_rate": float((clean > 0).mean()),
        "best_day": float(clean.max()),
        "worst_day": float(clean.min()),
    }


def _rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std().replace(0, np.nan)
    return (mean / std * np.sqrt(252)).fillna(0.0)


def _monthly_returns(returns: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    monthly = (1 + returns).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    return pivot


def _drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    return cum / cum.cummax() - 1


# ---------------------------------------------------------------------------
# Public simulation API
# ---------------------------------------------------------------------------
def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    if not _STATE["ready"]:
        raise RuntimeError("simulator not ready yet — wait for warmup")

    start = params.get("start", _STATE["date_min"])
    end = params.get("end", _STATE["date_max"])
    target_vol = float(params.get("target_vol", 0.16))
    max_leverage = float(params.get("max_leverage", 2.0))
    sma_window = int(params.get("sma_window", 200))
    risk_free = float(params.get("risk_free", 0.04))
    include_vol_managed = bool(params.get("include_vol_managed", True))
    include_trend_filtered = bool(params.get("include_trend_filtered", True))
    included_selections = set(params.get("included_selections", ["Cross-Sectional Momentum", "Low Volatility", "Small-Cap Tilt"]))
    selection_trend_overlay = bool(params.get("selection_trend_overlay", True))
    combo_method = str(params.get("combo_method", "hrp")).lower()  # equal, inverse_vol, risk_parity, hrp
    combined_trend_overlay = bool(params.get("combined_trend_overlay", False))
    rolling_window = int(params.get("rolling_window", 252))

    valid_methods = {"equal": "equal", "inverse_vol": "inverse_vol", "risk_parity": "risk_parity", "hrp": "hrp"}
    if combo_method not in valid_methods:
        raise ValueError(f"unknown combo_method: {combo_method}")

    benchmark_full = _STATE["benchmark"]
    bench_index_full = _STATE["bench_index"]

    window = (pd.Timestamp(start), pd.Timestamp(end))
    benchmark = benchmark_full.loc[window[0]:window[1]]
    bench_index = bench_index_full.loc[window[0]:window[1]]

    if benchmark.empty:
        raise ValueError("date window produced no data")

    sleeves: Dict[str, pd.Series] = {}

    if include_vol_managed:
        vm = vol_managed_returns(
            benchmark, target_vol=target_vol, max_leverage=max_leverage,
        )
        sleeves["Vol-Managed Equity"] = vm

    if include_trend_filtered:
        base_for_trend = sleeves.get("Vol-Managed Equity")
        if base_for_trend is None:
            base_for_trend = vol_managed_returns(benchmark, target_vol=target_vol, max_leverage=max_leverage)
        tf = trend_filtered_returns(
            bench_index, base_for_trend,
            risk_free_annual=risk_free, sma_window=sma_window,
        )
        sleeves["Trend-Filtered Equity"] = tf

    for name, series in _STATE["selection_returns"].items():
        if name not in included_selections:
            continue
        sliced = series.loc[window[0]:window[1]]
        if selection_trend_overlay:
            sliced = trend_filtered_returns(
                bench_index, sliced,
                risk_free_annual=risk_free, sma_window=sma_window,
            )
            label = f"{name} + Trend"
        else:
            label = name
        sleeves[label] = sliced

    if not sleeves:
        raise ValueError("no sleeves enabled")

    sleeve_frame = pd.DataFrame(sleeves).dropna(how="all")
    if sleeve_frame.empty:
        raise ValueError("all sleeves produced empty series")
    sleeve_frame = sleeve_frame.fillna(0.0)

    # Combination
    if combo_method == "equal" or len(sleeves) == 1:
        combined = sleeve_frame.mean(axis=1)
    else:
        combined = combine_strategy_returns(
            {name: ret for name, ret in sleeves.items()},
            method=valid_methods[combo_method], lookback=252, rebalance_freq="ME",
        )
        combined = combined.reindex(sleeve_frame.index).fillna(0.0)

    if combined_trend_overlay:
        combined = trend_filtered_returns(
            bench_index, combined,
            risk_free_annual=risk_free, sma_window=sma_window,
        )

    # Build response payload
    dates = [d.strftime("%Y-%m-%d") for d in sleeve_frame.index]

    def _cum(series: pd.Series) -> List[float]:
        return [float(v) for v in ((1 + series.fillna(0)).cumprod() - 1).values]

    response = {
        "dates": dates,
        "benchmark": {
            "name": "Equal-Weight Universe",
            "cumulative": _cum(benchmark.reindex(sleeve_frame.index).fillna(0)),
            "returns": [float(v) for v in benchmark.reindex(sleeve_frame.index).fillna(0).values],
        },
        "sleeves": {
            name: {
                "cumulative": _cum(series),
                "returns": [float(v) for v in series.values],
            }
            for name, series in sleeve_frame.items()
        },
        "combined": {
            "name": f"Combined Book ({combo_method.upper()}{'+Trend' if combined_trend_overlay else ''})",
            "cumulative": _cum(combined),
            "returns": [float(v) for v in combined.values],
            "drawdown": [float(v) for v in _drawdown_series(combined).values],
            "rolling_sharpe": [float(v) for v in _rolling_sharpe(combined, rolling_window).values],
        },
        "metrics": {
            "Benchmark": _annualize(benchmark, risk_free),
            **{name: _annualize(series, risk_free) for name, series in sleeve_frame.items()},
            "Combined": _annualize(combined, risk_free),
        },
        "correlation": {
            "labels": list(sleeve_frame.columns),
            "matrix": [[float(v) for v in row] for row in sleeve_frame.corr().values],
        },
        "monthly_heatmap": _monthly_heatmap_payload(combined),
        "params_echo": {
            "start": start, "end": end, "target_vol": target_vol,
            "max_leverage": max_leverage, "sma_window": sma_window,
            "risk_free": risk_free, "combo_method": combo_method,
            "selection_trend_overlay": selection_trend_overlay,
            "combined_trend_overlay": combined_trend_overlay,
            "rolling_window": rolling_window,
            "n_sleeves": len(sleeves),
        },
    }
    return response


def _monthly_heatmap_payload(returns: pd.Series) -> Dict[str, Any]:
    pivot = _monthly_returns(returns)
    if pivot.empty:
        return {"years": [], "months": list(range(1, 13)), "values": []}
    return {
        "years": [int(y) for y in pivot.index],
        "months": list(range(1, 13)),
        "values": [[None if pd.isna(v) else float(v) for v in row] for row in pivot.values],
    }
