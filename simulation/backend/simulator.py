"""Single-strategy simulator — every knob maps to a real line in the repo.

The catalog below is the single source of truth. Each entry:
  - binds to an actual strategy class in strategies/research_strategies.py
  - lists only the parameters that exist on that class's __init__ signature
  - exposes the class's live source code via inspect.getsource() so the
    frontend shows exactly what the backend is about to run

The engine knobs come from BacktestConfig — every field we expose is
consumed by build_target_weights() or run_weight_backtest() in the repo.
Nothing is invented.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtester import research_backtester as rbt
from backtester.research_data import ResearchDataset, load_wharton_research_dataset
from strategies import research_strategies as rs

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "backtester" / "WhartonDataSource.parquet"

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "ready": False, "loading": False, "message": "idle", "error": None,
    "dataset": None, "benchmark": None,
    "date_min": None, "date_max": None, "tickers": 0,
}
_SIM_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Strategy catalog — declarative, each field mirrors the actual source
# ---------------------------------------------------------------------------
STRATEGIES = [
    {
        "id": "momentum",
        "cls": rs.CrossSectionalMomentumStrategy,
        "label": "Cross-Sectional Momentum",
        "summary": "Rank stocks by trailing return (skip the last month), long the top quintile.",
        "params": [
            {
                "name": "lookback_days", "type": "int", "default": 126,
                "min": 21, "max": 504, "step": 21,
                "desc": "Days of history used for the return ranking.",
            },
            {
                "name": "skip_days", "type": "int", "default": 21,
                "min": 0, "max": 63, "step": 1,
                "desc": "Days to skip at the front (avoid 1-month reversal).",
            },
        ],
    },
    {
        "id": "lowvol",
        "cls": rs.LowVolatilityStrategy,
        "label": "Low Volatility",
        "summary": "Rank stocks by trailing realized vol (ascending), long the calmest quintile.",
        "params": [
            {
                "name": "window", "type": "int", "default": 126,
                "min": 21, "max": 504, "step": 21,
                "desc": "Rolling window (in trading days) for realized vol.",
            },
        ],
    },
    {
        "id": "smallcap",
        "cls": rs.SmallCapTiltStrategy,
        "label": "Small-Cap Tilt",
        "summary": "Rank by -log(market cap), long the smallest quintile.",
        "params": [],
    },
]
STRATEGY_BY_ID = {s["id"]: s for s in STRATEGIES}

# Engine knobs — a strict subset of BacktestConfig that actually fires on
# every run in build_target_weights / run_weight_backtest.
ENGINE_PARAMS = [
    {
        "name": "rebalance_frequency", "type": "choice", "default": "ME",
        "choices": [
            {"value": "W",  "label": "Weekly"},
            {"value": "ME", "label": "Monthly (end-of-month)"},
            {"value": "QE", "label": "Quarterly (end-of-quarter)"},
        ],
        "desc": "Pandas resample rule used by _get_rebalance_dates().",
    },
    {
        "name": "signal_lag", "type": "int", "default": 1,
        "min": 0, "max": 5, "step": 1,
        "desc": "Days to delay target weights before execution — 1 = trade on tomorrow's open.",
    },
    {
        "name": "transaction_cost_bps", "type": "float", "default": 10.0,
        "min": 0.0, "max": 50.0, "step": 0.5,
        "desc": "Round-trip cost in basis points applied to turnover each rebalance.",
    },
    {
        "name": "long_quantile", "type": "float", "default": 0.20,
        "min": 0.05, "max": 0.50, "step": 0.05,
        "desc": "Fraction of the cross-section to go long (top nlargest).",
    },
    {
        "name": "max_position_weight", "type": "float", "default": 0.08,
        "min": 0.01, "max": 0.25, "step": 0.01,
        "desc": "Cap on any single name's portfolio weight.",
    },
    {
        "name": "leverage", "type": "float", "default": 1.0,
        "min": 0.5, "max": 2.0, "step": 0.1,
        "desc": "Gross exposure target (1.0 = fully invested long-only).",
    },
    {
        "name": "construction_method", "type": "choice", "default": "inverse_vol",
        "choices": [
            {"value": "equal_weight", "label": "Equal weight"},
            {"value": "inverse_vol",  "label": "Inverse volatility"},
            {"value": "mean_variance", "label": "Mean-variance"},
        ],
        "desc": "How the selected names are weighted after ranking.",
    },
]


# ---------------------------------------------------------------------------
# Warmup — load dataset only
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
            _STATE.update({
                "ready": True, "loading": False, "message": "ready",
                "dataset": dataset, "benchmark": dataset.benchmark_returns,
                "date_min": str(dataset.prices.index.min().date()),
                "date_max": str(dataset.prices.index.max().date()),
                "tickers": len(dataset.tickers),
            })
    except Exception as exc:
        with _LOCK:
            _STATE["loading"] = False
            _STATE["error"] = str(exc)
            _STATE["message"] = f"error: {exc}"
        raise


def status() -> Dict[str, Any]:
    with _LOCK:
        return {
            "ready": _STATE["ready"], "loading": _STATE["loading"],
            "message": _STATE["message"], "error": _STATE["error"],
            "date_min": _STATE["date_min"], "date_max": _STATE["date_max"],
            "tickers": _STATE["tickers"],
        }


# ---------------------------------------------------------------------------
# Catalog — expose actual source code
# ---------------------------------------------------------------------------
def catalog() -> Dict[str, Any]:
    strategies = []
    for entry in STRATEGIES:
        src = inspect.getsource(entry["cls"])
        strategies.append({
            "id": entry["id"], "label": entry["label"],
            "summary": entry["summary"],
            "params": entry["params"],
            "source": src,
            "source_file": f"strategies/research_strategies.py",
        })
    engine_sources = {
        "BacktestConfig": inspect.getsource(rbt.BacktestConfig),
        "build_target_weights": inspect.getsource(rbt.build_target_weights),
        "run_weight_backtest": inspect.getsource(rbt.run_weight_backtest),
    }
    return {
        "strategies": strategies,
        "engine_params": ENGINE_PARAMS,
        "engine_sources": engine_sources,
        "engine_source_file": "backtester/research_backtester.py",
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _cache_key(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()


def _metrics(returns: pd.Series) -> Dict[str, float]:
    clean = returns.dropna()
    if len(clean) < 2:
        return {}
    cum = (1 + clean).cumprod()
    ann_ret = cum.iloc[-1] ** (252 / len(clean)) - 1
    std = clean.std(ddof=0)
    sharpe = clean.mean() / std * np.sqrt(252) if std > 0 else 0.0
    downside = clean[clean < 0]
    sortino = clean.mean() / downside.std(ddof=0) * np.sqrt(252) if len(downside) and downside.std(ddof=0) > 0 else 0.0
    dd = (cum / cum.cummax() - 1).min()
    return {
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(std * np.sqrt(252)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd),
        "calmar": float(ann_ret / abs(dd)) if dd else 0.0,
        "win_rate": float((clean > 0).mean()),
    }


def _drawdown(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    return cum / cum.cummax() - 1


def _sample_holdings(target_weights: pd.DataFrame, max_rows: int = 8, top_n: int = 10) -> List[Dict[str, Any]]:
    """Pull the last few rebalance snapshots — non-zero holdings only."""
    rebalance_rows = []
    prev = None
    for date, row in target_weights.iterrows():
        values = row.values
        if prev is None or not np.allclose(values, prev, equal_nan=False):
            rebalance_rows.append(date)
            prev = values
    rebalance_rows = rebalance_rows[-max_rows:]
    snapshots = []
    for date in rebalance_rows:
        row = target_weights.loc[date]
        nonzero = row[row.abs() > 1e-6].sort_values(ascending=False).head(top_n)
        snapshots.append({
            "date": str(date.date()),
            "n_positions": int((row.abs() > 1e-6).sum()),
            "gross_exposure": float(row.abs().sum()),
            "net_exposure": float(row.sum()),
            "top": [{"ticker": t, "weight": float(w)} for t, w in nonzero.items()],
        })
    return snapshots


def run_simulation(body: Dict[str, Any]) -> Dict[str, Any]:
    if not _STATE["ready"]:
        raise RuntimeError("simulator not ready — wait for warmup")

    strategy_id = body.get("strategy_id", "momentum")
    if strategy_id not in STRATEGY_BY_ID:
        raise ValueError(f"unknown strategy_id: {strategy_id}")
    strat_entry = STRATEGY_BY_ID[strategy_id]

    strategy_params = body.get("strategy_params", {}) or {}
    engine_params = body.get("engine_params", {}) or {}
    start = body.get("start") or _STATE["date_min"]
    end = body.get("end") or _STATE["date_max"]

    # Build strategy instance — only pass kwargs that actually exist
    kw_allowed = {p["name"] for p in strat_entry["params"]}
    kwargs = {k: strategy_params[k] for k in kw_allowed if k in strategy_params}
    strat_instance = strat_entry["cls"](**kwargs)

    # Build config — only pass keys that actually exist on BacktestConfig
    config_kwargs: Dict[str, Any] = {
        "long_only": True,
        "short_quantile": 0.0,
        "min_names": 10,
    }
    config_kwargs.update({
        k: engine_params[k]
        for k in ("rebalance_frequency", "signal_lag", "transaction_cost_bps",
                  "long_quantile", "max_position_weight", "leverage", "construction_method")
        if k in engine_params
    })
    config = rbt.BacktestConfig(**config_kwargs)

    cache_payload = {
        "strategy_id": strategy_id, "strategy_params": kwargs,
        "engine_params": config_kwargs, "start": start, "end": end,
    }
    key = _cache_key(cache_payload)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]

    dataset = _STATE["dataset"]
    window = (pd.Timestamp(start), pd.Timestamp(end))

    # Slice the dataset to the user's window
    prices = dataset.prices.loc[window[0]:window[1]]
    returns = dataset.returns.loc[window[0]:window[1]]
    benchmark = _STATE["benchmark"].loc[window[0]:window[1]]
    if prices.empty:
        raise ValueError("date window produced no price data")

    sliced = ResearchDataset(
        prices=prices, returns=returns, benchmark_returns=benchmark,
        volumes=dataset.volumes.loc[window[0]:window[1]] if dataset.volumes is not None else None,
        dividends=dataset.dividends.loc[window[0]:window[1]] if dataset.dividends is not None else None,
        eps=dataset.eps.loc[window[0]:window[1]] if dataset.eps is not None else None,
        market_caps=dataset.market_caps.loc[window[0]:window[1]] if dataset.market_caps is not None else None,
        metadata=dataset.metadata,
        events=dataset.events,
    )

    scores = strat_instance.generate(sliced).scores
    target_weights = rbt.build_target_weights(scores, sliced.returns, config)
    result = rbt.run_weight_backtest(
        prices=sliced.prices, target_weights=target_weights, config=config,
        benchmark_returns=sliced.benchmark_returns, strategy_name=strat_entry["label"],
    )

    gross_cum = (1 + result.gross_returns.fillna(0)).cumprod() - 1
    net_cum = (1 + result.net_returns.fillna(0)).cumprod() - 1
    bench_cum = (1 + benchmark.reindex(result.net_returns.index).fillna(0)).cumprod() - 1

    turnover_ann = float(result.turnover.mean() * 252)
    tcost_drag_ann = float(result.transaction_costs.sum() / len(result.transaction_costs) * 252)
    n_rebalances = int((result.turnover > 0).sum())
    avg_positions = float((result.target_weights.abs() > 1e-6).sum(axis=1).replace(0, np.nan).mean())

    dates = [d.strftime("%Y-%m-%d") for d in result.net_returns.index]

    response = {
        "strategy": {"id": strategy_id, "label": strat_entry["label"]},
        "dates": dates,
        "cumulative_net": [float(v) for v in net_cum.values],
        "cumulative_gross": [float(v) for v in gross_cum.values],
        "cumulative_benchmark": [float(v) for v in bench_cum.values],
        "drawdown": [float(v) for v in _drawdown(result.net_returns).values],
        "metrics_net": _metrics(result.net_returns),
        "metrics_gross": _metrics(result.gross_returns),
        "metrics_benchmark": _metrics(benchmark.reindex(result.net_returns.index).fillna(0)),
        "order_summary": {
            "n_rebalances": n_rebalances,
            "avg_positions": avg_positions if not np.isnan(avg_positions) else 0.0,
            "turnover_annualized": turnover_ann,
            "tcost_drag_annualized": tcost_drag_ann,
            "total_tcost_cumulative": float(result.transaction_costs.sum()),
        },
        "recent_holdings": _sample_holdings(result.target_weights),
        "params_echo": cache_payload,
    }
    _SIM_CACHE[key] = response
    return response
