"""Single-strategy simulator — every knob maps to a real line in the repo.

MVP scope: two strategies (momentum, short-term reversal) and three engine
knobs (transaction_cost_bps, long_quantile, signal_lag). Other BacktestConfig
fields are held at sensible defaults; extend CATALOG / ENGINE_PARAMS to grow.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
from pathlib import Path
from typing import Any, Dict, List

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
# Catalog — declarative, each field mirrors an actual line of source
# ---------------------------------------------------------------------------
STRATEGIES = [
    {
        "id": "momentum",
        "cls": rs.CrossSectionalMomentumStrategy,
        "cls_name": "CrossSectionalMomentumStrategy",
        "label": "Momentum",
        "summary": "Long the top-quintile winners over the last lookback_days (skipping the last skip_days).",
        "params": [
            {"name": "lookback_days", "type": "int", "default": 126, "min": 21, "max": 504, "step": 21},
            {"name": "skip_days",     "type": "int", "default": 21,  "min": 0,  "max": 63,  "step": 1},
        ],
    },
    {
        "id": "mean_reversion",
        "cls": rs.ShortTermReversalStrategy,
        "cls_name": "ShortTermReversalStrategy",
        "label": "Mean Reversion",
        "summary": "Long the top-quintile recent losers over the last lookback_days.",
        "params": [
            {"name": "lookback_days", "type": "int", "default": 5, "min": 1, "max": 21, "step": 1},
        ],
    },
]
STRATEGY_BY_ID = {s["id"]: s for s in STRATEGIES}

ENGINE_PARAMS = [
    {"name": "transaction_cost_bps", "type": "float", "default": 10.0, "min": 0.0, "max": 50.0, "step": 0.5},
    {"name": "long_quantile",        "type": "float", "default": 0.20, "min": 0.05, "max": 0.50, "step": 0.05},
    {"name": "signal_lag",           "type": "int",   "default": 1,    "min": 0,   "max": 3,    "step": 1},
]

# Fields held constant — shown in the UI so nothing is hidden.
GICS_SECTORS = {
    "10": "Energy", "15": "Materials", "20": "Industrials",
    "25": "Consumer Discretionary", "30": "Consumer Staples",
    "35": "Health Care", "40": "Financials",
    "45": "Information Technology", "50": "Communication Services",
    "55": "Utilities", "60": "Real Estate",
}


FIXED_ENGINE = {
    "rebalance_frequency": "ME",
    "max_position_weight": 0.08,
    "leverage": 1.0,
    "construction_method": "inverse_vol",
    "long_only": True,
    "short_quantile": 0.0,
    "min_names": 10,
}


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
def warmup(start: str = "2000-01-01", end: str = "2025-12-31") -> None:
    with _LOCK:
        if _STATE["loading"] or _STATE["ready"]:
            return
        _STATE["loading"] = True
        _STATE["message"] = "loading Wharton parquet"
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
# Catalog endpoint
# ---------------------------------------------------------------------------
def catalog() -> Dict[str, Any]:
    strategies = []
    for entry in STRATEGIES:
        src = inspect.getsource(entry["cls"])
        strategies.append({
            "id": entry["id"],
            "label": entry["label"],
            "cls_name": entry["cls_name"],
            "summary": entry["summary"],
            "params": entry["params"],
            "source": src,
            "source_file": "strategies/research_strategies.py",
        })
    return {
        "strategies": strategies,
        "engine_params": ENGINE_PARAMS,
        "fixed_engine": FIXED_ENGINE,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _metrics(returns: pd.Series) -> Dict[str, float]:
    clean = returns.dropna()
    if len(clean) < 2:
        return {}
    cum = (1 + clean).cumprod()
    ann_ret = cum.iloc[-1] ** (252 / len(clean)) - 1
    std = clean.std(ddof=0)
    sharpe = clean.mean() / std * np.sqrt(252) if std > 0 else 0.0
    dd = (cum / cum.cummax() - 1).min()
    return {
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(std * np.sqrt(252)),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "win_rate": float((clean > 0).mean()),
    }


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

    # Build strategy with only the declared kwargs
    kw_allowed = {p["name"] for p in strat_entry["params"]}
    strat_kwargs = {k: strategy_params[k] for k in kw_allowed if k in strategy_params}
    strat_instance = strat_entry["cls"](**strat_kwargs)

    # Build BacktestConfig: fixed defaults + the 3 exposed knobs
    cfg_kwargs = dict(FIXED_ENGINE)
    for p in ENGINE_PARAMS:
        if p["name"] in engine_params:
            cfg_kwargs[p["name"]] = engine_params[p["name"]]
    config = rbt.BacktestConfig(**cfg_kwargs)

    key = _cache_key({
        "strategy_id": strategy_id, "strategy_params": strat_kwargs,
        "engine_params": cfg_kwargs, "start": start, "end": end,
    })
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]

    dataset = _STATE["dataset"]
    window = (pd.Timestamp(start), pd.Timestamp(end))
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

    net_cum = (1 + result.net_returns.fillna(0)).cumprod() - 1
    bench_cum = (1 + benchmark.reindex(result.net_returns.index).fillna(0)).cumprod() - 1

    turnover_ann = float(result.turnover.mean() * 252)
    tcost_drag_ann = float(result.transaction_costs.mean() * 252)

    response = {
        "strategy": {"id": strategy_id, "label": strat_entry["label"]},
        "dates": [d.strftime("%Y-%m-%d") for d in result.net_returns.index],
        "cumulative_net": [float(v) for v in net_cum.values],
        "cumulative_benchmark": [float(v) for v in bench_cum.values],
        "metrics_net": _metrics(result.net_returns),
        "metrics_benchmark": _metrics(benchmark.reindex(result.net_returns.index).fillna(0)),
        "order_summary": {
            "turnover_annualized": turnover_ann,
            "tcost_drag_annualized": tcost_drag_ann,
        },
        "survivorship_audit": result.survivorship_audit or {},
    }
    _SIM_CACHE[key] = response
    return response


# ---------------------------------------------------------------------------
# Data inspector
# ---------------------------------------------------------------------------
def data_overview() -> Dict[str, Any]:
    """Per-ticker summary rows for the table view."""
    if not _STATE["ready"]:
        raise RuntimeError("simulator not ready")
    dataset = _STATE["dataset"]
    prices = dataset.prices
    meta = dataset.metadata if dataset.metadata is not None else pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for tic in prices.columns:
        series = prices[tic].dropna()
        if series.empty:
            continue
        start = series.index.min()
        end = series.index.max()
        total_return = float(series.iloc[-1] / series.iloc[0] - 1)
        n_days = int(series.shape[0])
        row: Dict[str, Any] = {
            "ticker": tic,
            "start": str(start.date()),
            "end": str(end.date()),
            "n_days": n_days,
            "total_return": total_return,
            "first_price": float(series.iloc[0]),
            "last_price": float(series.iloc[-1]),
        }
        if tic in meta.index:
            row["name"] = str(meta.at[tic, "conm"]) if "conm" in meta.columns and pd.notna(meta.at[tic, "conm"]) else ""
            sector_code = str(meta.at[tic, "gsector"]).split(".")[0] if "gsector" in meta.columns and pd.notna(meta.at[tic, "gsector"]) else ""
            row["sector"] = GICS_SECTORS.get(sector_code, sector_code)
        else:
            row["name"] = ""
            row["sector"] = ""
        rows.append(row)

    return {
        "source_file": "backtester/WhartonDataSource.parquet",
        "date_min": _STATE["date_min"],
        "date_max": _STATE["date_max"],
        "n_tickers": len(rows),
        "tickers": rows,
    }


def ticker_detail(ticker: str) -> Dict[str, Any]:
    """Price history + summary stats for a single ticker."""
    if not _STATE["ready"]:
        raise RuntimeError("simulator not ready")
    dataset = _STATE["dataset"]
    ticker = ticker.upper()
    if ticker not in dataset.prices.columns:
        raise ValueError(f"ticker not found: {ticker}")
    series = dataset.prices[ticker].dropna()
    returns = series.pct_change().dropna()
    ann_vol = float(returns.std(ddof=0) * np.sqrt(252)) if len(returns) > 1 else 0.0
    ann_ret = float((series.iloc[-1] / series.iloc[0]) ** (252 / max(len(series), 1)) - 1)
    max_dd = float((series / series.cummax() - 1).min())

    meta: Dict[str, Any] = {}
    if dataset.metadata is not None and ticker in dataset.metadata.index:
        row = dataset.metadata.loc[ticker]
        for col in ("conm", "gsector", "gind", "sic", "naics", "exchg", "fic"):
            if col in row.index and pd.notna(row[col]):
                val = str(row[col])
                if col == "gsector":
                    code = val.split(".")[0]
                    meta[col] = f"{code} — {GICS_SECTORS.get(code, '?')}"
                else:
                    meta[col] = val

    return {
        "ticker": ticker,
        "metadata": meta,
        "n_days": int(len(series)),
        "start": str(series.index.min().date()),
        "end": str(series.index.max().date()),
        "first_price": float(series.iloc[0]),
        "last_price": float(series.iloc[-1]),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd,
        "dates": [d.strftime("%Y-%m-%d") for d in series.index],
        "prices": [float(v) for v in series.values],
    }
