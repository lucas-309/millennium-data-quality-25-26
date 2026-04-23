"""Single-strategy simulator — every knob maps to a real line in the repo.

Universe: today's 503 S&P 500 constituents (Wikipedia), prices sourced from
the yfinance per-ticker pickle cache under ``data_cache/yfinance``. Names
that joined the index recently appear from their first available trading
day and ffill no data before it — the survivorship audit panel quantifies
the bias.

MVP scope: two strategies (momentum, short-term reversal) and three engine
knobs (transaction_cost_bps, long_quantile, signal_lag). Other BacktestConfig
fields are held at sensible defaults; extend CATALOG / ENGINE_PARAMS to grow.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtester import research_backtester as rbt
from backtester.research_data import ResearchDataset
from backtester.sp500_universe import load_sp500_universe
from backtester.yfinance_cache import list_cached_tickers
from backtester.yfinance_loader import load_yfinance_research_dataset
from strategies import research_strategies as rs

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "ready": False, "loading": False, "message": "idle", "error": None,
    "dataset": None, "benchmark": None,
    "date_min": None, "date_max": None, "tickers": 0,
    "universe_label": None,
}
_SIM_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Source-code cleanup — strip the narrative class attributes so the live
# code panel shows the actual *logic* (constructor + generate_scores), not
# the motivational prose that drifts with every re-read.
# ---------------------------------------------------------------------------
_RATIONALE_ATTRS = {
    "name", "motivation", "economic_rationale",
    "why_it_works", "why_it_fails",
    "uses_event_data", "event_type",
    "backtest_overrides", "combine_in_portfolio",
}


def _clean_strategy_source(src: str) -> str:
    cleaned: List[str] = []
    prev_blank = False
    for line in src.split("\n"):
        stripped = line.strip()
        head = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if head in _RATIONALE_ATTRS:
            continue
        if stripped == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        cleaned.append(line)
    return "\n".join(cleaned).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Catalog — declarative, each field mirrors an actual line of source
# ---------------------------------------------------------------------------
STRATEGIES = [
    {
        "id": "momentum",
        "cls": rs.CrossSectionalMomentumStrategy,
        "cls_name": "CrossSectionalMomentumStrategy",
        "label": "Momentum",
        "summary": "Rank stocks by their return from (today − lookback_days) to (today − skip_days); long the top quantile, optionally short the bottom.",
        "params": [
            {"name": "lookback_days", "type": "int", "default": 126, "min": 21, "max": 504, "step": 21,
             "help": "Return window (126 ≈ 6mo)."},
            {"name": "skip_days",     "type": "int", "default": 21,  "min": 0,  "max": 63,  "step": 1,
             "help": "Skip last N days to drop short-term reversal (21 ≈ 1mo)."},
        ],
        "engine_overrides": [
            {"name": "short_quantile", "type": "float", "default": 0.0, "min": 0.0, "max": 0.40, "step": 0.05,
             "help": "Bottom fraction sold short (0 = long-only, long-short becomes dollar-neutral)."},
        ],
    },
    {
        "id": "mean_reversion",
        "cls": rs.ShortTermReversalStrategy,
        "cls_name": "ShortTermReversalStrategy",
        "label": "Mean Reversion",
        "summary": "Score stocks by their negated recent return; long the biggest recent losers, expecting a bounce.",
        "params": [
            {"name": "lookback_days", "type": "int", "default": 5, "min": 1, "max": 21, "step": 1,
             "help": "Recent-return window; biggest losers rank highest."},
        ],
        "engine_overrides": [],
    },
]
STRATEGY_BY_ID = {s["id"]: s for s in STRATEGIES}

ENGINE_PARAMS = [
    {"name": "transaction_cost_bps", "type": "float", "default": 10.0, "min": 0.0, "max": 50.0, "step": 0.5,
     "help": "Per-side trading cost in bps."},
    {"name": "long_quantile",        "type": "float", "default": 0.20, "min": 0.05, "max": 0.50, "step": 0.05,
     "help": "Top fraction held long (0.20 = top 20%)."},
    {"name": "signal_lag",           "type": "int",   "default": 1,    "min": 0,   "max": 3,    "step": 1,
     "help": "Days between signal and execution."},
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
# Warmup — load the yfinance S&P 500 cache into a ResearchDataset
# ---------------------------------------------------------------------------
def warmup(start: str = "2005-01-01", end: str = None) -> None:
    if end is None:
        end = date.today().strftime("%Y-%m-%d")
    with _LOCK:
        if _STATE["loading"] or _STATE["ready"]:
            return
        _STATE["loading"] = True
        _STATE["message"] = "loading S&P 500 yfinance cache"
        _STATE["error"] = None
    try:
        cached = list_cached_tickers()
        if not cached:
            raise RuntimeError(
                "yfinance cache is empty — run `python run_sp500_fetch.py` first"
            )
        snap = load_sp500_universe()
        dataset = load_yfinance_research_dataset(
            start_date=start, end_date=end,
            tickers=cached,
            min_history=21,  # let recent SP500 joiners in; strategies skip NaN-score days
            sp500_frame=snap.frame,
        )
        with _LOCK:
            _STATE.update({
                "ready": True, "loading": False, "message": "ready",
                "dataset": dataset, "benchmark": dataset.benchmark_returns,
                "date_min": str(dataset.prices.index.min().date()),
                "date_max": str(dataset.prices.index.max().date()),
                "tickers": len(dataset.tickers),
                "universe_label": f"S&P 500 yfinance ({len(dataset.tickers)} tickers)",
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
            "universe_label": _STATE["universe_label"],
        }


# ---------------------------------------------------------------------------
# Catalog endpoint
# ---------------------------------------------------------------------------
def catalog() -> Dict[str, Any]:
    strategies = []
    for entry in STRATEGIES:
        src = _clean_strategy_source(inspect.getsource(entry["cls"]))
        strategies.append({
            "id": entry["id"],
            "label": entry["label"],
            "cls_name": entry["cls_name"],
            "summary": entry["summary"],
            "params": entry["params"],
            "engine_overrides": entry.get("engine_overrides", []),
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


def _normalise_tickers(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = raw.replace("\n", ",").split(",")
    else:
        items = list(raw)
    out = []
    seen = set()
    for t in items:
        t = str(t).strip().upper().replace(".", "-")
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


#: Cap a single /api/universe/add call so a typo'd paste can't block the server
#: for minutes or trip yfinance's rate limiter.
_MAX_UNIVERSE_ADD = 50


def add_tickers_to_universe(tickers: List[str]) -> Dict[str, Any]:
    """Fetch any missing tickers into the yfinance cache, then rebuild the
    in-memory dataset so the new names are immediately usable by the
    simulator.
    """
    from backtester.yfinance_cache import fetch_tickers, list_cached_tickers

    wanted = _normalise_tickers(tickers)
    if not wanted:
        raise ValueError("no tickers provided")
    if len(wanted) > _MAX_UNIVERSE_ADD:
        raise ValueError(
            f"too many tickers ({len(wanted)}); max per request is {_MAX_UNIVERSE_ADD}. "
            "Split the list and submit in batches."
        )

    cached_before = set(list_cached_tickers())
    to_fetch = [t for t in wanted if t not in cached_before]

    fetched: List[str] = []
    failed: List[Dict[str, str]] = []
    if to_fetch:
        results = fetch_tickers(
            to_fetch,
            start="2005-01-01",
            pause_seconds=0.4,
            max_retries=4,
            fetch_info=True,
        )
        for r in results:
            if r.ok:
                fetched.append(r.ticker)
            else:
                failed.append({"ticker": r.ticker, "error": r.error or "unknown"})

    # Always rebuild the dataset so newly-cached tickers become visible.
    with _LOCK:
        _STATE["ready"] = False
        _STATE["loading"] = False
    _SIM_CACHE.clear()
    warmup()

    return {
        "requested": wanted,
        "already_cached": sorted(set(wanted) & cached_before),
        "fetched": sorted(fetched),
        "failed": failed,
        "universe_size_after": _STATE["tickers"],
    }


def run_simulation(body: Dict[str, Any]) -> Dict[str, Any]:
    # Snapshot the shared dataset under the lock so a concurrent
    # add_tickers_to_universe rebuild can't swap the panel out from under us
    # mid-run (KeyError on dropped columns, stale prices, etc.).
    with _LOCK:
        if not _STATE["ready"]:
            raise RuntimeError("simulator not ready — wait for warmup")
        dataset = _STATE["dataset"]
        benchmark_full = _STATE["benchmark"]
        date_min = _STATE["date_min"]
        date_max = _STATE["date_max"]

    strategy_id = body.get("strategy_id", "momentum")
    if strategy_id not in STRATEGY_BY_ID:
        raise ValueError(f"unknown strategy_id: {strategy_id}")
    strat_entry = STRATEGY_BY_ID[strategy_id]

    strategy_params = body.get("strategy_params", {}) or {}
    engine_params = body.get("engine_params", {}) or {}
    engine_overrides_in = body.get("engine_overrides", {}) or {}
    # Distinguish "tickers not provided / null" (= full universe) from
    # "tickers: [] explicitly" (= user narrowed to zero names, should fail
    # loudly rather than silently run the full cache).
    raw_tickers = body.get("tickers")
    if raw_tickers is None:
        custom_tickers = []
    else:
        custom_tickers = _normalise_tickers(raw_tickers)
        if not custom_tickers:
            raise ValueError(
                "tickers filter is explicitly empty — nothing to run on. "
                "Send tickers: null (or omit the field) for the full universe."
            )
    start = body.get("start") or date_min
    end = body.get("end") or date_max

    # Strategy constructor args (whitelisted)
    kw_allowed = {p["name"] for p in strat_entry["params"]}
    strat_kwargs = {k: strategy_params[k] for k in kw_allowed if k in strategy_params}
    strat_instance = strat_entry["cls"](**strat_kwargs)

    # BacktestConfig: fixed defaults + global engine knobs + per-strategy overrides
    cfg_kwargs = dict(FIXED_ENGINE)
    for p in ENGINE_PARAMS:
        if p["name"] in engine_params:
            cfg_kwargs[p["name"]] = engine_params[p["name"]]
    for p in strat_entry.get("engine_overrides", []):
        if p["name"] in engine_overrides_in:
            cfg_kwargs[p["name"]] = engine_overrides_in[p["name"]]
    # Auto-flip long_only based on short_quantile — a positive short leg
    # means we actually want to short (otherwise build_target_weights silently
    # ignores the short side).
    cfg_kwargs["long_only"] = float(cfg_kwargs.get("short_quantile", 0.0)) <= 0.0

    # Custom ticker subset — restrict to tickers already in memory; if any
    # are missing the caller should POST /api/universe/add first.
    universe_note = None
    selected_tickers = None
    if custom_tickers:
        available = set(dataset.tickers)
        selected_tickers = sorted(t for t in custom_tickers if t in available)
        missing = sorted(t for t in custom_tickers if t not in available)
        if not selected_tickers:
            raise ValueError(
                "none of the requested tickers are in the cache — run "
                "/api/universe/add first. Missing: " + ", ".join(missing)
            )
        universe_note = {"selected": selected_tickers, "missing": missing}
        # Scale min_names for small custom universes so the strategy actually
        # forms portfolios instead of skipping every rebalance.
        cfg_kwargs["min_names"] = min(cfg_kwargs.get("min_names", 10), max(2, len(selected_tickers) // 4))
        # Also cap max_position_weight so the per-leg budget can fit even when
        # only a handful of names qualify.
        n_long = max(1, int(len(selected_tickers) * float(cfg_kwargs["long_quantile"])))
        cfg_kwargs["max_position_weight"] = max(cfg_kwargs.get("max_position_weight", 0.08), 1.0 / n_long)

    config = rbt.BacktestConfig(**cfg_kwargs)

    key = _cache_key({
        "strategy_id": strategy_id, "strategy_params": strat_kwargs,
        "engine_params": cfg_kwargs, "start": start, "end": end,
        "tickers": selected_tickers,
    })
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]

    window = (pd.Timestamp(start), pd.Timestamp(end))

    prices = dataset.prices.loc[window[0]:window[1]]
    if selected_tickers:
        prices = prices[selected_tickers]
    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise ValueError("date window produced no price data")

    cols = prices.columns
    returns = dataset.returns.loc[window[0]:window[1]].reindex(columns=cols)
    if selected_tickers:
        benchmark = returns.mean(axis=1, skipna=True).fillna(0.0).rename("custom_universe")
    else:
        benchmark = benchmark_full.loc[window[0]:window[1]]

    def _opt_slice(panel):
        if panel is None:
            return None
        return panel.loc[window[0]:window[1]].reindex(columns=cols)

    sliced = ResearchDataset(
        prices=prices,
        returns=returns,
        benchmark_returns=benchmark,
        volumes=_opt_slice(dataset.volumes),
        dividends=_opt_slice(dataset.dividends),
        eps=_opt_slice(dataset.eps),
        market_caps=_opt_slice(dataset.market_caps),
        metadata=dataset.metadata.reindex(cols) if dataset.metadata is not None else None,
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
        "universe": {
            "n_tickers": int(prices.shape[1]),
            "custom": bool(selected_tickers),
            "selected": selected_tickers if selected_tickers else None,
            "missing_from_cache": universe_note["missing"] if universe_note else [],
        },
        "config": {
            "long_only": cfg_kwargs["long_only"],
            "long_quantile": float(cfg_kwargs["long_quantile"]),
            "short_quantile": float(cfg_kwargs.get("short_quantile", 0.0)),
            "leverage": float(cfg_kwargs.get("leverage", 1.0)),
        },
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
        "source_file": "data_cache/yfinance/*.pkl (current S&P 500)",
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
