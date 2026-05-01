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
import os
import threading
import time
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtester import research_backtester as rbt
from backtester.research_data import ResearchDataset, load_wharton_research_dataset
from backtester.sp500_universe import load_sp500_universe
from backtester.yfinance_cache import list_cached_tickers
from backtester.yfinance_loader import load_yfinance_research_dataset
from strategies import research_strategies as rs

# Two data sources are supported:
#   - "yfinance": today's S&P 500 (~503 tickers), live cache via run_sp500_fetch.py
#   - "wharton" : the WhartonDataSource4.parquet panel (~865 unique tickers across
#                 all-time SP membership; static file committed to the repo)
DEFAULT_SOURCE = "yfinance"
SUPPORTED_SOURCES = ("yfinance", "wharton")
WHARTON_PARQUET = Path(__file__).resolve().parent.parent.parent / "backtester" / "WhartonDataSource4.parquet"

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "ready": False, "loading": False, "message": "idle", "error": None,
    "dataset": None, "benchmark": None,
    "date_min": None, "date_max": None, "tickers": 0,
    "universe_label": None,
    "data_source": DEFAULT_SOURCE,
    "load_token": 0,
}
# Per-run simulation cache, bounded so a worker that runs hundreds of
# parameter sweeps doesn't accumulate result blobs forever. Each entry is a
# few hundred KB (date arrays + cum series + audit data); 64 entries caps
# the cache at ~10-20 MB, well below the Fly VM's working budget.
_SIM_CACHE_MAX = 64
_SIM_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

# Per-source dataset cache. Holding both yfinance and wharton panels resident
# pushes a 1GB Fly VM over its limit, so this cache is opt-in. Set
# MILLENNIUM_DATASET_CACHE=1 in local dev (where toggling sources is the
# common UX) to make return trips ~instant. Default is off — production VMs
# load one source at a time and free the previous panel on switch.
_DATASET_CACHE_ENABLED = os.environ.get("MILLENNIUM_DATASET_CACHE", "0").lower() in {"1", "true", "yes", "on"}
_DATASET_CACHE: Dict[str, Dict[str, Any]] = {}


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
_SHORT_QUANTILE_OVERRIDE = {
    "name": "short_quantile", "type": "float", "default": 0.0,
    "min": 0.0, "max": 0.40, "step": 0.05,
    "help": "Bottom fraction sold short.",
}

STRATEGIES = [
    {
        "id": "momentum",
        "cls": rs.CrossSectionalMomentumStrategy,
        "cls_name": "CrossSectionalMomentumStrategy",
        "label": "Momentum",
        "formula": "score(t, i) = ln( P[t − skip, i] / P[t − lookback, i] )",
        "summary": "Rank stocks by their return from (today − lookback_days) to (today − skip_days); long the top quantile, optionally short the bottom.",
        "params": [
            {"name": "lookback_days", "type": "int", "default": 126, "min": 21, "max": 504, "step": 21,
             "help": "Return window (126 ≈ 6mo)."},
            {"name": "skip_days",     "type": "int", "default": 21,  "min": 0,  "max": 63,  "step": 1,
             "help": "Skip last N days to drop short-term reversal (21 ≈ 1mo)."},
        ],
        "engine_overrides": [_SHORT_QUANTILE_OVERRIDE],
    },
    {
        "id": "mean_reversion",
        "cls": rs.ShortTermReversalStrategy,
        "cls_name": "ShortTermReversalStrategy",
        "label": "Mean Reversion",
        "formula": "score(t, i) = − ln( P[t, i] / P[t − lookback, i] )",
        "summary": "Score stocks by their negated recent return; long the biggest recent losers, expecting a bounce.",
        "params": [
            {"name": "lookback_days", "type": "int", "default": 5, "min": 1, "max": 21, "step": 1,
             "help": "Recent-return window; biggest losers rank highest."},
        ],
        "engine_overrides": [_SHORT_QUANTILE_OVERRIDE],
    },
    {
        "id": "value_composite",
        "cls": rs.ValueCompositeStrategy,
        "cls_name": "ValueCompositeStrategy",
        "label": "Value Composite",
        "formula": "score(t, i) = z(div_yield) + z(EPS / P) + z(− log market_cap)",
        "summary": (
            "Fama & French (1992, 2015 — Five-Factor Model). Cross-sectional "
            "value blend of trailing dividend yield, earnings yield (E/P), and "
            "a small-size tilt, each cross-sectionally z-scored and summed. "
            "Long the cheapest decile, optionally short the most expensive."
        ),
        "params": [
            {"name": "trailing_days", "type": "int", "default": 252, "min": 60, "max": 756, "step": 21,
             "help": "Window for trailing dividend sum (252 = 1y)."},
        ],
        "engine_overrides": [
            {**_SHORT_QUANTILE_OVERRIDE, "default": 0.10,
             "help": "Bottom decile (most expensive names) sold short — recovers the long-short HML sleeve."},
        ],
    },
    {
        "id": "moving_average",
        "cls": rs.MovingAverageCrossoverStrategy,
        "cls_name": "MovingAverageCrossoverStrategy",
        "label": "Simple Moving Average",
        "formula": "score(t, i) = MA_short(P, t, i) / MA_long(P, t, i) − 1",
        "summary": "Cross-sectional moving-average crossover. Long names in the strongest uptrend (short MA above long MA), short the weakest.",
        "params": [
            {"name": "short_window", "type": "int", "default": 50,  "min": 5,  "max": 100, "step": 5,
             "help": "Span of the fast moving average."},
            {"name": "long_window",  "type": "int", "default": 200, "min": 50, "max": 400, "step": 10,
             "help": "Span of the slow moving average."},
        ],
        "engine_overrides": [_SHORT_QUANTILE_OVERRIDE],
    },
    {
        "id": "residual_momentum",
        "cls": rs.ResidualMomentumStrategy,
        "cls_name": "ResidualMomentumStrategy",
        "label": "Residual Momentum",
        "formula": (
            "ε_i(t) = r_i(t) − β_i(t)·r_m(t);   "
            "score(t, i) = ⟨ ε_i(τ) / σ̂_ε,i(τ) ⟩  for  τ ∈ [t − skip − lookback, t − skip]"
        ),
        "summary": (
            "Blitz, Huij & Martens (2011, JFE). Regress each stock's daily "
            "returns on the equal-weight benchmark over a rolling window, "
            "standardize the residuals by their own rolling std, then take "
            "the mean of those standardized residuals over a 12-1-style "
            "lookback (skipping the most recent month). Same direction as "
            "vanilla momentum but with market beta residualized away — "
            "survives the 2009-style momentum crash that punishes raw 12-1."
        ),
        "params": [
            {"name": "beta_window", "type": "int", "default": 252, "min": 63, "max": 504, "step": 21,
             "help": "Rolling window for market-beta regression (252 ≈ 1y)."},
            {"name": "lookback_days", "type": "int", "default": 126, "min": 21, "max": 252, "step": 21,
             "help": "Window for averaging standardized residuals (126 ≈ 6mo)."},
            {"name": "skip_days", "type": "int", "default": 21, "min": 0, "max": 63, "step": 1,
             "help": "Skip the most recent N days to avoid short-term reversal contamination."},
        ],
        "engine_overrides": [_SHORT_QUANTILE_OVERRIDE],
    },
    {
        "id": "customer_supplier_momentum",
        "cls": rs.CustomerSupplierMomentumStrategy,
        "cls_name": "CustomerSupplierMomentumStrategy",
        "label": "Customer-Supplier Momentum",
        "formula": "score(t, i) = mean over j ∈ customers(i) of ln( P[t, j] / P[t − lookback, j] )",
        "summary": "Cohen & Frazzini (2008). Long suppliers whose major customers had strong recent returns; short those tied to weak customers. Uses a curated supplier→customer map.",
        "params": [
            {"name": "customer_lookback_days", "type": "int", "default": 21, "min": 5, "max": 63, "step": 1,
             "help": "Recent-return window on the customer side (21 ≈ 1mo)."},
            {"name": "min_customers", "type": "int", "default": 1, "min": 1, "max": 5, "step": 1,
             "help": "Minimum customers with valid returns required to score a supplier."},
        ],
        "engine_overrides": [
            {**_SHORT_QUANTILE_OVERRIDE, "default": 0.20,
             "help": "Bottom fraction (suppliers tied to weak customers) sold short."},
        ],
    },
    {
        "id": "pead",
        "cls": rs.PEADStrategy,
        "cls_name": "PEADStrategy",
        "label": "Post-Earnings Announcement Drift",
        "formula": "score(t, i) = SUE(announce(i))   for t ∈ [announce(i), announce(i) + holding]",
        "summary": "Bernard & Thomas (1989). On each ticker's announcement date set score = SUE; hold for `holding_days`. Long top decile, short bottom decile.",
        "params": [
            {"name": "holding_days", "type": "int", "default": 60, "min": 5, "max": 90, "step": 5,
             "help": "Trading days to keep the SUE signal active after the announcement."},
        ],
        "engine_overrides": [
            {**_SHORT_QUANTILE_OVERRIDE, "default": 0.10,
             "help": "Bottom-decile (negative SUE) names sold short — recovers the canonical long-short PEAD sleeve."},
        ],
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
# Warmup — load the chosen data source into a ResearchDataset
# ---------------------------------------------------------------------------
def _normalise_source(source: str = None) -> str:
    if source is None:
        source = _STATE.get("data_source") or DEFAULT_SOURCE
    source = str(source).strip().lower()
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unknown data source: {source!r}")
    return source


def _normalize_wharton_columns(panel):
    """Wharton/Compustat tickers use dot notation (BRK.B, BF.B); the rest of
    the simulator (and the frontend) speaks dash notation (BRK-B). Rename
    columns at load time so downstream code can stay source-agnostic."""
    if panel is None or not hasattr(panel, "columns"):
        return panel
    rename = {c: c.replace(".", "-") for c in panel.columns if isinstance(c, str)}
    if rename:
        panel = panel.rename(columns=rename)
    return panel


def _load_wharton_dataset(start: str, end: str) -> ResearchDataset:
    if not WHARTON_PARQUET.exists():
        raise RuntimeError(
            f"Wharton parquet not found at {WHARTON_PARQUET}. "
            "Switch back to the yfinance source or restore the file."
        )
    dataset = load_wharton_research_dataset(
        file_path=str(WHARTON_PARQUET),
        start_date=start,
        end_date=end,
        tickers=None,                # load full universe in the parquet
        use_total_return=True,
        min_history=21,
    )
    # Rename ticker columns to the dash convention used everywhere else.
    return ResearchDataset(
        prices=_normalize_wharton_columns(dataset.prices),
        returns=_normalize_wharton_columns(dataset.returns),
        benchmark_returns=dataset.benchmark_returns,
        volumes=_normalize_wharton_columns(dataset.volumes),
        dividends=_normalize_wharton_columns(dataset.dividends),
        eps=_normalize_wharton_columns(dataset.eps),
        market_caps=_normalize_wharton_columns(dataset.market_caps),
        metadata=(
            dataset.metadata.rename(
                index={i: i.replace(".", "-") for i in dataset.metadata.index if isinstance(i, str)}
            )
            if dataset.metadata is not None
            else None
        ),
        events=dataset.events,
    )


def warmup(start: str = "2005-01-01", end: str = None, source: str = None) -> None:
    if end is None:
        end = date.today().strftime("%Y-%m-%d")
    source = _normalise_source(source)
    with _LOCK:
        if _STATE["loading"] and _STATE.get("data_source") == source:
            return
        # If the requested source matches the already-loaded one, nothing to do.
        if _STATE["ready"] and _STATE.get("data_source") == source:
            return
        # Cache hit: promote the previously-loaded dataset for this source
        # back into _STATE without touching disk. ~instantaneous.
        cached = _DATASET_CACHE.get(source) if _DATASET_CACHE_ENABLED else None
        if cached is not None:
            _STATE.update({
                "ready": True,
                "loading": False,
                "message": "ready",
                "error": None,
                "dataset": cached["dataset"],
                "benchmark": cached["benchmark"],
                "date_min": cached["date_min"],
                "date_max": cached["date_max"],
                "tickers": cached["tickers"],
                "universe_label": cached["universe_label"],
                "data_source": source,
            })
            # Per-run cache was tied to the previous source's universe — drop it.
            _SIM_CACHE.clear()
            return
        token = int(_STATE.get("load_token", 0)) + 1
        _STATE["load_token"] = token
        _STATE["loading"] = True
        _STATE["ready"] = False
        # Drop the previous dataset reference so the GC can free it before
        # we allocate the new one — without this, peak memory during a
        # source switch ≈ size(old) + size(new), which OOMs a 1GB Fly box.
        if not _DATASET_CACHE_ENABLED:
            _STATE["dataset"] = None
            _STATE["benchmark"] = None
            _DATASET_CACHE.clear()
        _STATE["data_source"] = source
        _STATE["message"] = f"loading {source} dataset"
        _STATE["error"] = None
    try:
        if source == "yfinance":
            cached = list_cached_tickers()
            if not cached:
                raise RuntimeError(
                    "yfinance cache is empty — run `python run_sp500_fetch.py` first"
                )
            snap = load_sp500_universe()
            dataset = load_yfinance_research_dataset(
                start_date=start, end_date=end,
                tickers=cached,
                min_history=21,
                sp500_frame=snap.frame,
            )
            label = f"S&P 500 yfinance ({len(dataset.tickers)} tickers)"
        elif source == "wharton":
            dataset = _load_wharton_dataset(start=start, end=end)
            label = f"Wharton WRDS panel ({len(dataset.tickers)} tickers)"
        else:  # pragma: no cover — guarded above
            raise ValueError(f"unknown data source: {source!r}")

        # Switching sources invalidates the per-run cache (different prices,
        # different universe → identical request would otherwise stale-hit).
        _SIM_CACHE.clear()
        date_min = str(dataset.prices.index.min().date())
        date_max = str(dataset.prices.index.max().date())
        n_tickers = len(dataset.tickers)
        with _LOCK:
            if token != _STATE.get("load_token") or _STATE.get("data_source") != source:
                return
            _STATE.update({
                "ready": True, "loading": False, "message": "ready",
                "dataset": dataset, "benchmark": dataset.benchmark_returns,
                "date_min": date_min, "date_max": date_max,
                "tickers": n_tickers,
                "universe_label": label,
                "data_source": source,
            })
            # Memoize for subsequent toggles back to this source — only when
            # the env flag is set. Off by default to keep the prod VM under
            # its memory cap; turn on locally for snappy source toggles.
            if _DATASET_CACHE_ENABLED:
                _DATASET_CACHE[source] = {
                    "dataset": dataset,
                    "benchmark": dataset.benchmark_returns,
                    "date_min": date_min,
                    "date_max": date_max,
                    "tickers": n_tickers,
                    "universe_label": label,
                }
    except Exception as exc:
        with _LOCK:
            if token != _STATE.get("load_token") or _STATE.get("data_source") != source:
                return
            _STATE["loading"] = False
            _STATE["error"] = str(exc)
            _STATE["message"] = f"error: {exc}"
        raise


def ensure_ready(source: str = None, timeout: float = 120.0) -> None:
    """Block until the requested data source is loaded on this worker."""
    source = _normalise_source(source)
    deadline = time.monotonic() + timeout
    while True:
        with _LOCK:
            ready = _STATE["ready"] and _STATE.get("data_source") == source
            loading = _STATE["loading"] and _STATE.get("data_source") == source
            error = _STATE["error"] if _STATE.get("data_source") == source else None
        if ready:
            return
        if error and not loading:
            raise RuntimeError(error)
        if time.monotonic() > deadline:
            raise RuntimeError(f"timed out loading {source} dataset")
        if loading:
            time.sleep(0.25)
            continue
        warmup(source=source)


def status(source: str = None) -> Dict[str, Any]:
    requested_source = _normalise_source(source) if source else None
    with _LOCK:
        payload = {
            "ready": _STATE["ready"], "loading": _STATE["loading"],
            "message": _STATE["message"], "error": _STATE["error"],
            "date_min": _STATE["date_min"], "date_max": _STATE["date_max"],
            "tickers": _STATE["tickers"],
            "universe_label": _STATE["universe_label"],
            "data_source": _STATE.get("data_source", DEFAULT_SOURCE),
            "available_sources": list(SUPPORTED_SOURCES),
        }
    if requested_source and payload["data_source"] != requested_source:
        payload["ready"] = False
        payload["loading"] = True
        payload["message"] = f"loading {requested_source} dataset"
        payload["error"] = None
        payload["date_min"] = None
        payload["date_max"] = None
        payload["tickers"] = 0
        payload["universe_label"] = None
        payload["data_source"] = requested_source
    return payload


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
            "formula": entry.get("formula", ""),
            "params": entry["params"],
            "engine_overrides": entry.get("engine_overrides", []),
            "source": src,
            "source_file": "strategies/research_strategies.py",
        })
    return {
        "strategies": strategies,
        "engine_params": ENGINE_PARAMS,
        "fixed_engine": FIXED_ENGINE,
        "data_source": _STATE.get("data_source", DEFAULT_SOURCE),
        "available_sources": list(SUPPORTED_SOURCES),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _metrics(returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    clean = returns.dropna()
    if len(clean) < 2:
        return {}
    cum = (1 + clean).cumprod()
    ann_ret = float(cum.iloc[-1] ** (252 / len(clean)) - 1)
    std = float(clean.std(ddof=0))
    ann_vol = float(std * np.sqrt(252))
    sharpe = float(clean.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    dd = float((cum / cum.cummax() - 1).min())

    # Sortino — only penalize downside deviation. Treat zero-only-positives
    # as infinite Sortino by capping at the Sharpe value (shouldn't happen
    # on a real strategy but guards against div-by-zero).
    downside = clean[clean < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
    sortino = float(clean.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # Calmar — annualized return / abs(max drawdown). Undefined when
    # drawdown is exactly zero; report None rather than infinity.
    calmar = float(ann_ret / abs(dd)) if dd < 0 else None

    # Profit factor — sum of positive returns / abs(sum of negatives)
    pos_sum = float(clean[clean > 0].sum())
    neg_sum = float(abs(clean[clean < 0].sum()))
    profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else None

    out: Dict[str, Any] = {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "win_rate": float((clean > 0).mean()),
        "hit_rate": float((clean > 0).mean()),       # alias — quants prefer "hit rate"
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
    }

    # Benchmark-relative — Information Ratio + Beta. Only meaningful when
    # we actually have an aligned benchmark series.
    if benchmark is not None:
        aligned = pd.concat([clean, benchmark], axis=1, join="inner").dropna()
        if len(aligned) >= 2:
            aligned.columns = ["s", "b"]
            active = aligned["s"] - aligned["b"]
            active_std = float(active.std(ddof=0))
            info_ratio = float(active.mean() / active_std * np.sqrt(252)) if active_std > 0 else 0.0
            bench_var = float(aligned["b"].var(ddof=0))
            beta = float(aligned[["s", "b"]].cov().iat[0, 1] / bench_var) if bench_var > 0 else 0.0
            out["info_ratio"] = info_ratio
            out["beta"] = beta
        else:
            out["info_ratio"] = None
            out["beta"] = None
    else:
        out["info_ratio"] = None
        out["beta"] = None

    return out


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
    if _STATE.get("data_source") == "wharton":
        raise RuntimeError(
            "Wharton dataset is static — no live fetch. Switch the data "
            "source to yfinance to add new tickers, or pick a subset of "
            "the existing Wharton universe via Apply/Exclude."
        )
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
        # Drop the yfinance dataset cache entry so warmup() re-reads disk
        # rather than handing back the pre-fetch panel that's missing the
        # tickers we just added.
        _DATASET_CACHE.pop("yfinance", None)
    _SIM_CACHE.clear()
    warmup()

    return {
        "requested": wanted,
        "already_cached": sorted(set(wanted) & cached_before),
        "fetched": sorted(fetched),
        "failed": failed,
        "universe_size_after": _STATE["tickers"],
    }


_MAX_OVERRIDE_SIZE = 64 * 1024  # 64 KB cap on user-edited source — keeps exec() bounded.


def _compile_strategy_override(source: str, fallback_cls):
    """Compile a user-edited SignalStrategy class definition and return the
    resulting class. Re-executes only the class body — module-level imports
    are pre-populated in the namespace so the user doesn't have to repeat
    them. Falls back to ``fallback_cls`` on parse errors with a helpful
    error message.
    """
    if not isinstance(source, str):
        raise ValueError("strategy_source_override must be a string")
    if len(source) > _MAX_OVERRIDE_SIZE:
        raise ValueError(
            f"strategy_source_override too large ({len(source)} bytes; max {_MAX_OVERRIDE_SIZE})"
        )
    # Strip ANY 'from ... import' / 'import' lines that would re-import the
    # module — the namespace already has everything they need, and re-imports
    # can shadow our SignalStrategy ABC, which breaks subclass detection.
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "pd": pd, "np": np,
        "pandas": pd, "numpy": np,
        "ResearchDataset": ResearchDataset,
    }
    # Re-export every name from research_strategies so the user can call
    # _cross_sectional_zscore, reference SignalStrategy, etc.
    from strategies import research_strategies as _rs_mod
    for attr in dir(_rs_mod):
        if not attr.startswith("_") or attr == "_cross_sectional_zscore" or attr == "_sector_neutral_zscore":
            namespace.setdefault(attr, getattr(_rs_mod, attr))
    namespace["SignalStrategy"] = _rs_mod.SignalStrategy
    try:
        compiled = compile(source, "<live-edit>", "exec")
        exec(compiled, namespace)
    except SyntaxError as exc:
        raise ValueError(f"syntax error in edited code (line {exc.lineno}): {exc.msg}") from exc
    except Exception as exc:
        raise ValueError(f"error executing edited code: {exc}") from exc
    # Find the SignalStrategy subclass the user defined (skip the imported
    # ABC itself, and skip the fallback class which is also in the namespace).
    candidates = [
        v for v in namespace.values()
        if isinstance(v, type)
        and issubclass(v, _rs_mod.SignalStrategy)
        and v is not _rs_mod.SignalStrategy
        and v is not fallback_cls
    ]
    if not candidates:
        raise ValueError(
            "edited code did not define a new SignalStrategy subclass. "
            "Keep the `class XxxStrategy(SignalStrategy):` declaration."
        )
    # Prefer the LAST candidate so a user editing multiple class blocks gets
    # the bottom-most one (matches Python class-redefinition semantics).
    return candidates[-1]


def run_simulation(body: Dict[str, Any]) -> Dict[str, Any]:
    requested_source = body.get("data_source") or body.get("source")
    if requested_source:
        ensure_ready(requested_source)
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
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"start date must be on or before end date ({start} > {end})")

    # Strategy constructor args (whitelisted to the catalog params).
    kw_allowed = {p["name"] for p in strat_entry["params"]}
    strat_kwargs = {k: strategy_params[k] for k in kw_allowed if k in strategy_params}

    # Live-code override: if the body carries an edited class definition,
    # compile it on the fly and use it instead of the catalog class. The
    # catalog params still drive the constructor — but the user's class may
    # accept fewer/more kwargs, so we filter to whatever the new class
    # actually exposes.
    source_override = body.get("strategy_source_override")
    edited_cls_name: Optional[str] = None
    if source_override:
        live_cls = _compile_strategy_override(source_override, fallback_cls=strat_entry["cls"])
        edited_cls_name = live_cls.__name__
        sig = inspect.signature(live_cls.__init__)
        accepted = {name for name in sig.parameters if name != "self"}
        live_kwargs = {k: v for k, v in strat_kwargs.items() if k in accepted}
        try:
            strat_instance = live_cls(**live_kwargs)
        except TypeError as exc:
            raise ValueError(f"edited class constructor rejected the catalog params: {exc}") from exc
    else:
        strat_instance = strat_entry["cls"](**strat_kwargs)

    # BacktestConfig: fixed defaults + global engine knobs (defaults seeded
    # from the catalog so the response config block is always populated, even
    # when the caller omits engine_params entirely) + per-strategy overrides.
    cfg_kwargs = dict(FIXED_ENGINE)
    for p in ENGINE_PARAMS:
        cfg_kwargs[p["name"]] = engine_params.get(p["name"], p["default"])
    for p in strat_entry.get("engine_overrides", []):
        cfg_kwargs[p["name"]] = engine_overrides_in.get(p["name"], p["default"])
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
        # forms portfolios instead of skipping every rebalance. Floor drops
        # to 1 for a single-ticker universe — otherwise the default floor of
        # 2 blocks every rebalance and the sim returns a flat zero curve.
        n_sel = len(selected_tickers)
        floor = 1 if n_sel <= 1 else 2
        cfg_kwargs["min_names"] = min(cfg_kwargs.get("min_names", 10), max(floor, n_sel // 4))
        # Also cap max_position_weight so the per-leg budget can fit even when
        # only a handful of names qualify.
        n_long = max(1, int(n_sel * float(cfg_kwargs["long_quantile"])))
        cfg_kwargs["max_position_weight"] = max(cfg_kwargs.get("max_position_weight", 0.08), 1.0 / n_long)

    config = rbt.BacktestConfig(**cfg_kwargs)

    key = _cache_key({
        "strategy_id": strategy_id, "strategy_params": strat_kwargs,
        "engine_params": cfg_kwargs, "start": start, "end": end,
        "tickers": selected_tickers,
        # Edited source goes into the cache key so two runs with different
        # edits don't collide. Use a hash so we don't bloat the key.
        "src_hash": hashlib.md5(source_override.encode()).hexdigest() if source_override else None,
    })
    if key in _SIM_CACHE:
        # Touch on hit so the LRU eviction order stays meaningful.
        _SIM_CACHE.move_to_end(key)
        return _SIM_CACHE[key]

    window = (start_ts, end_ts)

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

    # Drawdown curve — fraction underwater from running peak. Same formula
    # as backtester/tearsheet.py:38 but emitted as a JSON-friendly list.
    net_dd = (net_cum.add(1) / net_cum.add(1).cummax() - 1)
    bench_dd = (bench_cum.add(1) / bench_cum.add(1).cummax() - 1)

    turnover_ann = float(result.turnover.mean() * 252)
    tcost_drag_ann = float(result.transaction_costs.mean() * 252)

    # Monthly returns — port of backtester/tearsheet.py:76. Emit a list of
    # {year, month, ret} cells so the frontend can pivot it into a heatmap.
    monthly_ret = (
        result.net_returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    )
    monthly_payload = [
        {"year": int(d.year), "month": int(d.month), "ret": float(v)}
        for d, v in monthly_ret.items()
        if pd.notna(v)
    ]

    response = {
        "strategy": {
            "id": strategy_id,
            "label": strat_entry["label"],
            "edited_cls_name": edited_cls_name,
            "live_edit": bool(source_override),
        },
        "dates": [d.strftime("%Y-%m-%d") for d in result.net_returns.index],
        "cumulative_net": [float(v) for v in net_cum.values],
        "cumulative_benchmark": [float(v) for v in bench_cum.values],
        "cumulative_drawdown": [float(v) for v in net_dd.values],
        "cumulative_drawdown_benchmark": [float(v) for v in bench_dd.values],
        "monthly_returns": monthly_payload,
        "metrics_net": _metrics(
            result.net_returns,
            benchmark=benchmark.reindex(result.net_returns.index).fillna(0),
        ),
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
    # Bounded LRU: evict the oldest entries until we're back within the cap.
    while len(_SIM_CACHE) > _SIM_CACHE_MAX:
        _SIM_CACHE.popitem(last=False)
    return response


def signal_preview(body: Dict[str, Any]) -> Dict[str, Any]:
    """Run the strategy's score function on the latest available date and
    return the top-N longs / bottom-N shorts, with current price for
    display. Skips the full backtest — just a single forward pass through
    `strategy.generate(dataset).scores`.
    """
    requested_source = body.get("data_source") or body.get("source")
    if requested_source:
        ensure_ready(requested_source)
    with _LOCK:
        if not _STATE["ready"]:
            raise RuntimeError("simulator not ready — wait for warmup")
        dataset = _STATE["dataset"]

    strategy_id = body.get("strategy_id", "momentum")
    if strategy_id not in STRATEGY_BY_ID:
        raise ValueError(f"unknown strategy_id: {strategy_id}")
    strat_entry = STRATEGY_BY_ID[strategy_id]

    strategy_params = body.get("strategy_params", {}) or {}
    kw_allowed = {p["name"] for p in strat_entry["params"]}
    strat_kwargs = {k: strategy_params[k] for k in kw_allowed if k in strategy_params}
    strat_instance = strat_entry["cls"](**strat_kwargs)

    top_n = int(body.get("top_n") or 10)
    top_n = max(1, min(50, top_n))
    short_quantile = float(
        (body.get("engine_overrides") or {}).get("short_quantile", 0.0)
    )
    include_shorts = short_quantile > 0.0

    try:
        signal = strat_instance.generate(dataset)
    except Exception as exc:
        raise RuntimeError(f"strategy.generate failed: {exc}") from exc

    scores = signal.scores
    if scores is None or scores.empty:
        return {
            "as_of": None, "longs": [], "shorts": [], "include_shorts": include_shorts,
            "strategy_id": strategy_id, "n_universe": 0,
        }

    # Find the last row with at least N finite values; walk backwards a few
    # days if today's row is mostly NaN (e.g. weekends, holidays, partial
    # data on the trailing edge).
    last_row = None
    last_date = None
    for ts in scores.index[::-1]:
        row = scores.loc[ts].dropna()
        if len(row) >= max(top_n, 5):
            last_row = row
            last_date = ts
            break
    if last_row is None:
        return {
            "as_of": None, "longs": [], "shorts": [], "include_shorts": include_shorts,
            "strategy_id": strategy_id, "n_universe": 0,
        }

    sorted_desc = last_row.sort_values(ascending=False)
    n = len(sorted_desc)

    last_prices = dataset.prices.loc[last_date] if last_date in dataset.prices.index else dataset.prices.iloc[-1]

    def _row(ticker: str, score: float, rank: int) -> Dict[str, Any]:
        price = last_prices.get(ticker)
        return {
            "ticker": ticker,
            "score": float(score),
            "rank": rank,
            "last_price": float(price) if pd.notna(price) else None,
        }

    longs = [_row(t, s, i + 1) for i, (t, s) in enumerate(sorted_desc.head(top_n).items())]
    shorts: List[Dict[str, Any]] = []
    if include_shorts:
        # Bottom-N (worst scores). Rank from the bottom upwards: rank 1 = worst.
        bottom = sorted_desc.tail(top_n).iloc[::-1]
        shorts = [_row(t, s, i + 1) for i, (t, s) in enumerate(bottom.items())]

    return {
        "as_of": str(last_date.date()) if hasattr(last_date, "date") else str(last_date),
        "longs": longs,
        "shorts": shorts,
        "include_shorts": include_shorts,
        "strategy_id": strategy_id,
        "strategy_label": strat_entry["label"],
        "n_universe": int(n),
        "top_n": top_n,
    }


# ---------------------------------------------------------------------------
# Data inspector
# ---------------------------------------------------------------------------
def data_overview(source: str = None) -> Dict[str, Any]:
    """Per-ticker summary rows for the table view."""
    if source:
        ensure_ready(source)
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

    src = _STATE.get("data_source", DEFAULT_SOURCE)
    if src == "wharton":
        source_file = "backtester/WhartonDataSource4.parquet (Wharton WRDS)"
    else:
        source_file = "data_cache/yfinance/*.pkl (current S&P 500)"
    return {
        "source_file": source_file,
        "date_min": _STATE["date_min"],
        "date_max": _STATE["date_max"],
        "n_tickers": len(rows),
        "tickers": rows,
    }


def ticker_detail(ticker: str, source: str = None) -> Dict[str, Any]:
    """Price history + summary stats for a single ticker."""
    if source:
        ensure_ready(source)
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
