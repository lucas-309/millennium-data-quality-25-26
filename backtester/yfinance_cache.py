"""Rate-limit-aware yfinance cache keyed by ticker.

Design goals (in priority order):
  1. Never hammer yfinance. Default throttling is 0.6s between tickers, with
     exponential backoff on 429/JSONDecode errors.
  2. Be resumable. Each ticker is its own pickle file, so a crash mid-run loses
     nothing already fetched.
  3. Be forgiving. Tickers that fail after max retries are recorded in
     ``_failed.json`` so the next run can retry selectively (or skip).

Layout:
  data_cache/yfinance/
    AAPL.pkl        # {"history": DataFrame, "dividends": Series, "splits": Series,
                    #  "info": dict, "fetched_at": ISO ts, "start": ..., "end": ...}
    MSFT.pkl
    ...
    _failed.json    # {ticker: {"error": ..., "attempts": n, "last_tried": ts}}

Each ``history`` DataFrame is the full auto_adjust=False yfinance output with
columns Open/High/Low/Close/Adj Close/Volume indexed by date.
"""
from __future__ import annotations

import json
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as curl_requests  # type: ignore
except ImportError:  # pragma: no cover
    curl_requests = None  # type: ignore

CACHE_ROOT = Path(__file__).resolve().parent.parent / "data_cache" / "yfinance"
FAILED_FILE = CACHE_ROOT / "_failed.json"


def _make_session():
    """A browser-fingerprinted session is required since yfinance tightened its
    public endpoints in 2024. curl_cffi's chrome impersonation is the
    community-standard fix — without it every call returns 429.
    """
    if curl_requests is None:
        return None
    return curl_requests.Session(impersonate="chrome")


@dataclass
class FetchResult:
    ticker: str
    ok: bool
    path: Optional[Path] = None
    error: Optional[str] = None
    source: str = "fresh"  # "fresh" | "cache" | "skipped"


def _ensure_dir() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str) -> Path:
    return CACHE_ROOT / f"{ticker.upper()}.pkl"


def _read_failed() -> dict:
    if FAILED_FILE.exists():
        try:
            return json.loads(FAILED_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_failed(d: dict) -> None:
    _ensure_dir()
    FAILED_FILE.write_text(json.dumps(d, indent=2, sort_keys=True))


def _record_failure(ticker: str, error: str) -> None:
    failed = _read_failed()
    prev = failed.get(ticker, {"attempts": 0})
    failed[ticker] = {
        "error": error,
        "attempts": prev.get("attempts", 0) + 1,
        "last_tried": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_failed(failed)


def _clear_failure(ticker: str) -> None:
    failed = _read_failed()
    if ticker in failed:
        del failed[ticker]
        _write_failed(failed)


def _fetch_one(
    ticker: str,
    start: str,
    end: str,
    pause_base: float = 0.6,
    max_retries: int = 4,
    fetch_info: bool = True,
    session=None,
) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            tk = yf.Ticker(ticker, session=session) if session is not None else yf.Ticker(ticker)
            hist = tk.history(
                start=start, end=end, auto_adjust=False, actions=True,
                raise_errors=False, timeout=30,
            )
            if hist.empty:
                raise RuntimeError("yfinance returned empty history")
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            dividends = tk.dividends
            if dividends.index.tz is not None:
                dividends.index = dividends.index.tz_localize(None)
            splits = tk.splits
            if splits.index.tz is not None:
                splits.index = splits.index.tz_localize(None)
            info: dict = {}
            if fetch_info:
                try:
                    info = dict(tk.info) if tk.info else {}
                except Exception:  # info API is notoriously flaky
                    info = {}
            return {
                "history": hist,
                "dividends": dividends,
                "splits": splits,
                "info": info,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "start": start,
                "end": end,
            }
        except Exception as exc:
            last_error = exc
            delay = pause_base * (2 ** attempt) + random.uniform(0, pause_base)
            time.sleep(delay)
    raise RuntimeError(f"{ticker}: {type(last_error).__name__}: {last_error}")


def fetch_tickers(
    tickers: Iterable[str],
    start: str = "2005-01-01",
    end: Optional[str] = None,
    pause_seconds: float = 0.6,
    max_retries: int = 4,
    force: bool = False,
    skip_known_failures: bool = True,
    fetch_info: bool = True,
    progress_every: int = 25,
) -> list[FetchResult]:
    """Download ticker data with caching. Safe to run repeatedly."""
    _ensure_dir()
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    tickers = [t.upper().replace(".", "-") for t in tickers]  # yf uses '-' for class shares
    failed = _read_failed() if skip_known_failures else {}
    results: list[FetchResult] = []
    session = _make_session()
    if session is None:
        print("warning: curl_cffi not installed — yfinance will likely rate-limit. "
              "Run `pip install curl_cffi` and retry.")

    for i, ticker in enumerate(tickers, 1):
        path = _cache_path(ticker)
        if path.exists() and not force:
            results.append(FetchResult(ticker, ok=True, path=path, source="cache"))
            continue
        if skip_known_failures and ticker in failed and failed[ticker].get("attempts", 0) >= 3:
            results.append(FetchResult(
                ticker, ok=False,
                error=f"skipped after {failed[ticker]['attempts']} prior failures",
                source="skipped",
            ))
            continue

        try:
            payload = _fetch_one(
                ticker, start=start, end=end,
                pause_base=pause_seconds, max_retries=max_retries,
                fetch_info=fetch_info, session=session,
            )
            with path.open("wb") as fh:
                pickle.dump(payload, fh)
            _clear_failure(ticker)
            results.append(FetchResult(ticker, ok=True, path=path, source="fresh"))
        except Exception as exc:
            _record_failure(ticker, str(exc))
            results.append(FetchResult(ticker, ok=False, error=str(exc)))

        if i % progress_every == 0:
            ok = sum(1 for r in results if r.ok)
            print(f"  progress {i}/{len(tickers)}: {ok} ok, {i - ok} failed")

        if results[-1].source == "fresh":
            time.sleep(pause_seconds + random.uniform(0, pause_seconds * 0.5))

    return results


def load_cached(ticker: str) -> Optional[dict]:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def list_cached_tickers() -> list[str]:
    _ensure_dir()
    return sorted(p.stem for p in CACHE_ROOT.glob("*.pkl") if not p.name.startswith("_"))


def cache_summary() -> dict:
    tickers = list_cached_tickers()
    failed = _read_failed()
    return {
        "cached": len(tickers),
        "failed_tracked": len(failed),
        "cache_root": str(CACHE_ROOT),
    }
