"""Bulk-download the current S&P 500 into the yfinance cache.

Safe to run repeatedly — already-cached tickers are skipped. Pass --force to
re-fetch. Tickers that fail max-retry attempts are written to
``data_cache/yfinance/_failed.json`` and skipped on subsequent runs until the
record is cleared.

Usage:
  python run_sp500_fetch.py                              # default 2005-01-01..today
  python run_sp500_fetch.py --start 2010-01-01
  python run_sp500_fetch.py --tickers AAPL MSFT          # download just a subset
  python run_sp500_fetch.py --refresh-universe           # re-scrape Wikipedia
  python run_sp500_fetch.py --pause 0.8                  # slower per-ticker pace
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from backtester.sp500_universe import load_sp500_universe
from backtester.yfinance_cache import cache_summary, fetch_tickers, list_cached_tickers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--pause", type=float, default=0.6, help="seconds between tickers")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    ap.add_argument("--no-info", action="store_true", help="skip Ticker.info calls (faster)")
    ap.add_argument("--refresh-universe", action="store_true", help="re-scrape Wikipedia")
    ap.add_argument("--tickers", nargs="+", help="override universe with this list")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"custom ticker list: {len(tickers)} symbols")
    else:
        snap = load_sp500_universe(refresh=args.refresh_universe)
        tickers = snap.tickers
        print(f"S&P 500 universe ({snap.fetched_at}): {len(tickers)} tickers")

    already = set(list_cached_tickers())
    fresh_count = sum(1 for t in tickers if t not in already) if not args.force else len(tickers)
    print(f"already cached: {len(already & set(tickers))}  | fresh to fetch: {fresh_count}")
    if fresh_count == 0 and not args.force:
        print("nothing to do")
        return

    t0 = time.time()
    results = fetch_tickers(
        tickers,
        start=args.start, end=args.end,
        pause_seconds=args.pause, max_retries=args.retries,
        force=args.force, fetch_info=not args.no_info,
    )
    elapsed = time.time() - t0

    ok = sum(1 for r in results if r.ok)
    fresh = sum(1 for r in results if r.source == "fresh")
    cached = sum(1 for r in results if r.source == "cache")
    failed = [r for r in results if not r.ok]

    print(f"\ndone in {elapsed:.1f}s — {ok}/{len(results)} ok  (fresh={fresh}, cached={cached}, failed={len(failed)})")
    if failed[:10]:
        print("first failures:")
        for r in failed[:10]:
            print(f"  {r.ticker}: {r.error}")
    print("cache summary:", cache_summary())


if __name__ == "__main__":
    main()
