"""Current S&P 500 constituents, scraped from Wikipedia and cached locally.

This gives us a survivor-biased universe (today's members only), which is the
same bias the Wharton parquet has. Historical membership changes would need
a paid source (e.g. Compustat Index Constituents). Documented honestly.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

CACHE_FILE = Path(__file__).resolve().parent.parent / "sp500_universe.pkl"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass
class SP500Snapshot:
    fetched_at: str
    tickers: list[str]
    frame: pd.DataFrame  # columns: ticker, security, sector, subindustry, hq, date_added, cik, founded


def _scrape_wikipedia() -> pd.DataFrame:
    req = Request(WIKI_URL, headers={"User-Agent": "millennium-data-quality/1.0"})
    html = urlopen(req, timeout=30).read()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"}) or soup.find("table", class_="wikitable")
    if table is None:
        raise RuntimeError("Wikipedia S&P 500 table not found")

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if len(cells) >= 4:
            rows.append(cells[:8] + [""] * (8 - len(cells)))
    if not rows:
        raise RuntimeError("Wikipedia S&P 500 table parsed empty")

    cols = ["ticker", "security", "sector", "subindustry", "hq", "date_added", "cik", "founded"]
    frame = pd.DataFrame(rows, columns=cols)
    frame["ticker"] = frame["ticker"].str.replace(".", "-", regex=False).str.upper()
    frame = frame.drop_duplicates("ticker").reset_index(drop=True)
    return frame


def load_sp500_universe(refresh: bool = False) -> SP500Snapshot:
    if not refresh and CACHE_FILE.exists():
        with CACHE_FILE.open("rb") as fh:
            return pickle.load(fh)
    frame = _scrape_wikipedia()
    snap = SP500Snapshot(
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tickers=frame["ticker"].tolist(),
        frame=frame,
    )
    with CACHE_FILE.open("wb") as fh:
        pickle.dump(snap, fh)
    return snap


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    snap = load_sp500_universe(refresh=args.refresh)
    print(f"fetched_at: {snap.fetched_at}")
    print(f"tickers: {len(snap.tickers)}")
    print(snap.frame.head().to_string(index=False))
