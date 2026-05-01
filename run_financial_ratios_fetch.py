"""Compile FinancialRaitio.csv → backtester/financial_ratios.parquet.

The Wharton WRDS Financial Ratios export is a 71MB CSV with 78 columns and
~150k rows. Parsing the CSV every warmup is slow; the simulator wants a
pivoted wide-form panel keyed by (date × ticker) per ratio. This script
runs once after each refresh of the source CSV and emits a parquet that
the backend can mmap in milliseconds.

Output schema (Fama & French 1992 / 2015 value sleeve inputs):
  index   = public_date (month-end Timestamp)
  columns = MultiIndex (ratio, ticker), where ratio ∈ {bm, ep, cfp, dy}
            bm  = book / market           (CSV column "bm", as-is)
            ep  = earnings / price        (1 / "pe_op_basic", winsorized)
            cfp = cash flow / price       (1 / "pcf",        winsorized)
            dy  = dividend yield          ("divyield" string "X.X%" → X.X/100)

Run: `python run_financial_ratios_fetch.py`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "FinancialRaitio.csv"
OUT_PATH = Path(__file__).resolve().parent / "backtester" / "financial_ratios.parquet"

# We only read what we need — a pure-Python read of the full 78-col file
# costs ~3s; trimming to 6 columns drops it to ~1s and halves the memory
# footprint, which matters on the Fly box during cold-start warmup.
USECOLS = ["Ticker", "public_date", "bm", "pe_op_basic", "pcf", "divyield"]


def _parse_divyield(series: pd.Series) -> pd.Series:
    """The CSV stores divyield as strings like "3.08%". Strip the %, divide
    by 100 so all four output ratios live on the same fraction-of-price scale."""
    cleaned = series.astype("string").str.rstrip("%").str.strip()
    return pd.to_numeric(cleaned, errors="coerce") / 100.0


def _winsorize(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Trim the 1% tails before inversion so a tiny earnings number doesn't
    explode 1/PE into a millennial outlier that dominates the cross-section
    z-score. We *clip* rather than drop to keep coverage."""
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return series
    lo = finite.quantile(lower_q)
    hi = finite.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"missing source: {CSV_PATH}")

    print(f"reading {CSV_PATH.name} ({CSV_PATH.stat().st_size / 1e6:.0f} MB)…")
    df = pd.read_csv(CSV_PATH, usecols=USECOLS, low_memory=False)
    print(f"  raw rows: {len(df):,}")

    df = df.dropna(subset=["Ticker", "public_date"])
    df["public_date"] = pd.to_datetime(df["public_date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df[df["Ticker"] != ""]

    # Wharton tickers occasionally use dot-class notation (BRK.B). The rest
    # of the simulator speaks dash notation (BRK-B); align here so a join on
    # Ticker matches the price panel.
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)

    # Some (gvkey, public_date) pairs collide — same ticker, multiple parents
    # in the WRDS dump. Keep the row with the most non-null ratio fields.
    df["_score"] = df[["bm", "pe_op_basic", "pcf", "divyield"]].notna().sum(axis=1)
    df = (
        df.sort_values(["Ticker", "public_date", "_score"])
        .drop_duplicates(subset=["Ticker", "public_date"], keep="last")
        .drop(columns="_score")
    )
    print(f"  deduped rows: {len(df):,}")
    print(f"  unique tickers: {df['Ticker'].nunique():,}")
    print(f"  date range: {df['public_date'].min().date()} → {df['public_date'].max().date()}")

    df["dy"] = _parse_divyield(df["divyield"])
    df["ep"] = _winsorize(1.0 / df["pe_op_basic"].replace(0, np.nan))
    df["cfp"] = _winsorize(1.0 / df["pcf"].replace(0, np.nan))
    df["bm_w"] = _winsorize(df["bm"])

    panels = {
        "bm": df.pivot(index="public_date", columns="Ticker", values="bm_w"),
        "ep": df.pivot(index="public_date", columns="Ticker", values="ep"),
        "cfp": df.pivot(index="public_date", columns="Ticker", values="cfp"),
        "dy": df.pivot(index="public_date", columns="Ticker", values="dy"),
    }

    # Stitch each ratio panel into a single MultiIndex-column DataFrame so
    # the backend can read the file once and slice ratios by name.
    combined = pd.concat(panels, axis=1)
    combined.columns.names = ["ratio", "ticker"]
    combined = combined.sort_index().sort_index(axis=1)
    print(f"  combined shape: {combined.shape}  (months × ratio·tickers)")
    print(f"  per-ratio coverage:")
    for r in panels:
        cov = combined[r].notna().mean().mean() * 100
        print(f"    {r}: {cov:.1f}% non-null")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"wrote {OUT_PATH.relative_to(Path(__file__).resolve().parent)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
