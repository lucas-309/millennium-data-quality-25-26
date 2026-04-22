"""Walk through a concrete split-adjustment example from the Wharton parquet.

The mentor asked for NVDA's 10-for-1 split on 2024-06-10. NVDA is not in the
167-ticker S&P subset we received, so we use AAPL's 4-for-1 split on
2020-08-31 — same mechanics, same adjustment math.

Prints a before/after table of the raw Compustat fields and the derived
split-adjusted / total-return price, so the deck can show the formula
applied to real data.

Usage:
  python run_split_demo.py                       # AAPL 4-for-1 around 2020-08-31
  python run_split_demo.py --ticker MSFT         # inspect another name
  python run_split_demo.py --all-splits          # list every split in the panel
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtester.data_source import WhartonResearchDataSource

DATA_PATH = Path(__file__).resolve().parent / "backtester" / "WhartonDataSource.parquet"


def _load_raw(ticker: str) -> pd.DataFrame:
    source = WhartonResearchDataSource(file_path=str(DATA_PATH), use_total_return=True)
    raw = source.get_raw_data(
        tickers=[ticker],
        start_date="1990-01-01",
        end_date="2026-01-01",
        columns=["prccd", "ajexdi", "trfd", "cshoc",
                 "split_adjusted_close", "total_return_reference"],
    )
    raw["datadate"] = pd.to_datetime(raw["datadate"])
    return raw.sort_values("datadate").reset_index(drop=True)


def _detect_splits(raw: pd.DataFrame, threshold: float = 1.2) -> pd.DataFrame:
    """A split is a step change in the adjustment factor AJEXDI.

    Ratio = ajexdi_t / ajexdi_{t-1}. For a 4-for-1 split ratio ≈ 4.
    """
    if raw.empty:
        return raw
    df = raw.copy()
    df["ajexdi_prev"] = df["ajexdi"].shift(1)
    df["split_ratio"] = df["ajexdi"] / df["ajexdi_prev"]
    return df[df["split_ratio"] >= threshold][["datadate", "ajexdi_prev", "ajexdi", "split_ratio"]]


def _window(raw: pd.DataFrame, date: str, days: int = 3) -> pd.DataFrame:
    ts = pd.Timestamp(date)
    mask = (raw["datadate"] >= ts - pd.Timedelta(days=days))
    mask &= (raw["datadate"] <= ts + pd.Timedelta(days=days))
    return raw.loc[mask]


def _format_row(r: pd.Series) -> str:
    return (
        f"  {r['datadate'].date()}  "
        f"prccd={r['prccd']:>9.2f}  "
        f"ajexdi={r['ajexdi']:>7.4f}  "
        f"trfd={r['trfd']:>6.3f}  "
        f"split_adj={r['split_adjusted_close']:>8.4f}  "
        f"total_ret={r['total_return_reference']:>9.4f}"
    )


def demo(ticker: str, split_date: str, ratio_label: str) -> None:
    raw = _load_raw(ticker)
    if raw.empty:
        print(f"ticker {ticker} not present in the Wharton parquet")
        return

    rows = _window(raw, split_date, days=4)
    print(f"\n=== {ticker}: {ratio_label} stock split on {split_date} ===")
    print("Raw Compustat fields (prccd, ajexdi, trfd) and derived adjusted series:\n")
    for _, row in rows.iterrows():
        print(_format_row(row))

    before = rows[rows["datadate"] < split_date].iloc[-1] if (rows["datadate"] < split_date).any() else None
    after = rows[rows["datadate"] >= split_date].iloc[0] if (rows["datadate"] >= split_date).any() else None
    if before is not None and after is not None:
        raw_drop = after["prccd"] / before["prccd"]
        adj_drop = after["split_adjusted_close"] / before["split_adjusted_close"]
        print(
            f"\nRaw PRCCD step: {before['prccd']:.2f} → {after['prccd']:.2f} "
            f"({raw_drop:.3f}×)  — the {ratio_label} split is visible unadjusted."
        )
        print(
            f"Split-adjusted step: {before['split_adjusted_close']:.4f} → "
            f"{after['split_adjusted_close']:.4f}  ({adj_drop:.3f}×)  "
            f"— the split is fully neutralized."
        )
    print("\nFormulas (see backtester/data_source.py:348,352):")
    print("  split_adjusted_close   = PRCCD / AJEXDI")
    print("  total_return_reference = (PRCCD / AJEXDI) * TRFD")


def list_all_splits(ticker: str) -> None:
    raw = _load_raw(ticker)
    if raw.empty:
        print(f"ticker {ticker} not present in the Wharton parquet")
        return
    splits = _detect_splits(raw)
    if splits.empty:
        print(f"no splits detected for {ticker}")
        return
    print(f"\n=== Splits detected for {ticker} ===")
    for _, row in splits.iterrows():
        print(f"  {row['datadate'].date()}  "
              f"ajexdi {row['ajexdi_prev']:.4f} → {row['ajexdi']:.4f}  "
              f"≈ {row['split_ratio']:.2f}-for-1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--split-date", default="2020-08-31",
                    help="demo split date (default: AAPL 4-for-1 on 2020-08-31)")
    ap.add_argument("--ratio-label", default="4-for-1")
    ap.add_argument("--all-splits", action="store_true",
                    help="list all splits in the ticker's history")
    args = ap.parse_args()

    print("NVDA's 10-for-1 on 2024-06-10 is the mentor's canonical example.")
    print("NVDA is absent from the 167-ticker Wharton pull we received, so we")
    print(f"demonstrate the identical mechanics on {args.ticker} instead.\n")

    if args.all_splits:
        list_all_splits(args.ticker)
    else:
        demo(args.ticker, args.split_date, args.ratio_label)


if __name__ == "__main__":
    main()
