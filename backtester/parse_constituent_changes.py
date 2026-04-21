"""
Parse `Constituent_changes.xlsx` (S&P 500 constituent changes) without external deps.

The workbook in this repo has a single sheet with the layout:
  A: Effective Date (Excel serial)
  B: Added Ticker
  C: Added Security
  D: Removed Ticker
  E: Removed Security
  F: Reason

This script extracts:
  - a row-level CSV of changes
  - unique tickers (WRDS format and yfinance format)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS_MAIN = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _excel_serial_to_date(serial: str) -> str:
    """
    Convert Excel 1900-date-system serial to ISO date.
    Uses the common base of 1899-12-30.
    """
    try:
        days = int(float(serial))
    except Exception:
        return ""
    base = dt.date(1899, 12, 30)
    return (base + dt.timedelta(days=days)).isoformat()


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    xml = zf.read("xl/sharedStrings.xml")
    root = ET.fromstring(xml)
    strings: list[str] = []
    for si in root.findall("m:si", NS_MAIN):
        # shared strings can have <t> or rich text with <r><t>
        texts = [t.text or "" for t in si.findall(".//m:t", NS_MAIN)]
        strings.append("".join(texts))
    return strings


_CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _parse_sheet_cells(zf: zipfile.ZipFile, shared_strings: list[str]) -> list[dict[str, str]]:
    xml = zf.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(xml)

    records: list[dict[str, str]] = []

    sheet_data = root.find("m:sheetData", NS_MAIN)
    if sheet_data is None:
        return records

    for row in sheet_data.findall("m:row", NS_MAIN):
        row_idx = row.attrib.get("r", "")
        if not row_idx.isdigit() or int(row_idx) <= 2:
            continue  # skip header rows

        values: dict[str, str] = {}
        for cell in row.findall("m:c", NS_MAIN):
            ref = cell.attrib.get("r")
            if not ref:
                continue
            m = _CELL_REF_RE.match(ref)
            if not m:
                continue
            col = m.group(1)
            if col not in {"A", "B", "D", "F"}:
                continue

            v = cell.find("m:v", NS_MAIN)
            if v is None or v.text is None:
                continue

            cell_type = cell.attrib.get("t")
            raw = v.text
            if cell_type == "s":
                try:
                    values[col] = shared_strings[int(raw)]
                except Exception:
                    continue
            else:
                values[col] = raw

        # Only store rows that have at least one of added/removed tickers.
        added = (values.get("B") or "").strip()
        removed = (values.get("D") or "").strip()
        if not added and not removed:
            continue

        effective = _excel_serial_to_date(values.get("A", ""))
        reason = (values.get("F") or "").strip()

        records.append(
            {
                "effective_date": effective,
                "added_ticker": added,
                "removed_ticker": removed,
                "reason": reason,
            }
        )

    return records


def _yfinance_ticker(ticker: str) -> str:
    # Common mapping: BRK.B -> BRK-B, BF.B -> BF-B, etc.
    return ticker.replace(".", "-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx",
        default=str(Path(__file__).with_name("Constituent_changes.xlsx")),
        help="Path to Constituent_changes.xlsx",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory to write output artifacts",
    )
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing xlsx: {xlsx_path}")

    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared_strings = _load_shared_strings(zf)
        records = _parse_sheet_cells(zf, shared_strings)

    # Artifacts
    parsed_csv = out_dir / "constituent_changes_parsed.csv"
    wrds_txt = out_dir / "constituent_change_tickers_wrds.txt"
    yfin_txt = out_dir / "constituent_change_tickers_yfinance.txt"

    with parsed_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["effective_date", "added_ticker", "removed_ticker", "reason"]
        )
        w.writeheader()
        w.writerows(records)

    tickers_wrds: set[str] = set()
    for r in records:
        if r["added_ticker"]:
            tickers_wrds.add(r["added_ticker"])
        if r["removed_ticker"]:
            tickers_wrds.add(r["removed_ticker"])

    tickers_wrds_sorted = sorted(tickers_wrds)
    tickers_yfin_sorted = sorted({_yfinance_ticker(t) for t in tickers_wrds_sorted})

    wrds_txt.write_text("\n".join(tickers_wrds_sorted) + "\n", encoding="utf-8")
    yfin_txt.write_text("\n".join(tickers_yfin_sorted) + "\n", encoding="utf-8")

    print(f"Wrote: {parsed_csv}")
    print(f"Wrote: {wrds_txt} ({len(tickers_wrds_sorted)} tickers)")
    print(f"Wrote: {yfin_txt} ({len(tickers_yfin_sorted)} tickers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

