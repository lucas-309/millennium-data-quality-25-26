from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import pandas as pd

from .data_source import WhartonDataSource, YahooFinanceDataSource
from .research_backtester import BacktestResult
from .research_data import ResearchDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPY_TICKER_REMAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "-": "USD",
}


def generate_data_quality_report(
    source: WhartonDataSource,
    start_date: str,
    end_date: str,
    tickers: Optional[list[str]] = None,
) -> Dict[str, Any]:
    raw = source.get_raw_data(tickers=tickers, start_date=start_date, end_date=end_date)
    if raw.empty:
        return {"row_count": 0}

    key_columns = [column for column in ["prccd", "ajexdi", "trfd", "cshtrd", "divd", "eps"] if column in raw.columns]
    split_adjusted = raw["split_adjusted_close"] if "split_adjusted_close" in raw.columns else pd.Series(dtype=float)
    total_return_reference = raw["total_return_reference"] if "total_return_reference" in raw.columns else pd.Series(dtype=float)

    return {
        "row_count": int(len(raw)),
        "ticker_count": int(raw["tic"].nunique()),
        "start_date": str(raw["datadate"].min().date()),
        "end_date": str(raw["datadate"].max().date()),
        "duplicate_ticker_dates": int(raw.duplicated(subset=["tic", "datadate"]).sum()),
        "missing_by_field": raw[key_columns].isna().sum().to_dict() if key_columns else {},
        "non_positive_close_count": int((raw["prccd"] <= 0).fillna(False).sum()) if "prccd" in raw.columns else 0,
        "split_adjustment_events": int((raw.get("ajexdi", pd.Series(dtype=float)).fillna(1.0) != 1.0).sum()),
        "dividend_event_count": int((raw.get("divd", pd.Series(dtype=float)).fillna(0.0) > 0).sum()),
        "volume_missing_count": int(raw.get("cshtrd", pd.Series(dtype=float)).isna().sum()) if "cshtrd" in raw.columns else 0,
        "split_adjusted_non_positive_count": int((split_adjusted <= 0).fillna(False).sum()) if not split_adjusted.empty else 0,
        "total_return_reference_non_positive_count": int((total_return_reference <= 0).fillna(False).sum()) if not total_return_reference.empty else 0,
    }


def validate_spy_reconstruction(
    holdings_file: str,
    start_date: str,
    end_date: str,
    data_source: Optional[YahooFinanceDataSource] = None,
) -> Dict[str, Any]:
    source = data_source or YahooFinanceDataSource()
    holdings = source.read_spy_holdings(holdings_file)
    if holdings.empty:
        return {"error": f"Unable to read holdings file: {holdings_file}"}

    holdings = holdings.copy()
    holdings["Ticker"] = holdings["Ticker"].replace(SPY_TICKER_REMAP)

    tickers = ["SPY"] + holdings["Ticker"].tolist()
    price_data = source.get_historical_data(tickers, start_date, end_date)
    if price_data.empty or "SPY" not in price_data.columns:
        return {"error": "Unable to load SPY benchmark series for reconstruction."}

    weighted_portfolio = source.calculate_weighted_portfolio(holdings, price_data.drop(columns=["SPY"], errors="ignore"))
    verification = source.verify_spy_vs_constituents(price_data["SPY"], weighted_portfolio)

    summary = {
        "constituent_count": int(len(holdings)),
        "available_constituent_count": int(sum(1 for ticker in holdings["Ticker"] if ticker in price_data.columns)),
        "within_threshold": bool(verification.get("within_threshold", False)),
        "max_difference": float(verification.get("max_difference", float("nan"))),
        "mean_difference": float(verification.get("mean_difference", float("nan"))),
        "pearson_coeff": float(verification.get("pearson_coeff", float("nan"))),
        "spearman_coeff": float(verification.get("spearman_coeff", float("nan"))),
    }

    worst_days = verification.get("worst_days")
    if isinstance(worst_days, pd.Series):
        summary["worst_days"] = {str(index.date()): float(value) for index, value in worst_days.items()}

    return summary


def plot_portfolio_curves(results: Dict[str, BacktestResult], output_dir: Path) -> None:
    if not results:
        return

    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        cumulative = result.portfolio_value / result.portfolio_value.iloc[0] - 1.0
        plt.plot(cumulative.index, cumulative, label=name, linewidth=2)

    plt.title("Cumulative Portfolio Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "cumulative_returns.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_event_study(event_study: pd.DataFrame, title: str, output_path: Path) -> None:
    if event_study is None or event_study.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(event_study.index, event_study["median"], label="Median", linewidth=2)
    plt.plot(event_study.index, event_study["p25"], label="P25", linestyle="--", alpha=0.8)
    plt.plot(event_study.index, event_study["p75"], label="P75", linestyle="--", alpha=0.8)
    plt.plot(event_study.index, event_study["p10"], label="P10", linestyle=":", alpha=0.7)
    plt.plot(event_study.index, event_study["p90"], label="P90", linestyle=":", alpha=0.7)
    plt.axvline(0, color="black", linewidth=1, alpha=0.5)
    plt.title(title)
    plt.xlabel("Relative Day")
    plt.ylabel("Cumulative Residual Return")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def export_research_outputs(
    dataset: ResearchDataset,
    results: Dict[str, BacktestResult],
    signal_correlation: pd.DataFrame,
    output_dir: str,
    data_quality_report: Optional[Dict[str, Any]] = None,
    spy_validation: Optional[Dict[str, Any]] = None,
) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([data_quality_report or {}]).to_csv(target_dir / "data_quality_report.csv", index=False)
    pd.DataFrame([spy_validation or {}]).to_csv(target_dir / "spy_validation.csv", index=False)
    signal_correlation.to_csv(target_dir / "signal_correlation.csv")
    dataset.benchmark_returns.to_csv(target_dir / "benchmark_returns.csv", header=True)

    summary_rows = []
    for name, result in results.items():
        metrics_row = {"strategy": name}
        metrics_row.update(result.metrics)
        summary_rows.append(metrics_row)

        result.portfolio_value.rename("portfolio_value").to_csv(target_dir / f"{name}_portfolio_value.csv")
        result.net_returns.rename("net_returns").to_csv(target_dir / f"{name}_net_returns.csv")
        result.turnover.rename("turnover").to_csv(target_dir / f"{name}_turnover.csv")
        changed_rows = result.target_weights.ne(result.target_weights.shift()).any(axis=1)
        weight_snapshots = result.target_weights.loc[changed_rows].copy()
        weight_snapshots.to_csv(target_dir / f"{name}_weights.csv")
        if result.lag_table is not None:
            result.lag_table.to_csv(target_dir / f"{name}_lag_table.csv")
        if result.event_study is not None:
            result.event_study.to_csv(target_dir / f"{name}_event_study.csv")
            try:
                plot_event_study(result.event_study, f"{name} Event Study", target_dir / f"{name}_event_study.png")
            except Exception:
                pass

    pd.DataFrame(summary_rows).to_csv(target_dir / "strategy_summary.csv", index=False)
    try:
        plot_portfolio_curves(results, target_dir)
    except Exception:
        pass
