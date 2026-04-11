"""Historical regime stress testing.

Replay the backtest across infamous drawdown periods and report the worst
outcomes. Every pod at a real shop gets tested against these regimes before
going live.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


HISTORICAL_REGIMES: Dict[str, Tuple[str, str]] = {
    "GFC_Lehman": ("2008-09-01", "2009-03-31"),
    "Flash_Crash": ("2010-05-03", "2010-05-31"),
    "EU_Debt_Crisis": ("2011-07-01", "2011-10-31"),
    "China_Devaluation": ("2015-08-01", "2015-09-30"),
    "Brexit": ("2016-06-17", "2016-07-15"),
    "Q4_2018_Selloff": ("2018-10-01", "2018-12-31"),
    "Volmageddon": ("2018-02-01", "2018-02-28"),
    "COVID_Crash": ("2020-02-15", "2020-04-30"),
    "2022_Rate_Shock": ("2022-01-01", "2022-10-31"),
    "SVB_Regional_Banks": ("2023-03-01", "2023-05-31"),
}


def regime_performance(
    returns: pd.Series,
    start: str,
    end: str,
) -> Dict[str, float]:
    """Compute performance metrics over a single regime window."""
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    sub = returns.loc[mask]
    if len(sub) < 2:
        return {
            "n_days": len(sub),
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "worst_day": np.nan,
            "worst_week": np.nan,
        }

    cum = (1 + sub).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1

    weekly = sub.rolling(5, min_periods=1).sum()

    n = len(sub)
    vol = sub.std(ddof=0) * np.sqrt(252)
    return {
        "n_days": n,
        "cumulative_return": float(cum.iloc[-1] - 1),
        "annualized_return": float((cum.iloc[-1] ** (252 / n)) - 1),
        "annualized_vol": float(vol),
        "sharpe": float(sub.mean() / sub.std(ddof=0) * np.sqrt(252)) if sub.std(ddof=0) > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "worst_day": float(sub.min()),
        "worst_week": float(weekly.min()),
    }


def regime_report(
    returns: pd.Series,
    regimes: Dict[str, Tuple[str, str]] = None,
) -> pd.DataFrame:
    """Generate per-regime performance table."""
    regimes = regimes or HISTORICAL_REGIMES
    rows = []
    for name, (start, end) in regimes.items():
        metrics = regime_performance(returns, start, end)
        metrics["regime"] = name
        metrics["start"] = start
        metrics["end"] = end
        rows.append(metrics)
    df = pd.DataFrame(rows)
    cols = ["regime", "start", "end", "n_days", "cumulative_return",
            "annualized_return", "annualized_vol", "sharpe", "max_drawdown",
            "worst_day", "worst_week"]
    return df[cols]


def worst_drawdowns(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Identify the top N worst drawdown episodes with start, trough, recovery."""
    cum = (1 + returns.fillna(0)).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1

    episodes = []
    in_dd = False
    start_idx = None
    for i, (date, dd) in enumerate(drawdown.items()):
        if not in_dd and dd < 0:
            in_dd = True
            start_idx = i
        elif in_dd and dd >= 0:
            # Drawdown ended
            in_dd = False
            trough_idx = start_idx + drawdown.iloc[start_idx:i].values.argmin()
            episodes.append({
                "start": drawdown.index[start_idx],
                "trough": drawdown.index[trough_idx],
                "recovery": drawdown.index[i],
                "depth": float(drawdown.iloc[trough_idx]),
                "duration_days": (drawdown.index[i] - drawdown.index[start_idx]).days,
            })
    if in_dd and start_idx is not None:
        trough_idx = start_idx + drawdown.iloc[start_idx:].values.argmin()
        episodes.append({
            "start": drawdown.index[start_idx],
            "trough": drawdown.index[trough_idx],
            "recovery": pd.NaT,
            "depth": float(drawdown.iloc[trough_idx]),
            "duration_days": (drawdown.index[-1] - drawdown.index[start_idx]).days,
        })

    df = pd.DataFrame(episodes)
    if df.empty:
        return df
    return df.nsmallest(top_n, "depth").reset_index(drop=True)
