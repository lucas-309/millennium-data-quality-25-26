import yfinance as yf
import pandas as pd
from datetime import timedelta


def download_data(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False,
        threads=True # this enables parallel downloading
    )
    return data

def earnings_impact(ticker: str, earnings_dates: list[str] | None = None) -> pd.DataFrame:
    """
    For each earnings date, compute the price return of `ticker` and NDX
    at various horizons, then report the alpha (ticker return minus NDX return).
 
    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g. "AAPL").
    earnings_dates : list[str] | None
        List of earnings date strings in "YYYY-MM-DD" format.
        If None, attempts to fetch them from yfinance.
 
    Returns
    -------
    pd.DataFrame
        One row per earnings date with columns:
            earnings_date
            ticker_1d, ticker_1d_after, ticker_3d_after, ticker_7d_after
            ndx_1d,    ndx_1d_after,    ndx_3d_after,    ndx_7d_after
            alpha_1d,  alpha_1d_after,  alpha_3d_after,  alpha_7d_after
    """
 
    NDX_TICKER = "^NDX"
 
    # ── 1. Resolve earnings dates ──────────────────────────────────────────
    if earnings_dates is None:
        stock = yf.Ticker(ticker)
        cal = stock.get_calendar()
        if cal is None or cal.empty:
            raise ValueError(
                f"No earnings calendar found for {ticker}. "
                "Please supply earnings_dates manually."
            )
        # yfinance returns a DataFrame with 'Earnings Date' as a column
        earnings_dates = sorted(
            pd.to_datetime(cal["Earnings Date"]).dt.normalize().unique()
        )
    else:
        earnings_dates = sorted(pd.to_datetime(earnings_dates).normalize().unique())
 
    if not len(earnings_dates):
        raise ValueError("earnings_dates is empty.")
 
    # ── 2. Download price data for the whole span in one shot ─────────────
    # We need 1 trading day before the first date and ~10 calendar days
    # after the last date (to guarantee 7 trading days exist).
    global_start = min(earnings_dates) - timedelta(days=10)
    global_end   = max(earnings_dates) + timedelta(days=20)
 
    raw = download_data([ticker, NDX_TICKER], global_start, global_end)
 
    # ── 3. Extract adjusted close series ─────────────────────────────────
 # ── 3. Extract adjusted close series ─────────────────────────────────
    def get_adj_close(data: pd.DataFrame, sym: str) -> pd.Series:
        """Pull the 'Adj Close' column for a given symbol.
 
        yfinance column structure varies by version and number of tickers:
          - MultiIndex (Price, Ticker): data["Adj Close"][sym]  — newer yfinance
          - MultiIndex (Ticker, Price): data[sym]["Adj Close"]  — older yfinance
          - Flat columns              : data["Adj Close"]       — single ticker
        """
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0).unique().tolist()
            lvl1 = data.columns.get_level_values(1).unique().tolist()
 
            # newer yfinance: (Price, Ticker)
            if "Adj Close" in lvl0 and sym in lvl1:
                return data["Adj Close"][sym].dropna()
 
            # older yfinance: (Ticker, Price)
            if sym in lvl0 and "Adj Close" in lvl1:
                return data[sym]["Adj Close"].dropna()
 
            raise KeyError(
                f"Cannot locate 'Adj Close' for '{sym}'. "
                f"Level 0 labels: {lvl0[:10]}. Level 1 labels: {lvl1[:10]}."
            )
 
        # Flat columns — single ticker download
        if "Adj Close" in data.columns:
            return data["Adj Close"].dropna()
 
        raise KeyError(
            f"'Adj Close' not found in columns: {data.columns.tolist()}"
        )
 
    prices_ticker = get_adj_close(raw, ticker)
    prices_ndx    = get_adj_close(raw, NDX_TICKER)
 
 
    trading_days_ticker = prices_ticker.index.sort_values()
    trading_days_ndx    = prices_ndx.index.sort_values()
 
    # ── 4. Helper: find the nearest available trading day on or after `date` ─
    def nearest_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
        candidates = trading_days[trading_days >= date]
        return candidates[0] if len(candidates) else None
 
    def nth_trading_day_after(date: pd.Timestamp, n: int, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
        idx = trading_days.searchsorted(date)
        target_idx = idx + n
        return trading_days[target_idx] if target_idx < len(trading_days) else None
 
    # ── 5. Compute returns for each earnings date ─────────────────────────
    def pct_return(prices: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> float | None:
        """Return (p[t1] / p[t0]) - 1, or None if either date is missing."""
        if t0 is None or t1 is None:
            return None
        if t0 not in prices.index or t1 not in prices.index:
            return None
        return (prices[t1] / prices[t0]) - 1
 
    records = []
    for ed in earnings_dates:
        ed = pd.Timestamp(ed)
 
        # "Day before" is the reference (t0).  We look for the last trading
        # day strictly before the earnings date.
        pre_days_ticker = trading_days_ticker[trading_days_ticker < ed]
        pre_days_ndx    = trading_days_ndx[trading_days_ndx < ed]
 
        if not len(pre_days_ticker) or not len(pre_days_ndx):
            continue  # not enough history — skip
 
        t0_ticker = pre_days_ticker[-1]
        t0_ndx    = pre_days_ndx[-1]
 
        # The earnings date itself (or the next available trading day)
        t_earn_ticker = nearest_trading_day(ed, trading_days_ticker)
        t_earn_ndx    = nearest_trading_day(ed, trading_days_ndx)
 
        # Subsequent horizons counted from the earnings date
        t1_ticker  = nth_trading_day_after(t_earn_ticker, 1, trading_days_ticker) if t_earn_ticker else None
        t3_ticker  = nth_trading_day_after(t_earn_ticker, 3, trading_days_ticker) if t_earn_ticker else None
        t7_ticker  = nth_trading_day_after(t_earn_ticker, 7, trading_days_ticker) if t_earn_ticker else None
 
        t1_ndx     = nth_trading_day_after(t_earn_ndx, 1, trading_days_ndx) if t_earn_ndx else None
        t3_ndx     = nth_trading_day_after(t_earn_ndx, 3, trading_days_ndx) if t_earn_ndx else None
        t7_ndx     = nth_trading_day_after(t_earn_ndx, 7, trading_days_ndx) if t_earn_ndx else None
 
        # Returns
        r_ticker_0d = pct_return(prices_ticker, t0_ticker, t_earn_ticker)  # day-of
        r_ticker_1d = pct_return(prices_ticker, t0_ticker, t1_ticker)       # 1 day after
        r_ticker_3d = pct_return(prices_ticker, t0_ticker, t3_ticker)       # 3 days after
        r_ticker_7d = pct_return(prices_ticker, t0_ticker, t7_ticker)       # 7 days after
 
        r_ndx_0d = pct_return(prices_ndx, t0_ndx, t_earn_ndx)
        r_ndx_1d = pct_return(prices_ndx, t0_ndx, t1_ndx)
        r_ndx_3d = pct_return(prices_ndx, t0_ndx, t3_ndx)
        r_ndx_7d = pct_return(prices_ndx, t0_ndx, t7_ndx)
 
        def alpha(r_stk, r_ndx):
            if r_stk is None or r_ndx is None:
                return None
            return r_stk - r_ndx
 
        records.append({
            "earnings_date":   ed.date(),
            # Ticker returns
            "ticker_day_of":   r_ticker_0d,
            "ticker_1d_after": r_ticker_1d,
            "ticker_3d_after": r_ticker_3d,
            "ticker_7d_after": r_ticker_7d,
            # NDX returns
            "ndx_day_of":      r_ndx_0d,
            "ndx_1d_after":    r_ndx_1d,
            "ndx_3d_after":    r_ndx_3d,
            "ndx_7d_after":    r_ndx_7d,
            # Alpha (ticker − NDX)
            "alpha_day_of":    alpha(r_ticker_0d, r_ndx_0d),
            "alpha_1d_after":  alpha(r_ticker_1d, r_ndx_1d),
            "alpha_3d_after":  alpha(r_ticker_3d, r_ndx_3d),
            "alpha_7d_after":  alpha(r_ticker_7d, r_ndx_7d),
        })
 
    df = pd.DataFrame(records)
 
    # Format returns as percentages for readability
    return_cols = [c for c in df.columns if c != "earnings_date"]
    df[return_cols] = df[return_cols].apply(lambda col: col * 100)
 
    return df
 
 
# ── Quick demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = earnings_impact(
        "NVDA",
        earnings_dates=["2025-11-02", "2026-2-26"],
    )
    pd.set_option("display.float_format", "{:.2f}%".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(result.to_string(index=False))