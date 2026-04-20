import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = "AAPL"
data = yf.download(ticker, start="2026-04-01", end="2026-04-08")


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


def get_open_close(data, tickers, dates=None):
    results = {}
    for ticker in tickers:
        df = data[ticker][["Open", "Close"]]

        if dates is not None:
            df = df.loc[df.index.intersection(pd.to_datetime(dates))]

        results[ticker] = df
    return results

# given a ticker and an array of dates, it will return the price movement and alpha (baseline is ticker ^NDX)
# on that day
def _fetch_earnings(ticker):
    """Try multiple yfinance paths; return a DataFrame or None. Loud about failures."""
    t = yf.Ticker(ticker)

    # Path 1: get_earnings_dates
    try:
        e = t.get_earnings_dates(limit=40)
        if e is not None and not e.empty:
            print(f"[earnings] got {len(e)} rows from get_earnings_dates()")
            return e
        print("[earnings] get_earnings_dates() returned empty/None")
    except Exception as ex:
        print(f"[earnings] get_earnings_dates() raised: {ex}")

    # Path 2: earnings_dates attribute
    try:
        e = t.earnings_dates
        if e is not None and not e.empty:
            print(f"[earnings] got {len(e)} rows from .earnings_dates")
            return e
        print("[earnings] .earnings_dates returned empty/None")
    except Exception as ex:
        print(f"[earnings] .earnings_dates raised: {ex}")

    return None

def compute_return(data, ticker, date, offset_days):
    """
    Compute return for a given date and offset.
    offset_days = 0: same day return (close - open)/open
    offset_days > 0: return from date's close to future date's close
    """
    try:
        # Get data for specific ticker
        ticker_data = data[ticker]
        
        # Ensure datetime index
        if not isinstance(ticker_data.index, pd.DatetimeIndex):
            ticker_data.index = pd.to_datetime(ticker_data.index)
        
        if offset_days == 0:
            # Same day: (close - open) / open
            day_data = ticker_data.loc[ticker_data.index == date]
            if len(day_data) == 0:
                # Try to find the closest date
                closest_idx = ticker_data.index.get_indexer([date], method='nearest')[0]
                if closest_idx >= 0:
                    day_data = ticker_data.iloc[[closest_idx]]
                else:
                    return None
            
            if len(day_data) == 0:
                return None
                
            open_price = day_data['Open'].iloc[0]
            close_price = day_data['Close'].iloc[0]
            
            if open_price == 0:
                return None
                
            return (close_price - open_price) / open_price
        else:
            # Future date: get close on target date
            target_date = date + pd.Timedelta(days=offset_days)
            
            # Find the closest trading day on or after target_date
            future_dates = ticker_data.index[ticker_data.index >= target_date]
            if len(future_dates) == 0:
                return None
            
            actual_date = future_dates[0]
            
            # Get start and end data
            start_data = ticker_data.loc[ticker_data.index == date]
            end_data = ticker_data.loc[ticker_data.index == actual_date]
            
            if len(start_data) == 0:
                # Find closest start date
                start_idx = ticker_data.index.get_indexer([date], method='nearest')[0]
                if start_idx >= 0:
                    start_data = ticker_data.iloc[[start_idx]]
                else:
                    return None
            
            if len(end_data) == 0:
                end_data = ticker_data.loc[ticker_data.index == actual_date]
            
            if len(start_data) == 0 or len(end_data) == 0:
                return None
                
            start_close = start_data['Close'].iloc[0]
            end_close = end_data['Close'].iloc[0]
            
            if start_close == 0:
                return None
                
            return (end_close - start_close) / start_close
            
    except Exception as e:
        print(f"Error computing return for {ticker} on {date} with offset {offset_days}: {e}")
        return None


def get_earnings_data_for_dates(ticker, earnings_dates, limit=80):
    t = yf.Ticker(ticker)
    try:
        earnings = t.get_earnings_dates(limit=limit)
    except Exception as e:
        print(f"Could not fetch earnings dates for {ticker}: {e}")
        return pd.DataFrame()

    if earnings is None or earnings.empty:
        return pd.DataFrame()

    earnings = earnings.copy()
    earnings.index = pd.to_datetime(earnings.index).tz_localize(None).normalize()
    target_dates = pd.to_datetime(earnings_dates).normalize()
    return earnings.loc[earnings.index.intersection(target_dates)]


def earnings_impact(ticker, earnings_dates=None):
    if not earnings_dates:
        raise ValueError("earnings_dates cannot be empty")

    start_date = earnings_dates[0]
    
    # Extend end date to cover all horizons (5 trading days after last earnings date)
    last_date = pd.to_datetime(earnings_dates[-1])
    end_date = (last_date + pd.Timedelta(days=10)).strftime("%Y-%m-%d")  # Extra buffer for trading days

    # Download stock + index data (pass as list)
    tickers = [ticker, "^NDX"]
    data = download_data(tickers, start_date, end_date)
    
    print("Downloaded data info:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Index levels: {data.index.names}")
    print("\nFirst few rows:")
    print(data.head())
    
    # Separate data for stock and index
    stock_data = data[ticker]
    ndx_data = data["^NDX"]
        
    print(f"\nStock data ({ticker}) shape: {stock_data.shape}")
    print(stock_data.head())
    print(f"\nNDX data shape: {ndx_data.shape}")
    print(ndx_data.head())
    
    # Ensure datetime index
    stock_data.index = pd.to_datetime(stock_data.index)
    ndx_data.index = pd.to_datetime(ndx_data.index)

    results = []
    earnings_info = get_earnings_data_for_dates(ticker, earnings_dates)

    for date in earnings_dates:
        date = pd.to_datetime(date)

        # Check if date is within data range
        if date < stock_data.index.min() or date > stock_data.index.max():
            print(f"Skipping {date.strftime('%Y-%m-%d')}: outside stock data range ({stock_data.index.min().strftime('%Y-%m-%d')} to {stock_data.index.max().strftime('%Y-%m-%d')})")
            continue
            
        if date < ndx_data.index.min() or date > ndx_data.index.max():
            print(f"Skipping {date.strftime('%Y-%m-%d')}: outside NDX data range ({ndx_data.index.min().strftime('%Y-%m-%d')} to {ndx_data.index.max().strftime('%Y-%m-%d')})")
            continue

        row = {"date": date.strftime('%Y-%m-%d')}

        eps_est = None
        eps_rep = None
        eps_surprise_pct = None
        if not earnings_info.empty and date.normalize() in earnings_info.index:
            earnings_row = earnings_info.loc[date.normalize()]
            if isinstance(earnings_row, pd.DataFrame):
                earnings_row = earnings_row.iloc[0]

            eps_est = earnings_row.get("EPS Estimate")
            eps_rep = earnings_row.get("Reported EPS")

            if pd.notnull(eps_est) and pd.notnull(eps_rep) and eps_est != 0:
                eps_surprise_pct = (eps_rep - eps_est) / abs(eps_est)

        row["eps_estimate"] = eps_est
        row["eps_reported"] = eps_rep
        row["eps_surprise_pct"] = eps_surprise_pct

        # Time horizons (in calendar days, will find nearest trading day)
        horizons = {
            "0d": 0,    # Same day reaction (close vs open)
            "1d": 1,    # Next day close
            "3d": 3,    # 3 days later close
            "5d": 5     # 1 week later close
        }

        for label, offset in horizons.items():
            # Pass the full data and ticker to compute_return
            stock_ret = compute_return(data, ticker, date, offset)
            ndx_ret = compute_return(data, "^NDX", date, offset)

            row[f"stock_{label}"] = stock_ret
            row[f"ndx_{label}"] = ndx_ret

            if stock_ret is not None and ndx_ret is not None:
                row[f"alpha_{label}"] = stock_ret - ndx_ret
            else:
                row[f"alpha_{label}"] = None

        results.append(row)

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("\nNo results were computed. Check if earnings_dates are within the data range.")
        return df

    # Display raw numbers before formatting
    print("\n" + "="*80)
    print(f"Earnings Impact Analysis for {ticker}")
    print("="*80)
    
    # Create a formatted version for display
    display_df = df.copy()
    for col in display_df.columns:
        if col != 'date' and display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    
    print(display_df.to_string(index=False))
    print("="*80)
    
    # Also show raw numbers
    print("\nRaw returns (not percentages):")
    print(df.to_string(index=False))

    plot_df = df.dropna(subset=["alpha_1d", "eps_surprise_pct"]).copy()
    if not plot_df.empty:
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        ax1.plot(
            plot_df["date"],
            plot_df["alpha_1d"],
            color="tab:blue",
            marker="o",
            linewidth=2,
            label="1D Alpha",
        )
        ax2.plot(
            plot_df["date"],
            plot_df["eps_surprise_pct"],
            color="tab:orange",
            marker="s",
            linewidth=2,
            label="EPS Surprise %",
        )

        ax1.set_title(f"{ticker} Earnings: 1D Alpha vs EPS Surprise %")
        ax1.set_xlabel("Earnings Date")
        ax1.set_ylabel("1D Alpha", color="tab:blue")
        ax2.set_ylabel("EPS Surprise %", color="tab:orange")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping alpha vs EPS surprise plot: insufficient overlapping data.")

    return df



def plot_earnings_vs_baseline(ticker, start_date, end_date, baseline="^NDX", normalize=True, lag=0):
    """
    Plot ticker price vs baseline index over [start_date, end_date], and overlay
    EPS Estimate + Reported EPS as points on each earnings date that falls in the window.

    Parameters
    ----------
    ticker : str           e.g. "NVDA"
    start_date, end_date : str 'YYYY-MM-DD'
    baseline : str         index ticker, default '^NDX'
    normalize : bool       if True, rebase both series to 100 at start_date
    """
    # --- 1. Price data ---
    prices = download_data([ticker, baseline], start_date, end_date)
    stock = prices[ticker]["Close"].dropna()
    bench = prices[baseline]["Close"].dropna()

        # --- Apply lag ---
    if lag != 0:
        bench = bench.shift(lag)

    # Align after shift to avoid NaNs messing things up
    combined = pd.concat([stock, bench], axis=1).dropna()
    stock = combined.iloc[:, 0]
    bench = combined.iloc[:, 1]


    if normalize:
        stock_plot = stock / stock.iloc[0] * 100
        bench_plot = bench / bench.iloc[0] * 100
        ylabel = "Price (rebased to 100)"
    else:
        stock_plot = stock
        bench_plot = bench
        ylabel = "Close price"

    # --- 2. Earnings data ---
    earnings = get_earnings_data_for_dates(
        ticker,
        pd.date_range(start=start_date, end=end_date, freq="D"),
        limit=80
    )
    earnings_in_window = earnings if not earnings.empty else None

    # --- 3. Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_plot.index, stock_plot.values, label=ticker, linewidth=2)
    ax.plot(bench_plot.index, bench_plot.values, label=baseline, linewidth=2, alpha=0.8)

    # Overlay earnings markers on the stock line
    if earnings_in_window is not None and not earnings_in_window.empty:
        for earn_date, row in earnings_in_window.iterrows():
            # Find nearest trading day in our stock series
            idx = stock_plot.index.get_indexer([earn_date], method="nearest")[0]
            if idx < 0:
                continue
            plot_date = stock_plot.index[idx]
            y = stock_plot.iloc[idx]

            eps_est = row.get("EPS Estimate")
            eps_rep = row.get("Reported EPS")

            # Vertical line for the earnings date
            ax.axvline(plot_date, color="gray", linestyle="--", alpha=0.4)

            # Marker on the line itself
            ax.scatter(plot_date, y, color="red", zorder=5, s=60)

            # Annotation with EPS est / reported
            label_parts = [f"Earnings {plot_date.strftime('%Y-%m-%d')}"]
            if pd.notnull(eps_est):
                label_parts.append(f"Est: {eps_est:.2f}")
            if pd.notnull(eps_rep):
                label_parts.append(f"Rep: {eps_rep:.2f}")
            ax.annotate(
                "\n".join(label_parts),
                xy=(plot_date, y),
                xytext=(8, 12),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
            )

    ax.set_title(f"{ticker} vs {baseline} with earnings overlay")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    return earnings_in_window

def plot_alpha_vs_eps_surprise(
    ticker,
    start_date,
    end_date,
    baseline="^NDX",
    earnings_override=None,
):
    """
    For each earnings date in [start_date, end_date]:
      - Compute 1-day alpha = stock 1d return - baseline 1d return
      - Compute EPS surprise % = (Reported - Estimate) / |Estimate| * 100
    Scatterplot: x = EPS surprise %, y = 1d alpha %, with OLS fit and R^2.
    """
    import numpy as np

    # --- 1. Earnings data ---
    earnings = earnings_override if earnings_override is not None else _fetch_earnings(ticker)
    if earnings is None or earnings.empty:
        print("[alpha_vs_surprise] no earnings data; nothing to plot")
        return pd.DataFrame()

    earnings = earnings.copy()
    idx = pd.to_datetime(earnings.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    earnings.index = idx.normalize()
    earnings = earnings.sort_index()

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    earnings = earnings.loc[(earnings.index >= start) & (earnings.index <= end)]

    needed = ["EPS Estimate", "Reported EPS"]
    earnings = earnings.dropna(subset=[c for c in needed if c in earnings.columns])
    if earnings.empty:
        print("[alpha_vs_surprise] no earnings rows with both Estimate and Reported EPS")
        return pd.DataFrame()

    # --- 2. Price data ---
    price_start = (earnings.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    price_end = (earnings.index.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    data = download_data([ticker, baseline], price_start, price_end)

    # --- 3. Build comparison rows ---
    rows = []
    for earn_date, erow in earnings.iterrows():
        est = erow["EPS Estimate"]
        rep = erow["Reported EPS"]

        if est == 0 or pd.isna(est):
            eps_surprise_pct = None
        else:
            eps_surprise_pct = (rep - est) / abs(est) * 100

        stock_1d = compute_return(data, ticker, earn_date, 1)
        bench_1d = compute_return(data, baseline, earn_date, 1)
        alpha_1d = (stock_1d - bench_1d) * 100 if (stock_1d is not None and bench_1d is not None) else None
        
        rows.append({
            "date": earn_date,
            "EPS Estimate": est,
            "Reported EPS": rep,
            "EPS Surprise %": eps_surprise_pct,
            "Alpha 1d %": alpha_1d,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print("\n[alpha_vs_surprise] computed:")
    print(df.to_string(index=False))

    plot_df = df.dropna(subset=["EPS Surprise %", "Alpha 1d %"])
    if len(plot_df) < 2:
        print("[alpha_vs_surprise] need at least 2 points to fit a line")
        return df

    # --- 4. Scatterplot with OLS fit + R^2 ---
    x = plot_df["EPS Surprise %"].to_numpy(dtype=float)
    y = plot_df["Alpha 1d %"].to_numpy(dtype=float)

    # OLS: y = m*x + b
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    # R^2 = 1 - SS_res / SS_tot
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Pearson r (signed) is also useful context
    r = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(x, y, s=70, alpha=0.75, edgecolor="black", linewidth=0.8, color="#1f77b4",
               label=f"Earnings events (n={len(plot_df)})")

    # Regression line across the x-range
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#d62728", linewidth=2,
            label=f"OLS fit: y = {slope:.3f}x + {intercept:.3f}")

    # Reference lines at zero
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)

    # Stats box
    stats_txt = f"$R^2$ = {r_squared:.3f}\nPearson r = {r:.3f}\nn = {len(plot_df)}"
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9))

    ax.set_xlabel("EPS Surprise (%)")
    ax.set_ylabel("1-day Alpha (%)")
    ax.set_title(f"{ticker}: 1d alpha vs EPS surprise")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return df

def plot_alpha_horizons_vs_surprise(
    ticker,
    start_date,
    end_date,
    baseline="^NDX",
    horizons=(1, 3, 5),
    earnings_override=None,
    pool_tickers=None,
    plot_dps=False,         # NEW: if True, x-axis becomes DPS instead of EPS surprise %
):
    """
    Scatter alpha at multiple horizons against either EPS surprise % (default)
    or dividend-per-share (if plot_dps=True). Supports single-ticker and pooled modes.
    """
    import numpy as np

    pooled_mode = bool(pool_tickers)
    tickers_to_process = list(pool_tickers) if pooled_mode else [ticker]
    max_h = max(horizons)

    # --- Collect rows across tickers ---
    all_rows = []

    for tkr in tickers_to_process:
        # Earnings
        if pooled_mode:
            earnings = _fetch_earnings(tkr)
        else:
            earnings = earnings_override if earnings_override is not None else _fetch_earnings(tkr)

        if earnings is None or earnings.empty:
            print(f"[{tkr}] no earnings data, skipping")
            continue

        earnings = earnings.copy()
        idx = pd.to_datetime(earnings.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        earnings.index = idx.normalize()
        earnings = earnings.sort_index()

        start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
        earnings = earnings.loc[(earnings.index >= start) & (earnings.index <= end)]
        earnings = earnings.dropna(subset=[c for c in ["EPS Estimate", "Reported EPS"]
                                           if c in earnings.columns])
        if earnings.empty:
            print(f"[{tkr}] no usable earnings rows in window, skipping")
            continue

        # Dividend history for this ticker (only if needed)
        div_series = None
        if plot_dps:
            try:
                div_series = yf.Ticker(tkr).dividends
                if div_series is not None and not div_series.empty:
                    div_idx = pd.to_datetime(div_series.index)
                    if div_idx.tz is not None:
                        div_idx = div_idx.tz_localize(None)
                    div_series.index = div_idx.normalize()
                    div_series = div_series.sort_index()
                else:
                    div_series = None
            except Exception as ex:
                print(f"[{tkr}] dividend fetch failed: {ex}")
                div_series = None

        # Price data
        price_start = (earnings.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        price_end = (earnings.index.max() + pd.Timedelta(days=max_h + 10)).strftime("%Y-%m-%d")
        try:
            data = download_data([tkr, baseline], price_start, price_end)
        except Exception as ex:
            print(f"[{tkr}] price download failed: {ex}, skipping")
            continue

        for earn_date, erow in earnings.iterrows():
            est, rep = erow["EPS Estimate"], erow["Reported EPS"]
            surprise = (rep - est) / est * 100 if (est and not pd.isna(est)) else None

            print("ticker: " + str(earn_date) + " estimate: " + str(est) + " reported: " + str(rep))
            print(" surprise: " + str(surprise))
            # Most recent dividend paid strictly before this earnings date
            dps = None
            if div_series is not None:
                prior = div_series.loc[div_series.index < earn_date]
                if len(prior) > 0:
                    dps = float(prior.iloc[-1])

            row = {"ticker": tkr, "date": earn_date,
                   "EPS Surprise %": surprise, "DPS": dps}
            for h in horizons:
                s = compute_return(data, tkr, earn_date, h)
                b = compute_return(data, baseline, earn_date, h)
                row[f"Alpha {h}d %"] = (s - b) * 100 if (s is not None and b is not None) else None
            all_rows.append(row)

        if pooled_mode:
            print(f"[{tkr}] collected {len(earnings)} events")

    if not all_rows:
        print("[alpha_horizons] no events collected")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    print(f"\n[alpha_horizons] total events: {len(df)} across {df['ticker'].nunique()} ticker(s)")

    # --- Choose x-axis column based on flag ---
    if plot_dps:
        xcol = "DPS"
        xlabel = "Dividend per share ($)"
        df_fit = df.dropna(subset=[xcol])
        # Drop zero-DPS events — no dividend means the x is uninformative here
        df_fit = df_fit[df_fit[xcol] > 0]
        n_dropped = len(df) - len(df_fit)
        if n_dropped:
            print(f"[alpha_horizons] dropped {n_dropped} events with no prior dividend")
        if df_fit.empty:
            print("[alpha_horizons] no events with dividends; nothing to plot")
            return df
    else:
        xcol = "EPS Surprise %"
        xlabel = "EPS Surprise (%)"
        df_fit = df.dropna(subset=[xcol])

        # Winsorize surprise outliers in pooled mode (same as before)
        if pooled_mode and len(df_fit) >= 20:
            lo, hi = df_fit[xcol].quantile([0.01, 0.99])
            n_before = len(df_fit)
            df_fit = df_fit[(df_fit[xcol] >= lo) & (df_fit[xcol] <= hi)].copy()
            print(f"[alpha_horizons] winsorized to [{lo:.2f}, {hi:.2f}]: "
                  f"{n_before} → {len(df_fit)} events")

    # --- Plot ---
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.5 * len(horizons), 5.5),
                             sharex=True, sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    x_all = df_fit[xcol].dropna().to_numpy(dtype=float)
    if len(x_all) == 0:
        print("[alpha_horizons] no x values to plot")
        return df
    x_pad = (x_all.max() - x_all.min()) * 0.05 or 1.0
    x_lo, x_hi = x_all.min() - x_pad, x_all.max() + x_pad

    colors = ["#1f77b4", "#2ca02c", "#9467bd"]

    for ax, h, color in zip(axes, horizons, colors):
        ycol = f"Alpha {h}d %"
        sub = df_fit.dropna(subset=[xcol, ycol])
        x = sub[xcol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)

        dot_size = 25 if pooled_mode else 70
        dot_alpha = 0.45 if pooled_mode else 0.75
        ax.scatter(x, y, s=dot_size, alpha=dot_alpha, color=color,
                   edgecolor="black", linewidth=0.4, zorder=3)

        if len(sub) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x_lo, x_hi, 100)
            ax.plot(x_line, slope * x_line + intercept, color="#d62728", linewidth=2, zorder=4)

            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            r = np.corrcoef(x, y)[0, 1]

            stats = (f"$R^2$ = {r2:.3f}\n"
                     f"r = {r:.3f}\n"
                     f"slope = {slope:.3f}\n"
                     f"n = {len(sub)}")
        else:
            stats = f"n = {len(sub)} (need ≥2)"

        ax.text(0.03, 0.97, stats, transform=ax.transAxes,
                fontsize=10, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", ec="gray", alpha=0.9))

        ax.axhline(0, color="gray", linewidth=0.7, alpha=0.5)
        if not plot_dps:
            ax.axvline(0, color="gray", linewidth=0.7, alpha=0.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_title(f"{h}-day alpha")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Alpha (%)")

    x_desc = "DPS" if plot_dps else "EPS surprise"
    if pooled_mode:
        title = f"Pooled across {df['ticker'].nunique()} tickers: alpha vs {x_desc}"
    else:
        title = f"{ticker}: alpha at multiple horizons vs {x_desc}"
    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    small_tech_sample = [
        "AEHR", "YEXT", "CXM",
    ]

    
    tech_sample = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "COST", "ADBE", "NFLX", "AMD", "CSCO", "QCOM",
    "INTC", "TXN", "AMAT", "MU", "INTU", "BKNG",
    ]
    #plot_earnings_vs_baseline("NVDA", "2025-01-01", "2026-04-01", lag=10)
    
    #plot_alpha_horizons_vs_surprise("Tech Ssample", "2020-01-01", "2026-04-01", pool_tickers=small_tech_sample)

    # plot_alpha_horizons_vs_surprise("YEXT", "2023-01-01", "2026-04-01")
    # plot_alpha_horizons_vs_surprise("GS", "2023-01-01", "2026-04-01")
    plot_alpha_vs_eps_surprise("NVDA", "2023-01-01", "2026-04-01")
    #plot_alpha_vs_eps_surprise("AAPL", "2023-01-01", "2026-04-01")
    plot_alpha_vs_eps_surprise("MSFT", "2023-01-01", "2026-04-01")
    #plot_alpha_vs_eps_surprise("GOOG", "2023-01-01", "2026-04-01")
    #plot_alpha_vs_eps_surprise("META", "2023-01-01", "2026-04-01")
    #result = earnings_impact("GOOG", ["2026-02-25"])



    # --- testing get_open_close
    # tickers = ["^NDX", "MSFT", "SPY"]
    # start_date = "2026-04-01"
    # end_date = "2026-04-08"
    # data = download_data(tickers, start_date, end_date)

    # # Example 1: get all dates
    # all_prices = get_open_close(data, tickers)
    # print(all_prices)

    # # Example 2: specific dates only
    # dates = ["2026-03-01", "2026-04-01"]
    # filtered_prices = get_open_close(data, tickers, dates)

    # print(filtered_prices["^NDX"])