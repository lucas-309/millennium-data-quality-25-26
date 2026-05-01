from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtester.research_data import ResearchDataset


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    # A one-column frame has no cross-section: the per-row std is 0 and the
    # regular zscore collapses every value to NaN, which the backtester then
    # treats as "no signal" and skips every rebalance. Collapse finite values
    # to 0 (a single sample sits at its own mean) so downstream ranking still
    # picks the sole surviving name.
    if frame.shape[1] <= 1:
        return frame * 0.0
    means = frame.mean(axis=1, skipna=True)
    stds = frame.std(axis=1, skipna=True).replace(0, np.nan)
    return frame.sub(means, axis=0).div(stds, axis=0)


def _sector_neutral_zscore(frame: pd.DataFrame, metadata: Optional[pd.DataFrame], group_column: str = "gsector") -> pd.DataFrame:
    if metadata is None or metadata.empty or group_column not in metadata.columns:
        return _cross_sectional_zscore(frame)

    groups = metadata.dropna(subset=[group_column]).groupby(group_column).groups
    neutralized = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    covered_tickers: set[str] = set()

    for _, tickers in groups.items():
        group_tickers = [ticker for ticker in tickers if ticker in frame.columns]
        if not group_tickers:
            continue
        neutralized[group_tickers] = _cross_sectional_zscore(frame[group_tickers])
        covered_tickers.update(group_tickers)

    remaining = [ticker for ticker in frame.columns if ticker not in covered_tickers]
    if remaining:
        neutralized[remaining] = _cross_sectional_zscore(frame[remaining])

    return neutralized


@dataclass
class StrategyOutput:
    name: str
    scores: pd.DataFrame
    motivation: str
    economic_rationale: str
    why_it_works: str
    why_it_fails: str
    uses_event_data: bool = False
    event_type: Optional[str] = None
    backtest_overrides: dict[str, Any] = field(default_factory=dict)
    combine_in_portfolio: bool = True


class SignalStrategy(ABC):
    name: str = ""
    motivation: str = ""
    economic_rationale: str = ""
    why_it_works: str = ""
    why_it_fails: str = ""
    uses_event_data: bool = False
    event_type: Optional[str] = None
    backtest_overrides: dict[str, Any] = {}
    combine_in_portfolio: bool = True

    def generate(self, dataset: ResearchDataset) -> StrategyOutput:
        raw_scores = self.generate_scores(dataset)
        scores = self.normalize_scores(raw_scores, dataset).replace([np.inf, -np.inf], np.nan)
        return StrategyOutput(
            name=self.name,
            scores=scores,
            motivation=self.motivation,
            economic_rationale=self.economic_rationale,
            why_it_works=self.why_it_works,
            why_it_fails=self.why_it_fails,
            uses_event_data=self.uses_event_data,
            event_type=self.event_type,
            backtest_overrides=dict(self.backtest_overrides),
            combine_in_portfolio=self.combine_in_portfolio,
        )

    def normalize_scores(self, raw_scores: pd.DataFrame, dataset: ResearchDataset) -> pd.DataFrame:
        return _cross_sectional_zscore(raw_scores)

    @abstractmethod
    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        raise NotImplementedError


class SmallCapTiltStrategy(SignalStrategy):
    name = "Small-Cap Tilt"
    motivation = "Own smaller-cap names and short larger-cap peers within the same large-cap universe."
    economic_rationale = "Smaller names often carry a persistent risk premium and receive less broad institutional attention."
    why_it_works = "Market-cap ranks move slowly, so the signal keeps turnover low while monetizing the size spread."
    why_it_fails = "Size tends to lag in risk-off environments or when mega-cap leadership dominates index returns."
    backtest_overrides = {
        "rebalance_frequency": "ME",
        "construction_method": "equal_weight",
        "long_quantile": 0.1,
        "short_quantile": 0.1,
    }

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        if dataset.market_caps is None or dataset.market_caps.empty:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)
        return -np.log(dataset.market_caps.replace(0, np.nan))


class ValueCompositeStrategy(SignalStrategy):
    name = "Value Composite"
    motivation = "Long the cheapest names by a Fama-French value composite (B/M, E/P, CF/P, D/P)."
    economic_rationale = "Cross-sectional value spreads compensate investors for distress / multiple-mean-reversion risk."
    why_it_works = "Composite of four price-yield ratios is more robust than any single ratio and matches the canonical Fama-French (1992, 2015) HML construction."
    why_it_fails = "Value cycles are long; the factor can underperform for a decade (e.g. 2010-2020 growth dominance)."
    backtest_overrides = {
        "rebalance_frequency": "ME",
        "construction_method": "equal_weight",
        "long_quantile": 0.1,
        "short_quantile": 0.1,
    }
    combine_in_portfolio = False

    # Each ratio is already on a "yield-to-price" scale (higher = cheaper),
    # so we can z-score and sum without any sign flipping. Order matters
    # only insofar as `_cross_sectional_zscore` is applied per-ratio first.
    RATIOS = ("bm", "ep", "cfp", "dy")

    def __init__(self) -> None:
        # Constructor takes no params: the composite is fully specified by
        # the four ratios above, and the long/short quantiles live in the
        # engine overrides. Keeping the signature empty also lets the live
        # code editor accept edits without juggling kwargs.
        pass

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        if not dataset.fundamentals:
            # Source has no financial-ratio panel attached (yfinance, custom).
            # Returning an all-NaN frame causes the backtester to skip every
            # rebalance instead of raising — the run still completes with a
            # flat curve and the user can switch sources.
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)

        z_frames: list[pd.DataFrame] = []
        for ratio in self.RATIOS:
            panel = dataset.fundamentals.get(ratio)
            if panel is None or panel.empty:
                continue
            aligned = panel.reindex(index=dataset.prices.index, columns=dataset.prices.columns)
            z_frames.append(_cross_sectional_zscore(aligned))
        if not z_frames:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)

        # NaN-aware sum: a (date, ticker) cell with no ratios at all stays
        # NaN (so the backtester drops it from the cross-section), but a cell
        # missing one of four ratios still gets scored from the others. Plain
        # DataFrame.add with fill_value=0 mishandles the all-NaN case by
        # turning it into a numerical zero, which then competes for the top
        # quantile alongside actually-cheap names.
        stacked = np.stack([f.to_numpy(dtype=float) for f in z_frames], axis=0)
        summed = np.nansum(stacked, axis=0)
        all_nan = np.isnan(stacked).all(axis=0)
        summed[all_nan] = np.nan
        return pd.DataFrame(summed, index=z_frames[0].index, columns=z_frames[0].columns)


class EarningsRevisionStrategy(SignalStrategy):
    name = "EPS Revision"
    motivation = "Rank stocks by improving versus deteriorating earnings expectations."
    economic_rationale = "Analysts and the market often underreact to gradual changes in the earnings outlook."
    why_it_works = "EPS revisions update less frequently than price, so the signal can capture post-announcement drift with manageable turnover."
    why_it_fails = "Revision signals break when estimates lag reality or when macro shocks overwhelm company-specific fundamentals."
    uses_event_data = True
    event_type = "anncdate"
    backtest_overrides = {
        "rebalance_frequency": "ME",
        "construction_method": "equal_weight",
        "long_quantile": 0.1,
        "short_quantile": 0.1,
    }

    def __init__(self, lookback_days: int = 63):
        self.lookback_days = lookback_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        if dataset.eps is None or dataset.eps.empty:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)
        baseline = dataset.eps.shift(self.lookback_days).replace(0, np.nan)
        return dataset.eps.div(baseline) - 1.0


class SectorNeutralDividendYieldStrategy(SignalStrategy):
    name = "Sector-Neutral Dividend Yield"
    motivation = "Own stronger dividend-yield names while neutralizing simple sector composition effects."
    economic_rationale = "Dividend policy can signal cash-flow durability, but yield comparisons are cleaner inside sectors than across them."
    why_it_works = "Sector neutralization removes the largest structural industry skews and keeps the signal focused on within-sector payout support."
    why_it_fails = "Dividend yield can still become a value trap when payouts lag a deteriorating earnings outlook."
    uses_event_data = True
    event_type = "divdpaydate"
    backtest_overrides = {
        "rebalance_frequency": "ME",
        "construction_method": "equal_weight",
        "long_quantile": 0.1,
        "short_quantile": 0.1,
    }

    def __init__(self, trailing_days: int = 252):
        self.trailing_days = trailing_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        if dataset.dividends is None or dataset.dividends.empty:
            return pd.DataFrame(index=dataset.prices.index, columns=dataset.prices.columns, dtype=float)

        trailing_dividends = dataset.dividends.rolling(
            self.trailing_days,
            min_periods=max(self.trailing_days // 4, 1),
        ).sum()
        return trailing_dividends.div(dataset.prices.replace(0, np.nan))

    def normalize_scores(self, raw_scores: pd.DataFrame, dataset: ResearchDataset) -> pd.DataFrame:
        return _sector_neutral_zscore(raw_scores, dataset.metadata)


class CrossSectionalMomentumStrategy(SignalStrategy):
    name = "Cross-Sectional Momentum"
    motivation = "Rank stocks by medium-term return, long the top names."
    economic_rationale = "Medium-term winners continue to outperform due to slow information diffusion and underreaction to fundamentals."
    why_it_works = "12-1 momentum is the most persistent cross-sectional anomaly in equities across decades and markets."
    why_it_fails = "Crashes during sharp reversals and regime changes (e.g. March 2009)."

    def __init__(self, lookback_days: int = 126, skip_days: int = 21):
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        prices = dataset.prices
        return prices.shift(self.skip_days).div(prices.shift(self.lookback_days)) - 1.0


class ShortTermReversalStrategy(SignalStrategy):
    name = "Short-Term Reversal"
    motivation = "Short-term losers tend to bounce back over the following days."
    economic_rationale = "Overreaction to news and liquidity shocks creates transient price dislocations that mean-revert."
    why_it_works = "Daily/weekly reversal is one of the most persistent short-horizon anomalies in equities."
    why_it_fails = "Strongly trending markets suppress reversal; execution cost eats the signal at high turnover."

    def __init__(self, lookback_days: int = 5):
        self.lookback_days = lookback_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        # Negate recent return — biggest recent losers get the highest score
        # (i.e. long candidates), biggest recent winners get the lowest.
        return -(dataset.prices.div(dataset.prices.shift(self.lookback_days)) - 1.0)


class ResidualMomentumStrategy(SignalStrategy):
    name = "Residual Momentum"
    motivation = "Buy names with strong recent returns *after* stripping out exposure to the market."
    economic_rationale = (
        "Blitz, Huij & Martens (2011): vanilla 12-1 momentum is partly a "
        "compensation for high-beta exposure. Regressing each stock's daily "
        "returns on the market and ranking by the recent *residual* mean "
        "isolates the slow-information-diffusion channel and decorrelates "
        "the signal from broad-market beta."
    )
    why_it_works = (
        "Higher Sharpe than raw momentum across decades; smaller drawdowns "
        "during momentum crashes (e.g. March 2009) because market-beta "
        "exposure has been residualized away."
    )
    why_it_fails = (
        "Estimation error in rolling beta on names with thin history; "
        "structural breaks in factor structure (e.g. the COVID drawdown "
        "regime) can leave residuals contaminated until the window rolls."
    )

    def __init__(
        self,
        beta_window: int = 252,
        lookback_days: int = 126,
        skip_days: int = 21,
    ):
        self.beta_window = beta_window
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        returns = dataset.returns
        market = dataset.benchmark_returns
        if returns is None or returns.empty or market is None or market.empty:
            return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        # Vectorized rolling beta: cov_i(t) = rolling_cov(r_i, r_m); var_m(t) = rolling_var(r_m)
        # Pandas: df.rolling(W).cov(series) returns a DataFrame where each
        # column is the rolling covariance of that column with the series.
        rolling_cov = returns.rolling(self.beta_window, min_periods=self.beta_window // 2).cov(market)
        rolling_var = market.rolling(self.beta_window, min_periods=self.beta_window // 2).var()
        beta = rolling_cov.div(rolling_var.replace(0, np.nan), axis=0)

        # Residuals: r_i(t) − β_i(t) · r_m(t). Drop the drifting alpha
        # intercept — for daily data over our window it's near-zero and
        # introduces only noise.
        expected = beta.mul(market, axis=0)
        residuals = returns.subtract(expected)

        # Standardize residuals by their own rolling std so high-vol names
        # don't dominate the cross-section purely on noise (BHM normalize
        # at the *signal* step, not the score step — same idea).
        res_std = residuals.rolling(
            self.beta_window, min_periods=self.beta_window // 2,
        ).std().replace(0, np.nan)
        standardized = residuals.div(res_std)

        # 12-1 style signal — average standardized residual over the
        # lookback window, skipping the most recent skip_days to avoid
        # short-term reversal contamination.
        score = (
            standardized.shift(self.skip_days)
            .rolling(self.lookback_days, min_periods=max(self.lookback_days // 2, 1))
            .mean()
        )
        return score


class LowVolatilityStrategy(SignalStrategy):
    name = "Low Volatility"
    motivation = "Hold the lowest-volatility names, exploiting the low-vol anomaly."
    economic_rationale = "Leverage constraints and lottery preference cause low-vol stocks to be chronically underpriced relative to their risk-adjusted returns."
    why_it_works = "Low-vol long-only portfolios have matched or beaten broad equity returns with significantly smaller drawdowns across many decades."
    why_it_fails = "Underperforms speculative rallies and rotation toward high-beta growth names."

    def __init__(self, window: int = 126):
        self.window = window

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        vol = dataset.returns.rolling(self.window, min_periods=self.window // 2).std()
        return -vol  # higher score = lower vol = more desirable


# Curated supplier → customers mapping for the Customer-Supplier Momentum
# strategy. Without a CapIQ Compustat-Segment file we ship a hand-built
# subset of well-known supplier/customer pairs that the yfinance S&P 500
# universe covers out of the box. The strategy averages each customer's
# recent return into a per-supplier signal — long the suppliers whose
# customers ran up, short those whose customers lagged.
CUSTOMER_SUPPLIER_RELATIONSHIPS: dict[str, list[str]] = {
    # Suppliers feeding NVIDIA's GPU / accelerator stack
    "TSM":  ["NVDA", "AAPL", "AMD", "QCOM"],
    "ASML": ["NVDA", "INTC", "AMD", "TSM"],
    "AMAT": ["NVDA", "TSM", "INTC"],
    "LRCX": ["NVDA", "TSM", "INTC"],
    "KLAC": ["NVDA", "TSM", "INTC"],
    "MU":   ["NVDA", "AAPL", "DELL", "HPQ"],
    "MRVL": ["NVDA", "META", "GOOGL"],
    # Apple supply chain
    "AVGO": ["AAPL", "GOOGL", "META"],
    "QCOM": ["AAPL", "META", "GOOGL"],
    "SWKS": ["AAPL"],
    "QRVO": ["AAPL"],
    "CRUS": ["AAPL"],
    # Cloud / hyperscaler accelerators
    "NVDA": ["MSFT", "META", "GOOGL", "AMZN", "ORCL"],
    "AMD":  ["MSFT", "META", "GOOGL", "AMZN", "ORCL"],
    # Logistics
    "UPS":  ["AMZN", "WMT", "TGT"],
    "FDX":  ["AMZN", "WMT"],
    # EV / battery / auto
    "ALB":  ["TSLA", "GM", "F"],
    "PCAR": ["AMZN", "UPS", "FDX"],
    # Industrials / aerospace
    "HON":  ["BA", "LMT", "RTX"],
    "TXT":  ["BA", "LMT"],
    "GE":   ["BA", "LMT"],
    "PH":   ["BA", "LMT", "CAT"],
    # Defense supply chain
    "LDOS": ["LMT", "RTX"],
}


class CustomerSupplierMomentumStrategy(SignalStrategy):
    name = "Customer-Supplier Momentum"
    motivation = "Suppliers whose major customers had strong recent stock returns tend to rally next; weak-customer suppliers tend to drift down."
    economic_rationale = "Cohen & Frazzini (2008): the equity market underreacts to news that originates at a customer firm and only slowly diffuses to its suppliers along the supply chain."
    why_it_works = "Customer-level information is public but expensive to track; a static supplier→customer map plus customer-side returns recovers the diffusion premium with low turnover."
    why_it_fails = "Requires accurate relationship data — stale or thin maps push noise into the signal. Macro shocks that hit customers and suppliers simultaneously kill the lead/lag."

    def __init__(self, customer_lookback_days: int = 21, min_customers: int = 1):
        self.customer_lookback_days = customer_lookback_days
        self.min_customers = min_customers

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        prices = dataset.prices
        # Customer recent return = past N-day total return on the customer's stock.
        customer_returns = prices.pct_change(self.customer_lookback_days, fill_method=None)
        scores = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns, dtype=float)
        for supplier, customers in CUSTOMER_SUPPLIER_RELATIONSHIPS.items():
            if supplier not in scores.columns:
                continue
            tracked = [c for c in customers if c in customer_returns.columns]
            if len(tracked) < self.min_customers:
                continue
            cust_panel = customer_returns[tracked]
            valid_count = cust_panel.notna().sum(axis=1)
            avg = cust_panel.mean(axis=1, skipna=True)
            scores[supplier] = avg.where(valid_count >= self.min_customers)
        return scores


class PEADStrategy(SignalStrategy):
    name = "Post-Earnings Announcement Drift"
    motivation = "Stocks with positive earnings surprises drift up for weeks; negative-surprise names drift down."
    economic_rationale = "Bernard & Thomas (1989): the market underreacts to the information content of an earnings announcement, releasing the signal gradually over the following 60 trading days."
    why_it_works = "Standardised Unexpected Earnings (SUE) is a clean cross-sectional ranking that turns over only on announcement dates, so transaction-cost drag is bounded."
    why_it_fails = "Crowded over time — top-decile drift has shrunk in liquid large caps; pre-announcement leakage and analyst-cluster noise eat the signal at higher frequencies."

    uses_event_data = True
    event_type = "anncdate"

    _events_cache: Optional[pd.DataFrame] = None

    def __init__(self, holding_days: int = 60):
        self.holding_days = holding_days

    @classmethod
    def _load_events(cls) -> pd.DataFrame:
        if cls._events_cache is not None:
            return cls._events_cache
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "backtester" / "surprise earning.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"PEAD requires {path.name} (the SUE event panel) but it was "
                f"not found at {path}. If running in Docker, check .dockerignore "
                f"isn't excluding backtester/*.csv."
            )
        df = pd.read_csv(path, usecols=["TICKER", "OFTIC", "anndats", "suescore"])
        # OFTIC is the standard exchange ticker; TICKER is the WRDS internal id
        # (e.g. ARRA for Agilent). Prefer OFTIC, fall back to TICKER.
        df["ticker"] = df["OFTIC"].fillna(df["TICKER"]).astype(str).str.upper().str.replace(".", "-", regex=False)
        df["anndats"] = pd.to_datetime(df["anndats"], errors="coerce")
        df = df.dropna(subset=["anndats", "suescore", "ticker"])
        df = df[["ticker", "anndats", "suescore"]].sort_values(["ticker", "anndats"])
        cls._events_cache = df
        return df

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        events = self._load_events()
        prices = dataset.prices
        scores = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns, dtype=float)
        if events.empty:
            return scores
        index = prices.index
        if not index.is_monotonic_increasing:
            index = index.sort_values()
        relevant = events[events["ticker"].isin(prices.columns)]
        relevant = relevant[(relevant["anndats"] >= index.min()) & (relevant["anndats"] <= index.max())]
        for ticker, sub in relevant.groupby("ticker"):
            col = scores[ticker].copy()
            for ann_date, sue in zip(sub["anndats"].values, sub["suescore"].values):
                # Find the next trading day on or after the announcement.
                pos = index.searchsorted(pd.Timestamp(ann_date), side="left")
                if pos >= len(index):
                    continue
                end = min(pos + self.holding_days, len(index))
                col.iloc[pos:end] = float(sue)
            scores[ticker] = col
        return scores


class MovingAverageCrossoverStrategy(SignalStrategy):
    name = "Simple Moving Average Crossover"
    motivation = "Long names whose short SMA has crossed above the long SMA — the textbook trend-following signal."
    economic_rationale = "Trend-following monetises momentum at the single-name level: when the fast moving average leads the slow one, recent demand has been outpacing supply."
    why_it_works = "Simple windows produce sharper crossovers than exponential weighting, so cross-sectional ranks turn over decisively when a stock breaks trend; pairing with a low rebalance frequency keeps turnover manageable."
    why_it_fails = "Choppy or mean-reverting regimes whipsaw the signal; transaction costs and signal lag can wipe out the edge."

    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        prices = dataset.prices
        # Rolling SMA, not EMA — the label and the catalog formula both say
        # "MA". EMA's exponential decay smooths the cross-section so heavily
        # that small window tweaks barely move the rank ordering, which read
        # as "Sharpe doesn't change" from the UI. Hard-window SMA flips
        # observations in/out as the window slides, so 50→60 actually shifts
        # the portfolio.
        short_ma = prices.rolling(self.short_window, min_periods=self.short_window).mean()
        long_ma  = prices.rolling(self.long_window,  min_periods=self.long_window).mean()
        return (short_ma - long_ma).div(long_ma.replace(0, np.nan))


class MLRidgeStrategy(SignalStrategy):
    name = "ML Ridge (rolling OLS)"
    motivation = "Let a rolling ridge regression blend momentum, short-term reversal, volatility, and market beta rather than hand-picking weights."
    economic_rationale = "Each input carries an independent risk premium. A data-driven linear combination lets their relative payoffs rotate with regime instead of relying on static weights."
    why_it_works = "Two-year rolling fit averages out single-period noise while still tracking coefficient drift. Ridge stabilises coefficients when momentum and reversal are correlated at the sampling horizon."
    why_it_fails = "Structural breaks (2008, 2020) briefly invalidate the pre-break fit; when all features decorrelate from forward returns, coefficients collapse toward zero."

    def __init__(
        self,
        training_lookback_days: int = 504,
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        reversal_lookback: int = 5,
        vol_lookback: int = 60,
        beta_lookback: int = 60,
        forward_horizon: int = 5,
        refit_frequency: str = "ME",
        ridge_lambda: float = 1.0,
        holdout_start: Optional[str] = None,
    ):
        self.training_lookback_days = training_lookback_days
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.reversal_lookback = reversal_lookback
        self.vol_lookback = vol_lookback
        self.beta_lookback = beta_lookback
        self.forward_horizon = forward_horizon
        self.refit_frequency = refit_frequency
        self.ridge_lambda = ridge_lambda
        self.holdout_start = holdout_start  # optional: refit only up to here for a hard hold-out

    def _build_features(self, dataset: ResearchDataset) -> dict[str, pd.DataFrame]:
        prices = dataset.prices
        returns = dataset.returns
        benchmark = dataset.benchmark_returns.reindex(returns.index).fillna(0.0)

        span = max(self.momentum_lookback - self.momentum_skip, 1)
        mom = prices.pct_change(span, fill_method=None).shift(self.momentum_skip)
        rev = -prices.pct_change(self.reversal_lookback, fill_method=None)
        vol = returns.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2).std()
        cov = returns.rolling(self.beta_lookback, min_periods=self.beta_lookback // 2).cov(benchmark)
        var = benchmark.rolling(self.beta_lookback, min_periods=self.beta_lookback // 2).var().replace(0, np.nan)
        beta = cov.div(var, axis=0)

        return {
            "mom": _cross_sectional_zscore(mom),
            "rev": _cross_sectional_zscore(rev),
            "vol": _cross_sectional_zscore(vol),
            "beta": _cross_sectional_zscore(beta),
        }

    def generate_scores(self, dataset: ResearchDataset) -> pd.DataFrame:
        from backtester.research_backtester import _get_rebalance_dates

        returns = dataset.returns
        features = self._build_features(dataset)
        feature_names = list(features.keys())
        n_features = len(feature_names)

        panel = np.stack(
            [features[k].reindex_like(returns).values for k in feature_names],
            axis=-1,
        )

        log_r = np.log1p(returns.fillna(0.0).values)
        T, N = log_r.shape
        cum = np.concatenate([np.zeros((1, N)), np.cumsum(log_r, axis=0)], axis=0)
        fwd_log = np.full((T, N), np.nan)
        H = self.forward_horizon
        if T > H:
            fwd_log[: T - H] = cum[H + 1 : T + 1] - cum[1 : T - H + 1]
        fwd_returns = np.expm1(fwd_log)

        rebalance_dates = _get_rebalance_dates(returns.index, self.refit_frequency)
        scores_out = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
        positions = {d: i for i, d in enumerate(returns.index)}
        holdout_ts = pd.Timestamp(self.holdout_start) if self.holdout_start else None

        for date in rebalance_dates:
            pos = positions.get(date)
            if pos is None:
                continue
            train_end_pos = pos - H - 1
            if holdout_ts is not None and date >= holdout_ts:
                holdout_pos = positions.get(holdout_ts)
                if holdout_pos is not None:
                    train_end_pos = min(train_end_pos, holdout_pos - H - 1)
            train_start_pos = max(train_end_pos - self.training_lookback_days + 1, 0)
            if train_end_pos - train_start_pos < 100:
                continue

            X_slice = panel[train_start_pos : train_end_pos + 1].reshape(-1, n_features)
            y_slice = fwd_returns[train_start_pos : train_end_pos + 1].reshape(-1)
            mask = np.all(np.isfinite(X_slice), axis=1) & np.isfinite(y_slice)
            if mask.sum() < 100:
                continue
            X = X_slice[mask]
            y = y_slice[mask]

            XtX = X.T @ X + self.ridge_lambda * np.eye(n_features)
            try:
                coefs = np.linalg.solve(XtX, X.T @ y)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coefs)):
                continue

            X_today = panel[pos]
            valid = np.all(np.isfinite(X_today), axis=1)
            X_masked = np.where(valid[:, None], X_today, 0.0)
            with np.errstate(all="ignore"):
                preds = X_masked @ coefs
            scores_out.iloc[pos] = np.where(valid, preds, np.nan)

        return scores_out.ffill()


def build_default_strategy_suite() -> list[SignalStrategy]:
    return [
        SmallCapTiltStrategy(),
        ValueCompositeStrategy(),
        EarningsRevisionStrategy(),
        SectorNeutralDividendYieldStrategy(),
        CrossSectionalMomentumStrategy(),
        LowVolatilityStrategy(),
        MLRidgeStrategy(),
    ]
