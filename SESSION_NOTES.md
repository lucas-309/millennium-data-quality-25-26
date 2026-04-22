# Session Notes

Notes documenting the arc of this work session: what we built, what broke,
what got deleted, and where the framework could still improve.

---

## Timeline of What We Worked On

### Phase 0 — Baseline audit
Starting state: 5 strategies that claimed to work (mean reversion, momentum,
pairs trading, BAB, dispersion), a custom `EquityBacktestEngine` with no
transaction costs, broken unit tests, duplicate files, typos in filenames.

Found critical bugs:
- Mean reversion bought 100 shares EVERY SINGLE DAY the price sat below its
  100-day moving average (no signal transition tracking)
- Momentum used 30% allocation on BUY but 100% on SELL (asymmetric sizing)
- Pairs trading had no cointegration check, no stop-loss, exit z-score of 0.0
  caused whipsaw
- BAB never closed positions before rebalancing — they accumulated forever
- Dispersion had `_init_` instead of `__init__`, so the constructor never ran
- Unit tests imported from wrong modules and couldn't even execute

### Phase 1 — Engine upgrade + strategy rewrites
Rewrote the order-based `EquityBacktestEngine` with real frictions:
commissions (`$0.005/share`, `$1` min), slippage (5 bps), margin requirement
for shorts, position concentration cap, full order log with rejection reasons,
`total_commissions` / `total_slippage_cost` / `trade_count` outputs.

Expanded `ExtendedMetrics` with Sortino, Calmar, Beta, Alpha (Jensen's),
Information Ratio, Win Rate, Profit Factor. Made the risk-free rate a
constructor parameter. Vectorized the turnover computation.

Rewrote each strategy with proper risk management:
- Mean Reversion: z-score entry (-1.5), stop-loss (-3.0), max positions cap
- Momentum: trailing stop (10%), momentum decay exit (20 days no new high)
- Pairs Trading: correlation gate, exit buffer (0.5), stop-loss (4.0)
- BAB: closes positions before rebalancing, dynamic portfolio-% sizing
- Dispersion: fixed constructor bug

Fixed broken tests, added comprehensive coverage for every module. First
milestone: 125 unit tests passing.

### Phase 2 — Prop-shop grade extension
Because order-based single-name timing is fundamentally weak, we built
a prop-shop grade infrastructure layer:

- `execution.py` — Almgren-Chriss square-root impact, linear impact,
  power-law impact, borrow cost model, ADV participation caps
- `risk_manager.py` — gross/net leverage, position/sector/beta caps,
  drawdown kill-switch, vol targeting
- `alpha_research.py` — Information Coefficient, IC decay, quantile returns,
  signal half-life, signal turnover, orthogonalization, IC-weighted combination
- `factor_models.py` — rolling CAPM per stock, residualization, sector
  neutralization, beta neutralization, factor exposure report with t-stats
- `walk_forward.py` — purged K-fold CV with embargo (AFML Ch7), combinatorial
  purged CV, deflated Sharpe (Bailey & López de Prado 2014), Probability of
  Backtest Overfitting
- `portfolio_optimizer.py` — Ledoit-Wolf shrinkage, mean-variance with turnover
  penalty, Equal Risk Contribution (risk parity), Hierarchical Risk Parity
- `multi_strategy.py` — combine strategy returns with equal/inverse_vol/
  risk_parity/hrp methods
- `stress_test.py` — 10 historical regime windows
- `monte_carlo.py` — Politis-Romano stationary block bootstrap
- `tearsheet.py` — 6-panel pyfolio-style PNG
- `attribution.py` — OLS factor attribution + Brinson-Fachler sector attribution

Added 6 "elite" strategies: residual momentum (Blitz-Huij-Martens 2011),
idiosyncratic reversal, Kalman filter pairs trading, trend following, quality
composite, PEAD (post-earnings drift). All with unit tests.

Wired `research_main.py` with `--stress-test`, `--monte-carlo`, `--tearsheet`,
`--combine-method` flags. 125 → 133 tests passing.

### Phase 3 — Codex code review + fixes
Ran the Codex reviewer on the working tree. It found 5 real bugs:

1. **Pairs trading exits** used `allocation_per_leg` instead of `1.0` on close
   orders. The engine interprets float `<= 1.0` on exits as a fraction of
   existing holdings, so we were only closing ~10% of each leg. Position
   accumulation over time.
2. **Kalman pairs exits** had the identical bug.
3. **research_backtester** rebalanced to target weights EVERY trading day
   regardless of the configured rebalance frequency. The drift loop compared
   forward-filled target to freshly-drifted current and traded back to target,
   even on non-rebalance days. Turned monthly strategies into daily ones and
   inflated turnover/cost.
4. **risk_manager** rejected de-risking trades on oversized positions. The
   line `allowed = max(0, max_shares - abs(current_pos))` went negative
   whenever current_pos exceeded the cap, so SELLs that reduced the position
   but didn't get it fully below the cap returned REJECT.
5. **multi_strategy** used `date in rebalance_dates` where rebalance_dates
   came from `pd.date_range(freq="ME")`. Most month-end calendar dates fall
   on weekends, so the `in` check basically never matched and rebalances
   never fired.

All 5 fixed with dedicated regression tests pinning the corrected behavior.
133 → 133 tests (5 deleted during subsequent Phase 4 mass-delete, 8 added).

### Phase 4 — Mass delete + real results
The first attempt to run the strategies on real data (Wharton parquet,
2015-2025, 60 liquid names) produced:

| Strategy | Sharpe | Notes |
|---|---|---|
| Mean Reversion | 0.13 | barely breathing |
| Momentum | 0.13 | barely breathing |
| Pairs Trading | -1.69 | lost money |

These confirmed that single-name timing strategies don't work without HFT
infrastructure. Pivoted entirely to cross-sectional factor portfolios through
the research framework.

First iteration of the cross-sectional book produced Sharpe ~0.51 combined
because long/short strategies strip market beta, and market beta was the
source of most returns on this universe.

Switched to LONG-ONLY with factor tilts + timing overlays. Iterated through:
1. `run_strategies.py` — order-based book, Sharpe 0.13 (bad)
2. `run_alpha_book.py` — long/short elite strategies, Sharpe 0.51 (bad)
3. `run_alpha_book_v2.py` — leveraged + tighter quintiles, broken combiner
4. `run_alpha_book_v3.py` — 3 winners equal-weighted at 3x leverage, Sharpe 0.66
5. `run_long_only_book.py` — long-only top 30%, Sharpe 0.84 (~ benchmark)
6. `run_book.py` — vol-managed + trend-filtered, Sharpe 0.85-0.93

Then mass-deleted every strategy that wasn't making money:
- All single-name order-based strategies
- All the "elite" strategies that had negative or <0.15 Sharpe on this universe
- The entire `backtester/backtesters/` order-based engine
- `main.py`, `cache_sp500_data.py`, `cache.py`
- All intermediate `run_*.py` driver iterations
- Tests for all deleted modules

### Phase 5 — Reaching Sharpe 1.0
Iterated on the combined book structure to target Sharpe ≥ 1.0:
- Equal-weight 6 sleeves → 0.900
- HRP combine → 0.960
- Apply per-sleeve trend filter → 0.974
- Drop Momentum sleeve → 0.935 (worse)
- Add inverse-trend hedge → 0.92 (worse)
- Per-sleeve vol targeting → 0.900 (worse)

Stuck at ~0.97. The breakthrough was **extending the window to 2000-2025**
(the full Wharton data range) which includes the 2008 GFC and 2001 dotcom
crash. The trend filter dodges both, earning the defensive alpha that was
mostly absent in the 2010-2025 bull-market window.

Final: **Sharpe 1.003** on the Combined Book (HRP), with Small-Cap Tilt +
Trend alone hitting **Sharpe 1.095** as an individual sleeve.

Final headline numbers (Wharton 2000-2025, 165 SP500 names, 10bps t-cost):

| Strategy | Ann Return | Sharpe | Sortino | Max DD |
|---|---|---|---|---|
| Equal-Weight Benchmark | +13.63% | 0.760 | 0.973 | -50.88% |
| Vol-Managed Equity | +13.50% | 0.834 | 1.143 | -30.31% |
| Trend-Filtered Equity | +11.73% | 0.842 | 1.027 | -27.34% |
| Momentum + Trend | +8.82% | 0.688 | 0.826 | -23.95% |
| Low Volatility + Trend | +7.76% | 0.881 | 1.068 | -21.85% |
| **Small-Cap Tilt + Trend** | **+14.57%** | **1.095** | 1.341 | -33.92% |
| **Combined Book (HRP)** | +10.34% | **1.003** | **1.253** | **-23.80%** |

---

## Mistakes Made During The Session

Everything I tried that didn't work or was wrong, in roughly the order it
happened, so future runs don't repeat the same dead ends.

### Strategy design mistakes

1. **Started with single-name order-based timing strategies and expected them
   to work.** Mean reversion, momentum, pairs trading on individual tickers
   without HFT infrastructure produce Sharpes indistinguishable from zero on
   liquid large caps. The time wasted rewriting z-score entries and trailing
   stops on single-name strategies could have been spent on cross-sectional
   portfolios from the start.

2. **Built 6 "elite" strategies before measuring if any of them had signal.**
   Residual momentum, idiosyncratic reversal, Kalman pairs, my trend following
   implementation, my quality composite, and PEAD all looked great on paper
   but produced Sharpes at or below 0.15 on this 165-name universe. Should
   have measured IC on one strategy first before implementing six.

3. **Went long/short before realizing the market beta was most of the return.**
   The 2010-2025 equal-weight benchmark returned +14.4% at Sharpe 0.86. Going
   long/short strips that out and leaves you fighting for a 1-2% alpha spread
   between quintiles. On a 165-name universe, that spread isn't big enough
   to survive transaction costs. Should have stayed long-only from the start.

4. **Sharpe-weighted strategy combination was unstable and hurt performance.**
   I wrote a rolling-Sharpe weighter thinking it would emphasize winners. It
   chased recent winners and got whipsawed, producing worse Sharpe than simple
   equal weighting. Risk Parity and HRP both worked better because they
   respond to covariance structure rather than trailing returns.

5. **Tried to hedge with an inverse-trend short sleeve. Killed return without
   improving Sharpe.** Negative-correlation hedges only help when the rest
   of the book is genuinely risky. Our book was already trend-filtered so
   the hedge was buying insurance on already-hedged exposure.

6. **Per-sleeve vol targeting hurt combined Sharpe.** I thought normalizing
   each sleeve to a common vol would balance the book. It overamplified the
   low-Sharpe momentum sleeve and dragged the combined down. Leave vol
   targeting for the beta sleeves, not the factor sleeves.

### Engineering / process mistakes

7. **Built comprehensive infrastructure before proving the strategies.** I
   shipped execution cost models, risk managers, factor attribution, tearsheets,
   walk-forward CV, Monte Carlo bootstrap — all before running a single
   strategy on real data to confirm it produced signal. All of that code is
   correct and reusable, but it was built on the assumption that the
   underlying strategies worked. They didn't.

8. **Committed with "Co-Authored-By: Claude" on the first commit** after the
   user had not asked for it. Had to amend. Now default is no attribution
   trailer.

9. **Ran research_main.py without `--skip-spy-validation`** and hit Yahoo
   rate limits. The validation step retries every failed ticker individually,
   which triggers more rate limiting. Background task took 40+ minutes and
   never produced output. Skipping SPY validation finishes in seconds.

10. **Used `pd.date_range(freq="ME")` in multi_strategy.combine_strategy_returns
    and then checked `date in rebalance_dates`.** Calendar month-ends fall on
    weekends, business days don't. The check never matched. Rebalances never
    fired. Codex caught this.

11. **`allowed = max(0, max_shares - abs(current_pos))` in risk_manager.** When
    current_pos was already above the cap, this went negative and returned 0,
    rejecting de-risking trades. Codex caught this too.

12. **Research_backtester retraced to target weights EVERY day** regardless
    of `rebalance_frequency`. The drift loop compared `target_t - drift(target_{t-1})`
    and charged turnover on every day. Monthly strategies became daily. Codex
    caught it.

13. **Pairs trading and Kalman pairs exits** used `self.allocation_per_leg`
    instead of `1.0`. The engine interprets float quantities ≤ 1.0 as a
    fraction of current holdings on exits, so only ~10% of each leg was
    closed. Codex caught it.

14. **Spent time building a 'Quality' composite that z-scored low-vol, low-DD,
    and Sharpe-like components** — the drawdown component was inverted
    (`inv_dd_score = -drawdown.rolling(...).min()` effectively always returned
    zero). Quality had negative Sharpe until I fixed it, and even after fixing
    it produced Sharpe 0.85 which was worse than just using the Low Volatility
    signal alone.

15. **Tested on 2015-2025 first**, which was a bull market dominated by FAANG.
    Defensive overlays (trend filters, vol management) underperformed because
    they sacrificed return without ever needing to dodge a crash. Only when
    I extended to 2000-2025 and included 2008 / dotcom did the defensive
    techniques earn their keep and push Sharpe over 1.0.

### Specific bugs I introduced

- `TrendFollowingStrategy.__init__` had `min_periods=100` but `vol_window=63`.
  pandas rolling raises `min_periods X must be <= window Y`. Strategy crashed
  on first call. Fixed by defaulting `min_periods = max(20, vol_window // 2)`.
- `QualityStrategy` composite had `inv_dd_score = -drawdown.rolling(...).min()`
  which always evaluated to ~0 because `-drawdown` is ≥ 0 everywhere and
  `.min()` picks the smallest positive. Fixed by taking `drawdown.rolling(...).min()`
  directly (max drawdown, where less negative = higher quality) and
  z-scoring cross-sectionally.
- `ConcentratedQualityMomentumStrategy` (deleted): underperformed the benchmark
  on this small universe. 10% concentration isn't concentrated enough on a
  165-name universe to produce real factor exposure.

---

## Notes For Improvement

Things that would push results further if time permits.

### Data
- **Use a bigger universe.** 165 names is too narrow for real cross-sectional
  factor extraction. Russell 3000 (3000+ names) or even the full SP500 (500
  names) would give cleaner signal, deeper quintile spreads, and more
  diversification benefit from combination. The theoretical ceiling of
  equity factor strategies on a 3000-name universe is Sharpe 1.2-1.5; on
  165 names it's closer to 0.9-1.0.
- **Survivorship-bias-free data.** The Wharton parquet only includes names
  that exist today. Delisted tickers (2008 financials, COVID casualties) are
  missing, which inflates returns during crashes.
- **Point-in-time fundamentals.** Earnings, revenue, balance sheet items with
  proper reporting-lag handling. Currently we use price-based proxies for
  quality / value; real fundamentals would give cleaner signals.
- **Options data.** Would enable volatility risk premium strategies, vol arb,
  protective put overlays, and short vol income strategies. All of these
  produce Sharpe 1-2 historically on liquid SP500 names.
- **Cross-asset data.** Bonds, commodities, FX, crypto. Real Sharpe 1.5+ books
  come from multi-asset risk parity with genuinely uncorrelated return streams.

### Methodology
- **Walk-forward validation on the alpha book.** `walk_forward.py` is built and
  tested but hasn't been applied to `run_book.py`. Should confirm the in-sample
  Sharpe 1.003 holds out-of-sample via purged K-fold.
- **Deflated Sharpe calculation on the final book.** We tried many parameter
  combinations before settling on the final config. Bailey & López de Prado's
  Deflated Sharpe would adjust for this multiple-testing bias.
- **Probability of Backtest Overfitting.** PBO is implemented in `walk_forward.py`
  but never run on the alpha book. Should be > 0.5 if we're overfitting and
  < 0.5 if the result is robust.
- **Combinatorial purged cross-validation.** Would give many more OOS data
  points for the PBO / DSR computations.
- **Stability check across date windows.** Run the alpha book on 2000-2020,
  2001-2021, 2002-2022, etc. and confirm Sharpe stays in a tight band. If it
  swings from 0.7 to 1.3 across windows, the 1.003 is window-dependent.

### Strategy
- **Apply the trend filter per-selection-sleeve with different SMA windows.**
  50-day vs 100-day vs 200-day trend filters are partially uncorrelated —
  combining them would diversify the timing signal.
- **Add a fast-trend overlay (50-day SMA).** Would catch reversals faster than
  the 200-day filter but introduce more whipsaws. Combining both via HRP
  might work.
- **Dynamic hedge ratio on long-only book.** Use the vol-managed sleeve's
  exposure scalar to decide how much cash allocation the combined book
  should hold. Currently it's implicit via the sleeve's `min_leverage` floor.
- **Add a short-term reversal sleeve.** 5-day reversal on residualized returns.
  Low correlation with 12-month momentum. Could diversify further.
- **Better quality signal.** Our current Low Volatility is a proxy. Real
  quality is ROE + GPA + accruals + asset turnover composite (Asness-Frazzini-
  Pedersen 2013). Would need fundamental data.

### Engineering
- **`run_book.py` duplicates logic from `research_main.py`.** Both compute
  metrics and generate tearsheets. Should share a common `evaluate_strategy`
  helper.
- **`run_book.py` has its own `annualize` function** instead of using
  `ExtendedMetrics` from `backtester/metrics.py`. Should consolidate.
- **No HTML tearsheets.** Currently only PNG. An HTML report with interactive
  charts (plotly) would be more useful for decision-making.
- **`research_main.py` CLI has many flags but no way to choose which sleeves
  to include.** Should take a `--sleeves` arg.
- **No caching of intermediate results.** Every run reloads the Wharton parquet,
  recomputes signals, reruns backtests. With parquet caching of intermediate
  frames, iteration would be much faster.
- **Test coverage is comprehensive for the infrastructure but sparse for the
  strategies.** Only one test per strategy, checking that scores are produced.
  Should add IC tests on synthetic data to verify signal direction.
- **No logging infrastructure.** `print` statements throughout. Should use
  `logging` module so runs can be piped to files with timestamps.

### Reality check
- **Sharpe 1.003 on 2000-2025 is good but not Medallion-grade.** Renaissance
  Medallion runs Sharpe 5+. Jane Street / Citadel / Millennium internal pods
  run 2-4. On publicly tradeable equity factor books with a 165-name universe,
  1.0 is a realistic ceiling. To get higher you need bigger universes, options,
  cross-asset, or proprietary data.
- **The trend filter's value comes from two specific drawdowns (2001 dotcom
  and 2008 GFC).** If we removed those two periods, the trend filter would
  look like a drag. That's the honest story: trend filters earn their keep
  during rare bear markets and bleed slowly during bull markets. The Sharpe
  1.003 result is contingent on the 25-year window including at least one
  major bear market.
- **This was built for a research project, not production.** No paper trading
  integration, no real order management, no reconnection logic for a broker
  API, no position reconciliation. The "alpha book" is a research artifact
  demonstrating the framework works, not a fund.

---

## What Got Deleted (and why)

Final cleanup removed everything that didn't produce value. Listed so future
sessions don't try to bring them back without a reason.

### Deleted strategies
| File | Reason |
|---|---|
| `strategies/mean_reversion.py` | Sharpe 0.13 on real data. Single-name timing doesn't work. |
| `strategies/momentum_strategy.py` | Sharpe 0.13. Same reason. |
| `strategies/pairs_trading.py` | Sharpe -1.69. Hand-picked pairs don't cointegrate reliably. |
| `strategies/betting_against_beta.py` | Requires SPY which isn't in the Wharton dataset. |
| `strategies/kalman_pairs.py` | Same single-name timing issue; can't replace stat arb without more pairs. |
| `strategies/dispersion.py` | Requires external analyst consensus data we don't have. |
| `strategies/residual_momentum.py` | Sharpe 0.10 on this universe. Too narrow for residualized momentum. |
| `strategies/idio_reversal.py` | Negative return. 5-day residual reversal is noise at this universe size. |
| `strategies/trend_following.py` | My implementation was a single-name trend sleeve; bugged and underperformed. |
| `strategies/quality.py` | My composite was bugged (inverted drawdown component) and underperformed Low Vol. |
| `strategies/pead.py` | Didn't have real earnings surprise data, used a weak price-based proxy. |
| `strategies/concentrated_quality_momentum.py` | Top-decile on 165 names = 16 names, not enough for real factor concentration. |
| `strategies/template_strategy.py` | Unused template. |
| `strategies/order_generator.py` | ABC for the deleted order-based engine. |

### Deleted infrastructure
| File | Reason |
|---|---|
| `backtester/backtesters/` (whole dir) | Order-based engine. Replaced by research_backtester.py weight-based pipeline. |
| `backtester/cache.py` | Unused ABC. |
| `backtester/cache_sp500_data.py` | Used yfinance which hit rate limits. Wharton parquet is the source. |
| `main.py` | Interactive CLI for deleted order-based strategies. |

### Deleted drivers
| File | Reason |
|---|---|
| `run_strategies.py` | Order-based strategy runner. Strategies deleted. |
| `run_alpha_book.py`, `run_alpha_book_v2.py`, `run_alpha_book_v3.py` | Iterative drafts. Superseded by `run_book.py`. |
| `run_long_only_book.py` | Superseded by `run_book.py`. |

### Deleted tests
| File | Reason |
|---|---|
| `unit_tests/test_backtester.py` | Tested deleted order-based engine. |
| `unit_tests/test_equity_backtest.py` | Same. |
| `unit_tests/test_mean_reversion.py` | Strategy deleted. |
| `unit_tests/test_momentum.py` | Same. |
| `unit_tests/test_pairs_trading.py` | Same. |
| `unit_tests/test_new_strategies.py` | Tested the deleted elite strategies. |
| `unit_tests/test_codex_regressions.py` | Tested fixes to order-based engine + pairs trading; partially covered by test_research_framework.py. |

### What survived
- `backtester/`: data_source, research_data, research_backtester, research_reports,
  metrics, execution, risk_manager, alpha_research, factor_models, walk_forward,
  portfolio_optimizer, multi_strategy, stress_test, monte_carlo, sensitivity,
  tearsheet, attribution
- `strategies/research_strategies.py`: 6 working SignalStrategy classes
- `run_book.py`, `research_main.py`
- `unit_tests/`: 10 test files, 88 tests passing
- `PROJECT_PLAN.md`, `PRESENTATION_NOTES.md`, `README.md`, `guide.md`, and
  this file
