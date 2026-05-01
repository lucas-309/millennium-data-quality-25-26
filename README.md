# Millennium Quantitative Research Playground

A two-track quantitative research project:

1. **Live strategy simulator** — a deployed web app where you can pick a
   strategy, tune parameters, edit the engine code in the browser, and watch
   the equity curve update against either the live yfinance S&P 500 cache
   or a Wharton WRDS panel.
2. **Research toolkit** — long-only cross-sectional equity factor strategies
   on a local Wharton / Compustat daily dataset. Alpha research (IC, quantile
   spread, factor models, walk-forward validation), HRP portfolio
   optimization, stress testing, tearsheets.

## Live simulator

- Frontend: [cds-millennium-backtester.vercel.app](https://cds-millennium-backtester.vercel.app/)
- Backend: [cds-millennium-backtester.fly.dev](https://cds-millennium-backtester.fly.dev/api/status)

Three strategies in the catalog today:

| ID | Strategy | Source class |
|----|----------|--------------|
| `momentum` | Cross-sectional momentum (12-1) | `CrossSectionalMomentumStrategy` |
| `mean_reversion` | Short-term reversal | `ShortTermReversalStrategy` |
| `value_composite` | Fama-French value blend (div yield + E/P + size) | `ValueCompositeStrategy` |

Two data sources, switchable from the topbar:

- `yfinance` — today's ~503 S&P 500 constituents, refreshed daily via
  `run_sp500_fetch.py` and persisted to `data_cache/yfinance/`.
- `wharton` — the static Wharton WRDS panel
  (`backtester/WhartonDataSource4.parquet`, ~830 tickers across the historic
  S&P 500 membership). Survivorship-bias-aware.

Headline features:

- **Live editable engine code panel.** Edit the strategy's Python source in
  the browser and the backend re-compiles + runs your version on the same
  data + engine knobs.
- **Per-tab pinning.** Run a config, hit Pin, change params, and the old
  curve stays on the chart for comparison.
- **Survivorship audit panel** quantifies the upward bias from running on
  today's index members.
- **Universe filter** — restrict the run to a specific ticker subset (Apply
  / Exclude / Reset).

## Research toolkit (offline)

The research-grade alpha book lives in `run_book.py`: a 5-sleeve combination
of vol-managed equity, trend-filtered equity, and three long-only factor
sleeves (Momentum, Low Volatility, Small-Cap Tilt) via Hierarchical Risk
Parity.

Best result on Wharton 2000-2025 (165 SP500 names, 10bps t-cost, no
lookahead):

| Strategy | Ann Return | Sharpe | Sortino | Max DD |
|---|---|---|---|---|
| Equal-Weight Benchmark | +13.63% | 0.760 | 0.973 | -50.88% |
| Small-Cap Tilt + Trend | +14.57% | **1.095** | 1.341 | -33.92% |
| **Combined Book (HRP)** | +10.34% | **1.003** | **1.253** | **-23.80%** |

## Installation

```sh
git clone https://github.com/lucas-309/millennium-data-quality-25-26
cd millennium-data-quality-25-26
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Data files:

- `backtester/WhartonDataSource4.parquet` — the simulator's Wharton source.
- `backtester/financial_ratios.parquet` — Wharton WRDS Financial Ratios
  panel used by `ValueCompositeStrategy`.
- `backtester/surprise earning.csv` — SUE event panel (loaded lazily by the
  PEAD strategy when present).

## Quick start

### Run the live simulator locally

```sh
.venv/bin/python -m simulation.backend.server --host 127.0.0.1 --port 8765
# open simulation/frontend/index.html — or set up the dev workflow:
cd simulation/frontend && npm install && npm run watch
```

The frontend is a single TypeScript file that compiles to `app.js` and gets
served by the Python backend.

For snappy yfinance ↔ wharton toggling locally:

```sh
MILLENNIUM_DATASET_CACHE=1 .venv/bin/python -m simulation.backend.server
```

This holds both panels in memory so a return trip is a pointer-swap. Off by
default in production (the Fly VM only loads one source at a time and frees
the previous panel before reading the new one).

### Run the offline alpha book

```sh
.venv/bin/python run_book.py
```

Uses the default window (2000-01-01 to 2025-01-01), 16% target vol, 200-day
trend filter, long-only top 20% selection sleeves. Outputs go to
`book_results/`.

Customize:
```sh
.venv/bin/python run_book.py \
  --start 2010-01-01 --end 2025-01-01 \
  --target-vol 0.16 --max-leverage 2.0 \
  --sma-window 200 --selection-top-pct 0.20 \
  --output-dir my_book_results
```

### Run the research pipeline (multi-signal suite)

```sh
.venv/bin/python research_main.py --skip-spy-validation \
  --stress-test --monte-carlo 500 --tearsheet \
  --combine-method hrp
```

This runs the `build_default_strategy_suite()` slate through the
weight-based research backtester with lag tables, event studies,
tearsheets, stress regimes, Monte Carlo CIs, and a combined-book output.

## Architecture

See [guide.md](guide.md) for the current project structure and module
index. Research workflow detail: [PROJECT_PLAN.md](PROJECT_PLAN.md).
Presentation notes: [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md).
Session notes: [SESSION_NOTES.md](SESSION_NOTES.md).

## Deploy

- **Frontend (Vercel)** — auto-deploys from `main`. Static TypeScript build,
  rewrites `/api/*` to the Fly backend (see `simulation/frontend/vercel.json`).
- **Backend (Fly)** — `fly deploy --remote-only`. The VM runs
  `simulation/backend/server.py` on shared-cpu-4x with 8GB memory (sized for
  the wharton load + simulate peak; see `fly.toml`).

## Running tests

```sh
.venv/bin/python -m unittest discover -s unit_tests
```

88 tests covering the research framework, alpha research tools, factor
models, walk-forward validation, portfolio optimizer, stress testing,
Monte Carlo, tearsheet, execution costs, and risk manager.
