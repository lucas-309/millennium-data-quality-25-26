# millennium-data-quality-25-26 — agent notes

Canonical repo-ops file; `AGENTS.md` is a symlink to it.

## Deploy guardrails (cost footguns)

- The backend is ONE Fly machine (`cds-millennium-backtester`, shared-cpu-4x /
  8GB) that scales to zero. Always `fly deploy --ha=false --remote-only`; plain
  `fly deploy` adds a second machine. Two always-on machines cost ~$87/mo in
  May 2026, which is why the app was destroyed on 2026-06-02 and redeployed
  scale-to-zero on 2026-10-05.
- Do not change `auto_stop_machines` / `min_machines_running` in `fly.toml`
  unless Lucas asks. Idle cost must stay ~$0.
- Public IPs: free kinds only — `fly ips allocate-v4 --shared` and
  `fly ips allocate-v6`. A dedicated v4 is $2/mo. flyctl has skipped the IPv6
  step on a first deploy; check `fly ips list` after deploying.
- `data_cache/` and `sp500_universe.pkl` are gitignored but `COPY`'d into the
  image: run `python run_sp500_fetch.py --refresh-universe --prune --no-info`
  before `fly deploy`, or the build fails / ships a stale universe.
- The frontend deploys from `main` only (Vercel). Pushing `simulation` does
  not ship prod.

## State (2026-10-05)

Live: frontend on Vercel, backend one scale-to-zero machine. yfinance universe
refreshed to the Oct 2026 S&P 500 (503 tickers). Wharton panel is static and
ends 2025-12-10. Full runbook: README → Deploy.
