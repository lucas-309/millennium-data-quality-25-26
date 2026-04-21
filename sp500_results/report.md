# S&P 500 yfinance backtest

- Window: 2010-01-01 → 2024-12-31
- Universe: 499 tickers (of 503 current S&P 500 members)
- Rebalance: monthly, inverse-vol, long-only, top 20% by score
- Transaction cost: 10.0 bps per side

## Performance summary

| strategy | sharpe | sortino | ann_return | ann_vol | max_drawdown | ann_turnover | ann_tcost_drag | universe_size | inception_biased_count |
|---|---|---|---|---|---|---|---|---|---|
| Small-Cap Tilt | 1.400 | 1.848 | 0.326 | 0.219 | -0.400 | 2.229 | 0.002 | 499 | 76 |
| Value Composite | 1.283 | 1.633 | 0.301 | 0.225 | -0.465 | 3.390 | 0.003 | 499 | 76 |
| EPS Revision | — | — | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 499 | 76 |
| Sector-Neutral Dividend Yield | 0.888 | 1.100 | 0.159 | 0.186 | -0.435 | 3.130 | 0.003 | 499 | 76 |
| Cross-Sectional Momentum | 0.989 | 1.211 | 0.174 | 0.178 | -0.389 | 9.450 | 0.009 | 499 | 76 |
| Low Volatility | 0.961 | 1.144 | 0.127 | 0.134 | -0.343 | 3.491 | 0.003 | 499 | 76 |
| ML Ridge (rolling OLS) | 0.825 | 1.010 | 0.153 | 0.196 | -0.389 | 12.159 | 0.012 | 499 | 76 |

## Factor attribution (FF3 + MOM proxies)

| strategy | alpha_annual | alpha_tstat | r_squared | n_obs | beta_MKT | tstat_MKT | contrib_MKT | beta_SMB | tstat_SMB | contrib_SMB | beta_HML | tstat_HML | contrib_HML | beta_MOM | tstat_MOM | contrib_MOM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Small-Cap Tilt | 0.035 | 3.015 | 0.960 | 3774 | 1.021 | 235.923 | 0.188 | 0.733 | 55.423 | 0.074 | 0.081 | 7.460 | 0.011 | -0.012 | -1.952 | -0.000 |
| Value Composite | 0.002 | 0.126 | 0.941 | 3774 | 0.999 | 185.448 | 0.183 | 0.419 | 25.479 | 0.042 | 0.463 | 34.492 | 0.063 | -0.139 | -18.154 | -0.002 |
| EPS Revision | 0.000 | — | 0.000 | 3774 | 0.000 | — | 0.000 | 0.000 | — | 0.000 | 0.000 | — | 0.000 | 0.000 | — | 0.000 |
| Sector-Neutral Dividend Yield | -0.034 | -2.423 | 0.914 | 3774 | 0.905 | 169.021 | 0.166 | -0.028 | -1.734 | -0.003 | 0.280 | 20.950 | 0.038 | -0.178 | -23.392 | -0.002 |
| Cross-Sectional Momentum | -0.007 | -0.429 | 0.879 | 3774 | 0.911 | 148.925 | 0.167 | -0.008 | -0.422 | -0.001 | 0.088 | 5.745 | 0.012 | 0.423 | 48.737 | 0.005 |
| Low Volatility | 0.044 | 2.673 | 0.781 | 3774 | 0.706 | 113.981 | 0.130 | -0.381 | -20.130 | -0.038 | -0.049 | -3.163 | -0.007 | 0.041 | 4.659 | 0.000 |
| ML Ridge (rolling OLS) | -0.005 | -0.195 | 0.776 | 3774 | 0.960 | 105.137 | 0.176 | 0.137 | 4.923 | 0.014 | -0.185 | -8.111 | -0.025 | 0.068 | 5.252 | 0.001 |

## Notes & caveats

- Universe is today's S&P 500 constituents (scraped from Wikipedia). This is
  survivorship-biased — names that were in the index during the backtest but
  are now delisted or removed are absent.
- Market cap is a **current-shares proxy** (today's sharesOutstanding × daily
  Adj Close). It tracks price but does not reflect buybacks or issuance.
- EPS is the trailing-twelve-month scalar from yfinance.info, broadcast as a
  constant panel. **This effectively disables the EPS Revision strategy** —
  its signal is the change in EPS over time, and a constant panel has zero
  change. Run on the Wharton pull to get a working EPS Revision backtest.
- Returns use Adj Close (dividend-reinvested) so total return is captured.