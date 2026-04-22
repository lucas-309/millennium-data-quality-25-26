from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt


class Metrics(ABC):
    """Interface for calculating portfolio metrics."""

    @abstractmethod
    def calculate(self, portfolio_values: pd.Series, returns: pd.Series,
                  benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        pass


class ExtendedMetrics(Metrics):
    """Extended metrics calculator with Sharpe, Sortino, Calmar, alpha, beta, and more."""

    def __init__(self, risk_free_rate: float = 0.05, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate(self, portfolio_values: pd.Series, returns: pd.Series,
                  benchmark_returns: Optional[pd.Series] = None,
                  data: Optional[pd.DataFrame] = None,
                  daily_holdings_and_cash: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        metrics = {}
        daily_rf = self.risk_free_rate / self.trading_days

        # --- Return metrics ---
        metrics['Avg Daily Return'] = returns.mean()
        cumulative_return = (1 + returns).prod() - 1
        metrics['Cumulative Return'] = cumulative_return
        metrics['Log Return'] = np.log(1 + returns).mean()

        n_days = len(returns)
        annualized_return = (1 + cumulative_return) ** (self.trading_days / max(n_days, 1)) - 1
        metrics['Annualized Return'] = annualized_return

        # --- Risk metrics ---
        volatility = returns.std() * np.sqrt(self.trading_days)
        metrics['Volatility'] = volatility

        # Sharpe Ratio
        excess_returns = returns - daily_rf
        if excess_returns.std() > 0:
            metrics['Sharpe Ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days)
        else:
            metrics['Sharpe Ratio'] = 0.0

        # Sortino Ratio (downside deviation only)
        downside = excess_returns[excess_returns < 0]
        if len(downside) > 0:
            downside_std = np.sqrt((downside ** 2).mean())
            metrics['Sortino Ratio'] = excess_returns.mean() / downside_std * np.sqrt(self.trading_days) if downside_std > 0 else 0.0
        else:
            metrics['Sortino Ratio'] = float('inf') if excess_returns.mean() > 0 else 0.0

        # Max Drawdown
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values / running_max) - 1
        max_drawdown = drawdown.min()
        metrics['Max Drawdown'] = max_drawdown

        # Calmar Ratio
        if max_drawdown != 0:
            metrics['Calmar Ratio'] = annualized_return / abs(max_drawdown)
        else:
            metrics['Calmar Ratio'] = float('inf') if annualized_return > 0 else 0.0

        # --- Benchmark-relative metrics ---
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
            if len(aligned) > 1:
                aligned.columns = ['strategy', 'benchmark']

                # Beta
                cov = aligned['strategy'].cov(aligned['benchmark'])
                bench_var = aligned['benchmark'].var()
                beta = cov / bench_var if bench_var > 0 else 0.0
                metrics['Beta'] = beta

                # Alpha (Jensen's)
                bench_annual = (1 + aligned['benchmark']).prod() ** (self.trading_days / len(aligned)) - 1
                metrics['Alpha'] = annualized_return - (self.risk_free_rate + beta * (bench_annual - self.risk_free_rate))

                # Information Ratio
                active_returns = aligned['strategy'] - aligned['benchmark']
                tracking_error = active_returns.std() * np.sqrt(self.trading_days)
                if tracking_error > 0:
                    metrics['Information Ratio'] = active_returns.mean() * self.trading_days / (tracking_error)
                else:
                    metrics['Information Ratio'] = 0.0
            else:
                metrics['Beta'] = np.nan
                metrics['Alpha'] = np.nan
                metrics['Information Ratio'] = np.nan
        else:
            metrics['Beta'] = np.nan
            metrics['Alpha'] = np.nan
            metrics['Information Ratio'] = np.nan

        # --- Trade quality metrics ---
        positive_days = returns[returns > 0]
        negative_days = returns[returns < 0]

        metrics['Win Rate'] = len(positive_days) / len(returns) if len(returns) > 0 else 0.0

        neg_sum = abs(negative_days.sum())
        metrics['Profit Factor'] = positive_days.sum() / neg_sum if neg_sum > 0 else float('inf')

        # --- Turnover (vectorized) ---
        if data is not None and daily_holdings_and_cash is not None:
            tradeable_cols = [c for c in data.columns if c in daily_holdings_and_cash.columns]
            if tradeable_cols:
                aligned_holdings = daily_holdings_and_cash.reindex(data.index).ffill()
                holdings_diff = aligned_holdings[tradeable_cols].diff().abs()
                prices = data[tradeable_cols].reindex(aligned_holdings.index)
                traded_values = (holdings_diff * prices).sum(axis=1)

                # Portfolio value per day
                holding_values = (aligned_holdings[tradeable_cols] * prices).sum(axis=1)
                pv = aligned_holdings.get('Cash', 0) + holding_values
                pv = pv.replace(0, np.nan)

                daily_turnover = traded_values / pv
                daily_turnover = daily_turnover.dropna()
                metrics['Avg Daily Turnover'] = daily_turnover.mean()
                metrics['Annualized Turnover'] = daily_turnover.mean() * self.trading_days
            else:
                metrics['Avg Daily Turnover'] = np.nan
                metrics['Annualized Turnover'] = np.nan
        else:
            metrics['Avg Daily Turnover'] = np.nan
            metrics['Annualized Turnover'] = np.nan

        return metrics

    def plot_returns(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                     title: str = "Portfolio Returns", save_path: Optional[str] = None):
        plt.figure(figsize=(12, 8))

        cumulative_returns = (1 + returns).cumprod() - 1
        plt.plot(cumulative_returns.index, cumulative_returns, label=title, linewidth=2, color='#1f77b4')

        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
            cumulative_benchmark = (1 + aligned_benchmark).cumprod() - 1
            plt.plot(cumulative_benchmark.index, cumulative_benchmark, label="S&P 500 (Benchmark)",
                     linestyle='--', linewidth=2, alpha=0.8, color='#ff7f0e')

        plt.title(title, fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        import matplotlib.ticker as mtick
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Plot saved to {save_path}.")
        else:
            plt.show()
