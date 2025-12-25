from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

class Metrics(ABC):
    """Interface for calculating portfolio metrics."""
    
    @abstractmethod
    def calculate(self, portfolio_values: pd.Series, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate performance metrics from portfolio values and returns."""
        pass


class ExtendedMetrics(Metrics):
    """Extended metrics calculator implementation."""
    
    def calculate(self, portfolio_values: pd.Series, returns: pd.Series, benchmark_returns: pd.Series = None, data: pd.DataFrame = None, daily_holdings_and_cash: pd.DataFrame = None) -> Dict[str, float]:
        metrics = {}
        
        # Mean Returns (already captured by Daily Return)
        metrics['Daily Return'] = returns.mean()
        metrics['Cumulative Return'] = (1 + returns).prod() - 1
        metrics['Log Return'] = np.log(1 + returns).mean()

        # Volatility
        metrics['Volatility'] = returns.std() * np.sqrt(252)  # annualize volatility, 252 trading days in a yr

        risk_free_rate = 0.0045  
        excess_returns = returns - (risk_free_rate / 252)
        metrics['Sharpe Ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values / running_max) - 1
        metrics['Max Drawdown'] = drawdown.min()

        

        # Turnover calculation
        if data is not None and daily_holdings_and_cash is not None:
            daily_turnover_list = []
            # Ensure data and daily_holdings_and_cash are aligned by index
            aligned_holdings_cash = daily_holdings_and_cash.reindex(data.index).ffill()
            
            for i in range(1, len(aligned_holdings_cash)):
                current_date = aligned_holdings_cash.index[i]
                previous_date = aligned_holdings_cash.index[i-1]
                
                # Get holdings and cash for current and previous day
                current_day_data = aligned_holdings_cash.loc[current_date]
                previous_day_data = aligned_holdings_cash.loc[previous_date]
                
                # Get prices for the current day for all relevant tickers
                # Exclude 'Cash' from tickers, as it's not a tradable asset price
                tradeable_tickers = [col for col in data.columns if col in current_day_data.index]
                current_prices = data.loc[current_date, tradeable_tickers]
                
                total_traded_value_today = 0.0
                for ticker in tradeable_tickers:
                    current_holding_qty = current_day_data.get(ticker, 0)
                    previous_holding_qty = previous_day_data.get(ticker, 0)
                    
                    # Absolute change in holdings, multiplied by current price
                    traded_qty = abs(current_holding_qty - previous_holding_qty)
                    total_traded_value_today += traded_qty * current_prices[ticker]
                
                # Calculate previous day's total portfolio value (cash + holdings value)
                previous_portfolio_value = previous_day_data.get('Cash', 0.0)
                for ticker in tradeable_tickers:
                    previous_portfolio_value += previous_day_data.get(ticker, 0) * data.loc[previous_date, ticker]

                if previous_portfolio_value > 0:
                    daily_turnover_list.append(total_traded_value_today / previous_portfolio_value)
                else:
                    daily_turnover_list.append(0.0) # No portfolio value, no turnover

            if daily_turnover_list:
                metrics['Daily Turnover'] = np.array(daily_turnover_list).mean() * 252 # Annualize average daily turnover
                metrics['Average Turnover'] = np.array(daily_turnover_list).mean()
            else:
                metrics['Daily Turnover'] = np.nan
                metrics['Average Turnover'] = np.nan
        else:
            metrics['Daily Turnover'] = np.nan
            metrics['Average Turnover'] = np.nan

        return metrics

    def plot_returns(self, returns: pd.Series, benchmark_returns: pd.Series = None, title: str = "Portfolio Returns", save_path: str = None):
        plt.figure(figsize=(12, 8))
        
        # Calculate Cumulative Returns (Geometric)
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # Plot Strategy
        plt.plot(cumulative_returns.index, cumulative_returns, label="Momentum Strategy", linewidth=2, color='#1f77b4') # Blue
        
        if benchmark_returns is not None:
            # Align benchmark to portfolio dates and fill missing with 0 returns
            aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
            cumulative_benchmark = (1 + aligned_benchmark).cumprod() - 1
            # Plot Benchmark
            plt.plot(cumulative_benchmark.index, cumulative_benchmark, label="S&P 500 (Benchmark)", linestyle='--', linewidth=2, alpha=0.8, color='#ff7f0e') # Orange

        plt.title(title, fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Format y-axis as percentage
        import matplotlib.ticker as mtick
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Plot saved to {save_path}.")
        else:
            plt.show()