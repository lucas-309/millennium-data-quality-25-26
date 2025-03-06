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
    
    def calculate(self, portfolio_values: pd.Series, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        metrics = {}
        
        metrics['Daily Return'] = returns.mean()
        metrics['Cumulative Return'] = (1 + returns).prod() - 1
        metrics['Log Return'] = np.log(1 + returns).mean()

        # volatility is the standard deviation of returns
        metrics['Volatility'] = returns.std() * np.sqrt(252)  # annualize volatility, 252 trading days in a yr

        # TODO: this part is buggy, need to fix
        # if benchmark_returns is not None:
        #     metrics['Information Coefficient'] = returns.corr(benchmark_returns)
        # else:
        #     metrics['Information Coefficient'] = None

        risk_free_rate = 0.0045  
        excess_returns = returns - (risk_free_rate / 252)
        # sharpe ratio is the excess return over the risk free rate divided by the volatility
        metrics['Sharpe Ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values / running_max) - 1
        # max drawdown is the max loss from a peak to a trough in the portfolio value
        metrics['Max Drawdown'] = drawdown.min()

        # val at risk (VaR) 1 day horizon. 5% quantile. this is the max loss we can expect with 95% confidence
        metrics['VaR 5%'] = returns.quantile(0.05)
        return metrics

    def plot_returns(self, returns: pd.Series, title: str = "Portfolio Returns"):
        plt.figure(figsize=(10, 6))
        returns.cumsum().plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.grid(True)
        plt.show()