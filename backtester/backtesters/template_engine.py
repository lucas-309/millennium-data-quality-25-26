"""Template for new backtest engines. Copy this and make your own engine!"""

import pandas as pd
from typing import List, Dict, Any

from .backtest_engine import BacktestEngine


class TemplateEngine(BacktestEngine):
    """Your backtest engine description here."""
    
    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Args:
            orders: List of order dicts from OrderGenerator
            data: Price data DataFrame
            
        Returns:
            Dict with at least: {'portfolio_values': DataFrame with 'Portfolio Value' column}
        """
        cash = self.initial_cash
        holdings = {}
        portfolio_values = []
        
        # Your execution logic here
        
        portfolio_values_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index("Date")
        return {"portfolio_values": portfolio_values_df}