"""Template for new trading strategies. Copy this and make your own strategy!"""
import pandas as pd
from typing import List, Dict, Any

from strategies.order_generator import OrderGenerator

class TemplateStrategy(OrderGenerator):
    """Your strategy description here."""
    
    def __init__(self, window=20):
        self.window = window
    
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Args:
            data: DataFrame with tickers as columns, dates as index
            
        Returns:
            List of dicts: {'date': timestamp, 'ticker': str, 'type': 'BUY'|'SELL', 'quantity': int|float}
        """
        orders = []
        
        # Your logic here
        
        return orders