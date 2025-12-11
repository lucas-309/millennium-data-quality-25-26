from backtester.order_generator import OrderGenerator
import pandas as pd
from typing import List, Dict, Any

class MomentumOrderGenerator(OrderGenerator):
    """
    Momentum strategy implementation that buys when price approaches 52-week high
    and sells when price approaches 52-week low.
    """
    def __init__(self, window_days: int = 125, threshold: float = 0.02):
        """
        Initialize the momentum strategy.
        
        Args:
            window_days: Number of trading days for the lookback window (default 252 for 52 weeks).
            threshold: Percentage threshold (0-1) to define "approaching" (default 0.02 for 2%).
        """
        self.window_days = window_days
        self.threshold = threshold

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate orders based on the momentum strategy.
        
        Args:
            data: DataFrame with tickers as columns and dates as index.
            
        Returns:
            List of order dictionaries.
        """
        orders = []
        tickers = data.columns

        for ticker in tickers:
            ticker_data = data[ticker].to_frame(name='Adj Close')
            # Calculate 52-week high and low, shifted by 1 to avoid lookahead bias
            # The threshold is based on the *previous* 252 days, not including today
            ticker_data['52_week_high'] = ticker_data['Adj Close'].rolling(window=self.window_days).max().shift(1)
            ticker_data['52_week_low'] = ticker_data['Adj Close'].rolling(window=self.window_days).min().shift(1)
            
            in_position = False

            for date, row in ticker_data.iterrows():
                if pd.isna(row['52_week_high']) or pd.isna(row['52_week_low']):
                    continue
                
                price = row['Adj Close']
                high_target = row['52_week_high']
                low_target = row['52_week_low']
                
                # Check if price approaches 52-week high (within threshold)
                # Buy signal - Enter position if not already in one
                if not in_position and price >= high_target * (1 - self.threshold):
                     # BUY 30% of Portfolio Value (0.3)
                     orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": 0.3})
                     in_position = True
                
                # Check if price approaches 52-week low (within threshold)
                # Sell signal - Exit position if currently in one
                elif in_position and price <= low_target * (1 + self.threshold):
                     # SELL 100% of Holdings (1.0)
                     orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": 1.0})
                     in_position = False
                     
        return orders
