import pandas as pd
from typing import List, Dict, Any

from .order_generator import OrderGenerator

class MeanReversionOrderGenerator(OrderGenerator):
    """
    Mean reversion strategy implementation with 100-day rolling window.
    Trades on 2 z-score deviation
    """
    
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []
        tickers = data.columns

        for ticker in tickers:
            ticker_data = data[ticker].to_frame(name='Adj Close')
            
            # rolling data
            ticker_data['100_day_avg'] = ticker_data['Adj Close'].rolling(window=100).mean()
            ticker_data['100_day_std'] = ticker_data['Adj Close'].rolling(window=100).std()
            ticker_data['z_score'] = (ticker_data['Adj Close'] - ticker_data['100_day_avg']) / ticker_data['100_day_std']

            for date, row in ticker_data.iterrows():
                if pd.isna(row['z_score']):
                    continue

                if row['z_score'] < -2:  # Extremely oversold
                    orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": 100})
                elif row['z_score'] > -2:   # Extremely overbought
                    orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": 100})
                # otherwise, no order

        return orders