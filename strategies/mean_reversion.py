import pandas as pd
from typing import List, Dict, Any

from .order_generator import OrderGenerator

class MeanReversionOrderGenerator(OrderGenerator):
    """Mean reversion strategy implementation with 100-day rolling window."""
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []
        tickers = data.columns

        for ticker in tickers:
            ticker_data = data[ticker].to_frame(name='Adj Close')
            ticker_data['100_day_avg'] = ticker_data['Adj Close'].rolling(window=100).mean()

            for date, row in ticker_data.iterrows():
                if pd.isna(row['100_day_avg']):
                    continue
                if row['Adj Close'] < row['100_day_avg']:
                    orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": 100})
                else:
                    orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": 100})

        return orders