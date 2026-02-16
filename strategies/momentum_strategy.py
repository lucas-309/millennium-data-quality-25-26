from .order_generator import OrderGenerator
import pandas as pd
from typing import List, Dict, Any

class MomentumOrderGenerator(OrderGenerator):
    def __init__(self, window_days: int = 125, threshold: float = 0.02):
        self.window_days = window_days
        self.threshold = threshold

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []

        # Use columns from data (or keep your own list)
        tickers = list(data.columns)   # assuming columns are like "AAPL", "NVDA"

        # Track position state per ticker
        in_position: Dict[str, bool] = {ticker: False for ticker in tickers}

        for ticker in tickers: # parallelize tickers (bad when large)

            ticker_data = data[ticker].to_frame(name='Adj Close')
            ticker_data['52_week_high'] = (
                ticker_data['Adj Close']
                .rolling(window=self.window_days)
                .max()
                .shift(1)
            )
            ticker_data['52_week_low'] = (
                ticker_data['Adj Close']
                .rolling(window=self.window_days)
                .min()
                .shift(1)
            )

            for date, row in ticker_data.iterrows():
                if pd.isna(row['52_week_high']) or pd.isna(row['52_week_low']):
                    continue

                price = row['Adj Close']
                high_target = row['52_week_high']
                low_target = row['52_week_low']

                # BUY: near prior-window high
                if (not in_position[ticker]) and price >= high_target * (1 - self.threshold):
                    orders.append({
                        "date": date,
                        "type": "BUY",
                        "ticker": ticker,
                        "quantity": 0.3,  # 30% of portfolio
                    })
                    in_position[ticker] = True

                # SELL: near prior-window low
                elif in_position[ticker] and price <= low_target * (1 + self.threshold):
                    orders.append({
                        "date": date,
                        "type": "SELL",
                        "ticker": ticker,
                        "quantity": 1.0,  # 100% of holdings
                    })
                    in_position[ticker] = False

        orders.sort(key=lambda o: (o["date"], o["ticker"], o["type"]))

        return orders
