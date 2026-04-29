import pandas as pd
from typing import List, Dict, Any
from .order_generator import OrderGenerator

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    # calculates price changes
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss # relative strength (RS)
    return 100 - (100 / (1 + rs)) # RSI

class MovingAverageOrderGenerator(OrderGenerator):
    """"""
    def __init__(self, short_window=50, long_window=200, buffer=0.01, cooldown=14):
        self.short_window = short_window
        self.long_window = long_window
        self.buffer = buffer # 1% confirmation buffer

    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []
        tickers = data.columns
        held = {ticker: False for ticker in tickers}
        peak_price = {ticker: 0.0 for ticker in tickers}

        for ticker in tickers:
            ticker_data = data[ticker].to_frame(name='Price')
            # EMA instead of uniform avg. for faster reaction
            ticker_data['short_avg'] = ticker_data['Price'].ewm(span=self.short_window).mean()
            ticker_data['long_avg'] = ticker_data['Price'].ewm(span=self.long_window).mean()
            ticker_data['rsi'] = calculate_rsi(ticker_data['Price'], window=14)

            for date, row in ticker_data.iterrows():
                price = row['Price']
                is_bull_trend = row['short_avg'] > row['long_avg']

                # BUY when cross-over OR bullish pullback (RSI < 40)
                if not held[ticker]:
                    ma_cross_up = is_bull_trend and row['short_avg'] > row['long_avg'] * (1 + self.buffer)
                    rsi_pullback = is_bull_trend and row['rsi'] < 40
                    
                    if ma_cross_up or rsi_pullback:
                        orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": 0.5}) # Increased size
                        held[ticker] = True
                        peak_price[ticker] = price

                elif held[ticker]:
                    peak_price[ticker] = max(peak_price[ticker], price)
                    # 10% trailing stop
                    trailing_stop_hit = price < peak_price[ticker] * 0.90
                    ma_cross_down = row['short_avg'] < row['long_avg']

                    # SELL when moving avg. crosses OR (trailing stop is hit AND price < short_avg)
                    if ma_cross_down or (trailing_stop_hit and price < row['short_avg']):
                        orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": 1.0})
                        held[ticker] = False

        return orders