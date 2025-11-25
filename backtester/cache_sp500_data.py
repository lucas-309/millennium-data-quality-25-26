import pandas as pd
import yfinance as yf
import pickle
import requests
from io import StringIO

def fetch_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    sp500_table = pd.read_html(StringIO(response.text))[0]
    tickers = sp500_table['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    if 'SPY' not in tickers:
        tickers.append('SPY')
    return tickers

def download_data(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False,
        threads=True # this enables parallel downloading
    )
    return data

def calculate_vwap(data):
    vwap_data = {}
    for ticker in data.columns.levels[0]:
        adj_close = data[ticker]['Adj Close']
        volume = data[ticker]['Volume']
        vwap = (adj_close * volume).cumsum() / volume.cumsum()
        ticker_df = pd.DataFrame({
            'Adj Close': adj_close,
            'Volume': volume,
            'VWAP': vwap
        })
        vwap_data[ticker] = ticker_df
    return vwap_data

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    tickers = fetch_sp500_tickers()
    start_date = '2010-01-01'
    end_date = '2024-11-20'
    data = download_data(tickers, start_date, end_date)
    vwap_data = calculate_vwap(data)
    save_data(vwap_data, 'sp500_data.pkl')
    print("Data has been cached and saved to sp500_data.pkl")

if __name__ == '__main__':
    main()