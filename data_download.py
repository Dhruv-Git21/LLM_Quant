# data_download.py
# Script to download all main data types for the project.

import os
import pandas as pd
from src.data_utils import save_pickle

os.makedirs("data/raw/", exist_ok=True)

# S&P 500 tickers
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
tickers = [t for t in tickers if '-' not in t][:20]  # Short for demo

# Download OHLCV
import yfinance as yf
data = yf.download(tickers, start="2015-01-01", end="2024-06-01", group_by='ticker', threads=True)
save_pickle(data, 'data/raw/ohlcv.pkl')

# Download Fama-French daily factors
import pandas_datareader.data as web
ff_factors = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench')[0]
save_pickle(ff_factors, 'data/raw/ff_factors.pkl')

# Download sample news
news = pd.DataFrame({
    'date': ['2024-05-01', '2024-05-01'],
    'ticker': ['AAPL', 'MSFT'],
    'headline': [
        'Apple reports record quarterly earnings despite supply chain challenges.',
        'Microsoft launches new AI services, driving bullish investor sentiment.'
    ]
})
news.to_csv('data/raw/news.csv', index=False)

print("All main datasets downloaded to data/raw/")
