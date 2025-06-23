import yfinance as yf
import pandas as pd 

config = {
    "tickers":["AAPL"],
    "start_date":"2022-01-01",
    "end_date":"2024-12-31"
    
    
  }


def get_price_data(tickers, start_date, end_date):
    
    if len(tickers) == 1:
        tickers = tickers[0]

    

    data = yf.download(tickers, start = start_date, end = end_date)
    data = data["Close"]

    print(data)
    return data


def normalize_



