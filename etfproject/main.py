from data import get_price_data
from dataclasses import dataclass
from 


config = {
    "tickers":["AAPL", "PFE", "JNJ"],
    "start_date":"2022-01-01",
    "end_date":"2024-12-31"
    
    
  }



# fetching price data 

prices = get_price_data(config["tickers"], config["start_date"], config["end_date"])

    
