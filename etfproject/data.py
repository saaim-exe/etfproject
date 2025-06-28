import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 



def get_price_data(file_path):

    # read data from csv
    with open(file_path , "r") as file: 
        data = pd.read_csv(file)

    # set date as index, drop date column 
    data = data.set_index(pd.DatetimeIndex(data["Unnamed: 0"].values))
    data = data.drop(columns =["Unnamed: 0"])
    data = data.dropna()


    return data
                                    
def normalize(prices):
    
    return prices / prices.iloc[0]




