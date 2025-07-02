import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import statistics
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.objective_functions import portfolio_return



# daily returns

def get_daily_returns(prices):
    return prices.pct_change().dropna()

# annualized returns 

def annualized_returns(daily_returns):
    return daily_returns.mean() * 252


# annualized volatility 

def annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)


# annualized Sharpe Ratio 

def sharpe_ratio(daily_returns, risk_free = 0.01):
    daily_risk_free_rate = risk_free / 252
    excess_return = daily_returns - daily_risk_free_rate 
    sratio = np.sqrt(252) * (excess_return.mean() / daily_returns.std(ddof = 1))
    return sratio

# Max Drawdown 

def max_drawdown(price_df):
   peaks = price_df.cummax()
   drawdown = (price_df - peaks) / peaks
   return drawdown.min() 

# closing price plot 

def closing_price_plot(prices):
    plt.figure(figsize=(12, 6))
    tickers = prices.columns.tolist()

    for tick in tickers:
        plt.plot(prices.index, prices[tick], label=tick)

    plt.xlabel("Date")
    plt.ylabel("Adjusted Closing Price")
    plt.title("Adjusted Closing Prices Over Time")
    plt.legend(loc="upper left")
    plt.grid(True)

    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(30))  

    plt.tight_layout()
    plt.show()

# metric summary (returns, volatility, sharpe ratio, max drawdown)

def metric_summary(daily_returns, nprice):
    annual_returns = annualized_returns(daily_returns)
    annual_volatility = annualized_volatility(daily_returns)
    annual_sharpe_ratio = sharpe_ratio(daily_returns)
    mdd = max_drawdown(nprice)

    summary_df = pd.DataFrame({
        
        "Annual Return":annual_returns,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": annual_sharpe_ratio,
        "Max Drawdown": mdd

        })

    summary_df["Annual Return"] = (summary_df["Annual Return"] * 100).map("{:.2f}%".format)
    summary_df["Annual Volatility"] = (summary_df["Annual Volatility"] * 100).map("{:.2f}%".format)
    summary_df["Sharpe Ratio"] = summary_df["Sharpe Ratio"].map("{:.2f}".format)
    summary_df["Max Drawdown"] = (summary_df["Max Drawdown"] * 100).map("{:.2f}%".format)

    #print(summary_df)
    #summary_df.to_csv("etf_summary_metrics.csv")
    return summary_df


# maximizing sharpe ratio to assign weights

def sharpe_max_weights(prices):
    
    # clean data 
    prices = prices.dropna()
    prices = prices[(prices["ARKQ"] > 0) & (prices["BOTZ"] > 0)] #negative price values in these etfs


    # calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # optimize for max sharpe ratio 
    ef = EfficientFrontier(mu, S, solver= "SCS")
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef.save_weights_to_file("weights.csv")

    #print("Optimal Weights:\n", cleaned_weights)
    #ef.portfolio_performance(verbose=True)

    return cleaned_weights 

# performance metrics on sharpe max portfolio 

def evaluate_portfolio_performance(prices):
    
    #load weights
    weights_df = pd.read_csv("weights.csv", header= None)
    weights_df.columns = ['ticker', 'weight']
    weights_df.set_index('ticker', inplace=True)
    weights = weights_df['weight'].astype(float)



    #normalize 
    weights_series = weights / weights.sum()

    #daily returns 
    daily_returns = get_daily_returns(prices)

    #metrics @@
    #portfolio returns 
    daily_portfolio_returns = daily_returns.dot(weights_series)

    #annual returns
    a_returns = annualized_returns(daily_portfolio_returns)

    #annual volatility 
    a_volatility = annualized_volatility(daily_portfolio_returns)

    #sharpe ratio 
    s_ratio = sharpe_ratio(daily_portfolio_returns)

    # max drawdown
    portfolio_prices = prices.dot(weights_series)
    mdd = max_drawdown(portfolio_prices)

    #performance
    performance = {
        "Annual Return": f"{a_returns * 100:.2f}%",
        "Annual Volatility": f"{a_volatility * 100:.2f}%",
        "Sharpe Ratio": f"{s_ratio:.2f}",
        "Max Drawdown": f"{mdd * 100:.2f}%"
    }


    
    return performance, daily_portfolio_returns

def evaluate_equal_weight_benchmark(prices):
    benchmark_prices = prices[["ICLN", "QCLN", "ARKQ", "BOTZ", "ROBO"]]

    #create equal weight assetsd
    weights = np.repeat(1/5, 5)

    #daily returns 
    daily_returns = get_daily_returns(benchmark_prices)

    #portfolio daily returns
    portfolio_returns = daily_returns.dot(weights)

    #metrics @@
    a_returns = annualized_returns(portfolio_returns)
    a_volatility = annualized_volatility(portfolio_returns)
    s_ratio = sharpe_ratio(portfolio_returns)
    mdd = max_drawdown(benchmark_prices.dot(weights))

    #performance 
    benchmark_pf = {
        "Annual Return": f"{a_returns * 100:.2f}%",
        "Annual Volatility": f"{a_volatility * 100:.2f}%",
        "Sharpe Ratio": f"{s_ratio:.2f}",
        "Max Drawdown": f"{mdd * 100:.2f}%"
    }

    return benchmark_pf

#tracking error

def tracking_error(portfolio_returns, benchmark_returns):
    difference = portfolio_returns - benchmark_returns
    return np.std(difference) * np.sqrt(252)

# alpha 

def alpha(portfolio_annual_return, benchmark_annual_return):
    return portfolio_annual_return - benchmark_annual_return



    