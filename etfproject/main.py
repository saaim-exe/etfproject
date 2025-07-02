import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from data import get_price_data, normalize
from utils import annualized_returns, annualized_volatility, evaluate_equal_weight_benchmark, get_daily_returns, sharpe_ratio, closing_price_plot, metric_summary, sharpe_max_weights, evaluate_portfolio_performance
from dataclasses import dataclass


# fetching price data 
prices = get_price_data('sqig_etf_portfolio_data.csv')

#normalize price data

normalized_price = normalize(prices)

# ETF closing price plot 
closing_price_plot(prices)

# ETF performance metrics (returns, volatility, sharpe ratio, max drawdown)
daily_returns = get_daily_returns(prices)
metric_summary(daily_returns, normalized_price)

# assigning weights based on maximizing the sharpe ratio 

sharpe_max_weights(prices)

# evaluating performance of weighted portfolio 

pf_performance = evaluate_portfolio_performance(prices)
#print(pf_performance)

# evaluating performance of benchmarks 

benchmark_performance = evaluate_equal_weight_benchmark(prices)
#print(benchmark_performance)

performance_comparison = pd.DataFrame({
    
    "Sharpe Max Portfolio": pf_performance,
    "Equal Wegihted Benchmark": benchmark_performance    
    
    })

performance_comparison = performance_comparison.T

performance_comparison.to_csv("portfolio_vs_benchmark.csv")

print("\nPerformance comparison saved to 'portfolio_vs_benchmark.csv'")




