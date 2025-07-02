import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.objective_functions import portfolio_return 
from data import get_price_data, normalize
from utils import annualized_returns, annualized_volatility, evaluate_equal_weight_benchmark, get_daily_returns, sharpe_ratio, closing_price_plot, metric_summary, sharpe_max_weights, evaluate_portfolio_performance, tracking_error, alpha
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

#benchmark returns 
benchmark_prices = prices[["ICLN", "QCLN", "ARKQ", "BOTZ", "ROBO"]]
benchmark_returns = get_daily_returns(benchmark_prices).dot(np.repeat(1/5, 5))

#evaluate optimized portfolio (includes daily returns)
pf_performance, portfolio_returns = evaluate_portfolio_performance(prices)

#calculate alpha and tracking error 
te = tracking_error(portfolio_returns, benchmark_returns)
portfolio_annual = annualized_returns(portfolio_returns)
benchmark_annual = annualized_returns(benchmark_returns)
a = alpha(portfolio_annual, benchmark_annual)

print(f"Tracking Error: {te:.4f}")
print(f"Alpha (Annual Excess Return): {a:.4%}")

#evaluate benchmark
benchmark_performance = evaluate_equal_weight_benchmark(prices)

#saving portfolio comparison to csv
performance_comparison = pd.DataFrame({
    
    "Sharpe Max Portfolio": pf_performance,
    "Equal Weighted Benchmark": benchmark_performance    
    
    })

performance_comparison.loc["Alpha vs Benchmark"] = [f"{a:.4%}", ""]
performance_comparison.loc["Tracking Error"] = [f"{te:.4f}", ""]
performance_comparison = performance_comparison.T
performance_comparison.to_csv("portfolio_vs_benchmark.csv")

print("\nPerformance comparison saved to 'portfolio_vs_benchmark.csv'")




