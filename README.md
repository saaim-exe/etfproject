#ETF Portfolio Performance - SQIG Competition Submission 

This project constructs and evaluates a synthetic ETF portfolio using historical price data.
The goal was to optimize portfolio weights for maximum Sharpe ratio and compare the results to an equal-weight benchmark.

**Files**
- main.py - runs the full analysis
- utils.py - Helper functions (e.g. returns, risk metrics, plots)
- sqig_etf_portfolio_data.csv - Historical ETF prices
- weights.csv - Optimized weights (Sharpe-maximizing)
- portfolio_vs_benchmark.csv - Performance comparison
- closing_prices.png - Price trend plot

**Metrics Evaluated**

- Annual Returns
- Volatility
- Sharpe Ratio
- Max Drawdown
- Alpha (vs benchmark)
- Tracking Error

**Notes**
- The optimizer heavily favors ICLN and QCLN, which outperformed historically. Alpha is high (254%) due to backtested hindsight - results should not be interpreted as predictive.

**Run**
```bash
python main.py
