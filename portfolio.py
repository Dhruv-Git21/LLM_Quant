# src/portfolio.py
# Portfolio construction and backtest logic for long-short strategy.

import numpy as np
import pandas as pd

def run_long_short_strategy(preds, returns, quantile=0.1, rebalance_freq=5):
    """
    preds: DataFrame, columns = tickers, rows = dates
    returns: DataFrame, columns = tickers, rows = dates
    """
    portfolio_returns = []
    dates = returns.index
    n = int(len(preds.columns) * quantile)
    
    for i in range(0, len(dates)-rebalance_freq, rebalance_freq):
        pred_slice = preds.iloc[i]
        ret_slice = returns.iloc[i+1:i+1+rebalance_freq]
        top = pred_slice.nlargest(n).index
        bottom = pred_slice.nsmallest(n).index
        pnl = ret_slice[top].mean(axis=1) - ret_slice[bottom].mean(axis=1)
        portfolio_returns.extend(pnl.values)
    return pd.Series(portfolio_returns, index=dates[:len(portfolio_returns)])
