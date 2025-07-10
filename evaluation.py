# src/evaluation.py
# Performance metrics, evaluation plots, SHAP analysis.

import numpy as np
import matplotlib.pyplot as plt

def evaluate_performance(portfolio_returns):
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max).min()
    return {'Sharpe': sharpe, 'Max Drawdown': drawdown}

def plot_performance(portfolio_returns):
    cumulative = (1 + portfolio_returns).cumprod()
    plt.figure(figsize=(12,6))
    plt.plot(cumulative)
    plt.title("Cumulative Portfolio Returns")
    plt.xlabel("Time")
    plt.ylabel("Growth of $1")
    plt.grid(True)
    plt.show()

def plot_shap_values(model, features):
    import shap
    explainer = shap.Explainer(model)
    shap_values = explainer(features)
    shap.summary_plot(shap_values, features)
