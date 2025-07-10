# src/residual_model.py
# Factor regression and ML model for residual return forecasting.

import xgboost as xgb
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split

def compute_residuals(returns, factors):
    """Regress returns on factors, return residuals."""
    X = sm.add_constant(factors)
    model = sm.OLS(returns, X, missing='drop').fit()
    residuals = returns - model.predict(X)
    return residuals

def train_residual_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, y_test
