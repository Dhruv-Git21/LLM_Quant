# src/tech_fund_features.py
# Functions for computing technical and fundamental features.

import pandas as pd
import ta

def compute_technical_indicators(df):
    """Expects OHLCV DataFrame with columns: 'Open','High','Low','Close','Volume'"""
    out = pd.DataFrame(index=df.index)
    out['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    out['macd'] = macd.macd()
    out['macd_signal'] = macd.macd_signal()
    out['bb_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    out['bb_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    out['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    out['roc'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
    return out

def compute_fundamental_ratios(fund_df):
    # Example: pass a DataFrame of fundamental data (columns: netIncome, totalAssets, etc)
    # Returns ratios as new DataFrame
    out = pd.DataFrame(index=fund_df.index)
    if 'Net Income' in fund_df.columns and 'Total Assets' in fund_df.columns:
        out['roe'] = fund_df['Net Income'] / fund_df['Total Assets']
    if 'EBITDA' in fund_df.columns and 'Total Revenue' in fund_df.columns:
        out['ebitda_margin'] = fund_df['EBITDA'] / fund_df['Total Revenue']
    return out
