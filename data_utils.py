# src/data_utils.py
# Utilities for loading, cleaning, merging, and saving DataFrames.

import pandas as pd
import os

def load_pickle(path):
    return pd.read_pickle(path)

def save_pickle(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)

def load_csv(path):
    return pd.read_csv(path)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def merge_features(*dfs, on=None):
    """Merge multiple DataFrames on a shared index or column."""
    from functools import reduce
    if on is None:
        merged = reduce(lambda left, right: left.join(right, how='outer'), dfs)
    else:
        merged = reduce(lambda left, right: pd.merge(left, right, on=on, how='outer'), dfs)
    return merged
