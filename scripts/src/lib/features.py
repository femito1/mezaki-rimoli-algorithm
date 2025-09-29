import pandas as pd
from dataclasses import dataclass
import numpy as np

def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()

def vol_ann(logret: pd.Series, window: int) -> pd.Series:
    return logret.rolling(window).std(ddof=0) * np.sqrt(252)

def rsi(close: pd.Series, window: int) -> pd.Series:
    
    r = close.diff()

    avg_gains = r.clip(lower=0).ewm(
        alpha=1/window, 
        min_periods=window, 
        adjust=False).mean()

    eps = 10e-6

    avg_losses = -r.clip(upper=0).ewm(
        alpha=1/window, 
        min_periods=window, 
        adjust=False).mean()

    rs = avg_gains / (avg_losses + eps)
    return 100 - 100 / (1 + rs)

def build_features(datafeed: dataclass, symbol: str, cfg: dict, small_sma_window: int, big_sma_window: int, vol_window: int, rsi_window: int) -> pd.DataFrame:
    df = datafeed.history()
    df_sym = (df[df["symbol"] == symbol]
              .sort_values("datetime")
              .set_index("datetime")
              .copy())
    close = df_sym["close"].astype(float)
    logret = np.log(close).diff()
    n, k, j, w = small_sma_window, big_sma_window, vol_window, rsi_window
    df_sym[f"sma_{n}"] = sma(close, n)
    df_sym[f"sma_{k}"] = sma(close, k)
    df_sym[f"vol_{j}"] = vol_ann(logret, j)
    df_sym[f"rsi_{w}"] = rsi(close, w)
    return df_sym[[f"sma_{n}", f"sma_{k}", f"vol_{j}", f"rsi_{w}"]].dropna()

