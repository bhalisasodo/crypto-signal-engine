import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame):
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["returns"].rolling(10).std()
    df["ma_fast"] = df["close"].rolling(5).mean()
    df["ma_slow"] = df["close"].rolling(20).mean()

    return df.dropna()
