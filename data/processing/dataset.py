import numpy as np
import pandas as pd


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns=None,
    target_column="returns",
    seq_len=20,
):
    if feature_columns is None:
        feature_columns = ["returns", "volatility", "ma_fast", "ma_slow"]

    df = df.copy()
    df[target_column] = df[target_column].shift(-1)
    df = df.dropna()

    values = df[feature_columns].values.astype(np.float32)
    targets = df[target_column].values.astype(np.float32).reshape(-1, 1)

    X = []
    y = []
    for i in range(len(values) - seq_len + 1):
        X.append(values[i : i + seq_len])
        y.append(targets[i + seq_len - 1])

    if len(X) == 0:
        return np.empty((0, seq_len, len(feature_columns)), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    return np.stack(X), np.stack(y)


def train_test_split_sequences(X, y, test_size=0.2):
    if not 0.0 <= test_size < 1.0:
        raise ValueError("test_size must be in [0.0, 1.0)")

    split = int(len(X) * (1 - test_size))
    if split < 1:
        raise ValueError("Not enough data to split into train/test sets.")

    return X[:split], X[split:], y[:split], y[split:]
