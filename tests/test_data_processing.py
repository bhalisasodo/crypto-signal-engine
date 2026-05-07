import numpy as np
import pandas as pd
from data.processing.dataset import build_sequence_dataset, train_test_split_sequences
from data.processing.feature_engineering import create_features


def test_create_features_adds_expected_columns():
    data = {
        "close": np.arange(1.0, 31.0, dtype=np.float32)
    }
    df = pd.DataFrame(data)
    df["returns"] = df["close"].pct_change()
    transformed = create_features(df)

    assert "volatility" in transformed.columns
    assert "ma_fast" in transformed.columns
    assert "ma_slow" in transformed.columns
    assert transformed["returns"].notna().all()
    assert len(transformed) > 0


def test_build_sequence_dataset_shapes():
    values = np.arange(1.0, 31.0, dtype=np.float32)
    df = pd.DataFrame({
        "close": values,
        "returns": np.concatenate([[0.0], values[1:] / values[:-1] - 1]),
        "volatility": np.ones_like(values),
        "ma_fast": values,
        "ma_slow": values,
    })
    X, y = build_sequence_dataset(df, seq_len=5)

    assert X.ndim == 3
    assert X.shape[1:] == (5, 4)
    assert y.shape[0] == X.shape[0]


def test_train_test_split_sequences_valid_split():
    X = np.zeros((10, 5, 4), dtype=np.float32)
    y = np.zeros((10, 1), dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, test_size=0.3)

    assert X_train.shape[0] == 7
    assert X_test.shape[0] == 3
    assert y_train.shape[0] == 7
    assert y_test.shape[0] == 3


def test_train_test_split_sequences_invalid_size_raises():
    X = np.zeros((3, 5, 4), dtype=np.float32)
    y = np.zeros((3, 1), dtype=np.float32)
    try:
        train_test_split_sequences(X, y, test_size=1.0)
    except ValueError as exc:
        assert "test_size must be in [0.0, 1.0)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid test_size")
