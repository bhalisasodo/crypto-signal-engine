import joblib
import numpy as np

class FeatureScaler:
    def __init__(self, mean=None, std=None):
        self.mean = np.asarray(mean, dtype=np.float32) if mean is not None else None
        self.std = np.asarray(std, dtype=np.float32) if std is not None else None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean = X.mean(axis=(0, 1))
        self.std = X.std(axis=(0, 1))
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean) / self.std

    def save(self, path):
        joblib.dump({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        return cls(mean=data["mean"], std=data["std"])
