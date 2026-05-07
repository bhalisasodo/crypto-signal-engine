import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM

class HMMModel:
    def __init__(self, n_states=4, covariance_type="full", n_iter=200, model_path=None):
        self.n_states = n_states
        self.model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter)
        self.state_means = np.zeros(n_states, dtype=float)
        if model_path is not None and hasattr(model_path, "__len__") and model_path:
            try:
                self.load(model_path)
            except Exception:
                pass

    def _prepare_input(self, returns):
        returns = np.asarray(returns, dtype=float).reshape(-1, 1)
        return returns

    def fit(self, returns):
        returns = self._prepare_input(returns)
        self.model.fit(returns)

    def train(self, returns):
        returns = np.asarray(returns, dtype=float)
        self.fit(returns)
        hidden_states = self.model.predict(self._prepare_input(returns))
        state_means = []
        for state in range(self.n_states):
            mask = hidden_states == state
            if mask.any():
                state_means.append(returns[mask].mean())
            else:
                state_means.append(0.0)
        self.state_means = np.array(state_means, dtype=float)

    def predict_regime(self, returns):
        returns = self._prepare_input(returns)
        hidden_states = self.model.predict(returns)
        return hidden_states[-1]

    def predict_proba(self, returns):
        returns = self._prepare_input(returns)
        return self.model.predict_proba(returns)

    def save(self, path):
        joblib.dump({
            "n_states": self.n_states,
            "model": self.model,
            "state_means": self.state_means,
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.n_states = data.get("n_states", self.n_states)
        self.model = data["model"]
        self.state_means = np.asarray(data.get("state_means", np.zeros(self.n_states, dtype=float)), dtype=float)
        return self

    def state_direction(self, state, threshold=0.0):
        if self.state_means[state] > threshold:
            return 1
        if self.state_means[state] < -threshold:
            return -1
        return 0
