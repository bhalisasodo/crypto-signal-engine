from hmmlearn.hmm import GaussianHMM
import numpy as np

class HMMModel:
    def __init__(self, n_states=3):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)

    def train(self, returns):
        returns = returns.reshape(-1, 1)
        self.model.fit(returns)

    def predict_regime(self, returns):
        returns = returns.reshape(-1, 1)
        hidden_states = self.model.predict(returns)
        return hidden_states[-1]
