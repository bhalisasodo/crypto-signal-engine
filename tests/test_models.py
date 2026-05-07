import os
import tempfile
import numpy as np
from models.hmm_model import HMMModel
from models.ensemble import generate_signal


def test_hmm_model_fit_predict_save_load():
    returns = np.linspace(-0.01, 0.01, 100)
    hmm = HMMModel(n_states=3, n_iter=50)
    hmm.train(returns)

    regime = hmm.predict_regime(returns)
    assert 0 <= regime < 3
    assert isinstance(hmm.state_means, np.ndarray)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "hmm.pkl")
        hmm.save(path)
        loaded = HMMModel(model_path=path)

        assert loaded.n_states == hmm.n_states
        assert np.allclose(loaded.state_means, hmm.state_means)
        assert loaded.predict_regime(returns) == regime


def test_generate_signal_returns_expected_values():
    assert generate_signal(0, 0.5, state_means=[0.2, -0.1], threshold=0.0) == "BUY"
    assert generate_signal(1, -0.5, state_means=[0.2, -0.1], threshold=0.0) == "SELL"
    assert generate_signal(0, 0.0, state_means=[0.2, -0.1], threshold=0.0) == "HOLD"
