import os

import numpy as np
import torch
from data.ingestion.binance_client import BinanceClient
from data.processing.feature_engineering import create_features
from data.processing.scaler import FeatureScaler
from models.hmm_model import HMMModel
from models.nn_model import PricePredictor
from models.ensemble import generate_signal

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
HMM_PATH = os.path.abspath(os.path.join(MODEL_DIR, "hmm_model.pkl"))
NN_PATH = os.path.abspath(os.path.join(MODEL_DIR, "price_predictor.pth"))
SCALER_PATH = os.path.abspath(os.path.join(MODEL_DIR, "feature_scaler.pkl"))

class SignalPipeline:
    def __init__(self):
        self.client = BinanceClient()
        self.hmm = HMMModel(model_path=HMM_PATH) if os.path.exists(HMM_PATH) else HMMModel()
        self.nn = PricePredictor()
        self.scaler = None

        if os.path.exists(NN_PATH):
            self.nn = PricePredictor.load(NN_PATH)
        if os.path.exists(SCALER_PATH):
            self.scaler = FeatureScaler.load(SCALER_PATH)

    def _prepare_features(self, df):
        features = df[["returns", "volatility", "ma_fast", "ma_slow"]].values[-self.nn.seq_len:]
        if features.shape[0] < self.nn.seq_len:
            padding = self.nn.seq_len - features.shape[0]
            features = np.vstack([
                np.zeros((padding, features.shape[1]), dtype=np.float32),
                features.astype(np.float32)
            ])
        else:
            features = features.astype(np.float32)

        if self.scaler is not None:
            features = self.scaler.transform(features)

        return torch.tensor(features, dtype=torch.float32).view(1, self.nn.seq_len, self.nn.input_size)

    def run(self, symbol="BTCUSDT"):
        df = self.client.get_klines(symbol=symbol)
        df = create_features(df)

        returns = df["returns"].values
        if not os.path.exists(HMM_PATH):
            self.hmm.train(returns)

        regime = self.hmm.predict_regime(returns)

        features = self._prepare_features(df)
        prediction = self.nn(features).item()
        signal = generate_signal(regime, prediction)

        return {
            "symbol": symbol,
            "regime": int(regime),
            "prediction": float(prediction),
            "signal": signal
        }

    def live_run(self, symbol="BTCUSDT"):
        # Fetch recent data for live signal
        df = self.client.get_klines(symbol=symbol, limit=100)
        df = create_features(df)

        returns = df["returns"].values
        if not os.path.exists(HMM_PATH):
            self.hmm.train(returns)

        regime = self.hmm.predict_regime(returns)

        features = self._prepare_features(df)
        self.nn.eval()
        with torch.no_grad():
            prediction = self.nn(features).item()
        signal = generate_signal(int(regime), float(prediction), self.hmm.state_means)

        timestamp_value = df["timestamp"].iloc[-1] if not df.empty else None
        if hasattr(timestamp_value, "item"):
            timestamp_value = timestamp_value.item()

        return {
            "symbol": symbol,
            "regime": int(regime),
            "prediction": float(prediction),
            "signal": signal,
            "timestamp": int(timestamp_value) if timestamp_value is not None else None
        }
