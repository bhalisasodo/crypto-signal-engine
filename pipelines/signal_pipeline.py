import torch
from data.ingestion.binance_client import BinanceClient
from data.processing.feature_engineering import create_features
from models.hmm_model import HMMModel
from models.nn_model import PricePredictor
from models.ensemble import generate_signal

class SignalPipeline:
    def __init__(self):
        self.client = BinanceClient()
        self.hmm = HMMModel()
        self.nn = PricePredictor()

    def run(self, symbol="BTCUSDT"):
        df = self.client.get_klines(symbol=symbol)
        df = create_features(df)

        returns = df["returns"].values
        self.hmm.train(returns)
        regime = self.hmm.predict_regime(returns)

        features = df[["returns", "volatility", "ma_fast", "ma_slow"]].values[-5:]
        features = torch.tensor(features.flatten(), dtype=torch.float32)

        prediction = self.nn(features).item()
        signal = generate_signal(regime, prediction)

        return {
            "symbol": symbol,
            "regime": int(regime),
            "prediction": float(prediction),
            "signal": signal
        }
