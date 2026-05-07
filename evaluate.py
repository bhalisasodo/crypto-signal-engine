import argparse
import os

import numpy as np
import pandas as pd
import torch

from data.ingestion.binance_client import BinanceClient
from data.processing.dataset import build_sequence_dataset
from data.processing.feature_engineering import create_features
from data.processing.scaler import FeatureScaler
from models.ensemble import generate_signal
from models.hmm_model import HMMModel
from models.nn_model import PricePredictor
import metrics

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
HMM_PATH = os.path.join(MODEL_DIR, "hmm_model.pkl")
NN_PATH = os.path.join(MODEL_DIR, "price_predictor.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")


def load_data(args):
    if args.data_path:
        df = pd.read_csv(args.data_path)
        if "close" not in df.columns:
            raise ValueError("Data file must contain a 'close' column.")
        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()
        return df.dropna().reset_index(drop=True)

    client = BinanceClient()
    return client.get_klines(symbol=args.symbol, interval=args.interval, limit=args.limit)


def compute_strategy_returns(signals, next_returns):
    strategy_returns = np.zeros_like(next_returns)
    for i, signal in enumerate(signals):
        if signal == "BUY":
            strategy_returns[i] = next_returns[i]
        elif signal == "SELL":
            strategy_returns[i] = -next_returns[i]
        else:
            strategy_returns[i] = 0.0
    return strategy_returns


def evaluate(predictions, targets):
    mse = float(np.mean((predictions - targets) ** 2))
    directional = float((np.sign(predictions.flatten()) == np.sign(targets.flatten())).mean())
    return mse, directional


def main(args):
    df = load_data(args)
    df = create_features(df)

    if not os.path.exists(HMM_PATH) or not os.path.exists(NN_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Saved models or scaler not found. Run train.py first.")

    scaler = FeatureScaler.load(SCALER_PATH)
    nn = PricePredictor.load(NN_PATH, device=args.device)
    hmm = HMMModel(model_path=HMM_PATH)

    X, y = build_sequence_dataset(df, seq_len=args.seq_len)
    if X.shape[0] == 0:
        raise ValueError("Not enough data to build a sequence dataset. Increase limit or reduce seq_len.")

    X_scaled = scaler.transform(X)
    X_tensor = np.array(X_scaled, dtype=np.float32)
    nn.eval()
    with np.errstate(all='ignore'):
        preds = []
        for batch in np.array_split(X_tensor, max(1, len(X_tensor) // args.batch_size)):
            batch_tensor = np.array(batch, dtype=np.float32)
            preds.append(nn(torch.from_numpy(batch_tensor)).detach().cpu().numpy())
        preds = np.vstack(preds)

    mse, directional = evaluate(preds, y)
    print(f"Model evaluation: MSE={mse:.6f}, direction_acc={directional:.4f}")

    signals = []
    for i, pred in enumerate(preds):
        history_returns = df["returns"].values[: i + args.seq_len]
        regime = hmm.predict_regime(history_returns)
        signal = generate_signal(int(regime), float(pred[0]), hmm.state_means, threshold=args.threshold)
        signals.append(signal)
    next_returns = df["returns"].values[args.seq_len: args.seq_len + len(signals)]
    strategy_returns = compute_strategy_returns(signals, next_returns)

    buy_hold = np.nansum(next_returns)
    strategy = np.nansum(strategy_returns)
    print(f"Strategy cumulative return: {strategy:.6f}, buy-hold cumulative return: {buy_hold:.6f}")
    print(f"Signal counts: BUY={signals.count('BUY')}, SELL={signals.count('SELL')}, HOLD={signals.count('HOLD')}")

    # Save metrics for dashboard
    metrics_data = {
        "training_status": "Trained and evaluated",
        "mse": mse,
        "directional_accuracy": float(directional * 100),  # Convert to percentage
        "cumulative_return": float(strategy * 100),  # Convert to percentage
        "buy_hold_return": float(buy_hold * 100),  # Convert to percentage
        "signal_counts": {
            "BUY": signals.count("BUY"),
            "SELL": signals.count("SELL"),
            "HOLD": signals.count("HOLD")
        }
    }
    metrics.save_metrics(metrics_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained crypto signal models.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol to evaluate")
    parser.add_argument("--interval", default="1h", help="Kline interval")
    parser.add_argument("--limit", type=int, default=1000, help="Historical sample size")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length for the NN")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--data-path", default=None, help="Optional CSV file path for offline evaluation data")
    parser.add_argument("--threshold", type=float, default=0.0, help="Prediction threshold for signal generation")
    parser.add_argument("--device", default="cpu", help="Device for model evaluation")
    args = parser.parse_args()
    main(args)
