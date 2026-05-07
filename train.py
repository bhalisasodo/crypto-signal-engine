import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.ingestion.binance_client import BinanceClient
from data.processing.dataset import build_sequence_dataset, train_test_split_sequences
from data.processing.feature_engineering import create_features
from data.processing.scaler import FeatureScaler
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
        return df.dropna()

    client = BinanceClient()
    return client.get_klines(symbol=args.symbol, interval=args.interval, limit=args.limit)


def train_nn(
    model,
    X,
    y,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    device="cpu",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch}/{epochs}: loss={epoch_loss:.6f}")

    return model


def directional_accuracy(predictions, targets):
    predictions = np.sign(predictions.flatten())
    targets = np.sign(targets.flatten())
    return float((predictions == targets).mean())


def evaluate_model(model, scaler, X, y, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(scaler.transform(X)).to(device)
        preds = model(X_t).cpu().numpy()

    mse = float(np.mean((preds - y) ** 2))
    direction = directional_accuracy(preds, y)
    return mse, direction


def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(args)
    df = create_features(df)

    X, y = build_sequence_dataset(df, seq_len=args.seq_len)
    if X.shape[0] == 0:
        raise ValueError("Not enough data to build a sequence dataset. Increase limit or reduce seq_len.")

    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, test_size=args.test_size)
    scaler = FeatureScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler.save(SCALER_PATH)
    print(f"Saved feature scaler to {SCALER_PATH}")

    train_returns = df["returns"].values[: len(X_train) + args.seq_len - 1]
    hmm = HMMModel(n_states=args.hmm_states)
    hmm.train(train_returns)
    hmm.save(HMM_PATH)
    print(f"Saved HMM model to {HMM_PATH}")

    nn = PricePredictor(input_size=X.shape[2], seq_len=args.seq_len)
    nn = train_nn(
        nn,
        X_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    nn.save(NN_PATH)
    print(f"Saved neural model to {NN_PATH}")

    mse, direction = evaluate_model(nn, scaler, X_test, y_test, device=args.device)
    print(f"Evaluation on {len(X_test)} test samples: MSE={mse:.6f}, directional_accuracy={direction:.4f}")
    # Save training status
    current_metrics = metrics.load_metrics()
    current_metrics["training_status"] = "Trained"
    current_metrics["mse"] = float(mse)
    current_metrics["directional_accuracy"] = float(direction * 100)  # Convert to percentage
    metrics.save_metrics(current_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMM and NN models for the crypto signal engine.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol to train on")
    parser.add_argument("--interval", default="1h", help="Kline interval")
    parser.add_argument("--limit", type=int, default=1000, help="Historical sample size")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length for the NN")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hmm-states", type=int, default=4, help="Number of HMM hidden states")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of samples reserved for evaluation")
    parser.add_argument("--data-path", default=None, help="Optional CSV file path for offline training data")
    parser.add_argument("--device", default="cpu", help="Device for model training (cpu or cuda)")
    args = parser.parse_args()
    main(args)
