import json
import os

METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")

def save_metrics(metrics):
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)

def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {
        "training_status": "Not trained",
        "mse": None,
        "directional_accuracy": None,
        "cumulative_return": None,
        "buy_hold_return": None,
        "signal_counts": {"BUY": 0, "SELL": 0, "HOLD": 0}
    }