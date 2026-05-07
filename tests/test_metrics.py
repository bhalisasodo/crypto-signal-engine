import json
import os
from pathlib import Path
import metrics


def test_metrics_save_and_load(tmp_path):
    temp_file = tmp_path / "metrics.json"
    original = {
        "training_status": "Tested",
        "mse": 0.001,
        "directional_accuracy": 50.0,
        "cumulative_return": 10.0,
        "buy_hold_return": 2.0,
        "signal_counts": {"BUY": 1, "SELL": 1, "HOLD": 0},
    }

    old_metrics_file = metrics.METRICS_FILE
    try:
        metrics.METRICS_FILE = str(temp_file)
        metrics.save_metrics(original)
        loaded = metrics.load_metrics()
        assert loaded == original
    finally:
        metrics.METRICS_FILE = old_metrics_file
