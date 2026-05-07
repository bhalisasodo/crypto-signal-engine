import pandas as pd
import numpy as np
from pipelines.signal_pipeline import SignalPipeline


def test_live_run_returns_native_timestamp(monkeypatch):
    df = pd.DataFrame({
        "timestamp": np.arange(1, 101, dtype=np.int64),
        "open": np.ones(100, dtype=float),
        "high": np.ones(100, dtype=float),
        "low": np.ones(100, dtype=float),
        "close": np.linspace(100, 200, 100, dtype=float),
        "volume": np.ones(100, dtype=float),
        "returns": np.linspace(-0.01, 0.01, 100, dtype=float),
        "volatility": np.ones(100, dtype=float),
        "ma_fast": np.linspace(100, 200, 100, dtype=float),
        "ma_slow": np.linspace(100, 200, 100, dtype=float),
    })

    pipeline = SignalPipeline()
    monkeypatch.setattr(pipeline.client, "get_klines", lambda symbol, interval="1h", limit=100: df)
    result = pipeline.live_run("BTCUSDT")

    assert isinstance(result["timestamp"], int)
    assert result["symbol"] == "BTCUSDT"
    assert result["signal"] in {"BUY", "SELL", "HOLD"}
