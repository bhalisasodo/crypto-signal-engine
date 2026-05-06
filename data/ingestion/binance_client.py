from binance.client import Client
import pandas as pd
import os

class BinanceClient:
    def __init__(self):
        self.client = Client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_SECRET_KEY")
        )

    def get_klines(self, symbol="BTCUSDT", interval="1h", limit=500):
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_", "_", "_", "_", "_", "_"
        ])

        df["close"] = df["close"].astype(float)
        df["returns"] = df["close"].pct_change()

        return df.dropna()
