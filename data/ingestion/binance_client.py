from binance.client import Client
import pandas as pd
import os
import pickle
import time

CACHE_FILE = os.path.join(os.path.dirname(__file__), "data_cache.pkl")

class BinanceClient:
    def __init__(self):
        self.client = Client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_SECRET_KEY")
        )
        self.cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                self.cache = pickle.load(f)

    def get_klines(self, symbol="BTCUSDT", interval="1h", limit=500):
        key = f"{symbol}_{interval}_{limit}"
        if key in self.cache:
            print(f"Using cached data for {key}")
            return self.cache[key].copy()

        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
        except Exception as e:
            print(f"API error: {e}, retrying in 60 seconds...")
            time.sleep(60)
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

        df = df.dropna()
        self.cache[key] = df.copy()
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(self.cache, f)
        return df
