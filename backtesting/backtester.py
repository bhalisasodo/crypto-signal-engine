class Backtester:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, symbol="BTCUSDT", steps=50):
        results = []
        for _ in range(steps):
            results.append(self.pipeline.run(symbol))
        return results
