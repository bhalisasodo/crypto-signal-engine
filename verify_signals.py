import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipelines.signal_pipeline import SignalPipeline
from unittest.mock import MagicMock

def verify():
    print("Testing SignalPipeline with Risk Management Indicators...")
    
    # Mock SignalPipeline's dependencies
    pipeline = SignalPipeline()
    pipeline.client = MagicMock()
    pipeline.hmm = MagicMock()
    pipeline.nn = MagicMock()
    
    # Mock data
    df = pd.DataFrame({
        "timestamp": [1600000000000],
        "close": [60000.0],
        "volatility": [0.02], # 2%
        "ma_fast": [61000.0],
        "ma_slow": [60000.0],
        "returns": [0.01]
    })
    
    # Setup mocks
    pipeline.client.get_klines.return_value = df
    pipeline.hmm.predict_regime.return_value = 1 # Bullish regime
    pipeline.nn.return_value = MagicMock(item=lambda: 0.05) # Positive prediction
    
    # We need to mock create_features too since it's used in run
    import data.processing.feature_engineering
    data.processing.feature_engineering.create_features = lambda x: x
    
    # We need to mock generate_signal too
    import models.ensemble
    models.ensemble.generate_signal = lambda regime, prediction: "BUY"
    
    print("\nRunning BUY signal test...")
    result = pipeline.run("BTCUSDT")
    
    print(f"Signal: {result['signal']}")
    print(f"Price: {result['current_price']}")
    print(f"Volatility: {result['volatility']}")
    print(f"Stop Loss: {result['stop_loss']}")
    print(f"Take Profit: {result['take_profit']}")
    
    # Expected SL: 60000 - (60000 * 0.02 * 1.5) = 60000 - 1800 = 58200
    # Expected TP: 60000 + (60000 * 0.02 * 3.0) = 60000 + 3600 = 63600
    assert result['stop_loss'] == 58200.0
    assert result['take_profit'] == 63600.0
    print("BUY signal risk levels verified!")

    # Test SELL signal
    models.ensemble.generate_signal = lambda regime, prediction: "SELL"
    print("\nRunning SELL signal test...")
    result = pipeline.run("BTCUSDT")
    
    # Expected SL: 60000 + (60000 * 0.02 * 1.5) = 60000 + 1800 = 61800
    # Expected TP: 60000 - (60000 * 0.02 * 3.0) = 60000 - 3600 = 56400
    assert result['stop_loss'] == 61800.0
    assert result['take_profit'] == 56400.0
    print("SELL signal risk levels verified!")

    # Test Telegram message formatting
    from pipelines.telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier(bot_token="test", chat_id="test")
    message = notifier._format_signal_message(result)
    
    print("\nFormatted Telegram Message Preview:")
    print("-" * 30)
    print(message)
    print("-" * 30)
    
    assert "Stop Loss: $61,800.00" in message
    assert "Take Profit: $56,400.00" in message
    print("Telegram message format verified!")

if __name__ == "__main__":
    verify()
