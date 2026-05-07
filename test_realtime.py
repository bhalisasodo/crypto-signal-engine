#!/usr/bin/env python3
"""
Test script for real-time signal generation.
Run this to manually trigger a signal outside of the scheduler.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.signal_pipeline import SignalPipeline
from pipelines.telegram_notifier import send_trading_signal

def test_realtime_signal():
    """Generate and send a real-time signal for testing."""
    print("Generating real-time signal for BTCUSDT...")

    pipeline = SignalPipeline()
    signal_data = pipeline.live_run("BTCUSDT")

    print(f"Signal: {signal_data.get('signal')}")
    print(f"Confidence: {signal_data.get('confidence_pct', signal_data.get('confidence'))}%")
    print(f"Prediction: {signal_data.get('prediction_pct', signal_data.get('prediction'))}%")

    success = send_trading_signal(signal_data)
    if success:
        print("✅ Signal sent to Telegram successfully!")
    else:
        print("❌ Failed to send signal to Telegram (may be duplicate or disabled)")

if __name__ == "__main__":
    test_realtime_signal()