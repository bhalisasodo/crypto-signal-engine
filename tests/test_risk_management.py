import pytest
from pipelines.signal_pipeline import SignalPipeline

def test_risk_levels_buy():
    pipeline = SignalPipeline()
    pipeline.sl_multiplier = 1.0
    pipeline.tp_multiplier = 2.0
    
    entry_price = 100.0
    volatility = 0.02 # 2%
    
    sl, tp = pipeline._calculate_risk_levels("BUY", entry_price, volatility)
    
    # SL should be entry - (entry * vol * multiplier) = 100 - (100 * 0.02 * 1.0) = 98
    assert sl == 98.0
    # TP should be entry + (entry * vol * multiplier) = 100 + (100 * 0.02 * 2.0) = 104
    assert tp == 104.0

def test_risk_levels_sell():
    pipeline = SignalPipeline()
    pipeline.sl_multiplier = 1.0
    pipeline.tp_multiplier = 2.0
    
    entry_price = 100.0
    volatility = 0.02 # 2%
    
    sl, tp = pipeline._calculate_risk_levels("SELL", entry_price, volatility)
    
    # SL should be entry + (entry * vol * multiplier) = 100 + (100 * 0.02 * 1.0) = 102
    assert sl == 102.0
    # TP should be entry - (entry * vol * multiplier) = 100 - (100 * 0.02 * 2.0) = 96
    assert tp == 96.0

def test_risk_levels_hold():
    pipeline = SignalPipeline()
    sl, tp = pipeline._calculate_risk_levels("HOLD", 100.0, 0.02)
    assert sl is None
    assert tp is None
