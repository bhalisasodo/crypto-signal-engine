def generate_signal(regime, prediction):
    if regime == 2 and prediction > 0:
        return "BUY"
    elif regime == 0 and prediction < 0:
        return "SELL"
    return "HOLD"
