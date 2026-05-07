def generate_signal(regime, prediction, state_means=None, threshold=0.0):
    if state_means is not None and len(state_means) > regime:
        regime_direction = 1 if state_means[regime] > threshold else -1 if state_means[regime] < -threshold else 0
    else:
        regime_direction = 1 if regime >= 2 else -1 if regime <= 1 else 0

    if regime_direction > 0 and prediction > threshold:
        return "BUY"
    if regime_direction < 0 and prediction < -threshold:
        return "SELL"
    return "HOLD"
