from fastapi.testclient import TestClient
from app.main import app
from app.api import routes

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_dashboard_endpoint_returns_html():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Crypto Signal Engine" in response.text


def test_signal_endpoints_use_pipeline(monkeypatch):
    mocked_result = {"symbol": "BTCUSDT", "regime": 1, "prediction": 0.12, "signal": "BUY"}
    monkeypatch.setattr(routes.pipeline, "run", lambda symbol: mocked_result)
    monkeypatch.setattr(routes.pipeline, "live_run", lambda symbol: {**mocked_result, "timestamp": 0})

    response = client.get("/signal/BTCUSDT")
    assert response.status_code == 200
    assert response.json()["signal"] == "BUY"

    response = client.get("/live-signal/BTCUSDT")
    assert response.status_code == 200
    assert response.json()["timestamp"] == 0
