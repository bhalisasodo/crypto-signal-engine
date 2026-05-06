# Crypto Signal Engine

Minimal MVP for crypto trading signal generation using:
- HMM (regime detection)
- PyTorch NN (prediction)
- FastAPI (serving signals)

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoint
GET /signal/{symbol}
