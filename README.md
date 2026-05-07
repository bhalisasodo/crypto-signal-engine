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

Visit `http://localhost:8000/dashboard` for the UI dashboard.

## Training
```bash
python train.py --symbol BTCUSDT --interval 1h --limit 1000 --seq-len 20 --epochs 20
```

## Offline training
If you have historical OHLC data in CSV format, use:
```bash
python train.py --data-path data.csv --seq-len 20 --epochs 20
```

## Evaluation
```bash
python evaluate.py --symbol BTCUSDT --interval 1h --limit 1000 --seq-len 20
```

If you have offline data, use:
```bash
python evaluate.py --data-path data.csv --seq-len 20
```

## Endpoint
GET /signal/{symbol}
