# Crypto Signal Engine

Minimal MVP for crypto trading signal generation using:
- HMM (regime detection)
- PyTorch NN (prediction)
- FastAPI (serving signals)
- Telegram (real-time group notifications)

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run
```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/dashboard` for the UI dashboard.

## Telegram Integration

Send trading signals to a Telegram group in real-time to share with friends!

### Setup

1. **Create a Telegram Bot:**
   - Chat with [@BotFather](https://t.me/botfather) on Telegram
   - Send `/newbot` and follow the prompts
   - Copy the API token provided

2. **Create a Telegram Group:**
   - Create a new group or use an existing one
   - Add your bot to the group
   - Get the group chat ID using:
     ```bash
     curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates" | grep -o '"chat":{"id":-[0-9]*'
     ```
     Or send a message in the group and check the URL structure.

3. **Set Environment Variables:**
   ```bash
   export TELEGRAM_BOT_TOKEN="your_bot_token_here"
   export TELEGRAM_CHAT_ID="your_chat_id_here"
   ```

   Or create a local `.env` file in the project root and the app will load it automatically.

4. **Start the Engine:**
   ```bash
   uvicorn app.main:app --reload
   ```

Signals will now be sent to your Telegram group automatically! 🚀

### Docker Setup with Telegram

If using Docker Compose, add environment variables to `docker-compose.yml`:
```yaml
environment:
  - TELEGRAM_BOT_TOKEN=your_bot_token_here
  - TELEGRAM_CHAT_ID=your_chat_id_here
```

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

## Real-Time Signals

The system now automatically generates trading signals every 5 minutes, but **only sends them to Telegram when the signal changes**.

### How it works:
- Background scheduler runs continuously when the server is active
- Generates live signals for BTCUSDT every 5 minutes
- **Only sends to Telegram when the signal changes** (BUY ↔ SELL ↔ HOLD)
- Includes duplicate suppression as additional safety
- Runs alongside manual signal requests via API endpoints

### Signal Change Detection:
- Tracks the last sent signal for each symbol
- Compares new signals against the previous one
- Only sends notifications when the trading recommendation changes
- Prevents spam while keeping users informed of important changes

### Configuration:
The real-time scheduler starts automatically when you run the server. No additional configuration needed beyond your existing Telegram setup.

### Testing Real-Time Signals:
```bash
python test_realtime.py
```

This will manually trigger a signal generation and send it to Telegram for testing.

## Testing
```bash
pip install -r requirements-dev.txt
pytest
```

## Docker
Build and run with Docker:
```bash
docker build -t crypto-signal-engine .
docker run -p 8000:8000 -e TELEGRAM_BOT_TOKEN="token" -e TELEGRAM_CHAT_ID="chatid" crypto-signal-engine
```

Or with Docker Compose:
```bash
docker compose up --build
```
