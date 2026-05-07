from dotenv import load_dotenv
from fastapi import FastAPI
from app.api.routes import router
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pipelines.signal_pipeline import SignalPipeline
from pipelines.telegram_notifier import send_trading_signal
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Signal Engine")

app.include_router(router)

# Global scheduler instance
scheduler = AsyncIOScheduler()
pipeline = SignalPipeline()
last_sent_signal = {"symbol": None, "signal": None, "timestamp": None}

async def send_realtime_signal():
    """Background task to generate and send real-time signals only when they change."""
    try:
        logger.info("Generating real-time signal for BTCUSDT")
        signal_data = pipeline.live_run("BTCUSDT")

        # Check if signal has changed
        current_signal = signal_data.get("signal")
        current_symbol = signal_data.get("symbol")

        global last_sent_signal
        if (last_sent_signal["symbol"] == current_symbol and
            last_sent_signal["signal"] == current_signal):
            logger.info(f"Signal unchanged ({current_signal}) - skipping Telegram send")
            return

        # Signal has changed, send it
        success = send_trading_signal(signal_data)
        if success:
            # Update last sent signal
            last_sent_signal = {
                "symbol": current_symbol,
                "signal": current_signal,
                "timestamp": signal_data.get("timestamp")
            }
            logger.info(f"New signal sent to Telegram: {current_signal}")
        else:
            logger.error("Failed to send signal to Telegram")

    except Exception as e:
        logger.error(f"Error in real-time signal generation: {e}")

@app.on_event("startup")
async def startup_event():
    logger.info("Crypto Signal Engine API started")

    # Start the background scheduler for real-time signals
    scheduler.add_job(
        send_realtime_signal,
        trigger=IntervalTrigger(minutes=5),  # Send signal every 5 minutes
        id="realtime_signal",
        name="Real-time Signal Generation",
        max_instances=1
    )

    scheduler.start()
    logger.info("Background scheduler started - signals will be sent every 5 minutes")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Crypto Signal Engine API shutting down")
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler stopped")

