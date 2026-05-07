from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pipelines.signal_pipeline import SignalPipeline
from pipelines.telegram_notifier import send_trading_signal
import metrics
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
pipeline = SignalPipeline()
templates = Jinja2Templates(directory="templates")

def _attach_model_metrics(signal_data: dict):
    current_metrics = metrics.load_metrics()
    accuracy = current_metrics.get("directional_accuracy")
    cv_accuracy = current_metrics.get("cv_directional_accuracy")
    if accuracy is not None:
        signal_data["directional_accuracy"] = float(accuracy)
    if cv_accuracy is not None:
        signal_data["cv_directional_accuracy"] = float(cv_accuracy)
    return signal_data

@router.get("/signal/{symbol}")
def get_signal(symbol: str):
    logger.info(f"Generating signal for {symbol}")
    signal_data = pipeline.run(symbol.upper())
    signal_data = _attach_model_metrics(signal_data)
    # Send signal to Telegram
    send_trading_signal(signal_data)
    return signal_data

@router.get("/live-signal/{symbol}")
def get_live_signal(symbol: str):
    logger.info(f"Generating live signal for {symbol}")
    signal_data = pipeline.live_run(symbol.upper())
    signal_data = _attach_model_metrics(signal_data)
    # Send signal to Telegram
    send_trading_signal(signal_data)
    return signal_data

@router.get("/")
def root():
    logger.info("Root endpoint accessed")
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard", status_code=302)

@router.get("/dashboard")
def dashboard(request: Request):
    logger.info("Dashboard accessed")
    current_metrics = metrics.load_metrics()
    return templates.TemplateResponse(request, "dashboard.html", {"metrics": current_metrics})

@router.get("/health")
def health():
    logger.info("Health check")
    return {"status": "healthy"}

@router.get("/price-data/{symbol}")
def get_price_data(symbol: str):
    logger.info(f"Fetching price data for {symbol}")
    df = pipeline.client.get_klines(symbol=symbol.upper(), limit=100)
    prices = [float(x) for x in df["close"].tolist()]
    timestamps = [int(x) if hasattr(x, "__int__") else x for x in df["timestamp"].tolist()]
    return {"timestamps": timestamps, "prices": prices}

