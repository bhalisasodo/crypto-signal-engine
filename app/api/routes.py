from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pipelines.signal_pipeline import SignalPipeline
import metrics
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
pipeline = SignalPipeline()
templates = Jinja2Templates(directory="templates")

@router.get("/signal/{symbol}")
def get_signal(symbol: str):
    logger.info(f"Generating signal for {symbol}")
    return pipeline.run(symbol.upper())

@router.get("/live-signal/{symbol}")
def get_live_signal(symbol: str):
    logger.info(f"Generating live signal for {symbol}")
    return pipeline.live_run(symbol.upper())

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

