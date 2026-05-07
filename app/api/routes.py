from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pipelines.signal_pipeline import SignalPipeline
import metrics

router = APIRouter()
pipeline = SignalPipeline()
templates = Jinja2Templates(directory="templates")

@router.get("/signal/{symbol}")
def get_signal(symbol: str):
    return pipeline.run(symbol.upper())

@router.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard", status_code=302)

@router.get("/dashboard")
def dashboard(request: Request):
    current_metrics = metrics.load_metrics()
    return templates.TemplateResponse(request, "dashboard.html", {"metrics": current_metrics})

