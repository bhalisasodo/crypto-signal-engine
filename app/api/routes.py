from fastapi import APIRouter
from pipelines.signal_pipeline import SignalPipeline

router = APIRouter()
pipeline = SignalPipeline()

@router.get("/signal/{symbol}")
def get_signal(symbol: str):
    return pipeline.run(symbol.upper())
