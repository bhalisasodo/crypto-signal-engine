from fastapi import FastAPI
from app.api.routes import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Signal Engine")

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    logger.info("Crypto Signal Engine API started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Crypto Signal Engine API shutting down")

