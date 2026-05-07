from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Crypto Signal Engine")

app.include_router(router)

