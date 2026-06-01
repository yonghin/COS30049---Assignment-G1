import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.services.model_loader import load_models
from backend.services.analytics_service import initialize as init_analytics
from backend.routers import spam, malware, analytics, system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    app.state.registry = load_models()
    logger.info("Computing analytics...")
    app.state.analytics = init_analytics(app.state.registry)
    logger.info("All models loaded and analytics ready.")
    yield


app = FastAPI(title="NTCyber AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(spam.router,      prefix="/api/spam",      tags=["spam"])
app.include_router(malware.router,   prefix="/api/malware",   tags=["malware"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(system.router,    prefix="/api",           tags=["system"])
