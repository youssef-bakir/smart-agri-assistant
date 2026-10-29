# app/main.py
# What: FastAPI application entry point, mounting static UI and API routes
# Input: HTTP requests from browser or API clients
# Output: JSON API responses and static web assets (HTML/CSS/JS)
# How to run: uvicorn app.main:app --reload --port 8000
# Why: Unified lightweight single-process ASGI server without extra reverse proxies

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
from app.diagnose import router as diagnose_router
from app.predict import load_resources, is_model_loaded

# Configure structured application logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smart_agri.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up model and metadata on server startup."""
    logger.info("Initializing Smart Agri Assistant backend...")
    load_resources()
    if is_model_loaded():
        logger.info("ONNX model and metadata loaded successfully.")
    else:
        logger.warning("ONNX model file not found in models/ directory.")
    yield
    logger.info("Shutting down Smart Agri Assistant...")


app = FastAPI(
    title="Smart Agri Assistant",
    description="AI-powered plant leaf disease diagnostic system for Egyptian agriculture.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(diagnose_router)


@app.get("/api/health")
def health_check():
    """System health and model readiness check."""
    return {
        "status": "ok",
        "env": config.ENV,
        "model_loaded": is_model_loaded(),
    }


@app.get("/api/config")
def get_public_config():
    """Returns safe client-facing feature flags and settings."""
    return {
        "feature_accounts": config.FEATURE_ACCOUNTS,
        "feature_soil": config.FEATURE_SOIL,
        "max_upload_bytes": config.MAX_UPLOAD_BYTES,
        "min_image_side_px": config.MIN_IMAGE_SIDE_PX,
        "egyptian_crops": config.EGYPTIAN_CROPS,
    }


# Global error handler - never leaks stack traces to client
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error_code": "internal_server_error"},
    )


# Mount static frontend assets
web_dir = Path(config.BASE_DIR) / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
