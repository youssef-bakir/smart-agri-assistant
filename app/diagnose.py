# app/diagnose.py
# What: FastAPI router for disease diagnosis endpoints (/api/diagnose)
# Input: HTTP multipart image upload
# Output: Clean JSON responses with machine codes and prediction metadata
# How to run: Included in app/main.py FastAPI application
# Why: Strictly separates HTTP transport and validation from core inference logic

import logging
from typing import Dict, Any
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.predict import predict_image, get_knowledge

logger = logging.getLogger("smart_agri.diagnose")

router = APIRouter(prefix="/api", tags=["Diagnosis"])


@router.post("/diagnose")
async def diagnose_leaf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts an uploaded image file, performs leaf disease diagnosis,
    and returns machine-readable status codes, top-3 candidates, and metadata.
    Does NOT return localized sentences; frontend translates all codes.
    """
    if not file:
        raise HTTPException(status_code=400, detail={"code": "missing_file"})

    # Check content type if provided (fail fast for obvious non-images)
    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        return {
            "status": "bad_image",
            "error_code": "unsupported_media_type",
            "top_label": None,
            "top_probability": 0.0,
            "threshold": 0.35,
            "predictions": [],
            "reliability": None,
            "knowledge": None,
        }

    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail={"code": "read_error"})

    result = predict_image(image_bytes)
    return result


@router.get("/knowledge")
def get_knowledge_base() -> Dict[str, Any]:
    """Returns the bilingual disease knowledge dictionary."""
    return get_knowledge()
