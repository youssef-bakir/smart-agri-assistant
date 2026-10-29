# app/predict.py
# What: Disease inference engine using ONNX Runtime with thresholding and reliability checking
# Input: Raw image bytes (from client upload or API call)
# Output: Structured dictionary with status code, top-3 predictions, reliability, and knowledge
# How to run: Imported by app/diagnose.py or run directly for testing
# Why: CPU-friendly, memory-efficient inference without requiring heavy PyTorch at runtime

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import onnxruntime as ort

import config
import preprocess

# Cached global resources
_SESSION: Optional[ort.InferenceSession] = None
_LABELS: Optional[list] = None
_THRESHOLD: float = 0.35
_RELIABILITY: Dict[str, Any] = {}
_KNOWLEDGE: Dict[str, Any] = {}


def load_resources() -> None:
    """
    Loads model and metadata files into memory once on startup.
    Keeps inference fast and avoids reloading files on every scan.
    """
    global _SESSION, _LABELS, _THRESHOLD, _RELIABILITY, _KNOWLEDGE

    models_dir = Path(config.MODELS_DIR)
    app_dir = Path(config.BASE_DIR) / "app"

    # 1. Load class labels
    labels_path = models_dir / "labels.json"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            _LABELS = json.load(f)
    else:
        _LABELS = list(config.CLASSES)

    # 2. Load calibrated threshold
    threshold_path = models_dir / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            _THRESHOLD = float(t_data.get("threshold", config.DEFAULT_CONFIDENCE_THRESHOLD))
    else:
        _THRESHOLD = config.DEFAULT_CONFIDENCE_THRESHOLD

    # 3. Load class reliability metadata
    rel_path = models_dir / "class_reliability.json"
    if rel_path.exists():
        with open(rel_path, "r", encoding="utf-8") as f:
            _RELIABILITY = json.load(f)
    else:
        _RELIABILITY = {}

    # 4. Load disease knowledge base
    knowledge_path = app_dir / "knowledge.json"
    if knowledge_path.exists():
        with open(knowledge_path, "r", encoding="utf-8") as f:
            _KNOWLEDGE = json.load(f)
    else:
        _KNOWLEDGE = {}

    # 5. Initialize ONNX Runtime session
    onnx_path = models_dir / "model.onnx"
    if onnx_path.exists():
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSION = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )


def is_model_loaded() -> bool:
    """Returns True if ONNX inference session is initialized."""
    return _SESSION is not None


def get_knowledge() -> Dict[str, Any]:
    """Returns the loaded disease knowledge base."""
    return _KNOWLEDGE


def softmax(logits: np.ndarray) -> np.ndarray:
    """Computes numerically stable softmax probabilities from raw logits."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def predict_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Main prediction pipeline:
    1. Validates raw image bytes (size, format, dimensions).
    2. Runs ONNX inference.
    3. Calculates softmax probabilities and top-3 candidates.
    4. Applies validation-calibrated threshold to mark uncertain scans.
    5. Checks class reliability and attaches bilingual knowledge.
    """
    if _SESSION is None:
        load_resources()

    # Step 1: Input image validation
    is_valid, error_code, pil_img = preprocess.validate_image_bytes(image_bytes)
    if not is_valid:
        return {
            "status": error_code,
            "error_code": error_code,
            "top_label": None,
            "top_probability": 0.0,
            "threshold": _THRESHOLD,
            "predictions": [],
            "reliability": None,
            "knowledge": None,
        }

    # If model is missing on disk (e.g. before training), return graceful error
    if _SESSION is None:
        return {
            "status": "model_not_found",
            "error_code": "model_not_found",
            "top_label": None,
            "top_probability": 0.0,
            "threshold": _THRESHOLD,
            "predictions": [],
            "reliability": None,
            "knowledge": None,
        }

    # Step 2: Preprocessing for ONNX Runtime (NCHW, 224x224, normalized)
    input_tensor = preprocess.preprocess_for_onnx(pil_img)
    input_name = _SESSION.get_inputs()[0].name
    output_name = _SESSION.get_outputs()[0].name

    # Step 3: Run inference
    raw_outputs = _SESSION.run([output_name], {input_name: input_tensor})[0]
    logits = raw_outputs[0]

    # Step 4: Softmax probabilities
    probs = softmax(logits)

    # Step 5: Rank top-3 predictions
    top_indices = np.argsort(probs)[::-1][:3]
    predictions = []
    for idx in top_indices:
        label = _LABELS[idx] if _LABELS and idx < len(_LABELS) else f"class_{idx}"
        prob = float(probs[idx])
        predictions.append({
            "label": label,
            "probability": round(prob, 4),
        })

    top_label = predictions[0]["label"]
    top_prob = predictions[0]["probability"]

    # Step 6: Determine classification status
    if top_label == "unknown":
        status = "unknown"
    elif top_prob < _THRESHOLD:
        status = "uncertain"
    else:
        status = "ok"

    # Step 7: Reliability check (warn if low field recall or lab only)
    if top_label in _RELIABILITY:
        rel_entry = _RELIABILITY[top_label]
        reliability = {
            "is_reliable": False,
            "reason": rel_entry.get("reason", "low_field_recall"),
            "field_recall": rel_entry.get("field_recall"),
        }
    else:
        reliability = {
            "is_reliable": True,
            "reason": None,
            "field_recall": None,
        }

    # Step 8: Knowledge base lookup
    if status in ("ok", "uncertain") and top_label in _KNOWLEDGE:
        knowledge = _KNOWLEDGE[top_label]
    elif status == "unknown" and "unknown" in _KNOWLEDGE:
        knowledge = _KNOWLEDGE["unknown"]
    else:
        knowledge = None

    return {
        "status": status,
        "error_code": None,
        "top_label": top_label,
        "top_probability": top_prob,
        "threshold": _THRESHOLD,
        "predictions": predictions,
        "reliability": reliability,
        "knowledge": knowledge,
    }
