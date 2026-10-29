"""
tests/test_predict.py - Unit tests for ONNX model inference and post-processing logic.
"""
from io import BytesIO
from PIL import Image

from app.predict import predict_image, load_resources
import app.predict as p


def _create_sample_jpeg(width: int = 256, height: int = 256, color=(60, 140, 70)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_resources_loaded():
    load_resources()
    assert p._LABELS is not None
    assert len(p._LABELS) == 30
    assert 0.1 <= p._THRESHOLD <= 0.9


def test_predict_valid_image():
    img_bytes = _create_sample_jpeg(256, 256)
    res = predict_image(img_bytes)

    assert "status" in res
    assert res["status"] in ("ok", "uncertain", "unknown")
    assert res["error_code"] is None
    assert len(res["predictions"]) == 3
    assert res["top_label"] is not None
    assert 0.0 <= res["top_probability"] <= 1.0

    # Probabilities should be descending
    probs = [p["probability"] for p in res["predictions"]]
    assert probs[0] >= probs[1] >= probs[2]

    # Reliability dict must be present
    assert "reliability" in res
    assert isinstance(res["reliability"]["is_reliable"], bool)


def test_predict_bad_image():
    res = predict_image(b"NOT_AN_IMAGE_RANDOM_GARBAGE")
    assert res["status"] == "bad_image"
    assert res["error_code"] == "bad_image"
    assert len(res["predictions"]) == 0
    assert res["top_label"] is None


def test_predict_empty_bytes():
    res = predict_image(b"")
    assert res["status"] == "empty_file"
    assert res["error_code"] == "empty_file"
    assert len(res["predictions"]) == 0


def test_predict_small_image():
    small_bytes = _create_sample_jpeg(50, 50)
    res = predict_image(small_bytes)
    assert res["status"] == "image_too_small"
    assert res["error_code"] == "image_too_small"
    assert len(res["predictions"]) == 0
