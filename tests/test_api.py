"""
tests/test_api.py - Integration tests for FastAPI endpoints and static web assets.
"""
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_sample_jpeg_bytes(width: int = 256, height: int = 256) -> bytes:
    img = Image.new("RGB", (width, height), color=(40, 160, 50))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_config_endpoint():
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "egyptian_crops" in data
    assert len(data["egyptian_crops"]) > 5
    assert data["min_image_side_px"] == 100


def test_diagnose_valid_image():
    img_bytes = _create_sample_jpeg_bytes(256, 256)
    files = {"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
    res = client.post("/api/diagnose", files=files)
    assert res.status_code == 200

    data = res.json()
    assert data["status"] in ("ok", "uncertain", "unknown")
    assert len(data["predictions"]) == 3
    assert data["top_label"] is not None


def test_diagnose_non_image():
    files = {"file": ("document.txt", b"Hello, this is just plain text", "text/plain")}
    res = client.post("/api/diagnose", files=files)
    assert res.status_code == 200

    data = res.json()
    assert data["status"] == "bad_image"
    assert data["error_code"] == "unsupported_media_type"


def test_diagnose_corrupted_image():
    files = {"file": ("corrupt.jpg", b"NOT_VALID_IMAGE_DATA_12345", "image/jpeg")}
    res = client.post("/api/diagnose", files=files)
    assert res.status_code == 200

    data = res.json()
    assert data["status"] == "bad_image"
    assert data["error_code"] == "bad_image"


def test_diagnose_empty_file():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = client.post("/api/diagnose", files=files)
    assert res.status_code == 200

    data = res.json()
    assert data["status"] == "empty_file"


def test_static_pages_serve():
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "المساعد الزراعي الذكي" in res_index.text

    res_diag = client.get("/diagnose.html")
    assert res_diag.status_code == 200
    assert "diagnose.js" in res_diag.text

    res_ar_i18n = client.get("/i18n/ar.json")
    assert res_ar_i18n.status_code == 200
    assert "app_name" in res_ar_i18n.json()
