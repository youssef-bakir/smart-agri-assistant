"""
tests/test_preprocess.py - Unit tests for image validation, dHash, and ONNX preprocessing.
"""
from io import BytesIO
import numpy as np
from PIL import Image

import config
from preprocess import (
    validate_image_bytes,
    compute_dhash,
    dhash_distance,
    preprocess_for_onnx,
)


def _create_test_image(width: int, height: int, color=(100, 150, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_validate_image_empty():
    is_valid, err, img = validate_image_bytes(b"")
    assert not is_valid
    assert err == "empty_file"
    assert img is None


def test_validate_image_too_small():
    data = _create_test_image(80, 80)
    is_valid, err, img = validate_image_bytes(data)
    assert not is_valid
    assert err == "image_too_small"
    assert img is None


def test_validate_image_corrupted():
    corrupted_data = b"NOT_A_VALID_IMAGE_DATA_BYTES_XYZ"
    is_valid, err, img = validate_image_bytes(corrupted_data)
    assert not is_valid
    assert err == "bad_image"
    assert img is None


def test_validate_image_too_large():
    fake_huge_bytes = b"0" * (config.MAX_UPLOAD_BYTES + 1024)
    is_valid, err, img = validate_image_bytes(fake_huge_bytes)
    assert not is_valid
    assert err == "file_too_large"
    assert img is None


def test_validate_image_valid():
    data = _create_test_image(256, 256)
    is_valid, err, img = validate_image_bytes(data)
    assert is_valid
    assert err == "ok"
    assert img is not None
    assert img.size == (256, 256)


def test_dhash_identical_images():
    img1 = Image.new("RGB", (256, 256), color=(50, 100, 150))
    img2 = Image.new("RGB", (256, 256), color=(50, 100, 150))
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert h1 == h2
    assert dhash_distance(h1, h2) == 0


def test_dhash_different_images():
    # Gradient image vs solid image
    arr1 = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    arr2 = np.tile(np.linspace(255, 0, 256, dtype=np.uint8).reshape(-1, 1), (1, 256))
    img1 = Image.fromarray(arr1, mode="L")
    img2 = Image.fromarray(arr2, mode="L")
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert dhash_distance(h1, h2) > 0


def test_preprocess_for_onnx_shape_and_dtype():
    img = Image.new("RGB", (300, 400), color=(120, 80, 40))
    tensor = preprocess_for_onnx(img)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32

