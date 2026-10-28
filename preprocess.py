"""
preprocess.py - The single source of truth for image preprocessing.
Used across data preparation (1_prepare_data.py), training (2_train.py),
evaluation (3_export_and_eval.py), and the web app backend (app/predict.py).
"""
from io import BytesIO
from typing import Tuple, Optional
import numpy as np
from PIL import Image

import config

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """
    Computes an 8x8 perceptual difference hash (dHash, 64 bits).
    1. Converts image to grayscale.
    2. Resizes to (hash_size + 1, hash_size) -> (9, 8).
    3. Compares adjacent horizontal pixels (P[x, y] > P[x+1, y]).
    4. Packs the 64 boolean comparisons into a single 64-bit integer.
    """
    # Convert to grayscale and resize to 9 width x 8 height
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.int32)
    # Compare adjacent pixels in each row
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert boolean array to 64-bit integer
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def dhash_distance(hash1: int, hash2: int) -> int:
    """Calculates Hamming distance (number of differing bits) between two hashes."""
    return bin(hash1 ^ hash2).count("1")


def validate_image_bytes(image_bytes: bytes) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates uploaded image before feeding into model.
    Checks:
    - Max size <= 8 MB
    - Successfully decodes as image
    - Minimum side >= 100 px
    Returns (is_valid, error_code, PIL_image)
    """
    if not image_bytes or len(image_bytes) == 0:
        return False, "empty_file", None

    if len(image_bytes) > config.MAX_UPLOAD_BYTES:
        return False, "file_too_large", None

    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()  # Verify integrity
        # Reopen because verify closes or leaves PIL image in unusable state
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except Exception:
        return False, "bad_image", None

    w, h = img.size
    if min(w, h) < config.MIN_IMAGE_SIDE_PX:
        return False, "image_too_small", None

    return True, "ok", img


def preprocess_for_onnx(img: Image.Image) -> np.ndarray:
    """
    Prepares a PIL Image for ONNX Runtime inference without PyTorch.
    1. Converts to RGB.
    2. Resizes to (224, 224) using bilinear interpolation.
    3. Scales pixels to [0, 1] and normalizes with ImageNet mean/std.
    4. Transposes to shape (1, 3, 224, 224) (float32 NCHW).
    """
    rgb_img = img.convert("RGB")
    resized = rgb_img.resize(config.TRAIN_INPUT_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    normalized = (arr - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW -> NCHW
    chw = np.transpose(normalized, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0)
    return nchw.astype(np.float32)


def get_train_transforms():
    """
    Returns torchvision transforms for training on GPU.
    Lazy imports torchvision so inference / web app doesn't require torch.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])


def get_eval_transforms():
    """Returns torchvision transforms for evaluation / validation."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(config.TRAIN_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])

