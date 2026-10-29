"""
config.py - All configuration settings for Smart Agri Assistant.
Central place for paths, crop lists, training params, and feature flags.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

PACK_ZIP_PATH = DATA_DIR / "pack.zip"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

# Egyptian priority crops (marked in UI and prioritized in datasets)
EGYPTIAN_CROPS = [
    "tomato",
    "potato",
    "cucumber",
    "pepper",
    "eggplant",
    "onion",
    "squash",
    "maize",
    "wheat",
    "rice",
    "cotton",
    "grape",
    "citrus",
    "strawberry",
    "mango",
]

# Supported non-Egyptian crops (kept only if sufficient images exist)
OTHER_CROPS = [
    "apple",
    "cherry",
    "peach",
    "corn",
]

# Data preparation caps & thresholds
IMAGE_SIZE = (256, 256)
TRAIN_INPUT_SIZE = (256, 256)
JPEG_QUALITY = 90

# Definitive list of target classes (verified >= 100 training images + unknown)
CLASSES = [
    "apple___healthy",
    "apple___rust",
    "apple___scab",
    "corn___leaf_blight",
    "cotton___diseased",
    "cotton___healthy",
    "cucumber___anthracnose",
    "cucumber___bacterial_wilt",
    "cucumber___downy_mildew",
    "cucumber___gummy_stem_blight",
    "cucumber___healthy",
    "grape___black_rot",
    "grape___healthy",
    "pepper___bacterial_spot",
    "pepper___healthy",
    "potato___early_blight",
    "potato___healthy",
    "potato___late_blight",
    "squash___powdery_mildew",
    "tomato___bacterial_spot",
    "tomato___early_blight",
    "tomato___healthy",
    "tomato___late_blight",
    "tomato___leaf_mold",
    "tomato___mosaic_virus",
    "tomato___septoria_leaf_spot",
    "tomato___spider_mites",
    "tomato___target_spot",
    "tomato___yellow_leaf_curl_virus",
    "unknown",
]

CAP_LAB_PER_CLASS = 500
CAP_FIELD_PER_CLASS = 1500
CAP_UNKNOWN = 3000
MIN_TRAIN_IMAGES_PER_CLASS = 100

DHASH_SIZE = 8  # 8x8 perceptual hash for near-duplicate grouping

# Training defaults
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
TARGET_TRAIN_MINUTES = 40
MIN_THROUGHPUT_IMG_PER_SEC = 500

# App & Inference defaults
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB
MIN_IMAGE_SIDE_PX = 100
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

# Feature Flags
FEATURE_ACCOUNTS = os.getenv("FEATURE_ACCOUNTS", "true").lower() == "true"
FEATURE_SOIL = os.getenv("FEATURE_SOIL", "true").lower() == "true"

# Environment & Security
ENV = os.getenv("ENV", "dev")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-secret-key-change-in-production")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'app.db'}")

