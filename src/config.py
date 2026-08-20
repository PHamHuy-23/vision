import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables if .env file exists
env_path = BASE_DIR / ".env"
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass

# Data directory root path
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(BASE_DIR / "data"))).resolve()

# Directory structure conventions inside DATA_ROOT
CLIP_DIR_NAME = os.getenv("CLIP_DIR_NAME", "CLIP")
KEYFRAMES_DIR_NAME = os.getenv("KEYFRAMES_DIR_NAME", "keyframes")
MAPPING_DIR_NAME = os.getenv("MAPPING_DIR_NAME", "map_keyframes")

# Database & Vector storage paths
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "video_index.db"))).resolve()
CONSOLIDATED_VECTORS_PATH = Path(os.getenv("CONSOLIDATED_VECTORS_PATH", str(BASE_DIR / "all_vectors.npy"))).resolve()

# Allowed file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}
VECTOR_EXTENSIONS = {".npy"}

# CLIP Model Settings
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
