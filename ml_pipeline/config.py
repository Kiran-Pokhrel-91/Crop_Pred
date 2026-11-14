"""
Configuration module for Nepal Crop Prediction Pipeline.
Loads environment variables, defines directory paths, and sets geospatial constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _resolve_project_path(env_var: str, default: Path) -> Path:
    """Resolve a configured path, treating relative values as project-relative."""
    raw_value = os.getenv(env_var)
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


DATA_DIR = _resolve_project_path("DATA_DIR", BASE_DIR / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MODELS_DIR = _resolve_project_path("MODEL_DIR", BASE_DIR / "models")

for directory in [RAW_DIR, PROCESSED_DIR, FINAL_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

GEE_SERVICE_ACCOUNT = os.getenv("GEE_SERVICE_ACCOUNT")
GEE_PRIVATE_KEY_PATH = os.getenv("GEE_PRIVATE_KEY_PATH")
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID")

TEST_MODE = os.getenv("TEST_MODE", "False").lower() == "true"

NEPAL_CRS = "EPSG:4326"
NEPAL_PROJECTED_CRS = "EPSG:32645"

NEPAL_DISTRICTS_URL = (
    "https://raw.githubusercontent.com/manishacharya60/nepal-geojson/main/"
    "Nepal_Administrative_Boundary_District.geojson"
)

GRID_RESOLUTION_METERS = 5000.0
