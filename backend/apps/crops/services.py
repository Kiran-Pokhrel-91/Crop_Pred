"""Crop catalogue and seasonal metadata services."""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings

SUPPORTED_CROPS = [
    {
        "name": "Barley",
        "type": "cereal",
        "scientific_name": "Hordeum vulgare",
        "seasons": [{"season": "winter", "planting": "Oct-Nov", "harvest": "Apr-May"}],
    },
    {
        "name": "Buckwheat",
        "type": "pseudocereal",
        "scientific_name": "Fagopyrum esculentum",
        "seasons": [{"season": "autumn", "planting": "Aug-Sep", "harvest": "Nov-Dec"}],
    },
    {
        "name": "Maize",
        "type": "cereal",
        "scientific_name": "Zea mays",
        "seasons": [
            {"season": "summer", "planting": "Feb-Apr", "harvest": "Jun-Aug"},
            {"season": "winter-hills", "planting": "Sep-Oct", "harvest": "Jan-Feb"},
        ],
    },
    {
        "name": "Millet",
        "type": "cereal",
        "scientific_name": "Eleusine coracana",
        "seasons": [{"season": "summer", "planting": "Jun-Jul", "harvest": "Oct-Nov"}],
    },
    {
        "name": "Oilseed",
        "type": "oilseed",
        "scientific_name": "Brassica spp.",
        "seasons": [{"season": "winter", "planting": "Sep-Oct", "harvest": "Feb-Mar"}],
    },
    {
        "name": "Paddy",
        "type": "cereal",
        "scientific_name": "Oryza sativa",
        "seasons": [{"season": "monsoon", "planting": "Jun-Jul", "harvest": "Oct-Nov"}],
    },
    {
        "name": "Potato",
        "type": "tuber",
        "scientific_name": "Solanum tuberosum",
        "seasons": [{"season": "winter", "planting": "Sep-Oct", "harvest": "Jan-Feb"}],
    },
    {
        "name": "Sugarcane",
        "type": "cash_crop",
        "scientific_name": "Saccharum officinarum",
        "seasons": [{"season": "annual", "planting": "Feb-Mar", "harvest": "Dec-Mar"}],
    },
    {
        "name": "Wheat",
        "type": "cereal",
        "scientific_name": "Triticum aestivum",
        "seasons": [{"season": "winter", "planting": "Nov-Dec", "harvest": "Mar-Apr"}],
    },
]


def get_crop_catalogue() -> list[dict]:
    return SUPPORTED_CROPS


def get_crop_names() -> list[str]:
    return [item["name"] for item in SUPPORTED_CROPS]


def get_crop_seasons(crop: str | None = None) -> list[dict]:
    items = SUPPORTED_CROPS
    if crop:
        items = [item for item in items if item["name"].lower() == crop.lower()]
    return [{"crop": item["name"], "seasons": item["seasons"]} for item in items]


def get_model_info() -> dict:
    project_root = Path(getattr(settings, "BASE_DIR", Path(__file__).resolve().parents[3])).parent
    meta_path = project_root / "models" / "training_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
