"""Service layer for crop prediction requests.

This module bridges Django views and the ML pipeline. It keeps views thin and
isolates model loading/real-time feature fetching concerns.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(getattr(settings, "BASE_DIR", Path(__file__).resolve().parents[3])).parent
ML_PIPELINE_ROOT = PROJECT_ROOT
if str(ML_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT))


@lru_cache(maxsize=1)
def get_realtime_service():
    """Return a cached realtime prediction service instance."""
    from ml_pipeline.prediction.realtime import RealtimePredictionService

    logger.info("Initializing realtime prediction service")
    return RealtimePredictionService(initialize_earth_engine=False)


def run_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    """Run real-time prediction for validated request payload."""
    service = get_realtime_service()
    result = service.predict(
        latitude=payload.get("latitude"),
        longitude=payload.get("longitude"),
        date_value=payload["date"],
        district=payload.get("district"),
        crop=payload.get("crop"),
        crops=payload.get("crops"),
        use_realtime=payload.get("use_realtime", True),
    )
    return result
