"""Standalone inference for the trained crop-yield model."""

from pathlib import Path

import joblib
import pandas as pd

from ml_pipeline.config import MODELS_DIR


class CropYieldPredictor:
    """Load the persisted model artifact and predict crop yield."""

    def __init__(self, model_path: Path | None = None):
        path = model_path or (MODELS_DIR / "best_crop_model.joblib")
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.model_name = artifact.get("model_name", "unknown")
        self.features = artifact["features"]
        self.target = artifact.get("target", "yield_kg_ha")
        self.target_transform = artifact.get("target_transform")

    def predict(self, records: list[dict] | pd.DataFrame) -> list[dict]:
        """Return predictions for input records."""
        df = pd.DataFrame(records) if isinstance(records, list) else records.copy()
        missing = [col for col in self.features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        preds = self.model.predict(df[self.features])
        return [
            {
                "crop": row.get("crop"),
                "predicted_yield_kg_ha": float(pred),
                "model": self.model_name,
            }
            for row, pred in zip(df.to_dict("records"), preds)
        ]

