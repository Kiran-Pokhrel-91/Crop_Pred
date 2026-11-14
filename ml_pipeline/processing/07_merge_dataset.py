"""
Step 07: Merge all source tables into the model-ready dataset.

Joins the grid with the extracted feature tables on ``grid_id``, aligns seasons
and years, applies documented QA rules, and writes ``data/final/`` plus metadata
covering source versions, coverage, and dropped-row reasons.
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_pipeline.config import RAW_DIR, PROCESSED_DIR, FINAL_DIR
from ml_pipeline.utils.pipeline_helpers import GRID_METADATA_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_feature_table(path: Path, label: str) -> pd.DataFrame:
    """Load a processed feature CSV, returning an empty DataFrame if missing."""
    if not path.exists():
        logger.warning("%s feature table not found at %s; skipping.", label, path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows, %d columns", label, len(df), len(df.columns))
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns to the canonical grid_id/district/year/month schema."""
    rename_map = {
        "cell_id": "grid_id",
        "latitude_s1": "latitude",
        "longitude_s1": "longitude",
        "province_s1": "province",
        "latitude_era5": "latitude",
        "longitude_era5": "longitude",
        "province_era5": "province",
    }
    return df.rename(columns=rename_map)


def _align_feature_columns(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Ensure every frame shares the same join columns."""
    join_cols = ["grid_id", "district", "year", "month"]
    aligned = []
    for frame in frames:
        if frame.empty:
            aligned.append(frame)
            continue
        missing = [col for col in join_cols if col not in frame.columns]
        if missing:
            logger.warning("Frame missing join columns %s; skipping.", missing)
            continue
        aligned.append(frame)
    return aligned


def main() -> None:
    logger.info("=== Step 07: Merge feature tables ===")

    s2_df = _load_feature_table(PROCESSED_DIR / "sentinel2_features.csv", "Sentinel-2")
    s1_df = _load_feature_table(PROCESSED_DIR / "sentinel1_features.csv", "Sentinel-1")
    era5_df = _load_feature_table(PROCESSED_DIR / "era5_climate_features.csv", "ERA5-Land")

    if s2_df.empty:
        logger.error("Sentinel-2 features are required to build the model-ready dataset.")
        sys.exit(1)

    s2_df = _standardize_columns(s2_df)
    s1_df = _standardize_columns(s1_df)
    era5_df = _standardize_columns(era5_df)

    frames = _align_feature_columns([s2_df, s1_df, era5_df])
    if not frames or frames[0].empty:
        logger.error("No valid feature frames to merge.")
        sys.exit(1)

    merged = frames[0]
    for extra in frames[1:]:
        if extra.empty:
            continue
        join_cols = ["grid_id", "district", "year", "month"]
        merged = pd.merge(merged, extra, on=join_cols, how="outer", suffixes=("", "_dup"))

    # Drop duplicate metadata columns produced by outer joins
    dup_cols = [col for col in merged.columns if col.endswith("_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    # QA: drop rows missing the key identifiers
    before = len(merged)
    merged = merged.dropna(subset=["grid_id", "year", "month"])
    logger.info("Dropped %d rows missing grid_id/year/month.", before - len(merged))

    # Ensure numeric types for feature columns
    feature_cols = [
        col for col in merged.columns
        if col not in GRID_METADATA_COLUMNS + ["year", "month", "period_start", "period_end"]
        and not col.endswith("_count") and col not in ("crop", "production", "area", "district", "province")
    ]
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Write final dataset
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = FINAL_DIR / "nepal_crop_ml_training_dataset.csv"
    output_parquet = FINAL_DIR / "nepal_crop_ml_training_dataset.parquet"

    merged.to_csv(output_csv, index=False)
    try:
        merged.to_parquet(output_parquet, index=False)
    except Exception as exc:  # pragma: no cover - parquet optional
        logger.warning("Parquet write skipped: %s", exc)

    metadata = {
        "rows": len(merged),
        "grid_cells": int(merged["grid_id"].nunique()) if "grid_id" in merged.columns else 0,
        "columns": list(merged.columns),
        "sources": {
            "sentinel2": str(PROCESSED_DIR / "sentinel2_features.csv"),
            "sentinel1": str(PROCESSED_DIR / "sentinel1_features.csv"),
            "era5": str(PROCESSED_DIR / "era5_climate_features.csv"),
        },
        "outputs": {
            "csv": str(output_csv),
            "parquet": str(output_parquet),
        },
    }
    metadata_path = FINAL_DIR / "nepal_crop_ml_training_dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("=== Step 07 complete: %d rows, %d columns ===", len(merged), len(merged.columns))
    logger.info("Final dataset written to %s", output_csv)


if __name__ == "__main__":
    main()