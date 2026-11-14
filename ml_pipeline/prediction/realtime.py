"""Real-time feature collection and prediction for a point/date request.

Fetches recent Sentinel-2, Sentinel-1, and ERA5-Land features from Google Earth
Engine for the grid cell nearest to the requested location, builds the model
input contract, and runs the standalone crop-yield predictor.

This module is independent of Django and returns plain Python data structures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.config import MODELS_DIR, RAW_DIR
from ml_pipeline.prediction.predictor import CropYieldPredictor
from ml_pipeline.utils.gee_helpers import (
    ERA5_FEATURE_COLUMNS,
    S1_FEATURE_BANDS,
    S2_FEATURE_COLUMNS,
    build_era5_month_composite,
    build_s1_month_composite,
    build_s2_month_composite,
    grid_to_feature_collection,
    initialize_gee,
    make_feature_reducer,
)

logger = logging.getLogger(__name__)

SUPPORTED_CROPS = [
    "Barley",
    "Buckwheat",
    "Maize",
    "Millet",
    "Oilseed",
    "Paddy",
    "Potato",
    "Sugarcane",
    "Wheat",
]

NEPAL_BOUNDS = {
    "lat_min": 26.3,
    "lat_max": 30.5,
    "lon_min": 80.0,
    "lon_max": 88.3,
}

# Plausible yield bounds (kg/ha) used to keep predictions realistic when features are uncertain.
CROP_YIELD_BOUNDS = {
    "Barley": (400, 4000),
    "Buckwheat": (400, 2500),
    "Maize": (800, 9000),
    "Millet": (500, 2500),
    "Oilseed": (300, 2500),
    "Paddy": (1200, 7000),
    "Potato": (6000, 35000),
    "Sugarcane": (20000, 90000),
    "Wheat": (800, 7000),
}

CROP_SEASONS = {
    "Barley": [{"season": "winter", "months": [10, 11, 12, 1, 2, 3, 4, 5], "planting": "Oct-Nov", "harvest": "Apr-May"}],
    "Buckwheat": [{"season": "autumn", "months": [8, 9, 10, 11, 12], "planting": "Aug-Sep", "harvest": "Nov-Dec"}],
    "Maize": [
        {"season": "summer", "months": [2, 3, 4, 5, 6, 7, 8], "planting": "Feb-Apr", "harvest": "Jun-Aug"},
        {"season": "winter-hills", "months": [9, 10, 11, 12, 1, 2], "planting": "Sep-Oct", "harvest": "Jan-Feb"},
    ],
    "Millet": [{"season": "summer", "months": [6, 7, 8, 9, 10, 11], "planting": "Jun-Jul", "harvest": "Oct-Nov"}],
    "Oilseed": [{"season": "winter", "months": [9, 10, 11, 12, 1, 2, 3], "planting": "Sep-Oct", "harvest": "Feb-Mar"}],
    "Paddy": [{"season": "monsoon", "months": [6, 7, 8, 9, 10, 11], "planting": "Jun-Jul", "harvest": "Oct-Nov"}],
    "Potato": [{"season": "winter", "months": [9, 10, 11, 12, 1, 2], "planting": "Sep-Oct", "harvest": "Jan-Feb"}],
    "Sugarcane": [{"season": "annual", "months": list(range(1, 13)), "planting": "Feb-Mar", "harvest": "Dec-Mar"}],
    "Wheat": [{"season": "winter", "months": [11, 12, 1, 2, 3, 4], "planting": "Nov-Dec", "harvest": "Mar-Apr"}],
}


@dataclass
class GridCell:
    grid_id: str
    district: str
    province: str
    latitude: float
    longitude: float
    distance_km: float


class RealtimePredictionService:
    """Resolve a location/date to real-time features and yield predictions."""

    def __init__(
        self,
        grid_path: Path | None = None,
        model_path: Path | None = None,
        initialize_earth_engine: bool = True,
    ) -> None:
        self.grid_path = grid_path or (RAW_DIR / "grid.csv")
        self.grid = self._load_grid(self.grid_path)
        self.predictor = CropYieldPredictor(model_path=model_path or (MODELS_DIR / "best_crop_model.joblib"))
        self.metadata = self._load_metadata()
        self.crop_stats, self.district_crop_stats = self._load_crop_stats()
        self._gee_initialized = False
        if initialize_earth_engine:
            self.ensure_gee()

    def _load_grid(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Grid file not found: {path}")
        grid = pd.read_csv(path)
        required = {"grid_id", "district", "province", "latitude", "longitude"}
        missing = required.difference(grid.columns)
        if missing:
            raise ValueError(f"Grid is missing required columns: {sorted(missing)}")
        grid["grid_id"] = grid["grid_id"].astype(str)
        return grid

    def _load_metadata(self) -> dict[str, Any]:
        meta_path = MODELS_DIR / "training_metadata.json"
        if meta_path.exists():
            import json

            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {}

    def _load_crop_stats(self) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
        """Load historical crop yield stats for suitability and realistic clamping."""
        from ml_pipeline.config import FINAL_DIR

        crop_stats: dict[str, dict[str, float]] = {}
        district_stats: dict[tuple[str, str], dict[str, float]] = {}
        path = FINAL_DIR / "nepal_crop_ml_training_dataset.csv"
        if not path.exists():
            return crop_stats, district_stats
        try:
            df = pd.read_csv(path, usecols=["district", "crop", "yield_kg_ha"])
            df = df.dropna(subset=["district", "crop", "yield_kg_ha"])
            crop_grp = df.groupby("crop")["yield_kg_ha"]
            for crop, series in crop_grp:
                bounds = CROP_YIELD_BOUNDS.get(crop)
                low = float(series.quantile(0.05))
                high = float(series.quantile(0.95))
                if bounds:
                    low = max(low, bounds[0] * 0.7)
                    high = min(high, bounds[1] * 1.1)
                crop_stats[str(crop)] = {
                    "mean": float(series.mean()),
                    "p05": low,
                    "p95": high,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            dc_grp = df.groupby(["district", "crop"])["yield_kg_ha"].agg(["count", "mean", "min", "max"]).reset_index()
            for _, row in dc_grp.iterrows():
                key = (str(row["district"]).strip().lower(), str(row["crop"]))
                district_stats[key] = {
                    "count": float(row["count"]),
                    "mean": float(row["mean"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
        except Exception as exc:
            logger.warning("Failed to load crop stats: %s", exc)
        return crop_stats, district_stats

    def ensure_gee(self) -> None:
        if self._gee_initialized:
            return
        initialize_gee()
        self._gee_initialized = True

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371.0088
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
        return 2 * radius * math.asin(math.sqrt(a))

    def resolve_grid_cell(self, latitude: float, longitude: float) -> GridCell:
        lats = self.grid["latitude"].to_numpy(dtype=float)
        lons = self.grid["longitude"].to_numpy(dtype=float)
        # Fast approximate distance then exact haversine for nearest candidates.
        approx = (lats - latitude) ** 2 + (lons - longitude) ** 2
        idx = int(np.argmin(approx))
        # refine among nearest 50 by euclidean in degrees
        nearest_idx = np.argsort(approx)[:50]
        best = None
        for i in nearest_idx:
            row = self.grid.iloc[int(i)]
            dist = self._haversine_km(latitude, longitude, float(row["latitude"]), float(row["longitude"]))
            if best is None or dist < best.distance_km:
                best = GridCell(
                    grid_id=str(row["grid_id"]),
                    district=str(row["district"]),
                    province=str(row["province"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    distance_km=dist,
                )
        assert best is not None
        return best

    @staticmethod
    def validate_location(latitude: float, longitude: float) -> None:
        if not (NEPAL_BOUNDS["lat_min"] <= latitude <= NEPAL_BOUNDS["lat_max"]):
            raise ValueError("latitude must be within Nepal bounds (26.3 to 30.5).")
        if not (NEPAL_BOUNDS["lon_min"] <= longitude <= NEPAL_BOUNDS["lon_max"]):
            raise ValueError("longitude must be within Nepal bounds (80.0 to 88.3).")

    def resolve_district_location(self, district: str) -> dict[str, Any]:
        if not district:
            raise ValueError("district is required")
        subset = self.grid[self.grid["district"].str.lower() == district.strip().lower()]
        if subset.empty:
            raise ValueError(f"Unknown district: {district}")
        lat = float(subset["latitude"].mean())
        lon = float(subset["longitude"].mean())
        province = str(subset["province"].iloc[0])
        district_name = str(subset["district"].iloc[0])
        return {
            "district": district_name,
            "province": province,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "grid_cells": int(len(subset)),
        }

    @staticmethod
    def parse_date(value: str | date | datetime) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()

    @staticmethod
    def month_window(target: date) -> tuple[date, date, date]:
        start = date(target.year, target.month, 1)
        if start.month == 12:
            next_month = date(start.year + 1, 1, 1)
        else:
            next_month = date(start.year, start.month + 1, 1)
        end_inclusive = next_month - timedelta(days=1)
        return start, end_inclusive, next_month

    def _point_feature_collection(self, latitude: float, longitude: float, grid_id: str):
        import ee

        point = ee.Geometry.Point([longitude, latitude])
        return ee.FeatureCollection([ee.Feature(point, {"grid_id": grid_id})])

    def _reduce_image(self, image, latitude: float, longitude: float, grid_id: str, scale: float) -> dict[str, Any]:
        import ee

        fc = self._point_feature_collection(latitude, longitude, grid_id)
        reduced = image.reduceRegions(
            collection=fc,
            reducer=make_feature_reducer(),
            scale=scale,
            tileScale=4,
        )
        info = reduced.getInfo()
        features = info.get("features", [])
        if not features:
            return {}
        return features[0].get("properties", {}) or {}

    @staticmethod
    def _extract_mean(props: dict[str, Any], band: str) -> float | None:
        value = props.get(f"{band}_mean")
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return float(value)

    def fetch_sentinel2(self, latitude: float, longitude: float, grid_id: str, target: date) -> tuple[dict[str, Any], dict[str, Any]]:
        start, _, exclusive_end = self.month_window(target)
        image, count = build_s2_month_composite(
            aoi=self._point_feature_collection(latitude, longitude, grid_id).geometry(),
            start_date=start.isoformat(),
            end_date=exclusive_end.isoformat(),
        )
        meta = {"source": "Sentinel-2", "image_count": int(count), "period_start": start.isoformat()}
        if image is None:
            return {col: None for col in S2_FEATURE_COLUMNS}, {**meta, "status": "missing"}
        props = self._reduce_image(image, latitude, longitude, grid_id, scale=10.0)
        values = {col: self._extract_mean(props, col.replace("_mean", "")) for col in S2_FEATURE_COLUMNS}
        status = "ok" if any(v is not None for v in values.values()) else "empty"
        return values, {**meta, "status": status}

    def fetch_sentinel1(self, latitude: float, longitude: float, grid_id: str, target: date) -> tuple[dict[str, Any], dict[str, Any]]:
        start, _, exclusive_end = self.month_window(target)
        image, count = build_s1_month_composite(
            aoi=self._point_feature_collection(latitude, longitude, grid_id).geometry(),
            start_date=start.isoformat(),
            end_date=exclusive_end.isoformat(),
        )
        meta = {"source": "Sentinel-1", "image_count": int(count), "period_start": start.isoformat()}
        if image is None:
            return {col: None for col in S1_FEATURE_BANDS}, {**meta, "status": "missing"}
        props = self._reduce_image(image, latitude, longitude, grid_id, scale=10.0)
        # S1 internal bands map to output names by position in composite; reduce gives *_mean
        internal = ("VVm", "VVmed", "VVs", "VHm", "VHmed", "VHs", "VVVH", "VHVV")
        values = {}
        for internal_name, output_name in zip(internal, S1_FEATURE_BANDS):
            values[output_name] = self._extract_mean(props, internal_name)
        status = "ok" if any(v is not None for v in values.values()) else "empty"
        return values, {**meta, "status": status}

    def fetch_era5(self, latitude: float, longitude: float, grid_id: str, target: date) -> tuple[dict[str, Any], dict[str, Any]]:
        start, _, exclusive_end = self.month_window(target)
        image, count = build_era5_month_composite(
            aoi=self._point_feature_collection(latitude, longitude, grid_id).geometry(),
            start_date=start.isoformat(),
            end_date=exclusive_end.isoformat(),
        )
        meta = {"source": "ERA5-Land", "image_count": int(count), "period_start": start.isoformat()}
        if image is None:
            return {col: None for col in ERA5_FEATURE_COLUMNS}, {**meta, "status": "missing"}
        props = self._reduce_image(image, latitude, longitude, grid_id, scale=11132.0)
        values = {col: self._extract_mean(props, col.replace("_mean", "")) for col in ERA5_FEATURE_COLUMNS}
        status = "ok" if any(v is not None for v in values.values()) else "empty"
        return values, {**meta, "status": status}

    @staticmethod
    def _quality_score(sources: dict[str, dict[str, Any]], feature_values: dict[str, Any]) -> dict[str, Any]:
        expected = len(S2_FEATURE_COLUMNS) + len(S1_FEATURE_BANDS) + len(ERA5_FEATURE_COLUMNS)
        available = sum(1 for k in [*S2_FEATURE_COLUMNS, *S1_FEATURE_BANDS, *ERA5_FEATURE_COLUMNS] if feature_values.get(k) is not None)
        coverage = available / expected if expected else 0.0
        per_source = {}
        for name, meta in sources.items():
            per_source[name] = {
                "status": meta.get("status"),
                "image_count": meta.get("image_count", 0),
            }
        if coverage >= 0.9:
            level = "high"
        elif coverage >= 0.6:
            level = "medium"
        else:
            level = "low"
        return {
            "feature_coverage": round(coverage, 4),
            "available_features": available,
            "expected_features": expected,
            "level": level,
            "sources": per_source,
        }

    def _health_from_ndvi(self, ndvi: float | None) -> dict[str, Any]:
        if ndvi is None or (isinstance(ndvi, float) and math.isnan(ndvi)):
            return {"label": "unknown", "score": None, "reason": "NDVI unavailable"}
        if ndvi >= 0.6:
            label, score = "excellent", 0.95
        elif ndvi >= 0.4:
            label, score = "good", 0.8
        elif ndvi >= 0.2:
            label, score = "moderate", 0.6
        elif ndvi >= 0.1:
            label, score = "poor", 0.35
        else:
            label, score = "stressed", 0.15
        return {"label": label, "score": score, "ndvi": round(float(ndvi), 4)}

    def _confidence(self, crop: str, quality: dict[str, Any]) -> float:
        per_crop = {m.get("Crop"): m for m in self.metadata.get("per_crop_metrics", []) if m.get("Crop")}
        cv = self.metadata.get("cv_metrics", [{}])
        best_cv_r2 = None
        for item in cv:
            if item.get("Selected") or best_cv_r2 is None:
                best_cv_r2 = item.get("CV_R2_Mean", best_cv_r2)
        crop_r2 = None
        if crop in per_crop:
            crop_r2 = per_crop[crop].get("R2")
        model_component = 0.5
        if isinstance(best_cv_r2, (int, float)):
            model_component = max(0.0, min(1.0, float(best_cv_r2)))
        crop_component = model_component
        if isinstance(crop_r2, (int, float)):
            crop_component = max(0.0, min(1.0, (float(crop_r2) + 1.0) / 2.0))
        coverage = float(quality.get("feature_coverage", 0.0))
        confidence = 0.5 * coverage + 0.3 * model_component + 0.2 * crop_component
        return round(max(0.05, min(0.99, confidence)), 4)

    def _season_info(self, crop: str, month: int) -> dict[str, Any]:
        seasons = CROP_SEASONS.get(crop, [])
        in_season = False
        matched = None
        for s in seasons:
            if month in s.get("months", []):
                in_season = True
                matched = s
                break
        if not seasons:
            return {"in_season": None, "level": "unknown", "note": "Season data unavailable"}
        if in_season and matched:
            return {
                "in_season": True,
                "level": "good",
                "season": matched.get("season"),
                "planting": matched.get("planting"),
                "harvest": matched.get("harvest"),
                "note": f"In typical {matched.get('season')} season window for this crop.",
            }
        # nearest info
        first = seasons[0]
        return {
            "in_season": False,
            "level": "low",
            "season": first.get("season"),
            "planting": first.get("planting"),
            "harvest": first.get("harvest"),
            "note": "Selected date is outside typical planting/harvest window.",
        }

    def _suitability(self, district: str, crop: str) -> dict[str, Any]:
        key = (district.strip().lower(), crop)
        hist = self.district_crop_stats.get(key)
        if hist:
            # More historical records => stronger local evidence.
            evidence = min(1.0, hist["count"] / 20.0)
            score = 0.7 + 0.3 * evidence
            return {
                "score": round(score, 4),
                "level": "high" if score >= 0.85 else "medium",
                "historical_records": int(hist["count"]),
                "historical_avg_yield_kg_ha": round(hist["mean"], 2),
                "reason": "Crop has historical records in this district.",
            }
        return {
            "score": 0.2,
            "level": "low",
            "historical_records": 0,
            "historical_avg_yield_kg_ha": None,
            "reason": "No historical ground-truth records for this crop in the selected district.",
        }

    def _clamp_yield(self, crop: str, district: str, predicted: float, coverage: float) -> tuple[float, dict[str, Any]]:
        bounds = CROP_YIELD_BOUNDS.get(crop, (0.0, float("inf")))
        hard_low, hard_high = float(bounds[0]), float(bounds[1])
        stats = self.crop_stats.get(crop, {})
        low = float(stats.get("p05", hard_low))
        high = float(stats.get("p95", hard_high))
        low = max(low * 0.85, hard_low * 0.6)
        high = min(high * 1.15, hard_high * 1.15)

        hist = self.district_crop_stats.get((district.strip().lower(), crop))
        anchor = None
        if hist:
            anchor = float(hist["mean"])
            # Keep within a reasonable band around local history when available.
            low = max(low, anchor * 0.35)
            high = min(high, anchor * 2.2)
        elif coverage < 0.5:
            # For districts/cases with weak evidence and missing realtime features, stay near national typical.
            anchor = float(stats.get("mean", predicted))
            low = max(low, anchor * 0.5)
            high = min(high, anchor * 1.8)

        adjusted = min(max(float(predicted), low), high)
        return round(adjusted, 3), {
            "raw_predicted_yield_kg_ha": round(float(predicted), 3),
            "adjusted": abs(adjusted - float(predicted)) > 1e-6,
            "clamp_low_kg_ha": round(low, 3),
            "clamp_high_kg_ha": round(high, 3),
            "historical_anchor_kg_ha": None if anchor is None else round(anchor, 3),
        }

    def predict(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        date_value: str | date | datetime = "",
        district: str | None = None,
        crop: str | None = None,
        crops: list[str] | None = None,
        use_realtime: bool = True,
    ) -> dict[str, Any]:
        district_info = None
        if district and (latitude is None or longitude is None):
            district_info = self.resolve_district_location(district)
            latitude = district_info["latitude"]
            longitude = district_info["longitude"]
        if latitude is None or longitude is None:
            raise ValueError("Provide a district or both latitude and longitude.")
        self.validate_location(float(latitude), float(longitude))
        target = self.parse_date(date_value)
        cell = self.resolve_grid_cell(float(latitude), float(longitude))
        start, end_inclusive, _ = self.month_window(target)

        selected_crops = crops or ([crop] if crop else SUPPORTED_CROPS)
        unknown = [c for c in selected_crops if c not in SUPPORTED_CROPS]
        if unknown:
            raise ValueError(f"Unsupported crop(s): {', '.join(unknown)}")

        if use_realtime:
            self.ensure_gee()
            s2_vals, s2_meta = self.fetch_sentinel2(cell.latitude, cell.longitude, cell.grid_id, target)
            s1_vals, s1_meta = self.fetch_sentinel1(cell.latitude, cell.longitude, cell.grid_id, target)
            era5_vals, era5_meta = self.fetch_era5(cell.latitude, cell.longitude, cell.grid_id, target)
            sources_meta = {"sentinel2": s2_meta, "sentinel1": s1_meta, "era5": era5_meta}
        else:
            s2_vals = {col: None for col in S2_FEATURE_COLUMNS}
            s1_vals = {col: None for col in S1_FEATURE_BANDS}
            era5_vals = {col: None for col in ERA5_FEATURE_COLUMNS}
            sources_meta = {"sentinel2": {"status": "skipped"}, "sentinel1": {"status": "skipped"}, "era5": {"status": "skipped"}}

        feature_values = {
            "latitude": cell.latitude,
            "longitude": cell.longitude,
            **s2_vals,
            **s1_vals,
            **era5_vals,
        }
        quality = self._quality_score(sources_meta, feature_values)

        # Fill missing model features with neutral defaults only at inference edge.
        # Defaults use dataset-like annual/monthly aggregate scales to avoid unrealistic extrapolation.
        defaults = {
            "B2_mean": 0.05, "B3_mean": 0.07, "B4_mean": 0.07, "B8_mean": 0.25,
            "B11_mean": 0.2, "B12_mean": 0.13, "NDVI_mean": 0.46, "EVI_mean": 0.34,
            "NDWI_mean": -0.57, "SAVI_mean": 0.33, "MSAVI_mean": 0.30, "GNDVI_mean": 0.57, "NDBI_mean": -0.11,
            "VV_mean": -8.3, "VV_median": -8.3, "VV_std": 1.6, "VH_mean": -15.3, "VH_median": -15.3,
            "VH_std": 1.7, "VV_VH_mean": 4.8, "VH_VV_mean": 0.23,
            "temperature_c_mean": 11.5, "rainfall_mm_mean": 2180.0, "soil_moisture_mean": 0.31,
            "relative_humidity_mean": 71.4, "solar_radiation_mean": 4800.0, "wind_speed_mean": 0.48,
        }
        filled = dict(feature_values)
        for key, default in defaults.items():
            if filled.get(key) is None or (isinstance(filled.get(key), float) and math.isnan(filled[key])):
                filled[key] = default

        records = []
        for c in selected_crops:
            rec = dict(filled)
            rec["crop"] = c
            records.append(rec)

        preds = self.predictor.predict(records)
        ndvi = feature_values.get("NDVI_mean")
        health = self._health_from_ndvi(ndvi)
        coverage = float(quality.get("feature_coverage", 0.0))
        predictions = []
        for item in preds:
            c = item["crop"]
            conf = self._confidence(c, quality)
            suitability = self._suitability(cell.district, c)
            season = self._season_info(c, target.month)
            adjusted_yield, yield_info = self._clamp_yield(c, cell.district, float(item["predicted_yield_kg_ha"]), coverage)

            suitability_factor = float(suitability["score"])
            season_factor = 1.0 if season.get("in_season") else 0.85 if season.get("in_season") is False else 0.95
            recommendation_score = conf * (0.2 + 0.65 * suitability_factor + 0.15 * season_factor)
            if quality.get("level") == "low":
                recommendation_score *= 0.9
            recommendation_score = round(max(0.0, min(0.99, recommendation_score)), 4)

            # simple production estimate context (indicative, per hectare)
            estimated_production_note = "Estimated yield per hectare based on satellite/climate features and historical patterns."

            reasons = []
            if suitability["historical_records"] == 0:
                reasons.append("No historical cultivation records in this district")
            if season.get("in_season") is False:
                reasons.append("Outside typical seasonal window for selected date")
            elif season.get("in_season") is True:
                reasons.append("Within typical seasonal window")
            if yield_info["adjusted"]:
                reasons.append("Yield adjusted to realistic range for available evidence")
            if not reasons:
                reasons.append("Suitable crop with historical presence in district")

            predictions.append(
                {
                    "crop": c,
                    "predicted_yield_kg_ha": adjusted_yield,
                    "yield_ton_ha": round(adjusted_yield / 1000.0, 4),
                    "confidence": conf,
                    "health": {**health, "ndvi": ndvi if ndvi is None else round(float(ndvi), 4)},
                    "model": item.get("model", self.predictor.model_name),
                    "suitability": suitability,
                    "season": season,
                    "recommendation_score": recommendation_score,
                    "yield_adjustment": yield_info,
                    "recommendation_reason": "; ".join(reasons),
                    "estimated_production_note": estimated_production_note,
                }
            )
        predictions.sort(key=lambda x: (x["recommendation_score"], x["suitability"]["score"], x["confidence"]), reverse=True)
        recommended = predictions[0] if predictions else None

        request_info = {
            "date": target.isoformat(),
        }
        if district_info:
            request_info["district"] = district_info["district"]
        if latitude is not None:
            request_info["latitude"] = float(latitude)
        if longitude is not None:
            request_info["longitude"] = float(longitude)

        location = {
            "grid_id": cell.grid_id,
            "district": cell.district,
            "province": cell.province,
            "latitude": cell.latitude,
            "longitude": cell.longitude,
            "distance_km": round(cell.distance_km, 4),
        }
        if district_info:
            location["selected_district"] = district_info["district"]
            location["selected_province"] = district_info["province"]
            location["district_centroid"] = {
                "latitude": district_info["latitude"],
                "longitude": district_info["longitude"],
                "grid_cells": district_info["grid_cells"],
            }

        return {
            "status": "success",
            "request": request_info,
            "location": location,
            "period": {
                "year": target.year,
                "month": target.month,
                "start_date": start.isoformat(),
                "end_date": end_inclusive.isoformat(),
            },
            "data_quality": quality,
            "sources": sources_meta,
            "features": {k: (None if v is None else float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in feature_values.items() if k not in {"latitude", "longitude"}},
            "predictions": predictions,
            "recommended_crop": recommended,
            "model": {
                "name": self.predictor.model_name,
                "target": self.predictor.target,
                "target_transform": self.predictor.target_transform,
                "trained_on_rows": self.metadata.get("usable_rows"),
                "validation": self.metadata.get("validation"),
            },
        }
