"""Services for grid and satellite feature lookups from pipeline outputs."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from django.conf import settings


def _project_root() -> Path:
    return Path(getattr(settings, "BASE_DIR", Path(__file__).resolve().parents[3])).parent


@lru_cache(maxsize=1)
def load_grid() -> pd.DataFrame:
    path = _project_root() / "data" / "raw" / "grid.csv"
    if not path.exists():
        return pd.DataFrame(columns=["grid_id", "district", "province", "latitude", "longitude"])
    grid = pd.read_csv(path)
    grid["grid_id"] = grid["grid_id"].astype(str)
    return grid


def list_districts() -> list[dict[str, Any]]:
    grid = load_grid()
    if grid.empty:
        return []
    grouped = (
        grid.groupby(["province", "district"], as_index=False)
        .agg(grid_cells=("grid_id", "count"), latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .sort_values(["province", "district"])
    )
    return grouped.to_dict("records")


def get_grid_cell(grid_id: str) -> dict[str, Any] | None:
    grid = load_grid()
    row = grid[grid["grid_id"] == str(grid_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def find_nearest_cell(latitude: float, longitude: float) -> dict[str, Any] | None:
    grid = load_grid()
    if grid.empty:
        return None
    lats = grid["latitude"].to_numpy(dtype=float)
    lons = grid["longitude"].to_numpy(dtype=float)
    approx = (lats - latitude) ** 2 + (lons - longitude) ** 2
    idx = int(approx.argmin())
    row = grid.iloc[idx].to_dict()
    row["grid_id"] = str(row["grid_id"])
    # haversine
    r = 6371.0088
    p1, p2 = math.radians(latitude), math.radians(float(row["latitude"]))
    dphi = math.radians(float(row["latitude"]) - latitude)
    dl = math.radians(float(row["longitude"]) - longitude)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    row["distance_km"] = round(2 * r * math.asin(math.sqrt(a)), 4)
    return row


@lru_cache(maxsize=1)
def load_processed_features() -> dict[str, pd.DataFrame]:
    root = _project_root() / "data" / "processed"
    mapping = {
        "sentinel2": root / "sentinel2_features.csv",
        "sentinel1": root / "sentinel1_features.csv",
        "era5": root / "era5_climate_features.csv",
    }
    data = {}
    for key, path in mapping.items():
        if path.exists():
            df = pd.read_csv(path)
            df["grid_id"] = df["grid_id"].astype(str)
            data[key] = df
        else:
            data[key] = pd.DataFrame()
    return data


def get_cell_features(grid_id: str, year: int | None = None, month: int | None = None) -> dict[str, Any]:
    features = load_processed_features()
    result = {}
    for source, df in features.items():
        if df.empty:
            result[source] = []
            continue
        subset = df[df["grid_id"] == str(grid_id)]
        if year is not None and "year" in subset.columns:
            subset = subset[subset["year"] == year]
        if month is not None and "month" in subset.columns:
            subset = subset[subset["month"] == month]
        subset = subset.sort_values([c for c in ["year", "month"] if c in subset.columns])
        # limit payload
        result[source] = subset.head(24).where(pd.notnull(subset), None).to_dict("records")
    return result


@lru_cache(maxsize=1)
def load_final_dataset() -> pd.DataFrame:
    path = _project_root() / "data" / "final" / "nepal_crop_ml_training_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # keep minimal columns for dashboard
    keep = [c for c in [
        "district", "province", "year", "crop", "area_ha", "production_mt", "yield_kg_ha",
        "NDVI_mean", "temperature_c_mean", "rainfall_mm_mean", "soil_moisture_mean",
        "relative_humidity_mean", "solar_radiation_mean"
    ] if c in df.columns]
    return df[keep].copy()


def get_district_dashboard(district: str) -> dict[str, Any] | None:
    df = load_final_dataset()
    grid = load_grid()
    if df.empty:
        return None
    dname = district.strip().lower()
    subset = df[df["district"].astype(str).str.lower() == dname]
    if subset.empty:
        # try match from grid for basic info
        g = grid[grid["district"].astype(str).str.lower() == dname]
        if g.empty:
            return None
        return {
            "district": str(g.iloc[0]["district"]),
            "province": str(g.iloc[0]["province"]),
            "grid_cells": int(len(g)),
            "latest_year": None,
            "current_cultivation": [],
            "environment_status": {},
            "trends": [],
            "message": "No cultivation records available for this district in the dataset.",
        }

    district_name = str(subset.iloc[0]["district"])
    province = str(subset.iloc[0].get("province", ""))
    latest_year = int(subset["year"].max())
    latest = subset[subset["year"] == latest_year]

    # crop records (dedupe district-year-crop)
    crop_cols = [c for c in ["district", "year", "crop", "area_ha", "production_mt", "yield_kg_ha"] if c in latest.columns]
    crops = latest[crop_cols].drop_duplicates(subset=["crop"]).copy()
    crops = crops.sort_values(by=["area_ha", "production_mt", "yield_kg_ha"], ascending=[False, False, False], na_position="last")
    crop_records = []
    for _, row in crops.iterrows():
        rec = {"crop": str(row.get("crop"))}
        for col in ["area_ha", "production_mt", "yield_kg_ha"]:
            if col in row and pd.notna(row[col]):
                rec[col] = round(float(row[col]), 3)
            else:
                rec[col] = None
        if rec.get("yield_kg_ha") is not None:
            rec["yield_ton_ha"] = round(rec["yield_kg_ha"] / 1000.0, 4)
        crop_records.append(rec)

    # environmental/status indicators from latest year (means)
    env = {}
    for col, label in [
        ("NDVI_mean", "ndvi_mean"),
        ("temperature_c_mean", "temperature_c_mean"),
        ("rainfall_mm_mean", "rainfall_mm_mean"),
        ("soil_moisture_mean", "soil_moisture_mean"),
        ("relative_humidity_mean", "relative_humidity_mean"),
        ("solar_radiation_mean", "solar_radiation_mean"),
    ]:
        if col in latest.columns:
            val = latest[col].dropna().mean()
            if pd.notna(val):
                env[label] = round(float(val), 4)

    # simple vegetation status from NDVI
    ndvi = env.get("ndvi_mean")
    if ndvi is None:
        veg_status = {"label": "unknown", "note": "NDVI not available"}
    elif ndvi >= 0.6:
        veg_status = {"label": "excellent", "note": "Dense/healthy vegetation signal"}
    elif ndvi >= 0.4:
        veg_status = {"label": "good", "note": "Good vegetation cover"}
    elif ndvi >= 0.2:
        veg_status = {"label": "moderate", "note": "Moderate vegetation cover"}
    elif ndvi >= 0.1:
        veg_status = {"label": "poor", "note": "Sparse vegetation cover"}
    else:
        veg_status = {"label": "stressed", "note": "Very low vegetation signal"}
    env["vegetation_status"] = veg_status

    # yearly trends
    trend_cols = [c for c in ["year", "crop", "area_ha", "production_mt", "yield_kg_ha"] if c in subset.columns]
    trend_df = subset[trend_cols].drop_duplicates(subset=["year", "crop"]).sort_values(["year", "crop"])
    trends = []
    for _, row in trend_df.iterrows():
        rec = {"year": int(row["year"]), "crop": str(row["crop"])}
        for col in ["area_ha", "production_mt", "yield_kg_ha"]:
            if col in row and pd.notna(row[col]):
                rec[col] = round(float(row[col]), 3)
            else:
                rec[col] = None
        trends.append(rec)

    # summary totals for latest year
    summary = {
        "crop_count": len(crop_records),
        "total_area_ha": None,
        "total_production_mt": None,
    }
    if "area_ha" in crops.columns:
        val = crops["area_ha"].dropna().sum()
        summary["total_area_ha"] = round(float(val), 3) if pd.notna(val) and len(crops["area_ha"].dropna()) else None
    if "production_mt" in crops.columns:
        val = crops["production_mt"].dropna().sum()
        summary["total_production_mt"] = round(float(val), 3) if pd.notna(val) and len(crops["production_mt"].dropna()) else None

    grid_cells = int((grid["district"].astype(str).str.lower() == dname).sum()) if not grid.empty else None

    return {
        "district": district_name,
        "province": province,
        "grid_cells": grid_cells,
        "latest_year": latest_year,
        "years_available": sorted(int(y) for y in subset["year"].dropna().unique().tolist()),
        "summary": summary,
        "current_cultivation": crop_records,
        "environment_status": env,
        "trends": trends,
        "data_note": "Cultivation figures are based on latest available district records in the dataset (estimated/recorded values, not real-time field verification).",
    }
