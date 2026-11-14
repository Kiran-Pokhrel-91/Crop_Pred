"""
Helpers for historical crop statistics, district-name alignment, and the
Open Data Nepal CKAN API.
"""

import io
import logging
import re
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ODN_API_BASE = "https://api.opendatanepal.com/api/3/action"
ODN_PACKAGE_ID = (
    "statistical-information-on-nepalese-agriculture-data-package-"
    "fy-2064-65-2081-82-bs-2007-08"
)

DISTRICT_ALIASES = {
    "CHITWAN": "CHITAWAN",
    "KAVREPALANCHOK": "KABHREPALANCHOK",
    "KAPILVASTU": "KAPILBASTU",
    "MAKWANPUR": "MAKAWANPUR",
    "NAWALPARASI WEST": "PARASI",
    "NAWALPARASIWEST": "PARASI",
    "NAWALPARASI EAST": "NAWALPUR",
    "NAWALPARASIEast": "NAWALPUR",
    "TANAHUN": "TANAHU",
}

CROP_ALIASES = {
    "PADDY (EARLY)": "Paddy",
    "PADDY (MAIN)": "Paddy",
}


def canonical_crop(name: object) -> str:
    """Normalise crop labels and merge early/main paddy into one crop."""
    text = re.sub(r"\s+", " ", str(name).strip())
    return CROP_ALIASES.get(text.upper(), text)


def district_key(name: object) -> str:
    """Normalise a district name for matching across sources."""
    text = str(name).upper().strip()
    text = text.replace(".", " ").replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return DISTRICT_ALIASES.get(text, text)


def build_district_lookup(grid: gpd.GeoDataFrame) -> dict[str, str]:
    """Map normalised district names to the canonical names used by the grid."""
    lookup = {district_key(name): name for name in grid["district"].astype(str)}
    for alias, target in DISTRICT_ALIASES.items():
        if target in lookup:
            lookup[alias] = lookup[target]
    return lookup


def normalise_district_series(
    values: pd.Series, lookup: dict[str, str]
) -> tuple[pd.Series, list[str]]:
    """Return canonical grid district names and any unmatched source values."""
    mapped = values.map(lambda value: lookup.get(district_key(value)))
    unmatched = sorted(set(values[mapped.isna()].astype(str)))
    return mapped, unmatched


def fetch_odn_package(
    package_id: str = ODN_PACKAGE_ID,
    api_base: str = ODN_API_BASE,
    timeout: int = 60,
    attempts: int = 3,
) -> dict:
    """Fetch one CKAN package, retrying transient failures."""
    url = f"{api_base}/package_show"
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params={"id": package_id}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success"):
                raise RuntimeError(f"CKAN returned an error for package {package_id}.")
            return payload["result"]
        except Exception as exc:
            if attempt >= attempts:
                raise
            delay = 2**attempt
            logger.warning(
                "CKAN package request failed (attempt %d/%d): %s. Retrying in %ds.",
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"Could not fetch CKAN package {package_id}.")


def select_district_crop_resources(package: dict) -> dict[str, dict]:
    """Select the 2024/25 district cereal and cash-crop resources."""
    selected = {}
    for resource in package.get("resources", []):
        name = resource.get("name", "")
        if "Cereal Crops by District" in name:
            selected["cereal"] = resource
        elif "Cash Crops by District" in name:
            selected["cash"] = resource

    missing = {"cereal", "cash"}.difference(selected)
    if missing:
        raise RuntimeError(f"ODN package is missing district resources: {sorted(missing)}")
    return selected


def download_odn_csv(
    resource: dict,
    cache_path: Path,
    refresh: bool = False,
    timeout: int = 60,
) -> pd.DataFrame:
    """Download and cache one CKAN CSV resource."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    url = resource.get("url")
    if not url:
        raise RuntimeError(f"ODN resource has no download URL: {resource.get('name')}")

    logger.info("Downloading ODN resource: %s", resource.get("name"))
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    frame.to_csv(cache_path, index=False)
    return frame


def odn_district_crop_table(
    frame: pd.DataFrame,
    year: int,
    source: str,
    lookup: dict[str, str],
) -> pd.DataFrame:
    """Convert one long ODN district table into area, production, and yield rows."""
    required = {"Province", "District", "Crop", "Measure", "Value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"ODN table is missing columns: {sorted(missing)}")

    wide = frame.pivot_table(
        index=["Province", "District", "Crop"],
        columns="Measure",
        values="Value",
        aggfunc="sum",
    ).reset_index()
    for column in ("Area_Ha", "Production_Mt"):
        if column not in wide.columns:
            wide[column] = pd.NA

    wide["Crop"] = wide["Crop"].map(canonical_crop)
    grouped = wide.groupby(["Province", "District", "Crop"], as_index=False)[
        ["Area_Ha", "Production_Mt"]
    ].sum(min_count=1)

    area = pd.to_numeric(grouped["Area_Ha"], errors="coerce")
    production = pd.to_numeric(grouped["Production_Mt"], errors="coerce")
    grouped["yield_kg_ha"] = (production * 1000.0 / area.where(area > 0)).astype(float)

    districts, unmatched = normalise_district_series(grouped["District"], lookup)
    if unmatched:
        logger.warning("Unmatched ODN districts: %s", unmatched)

    grouped["district"] = districts
    grouped["province"] = grouped["Province"]
    grouped["year"] = year
    grouped["crop"] = grouped["Crop"]
    grouped["source"] = source
    grouped = grouped.rename(
        columns={"Area_Ha": "area_ha", "Production_Mt": "production_mt"}
    )

    columns = [
        "district",
        "province",
        "year",
        "crop",
        "area_ha",
        "production_mt",
        "yield_kg_ha",
        "source",
    ]
    return (
        grouped.loc[grouped["district"].notna(), columns]
        .drop_duplicates(["district", "year", "crop"])
        .reset_index(drop=True)
    )


def zenodo_yield_table(
    frame: pd.DataFrame,
    start_year: int,
    end_year: int,
    lookup: dict[str, str],
) -> pd.DataFrame:
    """Convert the wide Zenodo maize/wheat panel into long yield rows."""
    required = {"district", "province", "year", "maize_yield_kg_ha", "wheat_yield_kg_ha"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Zenodo panel is missing columns: {sorted(missing)}")

    frame = frame[frame["year"].between(start_year, end_year)].copy()
    rows = []
    for crop, column in (
        ("Maize", "maize_yield_kg_ha"),
        ("Wheat", "wheat_yield_kg_ha"),
    ):
        subset = frame[["district", "province", "year", column]].dropna(
            subset=[column]
        )
        subset = subset.rename(columns={column: "yield_kg_ha"})
        subset["crop"] = crop
        subset["area_ha"] = pd.NA
        subset["production_mt"] = pd.NA
        subset["source"] = "Zenodo_1981_2023_panel"
        rows.append(subset)

    long = pd.concat(rows, ignore_index=True)
    districts, unmatched = normalise_district_series(long["district"], lookup)
    if unmatched:
        logger.warning("Unmatched Zenodo districts: %s", unmatched)

    long["district"] = districts
    return (
        long.loc[long["district"].notna()]
        .drop_duplicates(["district", "year", "crop"])
        .reset_index(drop=True)
    )
