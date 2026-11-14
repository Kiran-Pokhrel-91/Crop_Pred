"""
Google Earth Engine helpers for authentication, cloud masking, optical and SAR
feature calculation, batched monthly composites, and zonal statistics.
"""

import logging
import time
from pathlib import Path
from typing import Callable, TypeVar

import ee
import geopandas as gpd
import pandas as pd

from ml_pipeline.config import (
    BASE_DIR,
    GEE_PRIVATE_KEY_PATH,
    GEE_PROJECT_ID,
    GEE_SERVICE_ACCOUNT,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

S2_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
S2_REFLECTANCE_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
S2_INDEX_BANDS = ("NDVI", "EVI", "NDWI", "SAVI", "MSAVI", "GNDVI", "NDBI")
S2_FEATURE_BANDS = (*S2_REFLECTANCE_BANDS, *S2_INDEX_BANDS)
S2_FEATURE_COLUMNS = tuple(f"{band}_mean" for band in S2_FEATURE_BANDS)

S1_COLLECTION_ID = "COPERNICUS/S1_GRD"
S1_INTERNAL_BANDS = ("VVm", "VVmed", "VVs", "VHm", "VHmed", "VHs", "VVVH", "VHVV")
S1_FEATURE_BANDS = (
    "VV_mean",
    "VV_median",
    "VV_std",
    "VH_mean",
    "VH_median",
    "VH_std",
    "VV_VH_mean",
    "VH_VV_mean",
)

ERA5_COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"
ERA5_FEATURE_BANDS = (
    "temperature_c",
    "rainfall_mm",
    "soil_moisture",
    "relative_humidity",
    "solar_radiation",
    "wind_speed",
)
ERA5_FEATURE_COLUMNS = tuple(f"{band}_mean" for band in ERA5_FEATURE_BANDS)

RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate limit",
    "too many requests",
    "quota",
    "deadline exceeded",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "backend error",
    "internal error",
    "connection reset",
    "connection aborted",
)


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in RETRYABLE_MARKERS)


def call_with_retry(
    operation: Callable[[], T],
    description: str = "Earth Engine request",
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> T:
    """Run an Earth Engine operation with exponential backoff for transient errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                description,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"{description} failed after {max_attempts} attempts.")


def initialize_gee(
    service_account: str | None = None,
    private_key_path: str | Path | None = None,
    project_id: str | None = None,
) -> None:
    """Initialize Earth Engine with a service account or local credentials."""
    account = service_account or GEE_SERVICE_ACCOUNT
    key_path = private_key_path or GEE_PRIVATE_KEY_PATH
    project = project_id or GEE_PROJECT_ID

    try:
        if account:
            if not key_path:
                raise ValueError(
                    "A service account was provided without a private key path."
                )

            resolved_key_path = Path(key_path).expanduser()
            if not resolved_key_path.is_absolute():
                resolved_key_path = BASE_DIR / resolved_key_path
            if not resolved_key_path.exists():
                raise FileNotFoundError(
                    f"GEE private key file not found: {resolved_key_path}"
                )

            logger.info("Authenticating GEE with service account: %s", account)
            credentials = ee.ServiceAccountCredentials(account, str(resolved_key_path))
            ee.Initialize(credentials, project=project)
        else:
            logger.info("Initializing GEE with default user credentials...")
            ee.Initialize(project=project) if project else ee.Initialize()

        logger.info("Google Earth Engine initialized successfully.")
    except Exception as exc:
        logger.error("Failed to initialize Google Earth Engine: %s", exc)
        raise


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask shadows, clouds, saturated pixels, and nodata using the SCL band."""
    scl = image.select("SCL")
    invalid = (
        scl.eq(0)
        .Or(scl.eq(1))
        .Or(scl.eq(3))
        .Or(scl.eq(8))
        .Or(scl.eq(9))
        .Or(scl.eq(10))
    )
    return image.updateMask(invalid.Not())


def add_sentinel2_indices(image: ee.Image) -> ee.Image:
    """Add the seven optical indices to a reflectance-scaled Sentinel-2 image."""
    nir = image.select("B8").multiply(0.0001).clamp(0, 1)
    red = image.select("B4").multiply(0.0001).clamp(0, 1)
    green = image.select("B3").multiply(0.0001).clamp(0, 1)
    blue = image.select("B2").multiply(0.0001).clamp(0, 1)
    swir1 = image.select("B11").multiply(0.0001).clamp(0, 1)

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI").clamp(-1, 1)
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI").clamp(-1, 1)
    gndvi = nir.subtract(green).divide(nir.add(green)).rename("GNDVI").clamp(-1, 1)
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI").clamp(-1, 1)

    evi = (
        image.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {"NIR": nir, "RED": red, "BLUE": blue},
        )
        .rename("EVI")
        .clamp(-1, 1)
    )
    savi = (
        image.expression(
            "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
            {"NIR": nir, "RED": red},
        )
        .rename("SAVI")
        .clamp(-1, 1)
    )
    msavi = (
        image.expression(
            "0.5 * (2 * NIR + 1 - sqrt((2 * NIR + 1)**2 - 8 * (NIR - RED)))",
            {"NIR": nir, "RED": red},
        )
        .rename("MSAVI")
        .clamp(-1, 1)
    )

    return image.addBands([ndvi, evi, ndwi, savi, msavi, gndvi, ndbi])


def build_s2_month_composite(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 70.0,
) -> tuple[ee.Image | None, int]:
    """Build one cloud-masked monthly median composite with optical indices."""
    collection = (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
    )
    image_count = int(
        call_with_retry(
            lambda: collection.size().getInfo(),
            description=f"Sentinel-2 scene count {start_date} to {end_date}",
        )
    )
    if image_count == 0:
        logger.warning("No Sentinel-2 scenes for %s to %s.", start_date, end_date)
        return None, 0

    composite = (
        collection.map(mask_s2_clouds)
        .map(add_sentinel2_indices)
        .median()
    )
    reflectance = composite.select(list(S2_REFLECTANCE_BANDS)).multiply(0.0001)
    indices = composite.select(list(S2_INDEX_BANDS))
    return reflectance.addBands(indices).select(list(S2_FEATURE_BANDS)), image_count


def prepare_sentinel1_collection(
    start_date: str,
    end_date: str,
    geometry: ee.Geometry | None = None,
    orbit_pass: str = "DESCENDING",
) -> ee.ImageCollection:
    """Filter Sentinel-1 GRD imagery to IW mode with dual VV/VH polarisation."""
    collection = (
        ee.ImageCollection(S1_COLLECTION_ID)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filterDate(start_date, end_date)
    )
    if geometry is not None:
        collection = collection.filterBounds(geometry)
    return collection


def add_sentinel1_features(image: ee.Image) -> ee.Image:
    """
    Add linear-power VV/VH and VH/VV ratios.

    Sentinel-1 bands are stored in decibels, so ratios use
    ``10 ** ((VV - VH) / 10)`` rather than dividing dB values directly.
    """
    vv = image.select("VV")
    vh = image.select("VH")

    vv_vh = (
        image.expression("pow(10, (VV - VH) / 10)", {"VV": vv, "VH": vh})
        .rename("VVVH")
        .clamp(0, 1000)
    )
    vh_vv = (
        image.expression("pow(10, (VH - VV) / 10)", {"VV": vv, "VH": vh})
        .rename("VHVV")
        .clamp(0, 1000)
    )
    return image.addBands([vv_vh, vh_vv])


def build_s1_month_composite(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    orbit_pass: str = "DESCENDING",
) -> tuple[ee.Image | None, int]:
    """Build one monthly mean, median, and std-dev Sentinel-1 composite."""
    collection = prepare_sentinel1_collection(
        start_date=start_date,
        end_date=end_date,
        geometry=aoi,
        orbit_pass=orbit_pass,
    )
    image_count = int(
        call_with_retry(
            lambda: collection.size().getInfo(),
            description=f"Sentinel-1 scene count {start_date} to {end_date}",
        )
    )
    if image_count == 0:
        logger.warning("No Sentinel-1 scenes for %s to %s.", start_date, end_date)
        return None, 0

    with_ratios = collection.map(add_sentinel1_features)
    mean_image = (
        with_ratios.select(["VV", "VH", "VVVH", "VHVV"])
        .mean()
        .rename(["VVm", "VHm", "VVVH", "VHVV"])
    )
    median_image = collection.select(["VV", "VH"]).median().rename(["VVmed", "VHmed"])
    std_image = (
        collection.select(["VV", "VH"])
        .reduce(ee.Reducer.stdDev())
        .rename(["VVs", "VHs"])
    )

    composite = ee.Image.cat([mean_image, median_image, std_image]).select(
        list(S1_INTERNAL_BANDS)
    )
    return composite, image_count


def build_era5_month_composite(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
) -> tuple[ee.Image | None, int]:
    """
    Build one monthly ERA5-Land composite with temperature, rainfall, soil
    moisture, relative humidity, solar radiation, and wind speed.
    """
    collection = (
        ee.ImageCollection(ERA5_COLLECTION_ID)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    image_count = int(
        call_with_retry(
            lambda: collection.size().getInfo(),
            description=f"ERA5-Land image count {start_date} to {end_date}",
        )
    )
    if image_count == 0:
        logger.warning("No ERA5-Land images for %s to %s.", start_date, end_date)
        return None, 0

    temperature_c = (
        collection.select("temperature_2m").mean().subtract(273.15).rename("temperature_c")
    )
    dewpoint_c = collection.select("dewpoint_temperature_2m").mean().subtract(273.15)
    relative_humidity = (
        temperature_c.expression(
            "100 * exp((17.625 * TD) / (243.04 + TD) - (17.625 * T) / (243.04 + T))",
            {"T": temperature_c, "TD": dewpoint_c},
        )
        .rename("relative_humidity")
        .clamp(0, 100)
    )
    rainfall_mm = (
        collection.select("total_precipitation_hourly")
        .sum()
        .multiply(1000)
        .rename("rainfall_mm")
    )
    soil_moisture = (
        collection.select("volumetric_soil_water_layer_1")
        .mean()
        .rename("soil_moisture")
    )
    solar_radiation = (
        collection.select("surface_net_solar_radiation_hourly")
        .sum()
        .multiply(1e-6)
        .rename("solar_radiation")
    )
    wind_speed = (
        collection.select("u_component_of_wind_10m")
        .mean()
        .pow(2)
        .add(collection.select("v_component_of_wind_10m").mean().pow(2))
        .sqrt()
        .rename("wind_speed")
    )

    composite = ee.Image.cat(
        [
            temperature_c,
            rainfall_mm,
            soil_moisture,
            relative_humidity,
            solar_radiation,
            wind_speed,
        ]
    ).select(list(ERA5_FEATURE_BANDS))
    return composite, image_count


def stack_monthly_images(
    images_by_month: dict[int, ee.Image],
    band_names: tuple[str, ...],
) -> ee.Image:
    """Prefix each month's bands with ``Mxx_`` and stack them into one image."""
    if not images_by_month:
        raise ValueError("At least one monthly image is required.")

    stacked = [
        image.select(list(band_names)).rename(
            [f"M{month:02d}_{band}" for band in band_names]
        )
        for month, image in sorted(images_by_month.items())
    ]
    return ee.Image.cat(stacked)


def reshape_batch_statistics(
    stats: pd.DataFrame,
    year: int,
    months: list[int],
    band_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> pd.DataFrame:
    """Convert one batched reduction into long rows with one feature per column."""
    if len(band_names) != len(output_names):
        raise ValueError("band_names and output_names must have equal length.")

    frames = []
    for month in months:
        prefix = f"M{month:02d}_"
        frame = pd.DataFrame({"grid_id": stats["grid_id"].astype(str)})

        for band, output in zip(band_names, output_names):
            column = f"{prefix}{band}_mean"
            frame[output] = stats[column] if column in stats.columns else pd.NA

        count_column = f"{prefix}{band_names[0]}_count"
        counts = (
            pd.to_numeric(stats[count_column], errors="coerce")
            if count_column in stats.columns
            else pd.Series(0, index=stats.index, dtype="float64")
        )
        frame["valid_pixels"] = counts.fillna(0).astype("int64")
        frame["year"] = year
        frame["month"] = month
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def make_feature_reducer() -> ee.Reducer:
    """Return per-band means plus valid-pixel counts."""
    return ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)


def grid_to_feature_collection(
    grid: gpd.GeoDataFrame,
    properties: tuple[str, ...] = ("grid_id",),
) -> ee.FeatureCollection:
    """Convert a grid GeoDataFrame to an Earth Engine FeatureCollection."""
    if "grid_id" not in grid.columns:
        raise ValueError("Grid must contain a grid_id column.")

    selected_columns = list(dict.fromkeys([*properties, "geometry"]))
    grid_wgs84 = grid if grid.crs and grid.crs.to_epsg() == 4326 else grid.to_crs(4326)
    return ee.FeatureCollection(grid_wgs84[selected_columns].__geo_interface__)


def zonal_statistics(
    image: ee.Image,
    grid: gpd.GeoDataFrame,
    scale: float = 100.0,
    chunk_size: int = 500,
    tile_scale: int = 4,
    reducer: ee.Reducer | None = None,
) -> pd.DataFrame:
    """Reduce an image over grid polygons in chunks and return one row per cell."""
    if grid.empty:
        raise ValueError("Cannot calculate zonal statistics for an empty grid.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    reducer = reducer or make_feature_reducer()
    frames = []

    for start in range(0, len(grid), chunk_size):
        chunk = grid.iloc[start : start + chunk_size]
        chunk_ids = chunk["grid_id"].astype(str).tolist()
        logger.info(
            "Reducing cells %d-%d of %d...", start + 1, start + len(chunk), len(grid)
        )

        reduced = image.reduceRegions(
            collection=grid_to_feature_collection(chunk),
            reducer=reducer,
            scale=scale,
            tileScale=tile_scale,
        )
        info = call_with_retry(
            lambda: reduced.getInfo(),
            description=f"zonal statistics for cells {start + 1}-{start + len(chunk)}",
        )

        records = []
        for feature in info.get("features", []):
            properties = feature.get("properties") or {}
            properties.setdefault("grid_id", feature.get("id"))
            records.append(properties)

        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            frame = pd.DataFrame({"grid_id": chunk_ids})
        elif "grid_id" not in frame.columns:
            frame["grid_id"] = None

        frame["grid_id"] = frame["grid_id"].astype(str)
        frame = (
            frame.drop_duplicates(subset=["grid_id"])
            .set_index("grid_id")
            .reindex(chunk_ids)
            .reset_index()
        )
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["grid_id"] = result["grid_id"].astype(str)
    return result.sort_values("grid_id").reset_index(drop=True)
