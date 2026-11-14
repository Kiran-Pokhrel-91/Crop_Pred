"""
Geospatial helper utilities for grid generation, coordinate transformations,
bounding box calculations, and spatial joins using GeoPandas and Shapely.
"""

import logging
from pathlib import Path
from typing import Any, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box
from shapely.prepared import prep

logger = logging.getLogger(__name__)

PROVINCE_NAMES = {
    1: "Koshi",
    2: "Madhesh",
    3: "Bagmati",
    4: "Gandaki",
    5: "Lumbini",
    6: "Karnali",
    7: "Sudurpashchim",
}

# Mapping from older/alternative district spellings to the canonical names
# used by the Phase 1 grid (post-2015 federal district names).
DISTRICT_NAME_ALIASES = {
    "chitwan": "Chitawan",
    "makwanpur": "Makawanpur",
    "makawanpur": "Makawanpur",
    "kavrepalanchok": "Kabhrepalanchok",
    "kabhrepalanchok": "Kabhrepalanchok",
    "kapilvastu": "Kapilbastu",
    "kapilbastu": "Kapilbastu",
    "parasi": "Parasi",
    "nawalparasi east": "Nawalpur",
    "nawalparasi west": "Nawalpur",
    "nawalpur": "Nawalpur",
    "tanahun": "Tanahu",
    "tanahu": "Tanahu",
    "rukum east": "Rukum East",
    "rukum west": "Rukum West",
    "sudur paschim": "Sudurpashchim",
    "sudurpashchim": "Sudurpashchim",
}


def normalize_district_name(name: Any) -> str:
    """Map an alternative/older district spelling to the canonical grid name."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "Unknown"
    text = str(name).strip()
    if not text:
        return "Unknown"
    return DISTRICT_NAME_ALIASES.get(text.lower(), text)


def normalize_province_name(name: Any) -> str:
    """Normalize province names to match the Phase 1 grid convention."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "Unknown"
    text = str(name).strip()
    return DISTRICT_NAME_ALIASES.get(text.lower(), text)

FINAL_COLUMNS = [
    "grid_id",
    "district",
    "province",
    "latitude",
    "longitude",
    "geometry",
]


def _read_remote_geojson(url: str) -> gpd.GeoDataFrame:
    """Download and parse a remote GeoJSON FeatureCollection."""
    logger.info("Downloading district boundaries from %s", url)
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    payload = response.json()

    if payload.get("type") != "FeatureCollection" or not payload.get("features"):
        raise ValueError("Remote GeoJSON is not a non-empty FeatureCollection.")

    return gpd.GeoDataFrame.from_features(
        payload["features"],
        crs="EPSG:4326",
    )


def load_nepal_districts(geojson_url: str) -> gpd.GeoDataFrame:
    """
    Load Nepal district boundaries from a GeoJSON URL or local path.
    Ensures CRS is set to EPSG:4326 and geometries are valid.
    """
    logger.info("Loading Nepal district boundaries from %s...", geojson_url)
    try:
        if geojson_url.startswith(("http://", "https://")):
            gdf = _read_remote_geojson(geojson_url)
        else:
            path = Path(geojson_url).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"District boundary file not found: {path}")
            gdf = gpd.read_file(path)

        if gdf.empty:
            raise ValueError("The district boundary dataset is empty.")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if gdf.empty:
            raise ValueError("No valid district geometries remain after loading.")

        try:
            gdf["geometry"] = gdf.geometry.make_valid()
        except Exception:
            logger.warning("Geometry repair was not available; using input geometries.")

        if len(gdf) != 77:
            logger.warning(
                "Expected 77 districts, but loaded %d features.",
                len(gdf),
            )
        else:
            logger.info("Loaded all 77 Nepal districts.")

        return gdf.reset_index(drop=True)
    except Exception as exc:
        logger.error("Failed to load Nepal districts GeoJSON: %s", exc)
        raise


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name, case-insensitively."""
    lookup = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _province_name_from_code(value: Any) -> str | None:
    """Map a numeric or NP-prefixed province code to its official name."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip().upper()
    if text.startswith("NP"):
        text = text[2:]

    try:
        code = int(text)
    except (TypeError, ValueError):
        return None

    return PROVINCE_NAMES.get(code)


def _standardize_admin_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalise common district/province column names to ``district``/``province``."""
    district_col = _find_column(
        list(gdf.columns),
        ("district", "dist_en", "dist_name", "adm2_en", "name"),
    )
    province_name_col = _find_column(
        list(gdf.columns),
        ("pr_name", "province_name", "prov_name", "adm1_en", "state", "province"),
    )
    province_code_col = _find_column(
        list(gdf.columns),
        ("province", "province_id", "province_code", "adm1_pcode", "adm1_code"),
    )

    if district_col is None:
        raise ValueError(
            "Could not identify a district column. Available columns: "
            f"{list(gdf.columns)}"
        )

    standardized = gdf.copy()
    standardized["district"] = (
        standardized[district_col].astype("string").str.strip()
    )

    province_names = pd.Series(pd.NA, index=standardized.index, dtype="string")
    if province_code_col is not None:
        province_names = standardized[province_code_col].map(_province_name_from_code)

    if province_names.isna().all() and province_name_col is not None:
        province_names = standardized[province_name_col].astype("string").str.strip()

    standardized["province"] = province_names.fillna("Unknown")

    if standardized[["district", "province"]].isna().any().any():
        raise ValueError("District/province fields contain missing values.")

    return standardized[["district", "province", "geometry"]]


def _largest_overlap_join(
    grid: gpd.GeoDataFrame,
    admin: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Assign each grid cell to the district with the largest overlapping area."""
    logger.info("Performing largest-overlap spatial join with district boundaries...")

    intersections = gpd.overlay(
        grid[["grid_id", "geometry"]],
        admin[["district", "province", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    intersections["overlap_area"] = intersections.geometry.area

    assignments = (
        intersections.sort_values("overlap_area", ascending=False, kind="stable")
        .drop_duplicates(subset=["grid_id"], keep="first")
        .set_index("grid_id")[["district", "province"]]
    )

    assigned = grid[["grid_id"]].join(assignments, on="grid_id")

    missing = assigned["district"].isna()
    if missing.any():
        logger.warning(
            "%d grid cells did not overlap a district; assigning nearest district.",
            int(missing.sum()),
        )
        nearest = gpd.sjoin_nearest(
            grid.loc[missing.values, ["grid_id", "geometry"]],
            admin[["district", "province", "geometry"]],
            how="left",
            distance_col="_distance",
        ).drop_duplicates(subset=["grid_id"], keep="first")
        nearest = nearest.set_index("grid_id")[["district", "province"]]
        assigned = assigned.set_index("grid_id")
        assigned.loc[nearest.index, ["district", "province"]] = nearest[
            ["district", "province"]
        ]
        assigned = assigned.reset_index()

    return assigned


def generate_grid_cells(
    boundary_gdf: gpd.GeoDataFrame,
    resolution_meters: float = 5000.0,
    target_crs: str = "EPSG:32645",
    output_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Generate a uniform 5 km x 5 km grid over Nepal and assign district/province
    attributes by maximum area overlap.

    Args:
        boundary_gdf: GeoDataFrame containing Nepal district boundaries.
        resolution_meters: Grid cell size in metres (5000 for 5 km).
        target_crs: Projected CRS used for metric grid construction.
        output_crs: Geographic CRS used for the returned grid.

    Returns:
        GeoDataFrame with grid_id, district, province, latitude, longitude,
        and geometry columns in ``output_crs``.
    """
    if resolution_meters <= 0:
        raise ValueError("resolution_meters must be greater than zero.")
    if boundary_gdf.empty:
        raise ValueError("boundary_gdf is empty; cannot generate a grid.")

    logger.info("Generating %.0fm grid over Nepal...", resolution_meters)

    admin = _standardize_admin_columns(boundary_gdf)

    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    projected_boundary = boundary_gdf.to_crs(target_crs)
    country_geom = projected_boundary.geometry.union_all()
    if country_geom.is_empty:
        raise ValueError("The union of district boundaries is empty.")

    minx, miny, maxx, maxy = country_geom.bounds
    prepared_boundary = prep(country_geom)

    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + resolution_meters, y + resolution_meters)
            if prepared_boundary.intersects(cell):
                grid_cells.append(cell)
            y += resolution_meters
        x += resolution_meters

    if not grid_cells:
        raise ValueError("No grid cells intersect the Nepal boundary.")

    logger.info("Created %d intersecting grid cells in metric space.", len(grid_cells))

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=target_crs)
    grid["grid_id"] = [f"GRID_{index:05d}" for index in range(len(grid))]

    centroids_projected = grid.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(
        centroids_projected,
        crs=target_crs,
    ).to_crs(output_crs)

    assignments = _largest_overlap_join(grid, admin.to_crs(target_crs))
    grid = grid.merge(assignments, on="grid_id", how="left")

    grid["district"] = grid["district"].fillna("Unknown")
    grid["province"] = grid["province"].fillna("Unknown")
    grid["latitude"] = centroids_wgs84.y.to_numpy()
    grid["longitude"] = centroids_wgs84.x.to_numpy()

    grid = grid.to_crs(output_crs)
    grid = grid[FINAL_COLUMNS].sort_values("grid_id").reset_index(drop=True)

    logger.info(
        "Assigned %d cells to %d districts across %d provinces.",
        len(grid),
        grid["district"].nunique(),
        grid["province"].nunique(),
    )

    return grid


def calculate_bounding_box(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """Return the (minx, miny, maxx, maxy) bounds of a GeoDataFrame."""
    if gdf.empty:
        raise ValueError("Cannot calculate bounds for an empty GeoDataFrame.")
    return tuple(float(value) for value in gdf.total_bounds)
