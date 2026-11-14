"""
Script 08: Build the district ground-truth panel and the final ML training set.

Historical yields come from the Open Data Nepal CKAN district tables for
2024/25 and the cached Zenodo 77-district maize/wheat panel for 2018-2023.
Satellite and climate features are aggregated from monthly grid cells to
annual and quarterly grid-cell features before the district-year labels are
joined. Outputs:

- ``data/processed/nepal_all_77_districts_dataset.csv``
- ``data/processed/ground_truth_labels.csv``
- ``data/final/nepal_crop_ml_training_dataset.csv``
"""

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import pandas as pd

from ml_pipeline.config import FINAL_DIR, PROCESSED_DIR, RAW_DIR
from ml_pipeline.utils.historical_helpers import (
    build_district_lookup,
    download_odn_csv,
    fetch_odn_package,
    odn_district_crop_table,
    select_district_crop_resources,
    zenodo_yield_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

QUARTER_MONTHS = {
    1: (1, 2, 3),
    2: (4, 5, 6),
    3: (7, 8, 9),
    4: (10, 11, 12),
}
SUM_FEATURES = {"rainfall_mm_mean", "solar_radiation_mean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge satellite, climate, and historical crop statistics."
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=RAW_DIR / "grid.geojson",
        help="Path to the Phase 1 grid GeoJSON.",
    )
    parser.add_argument(
        "--sentinel2",
        type=Path,
        default=PROCESSED_DIR / "sentinel2_features.csv",
        help="Combined Sentinel-2 monthly feature CSV.",
    )
    parser.add_argument(
        "--sentinel1",
        type=Path,
        default=PROCESSED_DIR / "sentinel1_features.csv",
        help="Combined Sentinel-1 monthly feature CSV.",
    )
    parser.add_argument(
        "--era5",
        type=Path,
        default=PROCESSED_DIR / "era5_climate_features.csv",
        help="Combined ERA5-Land monthly feature CSV.",
    )
    parser.add_argument(
        "--zenodo",
        type=Path,
        default=RAW_DIR / "ground_truth" / "nepal_maize_wheat_panel_1981_2023.csv",
        help="Cached Zenodo district maize/wheat panel.",
    )
    parser.add_argument(
        "--district-output",
        type=Path,
        default=PROCESSED_DIR / "nepal_all_77_districts_dataset.csv",
        help="Path for the all-77-districts annual panel.",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=PROCESSED_DIR / "ground_truth_labels.csv",
        help="Path for the long ground-truth label table.",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=FINAL_DIR / "nepal_crop_ml_training_dataset.csv",
        help="Path for the final ML training dataset.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="First year to include (default: 2018).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Last year to include (default: 2024).",
    )
    parser.add_argument(
        "--refresh-odn",
        action="store_true",
        help="Re-download the Open Data Nepal district tables.",
    )
    return parser.parse_args()


def required_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_monthly_features(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(required_file(path, label))
    missing = {"grid_id", "year", "month"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    frame["grid_id"] = frame["grid_id"].astype(str)
    return frame


NON_FEATURE_COLUMNS = {
    "grid_id",
    "district",
    "province",
    "latitude",
    "longitude",
    "year",
    "month",
    "period_start",
    "period_end",
    "valid_pixels",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return every numeric measurement column, excluding metadata and counts."""
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and not column.endswith("_count")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def aggregate_grid_year(
    frame: pd.DataFrame,
    prefix: str,
    include_quarters: bool = True,
) -> pd.DataFrame:
    """Aggregate monthly grid features to annual means and quarterly means."""
    features = feature_columns(frame)
    if not features:
        raise ValueError(f"No *_mean feature columns found for {prefix}.")

    aggregations = {
        column: ("sum" if column in SUM_FEATURES else "mean") for column in features
    }
    if f"{prefix}_image_count" in frame.columns:
        aggregations[f"{prefix}_image_count"] = "sum"
    aggregations["valid_pixels"] = "sum"

    annual = frame.groupby(["grid_id", "year"], as_index=False).agg(aggregations)
    annual = annual.rename(columns={"valid_pixels": f"{prefix}_valid_pixels"})

    if not include_quarters:
        return annual

    parts = [annual]
    for quarter, months in QUARTER_MONTHS.items():
        subset = frame[frame["month"].isin(months)]
        quarter_aggs = {
            column: ("sum" if column in SUM_FEATURES else "mean") for column in features
        }
        quarter_frame = subset.groupby(["grid_id", "year"], as_index=False).agg(
            quarter_aggs
        )
        quarter_frame = quarter_frame.rename(
            columns={column: f"{column}_Q{quarter}" for column in features}
        )
        parts.append(quarter_frame)

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["grid_id", "year"], how="left")
    return merged


def fetch_ground_truth(
    grid: gpd.GeoDataFrame,
    zenodo_path: Path,
    start_year: int,
    end_year: int,
    refresh_odn: bool,
) -> pd.DataFrame:
    """Load ODN 2024/25 and Zenodo 2018-2023 district yields into one table."""
    lookup = build_district_lookup(grid)
    frames = []

    try:
        package = fetch_odn_package()
        resources = select_district_crop_resources(package)
        cache_dir = RAW_DIR / "ground_truth"
        year = 2024
        for key, resource in resources.items():
            frame = download_odn_csv(
                resource,
                cache_dir / f"odn_{key}_by_district.csv",
                refresh=refresh_odn,
            )
            frames.append(
                odn_district_crop_table(
                    frame,
                    year=year,
                    source=f"ODN_{key}_2024_25",
                    lookup=lookup,
                )
            )
            logger.info("Loaded ODN %s table: %d rows.", key, len(frame))
    except Exception as exc:
        logger.warning("ODN download failed; using cache or Zenodo only: %s", exc)
        cache_dir = RAW_DIR / "ground_truth"
        for key in ("cereal", "cash"):
            cache_path = cache_dir / f"odn_{key}_by_district.csv"
            if cache_path.exists():
                frames.append(
                    odn_district_crop_table(
                        pd.read_csv(cache_path),
                        year=2024,
                        source=f"ODN_{key}_2024_25_cached",
                        lookup=lookup,
                    )
                )
                logger.info("Loaded cached ODN %s table.", key)

    zenodo = pd.read_csv(required_file(zenodo_path, "Zenodo ground-truth panel"))
    zenodo_long = zenodo_yield_table(zenodo, start_year, end_year, lookup)
    logger.info("Loaded Zenodo labels: %d rows.", len(zenodo_long))
    frames.append(zenodo_long)

    labels = pd.concat(frames, ignore_index=True)
    labels = labels[labels["year"].between(start_year, end_year)].copy()
    labels["crop"] = labels["crop"].astype(str).str.strip()
    labels = labels.drop_duplicates(["district", "year", "crop"], keep="first")
    labels = labels.sort_values(["district", "year", "crop"]).reset_index(drop=True)

    if labels.empty:
        raise RuntimeError("No ground-truth labels were loaded.")

    logger.info(
        "Ground truth: %d rows, %d districts, %d years, %d crops.",
        len(labels),
        labels["district"].nunique(),
        labels["year"].nunique(),
        labels["crop"].nunique(),
    )
    return labels


def district_climate_panel(
    era5: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Summarise monthly grid climate into one row per district and year."""
    frame = era5[era5["year"].between(start_year, end_year)].copy()
    features = feature_columns(frame)
    aggregations = {
        column: ("sum" if column in SUM_FEATURES else "mean") for column in features
    }
    aggregations["valid_pixels"] = "sum"

    grid_year = frame.groupby(["grid_id", "year"], as_index=False).agg(aggregations)
    grid_year = grid_year.merge(
        grid[["grid_id", "district", "province"]], on="grid_id", how="left"
    )

    district = (
        grid_year.groupby(["district", "province", "year"], as_index=False)
        .agg({**{column: "mean" for column in features}, "valid_pixels": "sum"})
        .rename(columns={"valid_pixels": "era5_valid_pixels"})
    )
    counts = (
        grid_year.groupby(["district", "year"], as_index=False)["grid_id"]
        .nunique()
        .rename(columns={"grid_id": "grid_cells"})
    )
    return district.merge(counts, on=["district", "year"], how="left")


def crop_slug(crop: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", crop.lower()).strip("_")


def build_district_dataset(
    grid: gpd.GeoDataFrame,
    labels: pd.DataFrame,
    era5: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    districts = grid[["district", "province"]].drop_duplicates().reset_index(drop=True)
    years = pd.DataFrame({"year": range(start_year, end_year + 1), "key": 1})
    panel = districts.assign(key=1).merge(years, on="key").drop(columns="key")

    climate = district_climate_panel(era5, grid, start_year, end_year)
    panel = panel.merge(climate, on=["district", "province", "year"], how="left")

    yield_wide = labels.pivot_table(
        index=["district", "year"], columns="crop", values="yield_kg_ha"
    )
    yield_wide.columns = [f"{crop_slug(crop)}_yield_kg_ha" for crop in yield_wide.columns]
    panel = panel.merge(yield_wide.reset_index(), on=["district", "year"], how="left")

    area = labels.dropna(subset=["area_ha"]).pivot_table(
        index=["district", "year"], columns="crop", values="area_ha"
    )
    if not area.empty:
        area.columns = [f"{crop_slug(crop)}_area_ha" for crop in area.columns]
        panel = panel.merge(area.reset_index(), on=["district", "year"], how="left")

    production = labels.dropna(subset=["production_mt"]).pivot_table(
        index=["district", "year"], columns="crop", values="production_mt"
    )
    if not production.empty:
        production.columns = [
            f"{crop_slug(crop)}_production_mt" for crop in production.columns
        ]
        panel = panel.merge(
            production.reset_index(), on=["district", "year"], how="left"
        )

    return panel.sort_values(["district", "year"]).reset_index(drop=True)


def build_final_dataset(
    grid: gpd.GeoDataFrame,
    labels: pd.DataFrame,
    sentinel2: pd.DataFrame,
    sentinel1: pd.DataFrame,
    era5: pd.DataFrame,
) -> pd.DataFrame:
    s2 = aggregate_grid_year(sentinel2, "s2")
    s1 = aggregate_grid_year(sentinel1, "s1")
    climate = aggregate_grid_year(era5, "era5")

    features = s2.merge(s1, on=["grid_id", "year"], how="outer")
    features = features.merge(climate, on=["grid_id", "year"], how="outer")
    features = features.merge(
        grid[["grid_id", "district", "province", "latitude", "longitude"]],
        on="grid_id",
        how="left",
    )

    final = features.merge(
        labels.drop(columns=["province"], errors="ignore"),
        on=["district", "year"],
        how="inner",
    )
    final = final[final["yield_kg_ha"].notna()].copy()
    final = final.sort_values(["year", "district", "grid_id", "crop"]).reset_index(
        drop=True
    )
    return final


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be on or before --end-year.")

    logger.info("=== Phase 5: historical ground truth and final merge ===")
    logger.info("Years: %d-%d", args.start_year, args.end_year)

    try:
        grid = gpd.read_file(required_file(args.grid, "Grid"))
        grid["grid_id"] = grid["grid_id"].astype(str)
        logger.info("Loaded %d grid cells and %d districts.", len(grid), grid["district"].nunique())

        labels = fetch_ground_truth(
            grid=grid,
            zenodo_path=args.zenodo,
            start_year=args.start_year,
            end_year=args.end_year,
            refresh_odn=args.refresh_odn,
        )
        args.labels_output.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(args.labels_output, index=False)
        logger.info("Wrote ground-truth labels to %s.", args.labels_output)

        era5 = load_monthly_features(args.era5, "ERA5 features")
        district_panel = build_district_dataset(
            grid, labels, era5, args.start_year, args.end_year
        )
        args.district_output.parent.mkdir(parents=True, exist_ok=True)
        district_panel.to_csv(args.district_output, index=False)
        logger.info(
            "Wrote all-77-districts panel: %d rows, %d districts.",
            len(district_panel),
            district_panel["district"].nunique(),
        )

        sentinel2 = load_monthly_features(args.sentinel2, "Sentinel-2 features")
        sentinel1 = load_monthly_features(args.sentinel1, "Sentinel-1 features")
        final = build_final_dataset(
            grid, labels, sentinel2, sentinel1, era5
        )
        args.final_output.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(args.final_output, index=False)
        parquet_path = args.final_output.with_suffix(".parquet")
        final.to_parquet(parquet_path, index=False)

        logger.info(
            "Wrote final training dataset: %d rows, %d cells, %d districts, %d crops.",
            len(final),
            final["grid_id"].nunique(),
            final["district"].nunique(),
            final["crop"].nunique(),
        )
        logger.info("Yield labels present: %d.", int(final["yield_kg_ha"].notna().sum()))
        logger.info("=== Phase 5 complete ===")
    except Exception as exc:
        logger.error("Phase 5 failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
