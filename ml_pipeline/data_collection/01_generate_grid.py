"""
Script 01: Generate 5km x 5km Spatial Grid for Nepal.
Loads Nepal district boundaries, generates the grid, assigns district/province metadata,
and exports `grid.geojson` and `grid.csv` to `data/raw/`.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_pipeline.config import (
    GRID_RESOLUTION_METERS,
    NEPAL_CRS,
    NEPAL_DISTRICTS_URL,
    NEPAL_PROJECTED_CRS,
    RAW_DIR,
)
from ml_pipeline.utils.geo_helpers import generate_grid_cells, load_nepal_districts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Starting Phase 1: Nepal Grid Generation ===")

    try:
        districts_gdf = load_nepal_districts(NEPAL_DISTRICTS_URL)
        logger.info("Loaded %d district/administrative polygons.", len(districts_gdf))

        grid_gdf = generate_grid_cells(
            boundary_gdf=districts_gdf,
            resolution_meters=GRID_RESOLUTION_METERS,
            target_crs=NEPAL_PROJECTED_CRS,
            output_crs=NEPAL_CRS,
        )
        logger.info("Successfully generated %d grid cells covering Nepal.", len(grid_gdf))

        geojson_path = RAW_DIR / "grid.geojson"
        csv_path = RAW_DIR / "grid.csv"

        logger.info("Exporting grid to GeoJSON: %s", geojson_path)
        grid_gdf.to_file(geojson_path, driver="GeoJSON")

        logger.info("Exporting grid to CSV: %s", csv_path)
        grid_gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)

        for output_path in (geojson_path, csv_path):
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Expected output was not written: {output_path}")

        logger.info("=== Phase 1 Completed Successfully! ===")
        logger.info("Grid cells: %d | GeoJSON: %s | CSV: %s", len(grid_gdf), geojson_path, csv_path)
    except Exception as exc:
        logger.error("Phase 1 failed with error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
