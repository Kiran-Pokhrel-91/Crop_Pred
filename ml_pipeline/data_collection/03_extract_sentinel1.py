"""
Script 03: Extract monthly Sentinel-1 SAR features for the Nepal 5 km grid.

Builds monthly mean, median, and standard deviation backscatter composites with
linear-power polarization ratios three months at a time, reduces them over grid
cells, and writes resumable monthly checkpoints plus
``data/processed/sentinel1_features.csv``.
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ee

from ml_pipeline.config import PROCESSED_DIR, RAW_DIR, TEST_MODE
from ml_pipeline.utils.gee_helpers import (
    S1_COLLECTION_ID,
    S1_FEATURE_BANDS,
    S1_INTERNAL_BANDS,
    build_s1_month_composite,
    initialize_gee,
    reshape_batch_statistics,
    stack_monthly_images,
    zonal_statistics,
)
from ml_pipeline.utils.pipeline_helpers import (
    assemble_month_frame,
    checkpoint_is_reusable,
    combine_checkpoints,
    load_grid,
    month_periods,
    quarter_batches,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract monthly Sentinel-1 SAR features for the Nepal grid."
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=RAW_DIR / "grid.geojson",
        help="Path to the Phase 1 grid GeoJSON.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date(2024, 1, 1) if TEST_MODE else date(2018, 1, 1),
        help="First date to process (YYYY-MM-DD). Defaults to 2024 in TEST_MODE.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2024, 12, 31),
        help="Last date to process (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Reduction scale in metres (default: 100).",
    )
    parser.add_argument(
        "--orbit-pass",
        choices=("DESCENDING", "ASCENDING"),
        default="DESCENDING",
        help="Sentinel-1 orbit pass to keep (default: DESCENDING).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Grid cells per Earth Engine reduction request (default: 500).",
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=4,
        help="Earth Engine tileScale for large reductions (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR / "sentinel1",
        help="Directory for monthly checkpoint tables.",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=PROCESSED_DIR / "sentinel1_features.csv",
        help="Path for the combined Sentinel-1 feature CSV.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=100 if TEST_MODE else None,
        help="Process only the first N grid cells (defaults to 100 in TEST_MODE).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute monthly checkpoints that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_date > args.end_date:
        raise ValueError("--start-date must be on or before --end-date.")
    if args.scale <= 0:
        raise ValueError("--scale must be greater than zero.")

    logger.info("=== Phase 3: Sentinel-1 SAR extraction ===")
    logger.info("Period: %s to %s", args.start_date, args.end_date)
    if TEST_MODE:
        logger.warning("TEST_MODE: %s cells, 2024 only.", args.max_cells or "all")

    try:
        grid = load_grid(args.grid, args.max_cells)
        feature_columns = list(S1_FEATURE_BANDS)
        logger.info("Loaded %d grid cells.", len(grid))

        initialize_gee()

        minx, miny, maxx, maxy = grid.total_bounds
        aoi = ee.Geometry.Rectangle(
            [float(minx), float(miny), float(maxx), float(maxy)],
            proj="EPSG:4326",
            geodesic=False,
        )

        checkpoint_dir = args.output_dir / "monthly"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        periods = month_periods(args.start_date, args.end_date)
        period_by_label = {period[0]: period for period in periods}
        logger.info("Processing %d months in quarterly batches.", len(periods))

        for (year, quarter), quarter_periods in quarter_batches(periods):
            missing = [
                period
                for period in quarter_periods
                if args.overwrite
                or not checkpoint_is_reusable(
                    checkpoint_dir / f"s1_{period[0]}.csv", len(grid), period[1]
                )
            ]
            if not missing:
                logger.info("%d Q%d already complete.", year, quarter)
                continue

            logger.info(
                "Processing %d Q%d (%d of %d months missing)...",
                year,
                quarter,
                len(missing),
                len(quarter_periods),
            )

            images: dict[int, ee.Image] = {}
            image_counts: dict[str, int] = {}
            for label, start, _, exclusive_end in missing:
                composite, image_count = build_s1_month_composite(
                    aoi=aoi,
                    start_date=start.isoformat(),
                    end_date=exclusive_end.isoformat(),
                    orbit_pass=args.orbit_pass,
                )
                image_counts[label] = image_count
                if composite is not None:
                    images[start.month] = composite

            if images:
                stacked = stack_monthly_images(images, S1_INTERNAL_BANDS)
                stats = zonal_statistics(
                    image=stacked,
                    grid=grid,
                    scale=args.scale,
                    chunk_size=args.chunk_size,
                    tile_scale=args.tile_scale,
                )
                monthly = reshape_batch_statistics(
                    stats,
                    year=year,
                    months=sorted(images),
                    band_names=S1_INTERNAL_BANDS,
                    output_names=S1_FEATURE_BANDS,
                )

                for month, group in monthly.groupby("month", sort=True):
                    label = f"{year:04d}-{int(month):02d}"
                    frame = assemble_month_frame(
                        grid=grid,
                        period=period_by_label[label],
                        stats=group.drop(columns=["year", "month"]),
                        feature_columns=feature_columns,
                        image_count=image_counts[label],
                        image_count_column="s1_image_count",
                    )
                    frame.to_csv(checkpoint_dir / f"s1_{label}.csv", index=False)
                    logger.info(
                        "%s: %d scenes, %d valid cells.",
                        label,
                        image_counts[label],
                        int((frame["valid_pixels"] > 0).sum()),
                    )

            for label, *_ in missing:
                if image_counts[label] != 0:
                    continue
                frame = assemble_month_frame(
                    grid=grid,
                    period=period_by_label[label],
                    stats=None,
                    feature_columns=feature_columns,
                    image_count=0,
                    image_count_column="s1_image_count",
                )
                frame.to_csv(checkpoint_dir / f"s1_{label}.csv", index=False)
                logger.warning("%s: no scenes; wrote a complete row set of NaN values.", label)

        final = combine_checkpoints(
            checkpoint_dir=checkpoint_dir,
            prefix="s1",
            periods=periods,
            output_csv=args.features_csv,
            metadata={
                "collection": S1_COLLECTION_ID,
                "grid_path": str(args.grid),
                "requested_start_date": args.start_date.isoformat(),
                "requested_end_date": args.end_date.isoformat(),
                "scale_m": args.scale,
                "orbit_pass": args.orbit_pass,
                "chunk_size": args.chunk_size,
                "tile_scale": args.tile_scale,
                "max_cells": args.max_cells,
                "test_mode": TEST_MODE,
            },
        )
        logger.info(
            "=== Phase 3 complete: %d rows, %d cells, %d months ===",
            len(final),
            final["grid_id"].nunique(),
            final["month"].nunique(),
        )
    except Exception as exc:
        logger.error("Phase 3 failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
