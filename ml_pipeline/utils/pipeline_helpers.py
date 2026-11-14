"""
Shared helpers for monthly periods, grid loading, checkpoint reuse, and output.

Both satellite extraction scripts use these helpers so that batching, metadata,
and file handling stay consistent across the seven-year run.
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRID_METADATA_COLUMNS = ["grid_id", "district", "province", "latitude", "longitude"]
PERIOD_COLUMNS = ["year", "month", "period_start", "period_end"]


def month_periods(start: date, end: date) -> list[tuple[str, date, date, date]]:
    """Return (label, period_start, period_end, exclusive_end) for each month."""
    periods = []
    cursor = date(start.year, start.month, 1)

    while cursor <= end:
        next_month = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
        period_start = max(cursor, start)
        period_end = min(next_month - timedelta(days=1), end)
        exclusive_end = min(next_month, end + timedelta(days=1))
        periods.append(
            (f"{cursor.year:04d}-{cursor.month:02d}", period_start, period_end, exclusive_end)
        )
        cursor = next_month

    return periods


def quarter_batches(
    periods: list[tuple[str, date, date, date]],
) -> list[tuple[tuple[int, int], list[tuple[str, date, date, date]]]]:
    """Group monthly periods into calendar-quarter batches."""
    batches: dict[tuple[int, int], list[tuple[str, date, date, date]]] = {}
    for period in periods:
        start = period[1]
        key = (start.year, (start.month - 1) // 3 + 1)
        batches.setdefault(key, []).append(period)
    return sorted(batches.items())


def year_batches(
    periods: list[tuple[str, date, date, date]],
) -> list[tuple[int, list[tuple[str, date, date, date]]]]:
    """Group monthly periods into calendar-year batches."""
    batches: dict[int, list[tuple[str, date, date, date]]] = {}
    for period in periods:
        batches.setdefault(period[1].year, []).append(period)
    return sorted(batches.items())


def load_grid(grid_path: Path, max_cells: int | None = None) -> gpd.GeoDataFrame:
    """Load, validate, and optionally truncate the Phase 1 grid."""
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    grid = gpd.read_file(grid_path)
    required = {*GRID_METADATA_COLUMNS, "geometry"}
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"Grid is missing required columns: {sorted(missing)}")

    grid = grid if grid.crs else grid.set_crs("EPSG:4326")
    grid = grid.to_crs("EPSG:4326")
    grid = grid[grid.geometry.notna() & ~grid.geometry.is_empty].copy()
    grid["grid_id"] = grid["grid_id"].astype(str)

    if max_cells is not None:
        if max_cells <= 0:
            raise ValueError("--max-cells must be greater than zero.")
        grid = grid.head(max_cells).copy()
        logger.warning("Processing only %d grid cells.", len(grid))

    return grid.reset_index(drop=True)


def checkpoint_is_reusable(path: Path, expected_rows: int, period_start: date) -> bool:
    """Return True when a monthly checkpoint matches the current grid and period."""
    if not path.exists():
        return False

    try:
        existing = pd.read_csv(path, usecols=["grid_id", "period_start"])
    except (ValueError, pd.errors.ParserError):
        return False

    return (
        len(existing) == expected_rows
        and existing["grid_id"].nunique() == expected_rows
        and (existing["period_start"] == period_start.isoformat()).all()
    )


def metadata_frame(
    grid: gpd.GeoDataFrame,
    period: tuple[str, date, date, date],
    image_count: int,
    image_count_column: str,
) -> pd.DataFrame:
    """Build the grid and period columns for one month."""
    _, period_start, period_end, _ = period
    frame = grid[GRID_METADATA_COLUMNS].copy()
    frame["year"] = period_start.year
    frame["month"] = period_start.month
    frame["period_start"] = period_start.isoformat()
    frame["period_end"] = period_end.isoformat()
    frame[image_count_column] = image_count
    return frame


def assemble_month_frame(
    grid: gpd.GeoDataFrame,
    period: tuple[str, date, date, date],
    stats: pd.DataFrame | None,
    feature_columns: list[str],
    image_count: int,
    image_count_column: str,
) -> pd.DataFrame:
    """
    Join one month of reduced values to grid metadata in a stable column order.

    ``stats`` must contain ``grid_id``, ``valid_pixels``, and one column per
    feature. Pass ``None`` to emit a complete month of missing values.
    """
    frame = metadata_frame(grid, period, image_count, image_count_column)

    if stats is None:
        stats = pd.DataFrame(
            {"grid_id": grid["grid_id"].astype(str), "valid_pixels": 0}
        )

    merged = frame.merge(stats, on="grid_id", how="left")
    for column in feature_columns:
        if column not in merged.columns:
            merged[column] = np.nan

    merged["valid_pixels"] = (
        pd.to_numeric(merged["valid_pixels"], errors="coerce").fillna(0).astype("int64")
    )
    ordered = (
        GRID_METADATA_COLUMNS
        + PERIOD_COLUMNS
        + [image_count_column, "valid_pixels"]
        + feature_columns
    )
    return merged[ordered].sort_values("grid_id").reset_index(drop=True)


def combine_checkpoints(
    checkpoint_dir: Path,
    prefix: str,
    periods: list[tuple[str, date, date, date]],
    output_csv: Path,
    metadata: dict,
) -> pd.DataFrame:
    """Combine requested monthly checkpoints into CSV, Parquet, and metadata."""
    paths = [checkpoint_dir / f"{prefix}_{label}.csv" for label, *_ in periods]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing monthly checkpoints: {', '.join(missing)}")

    final = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    final = final.sort_values(["grid_id", "year", "month"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = output_csv.with_suffix(".parquet")
    final.to_csv(output_csv, index=False)
    final.to_parquet(parquet_path, index=False)

    payload = {
        **metadata,
        "rows": len(final),
        "grid_cells": int(final["grid_id"].nunique()),
        "months": len(paths),
        "columns": list(final.columns),
        "outputs": {
            "csv": str(output_csv),
            "parquet": str(parquet_path),
            "checkpoints": str(checkpoint_dir),
        },
    }
    metadata_path = output_csv.with_name(f"{output_csv.stem}_metadata.json")
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info("Combined dataset written to %s.", output_csv)
    logger.info("Parquet copy and metadata written alongside the CSV.")
    return final
