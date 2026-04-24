"""Create split grids and calculate raster coverage for all So2Sat cities.

For each city the script:
  1. Loads patches_reference_{city}.gpkg (label polygons).
  2. Builds a 3×3 split grid over the city ROI using a fixed train/val/test
     checkerboard pattern.  The grid bbox comes from the reference CSV
     (so2sat_guppd_bounds.csv) if a match is found, otherwise falls back to
     the label polygon extent.
  3. Joins each polygon to its grid cell (by centroid).
  4. Calculates per-tile coverage from patches_reference_{city}.tif.
  5. Saves two GeoPackages to the city folder:
       patches_reference_{city}_split.gpkg  — polygons with split + grid_id
       {city}_grid.gpkg                     — grid tiles with split + coverage_pct + is_valid

Example:
    python src/create_city_grids.py \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --workers 4
"""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

# Fixed 3×3 train/val/test pattern (rows go top→bottom, columns left→right)
_SPLIT_PATTERN = [
    ["train", "test",  "train"],
    ["train", "train", "train"],
    ["train", "val",   "train"],
]

_COVERAGE_MIN = 0.02
_COVERAGE_MAX = 0.95

# Dir names that don't normalise cleanly to their CSV JRC_NAME_MAIN equivalent
_CITY_NAME_OVERRIDES: dict[str, str] = {
    "Sao Paulo": "São Paulo",
    "Dongying": "东营区",
}


def _load_city_bboxes(csv_path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Load city reference bboxes from CSV, keyed by JRC_NAME_MAIN."""
    df = pd.read_csv(csv_path)
    return {
        row["JRC_NAME_MAIN"]: (row["minx"], row["miny"], row["maxx"], row["maxy"])
        for _, row in df.iterrows()
    }


def _lookup_city_bbox(
    city: str, bbox_dict: dict[str, tuple[float, float, float, float]]
) -> tuple[float, float, float, float] | None:
    """Return (minx, miny, maxx, maxy) for a city dir name, or None if not found."""
    normalized = city.replace("_", " ")
    key = _CITY_NAME_OVERRIDES.get(normalized, normalized)
    return bbox_dict.get(key)


def _process_city(
    city_dir: Path,
    sub_tile_size: float,
    overwrite: bool,
    city_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[str, str | None]:
    """Process a single city. Returns (city_name, error_message_or_None)."""
    # Lazy imports so this function works in worker processes
    import warnings
    import geopandas as gpd

    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from utils.grid_split import create_split_grid_and_join, calculate_tile_coverage

    city = city_dir.name
    gpkg_path = city_dir / f"patches_reference_{city}.gpkg"
    tif_path  = city_dir / f"patches_reference_{city}.tif"
    out_split = city_dir / f"patches_reference_{city}_split.gpkg"
    out_grid  = city_dir / f"{city}_grid.gpkg"

    if not overwrite and out_split.exists() and out_grid.exists():
        return city, "skipped (already exists)"

    if not gpkg_path.exists():
        return city, f"missing {gpkg_path.name}"
    if not tif_path.exists():
        return city, f"missing {tif_path.name}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # 1. Load polygons
            polygons = gpd.read_file(gpkg_path)

            # 2. Build grid and join polygons
            #    target_crs=None → auto-estimates UTM per city
            if city_bbox is not None:
                _bbox_coords = city_bbox
                _bbox_crs = "EPSG:4326"
            else:
                logger.warning(f"  {city}: no CSV bbox found, falling back to label extent")
                _bbox_coords = tuple(polygons.total_bounds)
                _bbox_crs = str(polygons.crs)

            joined, grid = create_split_grid_and_join(
                bbox_coords=_bbox_coords,
                bbox_crs=_bbox_crs,
                polygons_gdf=polygons,
                sub_tile_size=sub_tile_size,
                grid_size=3,
                pattern=_SPLIT_PATTERN,
                target_crs=None,
                join_method="centroid",
            )

            # 3. Coverage — use the CRS the grid ended up in
            target_crs = str(grid.crs)
            grid_cov = calculate_tile_coverage(
                grid_gdf=grid,
                raster_paths=tif_path,
                target_crs=target_crs,
                band=1,
            )
            grid_cov["is_valid"] = grid_cov["coverage_pct"].between(
                _COVERAGE_MIN, _COVERAGE_MAX
            )

            # 4. Save
            joined.to_file(out_split, driver="GPKG")
            grid_cov.to_file(out_grid, driver="GPKG")

        return city, None

    except Exception as exc:
        return city, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create split grids and coverage for all So2Sat cities."
    )
    parser.add_argument(
        "--cities-dir",
        required=True,
        type=Path,
        help="Directory that contains one subfolder per city.",
    )
    parser.add_argument(
        "--sub-tile-size",
        type=float,
        default=1280.0,
        help="Size of each grid sub-tile in projected units (metres). Default: 1280.",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Process only these cities (default: all).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process cities that already have output files.",
    )
    parser.add_argument(
        "--bounds-csv",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "so2sat_guppd_bounds.csv",
        help="CSV with reference city bboxes (JRC_NAME_MAIN, minx, miny, maxx, maxy). "
             "Default: data/so2sat_guppd_bounds.csv",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    cities_dir = args.cities_dir
    if not cities_dir.exists():
        logger.error(f"cities-dir not found: {cities_dir}")
        sys.exit(1)

    # Load reference bboxes from CSV
    city_bboxes: dict[str, tuple[float, float, float, float]] = {}
    if args.bounds_csv.exists():
        city_bboxes = _load_city_bboxes(args.bounds_csv)
        logger.info(f"Loaded {len(city_bboxes)} city bboxes from {args.bounds_csv}")
    else:
        logger.warning(f"bounds-csv not found: {args.bounds_csv} — will use label extents")

    city_dirs = sorted(
        d for d in cities_dir.iterdir() if d.is_dir()
    )
    if args.cities:
        city_dirs = [d for d in city_dirs if d.name in args.cities]
        if not city_dirs:
            logger.error(f"None of {args.cities} found in {cities_dir}")
            sys.exit(1)

    logger.info(f"Processing {len(city_dirs)} cities with {args.workers} worker(s).")

    n_ok = n_skip = n_err = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_city, d, args.sub_tile_size, args.overwrite,
                    _lookup_city_bbox(d.name, city_bboxes),
                ): d.name
                for d in city_dirs
            }
            for fut in tqdm(as_completed(futures), total=len(futures), unit="city"):
                city, err = fut.result()
                if err is None:
                    n_ok += 1
                    logger.info(f"  ✓ {city}")
                elif err.startswith("skipped"):
                    n_skip += 1
                    logger.info(f"  – {city}: {err}")
                else:
                    n_err += 1
                    logger.error(f"  ✗ {city}: {err}")
    else:
        for city_dir in tqdm(city_dirs, unit="city"):
            city, err = _process_city(
                city_dir, args.sub_tile_size, args.overwrite,
                _lookup_city_bbox(city_dir.name, city_bboxes),
            )
            if err is None:
                n_ok += 1
                logger.info(f"  ✓ {city}")
            elif err.startswith("skipped"):
                n_skip += 1
                logger.info(f"  – {city}: {err}")
            else:
                n_err += 1
                logger.error(f"  ✗ {city}: {err}")

    logger.info(
        f"\nDone — {n_ok} processed, {n_skip} skipped, {n_err} errors."
    )


if __name__ == "__main__":
    main()
