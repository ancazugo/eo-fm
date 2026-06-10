"""Extract embedding crops for each grid tile produced by create_city_grids.py.

For each city the script:
  1. Loads {city}_grid.gpkg (created by create_city_grids.py).
  2. For every grid tile (optionally filtered to is_valid=True):
     a. Finds embedding tile(s) covering the tile footprint.
     b. Clips the embedding to the tile bounding box.
     c. Saves float32 .npy arrays named {city}_{grid_id:02d}.npy into
        split sub-folders under each city's own directory.

Output layout (inside each city folder):
    {city_dir}/{output_name}/{year}/train/{city}_{grid_id:02d}.npy
    {city_dir}/{output_name}/{year}/val/{city}_{grid_id:02d}.npy
    {city_dir}/{output_name}/{year}/test/{city}_{grid_id:02d}.npy

Example (AlphaEarth):
    python src/extract_grid_embeddings.py \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017 \\
        --embedding-name alpha_earth \\
        --output-name AlphaEarth \\
        --year 2017 \\
        --workers 8

Example (GeoTessera):
    python src/extract_grid_embeddings.py \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/GeoTessera/2017 \\
        --embedding-name tessera \\
        --output-name GeoTessera \\
        --year 2017 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
from loguru import logger
from tqdm import tqdm

from datasets.registry import EMBEDDING_REGISTRY
from datasets.tiles import build_tile_index as _build_tile_index, crop_patch as _crop_patch


# ---------------------------------------------------------------------------
# Worker for parallel execution
# ---------------------------------------------------------------------------

def _process_tile(args: tuple) -> tuple[str, bool]:
    """Worker: crop one grid tile and save as .npy.

    args = (city, grid_id, geom_wkt, geom_crs, tile_paths, output_path, skip_existing)
    """
    city, grid_id, geom_wkt, geom_crs, tile_paths, output_path, skip_existing = args
    from shapely import from_wkt
    geom = from_wkt(geom_wkt)
    ok = _crop_patch(geom, geom_crs, tile_paths, output_path, skip_existing)
    label = f"{city}/{grid_id:02d}"
    return label, ok


# ---------------------------------------------------------------------------
# Per-city processing
# ---------------------------------------------------------------------------

def _iter_city_tasks(
    city_dir: Path,
    tile_paths: list,
    strtree,
    output_name: str,
    year: str,
    only_valid: bool,
    skip_existing: bool,
) -> list[tuple]:
    """Build the list of (args) tuples for all grid tiles in one city."""
    from shapely.strtree import STRtree  # noqa: F401 — ensure import in workers

    city = city_dir.name
    grid_path = city_dir / f"{city}_grid.gpkg"
    if not grid_path.exists():
        logger.warning(f"  {city}: {grid_path.name} not found — skipping")
        return []

    grid_gdf = gpd.read_file(grid_path)
    if only_valid:
        grid_gdf = grid_gdf[grid_gdf["is_valid"]].reset_index(drop=True)

    if grid_gdf.empty:
        logger.warning(f"  {city}: no {'valid ' if only_valid else ''}tiles — skipping")
        return []

    # Reproject to EPSG:4326 for STRtree spatial query (tile index is in 4326)
    grid_4326 = grid_gdf.to_crs("EPSG:4326")
    grid_crs = str(grid_gdf.crs)

    tasks = []
    for (_, row), (_, row_4326) in zip(
        grid_gdf.iterrows(), grid_4326.iterrows()
    ):
        idxs = strtree.query(row_4326.geometry)
        if len(idxs) == 0:
            continue
        matched = [tile_paths[i] for i in idxs]
        split = row["split"]
        grid_id = int(row["grid_id"])
        out_path = (
            city_dir / output_name / year / split / f"{city}_{grid_id:02d}.npy"
        )
        tasks.append((
            city,
            grid_id,
            row["geometry"].wkt,
            grid_crs,
            matched,
            out_path,
            skip_existing,
        ))

    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embedding crops for each grid tile across all So2Sat cities."
    )
    parser.add_argument(
        "--cities-dir",
        required=True,
        type=Path,
        help="Directory containing one subfolder per city "
             "(each must have {city}_grid.gpkg produced by create_city_grids.py).",
    )
    parser.add_argument(
        "--embedding-dir",
        required=True,
        type=Path,
        help="Directory containing embedding tile files (.zarr or .tif).",
    )
    parser.add_argument(
        "--embedding-name",
        required=True,
        choices=sorted(EMBEDDING_REGISTRY),
        help="Embedding type key used to parse tile filenames.",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Sub-folder name written inside each city dir (e.g. 'AlphaEarth' or 'GeoTessera').",
    )
    parser.add_argument(
        "--year",
        required=True,
        help="Year sub-folder (e.g. '2017').",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Process only these city names (default: all).",
    )
    parser.add_argument(
        "--only-valid",
        action="store_true",
        help="Skip grid tiles where is_valid=False (low or saturated coverage).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tiles whose .npy file already exists (useful for resuming).",
    )
    args = parser.parse_args()

    # Ensure src/ is on sys.path when running directly
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    cities_dir = args.cities_dir
    if not cities_dir.exists():
        logger.error(f"cities-dir not found: {cities_dir}")
        sys.exit(1)

    city_dirs = sorted(d for d in cities_dir.iterdir() if d.is_dir())
    if args.cities:
        city_dirs = [d for d in city_dirs if d.name in args.cities]
        if not city_dirs:
            logger.error(f"None of {args.cities} found in {cities_dir}")
            sys.exit(1)

    # Build tile spatial index once for all cities
    tile_paths, strtree = _build_tile_index(args.embedding_dir, args.embedding_name, year=args.year)

    # Gather all tasks across cities
    logger.info(f"Building task list for {len(city_dirs)} cities …")
    all_tasks: list[tuple] = []
    for city_dir in city_dirs:
        tasks = _iter_city_tasks(
            city_dir,
            tile_paths,
            strtree,
            args.output_name,
            args.year,
            args.only_valid,
            args.skip_existing,
        )
        all_tasks.extend(tasks)

    logger.info(f"Total tiles to process: {len(all_tasks)}")
    if not all_tasks:
        logger.warning("Nothing to do.")
        return

    n_saved = n_skipped = n_no_coverage = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_tile, t): t[:2] for t in all_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="tile"):
                label, ok = fut.result()
                if ok:
                    n_saved += 1
                else:
                    n_no_coverage += 1
    else:
        for task in tqdm(all_tasks, unit="tile"):
            label, ok = _process_tile(task)
            if ok:
                n_saved += 1
            else:
                n_no_coverage += 1

    logger.info(
        f"\nDone — {n_saved} saved, {n_skipped} skipped (existing), "
        f"{n_no_coverage} skipped (no coverage)."
    )


if __name__ == "__main__":
    main()
