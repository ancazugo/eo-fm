import argparse
import os
from pathlib import Path

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger

from datasets.downloaders import download_tessera, download_alpha_earth, download_alpha_earth_coop
from utils.paths import TESSERA_DIR, ALPHA_EARTH_DIR

# Repo root relative to this file (src/)
_REPO_ROOT = Path(__file__).parent.parent


def _download_tessera_city(args: tuple) -> tuple[int, str, str | None]:
    idx, city_name, bbox, output_dir, year, output_format = args
    try:
        download_tessera(bbox=bbox, output_dir=output_dir, year=year, output_format=output_format,
                         cache_dir=Path(output_dir).parent / ".geotessera_cache")
        return (idx, city_name, None)
    except Exception as e:
        return (idx, city_name, str(e))


def _download_alpha_earth_city(args: tuple) -> tuple[int, str, str | None]:
    idx, city_name, bbox, output_dir, year, output_format = args
    try:
        download_alpha_earth(bbox=bbox, output_dir=output_dir, year=year, output_format=output_format)
        return (idx, city_name, None)
    except Exception as e:
        return (idx, city_name, str(e))


def _download_coop_city(args: tuple) -> tuple[int, str, str | None]:
    idx, city_name, bbox, output_dir, index_path, year = args
    try:
        download_alpha_earth_coop(
            index_path=index_path,
            output_dir=output_dir,
            bbox=tuple(bbox),
            year=year,
            workers=4,
            overwrite=False,
        )
        return (idx, city_name, None)
    except Exception as e:
        return (idx, city_name, str(e))


def _run_parallel(work_items: list, worker_fn, n_jobs: int, label: str) -> tuple[int, int]:
    success, errors = 0, 0
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(worker_fn, item): item for item in work_items}
        for future in as_completed(futures):
            try:
                idx, city_name, err = future.result()
                if err:
                    logger.error("[{}] Error for {} ({}): {}", label, city_name, idx, err)
                    errors += 1
                else:
                    logger.info("[{}] Done: {} ({})", label, city_name, idx)
                    success += 1
            except Exception:
                item = futures[future]
                logger.exception("[{}] Unexpected error for {} ({})", label, item[1], item[0])
                errors += 1
    return success, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Download embeddings for GUPPD cities.")
    parser.add_argument(
        "--tessera-jobs",
        type=int,
        default=None,
        help="Parallel workers for Tessera downloads (default: CPU count - 1).",
    )
    parser.add_argument(
        "--alpha-earth-jobs",
        type=int,
        default=2,
        help="Parallel workers for AlphaEarth/GEE downloads (default: 2, GEE rate-limited).",
    )
    parser.add_argument(
        "--output-format",
        choices=["zarr", "tif"],
        default="zarr",
        help="Output format for tiles: zarr (default) or tif.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2017,
        help="Year of embeddings to download (default: 2017).",
    )
    parser.add_argument(
        "--coop-jobs",
        type=int,
        default=4,
        help="Parallel workers for AlphaEarth coop downloads (default: 4).",
    )
    parser.add_argument(
        "--skip-tessera",
        action="store_true",
        help="Skip Tessera downloads.",
    )
    parser.add_argument(
        "--skip-alpha-earth",
        action="store_true",
        help="Skip GEE AlphaEarth downloads.",
    )
    parser.add_argument(
        "--skip-coop",
        action="store_true",
        help="Skip AlphaEarth coop downloads.",
    )
    parser.add_argument(
        "--bounds-csv",
        type=Path,
        default=_REPO_ROOT / "data" / "so2sat_guppd_bounds.csv",
        help="CSV with city bboxes (JRC_NAME_MAIN, minx, miny, maxx, maxy). "
             "Default: data/so2sat_guppd_bounds.csv",
    )
    args = parser.parse_args()

    logger.info("Loading city boundaries from {}", args.bounds_csv)
    df = pd.read_csv(args.bounds_csv)
    # columns: SMOD_ID, JRC_NAME_MAIN, minx, miny, maxx, maxy

    year = args.year
    tessera_dir = TESSERA_DIR / str(year)
    alpha_earth_dir = ALPHA_EARTH_DIR / str(year)
    coop_dir = ALPHA_EARTH_DIR / "coop"

    cities = [
        (idx, row["JRC_NAME_MAIN"], [row["minx"], row["miny"], row["maxx"], row["maxy"]])
        for idx, row in df.iterrows()
    ]
    logger.info("Found {} cities to process", len(cities))

    # --- Phase 1: Tessera ---
    if not args.skip_tessera:
        n_tessera = args.tessera_jobs or max(1, (os.cpu_count() or 4) - 1)
        logger.info("Phase 1/3: Tessera with {} workers", n_tessera)
        tessera_items = [
            (idx, city_name, bbox, tessera_dir, year, args.output_format)
            for idx, city_name, bbox in cities
        ]
        t_ok, t_err = _run_parallel(tessera_items, _download_tessera_city, n_tessera, "Tessera")
        logger.info("Tessera complete: {} succeeded, {} failed", t_ok, t_err)

    # --- Phase 2: AlphaEarth (GEE) ---
    if not args.skip_alpha_earth:
        n_alpha = args.alpha_earth_jobs
        logger.info("Phase 2/3: AlphaEarth (GEE) with {} workers", n_alpha)
        alpha_items = [
            (idx, city_name, bbox, alpha_earth_dir, year, args.output_format)
            for idx, city_name, bbox in cities
        ]
        a_ok, a_err = _run_parallel(alpha_items, _download_alpha_earth_city, n_alpha, "AlphaEarth")
        logger.info("AlphaEarth complete: {} succeeded, {} failed", a_ok, a_err)

    # --- Phase 3: AlphaEarth coop ---
    if not args.skip_coop:
        index_path = coop_dir / "aef_index.gpkg"
        if not index_path.exists():
            logger.error("aef_index.gpkg not found at {} — skipping coop phase", index_path)
        else:
            n_coop = args.coop_jobs
            logger.info("Phase 3/3: AlphaEarth coop with {} workers", n_coop)
            coop_items = [
                (idx, city_name, bbox, coop_dir, index_path, year)
                for idx, city_name, bbox in cities
            ]
            c_ok, c_err = _run_parallel(coop_items, _download_coop_city, n_coop, "Coop")
            logger.info("Coop complete: {} succeeded, {} failed", c_ok, c_err)


if __name__ == "__main__":
    main()
