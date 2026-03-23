import argparse
import os

import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger

from datasets.downloaders import download_tessera, download_alpha_earth


def _download_for_city(args: tuple) -> tuple[int, str | None]:
    """Download Tessera and AlphaEarth embeddings for a single city.

    Returns:
        (city_idx, error_message or None)
    """
    idx, bbox, tessera_dir, google_dir, year, output_format = args
    try:
        # download_tessera(bbox=bbox, output_dir=tessera_dir, year=year, output_format=output_format)
        download_alpha_earth(bbox=bbox, output_dir=google_dir, year=year, output_format=output_format)
        return (idx, None)
    except Exception as e:
        return (idx, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download embeddings for So2Sat cities.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Number of parallel workers. Defaults to (CPU count - 1), or 1 if unknown."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["zarr", "tif"],
        default="zarr",
        help="Output format for tiles: zarr (default) or tif.",
    )
    args = parser.parse_args()

    logger.info("Loading city boundaries")
    gdf = gpd.read_file(
        "/maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/so2sat_cities_buffer.shp"
    ).to_crs("EPSG:4326")

    year = 2017
    tessera_dir = f"/maps/acz25/phd-thesis-data/input/GeoTessera/{year}"
    google_dir = f"/maps/acz25/phd-thesis-data/input/Google/AlphaEarth/{year}"

    work_items = [
        (idx, list(row.geometry.bounds), tessera_dir, google_dir, year, args.output_format)
        for idx, row in gdf.iterrows()
    ]
    logger.info("Found {} cities to process", len(work_items))

    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 1)
    logger.info("Using {} parallel workers", n_jobs)

    success_count = 0
    error_count = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_download_for_city, item): item[0] for item in work_items}
        for future in as_completed(futures):
            city_idx = futures[future]
            try:
                idx, err = future.result()
                if err:
                    logger.error("Error downloading embeddings for city {}: {}", idx, err)
                    error_count += 1
                else:
                    logger.info("Finished downloading embeddings for city {}", idx)
                    success_count += 1
            except Exception:
                logger.exception("Unexpected error for city {}", city_idx)
                error_count += 1

    logger.info("Done: {} succeeded, {} failed", success_count, error_count)


if __name__ == "__main__":
    main()
