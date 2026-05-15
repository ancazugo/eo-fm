"""Download COOP tiles needed for missing So2Sat training patch extractions.

Finds training patches not yet extracted to training/AlphaEarthCoop/2017/,
queries aef_index.gpkg for the tiles that cover them, and downloads only the
tile files that are not already present locally.

Usage:
    python src/download_missing_coop_tiles.py [--workers N] [--year YEAR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely.geometry import box
from shapely.strtree import STRtree
from tqdm import tqdm

# Ensure src/ on path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from datasets.downloaders import download_alpha_earth_coop_tiles
from utils.paths import INPUT_DIR

SO2SAT_DIR = INPUT_DIR / "So2Sat-LCZ42" / "v4"
COOP_DIR = INPUT_DIR / "Google" / "AlphaEarth" / "coop"
_S3_PREFIX = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download missing AlphaEarth COOP tiles for So2Sat training patches."
    )
    parser.add_argument("--workers", type=int, default=8, help="Download threads (default: 8).")
    parser.add_argument("--year", type=int, default=2017, help="Embedding year (default: 2017).")
    args = parser.parse_args()

    # 1. Load training patches
    patches_path = SO2SAT_DIR / "patches_reference_rxr.gpkg"
    logger.info(f"Loading patches from {patches_path} …")
    patches = gpd.read_file(patches_path)
    train = patches[patches["dataset"] == "training"].reset_index(drop=True)
    logger.info(f"  {len(train)} training patches total")

    # 2. Find missing patch IDs
    extract_dir = SO2SAT_DIR / "training" / "AlphaEarthCoop" / str(args.year)
    extracted_ids = {p.stem.replace("patch_", "") for p in extract_dir.glob("patch_*.npy")}
    missing = train[~train["patch_id"].astype(str).isin(extracted_ids)]
    logger.info(f"  {len(missing)} patches not yet extracted")

    if missing.empty:
        logger.info("Nothing to do — all training patches already extracted.")
        return

    # 3. Build COOP tile spatial index
    index_path = COOP_DIR / "aef_index.gpkg"
    logger.info(f"Loading COOP tile index from {index_path} (year={args.year}) …")
    coop_idx = gpd.read_file(index_path, where=f"year = {args.year}")
    logger.info(f"  {len(coop_idx)} tiles in index")

    geoms = [
        box(r.wgs84_west, r.wgs84_south, r.wgs84_east, r.wgs84_north)
        for _, r in coop_idx.iterrows()
    ]
    tree = STRtree(geoms)

    # 4. Collect S3 paths of tiles that cover missing patches but aren't local
    needed: set[str] = set()
    for row in tqdm(missing.itertuples(index=False), total=len(missing), desc="scanning tiles"):
        for i in tree.query(row.geometry):
            tile_row = coop_idx.iloc[i]
            local = COOP_DIR / tile_row["path"].removeprefix(_S3_PREFIX)
            if not local.exists():
                needed.add(tile_row["path"])

    logger.info(f"  {len(needed)} unique tiles to download")

    if not needed:
        logger.info("All covering tiles are already present — re-run extract_so2sat_embeddings.py with --skip-existing.")
        return

    # 5. Download
    n_dl, n_err = download_alpha_earth_coop_tiles(
        s3_paths=list(needed),
        output_dir=COOP_DIR,
        workers=args.workers,
    )
    logger.info(f"Done: {n_dl} files downloaded, {n_err} errors.")
    logger.info(
        "Next step: re-run extract_so2sat_embeddings.py --splits training --skip-existing "
        "for alpha_earth_coop to extract the newly available patches."
    )


if __name__ == "__main__":
    main()
