"""Download AlphaEarth COOP tiles covering arbitrary city bboxes.

Companion to download_missing_coop_tiles.py (which is hardwired to So2Sat
training patches): queries aef_index.gpkg for the tiles intersecting the
bboxes of the requested cities (by SMOD_ID from a GUPPD bounds CSV) and
downloads the ones not already present locally.

Usage:
    python src/download_coop_tiles_bbox.py --smod-ids 30_4732 30_4693 ... \
        [--bounds-csv data/guppd_bounds.csv] [--year 2017] [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from shapely.geometry import box
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).parent))

from datasets.downloaders import download_alpha_earth_coop_tiles
from datasets.tiles import coop_wgs84_boxes, load_coop_index
from utils.paths import INPUT_DIR

COOP_DIR = INPUT_DIR / "Google" / "AlphaEarth" / "coop"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AlphaEarth COOP tiles covering city bboxes."
    )
    parser.add_argument("--smod-ids", nargs="+", required=True,
                        help="GUPPD SMOD_IDs of the cities (e.g. 30_4732).")
    parser.add_argument("--bounds-csv", default="data/guppd_bounds.csv",
                        help="CSV with SMOD_ID + minx/miny/maxx/maxy columns.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    bounds = pd.read_csv(args.bounds_csv)
    sel = bounds[bounds["SMOD_ID"].isin(args.smod_ids)]
    missing_ids = set(args.smod_ids) - set(sel["SMOD_ID"])
    if missing_ids:
        raise SystemExit(f"SMOD_IDs not found in {args.bounds_csv}: {sorted(missing_ids)}")
    logger.info(f"{len(sel)} cities: {', '.join(sel['JRC_NAME_MAIN'])}")

    logger.info(f"Loading COOP tile index from {COOP_DIR} (year={args.year}) …")
    coop_idx = load_coop_index(COOP_DIR, args.year)
    logger.info(f"  {len(coop_idx)} tiles in index")
    tree = STRtree(coop_wgs84_boxes(coop_idx))

    needed: set[str] = set()
    for city in sel.itertuples(index=False):
        city_needed, city_have = set(), 0
        for i in tree.query(box(city.minx, city.miny, city.maxx, city.maxy)):
            tile_row = coop_idx.iloc[i]
            if tile_row["is_local"]:
                city_have += 1
            else:
                city_needed.add(tile_row["path"])
        logger.info(f"  {city.JRC_NAME_MAIN}: {city_have} tiles local, {len(city_needed)} to download")
        needed |= city_needed

    if not needed:
        logger.info("All covering tiles already present.")
        return

    logger.info(f"Downloading {len(needed)} tiles …")
    n_dl, n_err = download_alpha_earth_coop_tiles(
        s3_paths=sorted(needed),
        output_dir=COOP_DIR,
        workers=args.workers,
    )
    logger.info(f"Done: {n_dl} files downloaded, {n_err} errors.")
    if n_err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
