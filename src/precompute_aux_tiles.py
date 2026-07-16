"""Precompute merged 4-band auxiliary-structural tiles for fast patch extraction.

For each ETH canopy tile (10 m grid) this reprojects the sibling GHSL 100 m
tiles onto that grid ONCE and writes a normalised 4-band GeoTIFF:
  band 0 ANBH/50 · 1 built fraction · 2 non-res built fraction · 3 canopy/50
NaN (GEE mask / ocean) → 0.

Doing the reproject once per tile (instead of once per patch inside the tile
handler) removes the per-patch memory spike that crashed the ProcessPool and
turns extraction from ~6 patch/s into a plain raster clip.

Output: {aux_dir}/merged_aux/aux_{lon}_{lat}.tif — the embedding-dir for
`extract_so2sat_embeddings.py --embedding-name aux_struct`.

Example:
    python src/precompute_aux_tiles.py --aux-dir ${DATA_DIR}/input/aux_struct --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rioxarray as rxr
from loguru import logger
from rasterio.enums import Resampling
from tqdm import tqdm


def build_one(canopy_path: Path, out_dir: Path, skip_existing: bool) -> tuple[str, bool]:
    suffix = canopy_path.name.removeprefix("canopy_")  # {lon}_{lat}.tif
    out_path = out_dir / f"aux_{suffix}"
    if skip_existing and out_path.exists():
        return suffix, True
    root = canopy_path.parent.parent
    try:
        canopy = rxr.open_rasterio(canopy_path).astype("float32")
        builth = rxr.open_rasterio(root / "ghs_built_h" / f"builth_{suffix}")
        builts = rxr.open_rasterio(root / "ghs_built_s" / f"builts_{suffix}")
        builth = builth.rio.reproject_match(canopy, resampling=Resampling.nearest)
        builts = builts.rio.reproject_match(canopy, resampling=Resampling.nearest)

        arr = np.stack([
            builth.values[0] / 50.0,
            builts.values[0] / 10_000.0,
            builts.values[1] / 10_000.0,
            canopy.values[0] / 50.0,
        ]).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        out = canopy.isel(band=slice(0, 1)).copy()  # inherit grid/CRS/transform
        out = out.reindex(band=[0, 1, 2, 3]).astype("float32")
        out.values = arr
        # Tiled + compressed so the patch extractor can do cheap windowed reads
        # (a 33 px clip touches ~1 block instead of loading the whole ~480 MB tile).
        out.rio.to_raster(
            out_path, tiled=True, blockxsize=256, blockysize=256,
            compress="DEFLATE", predictor=2, zlevel=6, num_threads="ALL_CPUS",
        )
        return suffix, True
    except Exception as e:  # noqa: BLE001
        logger.error(f"aux tile {suffix} failed: {e}")
        return suffix, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aux-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    canopy_dir = args.aux_dir / "eth_canopy_height"
    out_dir = args.aux_dir / "merged_aux"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = sorted(canopy_dir.glob("canopy_*.tif"))
    logger.info(f"{len(tiles)} canopy tiles → merged 4-band tiles in {out_dir}")

    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_one, t, out_dir, args.skip_existing): t for t in tiles}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="tile"):
            _, ok = fut.result()
            n_ok += ok
    logger.info(f"Done: {n_ok}/{len(tiles)} merged aux tiles written")


if __name__ == "__main__":
    main()
