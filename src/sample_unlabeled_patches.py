"""Sample unlabeled patches from Tessera tiles, weakly labeled by Demuzere LCZ.

Builds the unlabeled pool for semi-supervised (noisy-student) training:
candidate 320 m patch boxes are placed on a stride grid inside each Tessera
v1.1 0.1-degree tile, weakly labeled with the Demuzere et al. 2022 global
100 m LCZ map (majority vote over the ~3x3 pixels each patch covers), and
filtered by label purity, the map's per-pixel classification probability,
and non-overlap with the labeled So2Sat patches.

Class balance: rare LCZ classes are kept uncapped; all others stop at
--class-cap. Tiles are processed in seeded-random order until --n-patches
accepted patches exist (or tiles run out), with a per-tile/per-class cap for
geographic spread.

Output: a GeoPackage compatible with extract_so2sat_embeddings.py and
datasets/so2sat.py (columns: patch_id, dataset='unlabeled', LCZ_class,
demuzere_purity, demuzere_prob, tile_name, geometry EPSG:4326).

Example:
    python src/sample_unlabeled_patches.py \\
        --output data/patches_reference_unlabeled.gpkg \\
        --exclude-gpkg ${DATA_DIR}/input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg \\
        --n-patches 300000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from loguru import logger
from rasterio.windows import from_bounds
from shapely.geometry import box
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).parent))

_M_PER_DEG_LAT = 111_320.0
# LCZ classes that are rare in So2Sat's global train split (LCZ15 = 0.7%) or
# systematically weak for the current models; kept uncapped during sampling.
DEFAULT_RARE_CLASSES = (1, 4, 7, 10, 15, 16)


def candidate_stats(
    lcz: np.ndarray,
    prob: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    half_px: int,
    num_classes: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized majority vote over the (2*half_px+1)^2 window of each candidate.

    Returns (mode 1-17 or 0 if window all-nodata, purity in [0,1] over valid
    pixels, mean probability over valid pixels).
    """
    offs = np.arange(-half_px, half_px + 1)
    dy, dx = np.meshgrid(offs, offs, indexing="ij")
    wy = iy[:, None] + dy.ravel()[None, :]          # (n_cand, win)
    wx = ix[:, None] + dx.ravel()[None, :]
    wy = np.clip(wy, 0, lcz.shape[0] - 1)
    wx = np.clip(wx, 0, lcz.shape[1] - 1)
    win = lcz[wy, wx]                                # (n_cand, win) uint8
    win_prob = prob[wy, wx].astype(np.float32)

    valid = win > 0
    n_valid = valid.sum(axis=1)
    counts = (win[:, :, None] == np.arange(1, num_classes + 1)[None, None, :]).sum(axis=1)
    mode = counts.argmax(axis=1) + 1                 # (n_cand,) 1-17
    mode_count = counts.max(axis=1)

    purity = np.where(n_valid > 0, mode_count / np.maximum(n_valid, 1), 0.0)
    mean_prob = np.where(
        n_valid > 0,
        (win_prob * valid).sum(axis=1) / np.maximum(n_valid, 1),
        0.0,
    )
    mode = np.where(n_valid > 0, mode, 0)
    return mode, purity, mean_prob


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample Demuzere-weakly-labeled unlabeled patches from Tessera tiles."
    )
    parser.add_argument("--tiles-gpkg", type=Path,
                        default=Path("data/tessera_v1.1_global_2017_tiles.gpkg"),
                        help="Tessera tile footprints (tile_name + geometry, EPSG:4326).")
    parser.add_argument("--demuzere-dir", type=Path,
                        default=Path("/maps/acz25/phd-thesis-data/input/Demuzere_2022_complete"),
                        help="Directory with lcz_filter_v3.tif and lcz_probability_v3.tif.")
    parser.add_argument("--exclude-gpkg", type=Path, default=None,
                        help="Patches to exclude by overlap (e.g. So2Sat patches_reference_rxr.gpkg).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output GeoPackage path.")
    parser.add_argument("--n-patches", type=int, default=300_000)
    parser.add_argument("--patch-size-m", type=float, default=320.0)
    parser.add_argument("--stride-m", type=float, default=480.0,
                        help="Candidate grid stride (default 480 m: no candidate overlap).")
    parser.add_argument("--min-purity", type=float, default=0.65,
                        help="Min fraction of the Demuzere window agreeing with the mode.")
    parser.add_argument("--min-prob", type=float, default=50.0,
                        help="Min mean Demuzere classification probability (0-100).")
    parser.add_argument("--class-cap", type=int, default=None,
                        help="Max accepted patches per non-rare class "
                             "(default: 2 * n_patches / 17).")
    parser.add_argument("--rare-classes", type=int, nargs="+",
                        default=list(DEFAULT_RARE_CLASSES),
                        help="LCZ classes exempt from --class-cap.")
    parser.add_argument("--tile-class-cap", type=int, default=40,
                        help="Max accepted patches per class per tile (geographic spread).")
    parser.add_argument("--dry-run", type=int, default=None, metavar="N_TILES",
                        help="Process only N tiles and report yield statistics.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    class_cap = args.class_cap or (2 * args.n_patches) // 17
    rare = set(args.rare_classes)
    rng = np.random.default_rng(args.seed)

    tiles = gpd.read_file(args.tiles_gpkg)
    tiles = tiles.iloc[rng.permutation(len(tiles))].reset_index(drop=True)
    if args.dry_run:
        tiles = tiles.iloc[: args.dry_run]
    logger.info(f"Tiles to process: {len(tiles)}  target: {args.n_patches} patches  "
                f"class cap: {class_cap} (rare {sorted(rare)} uncapped)")

    exclude_tree = None
    if args.exclude_gpkg is not None:
        excl = gpd.read_file(args.exclude_gpkg)
        exclude_tree = STRtree(excl.geometry.values)
        logger.info(f"Excluding overlap with {len(excl)} patches from {args.exclude_gpkg.name}")

    lcz_src = rasterio.open(args.demuzere_dir / "lcz_filter_v3.tif")
    prob_src = rasterio.open(args.demuzere_dir / "lcz_probability_v3.tif")

    class_counts: Counter = Counter()
    records: list[dict] = []
    n_cand_total = n_rejected_quality = n_rejected_overlap = n_rejected_cap = 0

    for t_i, tile in enumerate(tiles.itertuples(index=False)):
        if len(records) >= args.n_patches and not args.dry_run:
            break
        minx, miny, maxx, maxy = tile.geometry.bounds
        mid_lat = 0.5 * (miny + maxy)
        m_per_deg_lon = _M_PER_DEG_LAT * max(np.cos(np.radians(mid_lat)), 0.05)

        # Candidate centers on a stride grid, inset by half a patch
        half_lat = 0.5 * args.patch_size_m / _M_PER_DEG_LAT
        half_lon = 0.5 * args.patch_size_m / m_per_deg_lon
        step_lat = args.stride_m / _M_PER_DEG_LAT
        step_lon = args.stride_m / m_per_deg_lon
        lats = np.arange(miny + half_lat, maxy - half_lat, step_lat)
        lons = np.arange(minx + half_lon, maxx - half_lon, step_lon)
        if len(lats) == 0 or len(lons) == 0:
            continue
        glon, glat = np.meshgrid(lons, lats)
        glon, glat = glon.ravel(), glat.ravel()
        n_cand_total += len(glon)

        # One window read per raster per tile
        try:
            win = from_bounds(minx, miny, maxx, maxy, lcz_src.transform)
            lcz = lcz_src.read(1, window=win, boundless=True, fill_value=0)
            prob = prob_src.read(1, window=win, boundless=True, fill_value=0)
        except Exception as e:
            logger.warning(f"{tile.tile_name}: Demuzere read failed ({e}) — skipping")
            continue
        if lcz.max() == 0:
            continue
        win_transform = lcz_src.window_transform(win)
        inv = ~win_transform
        cols, rows_ = inv * (glon, glat)
        ix = np.clip(cols.astype(int), 0, lcz.shape[1] - 1)
        iy = np.clip(rows_.astype(int), 0, lcz.shape[0] - 1)

        half_px = max(int(round(0.5 * args.patch_size_m / 100.0)), 1)
        mode, purity, mean_prob = candidate_stats(lcz, prob, iy, ix, half_px)

        ok = (mode > 0) & (purity >= args.min_purity) & (mean_prob >= args.min_prob)
        n_rejected_quality += int((~ok).sum())
        idxs = np.flatnonzero(ok)
        rng.shuffle(idxs)

        tile_counts: Counter = Counter()
        for i in idxs:
            cls = int(mode[i])
            if tile_counts[cls] >= args.tile_class_cap:
                n_rejected_cap += 1
                continue
            if cls not in rare and class_counts[cls] >= class_cap:
                n_rejected_cap += 1
                continue
            geom = box(glon[i] - half_lon, glat[i] - half_lat,
                       glon[i] + half_lon, glat[i] + half_lat)
            if exclude_tree is not None and len(exclude_tree.query(geom)) > 0:
                n_rejected_overlap += 1
                continue
            records.append(dict(
                patch_id=f"{len(records):07d}",
                dataset="unlabeled",
                LCZ_class=cls,
                demuzere_purity=float(purity[i]),
                demuzere_prob=float(mean_prob[i]),
                tile_name=tile.tile_name,
                geometry=geom,
            ))
            class_counts[cls] += 1
            tile_counts[cls] += 1

        if (t_i + 1) % 200 == 0:
            logger.info(f"  {t_i + 1}/{len(tiles)} tiles — {len(records)} accepted")

    lcz_src.close()
    prob_src.close()

    logger.info(f"Candidates: {n_cand_total}  rejected: quality {n_rejected_quality}, "
                f"caps {n_rejected_cap}, So2Sat overlap {n_rejected_overlap}")
    logger.info("Accepted per class: "
                + ", ".join(f"LCZ{c}: {class_counts[c]}" for c in sorted(class_counts)))

    if not records:
        logger.error("No patches accepted — check thresholds/paths.")
        sys.exit(1)

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.output, driver="GPKG")
    logger.info(f"Wrote {len(gdf)} patches to {args.output}")


if __name__ == "__main__":
    main()
