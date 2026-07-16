"""Extract embedding crops for all So2Sat LCZ42 patches.

For each patch in patches_reference_rxr.gpkg the script:
  1. Finds embedding tile(s) that cover the patch (no file I/O — filename parsing only).
  2. Clips the embedding to the patch bounding box.
  3. Saves the result as a float32 .npy file.

Note: unlike infer_roi.py, extraction does NOT clip alpha_earth_coop tiles to
their valid bbox (aef_index.gpkg wgs84_* bounds), so a patch at a UTM-zone
boundary can be filled from an adjacent tile's contaminated overhang pixels.

Output layout:
    {so2sat_dir}/{split}/{output_name}/{year}/patch_{patch_id}.npy

Example:
    python src/extract_so2sat_embeddings.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017 \\
        --embedding-name alpha_earth \\
        --output-name AlphaEarth \\
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

sys.path.insert(0, str(Path(__file__).parent))

from datasets.registry import EMBEDDING_REGISTRY
from datasets.tiles import build_tile_index as _build_tile_index, crop_patch as _crop_patch

# ---------------------------------------------------------------------------
# Worker for parallel execution
# ---------------------------------------------------------------------------

def _process_patch(args: tuple) -> tuple[str, bool]:
    patch_id, geom_wkt, patch_crs, tile_paths, output_path, skip_existing, dtype = args
    from shapely import from_wkt
    patch_geom = from_wkt(geom_wkt)
    ok = _crop_patch(patch_geom, patch_crs, tile_paths, output_path, skip_existing, dtype)
    return patch_id, ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embedding crops for all So2Sat LCZ42 patches."
    )
    parser.add_argument(
        "--so2sat-dir",
        required=True,
        type=Path,
        help="Root directory of the So2Sat-LCZ42 v4 dataset "
             "(must contain patches_reference_rxr.gpkg).",
    )
    parser.add_argument(
        "--patches-file",
        type=Path,
        default=None,
        help="Optional GeoPackage with patches to process. "
             "Defaults to patches_reference_rxr.gpkg in --so2sat-dir. "
             "Use to restrict extraction to a single city, e.g. "
             "cities/London/patches_reference_London.gpkg.",
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
        help="Embedding type key (used to parse tile filenames).",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Subfolder name for the output (e.g. 'GeoTessera' or 'AlphaEarth').",
    )
    parser.add_argument(
        "--year",
        required=True,
        help="Year subfolder for the output (e.g. '2017').",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation", "testing"],
        choices=["training", "validation", "testing", "unlabeled"],
        help="Dataset splits to process (default: the three So2Sat splits; "
             "'unlabeled' for semi-supervised patch pools).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="npy dtype for saved patches (float16 halves disk use; "
             "PatchDataset casts back to float32 on load). Not allowed for "
             "seamless, whose raw VQ indices exceed float16's exact-integer "
             "range and would be silently corrupted.",
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
        help="Skip patches whose .npy output file already exists (useful for resuming).",
    )
    args = parser.parse_args()
    if args.dtype == "float16" and args.embedding_name == "seamless":
        parser.error(
            "--dtype float16 corrupts seamless ESD indices (values > 2048 are "
            "not exactly representable); use float32."
        )

    # Ensure src/ is on sys.path when running directly
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # --- Load all patches once ---
    patches_path = args.patches_file if args.patches_file is not None else args.so2sat_dir / "patches_reference_rxr.gpkg"
    if not patches_path.exists():
        logger.error(f"Patches file not found at {patches_path}")
        sys.exit(1)

    logger.info(f"Loading patches from {patches_path} …")
    all_patches = gpd.read_file(patches_path)
    logger.info(f"  {len(all_patches)} patches total (crs: {all_patches.crs})")

    # --- Build tile spatial index (once for all splits) ---
    tile_paths, tree = _build_tile_index(args.embedding_dir, args.embedding_name, year=args.year)
    patch_crs = str(all_patches.crs)  # EPSG:4326

    # --- Process each split ---
    for split in args.splits:
        split_patches = all_patches[all_patches["dataset"] == split].reset_index(drop=True)
        if split_patches.empty:
            logger.warning(f"No patches found for split '{split}' — skipping.")
            continue

        out_dir = args.so2sat_dir / split / args.output_name / args.year
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"\n[{split}] {len(split_patches)} patches → {out_dir}"
        )

        n_saved = 0
        n_skipped = 0

        # Sort patches by centroid so tiles in the OS page cache are shared
        # across workers processing adjacent patches (critical for tesserav1.1
        # where each tile is ~110 MB and loaded fresh from npy each call).
        if args.embedding_name in ("tesserav1.1", "tesserav1.1_global", "aux_struct"):
            cx = split_patches.geometry.centroid.x
            cy = split_patches.geometry.centroid.y
            split_patches = split_patches.iloc[
                (cx + cy * 1000).argsort().values
            ].reset_index(drop=True)

        if args.workers > 1:
            # Build task list: only patches that intersect at least one tile
            tasks = []
            for row in split_patches.itertuples(index=False):
                patch_geom = row.geometry
                idxs = tree.query(patch_geom)
                if len(idxs) == 0:
                    n_skipped += 1
                    continue
                matched_paths = [tile_paths[i] for i in idxs]
                output_path = out_dir / f"patch_{row.patch_id}.npy"
                tasks.append((
                    row.patch_id,
                    patch_geom.wkt,
                    patch_crs,
                    matched_paths,
                    output_path,
                    args.skip_existing,
                    args.dtype,
                ))

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_process_patch, t): t[0] for t in tasks}
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=split,
                    unit="patch",
                ):
                    _, ok = fut.result()
                    if ok:
                        n_saved += 1
                    else:
                        n_skipped += 1

        else:
            for row in tqdm(
                split_patches.itertuples(index=False),
                total=len(split_patches),
                desc=split,
                unit="patch",
            ):
                patch_geom = row.geometry
                idxs = tree.query(patch_geom)
                if len(idxs) == 0:
                    n_skipped += 1
                    continue

                matched_paths = [tile_paths[i] for i in idxs]
                output_path = out_dir / f"patch_{row.patch_id}.npy"
                ok = _crop_patch(patch_geom, patch_crs, matched_paths, output_path,
                                 args.skip_existing, args.dtype)
                if ok:
                    n_saved += 1
                else:
                    n_skipped += 1

        logger.info(
            f"[{split}] done — {n_saved} saved, {n_skipped} skipped (no coverage)"
        )

    logger.info("All splits complete.")


if __name__ == "__main__":
    main()
