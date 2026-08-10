"""Report how much of a patch set an embedding's tiles actually cover.

Uses the same test extract_so2sat_embeddings.py applies: query the embedding's
tile index (datasets.tiles.build_tile_index) with each patch geometry, then
check whether the patch is covered by the union of the matched tile footprints.
Every patch is classified as

    full      every pixel backed by a tile → extracted as a complete crop
    partial   straddles a missing tile → truncated crop (skipped by the
              extractor's --skip-partial-coverage)
    none      no tile at all → never written

Patches are grouped by city using the reference bounds in
data/so2sat_guppd_bounds.csv.

Example:
    python src/check_embedding_coverage.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --embedding-dir /tessera/v2/large_student --embedding-name tesserav2 \\
        --year 2017 --out data/tessera_v2_2017_so2sat_coverage.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))

from datasets.registry import EMBEDDING_REGISTRY
from datasets.tiles import build_tile_index

CITY_BOUNDS = Path(__file__).resolve().parent.parent / "data" / "so2sat_guppd_bounds.csv"

FULL, PARTIAL, NONE = 2, 1, 0


def classify_coverage(geoms, tree) -> np.ndarray:
    """Per-patch coverage status against an STRtree of tile footprints."""
    idx_patch, idx_tile = tree.query(geoms, predicate="intersects")
    tile_geoms = tree.geometries
    n = len(geoms)

    order = np.argsort(idx_patch, kind="stable")
    idx_patch, idx_tile = idx_patch[order], idx_tile[order]
    starts = np.searchsorted(idx_patch, np.arange(n), side="left")
    ends = np.searchsorted(idx_patch, np.arange(n), side="right")

    status = np.full(n, NONE, dtype=np.int8)
    single = np.where((ends - starts) == 1)[0]
    if len(single):
        covered = shapely.covered_by(geoms[single], tile_geoms[idx_tile[starts[single]]])
        status[single] = np.where(covered, FULL, PARTIAL)
    for i in np.where((ends - starts) > 1)[0]:
        union = shapely.union_all(tile_geoms[idx_tile[starts[i]:ends[i]]])
        status[i] = FULL if shapely.covered_by(geoms[i], union) else PARTIAL
    return status


def assign_cities(patches: gpd.GeoDataFrame) -> pd.Series:
    """City name per patch, from the So2Sat reference bounding boxes."""
    bounds = pd.read_csv(CITY_BOUNDS)
    cities = gpd.GeoDataFrame(
        bounds,
        geometry=[box(r.minx, r.miny, r.maxx, r.maxy) for r in bounds.itertuples()],
        crs="EPSG:4326",
    )
    points = patches.copy()
    points["geometry"] = patches.geometry.representative_point()
    joined = gpd.sjoin(
        points, cities[["JRC_NAME_MAIN", "geometry"]], how="left", predicate="within"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    return pd.Series(
        joined["JRC_NAME_MAIN"].fillna("<outside city bboxes>").values, index=patches.index
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--so2sat-dir", required=True, type=Path,
                        help="So2Sat-LCZ42 v4 root (holds patches_reference_rxr.gpkg).")
    parser.add_argument("--patches-file", type=Path, default=None,
                        help="Patch GeoPackage (default: patches_reference_rxr.gpkg).")
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--embedding-name", required=True, choices=sorted(EMBEDDING_REGISTRY))
    parser.add_argument("--year", default=None, help="Tile year (required by some embeddings).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the per-city table to this CSV.")
    parser.add_argument("--patches-out", type=Path, default=None,
                        help="Write the per-patch status (patch_id, dataset, city, status) "
                             "to this CSV.")
    parser.add_argument("--missing-tiles-out", type=Path, default=None,
                        help="Write the 0.1-degree tile names that patches need but the "
                             "embedding does not have, to this text file.")
    args = parser.parse_args()

    patches_path = args.patches_file or args.so2sat_dir / "patches_reference_rxr.gpkg"
    if not patches_path.exists():
        logger.error(f"Patches file not found at {patches_path}")
        sys.exit(1)

    logger.info(f"Loading patches from {patches_path} …")
    patches = gpd.read_file(patches_path)
    logger.info(f"  {len(patches)} patches (crs: {patches.crs})")

    tile_paths, tree = build_tile_index(args.embedding_dir, args.embedding_name, year=args.year)

    status = classify_coverage(patches.geometry.values, tree)
    patches["city"] = assign_cities(patches)
    patches["status"] = status

    labels = {FULL: "full", PARTIAL: "partial", NONE: "none"}
    rows = []
    for city, group in patches.groupby("city"):
        rows.append({
            "city": city,
            "patches": len(group),
            "full": int((group.status == FULL).sum()),
            "partial": int((group.status == PARTIAL).sum()),
            "none": int((group.status == NONE).sum()),
            "full_pct": round(100 * (group.status == FULL).mean(), 2),
        })
    table = pd.DataFrame(rows).sort_values("full_pct", ascending=False)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))
    print(
        f"\nTOTAL {len(patches)} patches — "
        f"full {int((status == FULL).sum())} ({100 * (status == FULL).mean():.2f}%), "
        f"partial {int((status == PARTIAL).sum())} ({100 * (status == PARTIAL).mean():.2f}%), "
        f"none {int((status == NONE).sum())} ({100 * (status == NONE).mean():.2f}%)"
    )
    print(f"Tiles in index: {len(tile_paths)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        logger.info(f"Per-city table → {args.out}")

    if args.patches_out:
        args.patches_out.parent.mkdir(parents=True, exist_ok=True)
        out = patches[["patch_id", "dataset", "city"]].copy()
        out["status"] = [labels[s] for s in status]
        out.to_csv(args.patches_out, index=False)
        logger.info(f"Per-patch statuses → {args.patches_out}")

    if args.missing_tiles_out:
        # 0.1-degree cells the patches need, minus the ones the index holds.
        b = patches.geometry.bounds
        needed = set()
        for minx, miny, maxx, maxy in b.itertuples(index=False):
            for i in range(int(np.floor(minx / 0.1)), int(np.floor(maxx / 0.1)) + 1):
                for j in range(int(np.floor(miny / 0.1)), int(np.floor(maxy / 0.1)) + 1):
                    needed.add(f"grid_{round(i * 0.1 + 0.05, 2):g}_{round(j * 0.1 + 0.05, 2):g}")
        # Tile names carry decimal points (grid_-0.05_51.45), so strip only a
        # known raster extension — never split on ".".
        present = set()
        for path in tile_paths:
            name = Path(path).name
            for ext in (".tiff", ".tif", ".zarr", ".npy"):
                if name.endswith(ext):
                    name = name[: -len(ext)]
                    break
            present.add(name)
        missing = sorted(needed - present)
        args.missing_tiles_out.parent.mkdir(parents=True, exist_ok=True)
        args.missing_tiles_out.write_text("\n".join(missing) + "\n")
        logger.info(
            f"{len(missing)} of {len(needed)} needed tiles missing → {args.missing_tiles_out}"
        )


if __name__ == "__main__":
    main()
