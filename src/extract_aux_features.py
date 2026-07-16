"""Per-patch zonal statistics from auxiliary structural rasters (GHSL, canopy).

Downloads GEE aux rasters (0.5-degree UTM tiles, cached) covering the So2Sat
patches of the requested splits, then computes per-patch zonal features:

  ghs_built_h       -> builth_mean, builth_p90, builth_frac_gt10
  ghs_built_s       -> builts_frac, builts_nres_share
  eth_canopy_height -> canopy_mean, canopy_p90, canopy_std, canopy_cover3

Output: one parquet keyed by (dataset, patch_id), consumed by
ensemble_stacking.py --aux-parquet for the LOCO gate analysis.

Example:
    python src/extract_aux_features.py \\
        --global-gpkg ${DATA_DIR}/input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg \\
        --splits validation testing \\
        --aux-dir ${DATA_DIR}/input/aux_struct \\
        --download \\
        --output data/aux_features_valtest.parquet
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from pyproj import Transformer

from datasets.downloaders import GEE_DATASET_REGISTRY, download_gee_dataset

TILE_SIZE = 0.5


def needed_tiles(gdf: gpd.GeoDataFrame) -> set[tuple[float, float]]:
    """0.5-degree grid tiles containing each patch centroid."""
    cent = gdf.geometry.centroid
    lon = (np.floor(cent.x / TILE_SIZE) * TILE_SIZE).round(4)
    lat = (np.floor(cent.y / TILE_SIZE) * TILE_SIZE).round(4)
    return set(zip(lon.tolist(), lat.tolist()))


def tile_path(aux_dir: Path, dataset: str, lon: float, lat: float) -> Path:
    prefix = GEE_DATASET_REGISTRY[dataset]["prefix"]
    return aux_dir / dataset / f"{prefix}_{lon}_{lat}.tif"


def patch_stats(dataset: str, arr: np.ndarray) -> dict[str, float]:
    """Zonal features from the (bands, h, w) window of one patch; NaN = nodata."""
    if dataset == "ghs_built_h":
        v = arr[0][np.isfinite(arr[0])]
        if v.size == 0:
            return {}
        return {
            "builth_mean": float(v.mean()),
            "builth_p90": float(np.percentile(v, 90)),
            "builth_frac_gt10": float((v > 10.0).mean()),
        }
    if dataset == "ghs_built_s":
        built = arr[0][np.isfinite(arr[0])]
        nres = arr[1][np.isfinite(arr[1])]
        if built.size == 0:
            return {}
        tot = float(built.sum())
        return {
            "builts_frac": float(built.mean() / 10_000.0),  # m2 per 100m cell -> fraction
            "builts_nres_share": float(nres.sum() / tot) if tot > 0 else 0.0,
        }
    if dataset == "eth_canopy_height":
        v = arr[0][np.isfinite(arr[0])]
        v = v[v < 255]  # ETH nodata sentinel, if unmasked
        if v.size == 0:
            return {}
        return {
            "canopy_mean": float(v.mean()),
            "canopy_p90": float(np.percentile(v, 90)),
            "canopy_std": float(v.std()),
            "canopy_cover3": float((v > 3.0).mean()),
        }
    raise ValueError(f"unknown aux dataset {dataset}")


def extract_dataset(
    gdf: gpd.GeoDataFrame, dataset: str, aux_dir: Path,
) -> pd.DataFrame:
    """Zonal stats for all patches from one aux dataset's tile directory."""
    cent = gdf.geometry.centroid
    tile_of = list(zip(
        (np.floor(cent.x / TILE_SIZE) * TILE_SIZE).round(4).tolist(),
        (np.floor(cent.y / TILE_SIZE) * TILE_SIZE).round(4).tolist(),
    ))
    by_tile: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i, t in enumerate(tile_of):
        by_tile[t].append(i)

    rows: dict[int, dict[str, float]] = {}
    missing_tiles = 0
    for (lon, lat), idxs in sorted(by_tile.items()):
        tif = tile_path(aux_dir, dataset, lon, lat)
        if not tif.exists():
            missing_tiles += 1
            logger.warning(f"{dataset}: missing tile {tif.name} ({len(idxs)} patches -> NaN)")
            continue
        with rasterio.open(tif) as src:
            data = src.read().astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            inv = ~src.transform
            h, w = data.shape[1], data.shape[2]
            for i in idxs:
                minx, miny, maxx, maxy = gdf.geometry.iloc[i].bounds
                xs, ys = tr.transform([minx, maxx], [miny, maxy])
                # xee tiles can be south-up (transform.e > 0), so order rows/cols
                ca, ra = inv * (min(xs), min(ys))
                cb, rb = inv * (max(xs), max(ys))
                r0, r1 = int(np.floor(min(ra, rb))), int(np.ceil(max(ra, rb)))
                c0, c1 = int(np.floor(min(ca, cb))), int(np.ceil(max(ca, cb)))
                r0, r1 = max(r0, 0), min(r1, h)
                c0, c1 = max(c0, 0), min(c1, w)
                if r1 <= r0 or c1 <= c0:
                    continue
                rows[i] = patch_stats(dataset, data[:, r0:r1, c0:c1])
    if missing_tiles:
        logger.warning(f"{dataset}: {missing_tiles} tiles missing")
    df = pd.DataFrame.from_dict(rows, orient="index")
    logger.info(f"{dataset}: features for {len(df)}/{len(gdf)} patches")
    return df.reindex(range(len(gdf)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--global-gpkg", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["validation", "testing"],
                        choices=["training", "validation", "testing"])
    parser.add_argument("--datasets", nargs="+",
                        default=["ghs_built_h", "ghs_built_s", "eth_canopy_height"],
                        choices=[k for k in GEE_DATASET_REGISTRY if k != "demuzere_lcz"])
    parser.add_argument("--aux-dir", required=True, type=Path,
                        help="Root dir for downloaded aux tiles ({aux_dir}/{dataset}/*.tif)")
    parser.add_argument("--download", action="store_true",
                        help="Download missing GEE tiles before extracting")
    parser.add_argument("--download-only", action="store_true",
                        help="Download missing GEE tiles for the splits' patches, then exit "
                             "(no zonal stats; --output is ignored)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.download_only and args.output is None:
        parser.error("--output is required unless --download-only")

    gdf = gpd.read_file(args.global_gpkg)
    gdf = gdf[gdf["dataset"].isin(args.splits)].reset_index(drop=True)
    logger.info(f"{len(gdf)} patches in splits {args.splits}")

    tiles = sorted(needed_tiles(gdf))
    logger.info(f"{len(tiles)} aux tiles cover the patches")

    if args.download or args.download_only:
        for dataset in args.datasets:
            out_dir = args.aux_dir / dataset
            todo = [(lon, lat) for lon, lat in tiles
                    if not tile_path(args.aux_dir, dataset, lon, lat).exists()]
            logger.info(f"{dataset}: {len(todo)} tiles to download ({len(tiles) - len(todo)} cached)")
            for lon, lat in todo:
                bbox = [lon, lat, round(lon + TILE_SIZE, 4), round(lat + TILE_SIZE, 4)]
                try:
                    download_gee_dataset(bbox, out_dir, dataset=dataset)
                except Exception as e:  # noqa: BLE001 — keep going, stats warn on missing tiles
                    logger.error(f"{dataset} tile ({lon},{lat}) failed: {e}")

    if args.download_only:
        logger.info("--download-only: done.")
        return

    out = gdf[["patch_id", "dataset"]].copy()
    out["patch_id"] = out["patch_id"].astype(str)
    for dataset in args.datasets:
        feats = extract_dataset(gdf, dataset, args.aux_dir)
        out = pd.concat([out, feats.set_axis(out.index)], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    n_full = out.drop(columns=["patch_id", "dataset"]).notna().all(axis=1).sum()
    logger.info(f"Saved {args.output}: {len(out)} rows, {out.shape[1] - 2} features, "
                f"{n_full} rows complete")


if __name__ == "__main__":
    main()
