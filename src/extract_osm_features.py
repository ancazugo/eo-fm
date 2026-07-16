"""Per-patch OSM landuse features for the aux-data gate analysis.

Queries Overpass (via osmnx, cached) for landuse polygons per 0.5-degree tile
covering the So2Sat patches of the requested splits, then computes per-patch
cover fractions:

  osm_industrial_frac   landuse=industrial
  osm_commercial_frac   landuse=commercial|retail
  osm_residential_frac  landuse=residential
  osm_landuse_any_frac  any of the above (doubles as a mapping-completeness proxy)

NOTE: OSM completeness varies by city — use these as combiner features only
under the LOCO protocol, never as raw model input channels (see
docs/global_lcz_campaign_2026-07.md lessons on city-identity leakage).

Example:
    python src/extract_osm_features.py \\
        --global-gpkg ${DATA_DIR}/input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg \\
        --splits validation testing \\
        --cache-dir ${DATA_DIR}/input/aux_struct/osm_cache \\
        --output data/osm_features_valtest.parquet
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger

TILE_SIZE = 0.5
EQUAL_AREA = "EPSG:6933"
CATEGORIES = {
    "industrial": ["industrial"],
    "commercial": ["commercial", "retail"],
    "residential": ["residential"],
}
ALL_TAGS = [t for tags in CATEGORIES.values() for t in tags]


def fetch_tile_landuse(lon: float, lat: float, cache_dir: Path) -> gpd.GeoDataFrame:
    """Landuse polygons for one 0.5-degree tile, cached as GeoParquet."""
    cache = cache_dir / f"landuse_{lon}_{lat}.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)

    import osmnx as ox
    ox.settings.requests_timeout = 300
    ox.settings.cache_folder = str(cache_dir / "_overpass")
    try:
        gdf = ox.features_from_bbox(
            (lon, lat, round(lon + TILE_SIZE, 4), round(lat + TILE_SIZE, 4)),
            tags={"landuse": ALL_TAGS},
        )
    except ox._errors.InsufficientResponseError:
        gdf = gpd.GeoDataFrame({"landuse": []}, geometry=[], crs="EPSG:4326")
    if len(gdf):
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf[["landuse", "geometry"]].reset_index(drop=True)
    else:
        gdf = gpd.GeoDataFrame({"landuse": []}, geometry=[], crs="EPSG:4326")
    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache)
    logger.info(f"tile ({lon},{lat}): {len(gdf)} landuse polygons")
    return gdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--global-gpkg", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["validation", "testing"],
                        choices=["training", "validation", "testing"])
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gdf = gpd.read_file(args.global_gpkg)
    gdf = gdf[gdf["dataset"].isin(args.splits)].reset_index(drop=True)
    logger.info(f"{len(gdf)} patches in splits {args.splits}")

    cent = shapely.centroid(gdf.geometry.values)
    tile_of = list(zip(
        (np.floor(shapely.get_x(cent) / TILE_SIZE) * TILE_SIZE).round(4).tolist(),
        (np.floor(shapely.get_y(cent) / TILE_SIZE) * TILE_SIZE).round(4).tolist(),
    ))
    by_tile: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i, t in enumerate(tile_of):
        by_tile[t].append(i)
    logger.info(f"{len(by_tile)} tiles to query")

    frac = {cat: np.zeros(len(gdf)) for cat in CATEGORIES}
    for (lon, lat), idxs in sorted(by_tile.items()):
        lu = fetch_tile_landuse(lon, lat, args.cache_dir)
        if not len(lu):
            continue
        patches = gdf.geometry.iloc[idxs].to_crs(EQUAL_AREA)
        patch_area = patches.area.to_numpy()
        lu_ea = lu.to_crs(EQUAL_AREA)
        cat_of_poly = np.empty(len(lu_ea), dtype=object)
        for cat, tags in CATEGORIES.items():
            cat_of_poly[lu_ea["landuse"].isin(tags).to_numpy()] = cat

        tree = shapely.STRtree(lu_ea.geometry.values)
        pi, li = tree.query(patches.geometry.values, predicate="intersects")
        if len(pi) == 0:
            continue
        inter = shapely.area(shapely.intersection(
            patches.geometry.values[pi], lu_ea.geometry.values[li]))
        for cat in CATEGORIES:
            sel = cat_of_poly[li] == cat
            if not sel.any():
                continue
            sums = np.bincount(pi[sel], weights=inter[sel], minlength=len(idxs))
            # overlapping OSM polygons can push the sum past the patch area
            frac[cat][np.asarray(idxs)] += np.minimum(sums / patch_area, 1.0)

    out = gdf[["patch_id", "dataset"]].copy()
    out["patch_id"] = out["patch_id"].astype(str)
    for cat in CATEGORIES:
        out[f"osm_{cat}_frac"] = np.clip(frac[cat], 0.0, 1.0)
    out["osm_landuse_any_frac"] = np.clip(sum(frac.values()), 0.0, 1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    for cat in CATEGORIES:
        col = out[f"osm_{cat}_frac"]
        logger.info(f"osm_{cat}_frac: nonzero {float((col > 0).mean()):.1%}, mean {float(col.mean()):.4f}")
    logger.info(f"Saved {args.output}: {len(out)} rows")


if __name__ == "__main__":
    main()
