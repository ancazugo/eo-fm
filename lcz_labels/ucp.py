"""Stage 4b — zonal Urban Canopy Parameters (UCPs) per zone (block or patch).

The zone geometry is whatever GeoDataFrame is passed in: urban blocks (the
canonical pipeline; detected by a ``block_id`` column, keyed on it, cached as
``ucp_blocks_{hash}.parquet``, plus geometric descriptors) or the legacy 320 m
patch grid (``patch_id``/``dataset`` keying). All zonal internals take plain
polygon arrays and are geometry-agnostic.

Turns the Overture extract + per-building heights into one feature row per zone:

  bsf                  building surface fraction (ALL sources, clipped to patch)
  h_mean, h_max        footprint-area-weighted mean / max building height
  height_evidence_frac area share whose height tier is explicit/levels
  mean_footprint_area  mean single-building footprint (large-lowrise signal)
  f_water .. f_industrial_lu   land-cover area fractions (base themes)
  n_heavy_industry_poi count of works/chimney/power-plant evidence
  ghs_built_s          GHS-BUILT-S built fraction (raster cross-check)

Vector fractions use a shapely STRtree with per-patch union-of-intersection area
(so overlapping features are not double-counted). The single raster zonal
(GHS-BUILT-S) uses ``exactextract`` per 0.5-degree tile, falling back to a
rasterio window mean (the src/extract_aux_features.py formula) if exactextract
errors on a tile. Output: ``ucp_{aoi}.parquet`` keyed by ``patch_id`` (+dataset).

The land-cover class vocabularies below are semantic tag groupings (grounded in
the live Overture ``base`` schema), NOT tunable thresholds — the thresholds that
act on these fractions all live in the config.
"""

from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from loguru import logger
from pyproj import Transformer
from shapely import STRtree

from .config import LczLabelConfig
from .heights import compute_heights
from .overture import OvertureExtract

# ── Land-cover class vocabularies (Overture base theme `class` values) ────────
WATER_CLASSES = {"water", "river", "lake", "reservoir", "pond", "basin",
                 "lagoon", "wastewater", "oxbow"}
TREE_CLASSES = {"wood", "forest", "tree", "tree_row", "trees"}
LOWPLANT_CLASSES = {"grass", "grassland", "meadow", "farmland", "farmyard",
                    "village_green", "greenfield", "garden", "orchard", "cropland",
                    "allotments", "vineyard", "managed", "fairway", "green",
                    "rough", "pitch", "recreation_ground"}
SHRUB_CLASSES = {"scrub", "heath", "shrub", "shrubbery"}
SAND_CLASSES = {"sand", "beach", "dune"}
BARE_ROCK_CLASSES = {"rock", "scree", "bare_rock", "bedrock", "stone"}
PAVED_INFRA_CLASSES = {"runway", "apron", "taxiway", "helipad", "parking"}
INDUSTRIAL_LU_CLASSES = {"industrial", "works", "brownfield"}
# Heavy-industry evidence: actual works/chimney/power-plant (NOT substations or
# generators, which are common in ordinary large-lowrise/commercial areas and
# spuriously flag LCZ 8 as LCZ 10).
HEAVY_INDUSTRY_POI_CLASSES = {"works", "chimney", "power_plant", "plant"}

TILE_SIZE = 0.5


def _class_mask(gdf: gpd.GeoDataFrame, classes: set[str]) -> np.ndarray:
    if gdf.empty:
        return np.zeros(0, dtype=bool)
    c = gdf.get("class", pd.Series([None] * len(gdf))).astype("string")
    return c.isin(classes).fillna(False).to_numpy()


def _coverage_fraction(
    patch_geoms: np.ndarray, patch_areas: np.ndarray, cover_geoms: np.ndarray
) -> np.ndarray:
    """Per-patch fraction of area covered by the union of ``cover_geoms``."""
    frac = np.zeros(len(patch_geoms))
    cover_geoms = cover_geoms[shapely.area(cover_geoms) > 0] if len(cover_geoms) else cover_geoms
    if len(cover_geoms) == 0:
        return frac
    tree = STRtree(cover_geoms)
    for i, pg in enumerate(patch_geoms):
        idx = tree.query(pg, predicate="intersects")
        if len(idx) == 0:
            continue
        inter = shapely.intersection(pg, cover_geoms[idx])
        area = shapely.area(shapely.union_all(inter))
        frac[i] = area / patch_areas[i]
    return np.clip(frac, 0.0, 1.0)


def _poi_count(patch_geoms: np.ndarray, poi_geoms: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(patch_geoms), dtype=int)
    if len(poi_geoms) == 0:
        return counts
    tree = STRtree(poi_geoms)
    for i, pg in enumerate(patch_geoms):
        counts[i] = len(tree.query(pg, predicate="intersects"))
    return counts


def _building_stats(
    patch_geoms: np.ndarray, patch_areas: np.ndarray, buildings: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Per-patch building aggregates (bsf, heights, evidence, footprint, ML flag)."""
    n = len(patch_geoms)
    cols = ["bsf", "h_mean", "h_max", "height_evidence_frac", "height_none_frac",
            "mean_footprint_area", "median_footprint_area", "footprint_area_cv",
            "building_count_density", "f_google_source", "large_lowrise_frac",
            "n_tower", "built_area_ml_frac", "n_buildings"]
    out = {c: np.zeros(n) for c in cols}
    out["h_mean"][:] = np.nan
    out["h_max"][:] = np.nan
    out["median_footprint_area"][:] = np.nan
    out["footprint_area_cv"][:] = np.nan
    if buildings.empty:
        return pd.DataFrame(out)

    bgeom = buildings.geometry.values
    height = buildings["height_m"].to_numpy(dtype=float)
    tier = buildings["height_tier"].to_numpy(dtype=object)
    footprint = buildings["footprint_area_m2"].to_numpy(dtype=float)
    large = buildings["is_large_lowrise_type"].to_numpy(dtype=bool)
    tower = buildings["is_tower_type"].to_numpy(dtype=bool)
    trusted = buildings.get("is_osm_or_esri",
                            pd.Series(np.zeros(len(buildings), bool))).to_numpy(dtype=bool)
    google = buildings.get("is_google",
                           pd.Series(np.zeros(len(buildings), bool))).to_numpy(dtype=bool)
    tree = STRtree(bgeom)

    for i, pg in enumerate(patch_geoms):
        idx = tree.query(pg, predicate="intersects")
        if len(idx) == 0:
            continue
        inter_area = shapely.area(shapely.intersection(pg, bgeom[idx]))
        m = inter_area > 0
        if not m.any():
            continue
        idx, inter_area = idx[m], inter_area[m]
        tot = inter_area.sum()
        out["bsf"][i] = tot / patch_areas[i]
        out["n_buildings"][i] = len(idx)

        # Footprint morphology (LCZ 7 signals). Google Open Buildings segments
        # dense adjacent structures well; other ML sources merge them, inflating
        # the median — so median over Google footprints when they dominate.
        fp = footprint[idx]
        g_share = float(google[idx].mean())
        out["f_google_source"][i] = g_share
        fp_for_median = fp[google[idx]] if (g_share >= 0.5 and google[idx].any()) else fp
        out["mean_footprint_area"][i] = float(fp.mean())
        out["median_footprint_area"][i] = float(np.median(fp_for_median))
        out["footprint_area_cv"][i] = float(fp.std() / fp.mean()) if fp.mean() > 0 else 0.0
        out["building_count_density"][i] = len(idx) / (patch_areas[i] / 1.0e6)  # per km^2

        h = height[idx]
        hm = np.isfinite(h)
        if hm.any():
            w = inter_area[hm]
            out["h_mean"][i] = float((w * h[hm]).sum() / w.sum())
            out["h_max"][i] = float(h[hm].max())

        evid = np.isin(tier[idx], ("explicit", "levels"))
        out["height_evidence_frac"][i] = inter_area[evid].sum() / tot
        out["height_none_frac"][i] = inter_area[tier[idx] == "none"].sum() / tot
        out["large_lowrise_frac"][i] = inter_area[large[idx]].sum() / tot
        out["n_tower"][i] = int(tower[idx].sum())
        # ML-sourced (not OSM/Esri) share of built area — for the suspect-ML filter
        out["built_area_ml_frac"][i] = inter_area[~trusted[idx]].sum() / tot

    return pd.DataFrame(out)


def _road_stats(
    patch_geoms: np.ndarray, patch_areas: np.ndarray, n_buildings: np.ndarray,
    roads: gpd.GeoDataFrame | None, config: LczLabelConfig,
) -> pd.DataFrame:
    """Per-patch road_length_density (motorized km/km²) + buildings_per_road_km."""
    n = len(patch_geoms)
    road_km = np.zeros(n)
    if roads is not None and not roads.empty:
        motor = set(config.road_motorized_classes)
        rcls = roads.get("class", pd.Series([None] * len(roads))).astype("string")
        rgeom = roads.geometry.values[rcls.isin(motor).fillna(False).to_numpy()]
        if len(rgeom):
            tree = STRtree(rgeom)
            for i, pg in enumerate(patch_geoms):
                idx = tree.query(pg, predicate="intersects")
                if len(idx) == 0:
                    continue
                clipped = shapely.intersection(pg, rgeom[idx])
                road_km[i] = shapely.length(clipped).sum() / 1000.0
    area_km2 = patch_areas / 1.0e6
    road_density = np.divide(road_km, area_km2, out=np.zeros(n), where=area_km2 > 0)
    sentinel = config.router.buildings_per_road_km_sentinel
    bpr = np.where(
        road_km > 0, np.divide(n_buildings, np.maximum(road_km, 1e-9)),
        np.where(n_buildings > 0, sentinel, 0.0),
    )
    return pd.DataFrame({"road_length_density": road_density,
                         "buildings_per_road_km": bpr})


# ── GHS-BUILT-S raster zonal ──────────────────────────────────────────────────

def _ghs_tile_key(lon: float, lat: float) -> tuple[float, float]:
    return (round(float(np.floor(lon / TILE_SIZE) * TILE_SIZE), 4),
            round(float(np.floor(lat / TILE_SIZE) * TILE_SIZE), 4))


def ghs_built_s_fraction(patches_ll: gpd.GeoDataFrame, config: LczLabelConfig) -> np.ndarray:
    """GHS-BUILT-S built fraction per patch (patches in EPSG:4326).

    Uses exactextract per 0.5-degree UTM tile; on any tile-level error falls back
    to a rasterio window mean (built_surface m^2 per 100 m cell / 10000).
    """
    n = len(patches_ll)
    out = np.full(n, np.nan)
    # Bounds midpoint (avoids a geographic-CRS centroid warning; only used to
    # pick the 0.5-degree tile, so approximate centre is fine).
    b = patches_ll.geometry.bounds
    cx = (b["minx"] + b["maxx"]).to_numpy() / 2
    cy = (b["miny"] + b["maxy"]).to_numpy() / 2
    by_tile: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i in range(n):
        by_tile[_ghs_tile_key(cx[i], cy[i])].append(i)

    missing = 0
    for (lon, lat), idxs in by_tile.items():
        tif = config.rasters.ghs_built_s_dir / f"builts_{lon}_{lat}.tif"
        if not tif.exists():
            missing += 1
            continue
        sub = patches_ll.iloc[idxs]
        vals = _ghs_tile_zonal(tif, sub)
        for j, i in enumerate(idxs):
            out[i] = vals[j]
    if missing:
        logger.warning(f"GHS-BUILT-S: {missing} tiles missing (patches -> NaN)")
    return out


def _ghs_tile_zonal(tif, sub: gpd.GeoDataFrame) -> np.ndarray:
    """Mean built fraction for patches within one GHS-BUILT-S tile."""
    with rasterio.open(tif) as src:
        sub_utm = sub.to_crs(src.crs)
        try:
            from exactextract import exact_extract
            res = exact_extract(src, sub_utm, ["mean"], output="pandas")
            col = "mean" if "mean" in res.columns else "band_1_mean"
            return (res[col].to_numpy(dtype=float)) / 10_000.0
        except Exception as e:  # noqa: BLE001 — fall back to window mean
            logger.debug(f"exactextract failed on {tif.name} ({e}); window fallback")
            return _ghs_window_mean(src, sub_utm)


def _ghs_window_mean(src, sub_utm: gpd.GeoDataFrame) -> np.ndarray:
    band = src.read(1).astype(np.float64)
    if src.nodata is not None:
        band[band == src.nodata] = np.nan
    inv = ~src.transform
    h, w = band.shape
    out = np.full(len(sub_utm), np.nan)
    for k, geom in enumerate(sub_utm.geometry):
        minx, miny, maxx, maxy = geom.bounds
        ca, ra = inv * (minx, miny)
        cb, rb = inv * (maxx, maxy)
        r0, r1 = int(np.floor(min(ra, rb))), int(np.ceil(max(ra, rb)))
        c0, c1 = int(np.floor(min(ca, cb))), int(np.ceil(max(ca, cb)))
        r0, r1, c0, c1 = max(r0, 0), min(r1, h), max(c0, 0), min(c1, w)
        if r1 <= r0 or c1 <= c0:
            continue
        win = band[r0:r1, c0:c1]
        win = win[np.isfinite(win)]
        if win.size:
            out[k] = win.mean() / 10_000.0
    return out


# ── Orchestration ─────────────────────────────────────────────────────────────

def compute_ucp(
    grid: gpd.GeoDataFrame,
    extract: OvertureExtract,
    config: LczLabelConfig,
    aoi_name: str,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Compute the per-zone UCP table for one AOI (cached parquet).

    ``grid`` may be the block table (``block_id`` column -> block mode, adds
    geometric descriptors) or the legacy 320 m patch grid.
    """
    block_mode = "block_id" in grid.columns
    stem = "ucp_blocks" if block_mode else "ucp"
    cache = config.cache_dir / aoi_name / f"{stem}_{config.config_hash}.parquet"
    if cache.exists() and not force:
        logger.info(f"[{aoi_name}] UCP cache hit: {cache.name}")
        return pd.read_parquet(cache)

    utm = extract.utm_crs
    patches_utm = grid.to_crs(utm)
    pgeom = patches_utm.geometry.values
    pareas = shapely.area(pgeom)

    buildings = compute_heights(extract.buildings, config)
    lc = extract.landcover
    infra = extract.infrastructure

    if block_mode:
        df = pd.DataFrame({"block_id": grid["block_id"].to_numpy()})
    else:
        df = pd.DataFrame({"patch_id": grid["patch_id"].astype(str).to_numpy(),
                           "dataset": grid.get("dataset", pd.Series(["unlabeled"] * len(grid))).to_numpy()})

    # Building aggregates
    bstats = _building_stats(pgeom, pareas, buildings)
    df = pd.concat([df, bstats], axis=1)

    # Land-cover fractions (union-of-intersection area)
    def frac(gdf, classes):
        return _coverage_fraction(pgeom, pareas, gdf.geometry.values[_class_mask(gdf, classes)])

    df["f_water"] = frac(lc, WATER_CLASSES)
    df["f_trees"] = frac(lc, TREE_CLASSES)
    df["f_lowplants"] = frac(lc, LOWPLANT_CLASSES)
    df["f_shrub"] = frac(lc, SHRUB_CLASSES)
    df["f_sand"] = frac(lc, SAND_CLASSES)
    df["f_bare_rock"] = frac(lc, BARE_ROCK_CLASSES)
    df["f_industrial_lu"] = frac(lc, INDUSTRIAL_LU_CLASSES)

    # Paved infrastructure: aeroway/parking polygons from land-cover + infra themes
    paved_geoms = np.concatenate([
        lc.geometry.values[_class_mask(lc, PAVED_INFRA_CLASSES)],
        infra.geometry.values[_class_mask(infra, PAVED_INFRA_CLASSES)],
    ]) if not (lc.empty and infra.empty) else np.array([])
    df["f_paved_infra"] = _coverage_fraction(pgeom, pareas, paved_geoms)

    # Heavy-industry POI evidence (works polygons + power infrastructure)
    poi_geoms = np.concatenate([
        lc.geometry.values[_class_mask(lc, HEAVY_INDUSTRY_POI_CLASSES)],
        infra.geometry.values[_class_mask(infra, HEAVY_INDUSTRY_POI_CLASSES)],
    ]) if not (lc.empty and infra.empty) else np.array([])
    df["n_heavy_industry_poi"] = _poi_count(pgeom, poi_geoms)

    # Road metrics (motorized km/km² + buildings-per-road-km) for the LCZ 7 router
    rstats = _road_stats(pgeom, pareas, df["n_buildings"].to_numpy(), extract.roads, config)
    df = pd.concat([df, rstats], axis=1)

    # Million Neighborhoods informal-block coverage (NaN when the layer is absent)
    if extract.mn_blocks is not None and not extract.mn_blocks.empty:
        mn = extract.mn_blocks
        mn_informal = mn.geometry.values[mn["informal"].to_numpy(dtype=bool)]
        df["mn_informal_frac"] = _coverage_fraction(pgeom, pareas, mn_informal)
    else:
        df["mn_informal_frac"] = np.nan

    # GHS-BUILT-S cross-check (raster; tile lookup needs EPSG:4326 geometries)
    grid_ll = grid if (grid.crs and grid.crs.to_epsg() == 4326) else grid.to_crs("EPSG:4326")
    df["ghs_built_s"] = ghs_built_s_fraction(grid_ll, config)

    # Geometric descriptors (block mode): diagnostics + B3 GNN node features.
    # area_m2 stays on the block table (single source of truth).
    if block_mode:
        peri = shapely.length(pgeom)
        df["compactness"] = np.divide(4.0 * np.pi * pareas, peri**2,
                                      out=np.zeros(len(pgeom)), where=peri > 0)
        df["elongation"] = _elongation(pgeom)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    logger.info(f"[{aoi_name}] UCP: {len(df)} zones, {df.shape[1]} columns")
    return df


def _elongation(geoms: np.ndarray) -> np.ndarray:
    """1 - short/long side of the minimum rotated rectangle (0 = square-ish)."""
    out = np.zeros(len(geoms))
    rects = shapely.oriented_envelope(geoms)
    for i, r in enumerate(rects):
        coords = shapely.get_coordinates(r)
        if len(coords) < 4:
            continue
        e1 = float(np.hypot(*(coords[1] - coords[0])))
        e2 = float(np.hypot(*(coords[2] - coords[1])))
        lo, hi = min(e1, e2), max(e1, e2)
        out[i] = 1.0 - lo / hi if hi > 0 else 0.0
    return out
