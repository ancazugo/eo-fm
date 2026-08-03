"""Stage 3 — tiered per-building height model with GHS-BUILT-H raster backfill.

For each building we assign ``height_m`` and record the ``height_tier`` that
produced it (design principle 2: heights are only trusted from OSM/Esri):

  1. ``explicit`` — Overture ``height`` if sane (2-1000 m) AND ``is_osm_or_esri``.
  2. ``levels``   — ``num_floors * metres_per_floor + roof_height`` AND OSM/Esri.
  3. ``raster``   — GHS-BUILT-H ANBH (average net building height, JRC 100 m)
                    sampled at the building centroid.
  4. ``none``     — NaN.

GHS-BUILT-H download: JRC distributes 100 m tiles at
https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_H_GLOBE_R2023A/
GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100/V1-0/tiles/ (Mollweide, ~10x10 deg
tiles). This module instead reuses the repo's pre-downloaded per-0.5-degree UTM
tiles under ``rasters.ghs_built_h_dir`` (see src/datasets/downloaders.py
``ghs_built_h``); if a tile is absent, that building falls through to tier
``none`` rather than failing.
"""

from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import numpy as np
import rasterio
from loguru import logger
from pyproj import Transformer

from .config import LczLabelConfig

LARGE_LOWRISE_CLASSES = {"warehouse", "retail", "industrial", "hangar"}
TOWER_CLASSES = {"tower", "skyscraper"}


def _tile_key(lon: float, lat: float, size: float) -> tuple[float, float]:
    return (round(float(np.floor(lon / size) * size), 4),
            round(float(np.floor(lat / size) * size), 4))


def sample_ghs_built_h(points_lonlat: np.ndarray, config: LczLabelConfig) -> np.ndarray:
    """Sample GHS-BUILT-H (ANBH, metres) at each (lon, lat) point.

    ``points_lonlat`` is (N, 2). Returns (N,) float array; NaN where the tile is
    missing, the point is off-grid, or the raster is nodata. Tiles are grouped so
    each GeoTIFF is opened once (mirrors src/extract_aux_features.py).
    """
    n = len(points_lonlat)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    size = config.rasters.tile_size_deg
    prefix = "builth"
    by_tile: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i, (lon, lat) in enumerate(points_lonlat):
        by_tile[_tile_key(lon, lat, size)].append(i)

    missing = 0
    for (lon, lat), idxs in by_tile.items():
        tif = config.rasters.ghs_built_h_dir / f"{prefix}_{lon}_{lat}.tif"
        if not tif.exists():
            missing += 1
            continue
        with rasterio.open(tif) as src:
            band = src.read(1).astype(np.float64)
            if src.nodata is not None:
                band[band == src.nodata] = np.nan
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            inv = ~src.transform
            h, w = band.shape
            for i in idxs:
                lon_i, lat_i = points_lonlat[i]
                x, y = tr.transform(lon_i, lat_i)
                col, row = inv * (x, y)
                r, c = int(np.floor(row)), int(np.floor(col))
                if 0 <= r < h and 0 <= c < w:
                    out[i] = band[r, c]
    if missing:
        logger.warning(f"GHS-BUILT-H: {missing} tiles missing (buildings -> tier 'none')")
    return out


def compute_heights(
    buildings: gpd.GeoDataFrame, config: LczLabelConfig
) -> gpd.GeoDataFrame:
    """Add ``height_m``, ``height_tier``, ``footprint_area_m2`` and type flags.

    ``buildings`` must be in a metric (UTM) CRS. Returns a copy.
    """
    b = buildings.copy()
    if b.empty:
        for col, dt in [("height_m", float), ("footprint_area_m2", float)]:
            b[col] = np.array([], dtype=dt)
        b["height_tier"] = np.array([], dtype=object)
        b["is_large_lowrise_type"] = np.array([], dtype=bool)
        b["is_tower_type"] = np.array([], dtype=bool)
        return b

    n = len(b)
    height_m = np.full(n, np.nan)
    tier = np.array(["none"] * n, dtype=object)

    height = b.get("height", np.full(n, np.nan)).to_numpy(dtype=float, na_value=np.nan)
    num_floors = b.get("num_floors", np.full(n, np.nan)).to_numpy(dtype=float, na_value=np.nan)
    roof = b.get("roof_height", np.full(n, np.nan)).to_numpy(dtype=float, na_value=np.nan)
    trusted = b.get("is_osm_or_esri", np.zeros(n, dtype=bool)).to_numpy(dtype=bool, na_value=False)

    lo, hi = config.height_min_sane, config.height_max_sane

    # Tier 1: explicit height (trusted only)
    explicit = trusted & np.isfinite(height) & (height >= lo) & (height <= hi)
    height_m[explicit] = height[explicit]
    tier[explicit] = "explicit"

    # Tier 2: levels (trusted only, where not already explicit)
    levels_val = num_floors * config.metres_per_floor + np.nan_to_num(roof, nan=0.0)
    levels = (~explicit) & trusted & np.isfinite(num_floors) & (num_floors > 0)
    height_m[levels] = levels_val[levels]
    tier[levels] = "levels"

    # Tier 3: raster backfill (GHS-BUILT-H ANBH at centroid) for the rest
    remaining = ~(explicit | levels)
    if remaining.any():
        cent = b.geometry.centroid
        cent_ll = gpd.GeoSeries(cent, crs=b.crs).to_crs("EPSG:4326")
        pts = np.column_stack([cent_ll.x.to_numpy(), cent_ll.y.to_numpy()])
        anbh = sample_ghs_built_h(pts[remaining], config)
        rem_idx = np.flatnonzero(remaining)
        good = np.isfinite(anbh) & (anbh > 0)
        height_m[rem_idx[good]] = anbh[good]
        tier[rem_idx[good]] = "raster"

    b["height_m"] = height_m
    b["height_tier"] = tier
    b["footprint_area_m2"] = b.geometry.area

    cls = b.get("class", gpd.pd.Series([None] * n, index=b.index)).astype("string")
    b["is_large_lowrise_type"] = cls.isin(LARGE_LOWRISE_CLASSES).fillna(False).to_numpy()
    b["is_tower_type"] = cls.isin(TOWER_CLASSES).fillna(False).to_numpy()

    counts = {t: int((tier == t).sum()) for t in ("explicit", "levels", "raster", "none")}
    logger.info(f"Height tiers: {counts}")
    return b
