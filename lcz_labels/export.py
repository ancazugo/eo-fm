"""Stage 8 — export: bitmask contract, canonical raster grid, patch transfer.

One label contract for everything downstream (the ``lcz_train`` harness reads
nothing else):

* ``lcz_bitmask_{aoi}.tif``  — uint32; bit ``c-1`` set for every class ``c`` in
  a block's ``lcz_set`` (codes 1-17: built 1-10, natural A-G = 11-17). A hard
  label is a single bit, a coarse label several, unlabelled is 0. This single
  raster encodes hard/coarse/unlabelled uniformly.
* ``confidence_{aoi}.tif``   — uint8, confidence x 100.
* ``block_id_{aoi}.tif``     — uint32 ``block_idx`` (dense per-AOI index, 0 =
  no block); ALL blocks including unlabelled are burned so training-time
  erosion sees labelled/unlabelled boundaries too.

Rasters are written on the **canonical per-AOI grid**: local UTM at exactly
``export.raster_res_m`` (10 m, embedding-native), with the origin snapped to
resolution multiples so the grid is reproducible from the config alone. Part 2
warps embedding tiles onto this grid, which also resolves AOIs that straddle
UTM zones. No erosion at export time — erosion is a training transform.

The 320 m patch transfer (``patch_labels_{aoi}.parquet``) is the So2Sat
compatibility surface; nothing in Part 2 trains on it.
"""

from __future__ import annotations

import numpy as np
from rasterio.transform import Affine, from_origin

from .config import LczLabelConfig
from .grid import local_utm_crs, resolve_aoi_bbox

N_LCZ = 17


# ── Bitmask contract ──────────────────────────────────────────────────────────

def encode_lcz_set(lcz_sets: list[list[int] | None]) -> np.ndarray:
    """uint32 bitmask per entry: bit ``c-1`` for each class ``c`` in the set.

    ``None`` / empty sets encode to 0 (unlabelled).
    """
    out = np.zeros(len(lcz_sets), dtype=np.uint32)
    for i, s in enumerate(lcz_sets):
        if not s:
            continue
        v = 0
        for c in s:
            c = int(c)
            if not 1 <= c <= N_LCZ:
                raise ValueError(f"LCZ code {c} outside 1..{N_LCZ}")
            v |= 1 << (c - 1)
        out[i] = v
    return out


def decode_bitmask(values: np.ndarray) -> list[list[int]]:
    """Inverse of :func:`encode_lcz_set` — sorted class lists ([] for 0)."""
    return [
        [c for c in range(1, N_LCZ + 1) if int(v) >> (c - 1) & 1]
        for v in np.asarray(values, dtype=np.uint32)
    ]


# ── Canonical per-AOI raster grid ─────────────────────────────────────────────

def raster_grid(
    aoi_name: str, config: LczLabelConfig
) -> tuple[Affine, str, tuple[int, int]]:
    """(transform, utm_crs, (height, width)) of the AOI's canonical label grid.

    Local UTM at ``export.raster_res_m``, bounds = the AOI bbox projected to
    UTM and snapped outward to resolution multiples — deterministic from the
    config alone, so labels, embedding mosaics and predictions all align.
    """
    import geopandas as gpd
    from shapely.geometry import box

    aoi = config.aoi(aoi_name)
    bbox = resolve_aoi_bbox(aoi, config)
    utm = local_utm_crs(bbox, aoi.equal_area_crs)
    res = float(config.export.raster_res_m)
    ux0, uy0, ux1, uy1 = (
        gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(utm).total_bounds
    )
    minx = np.floor(ux0 / res) * res
    miny = np.floor(uy0 / res) * res
    maxx = np.ceil(ux1 / res) * res
    maxy = np.ceil(uy1 / res) * res
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    return from_origin(minx, maxy, res, res), utm, (height, width)
