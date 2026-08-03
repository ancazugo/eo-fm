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

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from loguru import logger
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from shapely import STRtree

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
        # lcz_set round-trips from parquet as a numpy array, not a list — avoid
        # `if not s` (ambiguous truth value for multi-element arrays).
        if s is None or len(s) == 0:
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


# ── Raster products ───────────────────────────────────────────────────────────

def _write_tif(path: Path, arr: np.ndarray, transform: Affine, crs: str) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype.name, crs=crs, transform=transform,
        nodata=0, compress="lzw", tiled=True,
    ) as dst:
        dst.write(arr, 1)
    return path


def write_rasters(
    labels: gpd.GeoDataFrame, aoi_name: str, config: LczLabelConfig
) -> dict[str, Path]:
    """Burn the three Stage 8 rasters on the canonical grid (no erosion).

    ``labels`` is the labelled block table (``block_idx, label_type, lcz_set,
    confidence`` + geometry). Unlabelled blocks stay 0 in the bitmask and
    confidence rasters but ARE burned into the ``block_id`` index raster.
    """
    transform, utm, (h, w) = raster_grid(aoi_name, config)
    g = labels if str(labels.crs) == utm else labels.to_crs(utm)
    out_dir = config.cache_dir / aoi_name
    out_dir.mkdir(parents=True, exist_ok=True)

    labelled = g[g["label_type"].isin(["hard", "coarse"])]
    kw = dict(out_shape=(h, w), transform=transform, fill=0, all_touched=False)
    if len(labelled):
        bits = encode_lcz_set(list(labelled["lcz_set"]))
        bitmask = rasterize(zip(labelled.geometry.values, bits.tolist()),
                            dtype="uint32", **kw)
        conf100 = np.clip(np.round(labelled["confidence"].to_numpy() * 100), 0, 100)
        conf = rasterize(zip(labelled.geometry.values, conf100.astype(np.uint8).tolist()),
                         dtype="uint8", **kw)
    else:
        bitmask = np.zeros((h, w), dtype=np.uint32)
        conf = np.zeros((h, w), dtype=np.uint8)
    block_idx = rasterize(zip(g.geometry.values, g["block_idx"].astype("uint32").tolist()),
                          dtype="uint32", **kw)

    paths = {
        "bitmask": _write_tif(out_dir / f"lcz_bitmask_{aoi_name}.tif", bitmask, transform, utm),
        "confidence": _write_tif(out_dir / f"confidence_{aoi_name}.tif", conf, transform, utm),
        "block_id": _write_tif(out_dir / f"block_id_{aoi_name}.tif", block_idx, transform, utm),
    }
    lab_px = int((bitmask != 0).sum())
    logger.info(f"[{aoi_name}] rasters {h}x{w} @ {config.export.raster_res_m} m ({utm}): "
                f"{lab_px / 1e6:.1f} Mpx labelled ({lab_px / (h * w):.0%})")
    return paths


# ── Patch transfer (So2Sat 320 m compatibility surface) ──────────────────────

def patch_transfer(
    labels: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    aoi_name: str,
    config: LczLabelConfig,
) -> pd.DataFrame:
    """Per-patch class fractions from the labelled blocks (vector overlay).

    Keyed on ``(dataset, patch_id)`` — patch_id repeats across So2Sat datasets.
    ``dominant_*`` picks the largest label category, where a category is a hard
    class or a distinct coarse set (never collapsed); ``boundary_flag`` marks
    patches whose dominant category covers < ``export.boundary_frac``. This is
    the So2Sat-comparison and probe surface; nothing in Part 2 trains on it.
    """
    utm = labels.crs
    grid_utm = grid.to_crs(utm)
    labelled = labels[labels["label_type"].isin(["hard", "coarse"])].reset_index(drop=True)
    lgeoms = labelled.geometry.values
    lconf = labelled["confidence"].to_numpy(dtype=float)
    lkeys = [
        (int(r),) if t == "hard" else tuple(sorted(int(c) for c in s))
        for t, r, s in zip(labelled["label_type"],
                           labelled["lcz"].fillna(-1), labelled["lcz_set"])
    ]
    tree = STRtree(lgeoms) if len(lgeoms) else None

    pgeoms = grid_utm.geometry.values
    pareas = shapely.area(pgeoms)
    rows = []
    for i, pg in enumerate(pgeoms):
        frac = np.zeros(N_LCZ)
        cat_area: dict[tuple, float] = {}
        conf_area = 0.0
        lab_area = 0.0
        if tree is not None:
            idx = tree.query(pg, predicate="intersects")
            if len(idx):
                inter = shapely.area(shapely.intersection(pg, lgeoms[idx]))
                for j, a in zip(idx, inter):
                    if a <= 0:
                        continue
                    key = lkeys[j]
                    cat_area[key] = cat_area.get(key, 0.0) + a
                    lab_area += a
                    conf_area += lconf[j] * a
                    share = a / len(key)
                    for c in key:              # coarse mass split across its set
                        frac[c - 1] += share
        row = {
            "dataset": grid_utm.iloc[i].get("dataset", "unlabeled"),
            "patch_id": str(grid_utm.iloc[i]["patch_id"]),
            "aoi": aoi_name,
        }
        row.update({f"f_lcz_{c}": frac[c - 1] / pareas[i] for c in range(1, N_LCZ + 1)})
        row["unlabelled_frac"] = max(0.0, 1.0 - lab_area / pareas[i])
        row["coarse_frac"] = sum(a for k, a in cat_area.items() if len(k) > 1) / pareas[i]
        if cat_area:
            dom_key, dom_area = max(cat_area.items(), key=lambda kv: kv[1])
            row["dominant_lcz"] = int(dom_key[0]) if len(dom_key) == 1 else None
            row["dominant_set"] = list(dom_key)
            row["dominant_frac"] = dom_area / pareas[i]
            row["mean_confidence"] = conf_area / lab_area if lab_area > 0 else float("nan")
        else:
            row.update(dominant_lcz=None, dominant_set=[], dominant_frac=0.0,
                       mean_confidence=float("nan"))
        row["boundary_flag"] = row["dominant_frac"] < config.export.boundary_frac
        if "LCZ_class" in grid_utm.columns:
            v = grid_utm.iloc[i]["LCZ_class"]
            row["so2sat_lcz"] = float(v) if v == v else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    out["label_year"] = config.label_year
    out["overture_release"] = config.overture_release
    out["config_hash"] = config.config_hash
    path = config.cache_dir / aoi_name / f"patch_labels_{aoi_name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    logger.info(f"[{aoi_name}] patch transfer: {len(out)} patches "
                f"(dominant_frac>=0.75: {(out['dominant_frac'] >= 0.75).mean():.0%})")
    return out


# ── Block vector product + training pairs ─────────────────────────────────────

_LEAD_COLS = ["block_id", "block_idx", "aoi", "block_kind", "label_type", "lcz",
              "lcz_set", "lcz_name", "confidence", "zone_id", "zone_area_ha",
              "zone_grade", "stable_2017_to_label_year", "change_score",
              "area_m2", "label_year", "overture_release", "config_hash"]


def write_blocks_parquet(
    labels: gpd.GeoDataFrame, aoi_name: str, config: LczLabelConfig
) -> Path:
    """``blocks_labelled_{aoi}.parquet`` — full provenance + diagnostics."""
    gdf = labels.copy()
    gdf["label_year"] = config.label_year
    gdf["overture_release"] = config.overture_release
    gdf["config_hash"] = config.config_hash
    lead = [c for c in _LEAD_COLS if c in gdf.columns]
    rest = [c for c in gdf.columns if c not in lead and c != "geometry"]
    gdf = gdf[lead + rest + ["geometry"]]
    path = config.cache_dir / aoi_name / f"blocks_labelled_{aoi_name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)
    n_hard = int((gdf["label_type"] == "hard").sum())
    n_coarse = int((gdf["label_type"] == "coarse").sum())
    logger.info(f"[{aoi_name}] wrote {path.name} ({len(gdf)} blocks, {n_hard} hard "
                f"incl. {int((gdf['lcz'] == 7).sum())} LCZ-7, {n_coarse} coarse)")
    return path


def to_training_pairs(labels: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Expand block labels into (block_id, year) training pairs.

    Stable blocks (``stable_2017_to_label_year``) pair with every requested
    embedding year; unstable ones only with their own ``label_year`` (epoch-
    locked — labels must not travel across years where the built environment
    changed; fast-growing informal fringes are exactly where this matters).

    CONSUMPTION CONTRACT (do not collapse a coarse label to one member!):
      * ``label_type == "hard"``   -> standard cross-entropy on the single ``lcz``.
      * ``label_type == "coarse"`` -> marginalised cross-entropy over ``lcz_set``,
        i.e. loss = ``-log sum_{c in lcz_set} p_c`` (the true class is one of the
        set members, e.g. {3,7} or {8,10}, but not which).
    ``lcz_set`` passes through untouched; ``lcz`` is null for coarse rows.
    ``zone_grade`` and ``block_kind`` ride along for filtering/stratification.
    """
    lab = labels[labels["label_type"].isin(["hard", "coarse"])]
    rows = []
    for r in lab.to_dict("records"):
        yrs = years if r.get("stable_2017_to_label_year") else [int(r["label_year"])]
        for y in yrs:
            rows.append({
                "block_id": r["block_id"], "aoi": r.get("aoi"),
                "block_kind": r.get("block_kind"),
                "zone_grade": bool(r.get("zone_grade", False)),
                "label_type": r["label_type"],
                "lcz": (int(r["lcz"]) if pd.notna(r["lcz"]) else None),
                "lcz_set": list(r["lcz_set"]),
                "confidence": r.get("confidence"), "year": y,
            })
    return pd.DataFrame(rows)
