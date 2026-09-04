"""H5/H6 — merge with So2Sat and emit the Stage 8 label contract.

This is the module that actually *writes labels*. Everything before it produces
either cleaned inputs or measurements.

Outputs per AOI, matching `lcz_labels`' contract exactly so `lcz_train` reads
them with no change:

* ``lcz_bitmask_{aoi}.tif``   uint32, bit ``c-1`` per class in the label set
* ``confidence_{aoi}.tif``    uint8, confidence x 100
* ``block_id_{aoi}.tif``      uint32, dense consensus-region index (0 = none)
* ``blocks_labelled_{aoi}.parquet``  one row per consensus region
* ``adjacency_{hash}.parquet``       schema-correct, empty (see below)

**Consensus regions play the role of blocks.** WUDAPT has no block geometry —
what it has is overlapping opinions. A region here is a connected component of
pixels carrying an identical `(lcz_set, source)`, extracted with
`rasterio.features.shapes`, which is exactly the unit `erosion_valid_mask` needs
for its block-interior logic to mean anything.

**Grid.** Rasters are written on the AOI's *labelled-footprint* grid rather than
the full GUPPD bbox: WUDAPT covers a small fraction of most city boxes, and a
full-bbox raster wastes disk and later mosaic space on unlabelled ground. Every
raster is self-describing (CRS + transform in the tif), so alignment is by
geo-reference, not by convention.

**Adjacency is written empty on purpose.** Consensus regions are islands in
unlabelled space, so a rook-contiguity graph over them would be near-empty and
misleading. An empty file with the right schema means
`lcz_train.datasets.block_graph_edges` returns an empty `edge_index` and
experiment B3 reports as *not run* rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger
from rasterio.features import rasterize, shapes

from lcz_labels.blocks import block_id_hash
from lcz_labels.classify import lcz_name
from lcz_labels.export import _write_tif, decode_bitmask, encode_lcz_set

from .config import WudaptConfig
from .consensus import ConsensusGrid, ConsensusResult
from .ingest import N_LCZ

__all__ = ["consensus_regions", "merge_so2sat", "write_stage8"]

_SO2SAT_CONFIDENCE = 1.0


def _dense(res: ConsensusResult, values: np.ndarray, dtype) -> np.ndarray:
    """Scatter a sparse per-pixel vector back onto the full grid."""
    out = np.zeros(res.grid.shape, dtype=dtype).ravel()
    out[res.index] = values
    return out.reshape(res.grid.shape)


def merge_so2sat(res: ConsensusResult, so2sat: np.ndarray | None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Burn WUDAPT consensus, then So2Sat over it. So2Sat always wins.

    Returns ``(bitmask, confidence_float, source_code)`` dense rasters, where
    source is 0 = unlabelled, 1 = wudapt, 2 = so2sat.

    Contested ground is *not* blended into an ambiguous set: the user's rule is
    that So2Sat is authoritative, so it replaces the WUDAPT label outright. The
    WUDAPT alternative is preserved per region in the parquet (`wudapt_alt`,
    `conflict`) for the change-vs-error triage, but never in the bitmask.
    """
    bitmask = _dense(res, res.bitmask, np.uint32)
    conf = _dense(res, res.confidence, np.float32)
    source = np.where(bitmask > 0, np.uint8(1), np.uint8(0))

    if so2sat is None:
        return bitmask, conf, source

    has = so2sat > 0
    if has.any():
        bitmask[has] = encode_lcz_set([[int(c)] for c in so2sat[has]])
        conf[has] = _SO2SAT_CONFIDENCE
        source[has] = 2
    return bitmask, conf, source


def consensus_regions(bitmask: np.ndarray, confidence: np.ndarray, source: np.ndarray,
                      grid: ConsensusGrid, aoi: str, config: WudaptConfig,
                      *, extra: dict[str, np.ndarray] | None = None) -> gpd.GeoDataFrame:
    """Connected components of identical ``(lcz_set, source)`` -> block table."""
    # One integer key per (label set, source) so components never merge across
    # either. bitmask is uint32 with at most 17 bits used, so shifting by 2 for
    # the source code stays inside uint32.
    key = (bitmask.astype(np.uint64) << np.uint64(2)) | source.astype(np.uint64)
    key = np.where(bitmask > 0, key, 0).astype(np.int64)

    geoms, keys = [], []
    for geom, val in shapes(key, mask=key > 0, connectivity=4, transform=grid.transform):
        geoms.append(shapely.geometry.shape(geom))
        keys.append(int(val))
    if not geoms:
        return gpd.GeoDataFrame(
            {"block_id": [], "lcz_set": [], "label_type": []},
            geometry=[], crs=grid.crs,
        )

    gdf = gpd.GeoDataFrame({"_key": keys}, geometry=geoms, crs=grid.crs)
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["area_m2"] >= config.regions.min_region_area_m2].reset_index(drop=True)
    if gdf.empty:
        return gdf.assign(block_id=[], lcz_set=[], label_type=[])

    gdf["source"] = np.where((gdf["_key"] & 0b11) == 2, "so2sat", "wudapt")
    bits = (gdf["_key"].to_numpy() >> 2).astype(np.uint32)
    sets = decode_bitmask(bits)
    gdf["lcz_set"] = sets
    gdf["label_type"] = ["hard" if len(s) == 1 else "coarse" for s in sets]
    gdf["lcz"] = [s[0] if len(s) == 1 else None for s in sets]
    gdf["lcz_name"] = [lcz_name(s[0]) if len(s) == 1 else None for s in sets]

    # Per-region aggregates: rasterise a dense region index, then bincount.
    gdf["block_idx"] = np.arange(1, len(gdf) + 1, dtype=np.uint32)
    idx = rasterize(zip(gdf.geometry.values, gdf["block_idx"]), out_shape=grid.shape,
                    transform=grid.transform, fill=0, dtype="uint32", all_touched=False)
    flat = idx.ravel()
    sel = flat > 0
    pos = flat[sel] - 1
    n = np.bincount(pos, minlength=len(gdf)).astype(float)
    n_safe = np.maximum(n, 1.0)

    gdf["n_pixels"] = n.astype(int)
    gdf["confidence"] = np.bincount(pos, weights=confidence.ravel()[sel],
                                    minlength=len(gdf)) / n_safe
    for name, arr in (extra or {}).items():
        gdf[name] = np.bincount(pos, weights=arr.ravel()[sel], minlength=len(gdf)) / n_safe

    # block_kind records whether the label had corroboration — lcz_train's
    # by_block_kind stratification then answers "is multi-annotator consensus
    # worth anything?" for free.
    n_eff = gdf.get("n_eff", pd.Series(np.zeros(len(gdf)))).to_numpy()
    gdf["block_kind"] = np.where(
        gdf["source"].to_numpy() == "so2sat", "so2sat",
        np.where(n_eff >= config.consensus.deep_n_eff,
                 "wudapt_consensus_deep", "wudapt_consensus_shallow"),
    )

    gdf["block_id"] = [block_id_hash(aoi, g) for g in gdf.geometry]
    gdf["aoi"] = aoi
    gdf["label_source"] = gdf["source"]        # `overture_release` is meaningless here
    gdf = gdf.sort_values("block_id").reset_index(drop=True)
    gdf["block_idx"] = np.arange(1, len(gdf) + 1, dtype=np.uint32)
    return gdf.drop(columns=["_key"])


def write_stage8(regions: gpd.GeoDataFrame, bitmask: np.ndarray, confidence: np.ndarray,
                 grid: ConsensusGrid, aoi: str, config: WudaptConfig) -> dict[str, Path]:
    """Write the three rasters + the block parquet + an empty adjacency file."""
    out_dir = config.aoi_dir(aoi)

    # block_id raster is rebuilt from the final, filtered region table so it can
    # never disagree with the parquet (regions below the area floor are dropped
    # from both).
    block_idx = rasterize(
        zip(regions.geometry.values, regions["block_idx"].astype("uint32")),
        out_shape=grid.shape, transform=grid.transform, fill=0,
        dtype="uint32", all_touched=False,
    ) if len(regions) else np.zeros(grid.shape, dtype=np.uint32)

    # Suppress pixels whose region was dropped, so the three rasters agree.
    keep = block_idx > 0
    bitmask = np.where(keep, bitmask, np.uint32(0))
    conf8 = np.clip(np.round(np.where(keep, confidence, 0.0) * 100), 0, 100).astype(np.uint8)

    paths = {
        "bitmask": _write_tif(out_dir / f"lcz_bitmask_{aoi}.tif", bitmask, grid.transform, grid.crs),
        "confidence": _write_tif(out_dir / f"confidence_{aoi}.tif", conf8, grid.transform, grid.crs),
        "block_id": _write_tif(out_dir / f"block_id_{aoi}.tif", block_idx, grid.transform, grid.crs),
    }

    tbl = regions.copy()
    tbl["config_hash"] = config.config_hash
    tbl["wudapt_release"] = config.wudapt_release
    lead = ["block_id", "block_idx", "aoi", "block_kind", "label_type", "lcz", "lcz_set",
            "lcz_name", "confidence", "area_m2", "label_year", "label_source", "config_hash"]
    lead = [c for c in lead if c in tbl.columns]
    rest = [c for c in tbl.columns if c not in lead and c != "geometry"]
    tbl = tbl[lead + rest + ["geometry"]]
    parquet = out_dir / f"blocks_labelled_{aoi}.parquet"
    tbl.to_parquet(parquet)
    paths["blocks"] = parquet

    adjacency = out_dir / f"adjacency_{config.config_hash}.parquet"
    pd.DataFrame({"block_a": pd.Series(dtype="int64"),
                  "block_b": pd.Series(dtype="int64"),
                  "shared_len_m": pd.Series(dtype="float64")}).to_parquet(adjacency, index=False)
    paths["adjacency"] = adjacency

    n_hard = int((tbl["label_type"] == "hard").sum())
    n_so2sat = int((tbl.get("source", pd.Series(dtype=str)) == "so2sat").sum())
    px = int((bitmask > 0).sum())
    logger.info(
        f"[{aoi}] Stage 8: {len(tbl):,} regions ({n_hard:,} hard, {n_so2sat:,} So2Sat-sourced) "
        f"| {px:,} labelled px on {grid.shape} @ {grid.res_m:g} m -> {out_dir}"
    )
    return paths
