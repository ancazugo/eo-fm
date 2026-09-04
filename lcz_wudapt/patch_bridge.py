"""H7 — the So2Sat 320 m patch bridge.

The Stage 8 export is 10 m pixel labels. The published benchmark
(`docs/global_lcz_campaign_2026-07.md`, kappa 0.6497 single / 0.6871 LOCO
ensemble) is 320 m **patch** classification, so comparing against it needs the
labels reshaped onto a So2Sat-compatible grid.

This is lossy, and the loss is the point to measure rather than assume:
**the median WUDAPT polygon is ~216 m across and only 31.8% reach 320 m**, so
most polygons are smaller than a single patch. A patch is emitted only when one
label category actually dominates it; everything else is mixed morphology that
would teach the classifier the wrong thing.

Output schema is exactly `patches_reference_rxr.gpkg`'s
(``patch_id, dataset, LCZ_class, geometry`` in EPSG:4326) plus a ``weight``
column, which is what `src/datasets/so2sat.py::build_pseudo_items` reads — so
`patch_classification.py --pseudo-gpkg` consumes it with no new training code,
and `src/extract_so2sat_embeddings.py --patches-file` extracts the npy.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger

from lcz_labels.export import patch_transfer
from lcz_labels.grid import _generate_grid, local_utm_crs

from .config import WudaptConfig

__all__ = ["build_patch_labels", "patch_yield", "region_centred_patches", "write_pseudo_gpkg"]


def _blocks_path(aoi: str, config: WudaptConfig) -> Path:
    return Path(config.cache_dir) / aoi / f"blocks_labelled_{aoi}.parquet"


def build_patch_labels(aoi: str, config: WudaptConfig) -> pd.DataFrame:
    """Per-patch class fractions for one AOI's Stage 8 regions.

    Grids the AOI's *labelled footprint* (not its GUPPD bbox) at
    ``labels.patch_size_m``, then reuses ``lcz_labels.export.patch_transfer``
    verbatim so the fractions mean exactly what they mean on the Overture path.
    """
    path = _blocks_path(aoi, config)
    if not path.exists():
        raise FileNotFoundError(f"no Stage 8 export for {aoi}; run `lcz_wudapt build` ({path})")
    blocks = gpd.read_parquet(path)
    if blocks.empty:
        raise ValueError(f"[{aoi}] Stage 8 export is empty")

    bbox = tuple(blocks.to_crs("EPSG:4326").total_bounds)
    utm = local_utm_crs(bbox, None)
    grid = _generate_grid(bbox, utm, config.labels.patch_size_m)

    # patch_transfer writes patch_labels_{aoi}.parquet into the LczLabelConfig's
    # cache_dir, which is the OVERTURE path's output directory. Redirect it to
    # this package's cache so the two label sources never overwrite each other's
    # files for a city they both cover.
    labels_cfg = config.labels.model_copy(update={"cache_dir": Path(config.cache_dir)})
    out = patch_transfer(blocks.to_crs(utm), grid, aoi, labels_cfg)

    # patch_transfer returns fractions keyed on (dataset, patch_id) and drops the
    # geometry; re-attach it from the grid we just generated.
    geo = grid.to_crs("EPSG:4326")[["patch_id", "geometry"]].copy()
    geo["patch_id"] = geo["patch_id"].astype(str)
    out["patch_id"] = out["patch_id"].astype(str)
    out = out.merge(geo, on="patch_id", how="left", validate="one_to_one")
    if out["geometry"].isna().any():
        raise ValueError(f"[{aoi}] patch geometry join lost {int(out.geometry.isna().sum())} rows")
    out["aoi"] = aoi
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def patch_yield(transfer: pd.DataFrame, config: WudaptConfig, *,
                min_dominant_frac: float = 0.75,
                max_unlabelled_frac: float = 0.25) -> pd.DataFrame:
    """Patches clean enough to train a 320 m classifier on.

    Three conditions, all necessary:

    * a **hard** dominant category (a coarse set has no single class to assign,
      and the patch pipeline's CE loss cannot represent one);
    * that category covers at least ``min_dominant_frac`` of the patch;
    * the patch is not mostly unlabelled.
    """
    df = transfer.copy()
    keep = (
        df["dominant_lcz"].notna()
        & (df["dominant_frac"] >= min_dominant_frac)
        & (df["unlabelled_frac"] <= max_unlabelled_frac)
    )
    out = df.loc[keep].copy()
    out["LCZ_class"] = out["dominant_lcz"].astype(int)
    out["weight"] = out["mean_confidence"].astype(float)
    return out


def write_pseudo_gpkg(aois: list[str], config: WudaptConfig, out_path: Path, *,
                      min_dominant_frac: float = 0.75,
                      max_unlabelled_frac: float = 0.25) -> tuple[Path, pd.DataFrame]:
    """Combine several AOIs into one `build_pseudo_items`-compatible gpkg.

    ``patch_id`` is only unique within an AOI (each grid restarts at 0000000),
    so ids are re-issued globally here — a collision would silently make two
    different patches look like one to ``build_patch_index``.
    """
    frames, stats = [], []
    for aoi in aois:
        try:
            transfer = build_patch_labels(aoi, config)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(f"[{aoi}] skipped: {exc}")
            continue
        kept = patch_yield(transfer, config, min_dominant_frac=min_dominant_frac,
                           max_unlabelled_frac=max_unlabelled_frac)
        stats.append({
            "aoi": aoi,
            "patches_gridded": len(transfer),
            "patches_touched": int((transfer["unlabelled_frac"] < 1.0).sum()),
            "patches_kept": len(kept),
            "yield_of_touched": len(kept) / max(int((transfer["unlabelled_frac"] < 1.0).sum()), 1),
            "n_classes": int(kept["LCZ_class"].nunique()) if len(kept) else 0,
        })
        if len(kept):
            frames.append(kept)

    report = pd.DataFrame(stats)
    if not frames:
        raise RuntimeError("no AOI produced a usable patch; check the yield report")

    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    combined["patch_id"] = [f"{i:07d}" for i in range(len(combined))]
    combined["dataset"] = "unlabeled"

    gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    keep_cols = ["patch_id", "dataset", "LCZ_class", "weight", "aoi",
                 "dominant_frac", "unlabelled_frac", "mean_confidence", "geometry"]
    gdf = gdf[[c for c in keep_cols if c in gdf.columns]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    logger.info(f"wrote {out_path} ({len(gdf):,} patches, {gdf.LCZ_class.nunique()} classes)")
    return out_path, report


# ── Region-centred sampling (the So2Sat construction) ────────────────────────
#
# Gridding a city blindly and keeping whatever happens to be pure is NOT how
# So2Sat was built, and it produces a badly distorted pool: measured over 66
# AOIs, water took 42% of the patches while LCZ 1 (Compact High-Rise) collapsed
# from 13.8% of the labelled area to 0.5%. A lake trivially dominates a 320 m
# square; a fragmented downtown parcel never does.
#
# So2Sat placed its patches INSIDE labelled polygons, which is why its coverage
# is clustered rather than wall-to-wall. Doing the same here — one or a few
# patches centred on each consensus region — restores the class balance
# (water 42% -> 8.6%, LCZ 1 0.5% -> 4.2%) at ~0.995 mean purity.


def _hard_class_plane(bitmask: np.ndarray) -> np.ndarray:
    """uint32 bitmask -> int16 class plane, keeping only single-class pixels."""
    from lcz_labels.export import decode_bitmask

    out = np.zeros(bitmask.shape, dtype=np.int16)
    for v in np.unique(bitmask[bitmask > 0]):
        s = decode_bitmask(np.array([v], dtype=np.uint32))[0]
        if len(s) == 1:
            out[bitmask == v] = s[0]
    return out


def region_centred_patches(aoi: str, config: WudaptConfig, *, patch_px: int = 32,
                           min_dominant_frac: float = 0.75,
                           min_labelled_frac: float = 0.5,
                           max_per_region: int = 4) -> gpd.GeoDataFrame:
    """320 m patches placed on consensus regions, So2Sat-style.

    ``max_per_region`` caps how many non-overlapping patches a single region may
    contribute. The cap is what controls the class balance: large regions are
    disproportionately water and natural classes, so an uncapped tiling
    re-introduces exactly the bias this placement exists to avoid.
    """
    import rasterio
    from rasterio.transform import xy

    d = Path(config.cache_dir) / aoi
    bm_path = d / f"lcz_bitmask_{aoi}.tif"
    if not bm_path.exists():
        raise FileNotFoundError(f"no Stage 8 export for {aoi}")
    with rasterio.open(bm_path) as src:
        B, transform, crs = src.read(1), src.transform, src.crs
    with rasterio.open(d / f"block_id_{aoi}.tif") as src:
        I = src.read(1)
    with rasterio.open(d / f"confidence_{aoi}.tif") as src:
        C = src.read(1)

    tbl = pd.read_parquet(d / f"blocks_labelled_{aoi}.parquet",
                          columns=["block_idx", "label_type", "lcz", "confidence", "block_kind"])
    hard = tbl[tbl["label_type"] == "hard"]
    if hard.empty:
        return gpd.GeoDataFrame(columns=["LCZ_class", "geometry"], geometry=[], crs="EPSG:4326")

    plane = _hard_class_plane(B)
    H, W = B.shape
    P = int(patch_px)

    # Group pixel positions by region in one pass (a per-region full-raster scan
    # is O(regions x pixels) and does not finish on city-sized rasters).
    flat = I.ravel()
    nz = np.flatnonzero(flat)
    order = np.argsort(flat[nz], kind="stable")
    nz = nz[order]
    keys = flat[nz]
    starts = np.searchsorted(keys, hard["block_idx"].to_numpy(), side="left")
    ends = np.searchsorted(keys, hard["block_idx"].to_numpy(), side="right")

    rows = []
    for st, en, kind in zip(starts, ends, hard["block_kind"].to_numpy()):
        if en <= st:
            continue
        pix = nz[st:en]
        cells = sorted({(int(y // P), int(x // P)) for y, x in zip(pix // W, pix % W)})
        if len(cells) > max_per_region:      # deterministic, spatially spread
            step = len(cells) / max_per_region
            cells = [cells[int(i * step)] for i in range(max_per_region)]
        for gy, gx in cells:
            y0 = max(0, min(gy * P, H - P))
            x0 = max(0, min(gx * P, W - P))
            win = plane[y0:y0 + P, x0:x0 + P]
            lab = win[win > 0]
            if lab.size < min_labelled_frac * P * P:
                continue
            vals, cnt = np.unique(lab, return_counts=True)
            frac = float(cnt.max()) / lab.size
            if frac < min_dominant_frac:
                continue
            conf = float(C[y0:y0 + P, x0:x0 + P][win > 0].mean()) / 100.0
            left, top = xy(transform, y0, x0, offset="ul")
            right, bottom = xy(transform, y0 + P, x0 + P, offset="ul")
            rows.append({
                "LCZ_class": int(vals[cnt.argmax()]),
                "dominant_frac": frac,
                "labelled_frac": float(lab.size) / (P * P),
                "weight": conf,
                "block_kind": kind,
                "aoi": aoi,
                "geometry": shapely.box(left, bottom, right, top),
            })
    if not rows:
        return gpd.GeoDataFrame(columns=["LCZ_class", "geometry"], geometry=[], crs="EPSG:4326")
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).to_crs("EPSG:4326")
    # A region's patches can coincide with a neighbour's; keep one per footprint.
    out = out.drop_duplicates(subset="geometry").reset_index(drop=True)
    return out
