"""Stage 7 — temporal stability mask (observed change products only).

Produces per patch a boolean ``stable_2017_to_label_year`` and a continuous
``change_score`` so that epoch-locked labels can be paired with embeddings of a
different year ONLY where the built environment is stable.

Primary source: Google Open Buildings 2.5D Temporal (annual built
presence/height). We ingest a user-provided directory of per-year GeoTIFFs
(``{google_temporal_dir}/{year}/*.tif``, band 1 = built fraction 0-1, band 2 =
mean building height m — see ``export_google_temporal_ee`` for the Earth Engine
export that produces this layout). Per patch we compute built fraction + mean
height for the baseline (2017) and label year and apply:

    stable  iff  |Δ built fraction| < 0.10  AND  height class unchanged.

Graceful degrade (design decision): when no temporal product is available for a
patch, ``stable = False`` (conservative — the label is used only in its own
epoch). This is the current default until the Google export is wired up.

We NEVER use OSM/Overture edit history for stability: mapping growth is not urban
growth.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger

from .config import LczLabelConfig

BUILT_DELTA_MAX = 0.10   # |Δ built fraction| threshold for "stable"


def export_google_temporal_ee(aoi_bbox, out_dir, years) -> str:
    """Documentation helper: the Earth Engine export for the temporal product.

    Not executed here (Earth Engine auth + long-running export). Run this snippet
    in an authenticated ``earthengine`` / ``geemap`` session to populate
    ``{out_dir}/{year}/built.tif`` used by :func:`compute_change_mask`::

        import ee; ee.Initialize()
        col = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')
        region = ee.Geometry.Rectangle(list(aoi_bbox))
        for year in years:
            img = (col.filterDate(f'{year}-01-01', f'{year}-12-31')
                      .filterBounds(region).mosaic()
                      .select(['building_fractional_count', 'building_height']))
            ee.batch.Export.image.toDrive(
                image=img.clip(region), description=f'obt_{year}',
                folder='obt', scale=4, region=region, maxPixels=1e10).start()

    Coverage: Africa, South/SE Asia, Latin America (~2016-present). Outside that
    footprint, GHS-BUILT-S multi-epoch or WSF-Evolution should be substituted
    (same statistic); until then those patches degrade to ``stable=False``.
    """
    return (f"See docstring: export {list(years)} of Google Open Buildings "
            f"Temporal for bbox {aoi_bbox} to {out_dir}")


def _height_class(h: float, config: LczLabelConfig) -> str:
    t = config.classification
    if not np.isfinite(h):
        return "none"
    if h >= t.height_high_min:
        return "high"
    if h >= t.height_mid_min:
        return "mid"
    if h > 0:
        return "low"
    return "nonbuilt"


def _sample_year(grid_ll: gpd.GeoDataFrame, year_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch (built_fraction, mean_height) from a year's GeoTIFF(s).

    Returns NaN arrays if the directory is absent or has no readable tiles.
    """
    n = len(grid_ll)
    built = np.full(n, np.nan)
    height = np.full(n, np.nan)
    if year_dir is None or not Path(year_dir).is_dir():
        return built, height
    tifs = sorted(Path(year_dir).glob("*.tif"))
    if not tifs:
        return built, height
    for tif in tifs:
        with rasterio.open(tif) as src:
            sub = grid_ll.to_crs(src.crs)
            bands = src.read().astype(np.float64)
            if src.nodata is not None:
                bands[bands == src.nodata] = np.nan
            inv = ~src.transform
            h, w = bands.shape[1], bands.shape[2]
            nb = bands.shape[0]
            for i, geom in enumerate(sub.geometry):
                minx, miny, maxx, maxy = geom.bounds
                ca, ra = inv * (minx, miny)
                cb, rb = inv * (maxx, maxy)
                r0, r1 = int(np.floor(min(ra, rb))), int(np.ceil(max(ra, rb)))
                c0, c1 = int(np.floor(min(ca, cb))), int(np.ceil(max(ca, cb)))
                r0, r1, c0, c1 = max(r0, 0), min(r1, h), max(c0, 0), min(c1, w)
                if r1 <= r0 or c1 <= c0:
                    continue
                win = bands[:, r0:r1, c0:c1]
                b0 = win[0][np.isfinite(win[0])]
                if b0.size and np.isnan(built[i]):
                    built[i] = float(b0.mean())
                if nb > 1:
                    b1 = win[1][np.isfinite(win[1])]
                    if b1.size and np.isnan(height[i]):
                        height[i] = float(b1.mean())
    return built, height


def stability(built0, h0, built1, h1, config) -> tuple[bool, float]:
    """(stable, change_score) from baseline vs label-year stats."""
    if not (np.isfinite(built0) and np.isfinite(built1)):
        return False, float("nan")   # no coverage -> conservative
    change_score = abs(built1 - built0)
    hclass_same = _height_class(h0, config) == _height_class(h1, config)
    stable = (change_score < BUILT_DELTA_MAX) and hclass_same
    return bool(stable), float(change_score)


def compute_change_mask(
    grid: gpd.GeoDataFrame, config: LczLabelConfig, aoi_name: str, *, force: bool = False
) -> pd.DataFrame:
    """Per-patch stability table (cached parquet)."""
    cache = config.cache_dir / aoi_name / f"change_{config.config_hash}.parquet"
    if cache.exists() and not force:
        logger.info(f"[{aoi_name}] change-mask cache hit: {cache.name}")
        return pd.read_parquet(cache)

    grid_ll = grid.to_crs("EPSG:4326")
    tdir = config.rasters.google_temporal_dir
    if tdir is None or not Path(tdir).is_dir():
        logger.warning(
            f"[{aoi_name}] no temporal product ({tdir}) — degrading to stable=False"
        )
        built0 = height0 = built1 = height1 = np.full(len(grid), np.nan)
    else:
        built0, height0 = _sample_year(grid_ll, Path(tdir) / str(config.change_baseline_year))
        built1, height1 = _sample_year(grid_ll, Path(tdir) / str(config.label_year))

    stable = np.zeros(len(grid), dtype=bool)
    score = np.full(len(grid), np.nan)
    for i in range(len(grid)):
        stable[i], score[i] = stability(built0[i], height0[i], built1[i], height1[i], config)

    out = pd.DataFrame({
        "patch_id": grid["patch_id"].astype(str).to_numpy(),
        "dataset": grid.get("dataset", pd.Series(["unlabeled"] * len(grid))).to_numpy(),
        "stable_2017_to_label_year": stable,
        "change_score": score,
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    logger.info(f"[{aoi_name}] change-mask: {int(stable.sum())}/{len(out)} stable")
    return out
