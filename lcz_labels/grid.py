"""Patch grid — reuse the existing So2Sat 320 m grid, or generate a matching one.

The pseudo-labels MUST land on the same grid the embedding pipeline uses, so for
So2Sat cities we read the authoritative ``patches_reference_{city}.gpkg``
(columns ``patch_id, dataset, LCZ_class, geometry``; EPSG:4326; 7-digit string
ids that are only unique *within* a ``dataset``). For any other AOI we generate a
fresh So2Sat-schema 320 m grid over the AOI bbox in local UTM, exactly mirroring
``src/sample_unlabeled_patches.py`` (``dataset='unlabeled'``, ``patch_id`` a
7-digit string, boxes reprojected back to EPSG:4326).

``LCZ_class`` (the So2Sat ground truth) is carried through when present so
``validate.py`` can join against it; it is ``NaN`` for generated grids.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import box

from .config import AOI, LczLabelConfig

# Dir names that don't normalise cleanly to their CSV JRC_NAME_MAIN equivalent
# (mirrors src/create_city_grids.py::_CITY_NAME_OVERRIDES).
_CITY_NAME_OVERRIDES: dict[str, str] = {
    "Sao Paulo": "São Paulo",
    "Dongying": "东营区",
}


def _load_city_bboxes(csv_path: Path) -> dict[str, tuple[float, float, float, float]]:
    df = pd.read_csv(csv_path)
    return {
        row["JRC_NAME_MAIN"]: (row["minx"], row["miny"], row["maxx"], row["maxy"])
        for _, row in df.iterrows()
    }


def _lookup_city_bbox(name: str, bbox_dict: dict) -> tuple | None:
    normalized = name.replace("_", " ")
    key = _CITY_NAME_OVERRIDES.get(normalized, normalized)
    return bbox_dict.get(key)


def _city_gpkg(config: LczLabelConfig, name: str) -> Path | None:
    """Path to the So2Sat patch gpkg for ``name`` (dir may use underscores)."""
    for dir_name in (name, name.replace(" ", "_")):
        p = config.cities_dir / dir_name / f"patches_reference_{dir_name}.gpkg"
        if p.exists():
            return p
    return None


def resolve_aoi_bbox(aoi: AOI, config: LczLabelConfig) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) EPSG:4326 from the AOI or the bounds CSV."""
    if aoi.bbox is not None:
        return tuple(aoi.bbox)  # type: ignore[return-value]
    if config.city_bounds_csv.exists():
        bbox = _lookup_city_bbox(aoi.name, _load_city_bboxes(config.city_bounds_csv))
        if bbox is not None:
            return bbox
    raise ValueError(
        f"AOI {aoi.name!r} has no bbox and is not in {config.city_bounds_csv}. "
        f"Provide bbox in the config."
    )


def local_utm_crs(bbox: tuple[float, float, float, float], override: str | None) -> str:
    """Local UTM CRS for area computations (auto-estimated unless overridden)."""
    if override:
        return override
    minx, miny, maxx, maxy = bbox
    g = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326")
    return str(g.estimate_utm_crs())


def _generate_grid(
    bbox: tuple[float, float, float, float], utm_crs: str, patch_size_m: float
) -> gpd.GeoDataFrame:
    """Non-overlapping 320 m patch grid tiling ``bbox`` (built in local UTM)."""
    minx, miny, maxx, maxy = bbox
    # Project the bbox corners to UTM and tile there so patches are true squares.
    corners = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326").to_crs(utm_crs)
    ux0, uy0, ux1, uy1 = corners.total_bounds
    xs = np.arange(ux0, ux1, patch_size_m)
    ys = np.arange(uy0, uy1, patch_size_m)
    geoms = [box(x, y, x + patch_size_m, y + patch_size_m) for y in ys for x in xs]
    if not geoms:
        raise ValueError(f"bbox {bbox} too small for {patch_size_m} m patches")
    gdf = gpd.GeoDataFrame(
        {
            "patch_id": [f"{i:07d}" for i in range(len(geoms))],
            "dataset": "unlabeled",
            "LCZ_class": np.nan,
        },
        geometry=geoms,
        crs=utm_crs,
    ).to_crs("EPSG:4326")
    return gdf


def load_grid(
    aoi_name: str, config: LczLabelConfig, *, force: bool = False
) -> gpd.GeoDataFrame:
    """Load (So2Sat) or generate the 320 m patch grid for one AOI.

    Returns a GeoDataFrame with columns ``patch_id, dataset, LCZ_class,
    geometry`` in EPSG:4326, plus an ``aoi`` column. Cached per AOI+config_hash
    as a GeoParquet under ``cache_dir/{aoi}/grid_{hash}.parquet``.
    """
    aoi = config.aoi(aoi_name)
    cache = config.cache_dir / aoi_name / f"grid_{config.config_hash}.parquet"
    if cache.exists() and not force:
        logger.info(f"[{aoi_name}] grid cache hit: {cache.name}")
        return gpd.read_parquet(cache)

    gpkg = _city_gpkg(config, aoi_name)
    if gpkg is not None:
        gdf = gpd.read_file(gpkg)
        for col in ("patch_id", "dataset", "LCZ_class", "geometry"):
            if col not in gdf.columns:
                gdf[col] = np.nan
        gdf = gdf[["patch_id", "dataset", "LCZ_class", "geometry"]].copy()
        gdf["patch_id"] = gdf["patch_id"].astype(str)
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        logger.info(f"[{aoi_name}] reusing So2Sat grid {gpkg.name}: {len(gdf)} patches")
    else:
        bbox = resolve_aoi_bbox(aoi, config)
        utm = local_utm_crs(bbox, aoi.equal_area_crs)
        gdf = _generate_grid(bbox, utm, config.patch_size_m)
        logger.info(
            f"[{aoi_name}] generated grid over {bbox} in {utm}: {len(gdf)} patches"
        )

    gdf["aoi"] = aoi_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache)
    return gdf
