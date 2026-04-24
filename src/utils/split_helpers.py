"""Shared train/val/test split helpers for train_unet.py and train_unet_raster.py."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from loguru import logger
from shapely.ops import unary_union


def concat_label_gdfs(paths: list[str | Path], label_col: str, target_crs) -> "gpd.GeoDataFrame":
    """Read and concatenate multiple vector label files into one GeoDataFrame.

    Args:
        paths: List of .gpkg / .geojson / .shp file paths.
        label_col: Column containing 1-based integer class values.
        target_crs: CRS to reproject each file to (typically embedding CRS).

    Returns:
        Concatenated GeoDataFrame in target_crs.
    """
    import geopandas as gpd
    import pandas as pd

    gdfs = []
    for p in paths:
        gdf = gpd.read_file(p).to_crs(target_crs)
        gdfs.append(gdf)
        logger.info(f"  {Path(p).name}: {len(gdf)} polygons")
    merged = pd.concat(gdfs, ignore_index=True)
    logger.info(f"Concatenated {len(paths)} label file(s): {len(merged)} polygons total")
    return merged


def assign_cities_by_fraction(
    paths: list[str | Path],
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Randomly assign city label files to train/val/test by fraction.

    Guarantees at least 1 city in train. Val and test may be empty when there
    are too few cities for the requested fractions.

    Args:
        paths: List of per-city label file paths.
        train_frac: Fraction of cities for training (e.g. 0.70).
        val_frac: Fraction of cities for validation (e.g. 0.15).
        seed: RNG seed for shuffling.

    Returns:
        (train_paths, val_paths, test_paths) as lists of Path objects.
    """
    rng = np.random.default_rng(seed)
    shuffled = [paths[i] for i in rng.permutation(len(paths))]
    n = len(shuffled)
    n_train = max(1, round(n * train_frac))
    n_val = round(n * val_frac)
    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train : n_train + n_val]
    test_paths = shuffled[n_train + n_val :]
    logger.info(
        f"City-level split — "
        f"train={[Path(p).stem for p in train_paths]}, "
        f"val={[Path(p).stem for p in val_paths]}, "
        f"test={[Path(p).stem for p in test_paths]}"
    )
    if not val_paths:
        logger.warning("No cities assigned to val — val evaluation may be empty")
    if not test_paths:
        logger.warning("No cities assigned to test — test evaluation may be empty")
    return train_paths, val_paths, test_paths


def roi_from_gdf(gdf: "gpd.GeoDataFrame"):
    """Return a Shapely geometry covering the bounding box of all polygons in gdf."""
    from shapely.geometry import box as shapely_box

    minx, miny, maxx, maxy = gdf.total_bounds
    return shapely_box(minx, miny, maxx, maxy)


def rasterize_city_paths(
    paths: list[str | Path],
    label_col: str,
    target_crs,
    embedding_res: float,
    out_dir: Path,
    rasterize_fn,
    bbox_poly=None,
) -> "gpd.GeoDataFrame":
    """Rasterize each city label file into its own TIF in out_dir.

    Each city gets a separate TIF (labels_000.tif, labels_001.tif, …) so the
    combined extent never spans multiple UTM zones in one huge raster.
    LCZLabelDataset pointed at out_dir will index all TIFs.

    Args:
        paths: Per-city vector label files.
        label_col: Column with 1-based integer class values.
        target_crs: CRS to reproject each GDF to (embedding CRS).
        embedding_res: Pixel resolution for rasterization.
        out_dir: Directory to write per-city TIFs into.
        rasterize_fn: ``rasterize_gdf`` callable from datasets.labels.
        bbox_poly: Optional Shapely polygon to clip each GDF (embedding CRS).

    Returns:
        Concatenated GeoDataFrame of all polygons (for ROI computation).
    """
    import geopandas as gpd
    import pandas as pd

    gdfs = []
    for i, p in enumerate(paths):
        gdf = gpd.read_file(p).to_crs(target_crs)
        if bbox_poly is not None:
            gdf = gdf[gdf.geometry.intersects(bbox_poly)].copy()
        if len(gdf) == 0:
            logger.warning(f"No polygons after bbox clip for {Path(p).name} — skipping")
            continue
        tif_path = out_dir / f"labels_{i:03d}.tif"
        rasterize_fn(gdf, label_col, tif_path, res=embedding_res)
        logger.info(f"  {Path(p).name}: {len(gdf)} polygons → {tif_path.name}")
        gdfs.append(gdf)

    if not gdfs:
        raise RuntimeError("No label polygons found after rasterizing all city paths.")

    return pd.concat(gdfs, ignore_index=True)
