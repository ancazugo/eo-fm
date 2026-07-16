"""Grid-based geographic train/val/test split with per-tile raster coverage."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from shapely.geometry import box


def create_split_grid_and_join(
    bbox_coords: tuple[float, float, float, float],
    bbox_crs: str,
    polygons_gdf: gpd.GeoDataFrame,
    sub_tile_size: float,
    grid_size: int = 3,
    pattern: str | list[list[str]] = "random",
    split_proportions: dict[str, float] | None = None,
    target_crs: str | None = None,
    join_method: str = "centroid",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create a grid over a bbox, assign train/val/test splits, and join polygons.

    Tiles are grouped into ``grid_size × grid_size`` macro-blocks. Each
    sub-tile inside a block gets the split assigned by ``pattern``.

    Args:
        bbox_coords: ``(minx, miny, maxx, maxy)`` of the region of interest.
        bbox_crs: CRS string for ``bbox_coords`` (e.g. ``"EPSG:4326"``).
        polygons_gdf: GeoDataFrame containing the label polygons.
        sub_tile_size: Size of each individual sub-tile in projected units
            (metres for UTM).
        grid_size: Width/height (in sub-tiles) of each macro-block (default 3).
        pattern: ``"random"`` to assign splits randomly, or a
            ``grid_size × grid_size`` list-of-lists of split names
            (``"train"``, ``"val"``, ``"test"``).
        split_proportions: Fractions used when ``pattern="random"``.
            Defaults to ``{"train": 0.70, "val": 0.15, "test": 0.15}``.
        target_crs: CRS to project into before gridding. Auto-estimated UTM
            from ``polygons_gdf`` when ``None``.
        join_method: ``"centroid"`` (polygon centroid falls inside tile) or
            ``"intersects"`` (any overlap).

    Returns:
        ``(joined_gdf, grid_gdf)`` — polygons with ``grid_id`` and ``split``
        columns, and the grid GeoDataFrame with split assignments.
    """
    if split_proportions is None:
        split_proportions = {"train": 0.70, "val": 0.15, "test": 0.15}

    if int(grid_size) <= 0:
        raise ValueError("grid_size must be a positive integer.")

    if pattern != "random":
        if len(pattern) != grid_size or any(len(row) != grid_size for row in pattern):
            raise ValueError(
                f"pattern must be a {grid_size}×{grid_size} 2-D list to match grid_size={grid_size}"
            )
    else:
        split_names = list(split_proportions.keys())
        split_probs = list(split_proportions.values())
        if not np.isclose(sum(split_probs), 1.0):
            raise ValueError("split_proportions must sum to 1.0")

    if polygons_gdf.crs is None:
        raise ValueError("polygons_gdf must have a defined CRS.")

    # --- CRS handling ---
    bbox_gdf = gpd.GeoDataFrame({"geometry": [box(*bbox_coords)]}, crs=bbox_crs)
    if bbox_gdf.crs != polygons_gdf.crs:
        bbox_gdf = bbox_gdf.to_crs(polygons_gdf.crs)

    if target_crs is not None:
        polygons_gdf = polygons_gdf.to_crs(target_crs)
        bbox_gdf = bbox_gdf.to_crs(target_crs)
    else:
        if not polygons_gdf.crs.is_projected:
            estimated_crs = polygons_gdf.estimate_utm_crs()
            warnings.warn(f"No target_crs provided. Auto-projecting to {estimated_crs}.")
            polygons_gdf = polygons_gdf.to_crs(estimated_crs)
            bbox_gdf = bbox_gdf.to_crs(estimated_crs)

    # --- Build grid ---
    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    macro_block_size = sub_tile_size * grid_size

    grid_geoms = []
    splits = []

    for x in np.arange(minx, maxx, macro_block_size):
        for y in np.arange(miny, maxy, macro_block_size):
            for i in range(grid_size):      # left → right
                for j in range(grid_size):  # bottom → top
                    sq = box(
                        x + i * sub_tile_size,
                        y + j * sub_tile_size,
                        x + (i + 1) * sub_tile_size,
                        y + (j + 1) * sub_tile_size,
                    )
                    grid_geoms.append(sq)
                    if pattern == "random":
                        splits.append(np.random.choice(split_names, p=split_probs))
                    else:
                        # pattern rows go top→bottom; j goes bottom→top
                        splits.append(pattern[(grid_size - 1) - j][i])

    grid_gdf = gpd.GeoDataFrame(
        {"grid_id": range(len(grid_geoms)), "split": splits},
        geometry=grid_geoms,
        crs=bbox_gdf.crs,
    )

    # --- Spatial join ---
    if join_method == "centroid":
        original_geom = polygons_gdf.geometry
        centroids = polygons_gdf.copy()
        centroids.geometry = centroids.geometry.centroid
        joined = gpd.sjoin(centroids, grid_gdf, how="inner", predicate="intersects")
        joined = joined.set_geometry(original_geom.loc[joined.index])
    elif join_method == "intersects":
        joined = gpd.sjoin(polygons_gdf, grid_gdf, how="inner", predicate="intersects")
    else:
        raise ValueError("join_method must be 'centroid' or 'intersects'")

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined, grid_gdf


def calculate_tile_coverage(
    grid_gdf: gpd.GeoDataFrame,
    raster_paths: str | Path | list[str | Path],
    target_crs: str,
    band: int = 1,
) -> gpd.GeoDataFrame:
    """Calculate the fraction of valid (non-NA) pixels in each grid tile.

    When multiple raster files are supplied they are merged in memory before
    computing coverage, so tiles that span the boundary of two files are handled
    correctly.

    Args:
        grid_gdf: GeoDataFrame of tile geometries.
        raster_paths: One or more raster file paths (``.tif``). Multiple files
            are merged in-memory using :func:`rasterio.merge.merge`.
        target_crs: CRS to evaluate everything in (should be projected, e.g.
            UTM, so pixel areas are consistent).
        band: Raster band to use for the nodata mask (1-indexed).

    Returns:
        Copy of ``grid_gdf`` (in ``target_crs``) with a ``coverage_pct`` column.
    """
    # Normalise to list of Path
    if isinstance(raster_paths, (str, Path)):
        raster_paths = [raster_paths]
    raster_paths = [Path(p) for p in raster_paths]

    target_rio_crs = CRS.from_user_input(target_crs)
    if not target_rio_crs.is_projected:
        warnings.warn(
            "target_crs is geographic (not projected). "
            "Pixel areas will vary, which may skew coverage values."
        )

    # Reproject grid if needed
    if grid_gdf.crs != target_rio_crs:
        warnings.warn(f"Reprojecting grid from {grid_gdf.crs} to {target_rio_crs}.")
        grid_proj = grid_gdf.to_crs(target_rio_crs)
    else:
        grid_proj = grid_gdf.copy()

    import rasterio

    # Open all source files; wrap in WarpedVRT when reprojection is needed
    src_files = [rasterio.open(p) for p in raster_paths]
    try:
        datasets = [
            WarpedVRT(src, crs=target_rio_crs) if src.crs != target_rio_crs else src
            for src in src_files
        ]

        # Merge into a single in-memory dataset when multiple files are provided
        if len(datasets) > 1:
            merged_data, merged_transform = merge(datasets)
            # Build a MemoryFile so we can use rasterio.mask normally
            from rasterio.io import MemoryFile

            profile = datasets[0].profile.copy()
            profile.update(
                crs=target_rio_crs,
                transform=merged_transform,
                width=merged_data.shape[2],
                height=merged_data.shape[1],
                count=merged_data.shape[0],
            )
            memfile = MemoryFile()
            with memfile.open(**profile) as mem_ds:
                mem_ds.write(merged_data)
            dataset_to_mask = memfile.open()
        else:
            dataset_to_mask = datasets[0]
            memfile = None

        coverages = []
        for geom in grid_proj.geometry:
            try:
                out_image, _ = mask(dataset_to_mask, [geom], crop=True, filled=False)
                band_data = out_image[band - 1]
                total_pixels = band_data.size
                if total_pixels == 0:
                    coverages.append(0.0)
                    continue
                valid_pixels = int(np.sum(~band_data.mask)) if np.ma.is_masked(band_data) else total_pixels
                coverages.append(valid_pixels / total_pixels)
            except ValueError:
                # Geometry does not overlap the raster extent
                coverages.append(0.0)

    finally:
        if memfile is not None:
            dataset_to_mask.close()
            memfile.close()
        for ds in datasets:
            ds.close()
        for src in src_files:
            src.close()

    grid_proj = grid_proj.copy()
    grid_proj["coverage_pct"] = coverages
    return grid_proj
