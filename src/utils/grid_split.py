"""Grid-based geographic train/val/test split with per-tile raster coverage."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


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


def _numpy_mosaic_grid_split(arrays: list[xr.DataArray]) -> np.ndarray:
    """Mosaic north-up DataArrays via coordinate-based pixel placement.

    Avoids rasterio.merge which rejects south-up (AlphaEarth) tiles.
    All arrays must already be normalised to north-up (descending y).
    """
    a0 = arrays[0]
    yres = abs(float(a0.y.values[0] - a0.y.values[1])) if len(a0.y) > 1 else 1.0
    xres = abs(float(a0.x.values[1] - a0.x.values[0])) if len(a0.x) > 1 else 1.0

    y_top   = max(float(a.y.values[0])  for a in arrays) + yres / 2
    y_bot   = min(float(a.y.values[-1]) for a in arrays) - yres / 2
    x_left  = min(float(a.x.values[0])  for a in arrays) - xres / 2
    x_right = max(float(a.x.values[-1]) for a in arrays) + xres / 2

    target_h = max(1, round((y_top - y_bot)    / yres))
    target_w = max(1, round((x_right - x_left) / xres))
    n_bands  = a0.sizes["band"]

    out = np.zeros((n_bands, target_h, target_w), dtype=np.float32)
    for a in arrays:
        vals    = a.values.astype(np.float32)
        ay, ax  = a.y.values, a.x.values
        row_idx = (np.floor((y_top  - ay) / yres + 0.5) - 1).astype(int)
        col_idx = (np.floor((ax - x_left) / xres + 0.5) - 1).astype(int)
        r_mask  = (row_idx >= 0) & (row_idx < target_h)
        c_mask  = (col_idx >= 0) & (col_idx < target_w)
        if not r_mask.any() or not c_mask.any():
            continue
        ri = np.where(r_mask)[0]
        ci = np.where(c_mask)[0]
        out[:, row_idx[r_mask][:, None], col_idx[c_mask][None, :]] = \
            vals[:, ri[:, None], ci[None, :]]
    return out


def _open_tile_as_dataarray(path: Path) -> xr.DataArray:
    """Open a .zarr or .tif embedding tile as an (band, y, x) DataArray with CRS set."""
    import rioxarray as rxr  # optional dep, imported lazily

    if path.suffix == ".zarr":
        ds = xr.open_zarr(str(path), chunks=False)
        da = ds["embedding"]
        # Normalise dim order: native tessera zarr stores as (y, x, band)
        if da.dims != ("band", "y", "x"):
            da = da.transpose("band", "y", "x")
        # CRS may be in spatial_ref variable rather than rio attributes
        if da.rio.crs is None and "spatial_ref" in ds:
            crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
            if crs_wkt:
                da = da.rio.write_crs(crs_wkt)
    else:
        da = rxr.open_rasterio(path, chunks=False)

    return da


def crop_embedding_to_polygon(
    polygon,
    polygon_crs: str,
    embedding_dir: str | Path,
    embedding_name: str,
    output_path: str | Path,
) -> np.ndarray:
    """Crop embedding tile(s) to the bounding box of a polygon and save as .npy.

    Works for any polygon shape — large grid tiles or small patch polygons
    (e.g. So2Sat patches). When the polygon spans more than one tile file the
    tiles are mosaicked before cropping.

    Args:
        polygon: Shapely geometry defining the crop extent.
        polygon_crs: CRS of ``polygon`` as an EPSG string or WKT
            (e.g. ``"EPSG:32737"``).
        embedding_dir: Directory containing ``.zarr`` or ``.tif`` tile files.
        embedding_name: Key in ``EMBEDDING_REGISTRY`` (``"tessera"`` or
            ``"alpha_earth"``). Used to locate the tiles that cover the polygon.
        output_path: Destination ``.npy`` file path.  Parent directories are
            created automatically.

    Returns:
        Float32 NumPy array of shape ``(bands, height, width)``.

    Raises:
        FileNotFoundError: If no embedding tiles are found for the polygon ROI.
        ValueError: If the polygon does not overlap any loaded tile.
    """
    from datasets.registry import find_tiles_for_roi

    # 1. Convert polygon bounds to EPSG:4326 for tile lookup
    proj_to_4326 = Transformer.from_crs(polygon_crs, "EPSG:4326", always_xy=True)
    polygon_4326 = shapely_transform(proj_to_4326.transform, polygon)
    roi = polygon_4326.bounds  # (minx, miny, maxx, maxy) in EPSG:4326

    # 2. Find tile files that intersect the roi
    tile_paths = find_tiles_for_roi(embedding_dir, roi, embedding_name)
    if not tile_paths:
        raise FileNotFoundError(
            f"No {embedding_name} tiles found in {embedding_dir} for roi {roi}"
        )

    # 3. Open each tile, clip to polygon bounds first, then normalise orientation.
    #    Clipping before any y-flip avoids loading the full ~244 MB south-up tile
    #    (AlphaEarth); zarr reads only the relevant chunks after clip_box.
    tile_crs_str: str | None = None
    polygon_in_tile = polygon
    arrays: list[xr.DataArray] = []

    for path in tile_paths:
        da = _open_tile_as_dataarray(path)
        if da.rio.crs is None:
            warnings.warn(f"Could not detect CRS for {path.name}; skipping.")
            continue

        if tile_crs_str is None:
            tile_crs_str = da.rio.crs.to_string()
            if tile_crs_str != polygon_crs:
                proj_to_tile = Transformer.from_crs(polygon_crs, tile_crs_str, always_xy=True)
                polygon_in_tile = shapely_transform(proj_to_tile.transform, polygon)

        minx, miny, maxx, maxy = polygon_in_tile.bounds
        try:
            clipped = da.rio.clip_box(minx, miny, maxx, maxy)
        except Exception:
            continue

        if clipped.size == 0:
            continue

        # Normalise south-up → north-up on the already-small clipped array
        if clipped.sizes.get("y", 0) > 1 and float(clipped.y.values[0]) < float(clipped.y.values[-1]):
            clipped = clipped.isel(y=slice(None, None, -1))

        arrays.append(clipped)

    if not arrays:
        raise ValueError(
            f"Polygon does not overlap any tile data. "
            f"Polygon bounds in tile CRS: {polygon_in_tile.bounds}"
        )

    # 4. Mosaic multiple tiles using coordinate-based numpy placement
    #    (rasterio.merge rejects south-up tiles, so we bypass it entirely)
    if len(arrays) > 1:
        result = _numpy_mosaic_grid_split(arrays)
    else:
        result = arrays[0].values.astype(np.float32)

    # 5. Save and return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)
    return result
