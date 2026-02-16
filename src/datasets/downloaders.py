"""Download functions for embedding and label datasets with tile-level caching."""

import math
import tempfile
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from loguru import logger


def download_tessera(
    bbox: list[float],
    output_dir: str | Path,
    year: int = 2024,
) -> Path:
    """Download Tessera embeddings for a bounding box with tile-level caching.

    Args:
        bbox: [west, south, east, north] in EPSG:4326.
        output_dir: Directory to save Zarr stores.
        year: Year of embeddings to download.

    Returns:
        Path to the output directory.
    """
    from geotessera import GeoTessera

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary directory for geotessera's intermediate .npy downloads
    # so they are cleaned up automatically after conversion to zarr.
    # Without this, GeoTessera defaults to cwd and leaves large .npy files behind.
    tmpdir = tempfile.mkdtemp(prefix="geotessera_")
    gt = GeoTessera(embeddings_dir=tmpdir)
    tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
    logger.info(f"Found {len(tiles)} tessera tiles for bbox={bbox}, year={year}")

    downloaded, skipped = 0, 0
    for tile_year, lon, lat in tiles:
        zarr_path = output_dir / f"grid_{lon}_{lat}_{tile_year}.zarr"
        if zarr_path.exists():
            skipped += 1
            continue

        embedding, crs, transform = gt.fetch_embedding(lon=lon, lat=lat, year=tile_year)
        # embedding shape: (H, W, 128), float32

        h, w, bands = embedding.shape
        # Build pixel coordinates from affine transform
        cols = np.arange(w)
        rows = np.arange(h)
        xs = transform.c + cols * transform.a + transform.b * 0  # x = c + col * a
        ys = transform.f + rows * transform.e + transform.d * 0  # y = f + row * e

        da = xr.DataArray(
            data=np.moveaxis(embedding, -1, 0),  # (128, H, W)
            dims=["band", "y", "x"],
            coords={"band": np.arange(bands), "y": ys, "x": xs},
        )
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        da = da.rio.write_crs(crs)
        da = da.rio.write_transform(transform)

        da.to_dataset(name="embedding").to_zarr(str(zarr_path))
        downloaded += 1
        logger.debug(f"Downloaded tile grid_{lon}_{lat}_{tile_year}")

    # Clean up intermediate .npy files from the temp directory
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    logger.info(f"Tessera download complete: {downloaded} downloaded, {skipped} cached")
    return output_dir


def download_google_satellite(
    bbox: list[float],
    output_dir: str | Path,
    year: int = 2021,
) -> Path:
    """Download Google Satellite Embeddings for a bounding box with tile-level caching.

    Tiles the bbox into 0.1-degree grid cells for tile-level caching.

    Args:
        bbox: [west, south, east, north] in EPSG:4326.
        output_dir: Directory to save Zarr stores.
        year: Year of embeddings to download.

    Returns:
        Path to the output directory.
    """
    from pyproj import CRS
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info

    from utils.gee import authenticate_ee, get_ic_as_xr, request_gee_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    authenticate_ee()

    west, south, east, north = bbox
    tile_size = 0.1

    # Generate 0.1-degree grid cells
    lon_starts = np.arange(
        math.floor(west / tile_size) * tile_size,
        east,
        tile_size,
    )
    lat_starts = np.arange(
        math.floor(south / tile_size) * tile_size,
        north,
        tile_size,
    )

    total_tiles = len(lon_starts) * len(lat_starts)
    logger.info(f"Google Satellite: {total_tiles} tiles for bbox={bbox}, year={year}")

    downloaded, skipped = 0, 0
    for lon in lon_starts:
        for lat in lat_starts:
            # Round to avoid floating point drift in filenames
            lon_r = round(lon, 2)
            lat_r = round(lat, 2)
            zarr_path = output_dir / f"gse_{lon_r}_{lat_r}_{year}.zarr"
            if zarr_path.exists():
                skipped += 1
                continue

            tile_bbox = [lon_r, lat_r, round(lon_r + tile_size, 2), round(lat_r + tile_size, 2)]

            # Find appropriate UTM CRS for this tile
            utm_info = query_utm_crs_info(
                datum_name="WGS 84",
                area_of_interest=AreaOfInterest(
                    west_lon_degree=tile_bbox[0],
                    south_lat_degree=tile_bbox[1],
                    east_lon_degree=tile_bbox[2],
                    north_lat_degree=tile_bbox[3],
                ),
            )
            utm_crs = CRS.from_authority(utm_info[0].auth_name, utm_info[0].code)

            ic = request_gee_image(
                "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                date=f"{year}-01-01",
                date_end=f"{year}-12-31",
                bbox=tile_bbox,
            )
            ds = get_ic_as_xr(ic, bbox=tile_bbox, utm_crs=str(utm_crs), scale=10)

            # Stack all bands into a single DataArray
            band_names = list(ds.data_vars)
            da = ds[band_names].to_dataarray(dim="band")
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            da = da.rio.write_crs(str(utm_crs))

            da.to_dataset(name="embedding").to_zarr(str(zarr_path))
            downloaded += 1
            logger.debug(f"Downloaded tile gse_{lon_r}_{lat_r}_{year}")

    logger.info(
        f"Google Satellite download complete: {downloaded} downloaded, {skipped} cached"
    )
    return output_dir


# --------------------------------------------------------------------------- #
# GEE label / generic raster datasets
# --------------------------------------------------------------------------- #

# Known GEE datasets with sensible defaults
GEE_DATASET_REGISTRY: dict[str, dict] = {
    "demuzere_lcz": {
        "ee_path": "RUB/RUBCLIM/LCZ/global_lcz_map/latest",
        "bands": ["LCZ_Filter"],
        "scale": 100,
        "crs": "utm",  # auto-detect UTM zone per tile so scale is in meters
        "dtype": np.uint8,
        "tile_size": 0.5,
        "prefix": "lcz",
    },
}


def download_gee_dataset(
    bbox: list[float],
    output_dir: str | Path,
    dataset: str = "demuzere_lcz",
    ee_path: str | None = None,
    bands: list[str] | None = None,
    scale: int | None = None,
    crs: str | None = None,
    dtype: np.dtype | None = None,
    tile_size: float | None = None,
    prefix: str | None = None,
) -> Path:
    """Download a GEE ImageCollection as tiled GeoTIFFs with tile-level caching.

    Uses a registry of known datasets for sensible defaults, but all parameters
    can be overridden for arbitrary GEE ImageCollections.

    Args:
        bbox: [west, south, east, north] in EPSG:4326.
        output_dir: Directory to save GeoTIFF tiles.
        dataset: Registry key for known datasets (e.g. "demuzere_lcz").
        ee_path: GEE ImageCollection path. Overrides registry default.
        bands: Band names to select. Overrides registry default.
        scale: Resolution in meters. Overrides registry default.
        crs: Output CRS string. Overrides registry default.
        dtype: NumPy dtype for output raster. Overrides registry default.
        tile_size: Tile size in degrees for caching grid. Overrides registry default.
        prefix: Filename prefix for tiles. Overrides registry default.

    Returns:
        Path to the output directory.
    """
    import ee

    from utils.gee import authenticate_ee

    # Resolve defaults from registry
    defaults = GEE_DATASET_REGISTRY.get(dataset, {})
    ee_path = ee_path or defaults.get("ee_path")
    bands = bands or defaults.get("bands")
    scale = scale or defaults.get("scale", 100)
    crs = crs or defaults.get("crs", "EPSG:4326")
    dtype = dtype or defaults.get("dtype", np.float32)
    tile_size = tile_size or defaults.get("tile_size", 0.5)
    prefix = prefix or defaults.get("prefix", "gee")

    if ee_path is None:
        raise ValueError(
            f"Unknown dataset '{dataset}' and no --ee-path provided. "
            f"Known datasets: {list(GEE_DATASET_REGISTRY)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    authenticate_ee()

    west, south, east, north = bbox

    lon_starts = np.arange(
        math.floor(west / tile_size) * tile_size, east, tile_size,
    )
    lat_starts = np.arange(
        math.floor(south / tile_size) * tile_size, north, tile_size,
    )

    total_tiles = len(lon_starts) * len(lat_starts)
    logger.info(
        f"GEE '{dataset}' ({ee_path}): {total_tiles} tiles for bbox={bbox}, scale={scale}m"
    )

    downloaded, skipped = 0, 0
    for lon in lon_starts:
        for lat in lat_starts:
            lon_r = round(lon, 4)
            lat_r = round(lat, 4)
            tif_path = output_dir / f"{prefix}_{lon_r}_{lat_r}.tif"
            if tif_path.exists():
                skipped += 1
                continue

            tile_bbox = [
                lon_r,
                lat_r,
                round(lon_r + tile_size, 4),
                round(lat_r + tile_size, 4),
            ]

            # Resolve CRS per tile: "utm" auto-detects the UTM zone
            if crs == "utm":
                from pyproj.aoi import AreaOfInterest
                from pyproj.database import query_utm_crs_info

                utm_info = query_utm_crs_info(
                    datum_name="WGS 84",
                    area_of_interest=AreaOfInterest(
                        west_lon_degree=tile_bbox[0],
                        south_lat_degree=tile_bbox[1],
                        east_lon_degree=tile_bbox[2],
                        north_lat_degree=tile_bbox[3],
                    ),
                )
                tile_crs = f"EPSG:{utm_info[0].code}"
            else:
                tile_crs = crs

            geometry = ee.Geometry.Rectangle(
                [tile_bbox[0], tile_bbox[1], tile_bbox[2], tile_bbox[3]]
            )
            ic = ee.ImageCollection(ee_path).filter(ee.Filter.bounds(geometry))

            ds = xr.open_dataset(
                ic, engine="ee", geometry=geometry, scale=scale, crs=tile_crs,
            )
            ds_merged = (
                ds.ffill(dim="time").bfill(dim="time").isel(time=0).drop_vars("time")
            )
            # xee returns X/Y for projected CRS, lon/lat for geographic CRS
            if "lon" in ds_merged.dims:
                ds_merged = ds_merged.transpose("lat", "lon").rename({"lat": "y", "lon": "x"})
            else:
                ds_merged = ds_merged.transpose("Y", "X").rename({"Y": "y", "X": "x"})

            # Select and stack requested bands
            if bands:
                da = ds_merged[bands[0]] if len(bands) == 1 else ds_merged[bands].to_dataarray(dim="band")
            else:
                band_names = list(ds_merged.data_vars)
                da = ds_merged[band_names[0]] if len(band_names) == 1 else ds_merged[band_names].to_dataarray(dim="band")

            da = da.compute().astype(dtype)
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            da = da.rio.write_crs(tile_crs)
            da.rio.to_raster(str(tif_path), dtype=dtype)

            downloaded += 1
            logger.debug(f"Downloaded tile {tif_path.name}")

    logger.info(
        f"GEE download complete: {downloaded} downloaded, {skipped} cached"
    )
    return output_dir
