"""Download functions for embedding and label datasets with tile-level caching."""

import math
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from loguru import logger

_COOP_S3_PREFIX = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/"


def download_alpha_earth_coop(
    index_path: str | Path,
    output_dir: str | Path,
    bbox: tuple[float, float, float, float],
    year: int,
    workers: int = 4,
    overwrite: bool = False,
) -> Path:
    """Download AlphaEarth coop tiles (.tiff + .vrt) from source.coop for a bbox and year.

    Reads the ``aef_index.gpkg`` registry to find tiles overlapping *bbox*,
    then downloads each tile's ``.tiff`` and ``.vrt`` files in parallel from
    the public HTTPS endpoint.  Already-present files are skipped unless
    *overwrite* is set.

    Local files are saved at ``output_dir/{year}/{utm_zone}/{filename}`` to
    mirror the S3 directory layout and avoid filename collisions across UTM
    zones.

    Args:
        index_path: Path to ``aef_index.gpkg`` (the tile registry GeoPackage).
        output_dir: Local root for downloaded tiles (typically the coop dir
            that also contains ``aef_index.gpkg``).
        bbox: ``(west, south, east, north)`` in EPSG:4326.
        year: Year of embeddings to download (2017–2025).
        workers: Number of parallel download threads.
        overwrite: Re-download tiles that already exist locally.

    Returns:
        Path to *output_dir*.
    """
    import requests
    import geopandas as gpd

    output_dir = Path(output_dir)
    index_path = Path(index_path)

    gdf = gpd.read_file(index_path, bbox=bbox, where=f"year = {year}")
    if len(gdf) == 0:
        logger.warning(f"No coop tiles found for year={year}, bbox={bbox}")
        return output_dir

    logger.info(f"AlphaEarth coop: {len(gdf)} tiles for year={year}, bbox={bbox}")

    def s3_to_local(s3_path: str) -> Path:
        return output_dir / s3_path.removeprefix(_COOP_S3_PREFIX)

    def stream_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            tmp.rename(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def download_tile(s3_path: str) -> tuple[int, int]:
        """Return (n_downloaded, n_skipped) for one tile's .tiff + .vrt pair."""
        https_url = s3_path.replace(
            "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/",
            "https://data.source.coop/tge-labs/aef/v1/annual/",
        )
        stem = https_url.rsplit(".", 1)[0]
        pairs = [
            (https_url, s3_to_local(s3_path)),
            (stem + ".vrt", s3_to_local(s3_path).with_suffix(".vrt")),
        ]
        downloaded = skipped = 0
        for url, dest in pairs:
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            try:
                stream_download(url, dest)
                downloaded += 1
                logger.debug(f"Downloaded {dest.name}")
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    logger.debug(f"Not found (skipping): {url}")
                else:
                    raise
        return downloaded, skipped

    total_dl = total_skip = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_tile, row["path"]): row["path"]
            for _, row in gdf.iterrows()
        }
        for future in as_completed(futures):
            try:
                dl, sk = future.result()
                total_dl += dl
                total_skip += sk
            except Exception as exc:
                path = futures[future]
                errors.append(f"{path}: {exc}")
                logger.error(f"Failed to download {path}: {exc}")

    logger.info(
        f"Coop download complete: {total_dl} downloaded, "
        f"{total_skip} cached, {len(errors)} errors"
    )
    if errors:
        logger.warning(f"Download errors (first 5): {errors[:5]}")

    return output_dir


def download_tessera(
    bbox: list[float],
    output_dir: str | Path,
    year: int = 2024,
    output_format: str = "zarr",
) -> Path:
    """Download Tessera embeddings for a bounding box with tile-level caching.

    Uses the native GeoTessera export methods (export_embedding_zarr /
    export_embedding_geotiff) which produce correctly geo-referenced tiles
    with proper CRS, transform, and metadata attributes.

    Args:
        bbox: [west, south, east, north] in EPSG:4326.
        output_dir: Directory to save tiles.
        year: Year of embeddings to download.
        output_format: Output format, either "zarr" or "tif".

    Returns:
        Path to the output directory.
    """
    if output_format not in ("zarr", "tif"):
        raise ValueError(f"output_format must be 'zarr' or 'tif', got '{output_format}'")

    from geotessera import GeoTessera

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary directory for geotessera's intermediate .npy cache so
    # large cache files are cleaned up automatically after export.
    tmpdir = tempfile.mkdtemp(prefix="geotessera_")
    gt = GeoTessera(embeddings_dir=tmpdir)
    tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=year)
    logger.info(f"Found {len(tiles)} tessera tiles for bbox={bbox}, year={year}")

    downloaded, skipped = 0, 0
    for tile_year, lon, lat in tiles:
        stem = f"grid_{lon}_{lat}_{tile_year}"
        ext = ".zarr" if output_format == "zarr" else ".tif"
        out_path = output_dir / f"{stem}{ext}"
        if out_path.exists():
            skipped += 1
            continue

        if output_format == "zarr":
            gt.export_embedding_zarr(lon=lon, lat=lat, output_path=out_path, year=tile_year)
        else:
            gt.export_embedding_geotiff(lon=lon, lat=lat, output_path=out_path, year=tile_year)
        downloaded += 1
        logger.debug(f"Downloaded tile {stem}")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    logger.info(f"Tessera download complete: {downloaded} downloaded, {skipped} cached")
    return output_dir


def download_alpha_earth(
    bbox: list[float],
    output_dir: str | Path,
    year: int = 2021,
    output_format: str = "zarr",
) -> Path:
    """Download AlphaEarth embeddings for a bounding box with tile-level caching.

    Tiles the bbox into 0.1-degree grid cells for tile-level caching.

    Args:
        bbox: [west, south, east, north] in EPSG:4326.
        output_dir: Directory to save tiles.
        year: Year of embeddings to download.
        output_format: Output format, either "zarr" or "tif".

    Returns:
        Path to the output directory.
    """
    if output_format not in ("zarr", "tif"):
        raise ValueError(f"output_format must be 'zarr' or 'tif', got '{output_format}'")
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
    logger.info(f"AlphaEarth: {total_tiles} tiles for bbox={bbox}, year={year}")

    downloaded, skipped = 0, 0
    for lon in lon_starts:
        for lat in lat_starts:
            # Round to avoid floating point drift in filenames
            lon_r = round(lon, 2)
            lat_r = round(lat, 2)
            stem = f"gse_{lon_r}_{lat_r}_{year}"
            out_path = output_dir / (f"{stem}.zarr" if output_format == "zarr" else f"{stem}.tif")
            if out_path.exists():
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
            try:
                ds = get_ic_as_xr(ic, bbox=tile_bbox, utm_crs=str(utm_crs), scale=10)
            except TypeError:
                # GEE returned an empty ImageCollection for this tile (no data / ocean)
                logger.debug(f"No GEE data for tile {stem}, skipping")
                skipped += 1
                continue

            # Stack all bands into a single DataArray
            band_names = list(ds.data_vars)
            da = ds[band_names].to_dataarray(dim="band")

            # Normalize y to descending (north-up), matching Tessera tile format.
            # xee returns ascending y for UTM projections; flip so the tile is
            # stored with y[0] > y[-1] (north at top) just like Tessera.
            if da.sizes.get("y", 0) > 1 and float(da.y.values[0]) < float(da.y.values[-1]):
                da = da.isel(y=slice(None, None, -1))

            # Use integer band indices (0, 1, …, 63) instead of string labels
            # ('A00', …, 'A63') to match the Tessera convention.
            da = da.assign_coords(band=np.arange(len(band_names)))

            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            da = da.rio.write_crs(str(utm_crs))
            da = da.rio.write_transform(da.rio.transform())

            if output_format == "zarr":
                da.to_dataset(name="embedding").to_zarr(str(out_path))
            else:
                da.rio.to_raster(str(out_path))
            downloaded += 1
            logger.debug(f"Downloaded tile {stem}")

    logger.info(
        f"AlphaEarth download complete: {downloaded} downloaded, {skipped} cached"
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
