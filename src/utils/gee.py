"""Google Earth Engine authentication and data request helpers."""

from utils.constants import GEE_PROJECT_NAME

from pathlib import Path

import ee
import xarray as xr


def authenticate_ee():
    """Authenticates the Earth Engine API."""
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_NAME, opt_url='https://earthengine-highvolume.googleapis.com')


def request_gee_image(ee_path: str, date: str | None = None, date_end: str | None = None, bbox: list = None) -> ee.ImageCollection:
    """Request a filtered ImageCollection from Google Earth Engine.

    Args:
        ee_path: GEE ImageCollection asset path (e.g. "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").
        date: Start date string for temporal filter (e.g. "2024-01-01"). Optional.
        date_end: End date string for temporal filter (e.g. "2024-12-31"). Optional.
        bbox: Bounding box [west, south, east, north] in EPSG:4326 for spatial filter. Optional.

    Returns:
        Filtered ee.ImageCollection.
    """
    dataset_ic = ee.ImageCollection(ee_path)

    if date and date_end:
        date = ee.Date(date)
        date_end = ee.Date(date_end)
        dataset_ic = dataset_ic.filter(ee.Filter.date(date, date_end))

    if bbox:
        bbox_fc = ee.Geometry.Rectangle([bbox[:2], bbox[2:]])
        dataset_ic = dataset_ic.filter(ee.Filter.bounds(bbox_fc))

    return dataset_ic


def get_ic_as_xr(dataset_ic: ee.ImageCollection, bbox: list, utm_crs: str, scale: int = 10, ic_output_path: Path = None) -> xr.Dataset:
    """Open a GEE ImageCollection as an xarray Dataset, clipped to a bounding box.

    Merges temporal dimension via forward/backward fill and selects the first time step.

    Args:
        dataset_ic: Earth Engine ImageCollection to convert.
        bbox: Bounding box [west, south, east, north] in EPSG:4326.
        utm_crs: Target CRS string for the output (e.g. "EPSG:32632").
        scale: Pixel resolution in meters.
        ic_output_path: If set, save the merged dataset to this Zarr path.

    Returns:
        xr.Dataset with spatial dims (y, x) and data variables for each band.
    """
    bbox_fc = ee.Geometry.Rectangle([bbox[:2], bbox[2:]])

    ic_ds = xr.open_dataset(dataset_ic, engine="ee", geometry=bbox_fc, scale=scale, crs=utm_crs)
    ic_ds_merged = ic_ds.ffill(dim="time").bfill(dim="time").isel(time=0).drop_vars("time")
    # xee returns X/Y for projected CRS, lon/lat for geographic CRS
    if "lon" in ic_ds_merged.dims:
        ic_ds_merged = ic_ds_merged.transpose("lat", "lon").rename({"lat": "y", "lon": "x"})
    else:
        ic_ds_merged = ic_ds_merged.transpose("Y", "X").rename({"Y": "y", "X": "x"})

    if ic_output_path:
        ic_ds_merged.to_zarr(ic_output_path)

    return ic_ds_merged
