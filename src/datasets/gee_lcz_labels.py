"""Fetch GEE raster datasets and burn them into LCZ label GeoTIFFs.

Two dataset sources, both used as additional (lower-priority) labels — the same
role as the OSM water raster produced by osm_lcz_labels.py.

  buildings     GOOGLE/Research/open-buildings-temporal/v1
                building_presence band → threshold → configurable LCZ built class
                Temporal: filter to the requested year before mosaicking.

  canopy_height projects/sat-io/open-datasets/facebook/meta-canopy-height
                Height band (metres) → height thresholds → LCZ 11/12 (Trees)
                Not temporally filtered (single snapshot).

Burn priority (last wins):
    buildings  <  canopy_height

Output is a uint8 GeoTIFF in a projected (UTM) CRS, 0 = no label.
The output folder is created next to the reference label file.

Usage
-----
    python src/datasets/gee_lcz_labels.py generate \\
        --ref-path /data/So2Sat-LCZ42/v4/cities/Nairobi/patches_reference_Nairobi.tif \\
        --year 2016 \\
        --datasets "buildings,canopy_height" \\
        --output-path /data/So2Sat-LCZ42/v4/cities/Nairobi/gee_labels/gee_lcz_2016.tif

    # Output defaults to <ref_path.parent>/gee_labels/gee_lcz_<year>.tif
    python src/datasets/gee_lcz_labels.py generate \\
        --ref-path /data/So2Sat-LCZ42/v4/cities/Nairobi/patches_reference_Nairobi.tif \\
        --year 2016
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import typer
from loguru import logger

app = typer.Typer(pretty_exceptions_enable=False)

# ── GEE asset config ──────────────────────────────────────────────────────────

BUILDINGS_ASSET = "GOOGLE/Research/open-buildings-temporal/v1"
BUILDINGS_BAND  = "building_presence"

CANOPY_ASSET = "projects/sat-io/open-datasets/facebook/meta-canopy-height"
CANOPY_BAND  = "cover_code"


# ── GEE download ──────────────────────────────────────────────────────────────


def _init_ee() -> None:
    """Initialize Earth Engine (no-op if already initialized)."""
    import ee
    from utils.gee import authenticate_ee

    try:
        ee.data.getIamPolicy({"resource": ""})          # cheap existence check
    except Exception:
        authenticate_ee()


def _download_band(
    asset: str,
    band: str,
    bbox_wgs84: tuple[float, float, float, float],
    target_crs_epsg: int,
    res_m: float,
    year: int | None = None,
) -> tuple[np.ndarray, object]:
    """Download one band of a GEE asset as a 2-D numpy array.

    Handles both ee.ImageCollection (temporal) and ee.Image (static) assets.
    When *year* is given the collection is filtered to that calendar year before
    mosaicking; when None the full collection is mosaicked.

    Args:
        asset: GEE asset path.
        band: Band / variable name to select.
        bbox_wgs84: (minx, miny, maxx, maxy) in EPSG:4326.
        target_crs_epsg: EPSG code of the projected output CRS (e.g. 32737).
        res_m: Pixel size in metres.
        year: Calendar year for temporal filtering (ImageCollections only).

    Returns:
        (arr, transform) where arr is float32 (height, width) north-up and
        transform is a rasterio Affine object.
    """
    import ee
    import xarray as xr
    from rasterio.transform import from_origin

    _init_ee()

    crs_str = f"EPSG:{target_crs_epsg}"

    # ── Get mosaic image ──────────────────────────────────────────────────────
    try:
        ic = ee.ImageCollection(asset)
        if year is not None:
            ic = ic.filter(ee.Filter.calendarRange(year, year, "year"))
        img = ic.select(band).mosaic()
        logger.debug(f"  {asset}: loaded as ImageCollection (year={year})")
    except Exception:
        img = ee.Image(asset).select(band)
        logger.debug(f"  {asset}: loaded as Image")

    # Set a dummy timestamp so xee's time dimension doesn't cause issues
    img = img.set("system:time_start", ee.Date("2000-01-01").millis())
    ic_for_xee = ee.ImageCollection([img])

    # ── Download via xee ──────────────────────────────────────────────────────
    minx, miny, maxx, maxy = bbox_wgs84
    bbox_geom = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    ds = xr.open_dataset(
        ic_for_xee,
        engine="ee",
        geometry=bbox_geom,
        scale=int(res_m),
        crs=crs_str,
    )
    # Drop time dimension (single-image collection)
    da = ds[band].isel(time=0, drop=True)

    # Normalise dim names (xee uses Y/X for projected, lat/lon for geographic)
    rename = {}
    for old, new in [("Y", "y"), ("X", "x"), ("lat", "y"), ("lon", "x")]:
        if old in da.dims:
            rename[old] = new
    if rename:
        da = da.rename(rename)

    # xee returns (X, Y) for projected CRS — transpose to (Y, X) = (rows, cols)
    if da.dims[0] == "x":
        da = da.transpose("y", "x")

    arr = da.values.astype(np.float32)  # (rows, cols)
    x_coords = da["x"].values
    y_coords = da["y"].values

    dx = abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else res_m
    dy = abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else res_m

    # Ensure north-up orientation (y decreasing top→bottom)
    if len(y_coords) > 1 and y_coords[0] < y_coords[-1]:
        arr = arr[::-1, :]
        y_coords = y_coords[::-1]

    # rasterio from_origin: west edge, north edge, pixel width, pixel height
    transform = from_origin(
        x_coords[0] - dx / 2,
        y_coords[0] + dy / 2,
        dx,
        dy,
    )
    logger.info(f"  Downloaded {asset!r} band={band!r}: {arr.shape} px, res={dx:.1f}m")
    return arr, transform


# ── Classification ────────────────────────────────────────────────────────────


def classify_buildings(
    arr: np.ndarray,
    presence_threshold: float,
    lcz_class: int,
) -> np.ndarray:
    """Threshold building_presence (0–1) to a uint8 LCZ label array.

    Pixels with arr >= presence_threshold are assigned lcz_class; all others 0.
    """
    labeled = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.isfinite(arr)
    labeled[valid & (arr >= presence_threshold)] = lcz_class
    n = int((labeled == lcz_class).sum())
    logger.info(f"  Buildings (presence): {n:,} px ≥ {presence_threshold} → LCZ {lcz_class}")
    return labeled


def classify_building_height(
    arr: np.ndarray,
    low_m: float = 3.0,
    mid_m: float = 10.0,
    high_m: float = 25.0,
    lcz_low: int = 3,
    lcz_mid: int = 2,
    lcz_high: int = 1,
) -> np.ndarray:
    """Classify building_height (metres) into LCZ built-type classes.

    Height tiers follow Stewart & Oke (2012) reference ranges:
        height >= high_m (≥25m, >10 floors)   → lcz_high  (default 1, High-Rise)
        mid_m <= height < high_m (10–25m)      → lcz_mid   (default 2, Mid-Rise)
        low_m <= height < mid_m  (3–10m)       → lcz_low   (default 3, Low-Rise)
        height < low_m  (or nodata)            → 0

    Note: without building density/BSF the compact vs open distinction (e.g.
    LCZ 1 vs 4) cannot be resolved from height alone.  The defaults (1/2/3)
    assume compact morphology; pass lcz_low=6, lcz_mid=5, lcz_high=4 for open.

    Higher tiers overwrite lower ones in the burn order, so high-rise always wins.

    Args:
        arr: Float array of building height in metres (0 = no building / nodata).
        low_m: Minimum height (m) for the low-rise tier.
        mid_m: Minimum height (m) for the mid-rise tier.
        high_m: Minimum height (m) for the high-rise tier.
        lcz_low: LCZ class for low-rise (default 3 = Compact Low-Rise).
        lcz_mid: LCZ class for mid-rise (default 2 = Compact Mid-Rise).
        lcz_high: LCZ class for high-rise (default 1 = Compact High-Rise).

    Returns:
        uint8 array with LCZ built classes, 0 elsewhere.
    """
    labeled = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.isfinite(arr) & (arr > 0)
    labeled[valid & (arr >= low_m)]  = lcz_low
    labeled[valid & (arr >= mid_m)]  = lcz_mid
    labeled[valid & (arr >= high_m)] = lcz_high
    n_high = int((labeled == lcz_high).sum())
    n_mid  = int((labeled == lcz_mid).sum())
    n_low  = int((labeled == lcz_low).sum())
    logger.info(
        f"  Buildings (height): "
        f"{n_low:,} px [{low_m}m,{mid_m}m) → LCZ {lcz_low}  |  "
        f"{n_mid:,} px [{mid_m}m,{high_m}m) → LCZ {lcz_mid}  |  "
        f"{n_high:,} px ≥{high_m}m → LCZ {lcz_high}"
    )
    return labeled


def classify_canopy_height(
    arr: np.ndarray,
    low_m: float,
    high_m: float,
    lcz_scattered: int = 12,
    lcz_dense: int = 11,
) -> np.ndarray:
    """Classify canopy height into LCZ tree classes.

    Thresholding:
        arr >= high_m             → lcz_dense    (LCZ 11 Dense Trees)
        low_m <= arr < high_m     → lcz_scattered (LCZ 12 Scattered Trees)
        arr < low_m  (or nodata)  → 0 (no label)

    Scattered trees are assigned first, then dense trees overwrite — so dense
    always wins in the overlap band at high_m.

    Args:
        arr: Float array of canopy height in metres.
        low_m: Minimum height for scattered trees (LCZ B).
        high_m: Minimum height for dense trees (LCZ A).
        lcz_scattered: LCZ class for scattered/short trees.
        lcz_dense: LCZ class for dense/tall trees.

    Returns:
        uint8 array with LCZ tree classes, 0 elsewhere.
    """
    labeled = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.isfinite(arr) & (arr > 0)
    labeled[valid & (arr >= low_m)] = lcz_scattered
    labeled[valid & (arr >= high_m)] = lcz_dense
    n_dense = int((labeled == lcz_dense).sum())
    n_scat  = int((labeled == lcz_scattered).sum())
    logger.info(
        f"  Canopy classified: {n_dense:,} px ≥ {high_m}m → LCZ {lcz_dense}  |  "
        f"{n_scat:,} px [{low_m}m, {high_m}m) → LCZ {lcz_scattered}"
    )
    return labeled


# ── Merging and writing ───────────────────────────────────────────────────────


def merge_label_layers(
    layers: list[tuple[np.ndarray, object]],
) -> tuple[np.ndarray, object]:
    """Merge LCZ label arrays in burn priority order (last layer wins).

    Args:
        layers: List of (label_array, affine_transform) tuples in ascending
            priority order (index 0 = lowest priority, −1 = highest priority).
            All arrays must have the same shape.

    Returns:
        (merged_array, transform_of_first_layer)
    """
    if not layers:
        raise ValueError("No layers to merge")

    merged = np.zeros(layers[0][0].shape, dtype=np.uint8)
    transform = layers[0][1]

    for arr, _ in layers:
        if arr.shape != merged.shape:
            logger.warning(
                f"Layer shape {arr.shape} differs from base {merged.shape} — "
                "resampling skipped; layer ignored"
            )
            continue
        mask = arr > 0
        merged[mask] = arr[mask]

    return merged, transform


def write_lcz_raster(
    layers: list[tuple[np.ndarray, object, str]],
    crs,
    output_path: str | Path,
    nodata: int = 0,
) -> Path:
    """Write one or more uint8 LCZ arrays as bands in a single GeoTIFF.

    Each layer becomes one band. Band descriptions are set from the layer names.

    Args:
        layers: List of (array, affine_transform, band_name) tuples.
            All arrays must have the same shape; the transform of the first
            layer is used for the output.
        crs: Rasterio/pyproj CRS for the output.
        output_path: Destination .tif path.
        nodata: No-data value (default 0 = unlabelled).

    Returns:
        Path to the written file.
    """
    import rasterio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr0, transform, _ = layers[0]
    n_bands = len(layers)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=arr0.shape[0],
        width=arr0.shape[1],
        count=n_bands,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        for i, (arr, _, name) in enumerate(layers, start=1):
            dst.write(arr, i)
            dst.update_tags(i, name=name)
            unique = np.unique(arr[arr != nodata]).tolist()
            logger.info(f"  Band {i} ({name}): unique classes = {unique}")

    logger.info(
        f"Wrote GEE LCZ raster: {output_path}  "
        f"({arr0.shape[1]}×{arr0.shape[0]} px, {n_bands} band(s))"
    )
    return output_path


# ── Shared helpers (mirrors osm_lcz_labels.py) ───────────────────────────────


def _read_ref_info(ref_path: str | Path):
    """Read (bounds_in_crs, crs, res) from a .tif or vector reference file."""
    import geopandas as gpd
    import rasterio

    ref_path = Path(ref_path)
    if ref_path.suffix.lower() in (".tif", ".tiff"):
        with rasterio.open(ref_path) as src:
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            return bounds, src.crs, src.res[0]

    gdf = gpd.read_file(ref_path)
    return tuple(gdf.total_bounds), gdf.crs, None


def _bounds_to_wgs84(bounds, crs):
    from pyproj import Transformer
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    minx, miny = tf.transform(bounds[0], bounds[1])
    maxx, maxy = tf.transform(bounds[2], bounds[3])
    return (minx, miny, maxx, maxy)


def _utm_crs_for_point(lon: float, lat: float):
    from pyproj import CRS
    zone = int((lon + 180) / 6) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


def _reproject_bounds(bounds, src_crs, dst_crs):
    from pyproj import Transformer
    tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny = tf.transform(bounds[0], bounds[1])
    maxx, maxy = tf.transform(bounds[2], bounds[3])
    return (minx, miny, maxx, maxy)


def _align_to_ref_grid(
    arr: np.ndarray,
    src_transform,
    src_crs,
    ref_bounds_utm: tuple[float, float, float, float],
    res_m: float,
    dst_crs,
    resampling=None,
) -> tuple[np.ndarray, object]:
    """Reproject a GEE-downloaded array onto the exact grid defined by ref_bounds_utm.

    GEE/xee snaps downloads to its own internal global grid, which may not align
    with the reference file's pixel grid. This function reprojects the array to an
    exact grid computed from the reference bounds and the requested resolution, so
    the output always covers exactly the reference extent at *res_m* metre pixels.

    Args:
        arr: 2-D float32 array (north-up) as returned by _download_band.
        src_transform: Affine transform of arr (from _download_band).
        src_crs: CRS of arr (same UTM as dst_crs when called from generate*).
        ref_bounds_utm: (minx, miny, maxx, maxy) in UTM (working) CRS.
        res_m: Target pixel size in metres.
        dst_crs: Target CRS (same as src_crs in normal use).
        resampling: rasterio Resampling method. Defaults to bilinear for float32.

    Returns:
        (aligned_arr, dst_transform) — aligned_arr has shape
        (ceil(height/res), ceil(width/res)) covering ref_bounds_utm exactly.
    """
    import math
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling

    if resampling is None:
        resampling = Resampling.bilinear

    minx, miny, maxx, maxy = ref_bounds_utm
    width  = math.ceil((maxx - minx) / res_m)
    height = math.ceil((maxy - miny) / res_m)
    dst_transform = from_origin(minx, maxy, res_m, res_m)

    dst = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=None,
        dst_nodata=0.0,
    )
    return dst, dst_transform


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def generate(
    ref_path: str = typer.Option(
        ...,
        help=(
            "Path to the city's reference label file (.tif or .gpkg). "
            "Defines spatial extent, CRS, and default resolution. "
            "The gee_labels/ output folder is created next to this file."
        ),
    ),
    year: int | None = typer.Option(
        None,
        help="Year for temporally filtered datasets (buildings). Required when using buildings.",
    ),
    datasets: str = typer.Option(
        "buildings,canopy_height",
        help="Comma-separated datasets to fetch: 'buildings', 'canopy_height', or both.",
    ),
    res: float | None = typer.Option(
        None,
        help="Pixel resolution in metres. Defaults to reference .tif resolution or 100 m.",
    ),
    # Buildings options
    buildings_asset: str = typer.Option(
        BUILDINGS_ASSET,
        help="GEE asset path for the buildings dataset.",
    ),
    buildings_band: str = typer.Option(
        BUILDINGS_BAND,
        help="Band name in the buildings asset ('building_presence' or 'building_height').",
    ),
    buildings_mode: str = typer.Option(
        "auto",
        help=(
            "Classification mode for the buildings band: "
            "'presence' (threshold 0–1 probability), "
            "'height' (height-tier thresholds in metres), "
            "'auto' (inferred from band name)."
        ),
    ),
    # presence mode
    buildings_presence_threshold: float = typer.Option(
        0.5,
        help="[presence mode] Minimum building_presence value to classify as built.",
    ),
    buildings_lcz_class: int = typer.Option(
        3,
        help="[presence mode] LCZ class to assign to all detected buildings.",
    ),
    # height mode
    buildings_low_m: float = typer.Option(
        3.0,
        help="[height mode] Min height (m) for low-rise tier (default → LCZ 3).",
    ),
    buildings_mid_m: float = typer.Option(
        10.0,
        help="[height mode] Min height (m) for mid-rise tier (default → LCZ 2).",
    ),
    buildings_high_m: float = typer.Option(
        25.0,
        help="[height mode] Min height (m) for high-rise tier (default → LCZ 1).",
    ),
    buildings_lcz_low: int = typer.Option(3, help="[height mode] LCZ class for low-rise."),
    buildings_lcz_mid: int = typer.Option(2, help="[height mode] LCZ class for mid-rise."),
    buildings_lcz_high: int = typer.Option(1, help="[height mode] LCZ class for high-rise."),
    # Canopy height options
    canopy_asset: str = typer.Option(
        CANOPY_ASSET,
        help="GEE asset path for the canopy height dataset.",
    ),
    canopy_band: str = typer.Option(
        CANOPY_BAND,
        help="Band name in the canopy height asset.",
    ),
    canopy_low_m: float = typer.Option(
        2.0,
        help="Canopy height (m) above which pixels become LCZ 12 (Scattered Trees).",
    ),
    canopy_high_m: float = typer.Option(
        5.0,
        help="Canopy height (m) above which pixels become LCZ 11 (Dense Trees).",
    ),
    output_path: str | None = typer.Option(
        None,
        help=(
            "Output GeoTIFF path. "
            "Defaults to <ref_path.parent>/gee_labels/gee_lcz_<year>.tif "
            "(or gee_lcz_all.tif when no year is given)."
        ),
    ),
) -> None:
    """Fetch GEE datasets and burn them into a LCZ raster label file.

    Output is written to <ref_path.parent>/gee_labels/ by default.
    Always works in a projected (UTM) CRS regardless of the reference file CRS.
    Canopy height overwrites buildings on overlap (higher priority).
    """
    requested = {d.strip() for d in datasets.split(",") if d.strip()}
    valid = {"buildings", "canopy_height"}
    unknown = requested - valid
    if unknown:
        raise typer.BadParameter(f"Unknown datasets: {unknown}. Valid: {valid}")

    if "buildings" in requested and year is None:
        raise typer.BadParameter("--year is required when using the buildings dataset.")

    # ── Reference file ────────────────────────────────────────────────────────
    ref_p = Path(ref_path)
    if not ref_p.exists():
        raise typer.BadParameter(f"ref-path does not exist: {ref_p}")

    bounds_ref, crs_ref, ref_res = _read_ref_info(ref_p)
    logger.info(f"Reference: {ref_p.name}  CRS: {crs_ref}  bounds: {bounds_ref}")

    # ── Always work in a projected UTM CRS ────────────────────────────────────
    bbox_wgs84 = _bounds_to_wgs84(bounds_ref, crs_ref)

    if crs_ref.is_geographic:
        lon_c = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
        lat_c = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
        working_crs = _utm_crs_for_point(lon_c, lat_c)
        logger.info(
            f"Ref CRS is geographic ({crs_ref}) — using {working_crs} "
            f"for metre-accurate rasterization"
        )
        if res is None:
            res = 100.0
            logger.info("No --res given; defaulting to 100 m")
    else:
        working_crs = crs_ref
        if res is None:
            res = ref_res if ref_res is not None else 100.0

    logger.info(f"Output resolution: {res} m  |  working CRS: {working_crs}")
    working_epsg = working_crs.to_epsg()

    # Pre-compute exact UTM bounds for grid alignment after download
    bounds_utm = (
        _reproject_bounds(bounds_ref, crs_ref, working_crs)
        if crs_ref.is_geographic
        else bounds_ref
    )

    # ── Output path ───────────────────────────────────────────────────────────
    if output_path is None:
        year_tag = str(year) if year is not None else "all"
        out_p = ref_p.parent / "osm_labels" / f"gee_lcz_{year_tag}.tif"
    else:
        out_p = Path(output_path)

    logger.info(f"Output: {out_p}")

    # ── Fetch and classify each dataset — one band per dataset ────────────────
    # layers: list of (labeled_array, transform, band_name)
    layers: list[tuple[np.ndarray, object, str]] = []

    if "buildings" in requested:
        logger.info(f"Fetching buildings ({buildings_asset}, year={year}) …")
        arr_b, transform_b = _download_band(
            asset=buildings_asset,
            band=buildings_band,
            bbox_wgs84=bbox_wgs84,
            target_crs_epsg=working_epsg,
            res_m=res,
            year=year,
        )
        arr_b, transform_b = _align_to_ref_grid(
            arr_b, transform_b, working_crs, bounds_utm, res, working_crs,
            resampling=None,  # bilinear
        )
        mode = buildings_mode
        if mode == "auto":
            mode = "height" if "height" in buildings_band else "presence"
            logger.info(f"  Auto-detected buildings mode: {mode!r}")

        if mode == "height":
            labeled_b = classify_building_height(
                arr_b,
                low_m=buildings_low_m,
                mid_m=buildings_mid_m,
                high_m=buildings_high_m,
                lcz_low=buildings_lcz_low,
                lcz_mid=buildings_lcz_mid,
                lcz_high=buildings_lcz_high,
            )
        else:
            labeled_b = classify_buildings(arr_b, buildings_presence_threshold, buildings_lcz_class)
        layers.append((labeled_b, transform_b, f"buildings_{buildings_band}"))

    if "canopy_height" in requested:
        logger.info(f"Fetching canopy height ({canopy_asset}) …")
        arr_c, transform_c = _download_band(
            asset=canopy_asset,
            band=canopy_band,
            bbox_wgs84=bbox_wgs84,
            target_crs_epsg=working_epsg,
            res_m=res,
            year=None,
        )
        arr_c, transform_c = _align_to_ref_grid(
            arr_c, transform_c, working_crs, bounds_utm, res, working_crs,
            resampling=None,  # bilinear
        )
        labeled_c = classify_canopy_height(arr_c, canopy_low_m, canopy_high_m)
        layers.append((labeled_c, transform_c, "canopy_height"))

    if not layers:
        logger.warning("No datasets fetched — nothing to write.")
        raise typer.Exit(0)

    # ── Write — each dataset as a separate band ────────────────────────────────
    write_lcz_raster(layers, working_crs, out_p)


@app.command()
def generate_features(
    ref_path: str = typer.Option(
        ...,
        help=(
            "Path to the city's reference label file (.tif or .gpkg). "
            "Defines spatial extent, CRS, and default resolution."
        ),
    ),
    year: int | None = typer.Option(
        None,
        help="Year for temporally filtered datasets (buildings). Required when using buildings.",
    ),
    datasets: str = typer.Option(
        "buildings,canopy_height",
        help="Comma-separated datasets to fetch: 'buildings', 'canopy_height', or both.",
    ),
    res: float | None = typer.Option(
        None,
        help="Pixel resolution in metres. Defaults to reference .tif resolution or 100 m.",
    ),
    buildings_asset: str = typer.Option(BUILDINGS_ASSET),
    buildings_band: str = typer.Option(
        "building_height",
        help="Band name in the buildings asset to download as raw values.",
    ),
    canopy_asset: str = typer.Option(CANOPY_ASSET),
    canopy_band: str = typer.Option(CANOPY_BAND),
    output_dir: str | None = typer.Option(
        None,
        help=(
            "Output directory for the per-dataset GeoTIFFs. "
            "Defaults to <ref_path.parent>/gee_features/."
        ),
    ),
) -> None:
    """Download raw GEE bands as float32 predictor features (no LCZ classification).

    Each requested dataset is written as a separate single-band float32 GeoTIFF:

        building_height_m_<year>.tif  — mean building height per pixel in metres
        canopy_height_m.tif           — canopy height per pixel in metres

    Nodata is 0.0 (absence of structure, not an error).
    Output is written to <ref_path.parent>/gee_features/ by default.
    """
    import rasterio

    requested = {d.strip() for d in datasets.split(",") if d.strip()}
    valid = {"buildings", "canopy_height"}
    unknown = requested - valid
    if unknown:
        raise typer.BadParameter(f"Unknown datasets: {unknown}. Valid: {valid}")

    if "buildings" in requested and year is None:
        raise typer.BadParameter("--year is required when using the buildings dataset.")

    # ── Reference file ────────────────────────────────────────────────────────
    ref_p = Path(ref_path)
    if not ref_p.exists():
        raise typer.BadParameter(f"ref-path does not exist: {ref_p}")

    bounds_ref, crs_ref, ref_res = _read_ref_info(ref_p)
    logger.info(f"Reference: {ref_p.name}  CRS: {crs_ref}  bounds: {bounds_ref}")

    bbox_wgs84 = _bounds_to_wgs84(bounds_ref, crs_ref)

    if crs_ref.is_geographic:
        lon_c = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
        lat_c = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
        working_crs = _utm_crs_for_point(lon_c, lat_c)
        logger.info(f"Ref CRS is geographic ({crs_ref}) — using {working_crs} for metre-accurate download")
        if res is None:
            res = 100.0
            logger.info("No --res given; defaulting to 100 m")
    else:
        working_crs = crs_ref
        if res is None:
            res = ref_res if ref_res is not None else 100.0

    logger.info(f"Output resolution: {res} m  |  working CRS: {working_crs}")
    working_epsg = working_crs.to_epsg()

    # Pre-compute exact UTM bounds for grid alignment after download
    bounds_utm = (
        _reproject_bounds(bounds_ref, crs_ref, working_crs)
        if crs_ref.is_geographic
        else bounds_ref
    )

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else ref_p.parent / "gee_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    year_tag = str(year) if year is not None else "all"

    def _write_single_band(arr: np.ndarray, transform, name: str, out_path: Path) -> None:
        with rasterio.open(
            out_path, "w",
            driver="GTiff", height=arr.shape[0], width=arr.shape[1],
            count=1, dtype=np.float32, crs=working_crs, transform=transform,
            nodata=0.0, compress="lzw",
        ) as dst:
            dst.write(arr, 1)
            dst.update_tags(1, name=name)
        logger.info(
            f"Wrote {name}: {out_path.name}  "
            f"({arr.shape[1]}×{arr.shape[0]} px, float32)"
        )

    written: list[Path] = []

    if "buildings" in requested:
        logger.info(f"Fetching buildings ({buildings_asset}, year={year}, band={buildings_band}) …")
        arr_b, transform_b = _download_band(
            asset=buildings_asset,
            band=buildings_band,
            bbox_wgs84=bbox_wgs84,
            target_crs_epsg=working_epsg,
            res_m=res,
            year=year,
        )
        arr_b, transform_b = _align_to_ref_grid(
            arr_b, transform_b, working_crs, bounds_utm, res, working_crs,
        )
        arr_b = np.nan_to_num(arr_b, nan=0.0).astype(np.float32)
        logger.info(
            f"  Buildings raw range: [{arr_b[arr_b > 0].min():.1f}, {arr_b.max():.1f}] m  "
            f"({int((arr_b > 0).sum()):,} non-zero px)"
        )
        out_b = out_dir / f"building_height_m_{year_tag}.tif"
        _write_single_band(arr_b, transform_b, "building_height_m", out_b)
        written.append(out_b)

    if "canopy_height" in requested:
        logger.info(f"Fetching canopy height ({canopy_asset}, band={canopy_band}) …")
        arr_c, transform_c = _download_band(
            asset=canopy_asset,
            band=canopy_band,
            bbox_wgs84=bbox_wgs84,
            target_crs_epsg=working_epsg,
            res_m=res,
            year=None,
        )
        arr_c, transform_c = _align_to_ref_grid(
            arr_c, transform_c, working_crs, bounds_utm, res, working_crs,
        )
        arr_c = np.nan_to_num(arr_c, nan=0.0).astype(np.float32)
        logger.info(
            f"  Canopy raw range: [{arr_c[arr_c > 0].min():.1f}, {arr_c.max():.1f}] m  "
            f"({int((arr_c > 0).sum()):,} non-zero px)"
        )
        out_c = out_dir / "canopy_height_m.tif"
        _write_single_band(arr_c, transform_c, "canopy_height_m", out_c)
        written.append(out_c)

    if not written:
        logger.warning("No datasets fetched — nothing to write.")
        raise typer.Exit(0)

    logger.info(f"Done. Wrote {len(written)} file(s) to {out_dir}")


if __name__ == "__main__":
    app()
