"""Raw source-embedding tile access: spatial index, tile opening, patch cropping.

Used by the extraction scripts (extract_so2sat_embeddings.py,
extract_grid_embeddings.py) and the ROI inference module (infer_roi.py).
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree


TESSERA_GRID_RE = re.compile(r"grid_(?P<lon>-?[\d.]+)_(?P<lat>-?[\d.]+)$")


@functools.lru_cache(maxsize=None)
def tessera_grid_geometry(tile_name: str):
    """CRS + affine transform of a tessera 0.1° tile, derived from its name.

    Tessera names a tile by the centre of its 0.1° cell (``grid_{lon}_{lat}``)
    and rasterises it at 10 m in the UTM zone of that centre, so the raster
    origin is that cell reprojected into the zone.  Pixel counts are not
    needed here — callers take them from the array they loaded.

    Tessera v2 ships no geoinfo tiff, so this is the only geometry source for
    it.  Checked against v1.1's global_0.1_degree_tiff_all on 699 tiles — 393
    random plus 306 in the Norway/Svalbard/polar/dateline bands where UTM has
    zone exceptions — with zero CRS, origin (<1 mm) or shape mismatches.
    """
    import rasterio.warp
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    m = TESSERA_GRID_RE.match(tile_name)
    if m is None:
        raise ValueError(f"Cannot parse lon/lat from tessera tile name: {tile_name}")
    lon, lat = float(m.group("lon")), float(m.group("lat"))

    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    crs = CRS.from_epsg(epsg)
    left, _, _, top = rasterio.warp.transform_bounds(
        "EPSG:4326", crs, lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05
    )
    return crs, from_origin(left, top, 10, 10)


def tessera_grid_bounds_4326(tile_name: str) -> tuple[float, float, float, float]:
    """WGS84 bounds of a tessera 0.1° tile's raster, from its name alone.

    Same quantity the tesserav1.1_global branch reads out of a geoinfo tiff
    (raster bounds reprojected to EPSG:4326), so both indexes hold comparable
    footprints — slightly larger than the nominal cell, since the UTM
    rectangle circumscribes the lat/lon quadrilateral.  The pixel counts
    follow from the reprojected cell (validated against the npy shape on 500
    v2 tiles), so no file has to be opened to index a tile.
    """
    import rasterio.warp

    crs, _ = tessera_grid_geometry(tile_name)
    m = TESSERA_GRID_RE.match(tile_name)
    lon, lat = float(m.group("lon")), float(m.group("lat"))
    left, bottom, right, top = rasterio.warp.transform_bounds(
        "EPSG:4326", crs, lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05
    )
    width, height = round((right - left) / 10), round((top - bottom) / 10)
    return rasterio.warp.transform_bounds(
        crs, "EPSG:4326", left, top - height * 10, left + width * 10, top
    )


# ---------------------------------------------------------------------------
# Tile index
# ---------------------------------------------------------------------------

def load_coop_index(coop_dir: Path, year: str | int | None = None):
    """Load ``aef_index.gpkg`` from a coop root directory.

    Single source of truth for coop tile metadata (also used by the coop
    download scripts). Returns the index GeoDataFrame filtered to *year*
    (if given), with two derived columns:

        local_path  Path — where the tile lives under *coop_dir*
                    (the S3 ``path`` column with the prefix stripped)
        is_local    bool — whether that file exists

    Use ``coop_wgs84_boxes`` to build shapely boxes over the reported
    valid bounds (``wgs84_*`` columns) for STRtree queries.
    """
    import geopandas as gpd
    from datasets.downloaders import COOP_S3_PREFIX

    index_path = Path(coop_dir) / "aef_index.gpkg"
    if not index_path.exists():
        raise FileNotFoundError(
            f"aef_index.gpkg not found at {index_path}. "
            "Pass the coop root directory (containing aef_index.gpkg)."
        )
    read_kwargs: dict = {}
    if year is not None:
        read_kwargs["where"] = f"year = {int(year)}"
    gdf = gpd.read_file(index_path, **read_kwargs)
    gdf["local_path"] = [
        Path(coop_dir) / p.removeprefix(COOP_S3_PREFIX) for p in gdf["path"]
    ]
    gdf["is_local"] = [p.exists() for p in gdf["local_path"]]
    return gdf


def coop_wgs84_boxes(coop_index) -> list:
    """Shapely boxes over a coop index's reported WGS84 valid bounds."""
    return [
        box(r.wgs84_west, r.wgs84_south, r.wgs84_east, r.wgs84_north)
        for r in coop_index.itertuples(index=False)
    ]


@functools.lru_cache(maxsize=4)
def build_tile_index(
    embedding_dir: Path,
    embedding_name: str,
    year: str | None = None,
) -> tuple[list[Path], STRtree]:
    """Glob the embedding directory once and build an STRtree over tile bboxes.

    Returns (tile_paths, tree) where tree is built over tile bounding boxes in
    EPSG:4326.  Indexing into tile_paths with positions returned by
    tree.query() gives the matching Path objects.

    Cached: multi-city callers (e.g. generate_seg_pseudo_rasters.py) reuse the
    index instead of re-scanning the tile directory per city.
    """
    from datasets.registry import EMBEDDING_REGISTRY

    # ── coop: spatial index from aef_index.gpkg ──────────────────────────────
    if embedding_name == "alpha_earth_coop":
        gdf = load_coop_index(embedding_dir, year)
        local = gdf[gdf["is_local"]]
        if local.empty:
            raise FileNotFoundError(
                f"No local coop tiles found under {embedding_dir} for year={year}."
            )
        paths = list(local["local_path"])
        logger.info(f"Tile index (coop): {len(paths)} local tiles from {embedding_dir}")
        return paths, STRtree(coop_wgs84_boxes(local))

    # ── tessera v1.1: spatial index from geoinfo tiffs ───────────────────────
    if embedding_name == "tesserav1.1":
        import rasterio
        import rasterio.warp

        geoinfo_dir = embedding_dir / "geoinfo"
        if not geoinfo_dir.exists():
            raise FileNotFoundError(
                f"geoinfo/ subdirectory not found under {embedding_dir}. "
                "Pass the year directory (e.g. .../GeoTessera/v1.1/2017) as --embedding-dir."
            )
        tiff_files = sorted(geoinfo_dir.glob("*.tiff")) + sorted(geoinfo_dir.glob("*.tif"))
        if not tiff_files:
            raise FileNotFoundError(f"No geoinfo tiff files found in {geoinfo_dir}")
        paths, geoms = [], []
        for p in tiff_files:
            with rasterio.open(p) as ds:
                l, b, r, t = rasterio.warp.transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
            geoms.append(box(l, b, r, t))
            paths.append(p)
        logger.info(f"Tile index (tesserav1.1): {len(paths)} tiles from {geoinfo_dir}")
        return paths, STRtree(geoms)

    # ── tessera v1.1 global: NPY subdirs + tiff_all for geo metadata ─────────
    if embedding_name == "tesserav1.1_global":
        import rasterio
        import rasterio.warp

        if year is None:
            raise ValueError("--year is required for tesserav1.1_global.")

        tiff_dir = embedding_dir / "global_0.1_degree_tiff_all"
        npy_root = embedding_dir / "global_0.1_degree_representation" / str(year)

        if not tiff_dir.exists():
            raise FileNotFoundError(
                f"global_0.1_degree_tiff_all/ not found under {embedding_dir}. "
                "Pass the v1.1 root (e.g. /tessera/v1.1) as --embedding-dir."
            )
        if not npy_root.exists():
            raise FileNotFoundError(
                f"global_0.1_degree_representation/{year}/ not found under {embedding_dir}."
            )

        paths, geoms = [], []
        for npy_dir in sorted(npy_root.iterdir()):
            if not npy_dir.is_dir():
                continue
            tiff_path = tiff_dir / f"{npy_dir.name}.tiff"
            if not tiff_path.exists():
                continue
            with rasterio.open(tiff_path) as ds:
                l, b, r, t = rasterio.warp.transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
            geoms.append(box(l, b, r, t))
            paths.append(npy_dir)
        logger.info(f"Tile index (tesserav1.1_global): {len(paths)} tiles from {npy_root}")
        return paths, STRtree(geoms)

    # ── tessera v2: NPY subdirs, geometry derived from the tile names ────────
    if embedding_name == "tesserav2":
        if year is None:
            raise ValueError("--year is required for tesserav2.")

        root = embedding_dir / "large_student"
        if not root.exists():
            root = embedding_dir  # --embedding-dir already points at a variant
        npy_root = root / str(year)
        if not npy_root.exists():
            raise FileNotFoundError(
                f"{year}/ not found under {root}. Pass the v2 variant root "
                "(e.g. /tessera/v2/large_student) as --embedding-dir."
            )

        # v2 has no geoinfo tiffs, so footprints come from the tile names
        # (tessera_grid_bounds_4326) rather than from opening 40k rasters.
        paths, geoms = [], []
        for npy_dir in sorted(npy_root.iterdir()):
            if not npy_dir.is_dir() or not TESSERA_GRID_RE.match(npy_dir.name):
                continue
            geoms.append(box(*tessera_grid_bounds_4326(npy_dir.name)))
            paths.append(npy_dir)
        if not paths:
            raise FileNotFoundError(f"No grid_* tile directories found in {npy_root}")
        logger.info(f"Tile index (tesserav2): {len(paths)} tiles from {npy_root}")
        return paths, STRtree(geoms)

    # ── seamless (ESD): spatial index from rasterio bounds ───────────────────
    if embedding_name == "seamless":
        import rasterio
        import rasterio.warp

        all_tiffs = sorted(embedding_dir.rglob("SDC30_EBD_V001_*.tiff"))
        all_tiffs += sorted(embedding_dir.rglob("SDC30_EBD_V001_*.tif"))
        if not all_tiffs:
            raise FileNotFoundError(
                f"No ESD tiles (SDC30_EBD_V001_*.tiff) found under {embedding_dir}"
            )
        paths, geoms = [], []
        for p in all_tiffs:
            with rasterio.open(p) as ds:
                l, b, r, t = rasterio.warp.transform_bounds(
                    ds.crs, "EPSG:4326", *ds.bounds
                )
            geoms.append(box(l, b, r, t))
            paths.append(p)
        logger.info(f"Tile index (seamless): {len(paths)} tiles from {embedding_dir}")
        return paths, STRtree(geoms)

    # ── zarr / tif: spatial index from filename patterns ─────────────────────
    meta = EMBEDDING_REGISTRY.get(embedding_name, {})
    pattern = meta.get("zarr_filename_pattern")
    tile_size = meta.get("zarr_tile_size")
    is_center = meta.get("zarr_filename_is_center")

    if pattern is None or tile_size is None or is_center is None:
        supported = sorted(
            k for k, m in EMBEDDING_REGISTRY.items() if m.get("zarr_filename_pattern")
        )
        raise ValueError(
            f"Embedding '{embedding_name}' has no filename-pattern metadata. "
            f"Embeddings with filename-indexed tiles: {supported} "
            "(tesserav1.1*/tesserav2/seamless are handled by their own branches above)."
        )

    regex = re.compile(pattern)
    half = tile_size / 2
    paths = []
    geoms = []

    all_files = sorted(embedding_dir.glob("*.zarr")) + sorted(embedding_dir.glob("*.tif"))
    for p in all_files:
        m = regex.match(p.name)
        if m is None:
            continue
        lon, lat = float(m.group("lon")), float(m.group("lat"))
        if is_center:
            geoms.append(box(lon - half, lat - half, lon + half, lat + half))
        else:
            geoms.append(box(lon, lat, lon + tile_size, lat + tile_size))
        paths.append(p)

    if not paths:
        raise FileNotFoundError(
            f"No {embedding_name} tiles found in {embedding_dir}"
        )

    logger.info(f"Tile index: {len(paths)} tiles loaded from {embedding_dir}")
    return paths, STRtree(geoms)


def build_coop_valid_bbox_map(
    embedding_dir: Path,
    year: str | None,
    matched_paths: list[Path],
) -> dict[Path, tuple[float, float, float, float]]:
    """Map coop tile path → reported WGS84 valid bbox from aef_index.gpkg.

    Coop tile data physically overshoots the UTM zone boundary (e.g. a UTM30
    tile extends ~0.5° into UTM31 territory) but the index clips the reported
    WGS84 bounds to the zone edge.  Intersecting clips with these bounds
    discards the contaminated overhang region so adjacent-zone tiles never
    overwrite clean predictions across the boundary.
    """
    idx = load_coop_index(embedding_dir, year)
    name_to_bounds: dict[str, tuple] = {
        Path(r["path"]).name: (
            r["wgs84_west"], r["wgs84_south"],
            r["wgs84_east"], r["wgs84_north"],
        )
        for _, r in idx.iterrows()
    }
    return {
        p: name_to_bounds[p.name]
        for p in matched_paths
        if p.name in name_to_bounds
    }


# ---------------------------------------------------------------------------
# Tile opening
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _open_tile_tessera11(path: Path) -> xr.DataArray:
    """Open a tessera v1.1 tile from a geoinfo tiff path.

    Finds the paired int8+scales npy files in the sibling infer_output/ directory,
    dequantizes via load_and_dequantize_tessera_representation(), and returns a
    (band, y, x) DataArray with the geoinfo tiff's CRS and pixel coordinates.
    """
    import rasterio
    import rioxarray  # noqa: F401

    m = re.match(r"grid_([-\d.]+)_([-\d.]+)\.tiff?", path.name)
    if m is None:
        raise ValueError(f"Cannot parse lon/lat from geoinfo filename: {path.name}")
    lon, lat = m.group(1), m.group(2)

    infer_dir = path.parent.parent / "infer_output"
    int8_files = sorted(infer_dir.glob(f"*_grid_{lon}_{lat}_all_data_emb128_int8.npy"))
    scales_files = sorted(infer_dir.glob(f"*_grid_{lon}_{lat}_all_data_emb128_scales.npy"))
    if not int8_files or not scales_files:
        raise FileNotFoundError(
            f"No int8/scales npy pair found for tile ({lon}, {lat}) in {infer_dir}"
        )

    from dequantize_embeddings import load_and_dequantize_tessera_representation
    arr_hwc = load_and_dequantize_tessera_representation(int8_files[0], scales_files[0])
    arr_chw = arr_hwc.transpose(2, 0, 1)  # (128, H, W)

    with rasterio.open(path) as ds:
        crs = ds.crs
        transform = ds.transform
        H, W = ds.height, ds.width

    x_coords = [transform.c + (i + 0.5) * transform.a for i in range(W)]
    y_coords = [transform.f + (i + 0.5) * transform.e for i in range(H)]

    da = xr.DataArray(
        arr_chw,
        dims=("band", "y", "x"),
        coords={"band": np.arange(arr_chw.shape[0]), "y": y_coords, "x": x_coords},
    )
    return da.rio.write_crs(crs)


@functools.lru_cache(maxsize=2)
def _open_tile_tessera_npy_dir(npy_dir: Path) -> xr.DataArray:
    """Open a tessera global tile (v1.1 or v2) from its NPY subdirectory.

    Loads {tile_name}.npy + {tile_name}_scales.npy from npy_dir, dequantizes,
    and returns a (band, y, x) DataArray with CRS and pixel coordinates.

    Geo metadata comes from the sibling global_0.1_degree_tiff_all/ TIFF when
    one exists (tesserav1.1_global's year-independent source), else from the
    tile name via tessera_grid_geometry — which is how tesserav2 tiles, and
    the v1.1 tiles with no tiff, are georeferenced.  The two agree exactly
    (see tessera_grid_geometry).

    npy_dir layouts:
        v1.1  .../global_0.1_degree_representation/{year}/grid_{lon}_{lat}/
        v2    .../large_student/{year}/grid_{lon}_{lat}/
    """
    import rasterio
    import rioxarray  # noqa: F401

    tile_name = npy_dir.name  # e.g. grid_-0.05_51.45

    int8_path = npy_dir / f"{tile_name}.npy"
    scales_path = npy_dir / f"{tile_name}_scales.npy"
    if not int8_path.exists() or not scales_path.exists():
        raise FileNotFoundError(f"NPY files not found in {npy_dir}")

    from dequantize_embeddings import load_and_dequantize_tessera_representation
    arr_hwc = load_and_dequantize_tessera_representation(int8_path, scales_path)
    arr_chw = arr_hwc.transpose(2, 0, 1)  # (128, H, W)
    H, W = arr_chw.shape[1], arr_chw.shape[2]

    # .../{representation_root}/{year}/grid_*/ → up 3 levels for the v1.1 root
    tiff_path = npy_dir.parent.parent.parent / "global_0.1_degree_tiff_all" / f"{tile_name}.tiff"
    if tiff_path.exists():
        with rasterio.open(tiff_path) as ds:
            crs, transform = ds.crs, ds.transform
            H, W = ds.height, ds.width
    else:
        crs, transform = tessera_grid_geometry(tile_name)

    x_coords = [transform.c + (i + 0.5) * transform.a for i in range(W)]
    y_coords = [transform.f + (i + 0.5) * transform.e for i in range(H)]

    da = xr.DataArray(
        arr_chw,
        dims=("band", "y", "x"),
        coords={"band": np.arange(arr_chw.shape[0]), "y": y_coords, "x": x_coords},
    )
    return da.rio.write_crs(crs)


def _open_tile_aux_struct(path: Path) -> xr.DataArray:
    """Open a precomputed 4-band auxiliary-structural tile (lazily).

    path = .../aux_struct/merged_aux/aux_{lon}_{lat}.tif (10 m grid), built by
    precompute_aux_tiles.py as a tiled+DEFLATE GeoTIFF. Bands, already
    normalised ~[0,1] with NaN→0:
      0 ANBH/50 · 1 built fraction · 2 non-res built fraction · 3 canopy/50.

    Returned lazily (not cached, no eager .astype) so crop_patch's clip_box
    reads only the ~1 block overlapping the patch instead of the whole ~480 MB
    tile — the striped/eager version made extraction ~10 patch/s. The merge/
    reproject is done once per tile at precompute time, not per patch here.
    """
    import rioxarray as rxr

    return rxr.open_rasterio(path, chunks={"band": -1, "x": 512, "y": 512})


def open_tile(path: Path) -> xr.DataArray:
    """Open a .zarr or .tif tile as a (band, y, x) DataArray with CRS set."""
    import rioxarray as rxr

    # Tessera global (v1.1 / v2): path is the NPY subdir
    if path.is_dir():
        return _open_tile_tessera_npy_dir(path)

    # Auxiliary structural bands: precomputed 4-band merged tile
    if path.name.startswith("aux_") and path.parent.name == "merged_aux":
        return _open_tile_aux_struct(path)

    # Tessera v1.1: geoinfo tiff with a sibling infer_output/ directory
    if path.suffix in (".tiff", ".tif") and path.parent.name == "geoinfo":
        infer_dir = path.parent.parent / "infer_output"
        if infer_dir.exists():
            return _open_tile_tessera11(path)

    if path.suffix == ".zarr":
        ds = xr.open_zarr(str(path), chunks=False)
        da = ds["embedding"]
        if da.dims != ("band", "y", "x"):
            da = da.transpose("band", "y", "x")
        if da.rio.crs is None and "spatial_ref" in ds:
            crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
            if crs_wkt:
                da = da.rio.write_crs(crs_wkt)
    else:
        da = rxr.open_rasterio(path)

    return da


# ---------------------------------------------------------------------------
# Patch cropping
# ---------------------------------------------------------------------------

def numpy_mosaic(arrays: list[xr.DataArray]) -> np.ndarray:
    """Mosaic multiple north-up DataArrays via coordinate-based pixel placement.

    Avoids rasterio.merge which rejects south-up rasters.  All input arrays
    must already be normalised to north-up (descending y) before calling this.
    Where tiles overlap, the last array in the list wins (last-writer-wins);
    callers that care about overlap quality must order or pre-clip the inputs.
    """
    a0 = arrays[0]
    yres = abs(float(a0.y.values[0] - a0.y.values[1])) if len(a0.y) > 1 else 1.0
    xres = abs(float(a0.x.values[1] - a0.x.values[0])) if len(a0.x) > 1 else 1.0

    # Output extent: half-pixel beyond outermost pixel centres
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

        r_mask = (row_idx >= 0) & (row_idx < target_h)
        c_mask = (col_idx >= 0) & (col_idx < target_w)
        if not r_mask.any() or not c_mask.any():
            continue

        ri = np.where(r_mask)[0]
        ci = np.where(c_mask)[0]
        out[:, row_idx[r_mask][:, None], col_idx[c_mask][None, :]] = \
            vals[:, ri[:, None], ci[None, :]]

    return out


def merge_multi_crs(arrays: list[xr.DataArray], patch_geom, patch_crs: str) -> np.ndarray:
    """Merge clipped tiles that do not share a CRS onto one grid.

    numpy_mosaic places pixels by raw coordinate value, which is meaningless
    across CRSs: tessera tiles carry per-tile UTM zones, so a patch straddling
    a zone boundary (lon 0 for London, ±6° elsewhere) mixed eastings from two
    zones into an array tens of thousands of pixels wide.

    The grid of the tile contributing the most pixels wins; the patch bounds
    are snapped to that tile's pixel grid, every array is warped onto it, and
    holes are filled from the remaining arrays in decreasing size order.
    """
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    ordered = sorted(arrays, key=lambda a: a.size, reverse=True)
    target = ordered[0]
    dst_crs = target.rio.crs
    tr = target.rio.transform()
    res_x, res_y = abs(tr.a), abs(tr.e)

    geom = patch_geom
    if patch_crs != dst_crs.to_string():
        t = Transformer.from_crs(patch_crs, dst_crs, always_xy=True)
        geom = shapely_transform(t.transform, geom)
    minx, miny, maxx, maxy = geom.bounds

    # Snap the output origin to the target tile's pixel grid so no resampling
    # shift is introduced for the dominant tile.
    left = tr.c + np.floor((minx - tr.c) / res_x) * res_x
    top = tr.f - np.floor((tr.f - maxy) / res_y) * res_y
    width = max(1, int(np.ceil((maxx - left) / res_x)))
    height = max(1, int(np.ceil((top - miny) / res_y)))
    dst_transform = from_origin(left, top, res_x, res_y)

    out = None
    for a in ordered:
        warped = a.rio.reproject(
            dst_crs,
            transform=dst_transform,
            shape=(height, width),
            resampling=Resampling.nearest,
            nodata=np.nan,
        ).values.astype(np.float32)
        if out is None:
            out = warped
            continue
        holes = np.isnan(out) & ~np.isnan(warped)
        out[holes] = warped[holes]
        if not np.isnan(out).any():
            break

    return np.nan_to_num(out, nan=0.0)


def crop_patch(
    patch_geom,
    patch_crs: str,
    tile_paths: list[Path],
    output_path: Path,
    skip_existing: bool = False,
    dtype: str = "float32",
    valid_bboxes: dict[Path, tuple[float, float, float, float]] | None = None,
) -> bool:
    """Clip *tile_paths* to *patch_geom*, mosaic if needed, save as .npy
    (``dtype``: float32 default, or float16 to halve disk use).

    Clips each tile to the patch bounds BEFORE any y-flip so that zarr only
    reads the relevant chunks (~15 MB) rather than the full ~244 MB tile.
    Uses a numpy coordinate mosaic instead of rasterio.merge to support
    south-up (AlphaEarth) tiles.

    ``valid_bboxes`` maps a tile path to its reported WGS84 valid bounds
    (see ``build_coop_valid_bbox_map``): the patch is intersected with them
    per tile, so coop pixels overshooting the tile's UTM zone never enter
    the mosaic — matching what infer_roi does at inference time.

    Returns True on success, False if the clip yields no data.
    """
    if skip_existing and output_path.exists():
        return True

    # Patch geometry in WGS84, for intersecting with the (WGS84) valid bboxes.
    patch_4326 = patch_geom
    if valid_bboxes and patch_crs.upper() not in ("EPSG:4326", "OGC:CRS84"):
        to_4326 = Transformer.from_crs(patch_crs, "EPSG:4326", always_xy=True)
        patch_4326 = shapely_transform(to_4326.transform, patch_geom)

    to_tile_crs: dict[str, Transformer] = {}
    arrays: list[xr.DataArray] = []
    for path in tile_paths:
        try:
            da = open_tile(path)
        except Exception as e:
            logger.warning(f"Skipping tile {path}: {e}")
            continue
        if da.rio.crs is None:
            continue

        clip_geom = patch_geom
        vb = (valid_bboxes or {}).get(path)
        if vb is not None:
            inter = patch_4326.intersection(box(*vb))
            if inter.is_empty:
                continue  # patch lies entirely in this tile's zone overhang
            if patch_4326 is not patch_geom:
                back = Transformer.from_crs("EPSG:4326", patch_crs, always_xy=True)
                inter = shapely_transform(back.transform, inter)
            clip_geom = inter

        # Patch bounds in tile CRS (transformer cached per CRS — coop tiles at
        # a zone boundary legitimately mix UTM zones within one patch).
        tile_crs_str = da.rio.crs.to_string()
        if tile_crs_str != patch_crs:
            t = to_tile_crs.get(tile_crs_str)
            if t is None:
                t = Transformer.from_crs(patch_crs, tile_crs_str, always_xy=True)
                to_tile_crs[tile_crs_str] = t
            clip_geom = shapely_transform(t.transform, clip_geom)

        minx, miny, maxx, maxy = clip_geom.bounds
        try:
            clipped = da.rio.clip_box(minx, miny, maxx, maxy)
        except Exception:
            continue

        if clipped.size == 0:
            continue

        # Normalise south-up → north-up on the already-small clipped array.
        if clipped.sizes.get("y", 0) > 1 and float(clipped.y.values[0]) < float(clipped.y.values[-1]):
            clipped = clipped.isel(y=slice(None, None, -1))

        arrays.append(clipped)

    if not arrays:
        return False

    if len(arrays) == 1:
        arr = arrays[0].values.astype(np.float32)
    elif len({a.rio.crs.to_string() for a in arrays}) > 1:
        arr = merge_multi_crs(arrays, patch_geom, patch_crs)
    else:
        arr = numpy_mosaic(arrays)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr.astype(np.float16) if dtype == "float16" else arr)
    return True


# Backward-compat aliases (older scripts imported the underscore names)
_open_tile_tessera11_global = _open_tile_tessera_npy_dir
_build_tile_index = build_tile_index
_open_tile = open_tile
_crop_patch = crop_patch
_numpy_mosaic = numpy_mosaic
