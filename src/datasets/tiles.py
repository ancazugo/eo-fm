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


# ---------------------------------------------------------------------------
# Tile index
# ---------------------------------------------------------------------------

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
        import geopandas as gpd

        index_path = embedding_dir / "aef_index.gpkg"
        if not index_path.exists():
            raise FileNotFoundError(
                f"aef_index.gpkg not found at {index_path}. "
                "Pass the coop root directory (containing aef_index.gpkg) as --embedding-dir."
            )
        read_kwargs: dict = {}
        if year is not None:
            read_kwargs["where"] = f"year = {int(year)}"
        gdf = gpd.read_file(index_path, **read_kwargs)

        _s3_prefix = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/"
        paths: list[Path] = []
        geoms = []
        for _, row in gdf.iterrows():
            local = embedding_dir / row["path"].removeprefix(_s3_prefix)
            if not local.exists():
                continue
            geoms.append(box(
                row["wgs84_west"], row["wgs84_south"],
                row["wgs84_east"], row["wgs84_north"],
            ))
            paths.append(local)

        if not paths:
            raise FileNotFoundError(
                f"No local coop tiles found under {embedding_dir} for year={year}."
            )
        logger.info(f"Tile index (coop): {len(paths)} local tiles from {index_path}")
        return paths, STRtree(geoms)

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
            "(tesserav1.1*/seamless are handled by their own branches above)."
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
    import geopandas as gpd

    idx = gpd.read_file(
        embedding_dir / "aef_index.gpkg",
        where=f"year = {int(year)}" if year else "",
    )
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
def _open_tile_tessera11_global(npy_dir: Path) -> xr.DataArray:
    """Open a tessera v1.1 global tile from its NPY subdirectory.

    Loads {tile_name}.npy + {tile_name}_scales.npy from npy_dir, dequantizes,
    and returns a (band, y, x) DataArray with CRS and pixel coordinates from
    the sibling global_0.1_degree_tiff_all/ TIFF (year-independent geo metadata).

    npy_dir layout: .../global_0.1_degree_representation/{year}/grid_{lon}_{lat}/
    """
    import rasterio
    import rioxarray  # noqa: F401

    tile_name = npy_dir.name  # e.g. grid_-0.05_51.45

    int8_path = npy_dir / f"{tile_name}.npy"
    scales_path = npy_dir / f"{tile_name}_scales.npy"
    if not int8_path.exists() or not scales_path.exists():
        raise FileNotFoundError(f"NPY files not found in {npy_dir}")

    # .../global_0.1_degree_representation/{year}/grid_*/ → go up 3 levels for v1.1 root
    tiff_path = npy_dir.parent.parent.parent / "global_0.1_degree_tiff_all" / f"{tile_name}.tiff"
    if not tiff_path.exists():
        raise FileNotFoundError(f"Geoinfo TIFF not found: {tiff_path}")

    from dequantize_embeddings import load_and_dequantize_tessera_representation
    arr_hwc = load_and_dequantize_tessera_representation(int8_path, scales_path)
    arr_chw = arr_hwc.transpose(2, 0, 1)  # (128, H, W)

    with rasterio.open(tiff_path) as ds:
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

    # Tessera v1.1 global: path is the NPY subdir
    if path.is_dir():
        return _open_tile_tessera11_global(path)

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


def crop_patch(
    patch_geom,
    patch_crs: str,
    tile_paths: list[Path],
    output_path: Path,
    skip_existing: bool = False,
    dtype: str = "float32",
) -> bool:
    """Clip *tile_paths* to *patch_geom*, mosaic if needed, save as .npy
    (``dtype``: float32 default, or float16 to halve disk use).

    Clips each tile to the patch bounds BEFORE any y-flip so that zarr only
    reads the relevant chunks (~15 MB) rather than the full ~244 MB tile.
    Uses a numpy coordinate mosaic instead of rasterio.merge to support
    south-up (AlphaEarth) tiles.

    Returns True on success, False if the clip yields no data.
    """
    if skip_existing and output_path.exists():
        return True

    tile_crs_str: str | None = None
    patch_in_tile = patch_geom

    arrays: list[xr.DataArray] = []
    for path in tile_paths:
        try:
            da = open_tile(path)
        except Exception as e:
            logger.warning(f"Skipping tile {path}: {e}")
            continue
        if da.rio.crs is None:
            continue

        # Compute patch bounds in tile CRS once (all tiles share the same CRS
        # within a geographic region for a given embedding type).
        if tile_crs_str is None:
            tile_crs_str = da.rio.crs.to_string()
            if tile_crs_str != patch_crs:
                t = Transformer.from_crs(patch_crs, tile_crs_str, always_xy=True)
                patch_in_tile = shapely_transform(t.transform, patch_geom)

        minx, miny, maxx, maxy = patch_in_tile.bounds
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

    arr = numpy_mosaic(arrays) if len(arrays) > 1 else arrays[0].values.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr.astype(np.float16) if dtype == "float16" else arr)
    return True


# Backward-compat aliases (older scripts imported the underscore names)
_build_tile_index = build_tile_index
_open_tile = open_tile
_crop_patch = crop_patch
_numpy_mosaic = numpy_mosaic
