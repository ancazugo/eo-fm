"""Extract embedding crops for all So2Sat LCZ42 patches.

For each patch in patches_reference_rxr.gpkg the script:
  1. Finds embedding tile(s) that cover the patch (no file I/O — filename parsing only).
  2. Clips the embedding to the patch bounding box.
  3. Saves the result as a float32 .npy file.

Output layout:
    {so2sat_dir}/{split}/{output_name}/{year}/patch_{patch_id}.npy

Example:
    python src/extract_so2sat_embeddings.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017 \\
        --embedding-name alpha_earth \\
        --output-name AlphaEarth \\
        --year 2017 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import functools
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from loguru import logger
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tile index helpers
# ---------------------------------------------------------------------------

def _build_tile_index(
    embedding_dir: Path,
    embedding_name: str,
    year: str | None = None,
) -> tuple[list[Path], STRtree]:
    """Glob the embedding directory once and build an STRtree over tile bboxes.

    Returns (tile_paths, tree) where tree is built over tile bounding boxes in
    EPSG:4326.  Indexing into tile_paths with positions returned by
    tree.query() gives the matching Path objects.
    """
    # Import here so sys.path adjustments made before the call take effect.
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
                f"No local coop tiles found under {embedding_dir} for year={year}. "
                "Run 'python src/cli.py download-coop' first."
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
        raise ValueError(
            f"Embedding '{embedding_name}' has no filename-pattern metadata. "
            "Only 'tessera', 'alpha_earth', and 'seamless' are supported."
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


# ---------------------------------------------------------------------------
# Per-patch crop
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _open_tile_tessera11(path: Path) -> xr.DataArray:
    """Open a tessera v1.1 tile from a geoinfo tiff path.

    Finds the paired int8+scales npy files in the sibling infer_output/ directory,
    dequantizes via load_and_dequantize_tessera_representation(), and returns a
    (band, y, x) DataArray with the geoinfo tiff's CRS and pixel coordinates.
    """
    import re
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


def _open_tile(path: Path) -> xr.DataArray:
    """Open a .zarr or .tif tile as a (band, y, x) DataArray with CRS set."""
    import rioxarray as rxr

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


def _numpy_mosaic(arrays: list[xr.DataArray]) -> np.ndarray:
    """Mosaic multiple north-up DataArrays via coordinate-based pixel placement.

    Avoids rasterio.merge which rejects south-up rasters.  All input arrays
    must already be normalised to north-up (descending y) before calling this.
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


def _crop_patch(
    patch_geom,
    patch_crs: str,
    tile_paths: list[Path],
    output_path: Path,
    skip_existing: bool = False,
) -> bool:
    """Clip *tile_paths* to *patch_geom*, mosaic if needed, save as float32 .npy.

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
        da = _open_tile(path)
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

    arr = _numpy_mosaic(arrays) if len(arrays) > 1 else arrays[0].values.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return True


# ---------------------------------------------------------------------------
# Worker for parallel execution
# ---------------------------------------------------------------------------

def _process_patch(args: tuple) -> tuple[str, bool]:
    patch_id, geom_wkt, patch_crs, tile_paths, output_path, skip_existing = args
    from shapely import from_wkt
    patch_geom = from_wkt(geom_wkt)
    ok = _crop_patch(patch_geom, patch_crs, tile_paths, output_path, skip_existing)
    return patch_id, ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embedding crops for all So2Sat LCZ42 patches."
    )
    parser.add_argument(
        "--so2sat-dir",
        required=True,
        type=Path,
        help="Root directory of the So2Sat-LCZ42 v4 dataset "
             "(must contain patches_reference_rxr.gpkg).",
    )
    parser.add_argument(
        "--patches-file",
        type=Path,
        default=None,
        help="Optional GeoPackage with patches to process. "
             "Defaults to patches_reference_rxr.gpkg in --so2sat-dir. "
             "Use to restrict extraction to a single city, e.g. "
             "cities/London/patches_reference_London.gpkg.",
    )
    parser.add_argument(
        "--embedding-dir",
        required=True,
        type=Path,
        help="Directory containing embedding tile files (.zarr or .tif).",
    )
    parser.add_argument(
        "--embedding-name",
        required=True,
        choices=["tessera", "tesserav1.1", "alpha_earth", "alpha_earth_coop", "seamless"],
        help="Embedding type key (used to parse tile filenames).",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="Subfolder name for the output (e.g. 'GeoTessera' or 'AlphaEarth').",
    )
    parser.add_argument(
        "--year",
        required=True,
        help="Year subfolder for the output (e.g. '2017').",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation", "testing"],
        choices=["training", "validation", "testing"],
        help="Dataset splits to process (default: all three).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip patches whose .npy output file already exists (useful for resuming).",
    )
    args = parser.parse_args()

    # Ensure src/ is on sys.path when running directly
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # --- Load all patches once ---
    patches_path = args.patches_file if args.patches_file is not None else args.so2sat_dir / "patches_reference_rxr.gpkg"
    if not patches_path.exists():
        logger.error(f"Patches file not found at {patches_path}")
        sys.exit(1)

    logger.info(f"Loading patches from {patches_path} …")
    all_patches = gpd.read_file(patches_path)
    logger.info(f"  {len(all_patches)} patches total (crs: {all_patches.crs})")

    # --- Build tile spatial index (once for all splits) ---
    tile_paths, tree = _build_tile_index(args.embedding_dir, args.embedding_name, year=args.year)
    patch_crs = str(all_patches.crs)  # EPSG:4326

    # --- Process each split ---
    for split in args.splits:
        split_patches = all_patches[all_patches["dataset"] == split].reset_index(drop=True)
        if split_patches.empty:
            logger.warning(f"No patches found for split '{split}' — skipping.")
            continue

        out_dir = args.so2sat_dir / split / args.output_name / args.year
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"\n[{split}] {len(split_patches)} patches → {out_dir}"
        )

        n_saved = 0
        n_skipped = 0

        # Sort patches by centroid so tiles in the OS page cache are shared
        # across workers processing adjacent patches (critical for tesserav1.1
        # where each tile is ~110 MB and loaded fresh from npy each call).
        if args.embedding_name == "tesserav1.1":
            cx = split_patches.geometry.centroid.x
            cy = split_patches.geometry.centroid.y
            split_patches = split_patches.iloc[
                (cx + cy * 1000).argsort().values
            ].reset_index(drop=True)

        if args.workers > 1:
            # Build task list: only patches that intersect at least one tile
            tasks = []
            for row in split_patches.itertuples(index=False):
                patch_geom = row.geometry
                idxs = tree.query(patch_geom)
                if len(idxs) == 0:
                    n_skipped += 1
                    continue
                matched_paths = [tile_paths[i] for i in idxs]
                output_path = out_dir / f"patch_{row.patch_id}.npy"
                tasks.append((
                    row.patch_id,
                    patch_geom.wkt,
                    patch_crs,
                    matched_paths,
                    output_path,
                    args.skip_existing,
                ))

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_process_patch, t): t[0] for t in tasks}
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=split,
                    unit="patch",
                ):
                    _, ok = fut.result()
                    if ok:
                        n_saved += 1
                    else:
                        n_skipped += 1

        else:
            for row in tqdm(
                split_patches.itertuples(index=False),
                total=len(split_patches),
                desc=split,
                unit="patch",
            ):
                patch_geom = row.geometry
                idxs = tree.query(patch_geom)
                if len(idxs) == 0:
                    n_skipped += 1
                    continue

                matched_paths = [tile_paths[i] for i in idxs]
                output_path = out_dir / f"patch_{row.patch_id}.npy"
                ok = _crop_patch(patch_geom, patch_crs, matched_paths, output_path, args.skip_existing)
                if ok:
                    n_saved += 1
                else:
                    n_skipped += 1

        logger.info(
            f"[{split}] done — {n_saved} saved, {n_skipped} skipped (no coverage)"
        )

    logger.info("All splits complete.")


if __name__ == "__main__":
    main()
