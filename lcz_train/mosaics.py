"""T2 (mosaic step) — cache one embedding mosaic per AOI, aligned to its label grid.

Decision: cache a per-AOI mosaic rather than read tiles on-the-fly in
``__getitem__``. The label raster grid (``lcz_labels.export.raster_grid``) is
canonical; every embedding tile overlapping the AOI is warped onto it once
via ``rasterio.warp.reproject`` (nearest — embedding vectors must never be
blended) and written to a float16 memmap + JSON sidecar. This is what
resolves AOIs straddling UTM zones (every tile reprojects into the AOI's own
UTM), and turns per-window sampling into cheap memmap slicing instead of a
per-window multi-tile CRS-aware read.

Never call ``clipped.rio.transform()`` after a north-up flip — it silently
returns a south-up affine (see ``src/infer_roi.py::_open_and_clip``). The
affine is always rebuilt from the (already north-up) coordinate arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

from lcz_labels.export import raster_grid


def _tile_to_northup_array(da) -> tuple[np.ndarray, Affine, str]:
    """(array, transform, crs) for one opened tile, north-up, manual affine."""
    if da.sizes.get("y", 0) > 1 and float(da.y.values[0]) < float(da.y.values[-1]):
        da = da.isel(y=slice(None, None, -1))
    x_vals, y_vals = da.x.values, da.y.values
    res_x = float(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else float(abs(da.rio.resolution()[0]))
    res_y = float(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else -float(abs(da.rio.resolution()[1]))
    transform = Affine(res_x, 0.0, float(x_vals[0]) - res_x / 2,
                       0.0, res_y, float(y_vals[0]) - res_y / 2)
    return np.asarray(da.values, dtype=np.float32), transform, da.rio.crs.to_string()


def warp_tile_onto_grid(
    arr: np.ndarray,
    src_transform: Affine,
    src_crs: str,
    dst_transform: Affine,
    dst_crs: str,
    dst_shape: tuple[int, int],
    *,
    dst: np.ndarray | None = None,
) -> np.ndarray:
    """Reproject one (C, H, W) tile array onto the destination grid (nearest).

    ``rasterio.warp.reproject`` recomputes EVERY pixel of ``destination`` from
    the current source alone — pixels outside the source's footprint are
    overwritten with ``dst_nodata``, not left untouched. So accumulating tiles
    by passing the same array as ``destination`` across calls silently erases
    every earlier tile outside the current one's footprint (only the last
    tile processed would ever survive). Instead we always warp into a fresh
    per-tile buffer and merge only its covered pixels into ``dst`` via
    ``np.copyto(..., where=...)`` — the same last-writer-wins merge pattern
    ``infer_roi.py`` uses for its own multi-tile mosaic.

    Writes the merge into ``dst`` in place when given, else allocates a fresh
    ``(C, *dst_shape)`` array. Zero (never covered) marks no-coverage —
    embeddings are never exactly all-zero in practice, but the dataset layer
    must additionally check the label rasters' own validity.
    """
    c = arr.shape[0]
    out = dst if dst is not None else np.zeros((c, *dst_shape), dtype=np.float32)
    tmp = np.zeros((c, *dst_shape), dtype=np.float32)
    reproject(
        source=arr, destination=tmp,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.nearest, src_nodata=None, dst_nodata=0.0,
    )
    covered = np.any(tmp != 0, axis=0)   # matches datasets.pool_blocks_mean's convention
    for band in range(c):
        np.copyto(out[band], tmp[band], where=covered)
    return out


def build_mosaic(
    aoi_name: str,
    year: int | str,
    embedding_name: str,
    embedding_dir: Path,
    lcz_config,
    mosaic_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Warp every tile covering the AOI onto its canonical label grid.

    Returns the path to the cached ``{aoi}_{year}.npy`` (float16, memmap-
    friendly, shape ``(C, H, W)``) with a ``.json`` sidecar carrying the
    transform/crs/shape/channel-count so consumers never re-derive them.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from datasets.tiles import build_tile_index, open_tile  # noqa: E402

    out_dir = Path(mosaic_dir) / embedding_name
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{aoi_name}_{year}.npy"
    json_path = out_dir / f"{aoi_name}_{year}.json"
    if npy_path.exists() and json_path.exists() and not force:
        logger.info(f"[{aoi_name}] mosaic cache hit: {npy_path.name}")
        return npy_path

    transform, utm_crs, (h, w) = raster_grid(aoi_name, lcz_config)
    from lcz_labels.grid import resolve_aoi_bbox

    bbox = resolve_aoi_bbox(lcz_config.aoi(aoi_name), lcz_config)
    tile_paths, tree = build_tile_index(Path(embedding_dir), embedding_name, str(year))
    hits = tree.query(box(*bbox))
    if len(hits) == 0:
        raise FileNotFoundError(f"[{aoi_name}] no {embedding_name} tiles cover {bbox}")

    mosaic = None
    n_channels = None
    for i in hits:
        try:
            da = open_tile(tile_paths[i])
        except Exception as e:  # noqa: BLE001 — one bad tile must not abort the AOI
            logger.warning(f"[{aoi_name}] failed to open tile {tile_paths[i]}: {e}")
            continue
        arr, src_transform, src_crs = _tile_to_northup_array(da)
        if n_channels is None:
            n_channels = arr.shape[0]
            mosaic = np.zeros((n_channels, h, w), dtype=np.float32)
        warp_tile_onto_grid(arr, src_transform, src_crs, transform, utm_crs, (h, w), dst=mosaic)

    if mosaic is None:
        raise FileNotFoundError(f"[{aoi_name}] all {len(hits)} candidate tiles failed to open")

    mosaic16 = mosaic.astype(np.float16)
    memmapped = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float16,
                                          shape=mosaic16.shape)
    memmapped[:] = mosaic16
    memmapped.flush()
    json_path.write_text(json.dumps({
        "aoi": aoi_name, "year": str(year), "embedding_name": embedding_name,
        "shape": list(mosaic16.shape), "crs": utm_crs,
        "transform": list(transform)[:6],
    }))
    cov = float((mosaic != 0).any(axis=0).mean())
    logger.info(f"[{aoi_name}] mosaic {mosaic16.shape} ({embedding_name}, {year}) "
                f"coverage={cov:.0%} -> {npy_path}")
    return npy_path


def load_mosaic(npy_path: Path) -> tuple[np.ndarray, dict]:
    """Open a cached mosaic as a read-only memmap + its sidecar metadata."""
    meta = json.loads(Path(npy_path).with_suffix(".json").read_text())
    arr = np.lib.format.open_memmap(npy_path, mode="r")
    return arr, meta
