"""Checkerboard geographic train/val/test split for raster label datasets."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from shapely.ops import unary_union

from loguru import logger


def _get_label_paths(label_path: str | Path) -> list[Path]:
    """Return list of GeoTIFF paths from a directory or single file."""
    p = Path(label_path)
    if p.is_dir():
        paths = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
        if not paths:
            raise FileNotFoundError(f"No GeoTIFF files found in {p}")
        return paths
    return [p]


def _make_grid_cells(roi, tile_size: float) -> list:
    """Divide roi bounds into a grid of tile_size × tile_size cells.

    Each cell is clipped to the roi shape. Cells with zero area are dropped.
    Returns a list of Shapely geometries.
    """
    minx, miny, maxx, maxy = roi.bounds
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + tile_size, y + tile_size)
            clipped = cell.intersection(roi)
            if not clipped.is_empty and clipped.area > 0:
                cells.append(clipped)
            y += tile_size
        x += tile_size
    return cells


def _count_valid_pixels(
    cell,
    label_paths: list[Path],
    nodata: int = 0,
    target_crs=None,
) -> tuple[int, int]:
    """Count valid (non-nodata) pixels in the cell across all label tif files.

    Args:
        cell: Shapely geometry of the grid cell.
        label_paths: Label GeoTIFF files to query.
        nodata: Nodata pixel value (default 0).
        target_crs: When provided, each label file is virtually reprojected to
            this CRS via WarpedVRT before querying. Cell bounds are then in the
            same CRS as the raster. Typically set to the embedding dataset's CRS
            so that label files in EPSG:4326 are transparently converted.

    Returns (valid_count, total_count). Files that don't overlap the cell are skipped.
    """
    valid = 0
    total = 0
    minx, miny, maxx, maxy = cell.bounds
    full_window = None
    for path in label_paths:
        with rasterio.open(path) as src:
            ds = WarpedVRT(src, crs=target_crs) if target_crs is not None else src
            with ds:
                if full_window is None:
                    full_window = Window(0, 0, ds.width, ds.height)
                fb = ds.bounds
                if maxx <= fb.left or minx >= fb.right or maxy <= fb.bottom or miny >= fb.top:
                    continue
                win = from_bounds(
                    max(minx, fb.left),
                    max(miny, fb.bottom),
                    min(maxx, fb.right),
                    min(maxy, fb.top),
                    ds.transform,
                )
                # Reprojection can produce fractional/edge-touching windows.
                # Clamp to valid pixel space before reading.
                win = win.round_offsets().round_lengths()
                try:
                    win = win.intersection(full_window)
                except Exception:
                    continue
                if win.width <= 0 or win.height <= 0:
                    continue
                try:
                    data = ds.read(1, window=win, boundless=False)
                except RasterioError:
                    continue
                total += data.size
                valid += int(np.sum(data != nodata))
    return valid, total


def get_raster_roi(path: str | Path, target_crs=None):
    """Return the bounding box of a label GeoTIFF as a Shapely Polygon.

    Args:
        path: Path to a GeoTIFF file or directory of GeoTIFFs (first file used).
        target_crs: When provided, the raster is virtually reprojected to this
            CRS via WarpedVRT before reading bounds. Use the embedding CRS so
            the returned ROI is directly usable as a sampler ROI.

    Returns:
        Shapely box in target_crs (or the raster's native CRS when None).
    """
    paths = _get_label_paths(path)
    with rasterio.open(paths[0]) as src:
        ds = WarpedVRT(src, crs=target_crs) if target_crs is not None else src
        with ds:
            b = ds.bounds
    return box(b.left, b.bottom, b.right, b.top)


def checkerboard_roi_split(
    roi,
    tile_size: float,
    label_path: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_valid_frac: float = 0.10,
    nodata: int = 0,
    seed: int = 42,
    roi_crs=None,
) -> tuple:
    """Split a ROI into train/val/test using a checkerboard grid.

    Args:
        roi: Shapely Polygon or MultiPolygon in the embedding CRS.
        tile_size: Size of each checkerboard tile in CRS units (e.g. metres for UTM).
        label_path: Directory of label GeoTIFFs or single GeoTIFF file.
            Used to filter tiles that lack sufficient labeled pixels.
        train_frac: Fraction of valid tiles assigned to training.
        val_frac: Fraction of valid tiles assigned to validation.
        test_frac: Fraction of valid tiles assigned to test.
        min_valid_frac: Minimum fraction of non-nodata pixels required to keep a tile.
            Tiles below this threshold are excluded from all splits.
        nodata: Nodata value in the label raster (default 0 for LCZ rasters).
        seed: Random seed for tile shuffling.
        roi_crs: CRS of the roi (typically the embedding CRS). Label rasters are
            virtually reprojected to this CRS via WarpedVRT so that cell bounds
            and raster bounds are in the same coordinate system.

    Returns:
        (train_roi, val_roi, test_roi) — each is a Shapely Polygon/MultiPolygon
        in the same CRS as roi, or None if no tiles were assigned to that split.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, "
            f"got {train_frac + val_frac + test_frac:.4f}"
        )

    label_paths = _get_label_paths(label_path)
    logger.info(f"Checkerboard split: tile_size={tile_size}, {len(label_paths)} label file(s)")

    all_cells = _make_grid_cells(roi, tile_size)
    logger.info(f"Grid cells generated: {len(all_cells)}")

    # Filter by valid pixel fraction
    valid_cells = []
    n_dropped = 0
    for cell in all_cells:
        valid, total = _count_valid_pixels(cell, label_paths, nodata=nodata, target_crs=roi_crs)
        if total == 0:
            n_dropped += 1
            continue
        frac = valid / total
        if frac >= min_valid_frac:
            valid_cells.append(cell)
        else:
            n_dropped += 1
            logger.debug(f"  Dropping tile (valid_frac={frac:.3f} < {min_valid_frac})")

    logger.info(
        f"Tiles after valid-pixel filter: {len(valid_cells)} kept, {n_dropped} dropped "
        f"(min_valid_frac={min_valid_frac})"
    )

    if not valid_cells:
        raise RuntimeError(
            "No checkerboard tiles passed the valid-pixel filter. "
            "Lower --min-valid-frac or increase --checkerboard-tile-size."
        )

    # Shuffle and assign to splits
    rng = random.Random(seed)
    rng.shuffle(valid_cells)
    n = len(valid_cells)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    # Test gets the remainder to avoid rounding gaps
    train_cells = valid_cells[:n_train]
    val_cells = valid_cells[n_train : n_train + n_val]
    test_cells = valid_cells[n_train + n_val :]

    logger.info(
        f"Split assignment: train={len(train_cells)}, val={len(val_cells)}, test={len(test_cells)} tiles"
    )

    def _to_roi(cells):
        if not cells:
            return None
        return unary_union(cells)

    return _to_roi(train_cells), _to_roi(val_cells), _to_roi(test_cells)
