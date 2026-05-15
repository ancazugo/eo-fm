"""Plot the first three channels of tessera or alphaearth embeddings for a city ROI.

Saves:
  <output_dir>/<city>_<embedding>_rgb.png  — percentile-normalised RGB image
  <output_dir>/<city>_<embedding>_rgb.tif  — clipped mosaic (bands 0-2, raw values)

Usage:
    python src/plot_embeddings.py \
        --embedding tessera \
        --embedding-path /maps/acz25/phd-thesis-data/input/GeoTessera/2017/ \
        --city Paris \
        --bbox "2.09,48.72,2.58,49.02" \
        --output-dir /tmp/

    python src/plot_embeddings.py \
        --embedding alphaearth \
        --embedding-path /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017/ \
        --city Nairobi \
        --bbox "36.45,-1.54,37.17,-0.96" \
        --output-dir /tmp/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from pyproj import CRS, Transformer
from rioxarray.merge import merge_arrays


# ---------------------------------------------------------------------------
# Tile helpers (adapted from tessera_rgb_plot.ipynb)
# ---------------------------------------------------------------------------

def _load_tile_rgb(path: Path, bands: list[int]) -> xr.DataArray:
    """Load only the requested bands from a zarr or tif tile."""
    path = Path(path)
    if path.suffix == ".zarr":
        ds = xr.open_zarr(str(path), chunks=False)
        da = ds["embedding"]
        if da.dims != ("band", "y", "x"):
            da = da.transpose("band", "y", "x")
        if da.rio.crs is None and "spatial_ref" in ds:
            da = da.rio.write_crs(ds["spatial_ref"].attrs["crs_wkt"])
        return da.isel(band=bands)
    else:
        da = rioxarray.open_rasterio(str(path), chunks=None)
        da = da.assign_coords(band=np.arange(len(da.band)))
        return da.isel(band=bands)


def _tile_bounds_wgs84(path: Path) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS84 for a tile."""
    path = Path(path)
    if path.suffix == ".zarr":
        m = re.match(r"grid_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)_\d+\.zarr", path.name)
        if m:
            lon, lat = float(m.group("lon")), float(m.group("lat"))
            half = 0.05
            return lon - half, lat - half, lon + half, lat + half
    with rasterio.open(str(path)) as src:
        left, bottom, right, top = src.bounds
        src_crs = CRS.from_user_input(src.crs)
        wgs84 = CRS.from_epsg(4326)
        if src_crs != wgs84:
            t = Transformer.from_crs(src_crs, wgs84, always_xy=True)
            left, bottom = t.transform(left, bottom)
            right, top = t.transform(right, top)
        return left, bottom, right, top


def _overlaps(tile_bounds: tuple, query: tuple) -> bool:
    tminx, tminy, tmaxx, tmaxy = tile_bounds
    qminx, qminy, qmaxx, qmaxy = query
    return tmaxx > qminx and tminx < qmaxx and tmaxy > qminy and tminy < qmaxy


def _percentile_normalise(arr: np.ndarray, lo: int = 2, hi: int = 98) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        band = arr[i].astype(np.float32)
        vmin, vmax = np.nanpercentile(band, [lo, hi])
        out[i] = np.clip((band - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return out


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def plot_embedding_roi(
    embedding_path: str | Path,
    bbox: tuple[float, float, float, float],
    city: str,
    embedding: str,
    output_dir: str | Path,
    rgb_bands: list[int] | None = None,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """Load embedding tiles overlapping bbox, merge, clip, and save PNG + TIF.

    Args:
        embedding_path: Directory containing zarr or tif tiles.
        bbox: (west, south, east, north) in WGS84.
        city: City name used for output filenames.
        embedding: Embedding name used for output filenames (e.g. "tessera").
        output_dir: Directory to write outputs.
        rgb_bands: 0-indexed band indices for R, G, B (default: [0, 1, 2]).
        dpi: PNG resolution.

    Returns:
        (png_path, tif_path)
    """
    if rgb_bands is None:
        rgb_bands = [0, 1, 2]

    embedding_path = Path(embedding_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{city}_{embedding}_rgb"
    png_path = output_dir / f"{stem}.png"
    tif_path = output_dir / f"{stem}.tif"

    # Discover tiles
    candidates = sorted(
        list(embedding_path.glob("*.zarr"))
        + list(embedding_path.glob("*.tif"))
        + list(embedding_path.glob("*.tiff"))
    )
    if not candidates:
        raise FileNotFoundError(f"No zarr/tif tiles found in {embedding_path}")

    # Filter to bbox
    tiles = []
    for p in candidates:
        try:
            if _overlaps(_tile_bounds_wgs84(p), bbox):
                tiles.append(p)
        except Exception as e:
            print(f"  Skipping {p.name}: {e}", file=sys.stderr)

    if not tiles:
        raise ValueError(f"No tiles overlap bbox {bbox} in {embedding_path}")

    print(f"Found {len(tiles)} tiles for {city}")

    # Load RGB bands from each tile
    rgb_das = []
    for i, tile in enumerate(tiles):
        da = _load_tile_rgb(tile, rgb_bands)
        rgb_das.append(da)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(tiles)} tiles loaded")

    # Merge into mosaic
    merged = merge_arrays(rgb_das, nodata=np.nan)
    print(f"Merged shape: {merged.shape}  (bands, H, W)")

    # Clip to bbox
    west, south, east, north = bbox
    clipped = merged.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north, crs="EPSG:4326")
    print(f"Clipped shape: {clipped.shape}")

    # Save TIF (raw values, float32)
    clipped.astype(np.float32).rio.to_raster(str(tif_path), driver="GTiff")
    print(f"Saved TIF: {tif_path}")

    # Normalise for PNG
    mosaic = clipped.values.astype(np.float32)
    mosaic_norm = _percentile_normalise(mosaic)
    mosaic_norm = np.nan_to_num(mosaic_norm, nan=0.0)
    rgb_hwc = np.moveaxis(mosaic_norm, 0, -1)

    # Save PNG
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_hwc, interpolation="nearest")
    ax.set_title(f"{city} — {embedding} — bands {rgb_bands} as RGB", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG: {png_path}")

    return png_path, tif_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot first 3 channels of tessera/alphaearth embeddings for a city ROI."
    )
    parser.add_argument("--embedding", required=True, help="Embedding name (e.g. tessera, alphaearth).")
    parser.add_argument("--embedding-path", required=True, help="Directory containing tile files.")
    parser.add_argument("--city", required=True, help="City name (used for output filenames).")
    parser.add_argument(
        "--bbox",
        required=True,
        help="Bounding box as 'west,south,east,north' in WGS84.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for PNG and TIF.")
    parser.add_argument(
        "--rgb-bands",
        default="0,1,2",
        help="Comma-separated 0-indexed band indices for R,G,B (default: 0,1,2).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    west, south, east, north = [float(x) for x in args.bbox.split(",")]
    rgb_bands = [int(b) for b in args.rgb_bands.split(",")]
    plot_embedding_roi(
        embedding_path=args.embedding_path,
        bbox=(west, south, east, north),
        city=args.city,
        embedding=args.embedding,
        output_dir=args.output_dir,
        rgb_bands=rgb_bands,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
