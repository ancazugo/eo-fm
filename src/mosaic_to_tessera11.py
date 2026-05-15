"""Convert a float32 tesserav1.1 mosaic (or pre-split tiles) to geoinfo + infer_output format.

The pipeline (_open_tile_tessera11 in extract_so2sat_embeddings.py) reads:
  <year_dir>/geoinfo/grid_{lon}_{lat}.tiff        -- 1-band uint8 spatial reference
  <year_dir>/infer_output/{prefix}_grid_{lon}_{lat}_all_data_emb128_int8.npy   (H,W,128) int8
  <year_dir>/infer_output/{prefix}_grid_{lon}_{lat}_all_data_emb128_scales.npy (H,W,1)   float32

Usage (Mode A — pre-split tiles, preferred):
  python src/mosaic_to_tessera11.py \
    --src-tiles .../GeoTessera/v1.1/2017/ \
    --output-dir .../GeoTessera/v1.1/2017_split/ \
    --prefix nairobi_mosaic

Usage (Mode B — large mosaic TIFF):
  python src/mosaic_to_tessera11.py \
    --src-mosaic .../nairobi_mosaic_emb128.tif \
    --output-dir .../GeoTessera/v1.1/2017_split/ \
    --prefix nairobi_mosaic
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows


def _quantize(data_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert dequantization: (128,H,W) float32 → int8 (H,W,128) + scales (H,W,1) float32.

    Matches the format expected by load_and_dequantize_tessera_representation():
      float32[h,w,c] = int8[h,w,c] * scales[h,w]
    Per-pixel scale = max(|embedding|, axis=channels) / 127 so the loudest channel
    uses the full int8 range.
    """
    data_hwc = data_chw.transpose(1, 2, 0).astype(np.float32)  # (H, W, 128)
    max_abs = np.abs(data_hwc).max(axis=2, keepdims=True)       # (H, W, 1)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    int8_hwc = np.round(data_hwc / scales).clip(-127, 127).astype(np.int8)
    return int8_hwc, scales


def _write_geoinfo(path: Path, crs, transform, width: int, height: int) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", count=1, dtype="uint8",
        crs=crs, transform=transform, width=width, height=height,
    ) as dst:
        dst.write(np.zeros((1, height, width), dtype=np.uint8))


def _save_tile(
    data_chw: np.ndarray,
    crs,
    transform,
    tile_name: str,
    geoinfo_dir: Path,
    infer_dir: Path,
    prefix: str,
) -> None:
    _, H, W = data_chw.shape
    m = re.match(r"grid_([-\d.]+)_([-\d.]+)$", tile_name)
    if m is None:
        raise ValueError(f"Cannot parse lon/lat from tile name: {tile_name!r}")
    lon_str, lat_str = m.group(1), m.group(2)

    _write_geoinfo(geoinfo_dir / f"{tile_name}.tiff", crs, transform, W, H)

    int8_hwc, scales = _quantize(data_chw)
    base = f"{prefix}_grid_{lon_str}_{lat_str}_all_data_emb128"
    np.save(infer_dir / f"{base}_int8.npy", int8_hwc)
    np.save(infer_dir / f"{base}_scales.npy", scales)
    print(f"  {tile_name}  ({H}×{W})")


def process_tiles_dir(src_dir: Path, output_dir: Path, prefix: str) -> None:
    """Mode A: process pre-split float32 tile TIFFs named grid_{lon}_{lat}_{year}.tif."""
    pattern = re.compile(r"(grid_[-\d.]+_[-\d.]+)_\d+\.tif$")
    tifs = sorted(src_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in {src_dir}")

    geoinfo_dir = output_dir / "geoinfo"
    infer_dir = output_dir / "infer_output"
    geoinfo_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    for tif in tifs:
        m = pattern.match(tif.name)
        if m is None:
            print(f"  skip {tif.name} (expected grid_LON_LAT_YEAR.tif)")
            continue
        tile_name = m.group(1)
        with rasterio.open(tif) as ds:
            _save_tile(ds.read(), ds.crs, ds.transform, tile_name, geoinfo_dir, infer_dir, prefix)


def process_mosaic(src: Path, output_dir: Path, prefix: str, min_pixels: int = 10) -> None:
    """Mode B: split mosaic into 0.1° tiles and process each."""
    GRID = 0.1
    geoinfo_dir = output_dir / "geoinfo"
    infer_dir = output_dir / "infer_output"
    geoinfo_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src) as ds:
        wl, wb, wr, wt = rasterio.warp.transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)

        lon = math.floor(wl / GRID) * GRID
        while lon < wr:
            lat = math.floor(wb / GRID) * GRID
            while lat < wt:
                lon_c = round(lon + GRID / 2, 2)
                lat_c = round(lat + GRID / 2, 2)
                tile_name = f"grid_{lon_c:.2f}_{lat_c:.2f}"

                cell_utm = rasterio.warp.transform_bounds(
                    "EPSG:4326", ds.crs, lon, lat, lon + GRID, lat + GRID
                )
                win = rasterio.windows.from_bounds(
                    *cell_utm, transform=ds.transform
                ).round_offsets().round_lengths()

                if win.width >= min_pixels and win.height >= min_pixels:
                    data = ds.read(window=win)
                    # rasterio clamps windows to dataset bounds; actual shape may be smaller
                    if data.shape[1] >= min_pixels and data.shape[2] >= min_pixels:
                        win_transform = rasterio.windows.transform(win, ds.transform)
                        _save_tile(data, ds.crs, win_transform, tile_name, geoinfo_dir, infer_dir, prefix)

                lat += GRID
            lon += GRID


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-tiles", type=Path,
                    help="Directory of pre-split float32 tile TIFFs (grid_LON_LAT_YEAR.tif)")
    ap.add_argument("--src-mosaic", type=Path,
                    help="Large float32 mosaic TIFF; used if --src-tiles is absent")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Output root; geoinfo/ and infer_output/ are created inside")
    ap.add_argument("--prefix", default="nairobi_mosaic",
                    help="Prefix for npy filenames (default: nairobi_mosaic)")
    ap.add_argument("--min-pixels", type=int, default=50,
                    help="Skip cells with fewer than this many pixels in either dimension (default: 50)")
    args = ap.parse_args()

    if args.src_tiles and args.src_tiles.exists():
        print(f"Mode A — pre-split tiles: {args.src_tiles}")
        process_tiles_dir(args.src_tiles, args.output_dir, args.prefix)
    elif args.src_mosaic and args.src_mosaic.exists():
        print(f"Mode B — mosaic: {args.src_mosaic}")
        process_mosaic(args.src_mosaic, args.output_dir, args.prefix, min_pixels=args.min_pixels)
    else:
        ap.error("Provide --src-tiles (directory) or --src-mosaic (TIFF).")

    print(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()
