"""Crop a GeoTIFF around a coordinate and save as PNG."""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

_src = Path(__file__).parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from utils.plot_lcz import lcz_colormap


def tif_to_png(
    tif_path: str | Path,
    lon: float,
    lat: float,
    output_path: str | Path | None = None,
    size: tuple[int, int] = (64, 64),
    bands: list[int] | None = None,
    lcz: bool = False,
    percentile_norm: bool = False,
) -> Path:
    """Crop a GeoTIFF around (lon, lat) to `size` pixels and save as PNG.

    If the crop window extends beyond the raster extent it is clamped, then
    the result is resized to `size` with nearest-neighbour resampling.

    Args:
        tif_path: Path to input GeoTIFF.
        lon: Longitude of the centre point (WGS84 / EPSG:4326).
        lat: Latitude of the centre point (WGS84 / EPSG:4326).
        output_path: Destination PNG path. Defaults to <tif_stem>_<lon>_<lat>.png
                     next to the input file.
        size: (width, height) in pixels for the output PNG.
        bands: 1-indexed list of bands to include (max 3 for RGB, or 1 for greyscale).
               Ignored when lcz=True.
        lcz: Apply the LCZ discrete colormap. Expects band 1 to contain values 0–17.
        percentile_norm: Compute 2–98 percentile bounds over the full raster (8× downsampled)
                         and apply them to the crop. Matches the colormap of plot_embeddings.py.
                         Ignored when lcz=True.

    Returns:
        Path to the written PNG file.
    """
    tif_path = Path(tif_path)
    w, h = size

    with rasterio.open(tif_path) as src:
        # Reproject centre coordinate into raster CRS
        src_crs = src.crs
        wgs84 = CRS.from_epsg(4326)
        if src_crs != wgs84:
            xs, ys = warp_transform(wgs84, src_crs, [lon], [lat])
            cx, cy = xs[0], ys[0]
        else:
            cx, cy = lon, lat

        # Convert geographic centre to pixel row/col
        row_c, col_c = rowcol(src.transform, cx, cy)

        # Compute crop window (may exceed raster bounds — clamped below)
        row_off = row_c - h // 2
        col_off = col_c - w // 2

        # Clamp to valid pixel space
        row_off_c = max(0, min(row_off, src.height - h))
        col_off_c = max(0, min(col_off, src.width - w))
        read_h = min(h, src.height - row_off_c)
        read_w = min(w, src.width - col_off_c)

        # Choose bands
        total_bands = src.count
        if lcz:
            bands = [1]
        elif bands is None:
            bands = list(range(1, min(total_bands, 3) + 1))
        else:
            bands = [b for b in bands if 1 <= b <= total_bands]
        if not bands:
            raise ValueError(f"No valid bands selected from file with {total_bands} bands.")

        # Compute percentile bounds from a downsampled full read before cropping
        if percentile_norm and not lcz:
            ds_h = max(1, src.height // 8)
            ds_w = max(1, src.width // 8)
            full = src.read(bands, out_shape=(len(bands), ds_h, ds_w)).astype(np.float32)
            pct_bounds = [
                (float(np.nanpercentile(full[i], 2)), float(np.nanpercentile(full[i], 98)))
                for i in range(full.shape[0])
            ]
        else:
            pct_bounds = None

        window = rasterio.windows.Window(col_off_c, row_off_c, read_w, read_h)
        data = src.read(bands, window=window)  # (C, H, W)

    if lcz:
        cmap, norm = lcz_colormap()
        rgba = cmap(norm(data[0]))  # (H, W, 4) float64 in [0, 1]
        img = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8), mode="RGB")
    else:
        def normalise(arr: np.ndarray, bounds: tuple[float, float] | None = None) -> np.ndarray:
            lo, hi = bounds if bounds else (float(arr.min()), float(arr.max()))
            if hi == lo:
                return np.zeros_like(arr, dtype=np.uint8)
            return np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        channels = [
            normalise(data[i], pct_bounds[i] if pct_bounds else None)
            for i in range(data.shape[0])
        ]

        if len(channels) == 1:
            img = Image.fromarray(channels[0], mode="L")
        elif len(channels) == 2:
            # Pad to RGB with a zero blue channel
            zero = np.zeros_like(channels[0])
            img = Image.fromarray(np.stack([channels[0], channels[1], zero], axis=-1), mode="RGB")
        else:
            img = Image.fromarray(np.stack(channels[:3], axis=-1), mode="RGB")

    # Resize to exact target size if crop was clamped at raster edges
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)

    # Determine output path
    if output_path is None:
        stem = tif_path.stem
        output_path = tif_path.parent / f"{stem}_{lon}_{lat}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a GeoTIFF around a coordinate and export as PNG."
    )
    parser.add_argument("tif", help="Path to input GeoTIFF.")
    parser.add_argument("lon", type=float, help="Longitude of centre point (WGS84).")
    parser.add_argument("lat", type=float, help="Latitude of centre point (WGS84).")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path.")
    parser.add_argument(
        "--size",
        default="64x64",
        help="Output size as WxH pixels, e.g. 128x128 (default: 64x64).",
    )
    parser.add_argument(
        "--bands",
        default=None,
        help="Comma-separated 1-indexed band list, e.g. 1,2,3 (default: first 1-3 bands).",
    )
    parser.add_argument(
        "--lcz",
        action="store_true",
        help="Apply LCZ discrete colormap (band 1, values 0–17).",
    )
    parser.add_argument(
        "--percentile-norm",
        action="store_true",
        help="Normalise using 2–98 percentile of the full raster (matches plot_embeddings.py output).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    w_str, h_str = args.size.lower().split("x")
    size = (int(w_str), int(h_str))

    bands = [int(b) for b in args.bands.split(",")] if args.bands else None

    out = tif_to_png(
        tif_path=args.tif,
        lon=args.lon,
        lat=args.lat,
        output_path=args.output,
        size=size,
        bands=bands,
        lcz=args.lcz,
        percentile_norm=args.percentile_norm,
    )
    print(out)


if __name__ == "__main__":
    main()
