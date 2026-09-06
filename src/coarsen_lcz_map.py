"""Coarsen and/or seam-repair an existing LCZ prediction GeoTIFF.

Post-processes an *already-produced* `infer_roi.py` prediction (no re-running
GPU inference) into a coarser, smoothed LCZ map — LCZ is a ~100 m urban-climate
concept, not a 10 m one — using the same majority-vote / Demuzere-et-al.-2020
Gaussian-likelihood filters `infer_roi.py --coarsen-to` uses for new runs
(`utils.lcz_smoothing`). Also exposes `--repair-seams` standalone, so tifs
produced before the infer_roi.py seam fix can be cleaned up without
regenerating them.

Usage:
    python src/coarsen_lcz_map.py \\
        --input .../seg-row4-coop-global-small/..._London.tif \\
        --output .../seg-row4-coop-global-small/..._London_100m_gaussian.tif \\
        --resolution 100 --method gaussian

    # Just repair seams, no resolution change:
    python src/coarsen_lcz_map.py \\
        --input .../..._London.tif --output .../..._London_repaired.tif \\
        --repair-seams
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).parent))

from utils.lcz_smoothing import gaussian_likelihood_filter, majority_pool, repair_seams
from utils.plot_lcz import save_lcz_map


def _geographic_extent(crs, transform, shape) -> tuple[float, float, float, float] | None:
    """Reproject the raster's own bounds to EPSG:4326 for PNG axis ticks."""
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds

    if crs is None:
        return None
    left, bottom, right, top = array_bounds(shape[0], shape[1], transform)
    return transform_bounds(crs, "EPSG:4326", left, bottom, right, top)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Existing LCZ prediction GeoTIFF (uint8, 1-17, 0=nodata).")
    p.add_argument("--output", required=True, type=Path, help="Output GeoTIFF path.")
    p.add_argument("--resolution", type=float, default=None,
                   help="Target resolution in metres (e.g. 100/200/300). Omit to "
                        "only --repair-seams without changing resolution.")
    p.add_argument("--method", choices=["gaussian", "majority"], default="gaussian",
                   help="'gaussian' (default): per-class Gaussian-likelihood filter "
                        "(Demuzere et al. 2020). 'majority': plain block-mode vote, "
                        "faster and blockier. Only used with --resolution.")
    p.add_argument("--gaussian-sigma", type=float, default=None,
                   help="Single sigma in metres overriding the per-class default "
                        "table for --method gaussian.")
    p.add_argument("--repair-seams", action="store_true",
                   help="Fill thin (<=2px) nodata seams — e.g. from independently "
                        "reprojected adjacent source tiles — before coarsening.")
    p.add_argument("--num-classes", type=int, default=17)
    args = p.parse_args()

    with rasterio.open(args.input) as src:
        raster = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = int(src.nodata) if src.nodata is not None else 0

    native_res_m = abs(transform.a)

    if args.repair_seams:
        raster, _ = repair_seams(raster)

    if args.resolution is not None:
        if args.resolution < native_res_m:
            raise SystemExit(
                f"--resolution ({args.resolution}) must be >= the input's native "
                f"resolution ({native_res_m:.4g}) — this coarsens, it does not upsample."
            )
        factor = max(1, round(args.resolution / native_res_m))
        if args.method == "majority":
            out_raster = majority_pool(raster, factor, nodata=nodata,
                                       num_classes=args.num_classes)
        else:
            kwargs = {"sigma_by_class": args.gaussian_sigma} if args.gaussian_sigma is not None else {}
            out_raster = gaussian_likelihood_filter(
                raster, native_res_m=native_res_m, out_res_m=args.resolution,
                nodata=nodata, num_classes=args.num_classes, **kwargs,
            )
        actual_res_m = native_res_m * factor
        out_transform = Affine(actual_res_m, 0.0, transform.c, 0.0, -actual_res_m, transform.f)
    else:
        out_raster = raster
        out_transform = transform

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(args.output), "w", driver="GTiff",
        height=out_raster.shape[0], width=out_raster.shape[1], count=1, dtype="uint8",
        crs=crs, transform=out_transform, nodata=nodata,
    ) as dst:
        dst.write(out_raster, 1)
    print(f"Saved: {args.output}")

    extent = _geographic_extent(crs, out_transform, out_raster.shape)
    png_path = args.output.with_suffix(".png")
    save_lcz_map(out_raster, args.output.stem, png_path, extent=extent)
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
