"""Which output resolution and which pooling rule should an LCZ map use?

`infer_roi.py` can pool its per-pixel probability volume onto an output cell
three ways (`--aggregate soft|majority|gaussian`) at any `--target-res`. Those
are not interchangeable and the difference is invisible by eye: every
combination produces a plausible-looking LCZ map. This scores them.

The model runs ONCE per city. `infer_roi.TileProbSource` yields the per-tile
probability volume, every (aggregate x resolution) combination accumulates from
that same volume, and each finished map is scored against So2Sat patch labels
rasterised onto that map's own grid. So the sweep costs one GPU pass, and every
row differs ONLY in the pooling — the predictions behind them are identical.

Two things the numbers do NOT say:

  * The comparison is fair between rows, but a row's absolute value is only as
    honest as the checkpoint. A per-city checkpoint has seen the city it is
    scored on; pass `--splits val test` to at least restrict scoring to the
    patches its own grid split held out, and read the result as "which pooling
    is better", not "how good is this model".
  * Scoring a 100 m map against 320 m patch labels gives every 100 m cell its
    patch's class. That is the honest thing to do with the labels that exist --
    So2Sat has no finer truth -- but it structurally cannot reward a pooling
    rule for resolving sub-patch detail correctly.

    python src/coarsen_bakeoff.py \\
        --checkpoint <run_dir>/resnet_small_..._global-best.pt \\
        --family resnet --preset small \\
        --embedding-name tesserav1.1_global --embedding-dir /tessera/v1.1 \\
        --year 2017 --city Nairobi \\
        --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities \\
        --target-res 320 100 --output-dir data/coarsen_bakeoff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from infer_roi import (
    AGGREGATE_METHODS,
    REPROJECT_BAND_CHUNK,
    TileProbSource,
    build_aggregate_volume,
    load_model_and_normalize,
)
from datasets.tiles import setup_output
from utils.lcz_smoothing import repair_seams


def _parse_args() -> argparse.Namespace:
    from datasets.registry import available_embeddings
    from models import MODEL_REGISTRY

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--family", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--preset", default="small",
                   choices=["nano", "small", "base", "medium", "large"])
    p.add_argument("--arch", default=None)
    p.add_argument("--num-classes", type=int, default=17)
    p.add_argument("--embedding-name", required=True, choices=available_embeddings())
    p.add_argument("--embedding-dir", required=True, type=Path)
    p.add_argument("--year", default=None)
    p.add_argument("--city", required=True, help="City directory name under --cities-dir.")
    p.add_argument("--cities-dir", required=True, type=Path,
                   help="Root with one subfolder per city ({city}_grid.gpkg + "
                        "patches_reference_{city}.gpkg).")
    p.add_argument("--bbox", default=None,
                   help="ROI 'west,south,east,north' in EPSG:4326, replacing the "
                        "lookup in {city}_grid.gpkg. Use when that file is not "
                        "reachable (a full or read-only labels mount).")
    p.add_argument("--gt-gpkg", type=Path, default=None,
                   help="Ground-truth GeoPackage, replacing the one under "
                        "--cities-dir. Same reason as --bbox; must carry "
                        "--label-col (and a 'split' column if --splits is given).")
    p.add_argument("--splits", nargs="*", default=None,
                   help="Restrict scored GT patches to these splits from "
                        "patches_reference_{city}_split.gpkg (e.g. val test). "
                        "Omit to score every labelled patch.")
    p.add_argument("--label-col", default="LCZ_class")
    p.add_argument("--aggregate", nargs="+", default=list(AGGREGATE_METHODS),
                   choices=list(AGGREGATE_METHODS))
    p.add_argument("--target-res", nargs="+", type=float, default=[320.0, 100.0],
                   help="Output resolutions in metres to score (default: 320 100).")
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--patch-physical-res", type=float, default=320.0)
    p.add_argument("--patch-physical-stride", type=float, default=None,
                   help="Classification stride in metres. Default: the FINEST "
                        "--target-res, so every row is pooled from one identical "
                        "set of predictions.")
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--margin-m", type=float, default=200.0)
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--gaussian-sigma", type=float, default=None)
    p.add_argument("--normalize", choices=["auto", "none", "channel"], default="auto")
    p.add_argument("--dequantize", action="store_true")
    p.add_argument("--accelerator", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--save-maps", action="store_true",
                   help="Also write each combination's GeoTIFF + PNG.")
    return p.parse_args()


def _rasterize_gt(gdf, label_col: str, transform, shape) -> np.ndarray:
    """Burn patch labels onto an output grid (uint8, 0 = unlabelled)."""
    from rasterio.features import rasterize

    if gdf.empty:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize(
        [(g, int(c)) for g, c in zip(gdf.geometry, gdf[label_col])],
        out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
    )


def _score(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    """Confusion-matrix metrics over pixels both maps call labelled."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

    from training.evaluate import _lcz_suite

    both = (gt > 0) & (pred > 0)
    n = int(both.sum())
    if n == 0:
        return {"n_pixels": 0}
    y_true = gt[both].astype(int) - 1
    y_pred = pred[both].astype(int) - 1
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_acc = np.divide(
        np.diag(cm), cm.sum(axis=1),
        out=np.zeros(num_classes, dtype=float), where=cm.sum(axis=1) > 0,
    )
    out = {
        "n_pixels": n,
        "oa": float((y_true == y_pred).mean()),
        "macro_acc": float(per_class_acc[cm.sum(axis=1) > 0].mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
    }
    out.update({k.replace("test_", ""): v
                for k, v in _lcz_suite(cm, num_classes).items()})
    return out


def main() -> None:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio.warp import Resampling, reproject

    from utils.cli import resolve_overlap
    from utils.runtime import resolve_dequantize, resolve_device

    args = _parse_args()
    resolve_overlap(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    city_dir = args.cities_dir / args.city

    if args.bbox is not None:
        parts = [float(v) for v in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be 'west,south,east,north'")
        bbox = tuple(parts)
    else:
        grid_gpkg = city_dir / f"{args.city}_grid.gpkg"
        if not grid_gpkg.exists():
            raise SystemExit(f"Missing {grid_gpkg} — pass --bbox instead.")
        west, south, east, north = gpd.read_file(grid_gpkg).to_crs("EPSG:4326").total_bounds
        bbox = (float(west), float(south), float(east), float(north))
    logger.info(f"{args.city}: bbox {bbox}")

    # --splits reads the split file (it is the one carrying the split column);
    # otherwise the plain reference file, which has every labelled patch.
    if args.gt_gpkg is not None:
        gt_path = args.gt_gpkg
    elif args.splits:
        gt_path = city_dir / f"patches_reference_{args.city}_split.gpkg"
    else:
        gt_path = city_dir / f"patches_reference_{args.city}.gpkg"
    if not gt_path.exists():
        raise SystemExit(f"Missing {gt_path} — pass --gt-gpkg instead.")

    gt_gdf = gpd.read_file(gt_path)
    if args.splits:
        if "split" not in gt_gdf.columns:
            raise SystemExit(f"--splits given but {gt_path.name} has no 'split' column.")
        gt_gdf = gt_gdf[gt_gdf["split"].isin(args.splits)]
        logger.info(f"GT restricted to splits {args.splits}: {len(gt_gdf)} patches")
    logger.info(f"Ground truth: {len(gt_gdf)} labelled patches from {gt_path.name}")

    device = resolve_device(args.accelerator)
    model, normalize = load_model_and_normalize(
        args.checkpoint, args.family, args.embedding_name, device,
        preset=args.preset, arch=args.arch, num_classes=args.num_classes,
        patch_size=args.patch_size, normalize_mode=args.normalize,
    )
    dequantize_fn, _ = resolve_dequantize(args.embedding_name, force=args.dequantize)

    # Fix the prediction stride across the whole sweep, at the finest requested
    # resolution: rows must differ only in how the same predictions are pooled.
    stride_m = args.patch_physical_stride or min(args.target_res)
    source = TileProbSource(
        model, args.family, args.embedding_name, args.embedding_dir, bbox,
        year=args.year, num_classes=args.num_classes, patch_size=args.patch_size,
        overlap=args.overlap, batch_size=args.batch_size, device=device,
        dequantize_fn=dequantize_fn, normalize=normalize, margin_m=args.margin_m,
        patch_physical_res_m=args.patch_physical_res,
        patch_physical_stride_m=stride_m,   # read only for classification families
    )
    crs = source.first_crs
    native_res_m = source.embedding_res_m
    logger.info(
        f"{'segmentation' if source.is_seg else 'classification'} @ "
        f"{native_res_m:g} m native"
        + ("" if source.is_seg else f", stride {stride_m:g} m")
    )

    # One accumulator per (aggregate, resolution); all at output resolution,
    # so the whole sweep costs a few tens of MB.
    grids: dict[float, tuple] = {}
    for res_m in args.target_res:
        if res_m < native_res_m:
            logger.warning(f"--target-res {res_m} is finer than {native_res_m} m — skipped")
            continue
        # Output CRS is the tile's own UTM zone, so its units are metres.
        transform, H, W = setup_output(bbox, crs, res_m)
        grids[res_m] = (transform, H, W, res_m)
        logger.info(f"  grid @ {res_m:g} m: {H}×{W} px")

    accum = {
        (agg, res_m): [
            np.zeros((args.num_classes, g[1], g[2]), dtype=np.float32),
            np.zeros((g[1], g[2]), dtype=np.float32),
        ]
        for agg in args.aggregate
        for res_m, g in grids.items()
    }

    # Reprojected in class chunks against one reused buffer per grid — same
    # reason as infer_roi: a full second copy of the volume per tile per
    # combination would dominate memory for no benefit.
    chunk = max(1, min(args.num_classes, REPROJECT_BAND_CHUNK))
    buffers = {
        res_m: (np.zeros((chunk, g[1], g[2]), dtype=np.float32),
                np.zeros((1, g[1], g[2]), dtype=np.float32))
        for res_m, g in grids.items()
    }

    for probs, tile_crs, tile_transform in source:
        valid = (probs.sum(axis=0) > 0).astype(np.float32)
        for res_m, (transform, H, W, _) in grids.items():
            warp = dict(src_transform=tile_transform, src_crs=tile_crs,
                        dst_transform=transform, dst_crs=crs,
                        resampling=Resampling.sum)
            warp_buf, mask_buf = buffers[res_m]
            # The validity mask does not depend on the aggregate, so it is
            # warped once per grid rather than once per combination.
            mask_buf[:] = 0.0
            reproject(source=valid[None], destination=mask_buf, **warp)
            for agg in args.aggregate:
                accum[(agg, res_m)][1] += mask_buf[0]

        for agg in args.aggregate:
            vol = build_aggregate_volume(probs, agg, native_res_m, args.gaussian_sigma)
            for res_m, (transform, H, W, _) in grids.items():
                warp = dict(src_transform=tile_transform, src_crs=tile_crs,
                            dst_transform=transform, dst_crs=crs,
                            resampling=Resampling.sum)
                warp_buf, _ = buffers[res_m]
                for c0 in range(0, args.num_classes, chunk):
                    c1 = min(c0 + chunk, args.num_classes)
                    view = warp_buf[: c1 - c0]
                    view[:] = 0.0
                    reproject(source=vol[c0:c1], destination=view, **warp)
                    accum[(agg, res_m)][0][c0:c1] += view

    gt_by_res = {
        res_m: _rasterize_gt(gt_gdf.to_crs(crs), args.label_col, g[0], (g[1], g[2]))
        for res_m, g in grids.items()
    }

    rows = []
    for (agg, res_m), (prob_accum, weight_accum) in accum.items():
        transform, H, W, res_units = grids[res_m]
        cell_px = max(1.0, res_units / native_res_m)
        covered = weight_accum >= args.min_coverage * cell_px * cell_px
        pred = np.zeros((H, W), dtype=np.uint8)
        if covered.any():
            mean_probs = prob_accum[:, covered] / weight_accum[None, covered]
            pred[covered] = (mean_probs.argmax(axis=0) + 1).astype(np.uint8)
        pred, _ = repair_seams(pred)

        row = {
            "city": args.city,
            "pipeline": "segmentation" if source.is_seg else "classification",
            "family": args.family,
            "preset": args.preset,
            "embedding": args.embedding_name,
            "aggregate": agg,
            "res_m": res_m,
            "stride_m": None if source.is_seg else stride_m,
            "splits": ",".join(args.splits) if args.splits else "all",
            "nodata_frac": float((pred == 0).mean()),
        }
        row.update(_score(pred, gt_by_res[res_m], args.num_classes))
        rows.append(row)
        logger.info(
            f"{agg:>8} @ {res_m:>5.0f} m — kappa {row.get('kappa', float('nan')):.4f}  "
            f"OA {row.get('oa', float('nan')):.4f}  "
            f"macroF1 {row.get('macro_f1', float('nan')):.4f}  "
            f"nodata {row['nodata_frac']:.3%}  n={row.get('n_pixels', 0)}"
        )

        if args.save_maps:
            from utils.plot_lcz import save_lcz_map

            stem = f"{args.city}_{args.family}_{agg}_{int(round(res_m))}m"
            tif = args.output_dir / f"{stem}.tif"
            with rasterio.open(
                str(tif), "w", driver="GTiff", height=H, width=W, count=1,
                dtype="uint8", crs=crs, transform=transform, nodata=0,
            ) as dst_ds:
                dst_ds.write(pred, 1)
            save_lcz_map(pred, f"{args.city} — {agg} @ {int(round(res_m))}m",
                         tif.with_suffix(".png"), extent=bbox)

    csv_path = args.output_dir / "coarsen_bakeoff.csv"
    df = pd.DataFrame(rows).sort_values(["res_m", "kappa"], ascending=[True, False])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)
    logger.info(f"Wrote {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
