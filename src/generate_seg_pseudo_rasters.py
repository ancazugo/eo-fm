"""Generate dense per-city pseudo-label rasters for segmentation distillation.

Runs a trained patch-classification teacher over each city with infer_roi's
soft-voting sliding window (overlapping 320 m patches, fine stride) and turns
the result into a dense label raster the segmentation pipeline can train on
(semantic_segmentation.py --label-source tif --label-tif-dir ...):

  1. infer_roi(..., save_confidence=True) → 10 m class + confidence rasters
  2. teacher class kept where confidence >= --min-conf, else 0 (nodata)
  3. val/test-split So2Sat patch footprints zeroed out (no transductive
     leakage into the pixels later used for evaluation)
  4. train-split So2Sat polygons burned in on top (GT overrides teacher)

Output convention matches the So2Sat raw rasters: uint8, 0 = nodata,
1-17 = LCZ class.

Outputs (to --output-dir):
  teacher_{city}.tif / teacher_{city}_conf.tif   raw teacher argmax + confidence
  pseudo_seg_{city}.tif (+ .png)                 final training label raster

Example:
    python src/generate_seg_pseudo_rasters.py \\
        --checkpoint <run_dir>/resnet_small_GeoTessera_v1.1_global_global-best.pt \\
        --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities --cities Nairobi \\
        --embedding-name tesserav1.1_global \\
        --embedding-dir /tessera/v1.1 --year 2017 \\
        --output-dir data/pseudo_seg_rasters
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from datasets.registry import EMBEDDING_REGISTRY
from infer_roi import infer_roi, load_model_and_normalize
from utils.runtime import resolve_dequantize, resolve_device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher soft-voted inference → dense pseudo-label rasters per city."
    )
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+",
                        help="Teacher checkpoint(s). Several = one ensemble teacher: "
                             "their softmax is averaged per patch, each model on its "
                             "own checkpoint-stored normalisation.")
    parser.add_argument("--family", default="resnet")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--cities-dir", required=True, type=Path,
                        help="Root directory with one subfolder per city "
                             "({city}_grid.gpkg + patches_reference_{city}_split.gpkg).")
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--embedding-name", required=True, choices=sorted(EMBEDDING_REGISTRY))
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--year", required=True)
    parser.add_argument("--label-col", default="LCZ_class")
    parser.add_argument("--patch-size", type=int, default=32,
                        help="Model input side in pixels (must match training).")
    parser.add_argument("--patch-physical-res", type=float, default=320.0,
                        help="Physical patch side in metres (default 320 = So2Sat).")
    parser.add_argument("--patch-physical-stride", type=float, default=80.0,
                        help="Stride between patch origins in metres — smaller = finer "
                             "soft voting (default 80 = 16 overlapping patches/pixel).")
    parser.add_argument("--out-res", type=float, default=10.0,
                        help="Output resolution in output-CRS units (default 10 m; "
                             "output CRS is the tile UTM zone).")
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="Keep teacher pixels with soft-voted winning prob >= this.")
    parser.add_argument("--holdout-mode", choices=["grid", "orig"], default="grid",
                        help="Which split defines the masked/burned patches: 'grid' = the "
                             "per-city grid split column (mask split in {val,test}, burn "
                             "split=train) — for grid-split seg evaluation; 'orig' = the "
                             "original So2Sat split (mask dataset in {validation,testing}, "
                             "burn dataset=training) — REQUIRED when the student will be "
                             "benchmarked on the original So2Sat test set "
                             "(eval_seg_on_patches --global-split), otherwise test GT "
                             "leaks into training rasters.")
    parser.add_argument("--teacher-dir", type=Path, default=None,
                        help="Directory holding teacher_{city}[_conf].tif (default: "
                             "--output-dir). Point at an existing run to reuse cached "
                             "teacher inference while writing pseudo rasters elsewhere.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--margin-m", type=float, default=200.0)
    parser.add_argument("--dequantize", action="store_true")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Reuse existing teacher_{city}.tif rasters and only "
                             "redo the post-processing (e.g. to sweep --min-conf).")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


class NormalizedEnsemble(torch.nn.Module):
    """Mean softmax of several classifiers, each fed its own normalisation.

    ``infer_roi`` applies ONE normalisation before the model and a softmax
    after it, but seeds of one recipe store slightly different channel stats
    (each computed on its own run). So the raw tile goes in, every member
    normalises it itself, and the forward returns ``log(mean softmax)`` --
    which ``infer_roi``'s softmax turns back into exactly the mean.
    """

    def __init__(self, members: list[tuple[torch.nn.Module, tuple | None]]):
        super().__init__()
        self.models = torch.nn.ModuleList(m for m, _ in members)
        for i, (_, norm) in enumerate(members):
            mean, std = norm if norm is not None else (np.zeros(1), np.ones(1))
            self.register_buffer(f"mean{i}", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
            self.register_buffer(f"std{i}", torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = 0
        for i, m in enumerate(self.models):
            z = (x - getattr(self, f"mean{i}")) / getattr(self, f"std{i}")
            p = p + torch.softmax(m(z), dim=1)
        return torch.log(p / len(self.models) + 1e-8)


def build_pseudo_raster(
    teacher_tif: Path,
    conf_tif: Path,
    split_gpkg: Path,
    label_col: str,
    min_conf: float,
    out_path: Path,
    holdout_mode: str = "grid",
) -> np.ndarray:
    """Fuse teacher raster + confidence + So2Sat split polygons into out_path.

    holdout_mode selects which split column defines held-out (masked) vs
    burnable (GT) patches — see the --holdout-mode CLI help.

    Returns the final uint8 raster (0 = nodata, 1-17 = class).
    """
    import rasterio
    from rasterio.features import rasterize as rio_rasterize

    with rasterio.open(teacher_tif) as src:
        teacher = src.read(1)
        profile = src.profile
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
    with rasterio.open(conf_tif) as src:
        conf = src.read(1)

    pseudo = np.where(conf >= min_conf, teacher, 0).astype(np.uint8)
    n_total = pseudo.size
    n_teacher = int((pseudo > 0).sum())

    sdf = gpd.read_file(split_gpkg)
    if str(sdf.crs) != str(crs):
        sdf = sdf.to_crs(crs)

    # Zero out held-out patch footprints so eval pixels never carry pseudo labels
    if holdout_mode == "orig":
        holdout = sdf[sdf["dataset"].isin(["validation", "testing"])]
        train = sdf[sdf["dataset"] == "training"]
    else:
        holdout = sdf[sdf["split"].isin(["val", "test"])]
        train = sdf[sdf["split"] == "train"]
    if len(holdout):
        holdout_mask = rio_rasterize(
            [(g, 1) for g in holdout.geometry],
            out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
        )
        pseudo[holdout_mask > 0] = 0

    # Burn train-split ground truth on top (GT overrides teacher)
    if len(train):
        gt = rio_rasterize(
            [(g, int(c)) for g, c in zip(train.geometry, train[label_col])],
            out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
        )
        pseudo = np.where(gt > 0, gt, pseudo)

    profile.update(dtype="uint8", nodata=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pseudo, 1)

    n_kept = int((pseudo > 0).sum())
    logger.info(
        f"  teacher kept {n_teacher / n_total:.1%} of pixels (conf >= {min_conf}); "
        f"final labelled {n_kept / n_total:.1%} "
        f"(holdout masked: {len(holdout)}, GT burned: {len(train)} patches)"
    )
    counts = np.bincount(pseudo.ravel(), minlength=18)
    logger.info("  per class: " + ", ".join(
        f"LCZ{c}: {n}" for c, n in enumerate(counts) if c > 0 and n > 0))
    return pseudo


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    dequantize_fn, _ = resolve_dequantize(args.embedding_name, force=args.dequantize)
    # load_model_and_normalize, not a bare state-dict load: it rebuilds the
    # family with the trained img_size and returns the checkpoint's own input
    # stats. The old path fed normalised-trained teachers raw embeddings.
    members = [
        load_model_and_normalize(
            ck, args.family, args.embedding_name, device,
            preset=args.preset, arch=args.arch, num_classes=args.num_classes,
            patch_size=args.patch_size,
        )
        for ck in args.checkpoint
    ]
    model = NormalizedEnsemble(members).to(device).eval()
    logger.info(f"Teacher: {args.family}/{args.preset} x{len(members)} "
                f"from {', '.join(str(c) for c in args.checkpoint)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    failed: list[str] = []
    for city in args.cities:
        try:
            n_ok += _process_city(args, city, model, device, dequantize_fn)
        except Exception:
            logger.exception(f"{city}: FAILED — continuing with next city")
            failed.append(city)
    logger.info(f"Done: {n_ok}/{len(args.cities)} cities"
                + (f"; failed: {', '.join(failed)}" if failed else ""))


def _process_city(args, city: str, model, device, dequantize_fn) -> int:
    """Generate teacher + pseudo rasters for one city. Returns 1 on success, 0 if skipped."""
    city_dir = args.cities_dir / city
    grid_gpkg = city_dir / f"{city}_grid.gpkg"
    split_gpkg = city_dir / f"patches_reference_{city}_split.gpkg"
    if not grid_gpkg.exists() or not split_gpkg.exists():
        logger.warning(f"{city}: missing {grid_gpkg.name} or {split_gpkg.name} — skipping")
        return 0

    teacher_dir = args.teacher_dir or args.output_dir
    teacher_tif = teacher_dir / f"teacher_{city}.tif"
    conf_tif = teacher_dir / f"teacher_{city}_conf.tif"

    if args.skip_inference and teacher_tif.exists() and conf_tif.exists():
        logger.info(f"{city}: reusing existing {teacher_tif.name}")
    else:
        grid_gdf = gpd.read_file(grid_gpkg)
        # Only the valid cells have extracted embeddings, so only they are ever
        # trained on -- and the full grid can be ~60x their area (Zurich: 161
        # of 9,657 cells), all of it inference nobody reads.
        valid = grid_gdf[grid_gdf["is_valid"]] if "is_valid" in grid_gdf.columns else grid_gdf
        valid_4326 = valid.to_crs("EPSG:4326")
        west, south, east, north = valid_4326.total_bounds
        logger.info(f"{city}: teacher inference over bbox "
                    f"({west:.3f}, {south:.3f}, {east:.3f}, {north:.3f})")
        infer_roi(
            model=model,
            model_type=args.family,
            embedding_name=args.embedding_name,
            embedding_dir=args.embedding_dir,
            bbox=(west, south, east, north),
            output_path=teacher_tif,
            # The grid's own CRS: the seg loader crops the label raster by the
            # tile polygon, and a polygon from another UTM zone crops a
            # rotated, resampled window rather than the tile's own pixels.
            out_crs=str(grid_gdf.crs),
            roi_geom_4326=valid_4326.union_all(),
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            device=device,
            dequantize_fn=dequantize_fn,
            out_res=args.out_res,
            year=args.year,
            city_name=city,
            title=f"Teacher — {city} — {args.embedding_name} — {args.family}/{args.preset}",
            margin_m=args.margin_m,
            patch_physical_res_m=args.patch_physical_res,
            patch_physical_stride_m=args.patch_physical_stride,
            save_confidence=True,
        )

    out_path = args.output_dir / f"pseudo_seg_{city}.tif"
    pseudo = build_pseudo_raster(
        teacher_tif, conf_tif, split_gpkg, args.label_col, args.min_conf, out_path,
        holdout_mode=args.holdout_mode,
    )

    from utils.plot_lcz import save_lcz_map
    save_lcz_map(pseudo, f"Pseudo labels — {city} (min-conf {args.min_conf})",
                 out_path.with_suffix(".png"))
    logger.info(f"{city}: wrote {out_path}")
    return 1


if __name__ == "__main__":
    main()
