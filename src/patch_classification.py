"""Patch-level classification on So2Sat patches.

Any classification family in the models registry can be trained
(``--family``: resnet, efficientnet, convnext, densenet, mobilenet, vit,
aspp, mlp, ...).

Two split modes:

  Per-city (default): specify --cities-dir and --cities.
    Uses patches_reference_{city}_split.gpkg (grid-based split column: train/val/test).

  Global (--global-split): uses patches_reference_rxr.gpkg directly.
    The 'dataset' column (training/validation/testing) defines the split — no
    city selection needed, all 400 k+ patches across 51 cities are included.

Label convention: LCZ_class 1-17 → 0-16 (class index)

Example (single city, AlphaEarth):
    python src/patch_classification.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi \\
        --output-name AlphaEarth --year 2017 \\
        --preset large --patch-size 32 \\
        --batch-size 64 --num-workers 4 --max-epochs 50 \\
        --embedding-name alpha_earth \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017 \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl

Example (global split, AlphaEarthCoop):
    python src/patch_classification.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --global-split \\
        --output-name AlphaEarthCoop --year 2017 \\
        --preset large --patch-size 32 \\
        --batch-size 256 --num-workers 8 --max-epochs 50 \\
        --embedding-name alpha_earth_coop \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from loguru import logger

# ── src/ must be on sys.path (run from repo root) ────────────────────────────

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from datasets.registry import EMBEDDING_REGISTRY
from datasets.so2sat import PatchDataModule, build_so2sat_items
from models import build_model, families_for, resolve_arch
from training import (
    LCZResNetModule,
    evaluate_classification,
    run_training_loop,
)
from utils.runtime import (
    detect_in_channels,
    init_run,
    resolve_dequantize,
    resolve_device,
    run_city_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a patch classifier on So2Sat patches "
                    "using the grid-based train/val/test split."
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Data")
    g.add_argument("--so2sat-dir", required=True, type=Path,
                   help="Root So2Sat directory that contains the "
                        "training/validation/testing subfolders with patch npy files.")
    g.add_argument("--global-split", action="store_true",
                   help="Use the global patches_reference_rxr.gpkg with its original "
                        "training/validation/testing split instead of per-city split GeoPackages.")
    g.add_argument("--global-gpkg", type=Path, default=None,
                   help="Path to global patches GPKG "
                        "(default: {so2sat_dir}/patches_reference_rxr.gpkg). "
                        "Only used with --global-split.")
    g.add_argument("--cities-dir", required=False, default=None, type=Path,
                   help="Directory containing one subfolder per city "
                        "(each must have patches_reference_{city}_split.gpkg). "
                        "Required unless --global-split is set.")
    g.add_argument("--cities", nargs="+", default=None,
                   help="City names to include. In per-city mode: selects cities for training. "
                        "In global-split mode: selects cities for post-training inference only.")
    g.add_argument("--output-name", required=True,
                   help="Embedding name used as subfolder in the So2Sat patch dirs "
                        "(e.g. AlphaEarth or GeoTessera).")
    g.add_argument("--year", required=True,
                   help="Year subfolder (e.g. 2017).")
    g.add_argument("--label-col", default="LCZ_class",
                   help="Column name for LCZ class in the split GDF (default: LCZ_class).")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Model")
    g.add_argument("--family", choices=families_for("classification"), default="resnet",
                   help="Model family (default: resnet).")
    g.add_argument("--preset", choices=["nano", "small", "base", "medium", "large"], default="large",
                   help="Size preset (default: large).")
    g.add_argument("--arch", default=None,
                   help="Override: any timm model name.")
    g.add_argument("--num-classes", type=int, default=17,
                   help="Number of output classes (default: 17).")
    g.add_argument("--patch-size", type=int, default=32,
                   help="Resize patches to this square size before feeding the model (default: 32).")
    g.add_argument("--sub-patch-size", type=int, default=None,
                   help="If set, sample sub-patches of this size (pixels) from each parent patch "
                        "instead of using the full patch. Sub-patches inherit the parent label.")
    g.add_argument("--sub-patch-stride", type=int, default=None,
                   help="Stride (pixels) for sub-patch sampling (default: sub-patch-size, "
                        "i.e. non-overlapping).")
    g.add_argument("--head-dropout", type=float, default=0.0,
                   help="Dropout before the final FC layer (default: 0.0).")

    # ── Training ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Training")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--class-weights", choices=["none", "inv_freq", "sqrt_inv_freq"],
                   default="none",
                   help="Per-class CE weights from train-split frequencies (default: none).")
    g.add_argument("--label-smoothing", type=float, default=0.0,
                   help="CE label smoothing (default: 0.0).")
    g.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="Mixup Beta(alpha, alpha) on training batches (default: 0.0 = off).")
    g.add_argument("--monitor", choices=["val_f1", "val_kappa"], default="val_f1",
                   help="Validation metric for checkpointing/early stopping (default: val_f1).")
    g.add_argument("--tta", action="store_true",
                   help="Test-time augmentation: average logits over flips/90° rotations.")
    g.add_argument("--max-epochs", type=int, default=50)
    g.add_argument("--early-stopping-patience", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    # ── Logging ───────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Logging")
    g.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for checkpoints and WandB run folders.")
    g.add_argument("--wandb-project", default="lcz-classification-dl")
    g.add_argument("--wandb-entity", default="phd-thesis-team")
    g.add_argument("--no-wandb", action="store_true",
                   help="Disable WandB logging.")
    g.add_argument("--run-name", default=None,
                   help="Optional WandB run name override.")
    g.add_argument("--dequantize", action="store_true",
                   help="Force dequantize when loading npy patches "
                        "(auto-applied for alpha_earth_coop and seamless).")
    g.add_argument("--checkpoint", type=Path, default=None,
                   help="Load model weights from this .pt file and skip training (inference only).")

    # ── Inference ─────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Inference")
    g.add_argument("--embedding-name", required=True,
                   choices=sorted(EMBEDDING_REGISTRY),
                   help="Embedding registry key for infer_roi.")
    g.add_argument("--embedding-dir", required=True, type=Path,
                   help="Directory containing raw source embedding tiles (.zarr or .tif).")
    g.add_argument("--overlap", type=int, default=None,
                   help="Overlap between adjacent patches in pixels for inference "
                        "(default: patch_size // 2).")
    g.add_argument("--margin-m", type=float, default=200.0,
                   help="Extra metres clipped around city bbox per tile for edge context (default: 200).")

    args = parser.parse_args()
    if args.overlap is None:
        args.overlap = args.patch_size // 2

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    # ── Item lists + dequantize ───────────────────────────────────────────────
    all_items, city_dirs = build_so2sat_items(
        args.so2sat_dir, args.output_name, args.year,
        global_split=args.global_split,
        global_gpkg=args.global_gpkg,
        cities_dir=args.cities_dir,
        cities=args.cities,
        label_col=args.label_col,
    )
    split_counts = {s: sum(1 for _, _, sp in all_items if sp == s)
                    for s in ("train", "val", "test")}
    logger.info(f"Total patches: {len(all_items)}  splits: {split_counts}")

    dequantize_fn, in_channels_override = resolve_dequantize(
        args.embedding_name, force=args.dequantize
    )
    in_channels = detect_in_channels(all_items[0][0], in_channels_override)

    # ── Model ─────────────────────────────────────────────────────────────────
    arch = resolve_arch(args.family, args.preset, args.arch)
    model = build_model(
        args.family, args.preset, args.arch,
        in_channels=in_channels,
        num_classes=args.num_classes,
        head_dropout=args.head_dropout,
        img_size=args.patch_size if args.family == "vit" else None,
    )
    class_weights = None
    if args.class_weights != "none":
        counts = np.bincount(
            [label for _, label, split in all_items if split == "train"],
            minlength=args.num_classes,
        ).astype(np.float64)
        weights = 1.0 / np.maximum(counts, 1)
        if args.class_weights == "sqrt_inv_freq":
            weights = np.sqrt(weights)
        weights /= weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class weights ({args.class_weights}): {np.round(weights, 3)}")

    task = LCZResNetModule(
        model=model,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        monitor=args.monitor,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"'{args.family}/{args.preset}' ({arch}): params={n_params:,}")

    # ── DataModule ────────────────────────────────────────────────────────────
    datamodule = PatchDataModule(
        all_items, args.patch_size, args.batch_size, args.num_workers,
        sub_patch_size=args.sub_patch_size,
        sub_patch_stride=args.sub_patch_stride,
        dequantize_fn=dequantize_fn,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    city_names = [d.name for d in city_dirs]
    run_cfg = dict(
        task="patch_classification",
        embedding=args.output_name,
        cities="all_so2sat" if args.global_split else city_names,
        year=args.year,
        family=args.family,
        preset=args.preset,
        arch=arch,
        in_channels=in_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        sub_patch_size=args.sub_patch_size,
        sub_patch_stride=args.sub_patch_stride,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=args.class_weights,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        monitor=args.monitor,
        tta=args.tta,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        n_params=n_params,
        data_source="so2sat_patches",
        split_source="global_so2sat" if args.global_split else "grid",
        **{f"{s}_patches": split_counts[s] for s in ("train", "val", "test")},
    )

    _run_label = "global" if args.global_split else "_".join(city_names[:3])
    run_dir = init_run(
        args.output_dir, run_cfg, args.run_name,
        default_name=f"{args.family}_{args.preset}_{_run_label}",
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        no_wandb=args.no_wandb,
    )
    model_name = f"{args.family}_{args.preset}_{args.output_name}_{_run_label}"

    # ── Train (or load checkpoint) ────────────────────────────────────────────
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        task.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        task = task.to(device)
        ckpt_path = args.checkpoint
    else:
        task, ckpt_path = run_training_loop(
            task_module=task,
            datamodule=datamodule,
            device=device,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience,
            run_dir=run_dir,
            model_name=model_name,
        )
    logger.info(f"Best checkpoint: {ckpt_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    datamodule.setup()
    evaluate_classification(
        task, datamodule.test_dataloader(), device, args.num_classes,
        run_dir, _run_label, use_wandb=not args.no_wandb, tta=args.tta,
    )

    if not args.no_wandb and wandb.run:
        wandb.finish()

    # ── Per-city inference raster + map ───────────────────────────────────────
    logger.info("Running full-ROI inference …")
    run_city_inference(
        task.model, args.family, "classification", args.preset,
        city_dirs, run_dir, device, dequantize_fn,
        embedding_name=args.embedding_name,
        embedding_dir=args.embedding_dir,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        year=args.year,
        margin_m=args.margin_m,
    )

    logger.info(f"Run complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
