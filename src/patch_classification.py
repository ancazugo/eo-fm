"""Patch-level classification on So2Sat patches.

Any classification family in the models registry can be trained
(``--family``: resnet, efficientnet, convnext, densenet, mobilenet, vit,
aspp, mlp, ...).

Three split modes:

  Per-city (default): specify --cities-dir and --cities.
    Uses patches_reference_{city}_split.gpkg (grid-based split column: train/val/test).

  Global (--global-split): uses patches_reference_rxr.gpkg directly.
    The 'dataset' column (training/validation/testing) defines the split — no
    city selection needed, every patch in the GeoPackage is included.

  Hybrid (--orig-test): grid-based train/val from the per-city GeoPackages,
    test set = the original So2Sat testing patches (comparable to the global
    benchmark without its training split).

Class imbalance / SSL levers: --class-weights, --sampler, --logit-adjustment,
--mixup-alpha, and --pseudo-gpkg (noisy-student pseudo-labels with per-sample
loss weights). Multiple --output-name/--embedding-name pairs fuse embeddings
by channel concatenation.

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

from datasets.registry import EMBEDDING_REGISTRY, get_nodata_predicate
from datasets.so2sat import PatchDataModule, build_so2sat_items
from models import build_model, resolve_arch
from training import (
    LCZResNetModule,
    evaluate_classification,
    run_training_loop,
)
from utils.cli import (
    add_inference_args,
    add_logging_args,
    add_model_args,
    add_training_args,
    resolve_overlap,
)
from utils.runtime import (
    detect_in_channels,
    init_run,
    load_checkpoint_weights,
    resolve_dequantize,
    resolve_device,
    run_city_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a patch classifier on So2Sat patches. Split modes: "
                    "per-city grid (default), --global-split (original So2Sat "
                    "split), or --orig-test (grid train/val + original test set)."
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
    g.add_argument("--orig-test", action="store_true",
                   help="Hybrid split: train/val come from the per-city grid split "
                        "(grid-test-cell patches fold into train) but the test set is "
                        "the original So2Sat testing patches. Trains on ALL cities; "
                        "--cities selects inference cities only. Mutually exclusive "
                        "with --global-split.")
    g.add_argument("--cities-dir", required=False, default=None, type=Path,
                   help="Directory containing one subfolder per city "
                        "(each must have patches_reference_{city}_split.gpkg). "
                        "Required unless --global-split is set.")
    g.add_argument("--cities", nargs="+", default=None,
                   help="City names to include. In per-city mode: selects cities for training. "
                        "In global-split mode: selects cities for post-training inference only.")
    g.add_argument("--output-name", required=True, nargs="+",
                   help="Embedding name(s) used as subfolder in the So2Sat patch dirs "
                        "(e.g. AlphaEarth or GeoTessera). Multiple names fuse the "
                        "embeddings by channel concatenation (must pair 1:1 with "
                        "--embedding-name).")
    g.add_argument("--year", required=True,
                   help="Year subfolder (e.g. 2017).")
    g.add_argument("--label-col", default="LCZ_class",
                   help="Column name for LCZ class in the split GDF (default: LCZ_class).")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = add_model_args(parser, "classification", default_family="resnet")
    g.add_argument("--arch", default=None,
                   help="Override: any timm model name.")
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
    g.add_argument("--nodata-mode", choices=["zero", "mask"], default="mask",
                   help="How to treat per-family nodata sentinels (see "
                        "datasets.registry.get_nodata_predicate): 'mask' emits a "
                        "'valid' channel and fills invalid pixels, 'zero' keeps the "
                        "pre-Phase-1 behaviour of passing them straight through "
                        "(default: mask).")

    # ── Training ──────────────────────────────────────────────────────────────
    g = add_training_args(parser, batch_size=64)
    g.add_argument("--class-weights", choices=["none", "inv_freq", "sqrt_inv_freq"],
                   default="none",
                   help="Per-class CE weights from train-split frequencies (default: none).")
    g.add_argument("--label-smoothing", type=float, default=0.0,
                   help="CE label smoothing (default: 0.0).")
    g.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="Mixup Beta(alpha, alpha) on training batches (default: 0.0 = off).")
    g.add_argument("--sampler", choices=["none", "balanced", "sqrt_balanced"],
                   default="none",
                   help="Class-balanced train sampling: per-sample weight 1/count "
                        "(balanced) or 1/sqrt(count) (sqrt_balanced) of the class "
                        "(default: none).")
    g.add_argument("--logit-adjustment", type=float, default=0.0,
                   help="Tau for logit-adjusted CE (Menon et al. 2021): training loss "
                        "sees logits + tau*log(train prior); eval uses raw logits "
                        "(default: 0 = off).")
    g.add_argument("--pseudo-gpkg", type=Path, default=None,
                   help="Pseudo-label GeoPackage (generate_pseudo_labels.py output): "
                        "its patches are appended to the train split with per-sample "
                        "loss weights.")
    g.add_argument("--pseudo-weight-scale", type=float, default=1.0,
                   help="Global multiplier on the pseudo-label sample weights (default: 1.0).")
    g.add_argument("--monitor", choices=["val_f1", "val_kappa"], default="val_f1",
                   help="Validation metric for checkpointing/early stopping (default: val_f1).")
    g.add_argument("--tta", action="store_true",
                   help="Test-time augmentation: average logits over flips/90° rotations.")
    g.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup epochs before cosine decay (0 = off).")

    # ── Logging ───────────────────────────────────────────────────────────────
    add_logging_args(parser)

    # ── Inference ─────────────────────────────────────────────────────────────
    g = add_inference_args(parser)
    g.add_argument("--embedding-name", required=True, nargs="+",
                   choices=sorted(EMBEDDING_REGISTRY),
                   help="Embedding registry key(s), one per --output-name. "
                        "Also used for infer_roi (single-embedding runs only).")

    args = parser.parse_args()
    resolve_overlap(args)
    if len(args.output_name) != len(args.embedding_name):
        parser.error("--output-name and --embedding-name must have the same length")
    fused = len(args.output_name) > 1
    if fused and args.sub_patch_size is not None:
        parser.error("--sub-patch-size is not supported with fused embeddings")
    output_label = "+".join(args.output_name)

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    # ── Item lists + dequantize ───────────────────────────────────────────────
    all_items, city_dirs = build_so2sat_items(
        args.so2sat_dir,
        args.output_name if fused else args.output_name[0],
        args.year,
        global_split=args.global_split,
        global_gpkg=args.global_gpkg,
        cities_dir=args.cities_dir,
        cities=args.cities,
        label_col=args.label_col,
        orig_test=args.orig_test,
    )
    n_pseudo = 0
    if args.pseudo_gpkg is not None:
        from datasets.so2sat import (build_patch_index, build_pseudo_items,
                                     merge_patch_indexes)
        indexes = [build_patch_index(args.so2sat_dir, n, args.year)
                   for n in args.output_name]
        pseudo_index = indexes[0] if not fused else merge_patch_indexes(indexes)
        pseudo_items = build_pseudo_items(
            args.pseudo_gpkg, pseudo_index, weight_scale=args.pseudo_weight_scale
        )
        n_pseudo = len(pseudo_items)
        all_items = all_items + pseudo_items

    split_counts = {s: sum(1 for it in all_items if it[2] == s)
                    for s in ("train", "val", "test")}
    logger.info(f"Total patches: {len(all_items)}  splits: {split_counts}"
                + (f"  (incl. {n_pseudo} pseudo-labeled)" if n_pseudo else ""))

    deq = [resolve_dequantize(e, force=args.dequantize) for e in args.embedding_name]
    if fused:
        dequantize_fn = [fn for fn, _ in deq]
        first_paths = all_items[0][0]
        in_channels = sum(
            detect_in_channels(p, override)
            for p, (_, override) in zip(first_paths, deq)
        )
        logger.info(f"Fused in_channels = {in_channels} ({output_label})")
    else:
        dequantize_fn, in_channels_override = deq[0]
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
    class_priors = None
    if args.class_weights != "none" or args.logit_adjustment > 0:
        counts = np.bincount(
            [it[1] for it in all_items if it[2] == "train"],
            minlength=args.num_classes,
        ).astype(np.float64)
    if args.class_weights != "none":
        weights = 1.0 / np.maximum(counts, 1)
        if args.class_weights == "sqrt_inv_freq":
            weights = np.sqrt(weights)
        weights /= weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class weights ({args.class_weights}): {np.round(weights, 3)}")
    if args.logit_adjustment > 0:
        priors = np.maximum(counts, 1) / counts.sum()
        class_priors = torch.tensor(priors, dtype=torch.float32)
        logger.info(f"Logit adjustment tau={args.logit_adjustment}, "
                    f"priors: {np.round(priors, 4)}")

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
        logit_adjustment_tau=args.logit_adjustment,
        class_priors=class_priors,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"'{args.family}/{args.preset}' ({arch}): params={n_params:,}")

    # ── DataModule ────────────────────────────────────────────────────────────
    nodata_predicate = [get_nodata_predicate(e) for e in args.embedding_name]
    if not fused:
        nodata_predicate = nodata_predicate[0]

    datamodule = PatchDataModule(
        all_items, args.patch_size, args.batch_size, args.num_workers,
        sub_patch_size=args.sub_patch_size,
        sub_patch_stride=args.sub_patch_stride,
        dequantize_fn=dequantize_fn,
        sampler=args.sampler,
        nodata_mode=args.nodata_mode,
        nodata_predicate=nodata_predicate,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    city_names = [d.name for d in city_dirs]
    _global_like = args.global_split or args.orig_test
    run_cfg = dict(
        task="patch_classification",
        embedding=output_label,
        embedding_name="+".join(args.embedding_name),
        cities="all_so2sat" if _global_like else city_names,
        year=args.year,
        family=args.family,
        preset=args.preset,
        arch=arch,
        in_channels=in_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        sub_patch_size=args.sub_patch_size,
        sub_patch_stride=args.sub_patch_stride,
        nodata_mode=args.nodata_mode,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=args.class_weights,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        sampler=args.sampler,
        logit_adjustment=args.logit_adjustment,
        pseudo_gpkg=str(args.pseudo_gpkg) if args.pseudo_gpkg else None,
        pseudo_patches=n_pseudo,
        pseudo_weight_scale=args.pseudo_weight_scale,
        monitor=args.monitor,
        tta=args.tta,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        n_params=n_params,
        data_source="so2sat_patches",
        split_source=("grid_orig_test" if args.orig_test
                      else "global_so2sat" if args.global_split else "grid"),
        **{f"{s}_patches": split_counts[s] for s in ("train", "val", "test")},
    )

    _run_label = "global" if _global_like else "_".join(city_names[:3])
    run_dir = init_run(
        args.output_dir, run_cfg, args.run_name,
        default_name=f"{args.family}_{args.preset}_{_run_label}",
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        no_wandb=args.no_wandb,
    )
    model_name = f"{args.family}_{args.preset}_{output_label}_{_run_label}"

    # ── Train (or load checkpoint) ────────────────────────────────────────────
    if args.checkpoint is not None:
        ckpt_path = load_checkpoint_weights(task, args.checkpoint, device)
    else:
        task, ckpt_path = run_training_loop(
            task_module=task,
            datamodule=datamodule,
            device=device,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience,
            run_dir=run_dir,
            model_name=model_name,
            warmup_epochs=args.warmup_epochs,
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
    if fused:
        logger.warning("Fused-embedding run: full-ROI inference from raw tiles "
                       "is not supported yet — skipping.")
        logger.info(f"Run complete. Outputs in {run_dir}")
        return

    logger.info("Running full-ROI inference …")
    run_city_inference(
        task.model, args.family, "classification", args.preset,
        city_dirs, run_dir, device, dequantize_fn,
        embedding_name=args.embedding_name[0],
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
