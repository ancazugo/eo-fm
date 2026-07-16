"""Semantic segmentation trained on grid tile embeddings.

Any segmentation family in the models registry can be trained
(``--family``: unet, resnet_unet, ...).

For each city the script:
  1. Reads {city}/{output_name}/{year}/{split}/{city}_{grid_id}.npy
  2. Loads the per-tile label mask from:
       gpkg (default): rasterises patches_reference_{city}_split.gpkg polygons
                       that fall in this tile → (H, W) label mask
       tif:            clips patches_reference_{city}.tif to tile bounds and
                       resizes to (H, W) by nearest-neighbour
  3. Trains with the grid-based train/val/test split and logs to WandB.

Label convention: raw 1-17 → 0-16 (class index), raw 0 (nodata) → -1 (ignore_index)

Example (single city, AlphaEarth, labels from GeoPackage):
    python src/semantic_segmentation.py \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi \\
        --output-name AlphaEarth --year 2017 \\
        --label-source gpkg \\
        --preset large --batch-size 16 --num-workers 4 \\
        --max-epochs 50 \\
        --embedding-name alpha_earth \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017 \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl

Example (ResNet-UNet, labels from raster TIF):
    python src/semantic_segmentation.py \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi Paris Berlin \\
        --output-name GeoTessera --year 2017 \\
        --label-source tif \\
        --family resnet_unet --preset base --batch-size 8 --num-workers 4 \\
        --max-epochs 50 \\
        --embedding-name tessera \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/GeoTessera/2017 \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import wandb
from loguru import logger

# ── src/ must be on sys.path (run from repo root) ────────────────────────────

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from datasets.grid_tiles import GridSegDataModule, build_city_tile_items
from datasets.registry import EMBEDDING_REGISTRY
from models import build_model, resolve_arch
from training import (
    LCZUNetModule,
    evaluate_segmentation,
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


def _fuse_items(items: list, output_names: list[str], year: str) -> tuple[list, int]:
    """Turn item npy_paths into per-source tuples by swapping the output-name
    path component ({city}/{name}/{year}/{split}/{file}). Items missing the
    npy in any extra source are dropped. Returns (fused_items, n_dropped)."""
    fused, dropped = [], 0
    for it in items:
        p0 = it[0]
        paths = [p0] + [
            p0.parents[3] / name / year / p0.parents[0].name / p0.name
            for name in output_names[1:]
        ]
        if all(p.exists() for p in paths[1:]):
            fused.append((tuple(paths), *it[1:]))
        else:
            dropped += 1
    return fused, dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a segmentation model on grid tile embeddings."
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Data")
    g.add_argument("--cities-dir", required=True, type=Path,
                   help="Root directory with one subfolder per city.")
    g.add_argument("--cities", nargs="+", default=None,
                   help="City names to include (default: all with grid GDF).")
    g.add_argument("--output-name", required=True, nargs="+",
                   help="Embedding folder name(s) inside each city dir (e.g. AlphaEarth). "
                        "Multiple names = channel fusion: per grid cell the sources' npys "
                        "are concatenated along channels (source 0 defines the grid; "
                        "cells missing in any source are dropped).")
    g.add_argument("--aux-channel-dropout", type=float, default=None,
                   help="With fused sources: per-sample probability of zeroing all "
                        "non-source-0 channels during training (default 0.3 when fused; "
                        "guards against OSM-completeness identity learning).")
    g.add_argument("--year", required=True,
                   help="Year subfolder (e.g. 2017).")
    g.add_argument("--label-source", choices=["gpkg", "tif"], default="gpkg",
                   help="Label source: 'gpkg' (rasterise polygon GeoPackage) or "
                        "'tif' (clip raster TIF). Default: gpkg.")
    g.add_argument("--label-tif-dir", type=Path, default=None,
                   help="Train on dense pseudo-label rasters "
                        "({dir}/pseudo_seg_{city}.tif from generate_seg_pseudo_rasters.py). "
                        "Implies --label-source tif for the train split; val/test labels "
                        "always come from the ground-truth gpkg.")
    g.add_argument("--label-col", default="LCZ_class",
                   help="Column name in the GeoPackage for LCZ class (default: LCZ_class).")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = add_model_args(parser, "segmentation", default_family="unet")
    g.add_argument("--bottleneck-dropout", type=float, default=0.3,
                   help="Bottleneck dropout probability (default: 0.3).")

    # ── Loss ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Loss")
    g.add_argument("--dice-weight", type=float, default=0.5,
                   help="Weight of Dice loss (0 = CE-only, 1 = Dice-only). Default: 0.5.")

    # ── Training ──────────────────────────────────────────────────────────────
    add_training_args(parser, batch_size=16)

    # ── Logging ───────────────────────────────────────────────────────────────
    add_logging_args(parser)

    # ── Inference ─────────────────────────────────────────────────────────────
    g = add_inference_args(parser)
    g.add_argument("--embedding-name", required=True,
                   choices=sorted(EMBEDDING_REGISTRY),
                   help="Embedding registry key for infer_roi.")
    g.add_argument("--patch-size", type=int, default=64,
                   help="Sliding-window patch size in pixels for inference (default: 64).")

    args = parser.parse_args()
    resolve_overlap(args)

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    # ── Collect cities ────────────────────────────────────────────────────────
    cities_dir = args.cities_dir
    city_dirs = sorted(d for d in cities_dir.iterdir() if d.is_dir())
    if args.cities:
        city_dirs = [d for d in city_dirs if d.name in args.cities]
        if not city_dirs:
            logger.error(f"None of {args.cities} found in {cities_dir}")
            raise SystemExit(1)

    # ── Build item lists ──────────────────────────────────────────────────────
    if args.label_tif_dir is not None:
        args.label_source = "tif"   # pseudo rasters are tifs

    all_items: list = []
    split_map: dict = {}
    eval_items: list | None = [] if args.label_tif_dir is not None else None
    for city_dir in city_dirs:
        items, sm = build_city_tile_items(
            city_dir, args.output_name[0], args.year,
            args.label_source, args.label_col,
            label_tif_dir=args.label_tif_dir,
        )
        all_items.extend(items)
        split_map.update(sm)
        if eval_items is not None:
            # val/test evaluated against ground-truth gpkg labels, not pseudo
            gt_items, _ = build_city_tile_items(
                city_dir, args.output_name[0], args.year, "gpkg", args.label_col,
            )
            eval_items.extend(gt_items)

    if not all_items:
        logger.error("No items found. Check --cities-dir, --output-name, --year.")
        raise SystemExit(1)

    fused = len(args.output_name) > 1
    if fused:
        all_items, n_drop = _fuse_items(all_items, args.output_name, args.year)
        logger.info(f"Fusion {' + '.join(args.output_name)}: "
                    f"{len(all_items)} tiles ({n_drop} dropped, missing a source)")
        if eval_items is not None:
            eval_items, _ = _fuse_items(eval_items, args.output_name, args.year)
    logger.info(f"Total tiles: {len(all_items)}")

    dequantize_fn, in_channels_override = resolve_dequantize(
        args.embedding_name, force=args.dequantize
    )
    first = all_items[0][0]
    if fused:
        base_channels = detect_in_channels(first[0], in_channels_override)
        in_channels = base_channels + sum(detect_in_channels(p) for p in first[1:])
        aux_dropout = 0.3 if args.aux_channel_dropout is None else args.aux_channel_dropout
        logger.info(f"Fused in_channels = {in_channels} "
                    f"(aux dropout {aux_dropout} on channels {base_channels}:)")
    else:
        base_channels = None
        aux_dropout = 0.0
        in_channels = detect_in_channels(first, in_channels_override)

    # ── Model ─────────────────────────────────────────────────────────────────
    arch = resolve_arch(args.family, args.preset)
    model = build_model(
        args.family, args.preset,
        in_channels=in_channels,
        num_classes=args.num_classes,
        bottleneck_dropout=args.bottleneck_dropout,
    )
    task = LCZUNetModule(
        model, args.num_classes,
        lr=args.lr, weight_decay=args.weight_decay,
        dice_weight=args.dice_weight, max_epochs=args.max_epochs,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"'{args.family}/{args.preset}' ({arch}): params={n_params:,}")

    # ── DataModule ────────────────────────────────────────────────────────────
    datamodule = GridSegDataModule(
        all_items, split_map, args.label_source,
        args.batch_size, args.num_workers,
        dequantize_fn=dequantize_fn,
        eval_items=eval_items,
        eval_label_source="gpkg" if eval_items is not None else None,
        aux_dropout_p=aux_dropout,
        aux_channel_start=base_channels,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    city_names = [d.name for d in city_dirs]
    run_cfg = dict(
        task="segmentation",
        embedding="+".join(args.output_name),
        embedding_name=args.embedding_name,
        cities=city_names,
        year=args.year,
        label_source=args.label_source,
        label_tif_dir=str(args.label_tif_dir) if args.label_tif_dir else None,
        aux_channel_dropout=aux_dropout,
        family=args.family,
        preset=args.preset,
        in_channels=in_channels,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dice_weight=args.dice_weight,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        n_params=n_params,
        data_source="grid_tiles",
    )

    _run_label = "_".join(city_names[:3])
    run_dir = init_run(
        args.output_dir, run_cfg, args.run_name,
        default_name=f"{args.family}_{args.preset}_{_run_label}",
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        no_wandb=args.no_wandb,
    )
    model_name = f"{args.family}_{args.preset}_{'+'.join(args.output_name)}_{_run_label}"

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
        )
    logger.info(f"Best checkpoint: {ckpt_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    datamodule.setup()   # re-create test dataloader after training
    evaluate_segmentation(
        task, datamodule.test_dataloader(), device, args.num_classes,
        run_dir, _run_label, use_wandb=not args.no_wandb,
    )

    if not args.no_wandb and wandb.run:
        wandb.finish()

    # ── Per-city inference raster + map ───────────────────────────────────────
    if fused:
        logger.warning("Fused multi-source model: full-ROI inference from raw "
                       "tiles is not supported yet — skipping city maps.")
        logger.info(f"Run complete. Outputs in {run_dir}")
        return
    logger.info("Running full-ROI inference …")
    run_city_inference(
        task.model, args.family, "segmentation", args.preset,
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
