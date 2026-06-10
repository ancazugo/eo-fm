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
from models import build_model, families_for, resolve_arch
from training import (
    LCZUNetModule,
    evaluate_segmentation,
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
        description="Train a segmentation model on grid tile embeddings."
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Data")
    g.add_argument("--cities-dir", required=True, type=Path,
                   help="Root directory with one subfolder per city.")
    g.add_argument("--cities", nargs="+", default=None,
                   help="City names to include (default: all with grid GDF).")
    g.add_argument("--output-name", required=True,
                   help="Embedding folder name inside each city dir (e.g. AlphaEarth).")
    g.add_argument("--year", required=True,
                   help="Year subfolder (e.g. 2017).")
    g.add_argument("--label-source", choices=["gpkg", "tif"], default="gpkg",
                   help="Label source: 'gpkg' (rasterise polygon GeoPackage) or "
                        "'tif' (clip raster TIF). Default: gpkg.")
    g.add_argument("--label-col", default="LCZ_class",
                   help="Column name in the GeoPackage for LCZ class (default: LCZ_class).")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Model")
    g.add_argument("--family", choices=families_for("segmentation"), default="unet",
                   help="Model family (default: unet).")
    g.add_argument("--preset", choices=["nano", "small", "base", "medium", "large"],
                   default="large",
                   help="Size preset (default: large).")
    g.add_argument("--num-classes", type=int, default=17,
                   help="Number of output classes (default: 17).")
    g.add_argument("--bottleneck-dropout", type=float, default=0.3,
                   help="Bottleneck dropout probability (default: 0.3).")

    # ── Loss ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Loss")
    g.add_argument("--dice-weight", type=float, default=0.5,
                   help="Weight of Dice loss (0 = CE-only, 1 = Dice-only). Default: 0.5.")

    # ── Training ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Training")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--max-epochs", type=int, default=50)
    g.add_argument("--early-stopping-patience", type=int, default=10)
    g.add_argument("--seed", type=int, default=411)
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
                   help="Force dequantize when loading npy tiles "
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
    g.add_argument("--patch-size", type=int, default=64,
                   help="Sliding-window patch size in pixels for inference (default: 64).")
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

    # ── Collect cities ────────────────────────────────────────────────────────
    cities_dir = args.cities_dir
    city_dirs = sorted(d for d in cities_dir.iterdir() if d.is_dir())
    if args.cities:
        city_dirs = [d for d in city_dirs if d.name in args.cities]
        if not city_dirs:
            logger.error(f"None of {args.cities} found in {cities_dir}")
            raise SystemExit(1)

    # ── Build item lists ──────────────────────────────────────────────────────
    all_items: list = []
    split_map: dict = {}
    for city_dir in city_dirs:
        items, sm = build_city_tile_items(
            city_dir, args.output_name, args.year,
            args.label_source, args.label_col,
        )
        all_items.extend(items)
        split_map.update(sm)

    if not all_items:
        logger.error("No items found. Check --cities-dir, --output-name, --year.")
        raise SystemExit(1)
    logger.info(f"Total tiles: {len(all_items)}")

    dequantize_fn, in_channels_override = resolve_dequantize(
        args.embedding_name, force=args.dequantize
    )
    in_channels = detect_in_channels(all_items[0][0], in_channels_override)

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
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    city_names = [d.name for d in city_dirs]
    run_cfg = dict(
        task="segmentation",
        embedding=args.output_name,
        cities=city_names,
        year=args.year,
        label_source=args.label_source,
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
    datamodule.setup()   # re-create test dataloader after training
    evaluate_segmentation(
        task, datamodule.test_dataloader(), device, args.num_classes,
        run_dir, _run_label, use_wandb=not args.no_wandb,
    )

    if not args.no_wandb and wandb.run:
        wandb.finish()

    # ── Per-city inference raster + map ───────────────────────────────────────
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
