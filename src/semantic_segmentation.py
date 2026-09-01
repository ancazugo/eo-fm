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

from datasets.grid_tiles import GridSegDataModule, build_city_tile_items, rasterize_polys  # noqa: F401
from datasets.channel_stats import (
    compute_grid_channel_stats,
    grid_stats_cache_path,
)
from datasets.registry import get_nodata_predicate
from utils.city_split import assign_city_roles
from datasets.registry import available_embeddings, provenance
from models import build_model, resolve_arch
from training import (
    LCZUNetModule,
    evaluate_segmentation,
    evaluate_segmentation_as_patches,
    run_training_loop,
    save_metrics_json,
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


def _restrict_to_common_tiles(
    items: list, require: list[str], year: str
) -> tuple[list, int, dict[str, int]]:
    """Drop grid cells that any of ``require`` cannot supply.

    Comparing two embedding arms means comparing them on the same ground. The
    2017 archives do not offer the same ground: AlphaEarth coop covers every
    valid tile in all 51 cities, while Tessera v1.1 global is short 993 tiles
    across 14 coastal cities -- 37 % of Qingdao and 32 % of Istanbul. Those
    tiles are absent from the archive, not unextracted, so the only way to make
    the arms comparable is to hold both to the intersection.

    Without this, a gap between "U-Net on AlphaEarth" and "U-Net on Tessera" is
    partly a gap in which cities each model saw, and the ablation stops
    isolating the embedding -- which is the one thing it exists to do.

    Only membership is filtered; channels are untouched. That is the difference
    from ``_fuse_items``, which stacks the sources into one tensor.

    Returns ``(kept_items, n_dropped, per_city_dropped)``.
    """
    kept, dropped = [], 0
    per_city: dict[str, int] = {}
    for it in items:
        p0 = it[0][0] if isinstance(it[0], tuple) else it[0]
        siblings = [
            p0.parents[3] / name / year / p0.parents[0].name / p0.name
            for name in require
        ]
        if all(sp.exists() for sp in siblings):
            kept.append(it)
        else:
            dropped += 1
            city = p0.parents[3].name
            per_city[city] = per_city.get(city, 0) + 1
    return kept, dropped, per_city


def _pixel_class_weights(all_items, split_map, scheme: str, num_classes: int):
    """Per-class CE weights from the TRAIN split's labelled pixel counts.

    Counted on polygon areas rather than by rasterising every tile: the burn is
    nearest-neighbour onto the tile grid, so area is the same quantity to
    within a pixel and costs one pass over the geometries instead of a full
    epoch of rasterisation.

    Only the train split is counted -- deriving weights from val or test would
    leak the label distribution of the evaluation set into training.
    """
    import numpy as np

    def _key(it):
        return it[0][0] if isinstance(it[0], tuple) else it[0]

    area = np.zeros(num_classes, dtype=np.float64)
    for it in all_items:
        if split_map.get(_key(it)) != "train":
            continue
        for entry in it[3] or []:
            cls = int(entry[1]) - 1          # 1-17 -> 0-16
            if 0 <= cls < num_classes:
                area[cls] += entry[0].area
    if area.sum() == 0:
        logger.warning("No labelled train polygons found — class weights disabled.")
        return None

    freq = np.where(area > 0, area, np.nan)
    inv = 1.0 / freq
    if scheme == "sqrt_inv_freq":
        inv = np.sqrt(inv)
    inv = np.nan_to_num(inv, nan=0.0)
    # Absent classes get the mean weight rather than 0, so that if they do turn
    # up in val or test they are not silently free to get wrong.
    inv[inv == 0] = inv[inv > 0].mean() if (inv > 0).any() else 1.0
    weights = inv / inv.mean()
    logger.info(
        f"Class weights ({scheme}): min={weights.min():.3f} "
        f"max={weights.max():.3f} over {int((area > 0).sum())} present classes"
    )
    return torch.from_numpy(weights.astype("float32"))


def _uid_tables(uid_registry: dict, all_items, eval_items):
    """Invert the UID registry into (uid -> 0-indexed label, uid -> key).

    The label has to come from the item polygons rather than the registry,
    because the registry is keyed on identity alone. The key is
    ``(city, dataset, patch_id)`` -- the triple that is actually unique, since
    patch_id restarts at 000000 in each of So2Sat's original splits.
    """
    uid_to_label: dict[int, int] = {}
    for items in (all_items, eval_items):
        for it in items or []:
            for entry in it[3] or []:
                if len(entry) >= 3:
                    uid_to_label[int(entry[2])] = int(entry[1]) - 1
    uid_to_key = {uid: key for key, uid in uid_registry.items()}
    return uid_to_label, uid_to_key


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
    g.add_argument("--split-mode", choices=["grid", "global"], default="grid",
                   help="'grid' (default): the within-city macro-block split. Section 7 "
                        "of the campaign report quarantines those numbers as "
                        "autocorrelation-inflated, so they cannot carry a headline. "
                        "'global': split by CITY, inheriting the So2Sat culture-10 "
                        "assignment, so results are comparable to the patch ladder. "
                        "Enforces tile-split purity, the proximity buffer and the "
                        "minimum labelled fraction. Requires --label-source gpkg.")
    g.add_argument("--val-inner-cities", type=int, default=6,
                   help="With --split-mode global: how many training-pool cities to "
                        "hold out for early stopping, chosen continent-stratified and "
                        "largest-first (default: 6). The culture cities' own validation "
                        "patches are NOT used for selection — they sit a median 2.55 km "
                        "from the test patches of the same cities.")
    g.add_argument("--buffer-km", type=float, default=1.3,
                   help="With --split-mode global: drop a tile that comes within this "
                        "distance of a patch in a different split (default: 1.3, about "
                        "one 128 px tile). 0 disables.")
    g.add_argument("--min-labelled-frac", type=float, default=0.01,
                   help="Drop tiles whose labelled area falls below this fraction, so "
                        "batches are not mostly ignore_index (default: 0.01).")
    g.add_argument("--require-embeddings", nargs="+", default=None,
                   metavar="OUTPUT_NAME",
                   help="Keep only grid cells that ALSO exist under each of these "
                        "output-name folders, so two embedding arms are trained and "
                        "scored on identical ground. Channels are NOT stacked (that "
                        "is --output-name fusion) — only membership is filtered. "
                        "Needed because Tessera v1.1 global is short 993 tiles "
                        "across 14 coastal cities (Qingdao 37%%, Istanbul 32%%) that "
                        "AlphaEarth coop covers, so an unrestricted comparison "
                        "confounds the embedding with which cities each model saw.")
    g.add_argument("--normalize", choices=["none", "channel"], default="channel",
                   help="Per-channel input standardisation using TRAIN-split "
                        "statistics over valid pixels (default: channel, matching "
                        "the patch pipeline). Statistics are computed through the "
                        "same dataset the model trains on, so they describe "
                        "post-dequantization values and cover fused channel "
                        "stacks. 'none' reproduces the historical unnormalised "
                        "segmentation path — every seg run before 2026-08 used it.")
    g.add_argument("--stats-sample", type=int, default=400,
                   help="Train tiles sampled to estimate channel statistics "
                        "(default: 400; a 128 px tile is 16k pixels, so this is "
                        "already millions of samples).")
    g.add_argument("--erode-px", type=float, default=2.0,
                   help="Shrink each patch footprint by this many pixels before "
                        "burning labels (default: 2). So2Sat patch edges carry "
                        "digitisation slop that WUDAPT explicitly tolerates, so edge "
                        "pixels are unreliable by construction. 0 disables.")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = add_model_args(parser, "segmentation", default_family="unet")
    g.add_argument("--bottleneck-dropout", type=float, default=0.3,
                   help="Bottleneck dropout probability (default: 0.3).")

    # ── Loss ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Loss")
    g.add_argument("--noise-sigma", type=float, default=0.05,
                   help="Gaussian augmentation noise, in units of the normalised "
                        "per-channel std (default: 0.05, historical value — untuned; "
                        "0 disables).")
    g.add_argument("--noise-prob", type=float, default=0.5,
                   help="Probability of adding augmentation noise (default: 0.5).")
    g.add_argument("--dice-weight", type=float, default=0.5,
                   help="Weight of Dice loss (0 = CE-only, 1 = Dice-only). Default: 0.5. "
                        "Consider 0.0 with sparse labels: Dice's denominator is then the "
                        "labelled subset, which is not the objective you want.")
    g.add_argument("--class-weights", choices=["none", "inv_freq", "sqrt_inv_freq"],
                   default="none",
                   help="Per-class CE weighting from the training-split pixel counts "
                        "(default: none). Do not stack with another imbalance mechanism: "
                        "Phase 0 found sqrt-freq sampling and logit adjustment both lost "
                        "on top of loss weighting.")
    g.add_argument("--label-smoothing", type=float, default=0.0,
                   help="CE label smoothing (default: 0.0).")
    g.add_argument("--monitor", choices=["val_miou", "val_kappa", "val_acc", "val_f1"],
                   default="val_miou",
                   help="Metric to checkpoint and early-stop on (default: val_miou). "
                        "Use val_kappa to select on the same quantity the patch "
                        "campaign reports.")

    # ── Training ──────────────────────────────────────────────────────────────
    add_training_args(parser, batch_size=16)

    # ── Logging ───────────────────────────────────────────────────────────────
    add_logging_args(parser)

    # ── Inference ─────────────────────────────────────────────────────────────
    g = add_inference_args(parser)
    g.add_argument("--embedding-name", required=True,
                   choices=available_embeddings(),
                   help="Embedding registry key for infer_roi. Deprecated and "
                        "pending entries are excluded (PLAN-v3).")
    g.add_argument("--patch-size", type=int, default=64,
                   help="Sliding-window patch size in pixels for INFERENCE only "
                        "(default: 64). This does not set the training tile size — that "
                        "is fixed when the grid is built (create_city_grids.py "
                        "--sub-tile-size, default 1280 m = 128 px at 10 m).")
    g.add_argument("--no-inference", action="store_true",
                   help="Skip the post-training full-ROI city rasters. Training "
                        "and both evaluation levels still run — this only drops "
                        "the map generation, which for 51 cities costs far more "
                        "than a short training run and is pure waste when the "
                        "point is a timing calibration or a metrics-only redo.")
    g.add_argument("--tta", action="store_true",
                   help="Average logits over the dihedral group (4 rotations x "
                        "{id, hflip}) at test time.")
    g.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup epochs before cosine decay (0 = off).")

    args = parser.parse_args()
    resolve_overlap(args)

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    # ── Collect cities ────────────────────────────────────────────────────────
    cities_dir = args.cities_dir
    city_dirs = sorted(d for d in cities_dir.iterdir() if d.is_dir())
    if args.cities and args.cities != ["all"]:
        city_dirs = [d for d in city_dirs if d.name in args.cities]
        if not city_dirs:
            logger.error(f"None of {args.cities} found in {cities_dir}")
            raise SystemExit(1)

    # ── City roles (global split only) ────────────────────────────────────────
    city_roles: dict = {}
    if args.split_mode == "global":
        if args.label_source != "gpkg":
            parser.error(
                "--split-mode global requires --label-source gpkg: the "
                "culture-10 assignment lives in the split GeoPackage's "
                "`dataset` column and a label raster does not carry it."
            )
        # Weight the val-inner choice by how many tiles a city actually has, so
        # stratification cannot hand back a one-tile city like Salvador.
        weights = {
            d.name: sum(1 for _ in (d / args.output_name[0] / args.year).rglob("*.npy"))
            for d in city_dirs
        }
        city_roles = assign_city_roles(
            [d.name for d in city_dirs],
            n_val_inner=args.val_inner_cities,
            seed=args.seed,
            city_weights=weights,
        )

    # ── Build item lists ──────────────────────────────────────────────────────
    if args.label_tif_dir is not None:
        args.label_source = "tif"   # pseudo rasters are tifs

    all_items: list = []
    split_map: dict = {}
    # Shared across cities so patch UIDs are unique run-wide: a bare patch_id
    # restarts at 000000 in each of So2Sat's three original splits.
    uid_registry: dict = {}
    split_kw = dict(
        split_mode=args.split_mode, buffer_km=args.buffer_km,
        min_labelled_frac=args.min_labelled_frac, uid_registry=uid_registry,
    )
    eval_items: list | None = [] if args.label_tif_dir is not None else None
    for city_dir in city_dirs:
        items, sm = build_city_tile_items(
            city_dir, args.output_name[0], args.year,
            args.label_source, args.label_col,
            label_tif_dir=args.label_tif_dir,
            city_role=city_roles.get(city_dir.name), **split_kw,
        )
        all_items.extend(items)
        split_map.update(sm)
        if eval_items is not None:
            # val/test evaluated against ground-truth gpkg labels, not pseudo
            gt_items, _ = build_city_tile_items(
                city_dir, args.output_name[0], args.year, "gpkg", args.label_col,
                city_role=city_roles.get(city_dir.name), **split_kw,
            )
            eval_items.extend(gt_items)

    if not all_items:
        logger.error("No items found. Check --cities-dir, --output-name, --year.")
        raise SystemExit(1)

    # Restrict to ground both arms can stand on, BEFORE fusion/stats/weights so
    # every downstream quantity is computed on the same tile set.
    if args.require_embeddings:
        require = [n for n in args.require_embeddings if n != args.output_name[0]]
        if require:
            before = len(all_items)
            all_items, n_drop, per_city = _restrict_to_common_tiles(
                all_items, require, args.year)
            if eval_items is not None:
                eval_items, _, _ = _restrict_to_common_tiles(
                    eval_items, require, args.year)
            worst = sorted(per_city.items(), key=lambda kv: -kv[1])[:5]
            logger.info(
                f"Common-tile restriction vs {', '.join(require)}: "
                f"{len(all_items)}/{before} tiles kept ({n_drop} dropped)"
                + (f" — worst: " + ", ".join(f"{c} {n}" for c, n in worst) if worst else "")
            )
            if not all_items:
                logger.error(
                    "--require-embeddings removed every tile. Check the "
                    "output-name spelling and that those folders are extracted."
                )
                raise SystemExit(1)
            # split_map is keyed by path and simply over-covers now; the
            # DataModule filters by membership, so stale keys are inert.

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
    channel_mean = channel_std = None
    if args.normalize == "channel":
        train_items = [
            it for it in all_items
            if split_map.get(it[0][0] if isinstance(it[0], tuple) else it[0]) == "train"
        ]
        if not train_items:
            parser.error("--normalize channel needs train-split tiles; none were found.")
        nodata_pred = get_nodata_predicate(args.embedding_name)
        # Cached: the estimate is a single-threaded pass over 400 cold tiles and
        # measured ~8 minutes, which every run of a sweep would otherwise repay
        # for an identical answer.
        cache = grid_stats_cache_path(
            args.output_dir, args.output_name, args.year, args.embedding_name,
            train_items, args.stats_sample, args.seed,
        )
        channel_mean, channel_std = compute_grid_channel_stats(
            train_items, dequantize_fn=dequantize_fn,
            nodata_predicate=nodata_pred,
            n_sample=args.stats_sample, seed=args.seed,
            cache_path=cache,
        )
        if len(channel_mean) != in_channels:
            parser.error(
                f"Channel statistics have {len(channel_mean)} channels but the "
                f"model expects {in_channels}. This usually means the "
                f"dequantize path disagrees with detect_in_channels."
            )
    else:
        logger.warning(
            "--normalize none: inputs are fed unnormalised. This is the "
            "historical segmentation path, not the patch pipeline's default."
        )

    class_weights = _pixel_class_weights(
        all_items, split_map, args.class_weights, args.num_classes,
    ) if args.class_weights != "none" else None
    task = LCZUNetModule(
        model, args.num_classes,
        lr=args.lr, weight_decay=args.weight_decay,
        dice_weight=args.dice_weight, max_epochs=args.max_epochs,
        class_weights=class_weights, label_smoothing=args.label_smoothing,
        monitor=args.monitor,
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
        noise_sigma=args.noise_sigma,
        noise_prob=args.noise_prob,
        erode_px=args.erode_px,
        emit_patch_uids=(args.label_source == "gpkg" or eval_items is not None),
        normalize=args.normalize,
        channel_mean=channel_mean,
        channel_std=channel_std,
        nodata_predicate=get_nodata_predicate(args.embedding_name),
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    city_names = [d.name for d in city_dirs]
    run_cfg = dict(
        task="segmentation",
        embedding="+".join(args.output_name),
        # Provenance on every run (Task 1.5.3, guard 3): `embedding_name` alone
        # does not say which Tessera archive a number came from.
        **provenance(args.embedding_name),
        cities=city_names,
        year=args.year,
        label_source=args.label_source,
        label_tif_dir=str(args.label_tif_dir) if args.label_tif_dir else None,
        aux_channel_dropout=aux_dropout,
        family=args.family,
        preset=args.preset,
        in_channels=in_channels,
        num_classes=args.num_classes,
        noise_sigma=args.noise_sigma,
        noise_prob=args.noise_prob,
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
            # The stats travel inside the checkpoint: infer_roi has to
            # reproduce them exactly or every map it makes is wrong. Provenance
            # rides along too (Task 1.5.3, guard 2).
            norm_meta={
                "normalize": args.normalize,
                "channel_mean": channel_mean,
                "channel_std": channel_std,
                **provenance(args.embedding_name),
                "year": args.year,
            },
            warmup_epochs=args.warmup_epochs,
        )
    logger.info(f"Best checkpoint: {ckpt_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    datamodule.setup()   # re-create test dataloader after training
    results = evaluate_segmentation(
        task, datamodule.test_dataloader(), device, args.num_classes,
        run_dir, _run_label, use_wandb=not args.no_wandb,
        tta=args.tta,
    ) or {}

    # Patch-level aggregation is the headline: it is the only level that can be
    # compared to the patch-classification ladder. Pixel-level mIoU is a masked
    # metric and is not comparable to anything published on dense maps.
    #
    # Naming: unsuffixed keys are PIXEL level (the historical segmentation keys,
    # kept so existing WandB runs stay comparable), `_patch` is this
    # aggregation and `_100m` the mode-pooled coarse suite. The plan asked for
    # an explicit `_pixel` suffix; suffixing only the four new LCZ metrics while
    # test_acc/test_kappa stayed bare would have been worse than either
    # convention on its own.
    uid_to_label, uid_to_key = _uid_tables(uid_registry, all_items, eval_items)
    patch_results = evaluate_segmentation_as_patches(
        task, datamodule.test_dataloader(), device, args.num_classes,
        run_dir, _run_label, use_wandb=not args.no_wandb,
        uid_to_label=uid_to_label, uid_to_key=uid_to_key, tta=args.tta,
    )
    if patch_results:
        results.update(patch_results)

    # The culture cities' validation patches. Scored and cached, never used for
    # selection: ensemble_stacking.py --city-holdout fits its combiner weights
    # on exactly these, and without this pass the LOCO protocol has no
    # segmentation probabilities to consume.
    culture_loader = datamodule.culture_val_dataloader()
    if culture_loader is not None:
        logger.info("Caching culture-city validation probs for the LOCO combiner …")
        evaluate_segmentation_as_patches(
            task, culture_loader, device, args.num_classes,
            run_dir, _run_label, use_wandb=False,
            uid_to_label=uid_to_label, uid_to_key=uid_to_key, tta=args.tta,
            metric_suffix="_culture_val", dataset_filter="validation",
        )

    results["split_mode"] = args.split_mode
    if city_roles:
        results["val_inner_cities"] = sorted(
            c for c, r in city_roles.items() if r == "val_inner"
        )
    save_metrics_json(results, run_dir)

    if not args.no_wandb and wandb.run:
        wandb.finish()

    # ── Per-city inference raster + map ───────────────────────────────────────
    if fused:
        logger.warning("Fused multi-source model: full-ROI inference from raw "
                       "tiles is not supported yet — skipping city maps.")
        logger.info(f"Run complete. Outputs in {run_dir}")
        return
    if args.no_inference:
        logger.info(f"--no-inference: skipping city rasters. Outputs in {run_dir}")
        return
    logger.info("Running full-ROI inference …")
    run_city_inference(
        task.model, args.family, "segmentation", args.preset,
        city_dirs, run_dir, device, dequantize_fn,
        embedding_name=args.embedding_name,
        embedding_dir=args.embedding_dir,
        num_classes=args.num_classes,
        normalize=(None if channel_mean is None
                   else (channel_mean, channel_std)),
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        year=args.year,
        margin_m=args.margin_m,
    )

    logger.info(f"Run complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
