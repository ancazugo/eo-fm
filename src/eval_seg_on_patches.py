"""Evaluate a segmentation checkpoint on the So2Sat patch benchmark.

Runs a trained segmentation model over the extracted So2Sat test patches
(the exact same 32×32 npy inputs the patch classifiers see), mean-pools the
per-pixel logits into one prediction per patch, and reports the standard
classification metric suite (OA, macro acc, F1, Cohen's kappa, confusion
matrix). This makes segmentation models directly comparable to the
patch-classification kappa numbers.

Caveat: the model sees only the 32×32 patch itself — none of the surrounding
context it had during tile training — so this is a conservative estimate.

Example (global split, comparable to the patch_classification benchmark):
    python src/eval_seg_on_patches.py \\
        --checkpoint <run_dir>/unet_large_...-best.pt \\
        --family unet --preset large \\
        --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \\
        --output-name GeoTessera_v1.1_global --year 2017 \\
        --embedding-name tesserav1.1_global --global-split \\
        --output-dir data/seg_patch_eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from datasets.registry import EMBEDDING_REGISTRY
from datasets.so2sat import PatchDataset, build_so2sat_items
from models import build_model, families_for
from training.evaluate import evaluate_classification
from training.tasks import LCZResNetModule
from utils.cli import add_eval_args
from utils.runtime import detect_in_channels, resolve_dequantize, resolve_device


class PatchPoolHead(nn.Module):
    """Adapter: (B, C, H, W) seg logits → (B, C) patch logits by mean pooling.

    zero_from: optionally zero input channels >= this index before the forward
    (robustness probe: how does a fused model behave without its aux modality?).
    """

    def __init__(self, seg_model: nn.Module, zero_from: int | None = None) -> None:
        super().__init__()
        self.seg_model = seg_model
        self.zero_from = zero_from

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_from is not None:
            x = x.clone()
            x[:, self.zero_from:] = 0.0
        return self.seg_model(x).mean(dim=(-2, -1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch-level benchmark evaluation of a segmentation checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--family", choices=families_for("segmentation"), default="unet")
    parser.add_argument("--preset", default="large",
                        choices=["nano", "small", "base", "medium", "large"])
    parser.add_argument("--bottleneck-dropout", type=float, default=0.3)
    parser.add_argument("--so2sat-dir", required=True, type=Path)
    parser.add_argument("--output-name", required=True, nargs="+",
                        help="Embedding output name(s); multiple = channel fusion "
                             "(only patches present in every source are used).")
    parser.add_argument("--year", required=True)
    parser.add_argument("--zero-aux-from", type=int, default=None,
                        help="Zero input channels >= this index before the forward "
                             "(fused-model robustness probe, e.g. 128).")
    parser.add_argument("--embedding-name", required=True, choices=sorted(EMBEDDING_REGISTRY))
    parser.add_argument("--global-split", action="store_true",
                        help="Use the original So2Sat train/val/test split (global GPKG).")
    parser.add_argument("--global-gpkg", type=Path, default=None)
    parser.add_argument("--orig-test", action="store_true",
                        help="Hybrid mode: original So2Sat testing patches as test set.")
    parser.add_argument("--cities-dir", type=Path, default=None)
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--label-col", default="LCZ_class")
    add_eval_args(parser, batch_size=512)
    args = parser.parse_args()

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    output_name = args.output_name if len(args.output_name) > 1 else args.output_name[0]
    all_items, _ = build_so2sat_items(
        args.so2sat_dir, output_name, args.year,
        global_split=args.global_split, global_gpkg=args.global_gpkg,
        cities_dir=args.cities_dir, cities=args.cities,
        label_col=args.label_col, orig_test=args.orig_test,
    )
    test_items = [it for it in all_items if it[2] == "test"]
    if not test_items:
        raise SystemExit("No test items found.")
    logger.info(f"Test patches: {len(test_items)}")

    dequantize_fn, override = resolve_dequantize(args.embedding_name)
    first = test_items[0][0]
    if isinstance(first, tuple):
        in_channels = detect_in_channels(first[0], override) + sum(
            detect_in_channels(p) for p in first[1:])
        # fusion: dequantize applies to source 0 only
        dequantize_fn = [dequantize_fn] + [None] * (len(first) - 1)
        logger.info(f"Fused in_channels = {in_channels}")
    else:
        in_channels = detect_in_channels(first, override)
    seg_model = build_model(
        args.family, args.preset,
        in_channels=in_channels, num_classes=args.num_classes,
        bottleneck_dropout=args.bottleneck_dropout,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    seg_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    task = LCZResNetModule(PatchPoolHead(seg_model, zero_from=args.zero_aux_from),
                           num_classes=args.num_classes)
    task = task.to(device).eval()
    logger.info(f"Loaded {args.family}/{args.preset} from {args.checkpoint}")

    ds = PatchDataset(test_items, args.patch_size, dequantize_fn=dequantize_fn)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate_classification(
        task, loader, device, args.num_classes,
        args.output_dir, run_label="seg_patch_eval", use_wandb=False, tta=args.tta,
    )
    if results:
        logger.info("Patch-benchmark results (comparable to patch_classification kappa):")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
