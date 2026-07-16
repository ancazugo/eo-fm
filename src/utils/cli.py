"""Shared argparse helpers for the training pipelines.

patch_classification.py and semantic_segmentation.py share most of their CLI
surface; these helpers define the common arguments once. Each helper creates
an argument group and returns it so callers can append script-specific flags.

Deliberately NOT shared:
  * the Data groups — split modes and label sources diverge per pipeline;
  * ``--embedding-name`` — classification accepts several (channel fusion),
    segmentation exactly one;
  * ``--patch-size`` — same flag, different meaning per pipeline
    (classification: training patch resize, lives in the Model group;
    segmentation: sliding-window inference size, lives in the Inference group).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from models import families_for


def add_model_args(
    parser: argparse.ArgumentParser, pipeline: str, *, default_family: str
) -> argparse._ArgumentGroup:
    """Add the shared Model group (--family/--preset/--num-classes)."""
    g = parser.add_argument_group("Model")
    g.add_argument("--family", choices=families_for(pipeline), default=default_family,
                   help=f"Model family (default: {default_family}).")
    g.add_argument("--preset", choices=["nano", "small", "base", "medium", "large"],
                   default="large",
                   help="Size preset (default: large).")
    g.add_argument("--num-classes", type=int, default=17,
                   help="Number of output classes (default: 17).")
    return g


def add_training_args(
    parser: argparse.ArgumentParser, *, batch_size: int, seed: int = 42
) -> argparse._ArgumentGroup:
    """Add the shared Training group; per-pipeline defaults via keywords."""
    g = parser.add_argument_group("Training")
    g.add_argument("--batch-size", type=int, default=batch_size)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--max-epochs", type=int, default=50)
    g.add_argument("--early-stopping-patience", type=int, default=10)
    g.add_argument("--seed", type=int, default=seed)
    g.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return g


def add_logging_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the shared Logging group (output dir, WandB, checkpoint loading)."""
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
    return g


def add_inference_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the shared Inference group; callers append --embedding-name etc."""
    g = parser.add_argument_group("Inference")
    g.add_argument("--embedding-dir", required=True, type=Path,
                   help="Directory containing raw source embedding tiles (.zarr or .tif).")
    g.add_argument("--overlap", type=int, default=None,
                   help="Overlap between adjacent patches in pixels for inference "
                        "(default: patch_size // 2).")
    g.add_argument("--margin-m", type=float, default=200.0,
                   help="Extra metres clipped around city bbox per tile for edge context (default: 200).")
    return g


def resolve_overlap(args: argparse.Namespace) -> None:
    """Default --overlap to half the patch size (in place)."""
    if args.overlap is None:
        args.overlap = args.patch_size // 2


def add_eval_args(
    parser: argparse.ArgumentParser, *, batch_size: int, patch_size: int = 32
) -> argparse.ArgumentParser:
    """Common inference block of the evaluation/SSL scripts (ensemble_eval,
    tta_city_adapt, generate_pseudo_labels, eval_seg_on_patches):
    --num-classes, --patch-size, --batch-size, --num-workers, --tta,
    --accelerator, --output-dir. Per-script defaults come in as keywords.
    """
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--patch-size", type=int, default=patch_size)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation: average probs over flips/90° rotations.")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def parse_model_spec(spec: str) -> dict:
    """argparse type= for the --model spec used by the ensemble/TTA scripts.

    Format: ``OUTPUT_NAME,EMBEDDING_NAME,CHECKPOINT[,FAMILY,PRESET]``
    (family/preset default to resnet/small).
    """
    parts = spec.split(",")
    if len(parts) == 3:
        parts += ["resnet", "small"]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"Bad --model spec {spec!r}; expected "
            "OUTPUT_NAME,EMBEDDING_NAME,CHECKPOINT[,FAMILY,PRESET]"
        )
    # A fused model joins several sources with '+', e.g.
    # "GeoTessera_v1.1_global+AuxStruct,tesserav1.1_global+aux_struct,<ckpt>".
    output_names = parts[0].split("+")
    embedding_names = parts[1].split("+")
    if len(output_names) != len(embedding_names):
        raise argparse.ArgumentTypeError(
            f"Bad --model spec {spec!r}: output/embedding source counts differ"
        )
    return dict(
        output_names=output_names, embedding_names=embedding_names,
        name=parts[0], checkpoint=Path(parts[2]), family=parts[3], preset=parts[4],
    )
