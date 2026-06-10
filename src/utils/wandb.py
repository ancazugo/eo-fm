"""Weights & Biases logging utilities."""

import re
from pathlib import Path

import numpy as np
import wandb
from loguru import logger


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    key: str = "confusion_matrix",
) -> None:
    """Log a confusion matrix to the active WandB run.

    Args:
        y_true: Ground truth labels (1-based LCZ class IDs).
        y_pred: Predicted labels (1-based LCZ class IDs).
        class_names: Optional list of class names for axis labels.
        key: WandB log key (use different keys to log train/test separately).
    """
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is None:
        from utils.constants import lcz_dict
        # wandb.plot.confusion_matrix indexes class_names by label value, so the list
        # must be padded so that class_names[lbl] is valid for every lbl in y_true/preds.
        max_label = max(labels) if labels else 0
        class_names = [""] * (max_label + 1)
        for lbl in labels:
            class_names[lbl] = lcz_dict.get(lbl, {}).get("name", str(lbl))

    wandb.log({
        key: wandb.plot.confusion_matrix(
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=class_names,
        )
    })
    logger.info(f"Logged confusion matrix '{key}' to WandB ({len(labels)} classes)")


def log_per_class_metrics(
    per_class_acc: np.ndarray,
    per_class_f1: np.ndarray,
    num_classes: int,
    prefix: str = "test",
) -> None:
    """Log per-class accuracy and F1 as a WandB Table."""
    from utils.constants import lcz_dict
    columns = ["class_id", "class_name", f"{prefix}_acc", f"{prefix}_f1"]
    data = []
    for cls_idx in range(num_classes):
        cls_id = cls_idx + 1
        cls_name = lcz_dict.get(cls_id, {}).get("name", str(cls_id))
        data.append([cls_id, cls_name, float(per_class_acc[cls_idx]), float(per_class_f1[cls_idx])])
    wandb.log({f"{prefix}_per_class_metrics": wandb.Table(columns=columns, data=data)})
    logger.info(f"Logged per-class metrics table '{prefix}_per_class_metrics'")


def log_prediction_raster(raster_path: str | Path) -> None:
    """Log a prediction GeoTIFF as a WandB artifact.

    Args:
        raster_path: Path to the GeoTIFF file to log.
    """
    raster_path = Path(raster_path)
    # WandB artifact names only allow [a-zA-Z0-9_\-.]; replace '=' with '-'
    artifact_name = re.sub(r"[^a-zA-Z0-9_\-.]", "-", raster_path.stem)
    artifact = wandb.Artifact(
        name=artifact_name,
        type="prediction_raster",
    )
    artifact.add_file(str(raster_path))
    wandb.log_artifact(artifact)
    logger.info(f"Logged prediction raster artifact: {raster_path.name}")
