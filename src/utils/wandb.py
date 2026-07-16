"""Weights & Biases logging utilities."""

import numpy as np
import wandb
from loguru import logger


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
