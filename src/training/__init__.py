"""Shared training library: task modules, generic loop, evaluation, augmentation."""

from training.augment import augment_batch, augment_images
from training.evaluate import (
    evaluate_classification,
    evaluate_segmentation,
    save_confusion_matrix,
)
from training.loop import run_training_loop
from training.tasks import LCZResNetModule, LCZUNetModule, MulticlassDiceLoss

__all__ = [
    "augment_batch",
    "augment_images",
    "evaluate_classification",
    "evaluate_segmentation",
    "save_confusion_matrix",
    "run_training_loop",
    "LCZResNetModule",
    "LCZUNetModule",
    "MulticlassDiceLoss",
]
