"""Shared training library: task modules, generic loop, evaluation, augmentation."""

from training.augment import augment_batch, augment_images
from training.evaluate import (
    dense_confusion_matrix,
    evaluate_classification,
    evaluate_segmentation,
    evaluate_segmentation_as_patches,
    per_city_metrics,
    predict_probs,
    save_confusion_matrix,
    save_metrics_json,
)
from training.lcz_metrics import (
    kappa_weighted,
    lcz_metrics_from_cm,
    load_similarity_matrix,
    oa_weighted,
)
from training.loop import run_training_loop
from training.tasks import LCZResNetModule, LCZUNetModule, MulticlassDiceLoss

__all__ = [
    "augment_batch",
    "augment_images",
    "dense_confusion_matrix",
    "evaluate_classification",
    "evaluate_segmentation",
    "evaluate_segmentation_as_patches",
    "kappa_weighted",
    "lcz_metrics_from_cm",
    "load_similarity_matrix",
    "oa_weighted",
    "per_city_metrics",
    "predict_probs",
    "save_confusion_matrix",
    "save_metrics_json",
    "run_training_loop",
    "LCZResNetModule",
    "LCZUNetModule",
    "MulticlassDiceLoss",
]
