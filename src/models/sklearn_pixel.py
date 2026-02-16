"""Functional sklearn pixel classifier: extract pixels from torchgeo datasets, train, evaluate."""

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from loguru import logger
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from torchgeo.datasets.geo import GeoDataset
from torchgeo.samplers import GridGeoSampler, Units

from conf import SklearnConfig


def extract_pixels_from_dataset(
    dataset: GeoDataset,
    patch_size: float = 256,
    stride: float = 256,
    n_samples_per_class: int | None = None,
    seed: int = 411,
    roi: Any | None = None,
    toi: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pixel-level features and labels from a torchgeo IntersectionDataset.

    Iterates patches via GridGeoSampler, flattens to (N, C) features + (N,) labels.

    Args:
        dataset: IntersectionDataset with "image" (embeddings) and "mask" (labels) keys.
        patch_size: Patch size in pixels for the GridGeoSampler.
        stride: Stride in pixels for the GridGeoSampler.
        n_samples_per_class: If set, subsample to this many pixels per class.
        seed: Random seed for subsampling.
        roi: Optional Shapely Polygon to restrict sampling spatially.
        toi: Optional pd.Interval to restrict sampling temporally.

    Returns:
        Tuple of (X, y) where X is (N, C) features and y is (N,) integer labels.
    """
    sampler = GridGeoSampler(dataset, size=patch_size, stride=stride, units=Units.PIXELS, roi=roi, toi=toi)

    all_features = []
    all_labels = []

    for bbox in sampler:
        sample = dataset[bbox]
        image = sample["image"]  # (C, H, W)
        mask = sample["mask"]  # (1, H, W) or (H, W)

        if isinstance(image, torch.Tensor):
            image = image.numpy()
            mask = mask.numpy()

        if mask.ndim == 3:
            mask = mask[0]  # (H, W)

        c, h, w = image.shape
        features = image.reshape(c, -1).T  # (H*W, C)
        labels = mask.reshape(-1)  # (H*W,)

        # Filter out nodata (0 or NaN)
        valid = (labels > 0) & ~np.isnan(labels) & ~np.any(np.isnan(features), axis=1)
        if valid.sum() > 0:
            all_features.append(features[valid])
            all_labels.append(labels[valid].astype(int))

    if not all_features:
        raise ValueError("No valid pixels found in dataset.")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    logger.info(f"Extracted {len(y)} valid pixels with {X.shape[1]} channels")

    if n_samples_per_class is not None:
        X, y = _subsample_per_class(X, y, n_samples_per_class, seed)

    return X, y


def _subsample_per_class(
    X: np.ndarray, y: np.ndarray, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to at most n_samples per class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    selected_X = []
    selected_y = []

    for cls in classes:
        mask = y == cls
        indices = np.where(mask)[0]
        n = min(n_samples, len(indices))
        chosen = rng.choice(indices, size=n, replace=False)
        selected_X.append(X[chosen])
        selected_y.append(y[chosen])

    X_sub = np.concatenate(selected_X, axis=0)
    y_sub = np.concatenate(selected_y, axis=0)
    logger.info(f"Subsampled to {len(y_sub)} pixels ({n_samples} max per class)")
    return X_sub, y_sub


def build_sklearn_classifier(config: SklearnConfig) -> ClassifierMixin:
    """Build a sklearn classifier from config."""
    if config.classifier == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=config.hidden_layer_sizes,
            alpha=config.alpha,
            learning_rate_init=config.learning_rate_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
    elif config.classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
        )
    else:
        raise ValueError(f"Unknown classifier: {config.classifier}")


def train_sklearn_classifier(
    dataset: GeoDataset,
    config: SklearnConfig,
    patch_size: float = 256,
    stride: float = 256,
    output_dir: Path | None = None,
    roi: Any | None = None,
    toi: Any | None = None,
) -> dict[str, Any]:
    """Extract pixels, run cross-validation, train final classifier, evaluate, and optionally save.

    Args:
        dataset: IntersectionDataset with "image" and "mask" keys.
        config: Sklearn training configuration.
        patch_size: Patch size for pixel extraction.
        stride: Stride for pixel extraction.
        output_dir: If set, save fitted model as joblib file in this directory.
        roi: Optional Shapely Polygon to restrict sampling spatially.
        toi: Optional pd.Interval to restrict sampling temporally.

    Returns:
        Dict with keys: classifier, y_test, y_pred, metrics, cv_results.
    """
    X, y = extract_pixels_from_dataset(
        dataset,
        patch_size=patch_size,
        stride=stride,
        n_samples_per_class=config.n_samples_per_class,
        seed=config.random_state,
        roi=roi,
        toi=toi,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    # Cross-validation on training set
    cv_classifier = build_sklearn_classifier(config)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    logger.info(f"Running {config.cv_folds}-fold cross-validation on {len(y_train)} training samples...")
    cv_results = cross_validate(cv_classifier, X_train, y_train, cv=cv, scoring=scoring)

    cv_summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        scores = cv_results[key]
        cv_summary[f"cv_{metric}_mean"] = float(np.mean(scores))
        cv_summary[f"cv_{metric}_std"] = float(np.std(scores))
        cv_summary[f"cv_{metric}_per_fold"] = scores.tolist()
        logger.info(f"CV {metric}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # Train final model on full training set and evaluate on held-out test set
    classifier = build_sklearn_classifier(config)
    logger.info(f"Training final {config.classifier} on {len(y_train)} samples...")
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    logger.info(f"Test metrics: {metrics}")
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    # Save model
    model_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"{config.classifier}_{timestamp}.joblib"
        joblib.dump(classifier, model_path)
        logger.info(f"Model saved to {model_path}")

    return {
        "classifier": classifier,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "cv_results": cv_summary,
        "model_path": model_path,
    }
