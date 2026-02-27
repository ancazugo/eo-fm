"""Functional sklearn pixel classifier: extract pixels from torchgeo datasets, train, evaluate."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from loguru import logger
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import rasterio
from rasterio.transform import from_bounds
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from torchgeo.datasets.geo import GeoDataset
from torchgeo.samplers import GridGeoSampler, Units

from conf import SklearnConfig


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def _split_per_class(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    n_samples_per_class: int | None = None,
    seed: int = 411,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (X, y) per class to maintain balanced class representation.

    For each class, independently splits pixels into train/test and subsamples.
    This ensures every class appears in both splits with equal representation
    regardless of class frequency in the original data.

    Args:
        X: Feature matrix (N, C).
        y: Label vector (N,).
        test_size: Fraction of samples per class assigned to the test set.
        n_samples_per_class: Maximum pixels per class per split after splitting.
        seed: Random seed.

    Returns:
        (X_train, y_train, X_test, y_test) with per-class balanced splits.
    """
    rng = np.random.default_rng(seed)
    X_trains, y_trains = [], []
    X_tests, y_tests = [], []

    for cls in np.unique(y):
        mask = y == cls
        X_cls, y_cls = X[mask], y[mask]
        n = len(X_cls)
        if n < 2:
            continue

        n_test = max(1, int(round(n * test_size)))
        if n - n_test < 1:
            continue

        idx = rng.permutation(n)
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        Xtr, ytr = X_cls[train_idx], y_cls[train_idx]
        Xte, yte = X_cls[test_idx], y_cls[test_idx]

        if n_samples_per_class is not None:
            n_tr = min(n_samples_per_class, len(Xtr))
            n_te = min(n_samples_per_class, len(Xte))
            Xtr = Xtr[rng.choice(len(Xtr), size=n_tr, replace=False)]
            ytr = np.full(n_tr, int(cls))
            Xte = Xte[rng.choice(len(Xte), size=n_te, replace=False)]
            yte = np.full(n_te, int(cls))

        X_trains.append(Xtr)
        y_trains.append(ytr)
        X_tests.append(Xte)
        y_tests.append(yte)

    if not X_trains:
        raise ValueError("Not enough samples to split per class.")

    X_train = np.concatenate(X_trains)
    y_train = np.concatenate(y_trains)
    X_test = np.concatenate(X_tests)
    y_test = np.concatenate(y_tests)

    logger.info(
        f"Per-class split → train: {len(y_train)} px "
        f"({len(np.unique(y_train))} classes), "
        f"test: {len(y_test)} px ({len(np.unique(y_test))} classes)"
    )
    return X_train, y_train, X_test, y_test


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


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_pixels_from_vector_labels(
    embedding_ds: GeoDataset,
    label_gdf,  # GeoDataFrame from VectorPatchLabelDataset.index
    test_size: float = 0.3,
    n_samples_per_class: int | None = None,
    seed: int = 411,
    toi: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract pixel features by iterating directly over labeled polygons.

    Splits polygons **per class** into train/test BEFORE extracting pixels,
    so all pixels from one polygon go to exactly one split — no leakage.

    Much faster than GridGeoSampler over the full area when labels are sparse
    vector patches (e.g. So2Sat-LCZ42).

    Args:
        embedding_ds: ZarrGeoDataset or similar embedding dataset.
        label_gdf: GeoDataFrame from VectorPatchLabelDataset.index with a
            'label' column and polygon geometries in the embedding's CRS.
        test_size: Fraction of polygons per class assigned to the test split.
        n_samples_per_class: Maximum pixels per class per split after extraction.
        seed: Random seed.
        toi: Optional pd.Interval to restrict by time (currently unused for
            tessera which uses a 1900-2100 sentinel time range).

    Returns:
        (X_train, y_train, X_test, y_test) arrays with polygon-level splits.
    """
    rng = np.random.default_rng(seed)
    t_slice = embedding_ds.bounds[2]

    X_trains, y_trains = [], []
    X_tests, y_tests = [], []
    n_ok, n_skip = 0, 0

    classes = sorted(int(c) for c in label_gdf["label"].unique() if c != 0)

    for cls_val in classes:
        cls_rows = label_gdf[label_gdf["label"] == cls_val]
        n_polys = len(cls_rows)
        if n_polys < 2:
            logger.warning(f"Class {cls_val}: only {n_polys} polygon(s), skipping")
            continue

        # Polygon-level split for this class
        poly_idx = rng.permutation(n_polys)
        n_test = max(1, int(round(n_polys * test_size)))
        test_poly_idx = poly_idx[:n_test]
        train_poly_idx = poly_idx[n_test:]

        def _extract_polys(indices):
            feats, lbls = [], []
            for i in indices:
                row = cls_rows.iloc[int(i)]
                minx, miny, maxx, maxy = row.geometry.bounds
                query = (slice(minx, maxx), slice(miny, maxy), t_slice)
                try:
                    sample = embedding_ds[query]
                except Exception:
                    nonlocal n_skip
                    n_skip += 1
                    return feats, lbls  # bail on first error for this polygon
                image = sample["image"]
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                c, h, w = image.shape
                f = image.reshape(c, -1).T  # (H*W, C)
                valid = ~np.any(np.isnan(f), axis=1)
                if valid.sum() > 0:
                    feats.append(f[valid])
                    lbls.append(np.full(int(valid.sum()), cls_val))
                    nonlocal n_ok
                    n_ok += 1
                else:
                    n_skip += 1
            return feats, lbls

        tr_feats, tr_lbls = _extract_polys(train_poly_idx)
        te_feats, te_lbls = _extract_polys(test_poly_idx)

        if not tr_feats or not te_feats:
            logger.warning(f"Class {cls_val}: insufficient valid polygons in one split, skipping")
            continue

        Xtr = np.concatenate(tr_feats)
        ytr = np.concatenate(tr_lbls)
        Xte = np.concatenate(te_feats)
        yte = np.concatenate(te_lbls)

        # Subsample within each split independently
        if n_samples_per_class is not None:
            n_tr = min(n_samples_per_class, len(Xtr))
            n_te = min(n_samples_per_class, len(Xte))
            Xtr = Xtr[rng.choice(len(Xtr), size=n_tr, replace=False)]
            ytr = np.full(n_tr, cls_val)
            Xte = Xte[rng.choice(len(Xte), size=n_te, replace=False)]
            yte = np.full(n_te, cls_val)

        X_trains.append(Xtr)
        y_trains.append(ytr)
        X_tests.append(Xte)
        y_tests.append(yte)

    if not X_trains:
        raise ValueError("No valid pixels found in labeled polygons.")

    X_train = np.concatenate(X_trains)
    y_train = np.concatenate(y_trains)
    X_test = np.concatenate(X_tests)
    y_test = np.concatenate(y_tests)

    logger.info(
        f"Extracted from {n_ok} polygons ({n_skip} skipped) — "
        f"train: {len(y_train)} px, test: {len(y_test)} px"
    )
    return X_train, y_train, X_test, y_test


def extract_pixels_from_dataset(
    dataset: GeoDataset,
    patch_size: float = 256,
    stride: float = 256,
    test_size: float = 0.3,
    n_samples_per_class: int | None = None,
    seed: int = 411,
    roi: Any | None = None,
    toi: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract pixel-level features and labels from a torchgeo IntersectionDataset.

    Iterates patches via GridGeoSampler, then applies a per-class train/test
    split with subsampling to keep classes balanced in both splits.

    Args:
        dataset: IntersectionDataset with "image" (embeddings) and "mask" (labels).
        patch_size: Patch size in pixels for the GridGeoSampler.
        stride: Stride in pixels for the GridGeoSampler.
        test_size: Fraction of pixels per class assigned to the test split.
        n_samples_per_class: Maximum pixels per class per split after splitting.
        seed: Random seed.
        roi: Optional Shapely Polygon to restrict sampling spatially.
        toi: Optional pd.Interval to restrict sampling temporally.

    Returns:
        (X_train, y_train, X_test, y_test) with per-class balanced splits.
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

        c, h, w = image.shape
        features = image.reshape(c, -1).T  # (H*W, C)

        # Handle patch-level labels from VectorPatchLabelDataset: shape (1, 1, 1)
        if mask.ndim == 3 and mask.shape[1] == 1 and mask.shape[2] == 1:
            labels = np.full(h * w, int(mask.flat[0]))
        elif mask.ndim == 3:
            labels = mask[0].reshape(-1)  # (H, W) → (H*W,)
        else:
            labels = mask.reshape(-1)

        valid = (labels > 0) & ~np.isnan(labels) & ~np.any(np.isnan(features), axis=1)
        if valid.sum() > 0:
            all_features.append(features[valid])
            all_labels.append(labels[valid].astype(int))

    if not all_features:
        raise ValueError("No valid pixels found in dataset.")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    logger.info(f"Extracted {len(y)} valid pixels with {X.shape[1]} channels")

    return _split_per_class(X, y, test_size=test_size, n_samples_per_class=n_samples_per_class, seed=seed)


# ---------------------------------------------------------------------------
# Classifier building + training
# ---------------------------------------------------------------------------

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
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier: {config.classifier}")


def train_sklearn_classifier(
    dataset: GeoDataset | None,
    config: SklearnConfig,
    patch_size: float = 256,
    stride: float = 256,
    output_dir: Path | None = None,
    roi: Any | None = None,
    toi: Any | None = None,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run cross-validation, train final classifier, evaluate, and optionally save.

    Accepts pre-split (X_train, y_train, X_test, y_test) from one of the
    extraction functions. When not provided, extracts from `dataset` using
    GridGeoSampler with per-class balanced splitting.

    Args:
        dataset: IntersectionDataset with "image" and "mask" keys.
            Ignored when pre-split arrays are provided.
        config: Sklearn training configuration.
        patch_size: Patch size for pixel extraction (raster labels only).
        stride: Stride for pixel extraction (raster labels only).
        output_dir: If set, save fitted model as joblib file here.
        roi: Optional Shapely Polygon to restrict sampling spatially.
        toi: Optional pd.Interval to restrict sampling temporally.
        X_train, y_train: Pre-split training features/labels.
        X_test, y_test: Pre-split test features/labels.

    Returns:
        Dict with keys: classifier, X_train, X_test, y_train, y_test,
        y_pred, metrics, cv_results, model_path.
    """
    if X_train is None or y_train is None:
        X_train, y_train, X_test, y_test = extract_pixels_from_dataset(
            dataset,
            patch_size=patch_size,
            stride=stride,
            test_size=config.test_size,
            n_samples_per_class=config.n_samples_per_class,
            seed=config.random_state,
            roi=roi,
            toi=toi,
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
        acc = metrics["accuracy"]
        model_path = output_dir / f"{config.classifier}-sklearn-acc={acc:.4f}.joblib"
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


# ---------------------------------------------------------------------------
# Full-area prediction
# ---------------------------------------------------------------------------

def _fill_tile_borders(
    raster: np.ndarray,
    tile_geoms,
    all_minx: float,
    all_maxy: float,
    out_h: int,
    out_w: int,
    res_x: float,
    res_y: float,
    trim_px: tuple[int, int, int, int],
) -> np.ndarray:
    """Replace border-contaminated pixels with nearest valid interior predictions.

    For each tile, pixels within ``trim_px`` of the tile edge are considered
    contaminated (the embedding model sees truncated context there).  Instead
    of zeroing them, this function fills them with the nearest-neighbour class
    from the tile interior using a Euclidean distance transform.  The result
    is a seamless, gap-free prediction map.

    Tessera contamination extents (measured): south ≈ 260 px, north ≈ 85 px,
    east ≈ 90 px, west ≈ 0 px.

    Args:
        raster: Output prediction array (out_h, out_w), modified in-place.
        tile_geoms: Iterable of Shapely geometries (one per tile).
        all_minx: West boundary of the output raster in dataset CRS units.
        all_maxy: North boundary of the output raster in dataset CRS units.
        out_h, out_w: Output raster dimensions in pixels.
        res_x, res_y: Pixel size in dataset CRS units (both positive).
        trim_px: (north, south, east, west) pixels to exclude near each tile edge.

    Returns:
        The (modified) raster array with border pixels replaced.
    """
    from scipy.ndimage import distance_transform_edt

    trim_north, trim_south, trim_east, trim_west = trim_px
    # valid_mask = True for interior pixels that are trusted
    valid_mask = np.zeros((out_h, out_w), dtype=bool)

    for geom in tile_geoms:
        minx, miny, maxx, maxy = geom.bounds
        int_minx = minx + trim_west  * res_x
        int_miny = miny + trim_south * res_y
        int_maxx = maxx - trim_east  * res_x
        int_maxy = maxy - trim_north * res_y

        if int_minx >= int_maxx or int_miny >= int_maxy:
            continue

        col0 = max(0, int(round((int_minx - all_minx) / res_x)))
        col1 = min(out_w, int(round((int_maxx - all_minx) / res_x)))
        row0 = max(0, int(round((all_maxy - int_maxy) / res_y)))
        row1 = min(out_h, int(round((all_maxy - int_miny) / res_y)))

        if col0 < col1 and row0 < row1:
            valid_mask[row0:row1, col0:col1] = True

    border_mask = ~valid_mask
    n_border = int(border_mask.sum())
    if n_border == 0:
        return raster

    # For each border pixel, find the nearest valid (interior) pixel and
    # copy its class.  distance_transform_edt with return_indices gives the
    # row/col of the nearest background (0 = valid) pixel for each foreground
    # (True = border) pixel.
    _, nn_idx = distance_transform_edt(border_mask, return_indices=True)
    raster[border_mask] = raster[nn_idx[0][border_mask], nn_idx[1][border_mask]]

    logger.info(
        f"Tile border fill: replaced {n_border:,} border pixels "
        f"({100 * n_border / valid_mask.size:.1f}% of raster) with nearest interior class "
        f"[trim N={trim_north} S={trim_south} E={trim_east} W={trim_west} px]"
    )
    return raster


def predict_sklearn_roi(
    dataset: GeoDataset,
    classifier: ClassifierMixin,
    patch_size: float = 256,
    stride: float = 256,
    roi: Any | None = None,
    toi: Any | None = None,
    output_path: str | Path = "prediction.tif",
    tile_border_trim: int = 0,
) -> dict[str, Any]:
    """Predict over the full ROI using a trained sklearn classifier and write a GeoTIFF.

    Args:
        dataset: GeoDataset (embedding only or intersection with labels).
        classifier: Fitted sklearn classifier.
        patch_size: Patch size in pixels for GridGeoSampler.
        stride: Stride in pixels for GridGeoSampler.
        roi: Optional Shapely Polygon to restrict sampling spatially.
        toi: Optional pd.Interval to restrict sampling temporally.
        output_path: Path to write the prediction GeoTIFF.
        tile_border_trim: Controls correction of tile-boundary artifacts.
            Tessera embeddings produce out-of-distribution features near tile
            edges (truncated model context), causing systematic misclassification.
            Pass an int N for symmetric trimming (all sides = N px) or a 4-tuple
            ``(north, south, east, west)`` for asymmetric trimming.  For tessera
            the recommended asymmetric values are ``(85, 260, 90, 0)`` — these
            cover the measured contamination extents.  Border pixels are NOT
            zeroed; instead they are replaced with the nearest valid prediction
            from the tile interior (seamless, gap-free output).  Set to 0 (default)
            to disable.  Only applied when ``dataset`` exposes a geopandas
            ``.index`` (i.e. ZarrGeoDataset).

    Returns:
        Dict with keys: y_true, y_pred, confusion_matrix, raster_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = GridGeoSampler(
        dataset, size=patch_size, stride=stride, units=Units.PIXELS, roi=roi, toi=toi,
    )

    all_preds = []
    all_truths = []
    patches = []  # (pred_patch, bbox) for raster assembly

    for bbox in sampler:
        sample = dataset[bbox]
        image = sample["image"]  # (C, H, W)
        mask = sample.get("mask")  # optional

        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        if mask is not None and mask.ndim == 3:
            mask = mask[0]  # (H, W)

        c, h, w = image.shape
        features = image.reshape(c, -1).T  # (H*W, C)

        pred_flat = classifier.predict(features)  # (H*W,)
        # Zero out pixels where the embedding has NaN values (nodata in source)
        pred_flat[np.any(np.isnan(features), axis=1)] = 0
        pred_patch = pred_flat.reshape(h, w).astype(np.uint8)
        patches.append((pred_patch, bbox))

        # Collect valid pixels for confusion matrix (only when labels are present)
        if mask is not None:
            labels = mask.reshape(-1)
            valid = (labels > 0) & ~np.isnan(labels) & ~np.any(np.isnan(features), axis=1)
            if valid.sum() > 0:
                all_truths.append(labels[valid].astype(int))
                all_preds.append(pred_flat[valid].astype(int))

    if not patches:
        raise ValueError("No patches found in ROI.")

    # Compute confusion matrix from all valid pixels
    y_true = np.concatenate(all_truths, axis=0) if all_truths else np.array([], dtype=int)
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=int)

    cm = None
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion matrix computed from {len(y_true)} valid pixels")

    def _bbox_bounds(b):
        """Return (minx, maxx, miny, maxy) from a sampler bbox.

        Supports both torchgeo BoundingBox (has .minx attribute) and
        GeoSlice tuple (slice_x, slice_y, slice_t) from torchgeo 0.9 API.
        """
        if hasattr(b, "minx"):
            return b.minx, b.maxx, b.miny, b.maxy
        # GeoSlice: (slice_x, slice_y, slice_t)
        return b[0].start, b[0].stop, b[1].start, b[1].stop

    # Assemble prediction raster — collect bounds from sampler bboxes
    all_minx = min(_bbox_bounds(b)[0] for _, b in patches)
    all_miny = min(_bbox_bounds(b)[2] for _, b in patches)
    all_maxx = max(_bbox_bounds(b)[1] for _, b in patches)
    all_maxy = max(_bbox_bounds(b)[3] for _, b in patches)

    # Determine pixel resolution from the first patch
    first_patch, first_bbox = patches[0]
    ph, pw = first_patch.shape
    fminx, fmaxx, fminy, fmaxy = _bbox_bounds(first_bbox)
    res_x = (fmaxx - fminx) / pw
    res_y = (fmaxy - fminy) / ph

    # Compute output raster dimensions
    out_w = int(round((all_maxx - all_minx) / res_x))
    out_h = int(round((all_maxy - all_miny) / res_y))
    output_raster = np.zeros((out_h, out_w), dtype=np.uint8)

    # Place each patch into the output raster
    for pred_patch, bbox in patches:
        bminx, _, _, bmaxy = _bbox_bounds(bbox)
        col_start = int(round((bminx - all_minx) / res_x))
        row_start = int(round((all_maxy - bmaxy) / res_y))  # raster origin is top-left
        ph, pw = pred_patch.shape
        # Clip to raster bounds
        row_end = min(row_start + ph, out_h)
        col_end = min(col_start + pw, out_w)
        patch_h = row_end - row_start
        patch_w = col_end - col_start
        if patch_h > 0 and patch_w > 0:
            output_raster[row_start:row_end, col_start:col_end] = pred_patch[:patch_h, :patch_w]

    # Determine CRS from the dataset
    crs = getattr(dataset, "crs", None)
    transform = from_bounds(all_minx, all_miny, all_maxx, all_maxy, out_w, out_h)

    # Fix border-contaminated pixels near tessera tile edges.
    if tile_border_trim and hasattr(dataset, "index"):
        trim = (
            (tile_border_trim,) * 4
            if isinstance(tile_border_trim, int)
            else tuple(tile_border_trim)
        )
        output_raster = _fill_tile_borders(
            output_raster, dataset.index.geometry,
            all_minx, all_maxy, out_h, out_w, res_x, res_y, trim,
        )

    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=out_h,
        width=out_w,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(output_raster, 1)

    logger.info(f"Prediction raster saved to {output_path} ({out_h}x{out_w})")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "raster_path": output_path,
    }
