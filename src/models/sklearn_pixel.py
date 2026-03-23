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


def split_label_gdf(
    gdf,
    val_size: float = 0.0,
    test_size: float = 0.3,
    seed: int = 411,
) -> tuple:
    """Stratified polygon-level split.

    Groups polygons by class label and assigns them to splits so every class
    is represented in every split. Uses integer positions (iloc) to handle
    geopandas GeoDataFrames with non-integer indices (e.g. pd.IntervalIndex).

    Args:
        gdf: GeoDataFrame with a 'label' column (e.g. VectorPatchLabelDataset.index).
        val_size: Fraction of polygons per class for the validation split.
            Pass 0.0 (default) to skip validation split (sklearn use-case).
        test_size: Fraction of polygons per class for the test split.
        seed: Random seed.

    Returns:
        ``(train_gdf, test_gdf)``          when val_size == 0
        ``(train_gdf, val_gdf, test_gdf)`` when val_size > 0
    """
    rng = np.random.default_rng(seed)
    labels_arr = gdf["label"].values
    classes = sorted(int(c) for c in np.unique(labels_arr) if c != 0)

    train_pos, val_pos, test_pos = [], [], []

    for cls_val in classes:
        cls_pos = np.where(labels_arr == cls_val)[0]
        n = len(cls_pos)
        if n < 2:
            train_pos.extend(cls_pos.tolist())
            continue

        shuffled = rng.permutation(n)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size))) if val_size > 0 else 0

        # Ensure at least 1 training sample
        while n - n_test - n_val < 1 and n_val > 0:
            n_val -= 1
        if n - n_test - n_val < 1:
            train_pos.extend(cls_pos.tolist())
            continue

        test_pos.extend(cls_pos[shuffled[:n_test]].tolist())
        val_pos.extend(cls_pos[shuffled[n_test:n_test + n_val]].tolist())
        train_pos.extend(cls_pos[shuffled[n_test + n_val:]].tolist())

    train_gdf = gdf.iloc[train_pos]
    test_gdf = gdf.iloc[test_pos]

    if val_size > 0:
        val_gdf = gdf.iloc[val_pos]
        logger.info(
            f"Polygon-level split: {len(train_gdf)} train, "
            f"{len(val_gdf)} val, {len(test_gdf)} test polygons"
        )
        return train_gdf, val_gdf, test_gdf

    logger.info(
        f"Polygon-level split: {len(train_gdf)} train polygons, {len(test_gdf)} test polygons"
    )
    return train_gdf, test_gdf


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
    """Extract pixel features with polygon-level train/test split.

    Polygons are pre-assigned to train or test via split_label_gdf() so no
    polygon ever contributes pixels to both splits. Within each split, all
    polygons for a class are merged via unary_union before extraction to avoid
    double-counting overlapping polygons.

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
    from shapely.ops import unary_union

    rng = np.random.default_rng(seed)
    t_slice = embedding_ds.bounds[2]

    train_gdf, test_gdf = split_label_gdf(label_gdf, val_size=0.0, test_size=test_size, seed=seed)

    def _extract(split_gdf, split_name):
        X_list, y_list = [], []
        n_ok = n_skip = 0
        classes = sorted(int(c) for c in split_gdf["label"].unique() if c != 0)

        for cls_val in classes:
            cls_rows = split_gdf[split_gdf["label"] == cls_val]
            merged = unary_union(cls_rows.geometry.values)
            components = list(merged.geoms) if hasattr(merged, "geoms") else [merged]

            all_feats = []
            for geom in components:
                minx, miny, maxx, maxy = geom.bounds
                query = (slice(minx, maxx), slice(miny, maxy), t_slice)
                try:
                    sample = embedding_ds[query]
                except Exception:
                    n_skip += 1
                    continue
                image = sample["image"]
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                c, h, w = image.shape
                f = image.reshape(c, -1).T  # (H*W, C)
                valid = ~np.any(np.isnan(f), axis=1)
                if valid.sum() > 0:
                    all_feats.append(f[valid])
                    n_ok += 1
                else:
                    n_skip += 1

            if not all_feats:
                logger.warning(f"{split_name} class {cls_val}: no valid pixels, skipping")
                continue

            X_cls = np.concatenate(all_feats)
            if n_samples_per_class is not None:
                n_sub = min(n_samples_per_class, len(X_cls))
                X_cls = X_cls[rng.choice(len(X_cls), size=n_sub, replace=False)]
            X_list.append(X_cls)
            y_list.append(np.full(len(X_cls), cls_val))

        return X_list, y_list, n_ok, n_skip

    X_trains, y_trains, n_ok_tr, n_skip_tr = _extract(train_gdf, "train")
    X_tests, y_tests, n_ok_te, n_skip_te = _extract(test_gdf, "test")

    if not X_trains:
        raise ValueError("No valid pixels found in train polygons.")
    if not X_tests:
        raise ValueError("No valid pixels found in test polygons.")

    X_train = np.concatenate(X_trains)
    y_train = np.concatenate(y_trains)
    X_test = np.concatenate(X_tests)
    y_test = np.concatenate(y_tests)

    logger.info(
        f"Extracted train: {n_ok_tr} components ({n_skip_tr} skipped) → {len(y_train)} px; "
        f"test: {n_ok_te} components ({n_skip_te} skipped) → {len(y_test)} px"
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
    elif config.classifier == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=-1,
        )
    elif config.classifier == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.lgbm_learning_rate,
            num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_state,
            verbose=-1,
        )
    elif config.classifier == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.xgb_learning_rate,
            max_depth=config.xgb_max_depth,
            n_jobs=-1,
            random_state=config.random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )
    elif config.classifier == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=config.logreg_C,
            max_iter=config.max_iter,
            random_state=config.random_state,
            n_jobs=-1,
            solver="saga",
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

    # XGBoost requires 0-indexed contiguous labels; remap with LabelEncoder.
    label_encoder = None
    if config.classifier == "xgboost":
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

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

    y_train_pred = classifier.predict(X_train)
    y_pred = classifier.predict(X_test)

    # Remap predictions back to original label space for XGBoost
    if label_encoder is not None:
        y_train = label_encoder.inverse_transform(y_train)
        y_test = label_encoder.inverse_transform(y_test)
        y_train_pred = label_encoder.inverse_transform(y_train_pred)
        y_pred = label_encoder.inverse_transform(y_pred)

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
        "y_train_pred": y_train_pred,
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
    """Replace border-contaminated pixels with nearest interior predictions, per tile.

    For each tile, pixels within ``trim_px`` of the tile edge are considered
    contaminated (the embedding model sees truncated context there).  Each tile
    fills its own border pixels using only its own interior predictions via a
    per-tile Euclidean distance transform.  Doing this per-tile (rather than
    globally) avoids cross-tile Voronoi banding artifacts that arise when the
    EDT finds interior pixels from a *different* tile.

    Tessera contamination extents (measured): south ≈ 260 px, north ≈ 85 px,
    east ≈ 90 px, west ≈ 0 px.
    Google AlphaEarth contamination extents (estimated): ~50 px on all sides.

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
    total_replaced = 0

    for geom in tile_geoms:
        minx, miny, maxx, maxy = geom.bounds

        # Convert tile extent to raster pixel coordinates
        tile_col0 = max(0, int(round((minx - all_minx) / res_x)))
        tile_col1 = min(out_w, int(round((maxx - all_minx) / res_x)))
        tile_row0 = max(0, int(round((all_maxy - maxy) / res_y)))
        tile_row1 = min(out_h, int(round((all_maxy - miny) / res_y)))

        if tile_col0 >= tile_col1 or tile_row0 >= tile_row1:
            continue

        tile_h = tile_row1 - tile_row0
        tile_w = tile_col1 - tile_col0

        # Interior bounds within this tile (clamp to tile dimensions)
        int_row0 = min(tile_h, trim_north)       # north trim from top
        int_row1 = max(0, tile_h - trim_south)   # south trim from bottom
        int_col0 = min(tile_w, trim_west)         # west trim from left
        int_col1 = max(0, tile_w - trim_east)     # east trim from right

        if int_row0 >= int_row1 or int_col0 >= int_col1:
            continue  # tile fully contaminated — skip

        # Interior mask within this tile (True = valid, not contaminated)
        interior = np.zeros((tile_h, tile_w), dtype=bool)
        interior[int_row0:int_row1, int_col0:int_col1] = True
        border = ~interior

        n_border_tile = int(border.sum())
        if n_border_tile == 0:
            continue

        # Per-tile EDT: find nearest interior pixel *within this tile* for each
        # border pixel.  Using only the tile's own interior avoids cross-tile
        # Voronoi banding that appears when the global EDT maps a border pixel
        # to an interior pixel from an adjacent tile.
        tile_slice = raster[tile_row0:tile_row1, tile_col0:tile_col1]
        _, nn_idx = distance_transform_edt(~interior, return_indices=True)
        tile_slice[border] = tile_slice[nn_idx[0][border], nn_idx[1][border]]

        total_replaced += n_border_tile

    logger.info(
        f"Tile border fill: replaced {total_replaced:,} border pixels "
        f"({100 * total_replaced / (out_h * out_w):.1f}% of raster) with nearest interior class "
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
    tile_border_trim: int | tuple[int, int, int, int] = 0,
    roi_edge_trim: int | tuple[int, int, int, int] = 0,
    gap_fill: bool = True,
) -> dict[str, Any]:
    """Predict over the full ROI using a trained sklearn classifier and write a GeoTIFF.

    Iterates non-overlapping patches via GridGeoSampler, predicts each patch,
    stitches them into a single raster, and saves as GeoTIFF.

    Args:
        dataset: GeoDataset (embedding only or intersection with labels).
        classifier: Fitted sklearn classifier.
        patch_size: Patch size in pixels for GridGeoSampler.
        stride: Stride in pixels for GridGeoSampler (default = patch_size → no overlap).
        roi: Optional Shapely Polygon to restrict sampling spatially.
        toi: Optional pd.Interval to restrict sampling temporally.
        output_path: Path to write the prediction GeoTIFF.
        tile_border_trim: Pixels to trim near each tile edge before nearest-interior
            fill. Pass a single int N for symmetric (N,N,N,N) or a
            (north, south, east, west) tuple. 0 disables the fill.
        roi_edge_trim: Pixels to zero at the outer boundary of the assembled raster
            before EDT gap-fill. Handles embedding contamination at the ROI's outer
            edges (not internal tile boundaries). Same format as tile_border_trim.

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
        # Zero out nodata pixels: NaN values or all-zero embeddings (outside tile bounds)
        nodata = np.any(np.isnan(features), axis=1) | np.all(features == 0, axis=1)
        pred_flat[nodata] = 0
        pred_patch = pred_flat.reshape(h, w).astype(np.uint8)
        patches.append((pred_patch, bbox))

        # Collect valid pixels for confusion matrix (only when labels are present)
        if mask is not None:
            labels = mask.reshape(-1)
            valid = (labels > 0) & ~np.isnan(labels) & ~nodata
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

    # Assemble prediction raster from sampler patches.
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

    # GridGeoSampler overshoots at north/east edges to fill a full patch_size
    # window. ZarrGeoDataset clips to tile extent and zero-pads at the bottom/
    # right of the returned tensor, so the overshot portion of the patch is zero.
    # These zeros land at the top/right of the assembled raster.  Trim any
    # all-zero rows/columns from the raster edges before gap-fill so EDT does
    # not propagate interior classes into a wide blank edge strip.
    row_has_data = np.any(output_raster > 0, axis=1)
    col_has_data = np.any(output_raster > 0, axis=0)
    if row_has_data.any() and col_has_data.any():
        r0 = int(np.argmax(row_has_data))
        r1 = int(len(row_has_data) - np.argmax(row_has_data[::-1]))
        c0 = int(np.argmax(col_has_data))
        c1 = int(len(col_has_data) - np.argmax(col_has_data[::-1]))
        if r0 > 0 or r1 < out_h or c0 > 0 or c1 < out_w:
            logger.info(
                f"Edge trim: removed {r0}N {out_h - r1}S {c0}W {out_w - c1}E all-zero "
                f"border rows/cols (GridGeoSampler overshoot)"
            )
            output_raster = output_raster[r0:r1, c0:c1]
            all_maxy -= r0 * res_y
            all_miny += (out_h - r1) * res_y
            all_minx += c0 * res_x
            all_maxx -= (out_w - c1) * res_x
            out_h, out_w = output_raster.shape

    # Apply tile-border fill if requested (handles embedding feature contamination at tile edges)
    if tile_border_trim:
        trim_px = tile_border_trim if isinstance(tile_border_trim, tuple) else (tile_border_trim,) * 4
        base = dataset.datasets[0] if hasattr(dataset, "datasets") else dataset
        if hasattr(base, "index") and hasattr(base.index, "geometry"):
            output_raster = _fill_tile_borders(
                output_raster, base.index.geometry,
                all_minx, all_maxy, out_h, out_w, res_x, res_y, trim_px,
            )
        else:
            logger.warning("tile_border_trim set but dataset has no GeoDataFrame index — skipping fill")

    # Zero the outer boundary of the raster (embedding context contamination at ROI edges).
    if roi_edge_trim:
        n, s, e, w = roi_edge_trim if isinstance(roi_edge_trim, tuple) else (roi_edge_trim,) * 4
        if n: output_raster[:n, :] = 0
        if s: output_raster[-s:, :] = 0
        if e: output_raster[:, -e:] = 0
        if w: output_raster[:, :w] = 0
        logger.info(f"ROI edge trim: zeroed outer N={n} S={s} E={e} W={w} px for EDT fill")

    # Fill any remaining zero (nodata) pixels with the nearest valid prediction.
    # Covers coverage gaps wider than tile_border_trim margins and areas with no tile
    # data (same behaviour as the torch UNet which always produces a prediction).
    if gap_fill:
        nodata_mask = output_raster == 0
        if nodata_mask.any():
            from scipy.ndimage import distance_transform_edt
            _, (iy, ix) = distance_transform_edt(nodata_mask, return_indices=True)
            n_filled = int(nodata_mask.sum())
            output_raster = output_raster[iy, ix]
            logger.info(f"Gap fill: replaced {n_filled:,} nodata pixels with nearest valid prediction")

    # Determine CRS from the dataset
    crs = getattr(dataset, "crs", None)
    transform = from_bounds(all_minx, all_miny, all_maxx, all_maxy, out_w, out_h)

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
