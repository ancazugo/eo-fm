"""Typer CLI for the eo-fm pipeline."""

from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

app = typer.Typer(help="Earth Observation Foundation Model — LCZ classification pipeline")


def _make_run_dir(base_dir: Path, embedding: str, year: int | None, bbox: list[str], run_name: str) -> Path:
    """Build and create a unique output directory for a training run.

    Pattern: ``{base_dir}/{embedding}_{year}_{bbox}_{run_name}/``

    Args:
        base_dir: Root output directory.
        embedding: Embedding name (e.g. "tessera").
        year: Year filter, or None.
        bbox: List of "west,south,east,north" strings; first element used if present.
        run_name: WandB run name or timestamp fallback.

    Returns:
        Created Path.
    """
    year_str = str(year) if year is not None else "all"
    if bbox:
        w, s, e, n = (float(v) for v in bbox[0].split(","))
        bbox_str = f"W{w:.1f}_S{s:.1f}_E{e:.1f}_N{n:.1f}"
    else:
        bbox_str = "global"
    run_dir = base_dir / f"{embedding}_{year_str}_{bbox_str}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_tile_border_trim(value: str) -> int | tuple[int, int, int, int]:
    """Parse tile_border_trim from CLI string to int or (N,S,E,W) tuple."""
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 4:
        return tuple(parts)  # type: ignore[return-value]
    raise typer.BadParameter("--tile-border-trim must be a single int or 'N,S,E,W'")


def _resolve_tile_border_trim(value: str, embedding: str) -> int | tuple[int, int, int, int]:
    """Parse tile_border_trim, falling back to embedding-specific registry default when 0.

    Pass ``"0"`` (the CLI default) to auto-apply the registered trim for the
    given embedding (e.g. ``(85, 260, 90, 0)`` for tessera).  Any explicit
    non-zero value overrides the registry default.
    """
    trim = _parse_tile_border_trim(value)
    if not trim:
        from datasets.registry import EMBEDDING_REGISTRY
        default = EMBEDDING_REGISTRY.get(embedding, {}).get("tile_border_trim")
        if default is not None:
            logger.info(f"Auto-applying {embedding} tile border trim defaults: {default}")
            return default
    return trim


def _resolve_roi_edge_trim(value: str, embedding: str) -> int | tuple[int, int, int, int]:
    """Parse roi_edge_trim, falling back to embedding-specific registry default when 0."""
    trim = _parse_tile_border_trim(value)
    if not trim:
        from datasets.registry import EMBEDDING_REGISTRY
        default = EMBEDDING_REGISTRY.get(embedding, {}).get("roi_edge_trim")
        if default is not None:
            logger.info(f"Auto-applying {embedding} ROI edge trim defaults: {default}")
            return default
    return trim


_VECTOR_SUFFIXES = {".gpkg", ".geojson", ".shp"}


def _detect_label_type(paths: list[str]) -> str:
    """Return 'vector' or 'raster' based on the first path's file suffix."""
    return "vector" if Path(paths[0]).suffix.lower() in _VECTOR_SUFFIXES else "raster"


def _concat_vector_gdfs(paths: list[str], label_col: str, crs) -> "gpd.GeoDataFrame":
    """Load and concatenate multiple vector label files into one GeoDataFrame.

    Each path is loaded as a VectorPatchLabelDataset and its .index GeoDataFrame
    (with 'label' column and pd.IntervalIndex) is concatenated. The resulting
    GeoDataFrame is compatible with split_label_gdf() and from_gdf().
    """
    import pandas as pd

    from datasets.labels import VectorPatchLabelDataset

    gdfs = []
    for p in paths:
        ds = VectorPatchLabelDataset(path=p, label_col=label_col, crs=crs)
        gdfs.append(ds.index)
    merged = pd.concat(gdfs)
    logger.info(f"Loaded {len(paths)} vector label file(s): {len(merged)} polygons total")
    return merged


def _assign_cities_by_fraction(
    paths: list[str],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Randomly assign city paths to train/val/test splits by fraction.

    Guarantees at least 1 city in train. Val and test may be empty if there
    are too few cities for the requested fractions.

    Returns:
        (train_paths, val_paths, test_paths)
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    shuffled = [paths[i] for i in rng.permutation(len(paths))]
    n = len(shuffled)
    n_train = max(1, round(n * train_frac))
    n_val = round(n * val_frac)
    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train : n_train + n_val]
    test_paths = shuffled[n_train + n_val :]
    logger.info(
        f"City-level split — "
        f"train={[Path(p).stem for p in train_paths]}, "
        f"val={[Path(p).stem for p in val_paths]}, "
        f"test={[Path(p).stem for p in test_paths]}"
    )
    if not val_paths:
        logger.warning("No cities assigned to val — val will use train ROI/data")
    if not test_paths:
        logger.warning("No cities assigned to test — test evaluation will be empty")
    return train_paths, val_paths, test_paths


def _compute_class_weights(train_gdf, num_classes: int):
    """Compute inverse-frequency class weights from a training GeoDataFrame.

    Weight for class c = total_polygons / (num_classes * count_c).
    Classes absent from the training set receive weight 0.

    Args:
        train_gdf: GeoDataFrame with a 'label' column (1-based class IDs).
        num_classes: Total number of classes.

    Returns:
        Float tensor of shape (num_classes,).
    """
    import numpy as np
    import torch

    counts = np.zeros(num_classes, dtype=float)
    for label_val, count in train_gdf["label"].value_counts().items():
        idx = int(label_val) - 1  # 1-based → 0-based
        if 0 <= idx < num_classes:
            counts[idx] = count
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * counts), 0.0)
    logger.info(
        f"Class weights (auto): {dict(enumerate(weights.round(3).tolist()))} "
        f"[{int((counts > 0).sum())}/{num_classes} classes present]"
    )
    return torch.tensor(weights, dtype=torch.float32)


def _parse_class_weights(value: str, num_classes: int):
    """Parse a comma-separated class weights string into a tensor.

    Args:
        value: Comma-separated floats, one per class (0-indexed).
        num_classes: Expected number of classes (for validation).

    Returns:
        Float tensor of shape (num_classes,).
    """
    import torch

    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != num_classes:
        raise typer.BadParameter(
            f"--class-weights has {len(parts)} values but num_classes={num_classes}"
        )
    return torch.tensor(parts, dtype=torch.float32)


def _parse_bbox(bbox: str | None, target_crs=None):
    """Parse a comma-separated bbox string into a Shapely Polygon.

    Args:
        bbox: "west,south,east,north" in EPSG:4326, or None.
        target_crs: If set, reproject the bbox from EPSG:4326 to this CRS.

    Returns:
        Shapely Polygon in the target CRS, or None if bbox is None.
    """
    if bbox is None:
        return None

    from pyproj import CRS, Transformer
    from shapely.geometry import box
    from shapely.ops import transform

    coords = [float(x) for x in bbox.split(",")]
    if len(coords) != 4:
        raise typer.BadParameter("bbox must have 4 values: west,south,east,north")
    west, south, east, north = coords
    polygon = box(west, south, east, north)

    if target_crs is not None:
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_user_input(target_crs)
        if src_crs != dst_crs:
            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            polygon = transform(transformer.transform, polygon)

    return polygon


def _parse_bboxes(bboxes: list[str] | None, target_crs=None):
    """Parse a list of bbox strings into a single Shapely geometry (union).

    Args:
        bboxes: List of "west,south,east,north" strings in EPSG:4326, or None/empty.
        target_crs: If set, reproject each bbox from EPSG:4326 to this CRS.

    Returns:
        Shapely Polygon or MultiPolygon (union of all bboxes), or None if empty.
    """
    if not bboxes:
        return None

    from shapely.ops import unary_union

    polygons = [_parse_bbox(b, target_crs) for b in bboxes]
    return unary_union(polygons)


def _parse_toi(year: int | None):
    """Build a pd.Interval for a year to use as temporal filter.

    Args:
        year: Year to filter to, or None.

    Returns:
        pd.Interval covering Jan 1 to Dec 31 of the year, or None.
    """
    if year is None:
        return None

    from datetime import datetime

    import pandas as pd

    return pd.Interval(
        left=pd.Timestamp(datetime(year, 1, 1)),
        right=pd.Timestamp(datetime(year, 12, 31)),
        closed="both",
    )


@app.command()
def train_sklearn(
    embedding: str = typer.Option(..., help="Embedding name: tessera, alpha_earth, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label: str = typer.Option("demuzere_lcz", help="Label dataset name"),
    label_path: List[str] = typer.Option([], help="Path(s) to label file(s) or directory (GeoPackage or GeoTIFF). Repeat to add cities: --label-path city1.gpkg --label-path city2.gpkg"),
    label_column: Optional[str] = typer.Option(None, help="Column name for class labels (required when --label-path points to a GeoPackage)."),
    split_mode: str = typer.Option("geographic", help="Split strategy: 'geographic' (polygon-level stratified for vector; checkerboard per city for raster) or 'city' (whole cities randomly assigned to splits)."),
    train_frac: float = typer.Option(0.70, help="Fraction of cities assigned to training (city split mode only)."),
    val_frac: float = typer.Option(0.15, help="Fraction of cities assigned to validation (city split mode only; unused in sklearn)."),
    checkerboard_tile_size: Optional[float] = typer.Option(None, help="Tile size in CRS units for checkerboard geographic split (raster labels only). Required when split_mode=geographic and using raster labels."),
    min_valid_frac: float = typer.Option(0.10, help="Min fraction of non-nodata pixels for a checkerboard tile to be included (raster labels only)."),
    classifier: str = typer.Option("mlp", help="Classifier type: mlp, random_forest, extra_trees, lgbm, xgboost, logistic_regression"),
    n_samples: int = typer.Option(2000, help="Max samples per class"),
    test_size: float = typer.Option(0.3, help="Test split ratio (geographic mode) or test fraction of cities (city mode)."),
    hidden_layer_sizes: str = typer.Option("100,50", help="MLP hidden layer sizes (comma-separated)"),
    alpha: float = typer.Option(0.0001, help="MLP regularization"),
    learning_rate_init: float = typer.Option(0.001, help="MLP learning rate"),
    max_iter: int = typer.Option(300, help="MLP/LogisticRegression max iterations"),
    n_estimators: int = typer.Option(100, help="RF/ExtraTrees/LightGBM/XGBoost n_estimators"),
    num_leaves: int = typer.Option(31, help="LightGBM num_leaves"),
    lgbm_learning_rate: float = typer.Option(0.1, help="LightGBM learning rate"),
    min_child_samples: int = typer.Option(20, help="LightGBM min_child_samples"),
    xgb_max_depth: int = typer.Option(6, help="XGBoost max_depth"),
    xgb_learning_rate: float = typer.Option(0.1, help="XGBoost learning rate"),
    logreg_c: float = typer.Option(1.0, help="Logistic Regression inverse regularization strength C"),
    patch_size: float = typer.Option(256, help="Patch size in pixels for raster label extraction and prediction map generation. Not used when --label-path is a vector file."),
    stride: float = typer.Option(256, help="Stride in pixels for raster label extraction. Defaults to patch_size (non-overlapping)."),
    pred_patch_size: Optional[int] = typer.Option(None, help="Patch size in pixels for prediction map generation (stitched map). Defaults to max(patch_size, 256)."),
    tile_border_trim: str = typer.Option("0", help="Fix tessera tile-boundary artifacts by replacing border pixels with the nearest valid interior prediction. Pass a single integer N for symmetric trimming or 'N,S,E,W' for asymmetric (e.g. '85,260,90,0' for tessera). Set to 0 to disable."),
    roi_edge_trim: str = typer.Option("0", help="Zero outer boundary pixels of the prediction raster before EDT fill. Handles embedding contamination at ROI edges (not internal tile boundaries). 'N,S,E,W' or single int. Auto-detected from registry when 0."),
    wandb_project: str = typer.Option("eo-fm", help="WandB project name"),
    no_wandb: bool = typer.Option(False, help="Disable WandB logging"),
    sweep: bool = typer.Option(False, help="Run WandB hyperparameter sweep"),
    sweep_count: int = typer.Option(20, help="Number of sweep trials"),
    cv_folds: int = typer.Option(5, help="Number of cross-validation folds"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save trained model (joblib)"),
    bbox: List[str] = typer.Option([], help="Spatial filter: west,south,east,north (EPSG:4326). Repeat to add multiple areas."),
    year: Optional[int] = typer.Option(None, help="Filter embeddings to this year"),
    seed: int = typer.Option(411, help="Random seed"),
    label_propagation: Optional[str] = typer.Option(None, help="Semi-supervised label propagation method: 'sklearn' (LabelSpreading, fast, subsampled) or 'iscen' (graph diffusion, scalable). Only supported with vector labels. Omit to disable."),
    lp_alpha: float = typer.Option(0.5, help="Label propagation alpha: diffusion clamping factor (iscen) or LabelSpreading alpha (sklearn)."),
    lp_k_neighbors: int = typer.Option(15, help="Label propagation k-NN graph connectivity."),
    lp_confidence_threshold: float = typer.Option(0.8, help="Minimum propagation confidence to accept a pseudo-label."),
    lp_max_pixels: int = typer.Option(2_000_000, help="Maximum pixels to extract from the full ROI for label propagation."),
    lp_max_iter: int = typer.Option(30, help="Maximum iterations for label propagation (power iterations for iscen, max_iter for sklearn)."),
) -> None:
    """Train a sklearn pixel classifier on embedding + label datasets."""
    import wandb

    from conf import LabelPropagationConfig, SklearnConfig, WandbConfig
    from datasets.labels import LCZLabelDataset, VectorPatchLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.sklearn_pixel import (
        extract_all_pixels_from_dataset,
        extract_pixels_from_gdf,
        extract_pixels_from_vector_labels,
        predict_sklearn_roi,
        train_sklearn_classifier,
    )
    from utils.paths import OUTPUT_DIR
    from utils.wandb import (
        log_confusion_matrix,
        log_prediction_raster,
        log_sklearn_cv_metrics,
        log_sklearn_metrics,
        run_sklearn_sweep,
    )

    if not label_path:
        raise typer.BadParameter("At least one --label-path is required.")

    parsed_hidden = tuple(int(x) for x in hidden_layer_sizes.split(","))

    raw_bbox = tuple(float(v) for v in bbox[0].split(",")) if bbox else None
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=raw_bbox, year=year)

    label_type = _detect_label_type(label_path)
    is_vector_labels = label_type == "vector"

    if is_vector_labels:
        if label_column is None:
            raise typer.BadParameter("--label-column is required when --label-path points to a GeoPackage.")
        label_ds = None  # built per-split below
    else:
        label_ds = LCZLabelDataset(paths=[Path(p) for p in label_path], crs=embedding_ds.crs)

    roi = _parse_bboxes(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)
    if roi is not None or toi is not None:
        logger.info(f"Filtering to ROI: bboxes={bbox}, year={year}")

    # Raster labels need an IntersectionDataset (for sweep and non-pre-split paths)
    dataset = None if is_vector_labels else embedding_ds & label_ds

    logger.info(f"Embedding: {embedding} ({embedding_path})")
    logger.info(f"Labels ({label_type}, {len(label_path)} file(s)), reprojected to {embedding_ds.crs}")

    sklearn_config = SklearnConfig(
        classifier=classifier,
        n_samples_per_class=n_samples,
        test_size=test_size,
        random_state=seed,
        hidden_layer_sizes=parsed_hidden,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        lgbm_learning_rate=lgbm_learning_rate,
        min_child_samples=min_child_samples,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        logreg_C=logreg_c,
        cv_folds=cv_folds,
    )

    resolved_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "models"
    wandb_config = WandbConfig(project=wandb_project, enabled=not no_wandb, sweep_count=sweep_count)

    # Pre-extract pixels. Vector labels iterate polygons directly (much faster
    # than GridGeoSampler). Raster labels use checkerboard geographic split when
    # split_mode=geographic (separate ROIs for train/test extraction).
    pre_X_train = pre_y_train = pre_X_test = pre_y_test = None
    lp_train_gdf = None  # Training polygon GeoDataFrame for label propagation (vector only)
    if is_vector_labels:
        logger.info(f"Vector labels — split_mode={split_mode}")
        if split_mode == "city" and len(label_path) > 1:
            test_frac = 1.0 - train_frac - val_frac
            train_paths, _val_paths, test_paths = _assign_cities_by_fraction(
                label_path, train_frac, val_frac, seed
            )
            train_gdf = _concat_vector_gdfs(train_paths, label_column, embedding_ds.crs)
            lp_train_gdf = train_gdf
            test_gdf = _concat_vector_gdfs(test_paths if test_paths else train_paths, label_column, embedding_ds.crs)
            logger.info("Extracting pixels from city-assigned train/test GDFs")
            pre_X_train, pre_y_train = extract_pixels_from_gdf(
                embedding_ds, train_gdf,
                n_samples_per_class=sklearn_config.n_samples_per_class,
                seed=sklearn_config.random_state,
                toi=toi,
            )
            pre_X_test, pre_y_test = extract_pixels_from_gdf(
                embedding_ds, test_gdf,
                n_samples_per_class=sklearn_config.n_samples_per_class,
                seed=sklearn_config.random_state,
                toi=toi,
            )
        else:
            # Geographic mode (or single city): concat all GDFs, polygon-level split
            merged_gdf = _concat_vector_gdfs(label_path, label_column, embedding_ds.crs)
            logger.info("Extracting pixels with polygon-level train/test split")
            pre_X_train, pre_y_train, pre_X_test, pre_y_test = extract_pixels_from_vector_labels(
                embedding_ds, merged_gdf,
                test_size=sklearn_config.test_size,
                n_samples_per_class=sklearn_config.n_samples_per_class,
                seed=sklearn_config.random_state,
                toi=toi,
            )
            # Expose training GDF for label propagation (same split as inside extract_pixels_from_vector_labels)
            if label_propagation:
                from models.sklearn_pixel import split_label_gdf
                lp_train_gdf, _ = split_label_gdf(
                    merged_gdf, test_size=sklearn_config.test_size, seed=sklearn_config.random_state
                )
    elif split_mode == "geographic" and checkerboard_tile_size is not None and len(label_path) > 1:
        # Multi-city raster with checkerboard: run per city, union ROIs, extract separately
        from shapely.ops import unary_union as _union
        from utils.geographic_split import checkerboard_roi_split, get_raster_roi
        train_rois, test_rois = [], []
        for lp in label_path:
            city_roi = _parse_bbox(bbox[label_path.index(lp)], target_crs=embedding_ds.crs) \
                if bbox and len(bbox) == len(label_path) \
                else get_raster_roi(lp, target_crs=embedding_ds.crs)
            tr, _vl, te = checkerboard_roi_split(
                city_roi, checkerboard_tile_size, lp,
                train_frac=train_frac, val_frac=0.0, test_frac=1.0 - train_frac,
                min_valid_frac=min_valid_frac, seed=seed,
                roi_crs=embedding_ds.crs,
            )
            if tr:
                train_rois.append(tr)
            if te:
                test_rois.append(te)
        train_roi_merged = _union(train_rois) if train_rois else roi
        test_roi_merged = _union(test_rois) if test_rois else roi
        logger.info("Extracting pixels from checkerboard train/test ROIs")
        pre_X_train, pre_y_train = extract_all_pixels_from_dataset(
            dataset, patch_size=patch_size, stride=stride,
            n_samples_per_class=sklearn_config.n_samples_per_class,
            seed=sklearn_config.random_state, roi=train_roi_merged, toi=toi,
        )
        pre_X_test, pre_y_test = extract_all_pixels_from_dataset(
            dataset, patch_size=patch_size, stride=stride,
            n_samples_per_class=sklearn_config.n_samples_per_class,
            seed=sklearn_config.random_state, roi=test_roi_merged, toi=toi,
        )
    elif split_mode == "city" and len(label_path) > 1:
        # Multi-city raster city mode: assign whole-city label files to splits
        from shapely.ops import unary_union as _union
        from utils.geographic_split import get_raster_roi
        test_frac = 1.0 - train_frac - val_frac
        train_paths, _val_paths, test_paths = _assign_cities_by_fraction(
            label_path, train_frac, val_frac, seed
        )
        train_lbl = LCZLabelDataset(paths=[Path(p) for p in train_paths], crs=embedding_ds.crs)
        test_lbl = LCZLabelDataset(paths=[Path(p) for p in (test_paths or train_paths)], crs=embedding_ds.crs)
        train_ds = embedding_ds & train_lbl
        test_ds = embedding_ds & test_lbl
        train_roi_merged = _union([get_raster_roi(p, target_crs=embedding_ds.crs) for p in train_paths])
        test_roi_merged = _union([get_raster_roi(p, target_crs=embedding_ds.crs) for p in test_paths]) if test_paths else None
        logger.info("Extracting pixels from city-assigned raster train/test datasets")
        pre_X_train, pre_y_train = extract_all_pixels_from_dataset(
            train_ds, patch_size=patch_size, stride=stride,
            n_samples_per_class=sklearn_config.n_samples_per_class,
            seed=sklearn_config.random_state, roi=train_roi_merged, toi=toi,
        )
        pre_X_test, pre_y_test = extract_all_pixels_from_dataset(
            test_ds, patch_size=patch_size, stride=stride,
            n_samples_per_class=sklearn_config.n_samples_per_class,
            seed=sklearn_config.random_state, roi=test_roi_merged, toi=toi,
        )

    # Optional label propagation: augment training set with pseudo-labeled pixels
    if label_propagation:
        if not is_vector_labels:
            logger.warning("--label-propagation is only supported with vector labels; skipping.")
        elif lp_train_gdf is None:
            logger.warning("Training GDF not available for label propagation; skipping.")
        else:
            from models.label_propagation import apply_label_propagation
            lp_config = LabelPropagationConfig(
                method=label_propagation,
                alpha=lp_alpha,
                k_neighbors=lp_k_neighbors,
                confidence_threshold=lp_confidence_threshold,
                max_pixels=lp_max_pixels,
                max_iter=lp_max_iter,
            )
            logger.info(
                f"Label propagation: method={label_propagation}, alpha={lp_alpha}, "
                f"k={lp_k_neighbors}, confidence_threshold={lp_confidence_threshold}, "
                f"max_pixels={lp_max_pixels}"
            )
            pre_X_train, pre_y_train = apply_label_propagation(
                embedding_ds, lp_train_gdf,
                method=label_propagation,
                config=lp_config,
                X_train_orig=pre_X_train,
                y_train_orig=pre_y_train,
                roi=roi, toi=toi,
                patch_size=patch_size,
                n_classes=17,
                n_samples_per_class=sklearn_config.n_samples_per_class,
                seed=seed,
            )

    if sweep and not no_wandb:
        if classifier == "lgbm":
            sweep_parameters = {
                "n_estimators": {"values": [100, 300, 500, 1000]},
                "lgbm_learning_rate": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.3},
                "num_leaves": {"distribution": "int_uniform", "min": 20, "max": 200},
                "min_child_samples": {"distribution": "int_uniform", "min": 10, "max": 100},
            }

            def run_trial():
                with wandb.init(dir=str(resolved_output_dir)) as run:
                    cfg = SklearnConfig(
                        classifier="lgbm",
                        n_samples_per_class=n_samples,
                        test_size=test_size,
                        random_state=seed,
                        n_estimators=wandb.config.n_estimators,
                        lgbm_learning_rate=wandb.config.lgbm_learning_rate,
                        num_leaves=wandb.config.num_leaves,
                        min_child_samples=wandb.config.min_child_samples,
                    )
                    result = train_sklearn_classifier(
                        dataset, cfg,
                        patch_size=patch_size, stride=stride,
                        X_train=pre_X_train, y_train=pre_y_train,
                        X_test=pre_X_test, y_test=pre_y_test,
                    )
                    log_sklearn_metrics(result["metrics"])
        elif classifier == "xgboost":
            sweep_parameters = {
                "n_estimators": {"values": [100, 300, 500, 1000]},
                "xgb_learning_rate": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.3},
                "xgb_max_depth": {"distribution": "int_uniform", "min": 3, "max": 10},
            }

            def run_trial():
                with wandb.init(dir=str(resolved_output_dir)) as run:
                    cfg = SklearnConfig(
                        classifier="xgboost",
                        n_samples_per_class=n_samples,
                        test_size=test_size,
                        random_state=seed,
                        n_estimators=wandb.config.n_estimators,
                        xgb_learning_rate=wandb.config.xgb_learning_rate,
                        xgb_max_depth=wandb.config.xgb_max_depth,
                    )
                    result = train_sklearn_classifier(
                        dataset, cfg,
                        patch_size=patch_size, stride=stride,
                        X_train=pre_X_train, y_train=pre_y_train,
                        X_test=pre_X_test, y_test=pre_y_test,
                    )
                    log_sklearn_metrics(result["metrics"])

        elif classifier == "random_forest":
            sweep_parameters = {
                "n_estimators": {"values": [100, 300, 500]},
                "rf_max_features": {"values": ["sqrt", "log2", 0.3, 0.5]},
                "rf_max_depth": {"values": [None, 10, 20, 30]},
                "rf_min_samples_leaf": {"distribution": "int_uniform", "min": 1, "max": 10},
            }

            def run_trial():
                with wandb.init(dir=str(resolved_output_dir)) as run:
                    cfg = SklearnConfig(
                        classifier="random_forest",
                        n_samples_per_class=n_samples,
                        test_size=test_size,
                        random_state=seed,
                        n_estimators=wandb.config.n_estimators,
                        rf_max_features=wandb.config.rf_max_features,
                        rf_max_depth=wandb.config.rf_max_depth,
                        rf_min_samples_leaf=wandb.config.rf_min_samples_leaf,
                    )
                    result = train_sklearn_classifier(
                        dataset, cfg,
                        patch_size=patch_size, stride=stride,
                        X_train=pre_X_train, y_train=pre_y_train,
                        X_test=pre_X_test, y_test=pre_y_test,
                    )
                    log_sklearn_metrics(result["metrics"])

        else:
            # MLP sweep (default)
            sweep_parameters = {
                "hidden_layer_sizes": {
                    "values": [
                        [64],
                        [128],
                        [256],
                        [512],
                        [100, 50],
                        [256, 128],
                        [512, 256],
                        [256, 128, 64],
                        [512, 256, 128],
                    ],
                },
                "alpha": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
                "learning_rate_init": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
            }

            def run_trial():
                with wandb.init(dir=str(resolved_output_dir)) as run:
                    cfg = SklearnConfig(
                        classifier="mlp",
                        n_samples_per_class=n_samples,
                        test_size=test_size,
                        random_state=seed,
                        hidden_layer_sizes=tuple(wandb.config.hidden_layer_sizes),
                        alpha=wandb.config.alpha,
                        learning_rate_init=wandb.config.learning_rate_init,
                        max_iter=max_iter,
                    )
                    result = train_sklearn_classifier(
                        dataset, cfg,
                        patch_size=patch_size, stride=stride,
                        X_train=pre_X_train, y_train=pre_y_train,
                        X_test=pre_X_test, y_test=pre_y_test,
                    )
                    log_sklearn_metrics(result["metrics"])

        run_sklearn_sweep(run_trial, wandb_config, sweep_parameters, dir=resolved_output_dir)
    else:
        import datetime

        if not no_wandb:
            wandb.init(project=wandb_project, config={"embedding": embedding, **sklearn_config.__dict__}, dir=str(resolved_output_dir))
            run_name = wandb.run.name
        else:
            run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = _make_run_dir(resolved_output_dir, embedding, year, bbox, run_name)
        logger.info(f"Run directory: {run_dir}")

        result = train_sklearn_classifier(
            dataset, sklearn_config, patch_size=patch_size, stride=stride,
            output_dir=run_dir, roi=roi, toi=toi,
            X_train=pre_X_train, y_train=pre_y_train,
            X_test=pre_X_test, y_test=pre_y_test,
        )

        if not no_wandb:
            log_sklearn_cv_metrics(result["cv_results"])
            log_sklearn_metrics(result["metrics"])
            log_confusion_matrix(result["y_train"], result["y_train_pred"], key="confusion_matrix_train")
            log_confusion_matrix(result["y_test"], result["y_pred"], key="confusion_matrix_test")

        logger.info(f"CV results: {result['cv_results']}")
        logger.info(f"Test metrics: {result['metrics']}")
        if result["model_path"]:
            logger.info(f"Model saved to: {result['model_path']}")

        # ROI prediction over full embedding area (not just labeled intersection).
        pred_patch = pred_patch_size if pred_patch_size is not None else max(int(patch_size), 256)
        model_stem = result["model_path"].stem if result["model_path"] else f"{classifier}-sklearn"
        pred_output = run_dir / f"{run_dir.name}_{model_stem}-prediction.tif"
        resolved_trim = _resolve_tile_border_trim(tile_border_trim, embedding)
        resolved_edge_trim = _resolve_roi_edge_trim(roi_edge_trim, embedding)
        pred_result = predict_sklearn_roi(
            embedding_ds, result["classifier"],
            patch_size=pred_patch, stride=pred_patch,
            roi=roi, toi=toi, output_path=pred_output,
            tile_border_trim=resolved_trim,
            roi_edge_trim=resolved_edge_trim,
        )
        if not no_wandb:
            if pred_result["y_true"] is not None and len(pred_result["y_true"]) > 0:
                log_confusion_matrix(pred_result["y_true"], pred_result["y_pred"])
            log_prediction_raster(pred_result["raster_path"])
            wandb.finish()


@app.command()
def train_lightning(
    embedding: str = typer.Option(..., help="Embedding name: tessera, alpha_earth, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label: str = typer.Option("demuzere_lcz", help="Label dataset name"),
    label_path: List[str] = typer.Option([], help="Path(s) to label file(s) or directory (GeoPackage or GeoTIFF). Repeat to add cities: --label-path city1.gpkg --label-path city2.gpkg"),
    split_mode: str = typer.Option("geographic", help="Split strategy: 'geographic' (polygon-level stratified for vector; checkerboard per city for raster) or 'city' (whole cities randomly assigned to train/val/test)."),
    task: str = typer.Option("classification", help="Task type: classification, segmentation"),
    model: str = typer.Option("resnet18", help=(
        "Classification: any timm model name (resnet18/34/50/101/152, vit_tiny_patch16_224, "
        "vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224). "
        "Segmentation: SMP architecture — unet, deeplabv3+, segformer, upernet, dpt. Pair with --backbone."
    )),
    backbone: Optional[str] = typer.Option(None, help=(
        "Segmentation only: SMP encoder backbone. "
        "ResNets: resnet18/34/50/101/152. "
        "Mix Transformers (SegFormer): mit_b0/b1/b2/b3/b4/b5. "
        "ViT (via timm-universal): timm-universal-vit_base_patch16_224, etc. "
        "Defaults to resnet50 if not set."
    )),
    num_classes: int = typer.Option(17, help="Number of output classes"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    max_epochs: int = typer.Option(50, help="Maximum training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    patch_size: float = typer.Option(256, help="Patch size in pixels"),
    stride: Optional[float] = typer.Option(None, help="Stride in pixels for GridGeoSampler (val/test); defaults to patch_size (non-overlapping). Set smaller than patch_size for overlapping tiles."),
    length: int = typer.Option(1000, help="Number of patches sampled per training epoch (RandomBatchGeoSampler). Set to roughly the number of training polygons for full coverage each epoch."),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    accelerator: str = typer.Option("auto", help="Lightning accelerator"),
    devices: int = typer.Option(1, help="Number of devices"),
    wandb_project: str = typer.Option("eo-fm", help="WandB project name"),
    no_wandb: bool = typer.Option(False, help="Disable WandB logging"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save model checkpoints"),
    bbox: List[str] = typer.Option([], help="Spatial filter: west,south,east,north (EPSG:4326). Repeat to add multiple areas."),
    year: Optional[int] = typer.Option(None, help="Filter embeddings to this year"),
    label_column: Optional[str] = typer.Option(None, help="Column name for class labels (required when --label-path points to a GeoPackage)."),
    weights: Optional[str] = typer.Option(None, help=(
        "Pretrained weights for backbone initialisation. "
        "Pass a torchgeo weight name (e.g. 'ResNet50_Weights.LANDSAT_TM_TOA_MOCO'), "
        "'imagenet' for ImageNet weights, or omit for random initialisation. "
        "Run `python -c \"from torchgeo.models import list_models; print(list_models())\"` "
        "to see available models."
    )),
    no_augment: bool = typer.Option(False, help="Disable training augmentations (random flip + rotation)."),
    seed: int = typer.Option(411, help="Random seed for polygon-level train/val/test split (vector labels) or checkerboard tile shuffling (raster labels)."),
    checkerboard_tile_size: Optional[float] = typer.Option(None, help=(
        "Tile size in CRS units (metres for UTM, degrees for WGS84) for checkerboard "
        "geographic train/val/test split. Only applies to raster labels. "
        "If omitted, all splits share the same ROI (no geographic separation)."
    )),
    train_frac: float = typer.Option(0.70, help="Fraction assigned to training. Checkerboard tiles (raster, geographic mode) or cities (city mode)."),
    val_frac: float = typer.Option(0.15, help="Fraction assigned to validation. Checkerboard tiles (raster, geographic mode) or cities (city mode)."),
    test_frac: float = typer.Option(0.15, help="Fraction assigned to test. Checkerboard tiles (raster, geographic mode) or cities (city mode)."),
    min_valid_frac: float = typer.Option(0.10, help=(
        "Minimum fraction of non-nodata pixels required to include a checkerboard tile. "
        "Tiles below this threshold are excluded from all splits (raster labels only)."
    )),
    pred_resolution: str = typer.Option("patch", help=(
        "Output resolution for classification prediction GeoTIFF. "
        "'patch' — one pixel per patch at label resolution (e.g. 320 m); matches label granularity. "
        "'pixel' — patch filled at embedding resolution (e.g. 10 m); each patch is a uniform block."
    )),
    class_weights: str = typer.Option("auto", help=(
        "Class weights for the loss function to counter class imbalance. "
        "'auto' computes inverse-frequency weights from the training polygon counts (default). "
        "'none' disables weighting (uniform loss). "
        "Or pass comma-separated floats, one per class (0-indexed): e.g. '1.0,2.5,0.8,...'."
    )),
    class_weights_cap: Optional[float] = typer.Option(None, help=(
        "Cap the maximum class weight to this value (e.g. 10.0). "
        "Prevents rare classes with extreme inverse-frequency weights from dominating the loss. "
        "Only applies when --class-weights auto. No cap by default."
    )),
    label_propagation: Optional[str] = typer.Option(None, help="Semi-supervised label propagation: 'sklearn' (LabelSpreading, fast, subsampled) or 'iscen' (graph diffusion, scalable). For vector labels, train polygons are rasterised first and training switches to LCZLabelDataset. Val/test labels are never augmented."),
    lp_alpha: float = typer.Option(0.5, help="LP diffusion clamping factor (iscen) or LabelSpreading alpha (sklearn)."),
    lp_k_neighbors: int = typer.Option(15, help="LP k-NN graph connectivity."),
    lp_confidence_threshold: float = typer.Option(0.8, help="Minimum LP confidence to accept a pseudo-label."),
    lp_max_pixels: int = typer.Option(2_000_000, help="Maximum pixels extracted from the ROI for LP."),
    lp_max_iter: int = typer.Option(30, help="LP max iterations (power iter for iscen, max_iter for sklearn)."),
) -> None:
    """Train a pure-PyTorch model (classification or segmentation) on embedding + label datasets."""
    import random

    import numpy as np
    import torch
    import wandb

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    from conf import SamplerConfig, TrainConfig, WandbConfig
    from datamodule import EmbeddingLabelDataModule
    from datasets.labels import LCZLabelDataset, VectorPatchLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.lightning_tasks import build_model, predict_dl_roi
    from utils.paths import OUTPUT_DIR
    from utils.wandb import init_wandb_run, log_confusion_matrix, log_prediction_raster

    if not label_path:
        raise typer.BadParameter("At least one --label-path is required.")

    raw_bbox = tuple(float(v) for v in bbox[0].split(",")) if bbox else None
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=raw_bbox, year=year)

    label_type = _detect_label_type(label_path)
    is_vector_labels = label_type == "vector"

    if is_vector_labels:
        if label_column is None:
            raise typer.BadParameter("--label-column is required when --label-path points to a GeoPackage.")
        label_ds = None  # built per-split below
    else:
        label_ds = LCZLabelDataset(paths=[Path(p) for p in label_path], crs=embedding_ds.crs)

    roi = _parse_bboxes(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)
    if roi is not None or toi is not None:
        logger.info(f"Filtering to ROI: bboxes={bbox}, year={year}")

    logger.info(f"Labels ({label_type}, {len(label_path)} file(s)) reprojected to embedding CRS: {embedding_ds.crs}")

    import datetime

    resolved_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "models"

    train_config = TrainConfig(
        task=task, model=model, backbone=backbone, num_classes=num_classes,
        lr=lr, max_epochs=max_epochs,
        weights=weights, output_dir=str(resolved_output_dir),
    )
    sampler_config = SamplerConfig(patch_size=patch_size, batch_size=batch_size, stride=stride, length=length)
    wandb_config = WandbConfig(project=wandb_project, enabled=not no_wandb)

    # Resolve class weights before building the task
    cw_tensor = None
    if class_weights.lower() == "none":
        pass  # uniform weighting
    elif class_weights.lower() == "auto":
        if not is_vector_labels:
            logger.warning("--class-weights auto requires vector labels; falling back to uniform weighting")
        # weights computed below after the train/val/test split
    else:
        cw_tensor = _parse_class_weights(class_weights, num_classes)

    if is_vector_labels:
        from models.sklearn_pixel import split_label_gdf

        logger.info(f"Vector labels — split_mode={split_mode}, {len(label_path)} file(s)")
        if split_mode == "city" and len(label_path) > 1:
            train_paths, val_paths, test_paths = _assign_cities_by_fraction(
                label_path, train_frac, val_frac, seed
            )
            train_gdf = _concat_vector_gdfs(train_paths, label_column, embedding_ds.crs)
            val_gdf = _concat_vector_gdfs(val_paths, label_column, embedding_ds.crs) if val_paths else train_gdf
            test_gdf = _concat_vector_gdfs(test_paths, label_column, embedding_ds.crs) if test_paths else val_gdf
        else:
            # Geographic mode (or single city): concat all, polygon-level 3-way split
            merged_gdf = _concat_vector_gdfs(label_path, label_column, embedding_ds.crs)
            train_gdf, val_gdf, test_gdf = split_label_gdf(
                merged_gdf, val_size=val_frac, test_size=test_frac, seed=seed,
            )

        if class_weights.lower() == "auto":
            cw_tensor = _compute_class_weights(train_gdf, num_classes)
            if class_weights_cap is not None:
                import torch
                cw_tensor = torch.clamp(cw_tensor, max=class_weights_cap)
                logger.info(f"Class weights after cap ({class_weights_cap}): {cw_tensor.tolist()}")

        val_label_ds = VectorPatchLabelDataset.from_gdf(val_gdf, label_col=label_column)
        test_label_ds = VectorPatchLabelDataset.from_gdf(test_gdf, label_col=label_column)
        if label_propagation is not None:
            # Rasterise train polygons → LP augment → LCZLabelDataset for training.
            # Val/test stay as VectorPatchLabelDataset (no pseudo-labels there).
            import tempfile as _tempfile
            from conf import LabelPropagationConfig
            from datasets.labels import rasterize_gdf as _rasterize_gdf
            from models.label_propagation import augment_label_raster
            _raw_res = embedding_ds.res
            _embedding_res = float(_raw_res[0]) if hasattr(_raw_res, "__len__") else float(_raw_res)
            _lp_seed_dir = Path(_tempfile.mkdtemp(prefix="eo_fm_lp_seed_"))
            _rasterize_gdf(train_gdf, label_column, _lp_seed_dir / "train_labels.tif", res=_embedding_res)
            _lp_aug_dir = Path(_tempfile.mkdtemp(prefix="eo_fm_lp_aug_"))
            _lp_cfg = LabelPropagationConfig(
                method=label_propagation, alpha=lp_alpha, k_neighbors=lp_k_neighbors,
                confidence_threshold=lp_confidence_threshold,
                max_pixels=lp_max_pixels, max_iter=lp_max_iter,
            )
            augment_label_raster(
                embedding_ds, _lp_seed_dir, label_propagation, _lp_cfg,
                _lp_aug_dir, roi=roi, toi=toi,
                patch_size=int(patch_size), n_classes=num_classes, seed=seed,
            )
            train_label_ds = LCZLabelDataset(paths=_lp_aug_dir, crs=embedding_ds.crs)
            logger.info(f"LP ({label_propagation}): train_label_ds → LCZLabelDataset from {_lp_aug_dir}")
        else:
            train_label_ds = VectorPatchLabelDataset.from_gdf(train_gdf, label_col=label_column)
        datamodule = EmbeddingLabelDataModule(
            embedding_ds=embedding_ds,
            train_label_ds=train_label_ds,
            val_label_ds=val_label_ds,
            test_label_ds=test_label_ds,
            sampler_config=sampler_config, task=task, num_workers=num_workers,
            train_roi=roi, val_roi=roi, test_roi=roi,
            train_toi=toi, val_toi=toi, test_toi=toi,
            augment=not no_augment,
        )
    else:
        # Raster labels
        logger.info(f"Raster labels — split_mode={split_mode}, {len(label_path)} file(s)")
        from shapely.geometry import box as shapely_box
        from shapely.ops import unary_union as _union
        from utils.geographic_split import checkerboard_roi_split, get_raster_roi

        if split_mode == "city" and len(label_path) > 1:
            # Assign whole cities to splits
            train_paths, val_paths, test_paths = _assign_cities_by_fraction(
                label_path, train_frac, val_frac, seed
            )
            train_roi = _union([get_raster_roi(p, target_crs=embedding_ds.crs) for p in train_paths])
            val_roi = _union([get_raster_roi(p, target_crs=embedding_ds.crs) for p in val_paths]) if val_paths else train_roi
            test_roi = _union([get_raster_roi(p, target_crs=embedding_ds.crs) for p in test_paths]) if test_paths else None
            # Build per-split label datasets so IntersectionDataset only loads relevant files
            _train_label_paths = [Path(p) for p in train_paths]
            if label_propagation is not None:
                import tempfile as _tempfile
                from conf import LabelPropagationConfig
                from models.label_propagation import augment_label_raster
                _lp_cfg = LabelPropagationConfig(
                    method=label_propagation, alpha=lp_alpha, k_neighbors=lp_k_neighbors,
                    confidence_threshold=lp_confidence_threshold,
                    max_pixels=lp_max_pixels, max_iter=lp_max_iter,
                )
                _lp_aug_dir = Path(_tempfile.mkdtemp(prefix="eo_fm_lp_aug_"))
                augment_label_raster(
                    embedding_ds, _train_label_paths, label_propagation, _lp_cfg,
                    _lp_aug_dir, roi=train_roi, toi=toi,
                    patch_size=int(patch_size), n_classes=num_classes, seed=seed,
                )
                train_label_ds = LCZLabelDataset(paths=_lp_aug_dir, crs=embedding_ds.crs)
                logger.info(f"LP ({label_propagation}): train_label_ds (city raster) → {_lp_aug_dir}")
            else:
                train_label_ds = LCZLabelDataset(paths=_train_label_paths, crs=embedding_ds.crs)
            val_label_ds = LCZLabelDataset(paths=[Path(p) for p in (val_paths or train_paths)], crs=embedding_ds.crs)
            test_label_ds = LCZLabelDataset(paths=[Path(p) for p in (test_paths or val_paths or train_paths)], crs=embedding_ds.crs)
            datamodule = EmbeddingLabelDataModule(
                embedding_ds=embedding_ds,
                train_label_ds=train_label_ds,
                val_label_ds=val_label_ds,
                test_label_ds=test_label_ds,
                sampler_config=sampler_config, task=task, num_workers=num_workers,
                train_roi=train_roi, val_roi=val_roi, test_roi=test_roi,
                train_toi=toi, val_toi=toi, test_toi=toi,
                augment=not no_augment,
            )
        elif checkerboard_tile_size is not None:
            if len(label_path) > 1:
                # Multi-city geographic: checkerboard per city, union ROIs
                train_rois, val_rois, test_rois = [], [], []
                for i, lp in enumerate(label_path):
                    city_roi = (
                        _parse_bbox(bbox[i], target_crs=embedding_ds.crs)
                        if bbox and len(bbox) == len(label_path)
                        else get_raster_roi(lp, target_crs=embedding_ds.crs)
                    )
                    tr, vl, te = checkerboard_roi_split(
                        roi=city_roi, tile_size=checkerboard_tile_size, label_path=lp,
                        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
                        min_valid_frac=min_valid_frac, seed=seed,
                        roi_crs=embedding_ds.crs,
                    )
                    if tr: train_rois.append(tr)
                    if vl: val_rois.append(vl)
                    if te: test_rois.append(te)
                train_roi = _union(train_rois) if train_rois else roi
                val_roi = _union(val_rois) if val_rois else None
                test_roi = _union(test_rois) if test_rois else None
            else:
                # Single city geographic checkerboard
                effective_roi = roi if roi is not None else shapely_box(*embedding_ds.bounds[:4])
                train_roi, val_roi, test_roi = checkerboard_roi_split(
                    roi=effective_roi, tile_size=checkerboard_tile_size, label_path=label_path[0],
                    train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
                    min_valid_frac=min_valid_frac, seed=seed,
                    roi_crs=embedding_ds.crs,
                )
            _checkerboard_label_ds = label_ds
            if label_propagation is not None:
                import tempfile as _tempfile
                from conf import LabelPropagationConfig
                from models.label_propagation import augment_label_raster
                _lp_cfg = LabelPropagationConfig(
                    method=label_propagation, alpha=lp_alpha, k_neighbors=lp_k_neighbors,
                    confidence_threshold=lp_confidence_threshold,
                    max_pixels=lp_max_pixels, max_iter=lp_max_iter,
                )
                _lp_aug_dir = Path(_tempfile.mkdtemp(prefix="eo_fm_lp_aug_"))
                augment_label_raster(
                    embedding_ds, label_path, label_propagation, _lp_cfg,
                    _lp_aug_dir, roi=train_roi, toi=toi,
                    patch_size=int(patch_size), n_classes=num_classes, seed=seed,
                )
                _checkerboard_label_ds = LCZLabelDataset(paths=_lp_aug_dir, crs=embedding_ds.crs)
                logger.info(f"LP ({label_propagation}): label_ds (checkerboard) → {_lp_aug_dir}")
            datamodule = EmbeddingLabelDataModule(
                embedding_ds=embedding_ds, label_ds=_checkerboard_label_ds,
                sampler_config=sampler_config, task=task, num_workers=num_workers,
                train_roi=train_roi, val_roi=val_roi, test_roi=test_roi,
                train_toi=toi, val_toi=toi, test_toi=toi,
                augment=not no_augment,
            )
        else:
            train_roi = val_roi = test_roi = roi
            _nosplit_label_ds = label_ds
            if label_propagation is not None:
                import tempfile as _tempfile
                from conf import LabelPropagationConfig
                from models.label_propagation import augment_label_raster
                _lp_cfg = LabelPropagationConfig(
                    method=label_propagation, alpha=lp_alpha, k_neighbors=lp_k_neighbors,
                    confidence_threshold=lp_confidence_threshold,
                    max_pixels=lp_max_pixels, max_iter=lp_max_iter,
                )
                _lp_aug_dir = Path(_tempfile.mkdtemp(prefix="eo_fm_lp_aug_"))
                augment_label_raster(
                    embedding_ds, label_path, label_propagation, _lp_cfg,
                    _lp_aug_dir, roi=roi, toi=toi,
                    patch_size=int(patch_size), n_classes=num_classes, seed=seed,
                )
                _nosplit_label_ds = LCZLabelDataset(paths=_lp_aug_dir, crs=embedding_ds.crs)
                logger.info(f"LP ({label_propagation}): label_ds (no-split) → {_lp_aug_dir}")
            datamodule = EmbeddingLabelDataModule(
                embedding_ds=embedding_ds, label_ds=_nosplit_label_ds,
                sampler_config=sampler_config, task=task, num_workers=num_workers,
                train_roi=train_roi, val_roi=val_roi, test_roi=test_roi,
                train_toi=toi, val_toi=toi, test_toi=toi,
                augment=not no_augment,
            )

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu")
    logger.info(f"Using device: {device}")

    wandb_run = None
    if not no_wandb:
        wandb_run = init_wandb_run(
            wandb_config,
            run_config={"embedding": embedding, **train_config.__dict__},
            dir=str(resolved_output_dir),
        )
        run_name = wandb_run.name
    else:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = _make_run_dir(resolved_output_dir, embedding, year, bbox, run_name)
    logger.info(f"Run directory: {run_dir}")

    # Build model + loss
    nn_model, loss_fn = build_model(train_config, embedding, class_weights=cw_tensor)
    nn_model = nn_model.to(device)
    if cw_tensor is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=cw_tensor.to(device), ignore_index=-1)
    else:
        loss_fn = loss_fn.to(device)

    logger.info(f"Training {task} with {model}, {embedding} embeddings")

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 15
    best_ckpt_path = run_dir / f"{model}-{task}-best.pt"

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    for epoch in range(max_epochs):
        nn_model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device).float()
            optimizer.zero_grad()
            logits = nn_model(images)
            targets = batch["label" if task == "classification" else "mask"].to(device)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        nn_model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device).float()
                logits = nn_model(images)
                targets = batch["label" if task == "classification" else "mask"].to(device)
                val_loss += loss_fn(logits, targets).item()
                n_val += 1
        val_loss /= max(1, n_val)
        scheduler.step()

        if wandb.run:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch + 1})
        logger.info(f"Epoch {epoch+1}/{max_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {"model_state_dict": nn_model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss},
                best_ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best checkpoint
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        nn_model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.4f}) from {best_ckpt_path}")

    pred_output = run_dir / f"{run_dir.name}_{model}-{task}-prediction.tif"
    pred_result = predict_dl_roi(
        nn_model, datamodule,
        task_type=task, output_path=pred_output,
        pred_resolution=pred_resolution,
        device=device,
    )

    if not no_wandb:
        if pred_result["y_true"] is not None and len(pred_result["y_true"]) > 0:
            log_confusion_matrix(pred_result["y_true"], pred_result["y_pred"])
        log_prediction_raster(pred_result["raster_path"])
        wandb.finish()


@app.command()
def predict_sklearn(
    model_path: str = typer.Option(..., help="Path to saved joblib model"),
    embedding: str = typer.Option(..., help="Embedding name: tessera, alpha_earth, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label_path: Optional[str] = typer.Option(None, help="Path to label data directory (optional; enables confusion matrix)"),
    bbox: List[str] = typer.Option([], help="Spatial filter: west,south,east,north (EPSG:4326). Repeat to add multiple areas."),
    year: Optional[int] = typer.Option(None, help="Filter embeddings to this year"),
    patch_size: float = typer.Option(256, help="Patch size in pixels"),
    stride: float = typer.Option(256, help="Stride in pixels"),
    tile_border_trim: str = typer.Option("0", help="Fix tessera tile-boundary artifacts by nearest-interior fill. Single int N or 'N,S,E,W'. Use '85,260,90,0' for tessera."),
    roi_edge_trim: str = typer.Option("0", help="Zero outer boundary pixels before EDT fill. Handles ROI-edge embedding contamination. 'N,S,E,W' or single int. Auto-detected from registry when 0."),
    no_gap_fill: bool = typer.Option(False, help="Disable EDT gap-fill (leave nodata pixels as 0). Useful for diagnostics."),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save prediction GeoTIFF"),
    wandb_project: str = typer.Option("eo-fm", help="WandB project name"),
    no_wandb: bool = typer.Option(False, help="Disable WandB logging"),
) -> None:
    """Load a saved sklearn model and predict over an ROI."""
    import joblib
    import wandb

    from datasets.labels import LCZLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.sklearn_pixel import predict_sklearn_roi
    from utils.paths import OUTPUT_DIR
    from utils.wandb import log_confusion_matrix, log_prediction_raster

    classifier = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    raw_bbox = tuple(float(v) for v in bbox[0].split(",")) if bbox else None
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=raw_bbox, year=year)
    if label_path is not None:
        label_ds = LCZLabelDataset(paths=label_path, crs=embedding_ds.crs)
        dataset = embedding_ds & label_ds
    else:
        dataset = embedding_ds
        logger.info("No label_path provided — running inference-only (no confusion matrix)")

    roi = _parse_bboxes(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)

    resolved_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "predictions"
    model_stem = Path(model_path).stem
    pred_output = resolved_output_dir / f"{embedding}-{model_stem}-prediction.tif"

    if not no_wandb:
        wandb.init(
            project=wandb_project,
            config={"model_path": model_path, "bboxes": bbox},
            dir=str(resolved_output_dir),
        )

    resolved_trim = _resolve_tile_border_trim(tile_border_trim, embedding)
    resolved_edge_trim = _resolve_roi_edge_trim(roi_edge_trim, embedding)
    pred_result = predict_sklearn_roi(
        dataset, classifier,
        patch_size=patch_size, stride=stride,
        roi=roi, toi=toi, output_path=pred_output,
        tile_border_trim=resolved_trim,
        roi_edge_trim=resolved_edge_trim,
        gap_fill=not no_gap_fill,
    )

    if not no_wandb:
        if pred_result["y_true"] is not None and len(pred_result["y_true"]) > 0:
            log_confusion_matrix(pred_result["y_true"], pred_result["y_pred"])
        log_prediction_raster(pred_result["raster_path"])
        wandb.finish()

    logger.info(f"Prediction saved to {pred_output}")


@app.command()
def predict_dl(
    checkpoint_path: str = typer.Option(..., help="Path to model checkpoint (.pt)"),
    embedding: str = typer.Option(..., help="Embedding name: tessera, alpha_earth, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label_path: Optional[str] = typer.Option(None, help="Path to label data directory or GeoPackage (optional; enables confusion matrix)"),
    label_column: Optional[str] = typer.Option(None, help="Column name for class labels (required when --label-path points to a vector file)."),
    task: str = typer.Option("classification", help="Task type: classification, segmentation"),
    model: str = typer.Option("resnet18", help=(
        "Classification: any timm model name (resnet18/34/50/101/152, vit_*). "
        "Segmentation: SMP architecture — unet, deeplabv3+, segformer, upernet, dpt. Pair with --backbone."
    )),
    backbone: Optional[str] = typer.Option(None, help=(
        "Segmentation only: SMP encoder backbone (resnet18/50, mit_b0-b5, timm-universal-vit_*). "
        "Defaults to resnet50 if not set."
    )),
    num_classes: int = typer.Option(17, help="Number of output classes"),
    batch_size: int = typer.Option(32, help="Batch size"),
    patch_size: float = typer.Option(256, help="Patch size in pixels"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    accelerator: str = typer.Option("auto", help="Lightning accelerator"),
    devices: int = typer.Option(1, help="Number of devices"),
    bbox: List[str] = typer.Option([], help="Spatial filter: west,south,east,north (EPSG:4326). Repeat to add multiple areas."),
    year: Optional[int] = typer.Option(None, help="Filter embeddings to this year"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save prediction GeoTIFF"),
    wandb_project: str = typer.Option("eo-fm", help="WandB project name"),
    no_wandb: bool = typer.Option(False, help="Disable WandB logging"),
    pred_resolution: str = typer.Option("patch", help=(
        "Output resolution for classification prediction GeoTIFF. "
        "'patch' — one pixel per patch at label resolution (e.g. 320 m). "
        "'pixel' — embedding resolution with uniform patch blocks (e.g. 10 m)."
    )),
    pred_stride: Optional[int] = typer.Option(None, help=(
        "Stride in pixels for prediction GridGeoSampler. Defaults to patch_size (non-overlapping). "
        "Set smaller than patch_size (e.g. patch_size//2) for overlapping patches: the probability "
        "maps are averaged over overlaps, smoothing out patch-edge artifacts."
    )),
) -> None:
    """Load a model checkpoint and predict over an ROI."""
    import torch
    import wandb

    from conf import SamplerConfig, TrainConfig
    from datamodule import EmbeddingLabelDataModule
    from datasets.labels import LCZLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.lightning_tasks import build_model, predict_dl_roi
    from utils.paths import OUTPUT_DIR
    from utils.wandb import log_confusion_matrix, log_prediction_raster

    raw_bbox = tuple(float(v) for v in bbox[0].split(",")) if bbox else None
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=raw_bbox, year=year)
    label_ds = None
    if label_path:
        from datasets.labels import VectorPatchLabelDataset
        if Path(label_path).suffix.lower() in (".gpkg", ".geojson", ".shp"):
            if label_column is None:
                raise typer.BadParameter("--label-column is required when --label-path is a vector file.")
            label_ds = VectorPatchLabelDataset(path=label_path, label_col=label_column, crs=embedding_ds.crs)
        else:
            label_ds = LCZLabelDataset(paths=label_path, crs=embedding_ds.crs)
    if label_ds is None:
        logger.info("No label_path provided — running inference-only (no confusion matrix)")

    roi = _parse_bboxes(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)

    train_config = TrainConfig(
        task=task, model=model, backbone=backbone, num_classes=num_classes,
    )
    sampler_config = SamplerConfig(patch_size=patch_size, batch_size=batch_size)

    # Build model architecture then load weights from checkpoint
    nn_model, _ = build_model(train_config, embedding)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Support both new format ("model_state_dict") and legacy Lightning format ("state_dict")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    nn_model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu")
    nn_model = nn_model.to(device)

    datamodule = EmbeddingLabelDataModule(
        embedding_ds=embedding_ds, label_ds=label_ds,
        sampler_config=sampler_config, task=task, num_workers=num_workers,
        test_roi=roi, test_toi=toi,
        pred_stride=pred_stride,
    )

    resolved_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "predictions"
    pred_output = resolved_output_dir / "prediction.tif"

    if not no_wandb:
        wandb.init(project=wandb_project, config={
            "checkpoint_path": checkpoint_path, "task": task, "bboxes": bbox,
        }, dir=str(resolved_output_dir))

    pred_result = predict_dl_roi(
        nn_model, datamodule,
        task_type=task, output_path=pred_output,
        pred_resolution=pred_resolution,
        device=device,
    )

    if not no_wandb:
        if pred_result["y_true"] is not None and len(pred_result["y_true"]) > 0:
            log_confusion_matrix(pred_result["y_true"], pred_result["y_pred"])
        log_prediction_raster(pred_result["raster_path"])
        wandb.finish()

    logger.info(f"Prediction saved to {pred_output}")


@app.command()
def download_embeddings(
    embedding: str = typer.Option(..., help="Embedding name: tessera, alpha_earth"),
    bbox: str = typer.Option(..., help="Bounding box: west,south,east,north (EPSG:4326)"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory (default: DATA_DIR/<embedding>)"),
    year: int = typer.Option(2024, help="Year of embeddings to download"),
    output_format: str = typer.Option("zarr", help="Output format: zarr or tif"),
) -> None:
    """Download embedding data with tile-level caching."""
    from datasets.downloaders import download_alpha_earth, download_tessera
    from utils.paths import ALPHA_EARTH_DIR, SEAMLESS_DIR, TESSERA_DIR

    if output_format not in ("zarr", "tif"):
        raise typer.BadParameter("output-format must be 'zarr' or 'tif'")

    if embedding == "seamless":
        raise typer.BadParameter(
            f"Seamless (EmbeddedSeamlessData) tiles are pre-existing GeoTIFFs — not downloadable "
            f"via this command. Place SDC30_EBD_V001_<MGRS>_<year>.tif files in "
            f"{SEAMLESS_DIR} and pass --embedding-path to the training commands."
        )

    parsed_bbox = [float(x) for x in bbox.split(",")]
    if len(parsed_bbox) != 4:
        raise typer.BadParameter("bbox must have 4 values: west,south,east,north")

    # Default output dirs based on embedding name
    if output_dir is None:
        default_dirs = {
            "tessera": TESSERA_DIR,
            "alpha_earth": ALPHA_EARTH_DIR,
        }
        if embedding not in default_dirs:
            raise typer.BadParameter(
                f"No default output dir for '{embedding}'. Provide --output-dir explicitly."
            )
        output_dir = str(default_dirs[embedding])

    if embedding == "tessera":
        download_tessera(bbox=parsed_bbox, output_dir=output_dir, year=year, output_format=output_format)
    elif embedding == "alpha_earth":
        download_alpha_earth(bbox=parsed_bbox, output_dir=output_dir, year=year, output_format=output_format)
    else:
        raise typer.BadParameter(
            f"Unsupported embedding '{embedding}' for download. Choose: tessera, alpha_earth"
        )

    logger.info(f"Embeddings downloaded to {output_dir}")


@app.command()
def download_coop(
    bbox: str = typer.Option(..., help="Bounding box: west,south,east,north (EPSG:4326)"),
    year: int = typer.Option(..., help="Year of embeddings to download (2017-2025)"),
    output_dir: Optional[str] = typer.Option(None, help="Coop root directory (must contain aef_index.gpkg). Default: AlphaEarth coop dir from .env"),
    workers: int = typer.Option(4, help="Number of parallel download threads"),
    overwrite: bool = typer.Option(False, help="Re-download already-present files"),
) -> None:
    """Download AlphaEarth coop tiles (.tiff + .vrt) from source.coop.

    The output directory must already contain ``aef_index.gpkg``.  Tiles are
    saved at ``{output_dir}/{year}/{utm_zone}/{filename}`` to mirror the S3
    layout.  Already-present files are skipped unless ``--overwrite`` is set.
    """
    from datasets.downloaders import download_alpha_earth_coop

    parsed_bbox = tuple(float(x) for x in bbox.split(","))
    if len(parsed_bbox) != 4:
        raise typer.BadParameter("bbox must have 4 values: west,south,east,north")

    if output_dir is None:
        from utils.paths import ALPHA_EARTH_DIR
        output_dir = str(ALPHA_EARTH_DIR / "coop")

    index_path = Path(output_dir) / "aef_index.gpkg"
    if not index_path.exists():
        raise typer.BadParameter(
            f"aef_index.gpkg not found at {index_path}. "
            "Download it from source.coop and place it in the output directory."
        )

    download_alpha_earth_coop(
        index_path=index_path,
        output_dir=output_dir,
        bbox=parsed_bbox,
        year=year,
        workers=workers,
        overwrite=overwrite,
    )
    logger.info(f"Coop tiles downloaded to {output_dir}")


@app.command()
def download_labels(
    bbox: str = typer.Option(..., help="Bounding box: west,south,east,north (EPSG:4326)"),
    dataset: str = typer.Option("demuzere_lcz", help="GEE dataset key or custom ee_path"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory (default: DATA_DIR/labels/<dataset>)"),
    scale: int = typer.Option(None, help="Resolution in meters (default: from dataset registry)"),
    bands: str = typer.Option(None, help="Comma-separated band names (default: from dataset registry)"),
    ee_path: str = typer.Option(None, help="Custom GEE ImageCollection path"),
) -> None:
    """Download label/raster data from Google Earth Engine with tile-level caching."""
    from datasets.downloaders import download_gee_dataset
    from utils.constants import DATA_DIR

    parsed_bbox = [float(x) for x in bbox.split(",")]
    if len(parsed_bbox) != 4:
        raise typer.BadParameter("bbox must have 4 values: west,south,east,north")

    if output_dir is None:
        output_dir = str(DATA_DIR / "labels" / dataset)

    parsed_bands = [b.strip() for b in bands.split(",")] if bands else None

    kwargs = {}
    if scale is not None:
        kwargs["scale"] = scale
    if parsed_bands is not None:
        kwargs["bands"] = parsed_bands
    if ee_path is not None:
        kwargs["ee_path"] = ee_path

    download_gee_dataset(
        bbox=parsed_bbox,
        output_dir=output_dir,
        dataset=dataset,
        **kwargs,
    )
    logger.info(f"Labels downloaded to {output_dir}")


if __name__ == "__main__":
    app()
