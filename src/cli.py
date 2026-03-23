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
    label_path: str = typer.Option(..., help="Path to label data directory or GeoPackage"),
    label_column: Optional[str] = typer.Option(None, help="Column name for class labels (required when --label-path points to a GeoPackage)."),
    classifier: str = typer.Option("mlp", help="Classifier type: mlp, random_forest, extra_trees, lgbm, xgboost, logistic_regression"),
    n_samples: int = typer.Option(2000, help="Max samples per class"),
    test_size: float = typer.Option(0.3, help="Test split ratio"),
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
) -> None:
    """Train a sklearn pixel classifier on embedding + label datasets."""
    import wandb

    from conf import SklearnConfig, WandbConfig
    from datasets.labels import LCZLabelDataset, VectorPatchLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.sklearn_pixel import (
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

    parsed_hidden = tuple(int(x) for x in hidden_layer_sizes.split(","))

    raw_bbox = tuple(float(v) for v in bbox[0].split(",")) if bbox else None
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=raw_bbox)

    if Path(label_path).suffix.lower() in (".gpkg", ".geojson", ".shp"):
        if label_column is None:
            raise typer.BadParameter("--label-column is required when --label-path is a vector file.")
        label_ds = VectorPatchLabelDataset(path=label_path, label_col=label_column, crs=embedding_ds.crs)
    else:
        label_ds = LCZLabelDataset(paths=label_path, crs=embedding_ds.crs)

    roi = _parse_bboxes(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)
    if roi is not None or toi is not None:
        logger.info(f"Filtering to ROI: bboxes={bbox}, year={year}")

    # Only create the intersection dataset when needed (raster labels or sweep)
    is_vector_labels = isinstance(label_ds, VectorPatchLabelDataset)
    dataset = None if is_vector_labels else embedding_ds & label_ds

    logger.info(f"Embedding: {embedding} ({embedding_path})")
    logger.info(f"Labels: {label} ({label_path}), reprojected to {embedding_ds.crs}")

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

    # Pre-extract pixels when using vector labels (direct polygon iteration, much faster).
    # Polygon-level split per class prevents pixels from the same polygon leaking
    # across train/test.
    pre_X_train = pre_y_train = pre_X_test = pre_y_test = None
    if is_vector_labels:
        logger.info("Vector labels detected — extracting pixels directly from labeled polygons")
        pre_X_train, pre_y_train, pre_X_test, pre_y_test = extract_pixels_from_vector_labels(
            embedding_ds, label_ds.index,
            test_size=sklearn_config.test_size,
            n_samples_per_class=sklearn_config.n_samples_per_class,
            seed=sklearn_config.random_state,
            toi=toi,
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
    label_path: str = typer.Option(..., help="Path to label data directory"),
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
    seed: int = typer.Option(411, help="Random seed for polygon-level train/val/test split (vector labels only)."),
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

    raw_bbox = tuple(float(v) for v in bbox[0].split(",")) if bbox else None
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=raw_bbox)

    if Path(label_path).suffix.lower() in (".gpkg", ".geojson", ".shp"):
        if label_column is None:
            raise typer.BadParameter("--label-column is required when --label-path is a vector file.")
        label_ds = VectorPatchLabelDataset(path=label_path, label_col=label_column, crs=embedding_ds.crs)
    else:
        label_ds = LCZLabelDataset(paths=label_path, crs=embedding_ds.crs)

    roi = _parse_bboxes(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)
    if roi is not None or toi is not None:
        logger.info(f"Filtering to ROI: bboxes={bbox}, year={year}")

    logger.info(f"Labels reprojected to embedding CRS: {embedding_ds.crs}")

    import datetime

    resolved_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "models"

    train_config = TrainConfig(
        task=task, model=model, backbone=backbone, num_classes=num_classes,
        lr=lr, max_epochs=max_epochs,
        weights=weights, output_dir=str(resolved_output_dir),
    )
    sampler_config = SamplerConfig(patch_size=patch_size, batch_size=batch_size, stride=stride, length=length)
    wandb_config = WandbConfig(project=wandb_project, enabled=not no_wandb)

    is_vector_labels = isinstance(label_ds, VectorPatchLabelDataset)

    # Resolve class weights before building the task
    cw_tensor = None
    if class_weights.lower() == "none":
        pass  # uniform weighting
    elif class_weights.lower() == "auto":
        if not is_vector_labels:
            logger.warning("--class-weights auto requires vector labels; falling back to uniform weighting")
        # weights computed below after the train/test split
    else:
        cw_tensor = _parse_class_weights(class_weights, num_classes)

    if is_vector_labels:
        from models.sklearn_pixel import split_label_gdf
        # 3-way polygon-level split: 70% train, 15% val, 15% test
        train_gdf, val_gdf, test_gdf = split_label_gdf(
            label_ds.index, val_size=0.15, test_size=0.15, seed=seed,
        )
        if class_weights.lower() == "auto":
            cw_tensor = _compute_class_weights(train_gdf, num_classes)
            if class_weights_cap is not None:
                import torch
                cw_tensor = torch.clamp(cw_tensor, max=class_weights_cap)
                logger.info(f"Class weights after cap ({class_weights_cap}): {cw_tensor.tolist()}")
        train_label_ds = VectorPatchLabelDataset.from_gdf(train_gdf, label_col=label_column)
        val_label_ds = VectorPatchLabelDataset.from_gdf(val_gdf, label_col=label_column)
        test_label_ds = VectorPatchLabelDataset.from_gdf(test_gdf, label_col=label_column)
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
        datamodule = EmbeddingLabelDataModule(
            embedding_ds=embedding_ds, label_ds=label_ds,
            sampler_config=sampler_config, task=task, num_workers=num_workers,
            train_roi=roi, val_roi=roi, test_roi=roi,
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
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=raw_bbox)
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
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=raw_bbox)
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
