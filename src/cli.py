"""Typer CLI for the eo-fm pipeline."""

from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

app = typer.Typer(help="Earth Observation Foundation Model — LCZ classification pipeline")


def _parse_tile_border_trim(value: str) -> int | tuple[int, int, int, int]:
    """Parse tile_border_trim from CLI string to int or (N,S,E,W) tuple."""
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 4:
        return tuple(parts)  # type: ignore[return-value]
    raise typer.BadParameter("--tile-border-trim must be a single int or 'N,S,E,W'")


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
    embedding: str = typer.Option(..., help="Embedding name: tessera, google_satellite, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label: str = typer.Option("demuzere_lcz", help="Label dataset name"),
    label_path: str = typer.Option(..., help="Path to label data directory or GeoPackage"),
    label_column: Optional[str] = typer.Option(None, help="Column name for class labels (required when --label-path points to a GeoPackage)."),
    classifier: str = typer.Option("mlp", help="Classifier type: mlp, random_forest"),
    n_samples: int = typer.Option(2000, help="Max samples per class"),
    test_size: float = typer.Option(0.3, help="Test split ratio"),
    hidden_layer_sizes: str = typer.Option("100,50", help="MLP hidden layer sizes (comma-separated)"),
    alpha: float = typer.Option(0.0001, help="MLP regularization"),
    learning_rate_init: float = typer.Option(0.001, help="MLP learning rate"),
    n_estimators: int = typer.Option(100, help="RandomForest n_estimators"),
    patch_size: float = typer.Option(256, help="Patch size in pixels for raster label extraction and prediction map generation. Not used when --label-path is a vector file."),
    stride: float = typer.Option(256, help="Stride in pixels for raster label extraction. Defaults to patch_size (non-overlapping)."),
    pred_patch_size: Optional[int] = typer.Option(None, help="Patch size in pixels for prediction map generation (stitched map). Defaults to max(patch_size, 256)."),
    tile_border_trim: str = typer.Option("0", help="Fix tessera tile-boundary artifacts by replacing border pixels with the nearest valid interior prediction. Pass a single integer N for symmetric trimming or 'N,S,E,W' for asymmetric (e.g. '85,260,90,0' for tessera). Set to 0 to disable."),
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
        n_estimators=n_estimators,
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
        if is_vector_labels:
            raise typer.BadParameter("WandB sweep is not yet supported with vector label datasets.")
        sweep_parameters = {
            "hidden_layer_sizes": {"values": [(50,), (100, 50), (128,), (64, 32)]},
            "alpha": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
            "learning_rate_init": {"values": [0.001, 0.0005, 0.005]},
        }

        def run_trial():
            with wandb.init() as run:
                cfg = SklearnConfig(
                    classifier="mlp",
                    n_samples_per_class=n_samples,
                    test_size=test_size,
                    random_state=seed,
                    hidden_layer_sizes=wandb.config.hidden_layer_sizes,
                    alpha=wandb.config.alpha,
                    learning_rate_init=wandb.config.learning_rate_init,
                )
                result = train_sklearn_classifier(dataset, cfg, patch_size=patch_size, stride=stride)
                log_sklearn_metrics(result["metrics"])

        run_sklearn_sweep(run_trial, wandb_config, sweep_parameters)
    else:
        if not no_wandb:
            wandb.init(project=wandb_project, config={"embedding": embedding, **sklearn_config.__dict__}, dir=str(resolved_output_dir))

        result = train_sklearn_classifier(
            dataset, sklearn_config, patch_size=patch_size, stride=stride,
            output_dir=resolved_output_dir, roi=roi, toi=toi,
            X_train=pre_X_train, y_train=pre_y_train,
            X_test=pre_X_test, y_test=pre_y_test,
        )

        if not no_wandb:
            log_sklearn_cv_metrics(result["cv_results"])
            log_sklearn_metrics(result["metrics"])

        logger.info(f"CV results: {result['cv_results']}")
        logger.info(f"Test metrics: {result['metrics']}")
        if result["model_path"]:
            logger.info(f"Model saved to: {result['model_path']}")

        # ROI prediction over full embedding area (not just labeled intersection).
        pred_patch = pred_patch_size if pred_patch_size is not None else max(int(patch_size), 256)
        model_stem = result["model_path"].stem if result["model_path"] else f"{classifier}-sklearn"
        pred_output = resolved_output_dir / f"{embedding}-{model_stem}-prediction.tif"
        pred_result = predict_sklearn_roi(
            embedding_ds, result["classifier"],
            patch_size=pred_patch, stride=pred_patch,
            roi=roi, toi=toi, output_path=pred_output,
            tile_border_trim=_parse_tile_border_trim(tile_border_trim),
        )
        if not no_wandb:
            if pred_result["y_true"] is not None and len(pred_result["y_true"]) > 0:
                log_confusion_matrix(pred_result["y_true"], pred_result["y_pred"])
            log_prediction_raster(pred_result["raster_path"])
            wandb.finish()


@app.command()
def train_lightning(
    embedding: str = typer.Option(..., help="Embedding name: tessera, google_satellite, seamless"),
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
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    accelerator: str = typer.Option("auto", help="Lightning accelerator"),
    devices: int = typer.Option(1, help="Number of devices"),
    wandb_project: str = typer.Option("eo-fm", help="WandB project name"),
    no_wandb: bool = typer.Option(False, help="Disable WandB logging"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save model checkpoints"),
    bbox: List[str] = typer.Option([], help="Spatial filter: west,south,east,north (EPSG:4326). Repeat to add multiple areas."),
    year: Optional[int] = typer.Option(None, help="Filter embeddings to this year"),
    label_column: Optional[str] = typer.Option(None, help="Column name for class labels (required when --label-path points to a GeoPackage)."),
    no_augment: bool = typer.Option(False, help="Disable training augmentations (random flip + rotation)."),
) -> None:
    """Train a Lightning model (classification or segmentation) on embedding + label datasets."""
    import lightning as L
    import wandb
    from lightning.pytorch.callbacks import ModelCheckpoint

    from conf import LightningConfig, SamplerConfig, WandbConfig
    from datamodule import EmbeddingLabelDataModule
    from datasets.labels import LCZLabelDataset, VectorPatchLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.lightning_tasks import build_task, predict_lightning_roi
    from utils.paths import OUTPUT_DIR
    from utils.wandb import get_wandb_logger, log_confusion_matrix, log_prediction_raster

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

    resolved_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "models"

    lightning_config = LightningConfig(
        task=task, model=model, backbone=backbone, num_classes=num_classes,
        lr=lr, max_epochs=max_epochs, accelerator=accelerator, devices=devices,
        output_dir=str(resolved_output_dir),
    )
    sampler_config = SamplerConfig(patch_size=patch_size, batch_size=batch_size, stride=stride)
    wandb_config = WandbConfig(project=wandb_project, enabled=not no_wandb)

    task_module = build_task(lightning_config, embedding)
    datamodule = EmbeddingLabelDataModule(
        embedding_ds=embedding_ds, label_ds=label_ds,
        sampler_config=sampler_config, task=task, num_workers=num_workers,
        train_roi=roi, val_roi=roi, test_roi=roi,
        train_toi=toi, val_toi=toi, test_toi=toi,
        augment=not no_augment,
    )

    loggers = []
    if not no_wandb:
        loggers.append(get_wandb_logger(
            wandb_config,
            run_config={"embedding": embedding, **lightning_config.__dict__},
            save_dir=str(resolved_output_dir),
        ))

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(resolved_output_dir),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,  # always keep latest regardless of val_loss (guards against NaN)
        filename=f"{model}-{task}-{{epoch:02d}}-{{val_loss:.4f}}",
    )

    trainer = L.Trainer(
        max_epochs=lightning_config.max_epochs,
        accelerator=lightning_config.accelerator,
        devices=lightning_config.devices,
        logger=loggers or None,
        callbacks=[checkpoint_callback],
    )

    logger.info(f"Training {task} with {model}, {embedding} embeddings")
    trainer.fit(task_module, datamodule=datamodule)

    # ROI prediction using best checkpoint
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        logger.info(f"Loading best checkpoint: {best_ckpt}")
        task_module = task_module.__class__.load_from_checkpoint(best_ckpt)

    if best_ckpt:
        pred_stem = Path(best_ckpt).stem  # e.g. unet_small-segmentation-epoch=36-val_loss=1.9976
    else:
        pred_stem = f"{model}-{task}"
    pred_output = resolved_output_dir / f"{embedding}-{pred_stem}-prediction.tif"
    pred_result = predict_lightning_roi(
        trainer, task_module, datamodule,
        task_type=task, output_path=pred_output,
    )

    if not no_wandb:
        if pred_result["y_true"] is not None and len(pred_result["y_true"]) > 0:
            log_confusion_matrix(pred_result["y_true"], pred_result["y_pred"])
        log_prediction_raster(pred_result["raster_path"])
        wandb.finish()


@app.command()
def predict_sklearn(
    model_path: str = typer.Option(..., help="Path to saved joblib model"),
    embedding: str = typer.Option(..., help="Embedding name: tessera, google_satellite, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label_path: Optional[str] = typer.Option(None, help="Path to label data directory (optional; enables confusion matrix)"),
    bbox: List[str] = typer.Option([], help="Spatial filter: west,south,east,north (EPSG:4326). Repeat to add multiple areas."),
    year: Optional[int] = typer.Option(None, help="Filter embeddings to this year"),
    patch_size: float = typer.Option(256, help="Patch size in pixels"),
    stride: float = typer.Option(256, help="Stride in pixels"),
    tile_border_trim: str = typer.Option("0", help="Fix tessera tile-boundary artifacts by nearest-interior fill. Single int N or 'N,S,E,W'. Use '85,260,90,0' for tessera."),
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

    pred_result = predict_sklearn_roi(
        dataset, classifier,
        patch_size=patch_size, stride=stride,
        roi=roi, toi=toi, output_path=pred_output,
        tile_border_trim=_parse_tile_border_trim(tile_border_trim),
    )

    if not no_wandb:
        if pred_result["y_true"] is not None and len(pred_result["y_true"]) > 0:
            log_confusion_matrix(pred_result["y_true"], pred_result["y_pred"])
        log_prediction_raster(pred_result["raster_path"])
        wandb.finish()

    logger.info(f"Prediction saved to {pred_output}")


@app.command()
def predict_lightning(
    checkpoint_path: str = typer.Option(..., help="Path to Lightning checkpoint (.ckpt)"),
    embedding: str = typer.Option(..., help="Embedding name: tessera, google_satellite, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label_path: Optional[str] = typer.Option(None, help="Path to label data directory (optional; enables confusion matrix)"),
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
) -> None:
    """Load a Lightning checkpoint and predict over an ROI."""
    import lightning as L
    import wandb

    from conf import LightningConfig, SamplerConfig
    from datamodule import EmbeddingLabelDataModule
    from datasets.labels import LCZLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.lightning_tasks import build_task, predict_lightning_roi
    from utils.paths import OUTPUT_DIR
    from utils.wandb import log_confusion_matrix, log_prediction_raster

    embedding_ds = create_embedding_dataset(embedding, embedding_path)
    label_ds = LCZLabelDataset(paths=label_path, crs=embedding_ds.crs) if label_path else None
    if label_ds is None:
        logger.info("No label_path provided — running inference-only (no confusion matrix)")

    roi = _parse_bboxes(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)

    lightning_config = LightningConfig(
        task=task, model=model, backbone=backbone, num_classes=num_classes,
        accelerator=accelerator, devices=devices,
    )
    sampler_config = SamplerConfig(patch_size=patch_size, batch_size=batch_size)

    task_module = build_task(lightning_config, embedding)
    # Load weights from checkpoint
    task_class = type(task_module)
    task_module = task_class.load_from_checkpoint(checkpoint_path)

    datamodule = EmbeddingLabelDataModule(
        embedding_ds=embedding_ds, label_ds=label_ds,
        sampler_config=sampler_config, task=task, num_workers=num_workers,
        test_roi=roi, test_toi=toi,
    )

    resolved_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "predictions"
    pred_output = resolved_output_dir / "prediction.tif"

    if not no_wandb:
        wandb.init(project=wandb_project, config={
            "checkpoint_path": checkpoint_path, "task": task, "bboxes": bbox,
        }, dir=str(resolved_output_dir))

    trainer = L.Trainer(accelerator=accelerator, devices=devices, logger=False)

    pred_result = predict_lightning_roi(
        trainer, task_module, datamodule,
        task_type=task, output_path=pred_output,
    )

    if not no_wandb:
        if pred_result["y_true"] is not None and len(pred_result["y_true"]) > 0:
            log_confusion_matrix(pred_result["y_true"], pred_result["y_pred"])
        log_prediction_raster(pred_result["raster_path"])
        wandb.finish()

    logger.info(f"Prediction saved to {pred_output}")


@app.command()
def download_embeddings(
    embedding: str = typer.Option(..., help="Embedding name: tessera, google_satellite"),
    bbox: str = typer.Option(..., help="Bounding box: west,south,east,north (EPSG:4326)"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for Zarr stores (default: DATA_DIR/<embedding>)"),
    year: int = typer.Option(2024, help="Year of embeddings to download"),
) -> None:
    """Download embedding data as Zarr stores with tile-level caching."""
    from datasets.downloaders import download_google_satellite, download_tessera
    from utils.paths import GOOGLE_SATELLITE_DIR, SEAMLESS_DIR, TESSERA_DIR

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
            "google_satellite": GOOGLE_SATELLITE_DIR,
        }
        if embedding not in default_dirs:
            raise typer.BadParameter(
                f"No default output dir for '{embedding}'. Provide --output-dir explicitly."
            )
        output_dir = str(default_dirs[embedding])

    if embedding == "tessera":
        download_tessera(bbox=parsed_bbox, output_dir=output_dir, year=year)
    elif embedding == "google_satellite":
        download_google_satellite(bbox=parsed_bbox, output_dir=output_dir, year=year)
    else:
        raise typer.BadParameter(
            f"Unsupported embedding '{embedding}' for download. Choose: tessera, google_satellite"
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
