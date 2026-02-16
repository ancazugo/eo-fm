"""Typer CLI for the eo-fm pipeline."""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

app = typer.Typer(help="Earth Observation Foundation Model — LCZ classification pipeline")


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
    label_path: str = typer.Option(..., help="Path to label data directory"),
    classifier: str = typer.Option("mlp", help="Classifier type: mlp, random_forest"),
    n_samples: int = typer.Option(2000, help="Max samples per class"),
    test_size: float = typer.Option(0.3, help="Test split ratio"),
    hidden_layer_sizes: str = typer.Option("100,50", help="MLP hidden layer sizes (comma-separated)"),
    alpha: float = typer.Option(0.0001, help="MLP regularization"),
    learning_rate_init: float = typer.Option(0.001, help="MLP learning rate"),
    n_estimators: int = typer.Option(100, help="RandomForest n_estimators"),
    patch_size: float = typer.Option(256, help="Patch size in pixels for extraction"),
    stride: float = typer.Option(256, help="Stride in pixels for extraction"),
    wandb_project: str = typer.Option("eo-fm", help="WandB project name"),
    no_wandb: bool = typer.Option(False, help="Disable WandB logging"),
    sweep: bool = typer.Option(False, help="Run WandB hyperparameter sweep"),
    sweep_count: int = typer.Option(20, help="Number of sweep trials"),
    cv_folds: int = typer.Option(5, help="Number of cross-validation folds"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save trained model (joblib)"),
    bbox: Optional[str] = typer.Option(None, help="Spatial filter: west,south,east,north (EPSG:4326)"),
    year: Optional[int] = typer.Option(None, help="Filter embeddings to this year"),
    seed: int = typer.Option(411, help="Random seed"),
) -> None:
    """Train a sklearn pixel classifier on embedding + label datasets."""
    import wandb

    from conf import SklearnConfig, WandbConfig
    from datasets.labels import LCZLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.sklearn_pixel import train_sklearn_classifier
    from utils.paths import OUTPUT_DIR
    from utils.wandb import log_sklearn_cv_metrics, log_sklearn_metrics, run_sklearn_sweep

    parsed_hidden = tuple(int(x) for x in hidden_layer_sizes.split(","))

    embedding_ds = create_embedding_dataset(embedding, embedding_path)
    label_ds = LCZLabelDataset(paths=label_path, crs=embedding_ds.crs)

    roi = _parse_bbox(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)
    if roi is not None or toi is not None:
        logger.info(f"Filtering to ROI: bbox={bbox}, year={year}")

    dataset = embedding_ds & label_ds

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

    if sweep and not no_wandb:
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
            wandb.init(project=wandb_project, config=sklearn_config.__dict__)

        result = train_sklearn_classifier(
            dataset, sklearn_config, patch_size=patch_size, stride=stride,
            output_dir=resolved_output_dir, roi=roi, toi=toi,
        )

        if not no_wandb:
            log_sklearn_cv_metrics(result["cv_results"])
            log_sklearn_metrics(result["metrics"])
            wandb.finish()

        logger.info(f"CV results: {result['cv_results']}")
        logger.info(f"Test metrics: {result['metrics']}")
        if result["model_path"]:
            logger.info(f"Model saved to: {result['model_path']}")


@app.command()
def train_lightning(
    embedding: str = typer.Option(..., help="Embedding name: tessera, google_satellite, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label: str = typer.Option("demuzere_lcz", help="Label dataset name"),
    label_path: str = typer.Option(..., help="Path to label data directory"),
    task: str = typer.Option("classification", help="Task type: classification, segmentation"),
    model: str = typer.Option("resnet18", help="Model name (classification) or backbone (segmentation)"),
    num_classes: int = typer.Option(17, help="Number of output classes"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    max_epochs: int = typer.Option(50, help="Maximum training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    patch_size: float = typer.Option(256, help="Patch size in pixels"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    accelerator: str = typer.Option("auto", help="Lightning accelerator"),
    devices: int = typer.Option(1, help="Number of devices"),
    wandb_project: str = typer.Option("eo-fm", help="WandB project name"),
    no_wandb: bool = typer.Option(False, help="Disable WandB logging"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save model checkpoints"),
    bbox: Optional[str] = typer.Option(None, help="Spatial filter: west,south,east,north (EPSG:4326)"),
    year: Optional[int] = typer.Option(None, help="Filter embeddings to this year"),
) -> None:
    """Train a Lightning model (classification or segmentation) on embedding + label datasets."""
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint

    from conf import LightningConfig, SamplerConfig, WandbConfig
    from datamodule import EmbeddingLabelDataModule
    from datasets.labels import LCZLabelDataset
    from datasets.registry import create_embedding_dataset
    from models.lightning_tasks import build_task
    from utils.paths import OUTPUT_DIR
    from utils.wandb import get_wandb_logger

    embedding_ds = create_embedding_dataset(embedding, embedding_path)
    label_ds = LCZLabelDataset(paths=label_path, crs=embedding_ds.crs)

    roi = _parse_bbox(bbox, target_crs=embedding_ds.crs)
    toi = _parse_toi(year)
    if roi is not None or toi is not None:
        logger.info(f"Filtering to ROI: bbox={bbox}, year={year}")

    logger.info(f"Labels reprojected to embedding CRS: {embedding_ds.crs}")

    resolved_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "models"

    lightning_config = LightningConfig(
        task=task, model=model, num_classes=num_classes,
        lr=lr, max_epochs=max_epochs, accelerator=accelerator, devices=devices,
        output_dir=str(resolved_output_dir),
    )
    sampler_config = SamplerConfig(patch_size=patch_size, batch_size=batch_size)
    wandb_config = WandbConfig(project=wandb_project, enabled=not no_wandb)

    task_module = build_task(lightning_config, embedding)
    datamodule = EmbeddingLabelDataModule(
        embedding_ds=embedding_ds, label_ds=label_ds,
        sampler_config=sampler_config, num_workers=num_workers,
        train_roi=roi, val_roi=roi, test_roi=roi,
        train_toi=toi, val_toi=toi, test_toi=toi,
    )

    loggers = []
    if not no_wandb:
        loggers.append(get_wandb_logger(wandb_config))

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(resolved_output_dir),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
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


@app.command()
def download_embeddings(
    embedding: str = typer.Option(..., help="Embedding name: tessera, google_satellite"),
    bbox: str = typer.Option(..., help="Bounding box: west,south,east,north (EPSG:4326)"),
    output_dir: str = typer.Option(None, help="Output directory for Zarr stores (default: DATA_DIR/<embedding>)"),
    year: int = typer.Option(2024, help="Year of embeddings to download"),
) -> None:
    """Download embedding data as Zarr stores with tile-level caching."""
    from datasets.downloaders import download_google_satellite, download_tessera
    from utils.constants import DATA_DIR
    from utils.paths import INPUT_DIR

    parsed_bbox = [float(x) for x in bbox.split(",")]
    if len(parsed_bbox) != 4:
        raise typer.BadParameter("bbox must have 4 values: west,south,east,north")

    # Default output dirs based on embedding name
    if output_dir is None:
        default_dirs = {
            "tessera": INPUT_DIR / "GeoTessera",
            "google_satellite": INPUT_DIR / "Google" / "AlphaEarth",
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
    output_dir: str = typer.Option(None, help="Output directory (default: DATA_DIR/labels/<dataset>)"),
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
