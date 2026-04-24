"""Standalone ResNet classification trainer for LCZ patch classification.

Architecture: timm ResNet variants (resnet18/34/50/101/152) with configurable
presets (nano/small/base/medium/large), CrossEntropy loss, ignore_index=-1 for
unlabeled patches.

Each patch receives a single class label via majority vote over valid mask pixels.
Designed for sparse vector labels (So2Sat GeoPackage).

Usage:
    # Single city (spatial x-axis split, default)
    python src/train_resnet.py train \\
        --embedding google_satellite \\
        --embedding-path /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017/ \\
        --label-path /maps/acz25/.../patches_reference_Nairobi.gpkg \\
        --label-column LCZ_class \\
        --preset base --patch-size 32 --batch-size 32 --num-classes 17 \\
        --bbox "36.45,-1.54,37.16,-0.96" --year 2017

    # Multiple cities, city-level split
    python src/train_resnet.py train \\
        --embedding tessera \\
        --embedding-path /maps/acz25/phd-thesis-data/input/GeoTessera/2017/ \\
        --label-path /maps/acz25/.../patches_reference_Nairobi.gpkg \\
        --label-path /maps/acz25/.../patches_reference_Paris.gpkg \\
        --label-column LCZ_class \\
        --split-mode city --train-frac 0.7 --val-frac 0.15 \\
        --preset base --patch-size 32 --batch-size 32 --num-classes 17 \\
        --year 2017
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import typer
from loguru import logger
from torch.utils.data import DataLoader
from torchgeo.datasets.geo import GeoDataset
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, Units
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from typing import List

from conf import WandbConfig
from datasets.labels import LCZLabelDataset, rasterize_gdf
from datasets.registry import create_embedding_dataset, get_in_channels
from utils.split_helpers import assign_cities_by_fraction, concat_label_gdfs, rasterize_city_paths, roi_from_gdf
from utils.wandb import init_wandb_run


app = typer.Typer(pretty_exceptions_enable=False)


# ─── ResNet Presets ───────────────────────────────────────────────────────────

RESNET_PRESETS: dict[str, str] = {
    "nano":   "resnet18",
    "small":  "resnet34",
    "base":   "resnet50",
    "medium": "resnet101",
    "large":  "resnet152",
}


def build_resnet(
    arch: str,
    in_channels: int,
    num_classes: int,
    head_dropout: float = 0.0,
) -> nn.Module:
    """Build a timm ResNet model with the given architecture.

    Args:
        arch: timm model name (e.g. "resnet50").
        in_channels: Number of embedding input channels.
        num_classes: Number of classification output classes.
        head_dropout: Dropout probability before the final linear layer (0 = off).

    Returns:
        timm ResNet nn.Module.
    """
    import timm

    model = timm.create_model(
        arch,
        in_chans=in_channels,
        num_classes=num_classes,
        pretrained=False,
        drop_rate=head_dropout,
    )
    return model


# ─── Augmentation ────────────────────────────────────────────────────────────

def augment_images(images: torch.Tensor) -> torch.Tensor:
    """Apply random flips and exact 90° rotations to images only.

    Args:
        images: (N, C, H, W) float tensor on any device.

    Returns:
        Augmented images with the same shape.
    """
    aug = []
    for img in images:
        if torch.rand(1) < 0.5:
            img = img.flip(-1)
        if torch.rand(1) < 0.5:
            img = img.flip(-2)
        k = torch.randint(0, 4, (1,)).item()
        if k:
            img = torch.rot90(img, k, dims=(-2, -1))
        if torch.rand(1) < 0.5:
            img = img + torch.randn_like(img) * 0.05
        aug.append(img)
    return torch.stack(aug)


# ─── Spatial split helper ─────────────────────────────────────────────────────

def _spatial_split(gdf, val_fraction: float = 0.15, test_fraction: float = 0.15):
    """Split GeoDataFrame spatially by centroid x-coordinate.

    Args:
        gdf: GeoDataFrame of label polygons in the target CRS.
        val_fraction: Fraction of the x range to use for validation.
        test_fraction: Fraction of the x range to use for test.

    Returns:
        (train_roi, val_roi, test_roi) as shapely boxes.
    """
    from shapely.geometry import box as shapely_box

    cx = gdf.geometry.centroid.x
    train_fraction = 1.0 - val_fraction - test_fraction
    split_x1 = cx.quantile(train_fraction)
    split_x2 = cx.quantile(train_fraction + val_fraction)

    minx, miny, maxx, maxy = gdf.total_bounds
    train_roi = shapely_box(minx, miny, split_x1, maxy)
    val_roi   = shapely_box(split_x1, miny, split_x2, maxy)
    test_roi  = shapely_box(split_x2, miny, maxx, maxy)
    return train_roi, val_roi, test_roi


# ─── DataModule ───────────────────────────────────────────────────────────────

class ResNetDataModule:
    """DataModule for patch-level classification with ResNet.

    Each patch gets a single class label via majority vote over valid mask pixels
    (those with value != -1 after 1-indexed → 0-indexed shift).

    Label convention:
    - LCZLabelDataset returns values 1-17 (1-indexed), 0 = nodata.
    - Collate shifts by -1: classes 0-16, nodata -1 (= ignore_index in CE loss).
    - Majority vote over valid pixels → single (N,) label tensor.

    Args:
        embedding_ds: GeoDataset providing embedding tensors.
        label_ds: LCZLabelDataset (RasterDataset with is_image=False).
        train_roi: Shapely geometry restricting training sampler.
        val_roi: Shapely geometry restricting validation sampler.
        test_roi: Shapely geometry restricting test sampler.
        patch_size: Square patch size in pixels.
        batch_size: Batch size for all loaders.
        length: Training patches per epoch (RandomBatchGeoSampler).
        num_workers: DataLoader worker processes.
        augment: Apply random flips + 90° rotation to images during training.
    """

    def __init__(
        self,
        embedding_ds: GeoDataset,
        label_ds: GeoDataset,
        train_roi,
        val_roi,
        test_roi,
        patch_size: int,
        batch_size: int,
        length: int,
        num_workers: int,
        augment: bool = True,
    ) -> None:
        self.embedding_ds = embedding_ds
        self.label_ds = label_ds
        self.train_roi = train_roi
        self.val_roi = val_roi
        self.test_roi = test_roi
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.length = length
        self.num_workers = num_workers
        self.augment = augment
        self.dataset = None

    def setup(self) -> None:
        self.dataset = self.embedding_ds & self.label_ds

    @staticmethod
    def _mask_to_label(mask: torch.Tensor) -> torch.Tensor:
        """Convert (N, H, W) integer masks to (N,) labels via majority vote.

        Args:
            mask: (N, H, W) long tensor, values 0-16 or -1 (nodata).

        Returns:
            (N,) long tensor; -1 for all-nodata patches.
        """
        B = mask.shape[0]
        flat = mask.view(B, -1)  # (B, H*W)
        labels = torch.full((B,), -1, dtype=torch.long)
        for i in range(B):
            valid_pixels = flat[i][flat[i] != -1]
            if valid_pixels.numel() > 0:
                labels[i] = valid_pixels.mode().values
        return labels

    @staticmethod
    def _collate(batch: list[dict]) -> dict:
        from torchgeo.datasets.utils import stack_samples
        collated = stack_samples(batch)
        if "mask" in collated:
            # 1-17 → 0-16, nodata 0 → -1
            mask = collated["mask"].squeeze(1).long() - 1  # (N, H, W)
            collated["label"] = ResNetDataModule._mask_to_label(mask)
            del collated["mask"]
        return collated

    def _train_collate(self, batch: list[dict]) -> dict:
        collated = self._collate(batch)
        if self.augment:
            collated["image"] = augment_images(collated["image"].float())
        return collated

    def train_dataloader(self) -> DataLoader:
        sampler = RandomBatchGeoSampler(
            self.embedding_ds,
            size=self.patch_size,
            batch_size=self.batch_size,
            length=self.length,
            roi=self.train_roi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._train_collate,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = GridGeoSampler(
            self.embedding_ds,
            size=self.patch_size,
            stride=self.patch_size,
            roi=self.val_roi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = GridGeoSampler(
            self.embedding_ds,
            size=self.patch_size,
            stride=self.patch_size,
            roi=self.test_roi,
            units=Units.PIXELS,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )


# ─── Module ───────────────────────────────────────────────────────────────────

class LCZResNetModule(nn.Module):
    """nn.Module wrapping a timm ResNet for multiclass LCZ patch classification.

    Loss: CrossEntropyLoss with ignore_index=-1 (skips all-nodata patches).
    Metrics: val_acc (top-1 accuracy), val_f1 (macro F1Score).

    Args:
        model: timm ResNet instance.
        num_classes: Number of classification classes.
        lr: Adam learning rate.
        weight_decay: Adam L2 regularization.
        max_epochs: Total training epochs (used for CosineAnnealingLR T_max).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        metric_kw = dict(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.val_acc  = Accuracy(**metric_kw)
        self.val_f1   = MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=-1)
        self.test_acc = Accuracy(**metric_kw)
        self.test_f1  = MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.float())


# ─── Training Loop ────────────────────────────────────────────────────────────

def _run_resnet_training_loop(
    task_module: LCZResNetModule,
    datamodule: ResNetDataModule,
    device: torch.device,
    max_epochs: int,
    early_stopping_patience: int,
    run_dir: Path,
    model_name: str,
) -> tuple[LCZResNetModule, Path | None]:
    """Run the pure-PyTorch training loop for a LCZResNetModule.

    Args:
        task_module: The LCZResNetModule to train (moved to device inside).
        datamodule: ResNetDataModule (setup() called inside).
        device: Device to train on.
        max_epochs: Maximum number of epochs.
        early_stopping_patience: Stop after this many epochs without val_f1 improvement.
        run_dir: Directory to save checkpoints.
        model_name: Stem for checkpoint filename.

    Returns:
        (task_module_with_best_weights, best_ckpt_path)
    """
    import wandb

    task_module = task_module.to(device)
    opt = torch.optim.Adam(
        task_module.parameters(), lr=task_module.lr, weight_decay=task_module.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    best_f1 = -1.0
    patience_counter = 0
    best_ckpt_path: Path | None = None

    for epoch in range(max_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        task_module.train()
        train_loss = 0.0
        n_valid = 0
        for batch in train_loader:
            images = batch["image"].to(device).float()
            labels = batch["label"].to(device)
            # Skip all-nodata batches
            if (labels != -1).sum() == 0:
                continue
            opt.zero_grad()
            logits = task_module.model(images)
            loss = task_module.ce_loss(logits, labels)
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_valid += 1
        train_loss /= max(1, n_valid)

        # ── Validate ───────────────────────────────────────────────────────
        task_module.eval()
        task_module.val_acc.reset()
        task_module.val_f1.reset()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device).float()
                labels = batch["label"].to(device)
                if (labels != -1).sum() == 0:
                    continue
                logits = task_module.model(images)
                loss = task_module.ce_loss(logits, labels)
                preds = logits.argmax(dim=1)
                task_module.val_acc(preds, labels)
                task_module.val_f1(preds, labels)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(1, n_val)
        val_acc = task_module.val_acc.compute().item()
        val_f1  = task_module.val_f1.compute().item()
        sched.step()

        if wandb.run:
            wandb.log({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "val_f1":     val_f1,
                "epoch":      epoch + 1,
            })
        logger.info(
            f"Epoch {epoch+1}/{max_epochs}  "
            f"loss={train_loss:.4f}  val_f1={val_f1:.4f}  val_acc={val_acc:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_ckpt_path = run_dir / f"{model_name}-best.pt"
            torch.save(
                {
                    "model_state_dict": task_module.model.state_dict(),
                    "epoch": epoch + 1,
                    "val_f1": val_f1,
                },
                best_ckpt_path,
            )
            logger.info(f"  → New best (val_f1={val_f1:.4f}), checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_ckpt_path and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        task_module.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model (val_f1={ckpt['val_f1']:.4f}) from {best_ckpt_path}")

    return task_module, best_ckpt_path


# ─── ROI Prediction ──────────────────────────────────────────────────────────

def predict_resnet_roi(
    task_module: LCZResNetModule,
    embedding_ds: GeoDataset,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    roi=None,
    stride: int | None = None,
    output_path: str | Path = "resnet_prediction.tif",
) -> Path:
    """Run ResNet over the full embedding ROI and write a GeoTIFF.

    Each patch receives a single predicted class. All pixels in the patch's
    bounding box are filled with that class value.

    Args:
        task_module: Trained LCZResNetModule (eval mode set internally).
        embedding_ds: GeoDataset providing embedding tensors.
        patch_size: Square patch size in pixels (must match training).
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        roi: Optional Shapely geometry to restrict prediction extent.
        stride: Grid stride in pixels. Defaults to patch_size (non-overlapping).
        output_path: Where to write the output GeoTIFF.

    Returns:
        Path to the saved GeoTIFF.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from torchgeo.datasets.utils import stack_samples

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stride = stride or patch_size

    sampler = GridGeoSampler(
        embedding_ds,
        size=patch_size,
        stride=stride,
        roi=roi,
        units=Units.PIXELS,
    )
    loader = DataLoader(
        embedding_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=stack_samples,
    )

    device = next(task_module.parameters()).device
    task_module.eval()

    res = embedding_ds.res[0]

    # (pred_class, minx, miny, maxx, maxy)
    patch_results: list[tuple] = []

    total = len(sampler) // batch_size + int(len(sampler) % batch_size > 0)
    logger.info(f"Predicting {len(sampler)} patches ({total} batches) …")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch["image"].to(device).float()
            preds = task_module.model(imgs).argmax(dim=1).cpu().numpy()  # (N,)
            bounds = batch["bounds"]  # (N, 9) tensor
            for j in range(preds.shape[0]):
                minx = float(bounds[j, 0])
                maxx = float(bounds[j, 1])
                miny = float(bounds[j, 3])
                maxy = float(bounds[j, 4])
                patch_results.append((int(preds[j]), minx, miny, maxx, maxy))
            if (i + 1) % 200 == 0:
                logger.info(f"  {i + 1}/{total} batches done")

    if not patch_results:
        raise RuntimeError("No predictions produced — check ROI and embedding paths")

    all_minx = min(p[1] for p in patch_results)
    all_miny = min(p[2] for p in patch_results)
    all_maxx = max(p[3] for p in patch_results)
    all_maxy = max(p[4] for p in patch_results)

    out_w = int(round((all_maxx - all_minx) / res))
    out_h = int(round((all_maxy - all_miny) / res))
    raster = np.full((out_h, out_w), fill_value=-1, dtype=np.int16)

    for cls, minx, miny, maxx, maxy in patch_results:
        col     = int(round((minx - all_minx) / res))
        row     = int(round((all_maxy - maxy) / res))
        ph      = int(round((maxy - miny) / res))
        pw      = int(round((maxx - minx) / res))
        # Clamp to raster bounds
        row_end = min(row + ph, out_h)
        col_end = min(col + pw, out_w)
        raster[row:row_end, col:col_end] = cls

    # Export 1-indexed classes (0 = nodata)
    export_raster = np.where(raster >= 0, raster + 1, 0).astype(np.uint8)

    crs = getattr(embedding_ds, "crs", None)
    transform = from_bounds(all_minx, all_miny, all_maxx, all_maxy, out_w, out_h)
    with rasterio.open(
        str(output_path), "w", driver="GTiff",
        height=out_h, width=out_w, count=1, dtype="uint8",
        crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(export_raster, 1)

    logger.info(f"Prediction raster saved: {output_path} ({out_h}×{out_w} px)")
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def train(
    # Data
    embedding: str = typer.Option(..., help="Embedding name: tessera, alpha_earth, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    label_path: List[str] = typer.Option([], help="Vector label file(s) (.gpkg/.geojson/.shp). Repeat for multiple cities."),
    label_column: str = typer.Option("LCZ_class", help="Column with 1-based integer class labels"),
    bbox: str | None = typer.Option(None, help="ROI bounding box 'west,south,east,north' (EPSG:4326)"),
    year: int | None = typer.Option(None, help="Year for temporal filtering of embedding tiles"),
    # Split
    split_mode: str = typer.Option("spatial", help="Split mode: 'spatial' (x-axis quantile), 'city' (whole cities by fraction)"),
    train_frac: float = typer.Option(0.70, help="Fraction of cities for training (city mode only)"),
    val_frac: float = typer.Option(0.15, help="Fraction of cities for validation (city mode only)"),
    # Model
    preset: str = typer.Option("base", help="ResNet size preset: nano (resnet18), small (resnet34), base (resnet50), medium (resnet101), large (resnet152)"),
    arch: str | None = typer.Option(None, help="Override preset: any timm model name (e.g. resnet50, resnext50_32x4d)"),
    head_dropout: float = typer.Option(0.0, help="Dropout probability before the final linear layer"),
    num_classes: int = typer.Option(17, help="Number of LCZ classification classes"),
    # Training
    patch_size: int = typer.Option(32, help="Square patch size in pixels"),
    batch_size: int = typer.Option(32, help="Training and evaluation batch size"),
    length: int = typer.Option(1000, help="Training patches per epoch (RandomBatchGeoSampler)"),
    num_workers: int = typer.Option(4, help="DataLoader worker processes"),
    augment: bool = typer.Option(True, help="Random flips + 90° rotation during training"),
    lr: float = typer.Option(1e-3, help="Adam learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Adam weight decay (L2 regularization)"),
    max_epochs: int = typer.Option(50, help="Maximum training epochs"),
    # Splits (spatial mode)
    val_size: float = typer.Option(0.15, help="Fraction of x range for validation (spatial mode)"),
    test_size: float = typer.Option(0.15, help="Fraction of x range for test (spatial mode)"),
    seed: int = typer.Option(411, help="Random seed for city-mode shuffling"),
    # Callbacks
    early_stopping_patience: int = typer.Option(10, help="EarlyStopping patience (monitors val_f1)"),
    accelerator: str = typer.Option("auto", help="Device: auto, gpu, cpu"),
    output_dir: str | None = typer.Option(None, help="Directory for checkpoint files"),
    # Logging
    wandb_project: str = typer.Option("lcz-classification-dl", help="WandB project name"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Disable WandB logging"),
) -> None:
    """Train a ResNet classification model for LCZ patch classification."""
    import datetime
    import random

    import wandb

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── Parse bbox ───────────────────────────────────────────────────────────
    bbox_tuple: tuple[float, float, float, float] | None = None
    if bbox:
        parts = [float(v) for v in bbox.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter("--bbox must be 'west,south,east,north'")
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])

    # ── Embedding dataset ────────────────────────────────────────────────────
    logger.info(f"Loading {embedding} embeddings from {embedding_path}")
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=bbox_tuple)
    in_channels = get_in_channels(embedding)

    raw_res = embedding_ds.res
    embedding_res = float(raw_res[0]) if hasattr(raw_res, "__len__") else float(raw_res)
    logger.info(f"Embedding: {in_channels} channels, res={embedding_res:.8f} CRS units/px")

    # ── Labels ────────────────────────────────────────────────────────────────
    import tempfile

    if not label_path:
        raise typer.BadParameter("--label-path is required (at least one path)")

    for lp in label_path:
        if Path(lp).suffix.lower() not in (".gpkg", ".geojson", ".shp", ".json"):
            raise typer.BadParameter(
                f"train_resnet.py only supports vector labels (.gpkg / .geojson / .shp). "
                f"Got: {lp}"
            )

    # ── Train/val/test split ──────────────────────────────────────────────────
    if split_mode == "city" and len(label_path) < 2:
        logger.warning("city mode with a single label path — falling back to spatial split")
        split_mode = "spatial"

    if split_mode == "city":
        train_paths, val_paths, test_paths = assign_cities_by_fraction(
            label_path, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        train_gdf = concat_label_gdfs(train_paths, label_column, embedding_ds.crs)
        val_gdf   = concat_label_gdfs(val_paths,   label_column, embedding_ds.crs) if val_paths else None
        test_gdf  = concat_label_gdfs(test_paths,  label_column, embedding_ds.crs) if test_paths else None

        train_roi = roi_from_gdf(train_gdf)
        val_roi   = roi_from_gdf(val_gdf)   if val_gdf  is not None else train_roi
        test_roi  = roi_from_gdf(test_gdf)  if test_gdf is not None else train_roi
        logger.info(
            f"City split: {len(train_gdf)} train / "
            f"{len(val_gdf) if val_gdf is not None else 0} val / "
            f"{len(test_gdf) if test_gdf is not None else 0} test polygons"
        )

        label_tmp_dir = Path(tempfile.mkdtemp(prefix="eo_fm_labels_"))
        all_paths = list(train_paths) + list(val_paths or []) + list(test_paths or [])
        gdf = rasterize_city_paths(
            all_paths, label_column, embedding_ds.crs, embedding_res,
            label_tmp_dir, rasterize_gdf,
        )

    else:  # spatial (default)
        logger.info(f"Loading {len(label_path)} vector label file(s)")
        label_tmp_dir = Path(tempfile.mkdtemp(prefix="eo_fm_labels_"))
        if len(label_path) == 1:
            label_tif_path = label_tmp_dir / "labels.tif"
            gdf = concat_label_gdfs(label_path, label_column, embedding_ds.crs)
            rasterize_gdf(gdf, label_column, label_tif_path, res=embedding_res)
            logger.info(f"Rasterized {len(gdf)} polygons → {label_tif_path}")
        else:
            gdf = rasterize_city_paths(
                label_path, label_column, embedding_ds.crs, embedding_res,
                label_tmp_dir, rasterize_gdf,
            )
        train_roi, val_roi, test_roi = _spatial_split(
            gdf, val_fraction=val_size, test_fraction=test_size
        )
        cx = gdf.geometry.centroid.x
        _tf = 1.0 - val_size - test_size
        n_train = (cx <= cx.quantile(_tf)).sum()
        n_val   = ((cx > cx.quantile(_tf)) & (cx <= cx.quantile(_tf + val_size))).sum()
        n_test  = len(gdf) - n_train - n_val
        logger.info(f"Spatial split: ~{n_train} train / ~{n_val} val / ~{n_test} test polygons")

    label_ds = LCZLabelDataset(
        paths=label_tmp_dir, crs=embedding_ds.crs, res=embedding_ds.res
    )

    datamodule = ResNetDataModule(
        embedding_ds=embedding_ds,
        label_ds=label_ds,
        train_roi=train_roi,
        val_roi=val_roi,
        test_roi=test_roi,
        patch_size=patch_size,
        batch_size=batch_size,
        length=length,
        num_workers=num_workers,
        augment=augment,
    )

    # ── ResNet model ──────────────────────────────────────────────────────────
    arch_name = arch if arch else RESNET_PRESETS.get(preset, "resnet50")
    logger.info(f"Building ResNet: preset={preset}, arch={arch_name}, head_dropout={head_dropout}")
    model = build_resnet(
        arch=arch_name,
        in_channels=in_channels,
        num_classes=num_classes,
        head_dropout=head_dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ResNet trainable parameters: {n_params:,}")

    task_module = LCZResNetModule(
        model=model,
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    run_config = {
        "embedding": embedding,
        "preset": preset,
        "arch": arch_name,
        "head_dropout": head_dropout,
        "num_classes": num_classes,
        "patch_size": patch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "n_params": n_params,
    }
    wandb_run = None
    if not no_wandb:
        wandb_run = init_wandb_run(WandbConfig(project=wandb_project), run_config=run_config)
        run_name = wandb_run.name
    else:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Output directory ──────────────────────────────────────────────────────
    year_str = str(year) if year is not None else "all"
    bbox_str = (
        f"W{bbox_tuple[0]:.1f}_S{bbox_tuple[1]:.1f}_E{bbox_tuple[2]:.1f}_N{bbox_tuple[3]:.1f}"
        if bbox_tuple else "global"
    )
    base_out = Path(output_dir) if output_dir else Path("/tmp/resnet_runs")
    run_dir = base_out / f"{embedding}_{year_str}_{bbox_str}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu"
    )
    logger.info(f"Using device: {device}")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting training …")
    task_module, best_ckpt_path = _run_resnet_training_loop(
        task_module=task_module,
        datamodule=datamodule,
        device=device,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        run_dir=run_dir,
        model_name=f"resnet-{preset}",
    )

    # ── Test evaluation ────────────────────────────────────────────────────────
    logger.info("Running test evaluation …")
    task_module.eval()
    task_module.test_acc.reset()
    task_module.test_f1.reset()
    test_preds: list[torch.Tensor] = []
    test_labels_list: list[torch.Tensor] = []
    test_loss_total = 0.0
    n_test_batches = 0

    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            images = batch["image"].to(device).float()
            labels = batch["label"].to(device)
            if (labels != -1).sum() == 0:
                continue
            logits = task_module.model(images)
            loss = task_module.ce_loss(logits, labels)
            preds = logits.argmax(dim=1)
            task_module.test_acc(preds, labels)
            task_module.test_f1(preds, labels)
            test_loss_total += loss.item()
            n_test_batches += 1
            test_preds.append(preds.cpu())
            test_labels_list.append(labels.cpu())

    test_loss = test_loss_total / max(1, n_test_batches)
    test_acc  = task_module.test_acc.compute().item()
    test_f1   = task_module.test_f1.compute().item()
    logger.info(f"Test results: loss={test_loss:.4f}  f1={test_f1:.4f}  acc={test_acc:.4f}")

    if wandb.run:
        wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1})

    if wandb.run and test_preds:
        y_pred_all = torch.cat(test_preds).numpy().ravel()
        y_true_all = torch.cat(test_labels_list).numpy().ravel()
        valid = y_true_all != -1
        if valid.sum() > 0:
            from utils.wandb import log_confusion_matrix
            log_confusion_matrix(
                y_true_all[valid] + 1,
                y_pred_all[valid] + 1,
                key="test_confusion_matrix",
            )

    # ── WandB artifact ────────────────────────────────────────────────────────
    if not no_wandb and best_ckpt_path:
        artifact = wandb.Artifact(
            name=f"resnet-{run_name}",
            type="model",
            metadata=run_config,
        )
        artifact.add_file(str(best_ckpt_path))
        wandb.log_artifact(artifact)
        logger.info(f"Logged model artifact: resnet-{run_name}")

    # ── ROI prediction ────────────────────────────────────────────────────────
    pred_output = run_dir / f"{run_dir.name}_resnet-{preset}-classification-prediction.tif"
    logger.info(f"Running ROI prediction → {pred_output}")
    predict_resnet_roi(
        task_module=task_module,
        embedding_ds=embedding_ds,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        output_path=pred_output,
    )
    if not no_wandb:
        from utils.wandb import log_prediction_raster
        log_prediction_raster(pred_output)

    if wandb_run:
        wandb.finish()


@app.command()
def predict(
    checkpoint_path: str = typer.Option(..., help="Path to .pt checkpoint file"),
    embedding: str = typer.Option(..., help="Embedding name: tessera, alpha_earth, seamless"),
    embedding_path: str = typer.Option(..., help="Path to embedding data directory"),
    preset: str = typer.Option("base", help="ResNet preset used at training time"),
    arch: str | None = typer.Option(None, help="Override preset: timm model name (must match training)"),
    head_dropout: float = typer.Option(0.0, help="Head dropout (must match training)"),
    num_classes: int = typer.Option(17, help="Number of classes (must match training)"),
    bbox: str | None = typer.Option(None, help="ROI 'west,south,east,north' (EPSG:4326)"),
    year: int | None = typer.Option(None, help="Year for temporal filtering"),
    patch_size: int = typer.Option(32, help="Patch size in pixels (must match training)"),
    stride: int | None = typer.Option(None, help="Grid stride in pixels; defaults to patch_size"),
    batch_size: int = typer.Option(32, help="Inference batch size"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    output_path: str | None = typer.Option(None, help="Output GeoTIFF path. Defaults to <checkpoint_dir>/<checkpoint_dir.name>_resnet-<preset>-classification-prediction.tif"),
    accelerator: str = typer.Option("auto", help="Device: auto, gpu, cpu"),
    wandb_project: str = typer.Option("lcz-classification-dl", help="WandB project name"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Disable WandB logging"),
) -> None:
    """Load a ResNet checkpoint and predict over the full ROI."""
    import wandb

    bbox_tuple: tuple[float, float, float, float] | None = None
    if bbox:
        parts = [float(v) for v in bbox.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter("--bbox must be 'west,south,east,north'")
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])

    logger.info(f"Loading {embedding} embeddings from {embedding_path}")
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=bbox_tuple)
    in_channels = get_in_channels(embedding)

    # ── Reconstruct model ────────────────────────────────────────────────────
    arch_name = arch if arch else RESNET_PRESETS.get(preset, "resnet50")
    model = build_resnet(
        arch=arch_name,
        in_channels=in_channels,
        num_classes=num_classes,
        head_dropout=head_dropout,
    )
    task_module = LCZResNetModule(model=model, num_classes=num_classes)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    task_module.model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu"
    )
    task_module = task_module.to(device)

    # ── Optional ROI from bbox ───────────────────────────────────────────────
    roi = None
    if bbox_tuple:
        from pyproj import CRS, Transformer
        from shapely.geometry import box
        from shapely.ops import transform as shapely_transform
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_user_input(embedding_ds.crs)
        if src_crs != dst_crs:
            t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            roi = shapely_transform(t.transform, box(*bbox_tuple))
        else:
            roi = box(*bbox_tuple)

    # ── WandB ────────────────────────────────────────────────────────────────
    if not no_wandb:
        wandb.init(
            project=wandb_project,
            config={
                "checkpoint": checkpoint_path,
                "embedding": embedding,
                "preset": preset,
                "arch": arch_name,
                "bbox": bbox,
            },
        )

    # ── Resolve output path ───────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_path).parent
    pred_name = f"{ckpt_dir.name}_resnet-{preset}-classification-prediction.tif"
    resolved_output = Path(output_path) if output_path else ckpt_dir / pred_name

    # ── Predict ──────────────────────────────────────────────────────────────
    pred_path = predict_resnet_roi(
        task_module=task_module,
        embedding_ds=embedding_ds,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        roi=roi,
        stride=stride,
        output_path=resolved_output,
    )

    if not no_wandb:
        from utils.wandb import log_prediction_raster
        log_prediction_raster(pred_path)
        wandb.finish()


if __name__ == "__main__":
    app()
