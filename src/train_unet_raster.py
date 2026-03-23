"""U-Net segmentation pipeline with raster-label data loading and fixed ROI inference.

Mirrors notebooks/torchgeo_raster_labels.ipynb as a CLI script.

Data loading
------------
- TIF embeddings : generic RasterDataset (filename_glob='*.tif')
- Zarr embeddings: ZarrGeoDataset (auto-detected when *.zarr stores are present)
- Labels         : GeoPackage / GeoJSON / SHP → rasterised to a single unified
                   GeoTIFF → LCZLabelDataset (RasterDataset, is_image=False)

Training
--------
- Single IntersectionDataset shared by train + val samplers
- West / east spatial split by polygon centroid x-coordinate (no spatial leakage)
- RandomBatchGeoSampler(roi=train_roi) — training
- GridGeoSampler(roi=val_roi)          — validation / test

Inference
---------
Follows the notebook exactly:
- GridGeoSampler over *embedding_ds* (NOT the intersection dataset)
- batch['bounds'] for per-patch coordinates
- Direct argmax stitching into a numpy raster
- Exports 1-indexed GeoTIFF (0 = nodata)

Models
------
UNet, MulticlassDiceLoss, LCZUNetModule imported unchanged from train_unet.py.
Only the data pipeline and inference loop are re-implemented here.

Usage
-----
    python src/train_unet_raster.py train \\
        --embedding-path /data/GeoTessera/2017/ \\
        --label-path /data/So2Sat-LCZ42/v4/patches_reference_Nairobi.gpkg \\
        --label-column LCZ_class \\
        --preset large --patch-size 32 --batch-size 16 --num-classes 17 \\
        --bbox "36.45,-1.54,37.16,-0.96" --year 2017 \\
        --output-dir /data/output/

    python src/train_unet_raster.py predict \\
        --checkpoint-path /data/output/run_dir/unet-large-best.pt \\
        --embedding-path /data/GeoTessera/2017/ \\
        --preset large --num-classes 17 --patch-size 32 \\
        --bbox "36.45,-1.54,37.16,-0.96"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import tempfile

import geopandas as gpd
import numpy as np
import rasterio
import torch
import typer
from loguru import logger
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, Units

# ── Models imported unchanged from train_unet ────────────────────────────────
from train_unet import (
    LCZUNetModule,
    UNet,
    augment_batch,
)

from conf import WandbConfig
from datasets.labels import LCZLabelDataset, rasterize_gdf
from utils.paths import OUTPUT_DIR
from utils.wandb import init_wandb_run, log_confusion_matrix, log_prediction_raster

app = typer.Typer(pretty_exceptions_enable=False)


# ── Generic GeoTIFF embedding dataset (notebook pattern) ─────────────────────

class TifEmbeddingDataset(RasterDataset):
    """Plain RasterDataset wrapper for any GeoTIFF embedding stack.

    Equivalent to the notebook's::

        class EmbeddingDataset(RasterDataset):
            filename_glob = '*.tif'
    """
    filename_glob = "*.tif"


# ── Embedding loading (TIF or Zarr, auto-detected) ────────────────────────────

def load_embedding_dataset(
    embedding_path: str | Path,
    bbox_tuple: tuple[float, float, float, float] | None = None,
    embedding_name: str | None = None,
) -> RasterDataset:
    """Return an embedding dataset for the given path.

    Auto-detects format:
    - *.zarr stores → ZarrGeoDataset (uses registry metadata if name is given)
    - *.tif files   → TifEmbeddingDataset (plain RasterDataset)

    Args:
        embedding_path: Directory containing embedding tiles.
        bbox_tuple: (west, south, east, north) in EPSG:4326. Passed to
            ZarrGeoDataset for CRS detection when tiles span multiple UTM zones.
        embedding_name: Optional registry key (e.g. "tessera") for Zarr metadata.

    Returns:
        A RasterDataset instance.
    """
    path = Path(embedding_path)
    zarr_stores = list(path.glob("*.zarr"))

    if zarr_stores:
        from datasets.zarr_dataset import ZarrGeoDataset
        from datasets.registry import EMBEDDING_REGISTRY

        meta = EMBEDDING_REGISTRY.get(embedding_name or "", {})
        logger.info(f"Detected Zarr store(s) in {path} — using ZarrGeoDataset")
        return ZarrGeoDataset(
            paths=path,
            tile_size=meta.get("zarr_tile_size"),
            filename_pattern=meta.get("zarr_filename_pattern"),
            filename_crs=meta.get("zarr_filename_crs"),
            filename_is_center=meta.get("zarr_filename_is_center", False),
            bbox=bbox_tuple,
        )

    tif_files = list(path.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(
            f"No *.tif or *.zarr files found in {path}. "
            "Point --embedding-path at the directory containing the tiles."
        )
    logger.info(f"Detected {len(tif_files)} GeoTIFF tile(s) — using TifEmbeddingDataset")
    return TifEmbeddingDataset(paths=str(path))


def detect_in_channels(embedding_ds: RasterDataset, embedding_path: str | Path) -> int:
    """Read band count from the embedding dataset, first tile file, or registry.

    Probe order:
    1. ``embedding_ds.bands`` (populated by some torchgeo datasets).
    2. First ``*.zarr`` store — reads the ``band`` dim of the ``embedding`` variable.
    3. First ``*.tif`` file — reads band count via rasterio.

    Args:
        embedding_ds: Already-loaded embedding dataset.
        embedding_path: Directory containing the tiles (for direct file probing).

    Returns:
        Number of bands (embedding dimension).
    """
    import xarray as xr

    # 1. Dataset-level attribute
    if hasattr(embedding_ds, "bands") and embedding_ds.bands:
        return len(embedding_ds.bands)

    path = Path(embedding_path)

    # 2. Zarr store — open first store, find the variable with a 'band' dim
    zarr_stores = sorted(path.glob("*.zarr"))
    if zarr_stores:
        ds = xr.open_zarr(zarr_stores[0], chunks=False)
        for var in ds.data_vars:
            arr = ds[var]
            if "band" in arr.dims:
                return arr.sizes["band"]

    # 3. GeoTIFF fallback
    tif_files = sorted(path.glob("*.tif"))
    if tif_files:
        with rasterio.open(tif_files[0]) as src:
            return src.count

    raise RuntimeError(f"Cannot determine band count from {embedding_path}")


# ── Label rasterisation ───────────────────────────────────────────────────────

def build_label_dataset(
    label_path: str | Path,
    label_column: str,
    embedding_ds: RasterDataset,
    bbox_poly=None,
) -> tuple[LCZLabelDataset, gpd.GeoDataFrame, Path]:
    """Load vector labels, rasterise them, and return a LCZLabelDataset.

    Follows the notebook's approach:
    1. Read GeoPackage and reproject to embedding CRS.
    2. Optionally clip to bbox polygon.
    3. Rasterise all polygons into one unified GeoTIFF (before any splitting).
    4. Return LCZLabelDataset pointing to that GeoTIFF.

    Args:
        label_path: Path to vector label file (.gpkg / .geojson / .shp).
        label_column: Column with 1-based integer class values.
        embedding_ds: Already-loaded embedding dataset (for CRS + res).
        bbox_poly: Optional Shapely polygon to clip labels (in embedding CRS).

    Returns:
        (label_ds, gdf, label_tif_dir) — dataset, projected+clipped GDF, temp dir.
    """
    gdf = gpd.read_file(label_path).to_crs(embedding_ds.crs)

    if bbox_poly is not None:
        gdf = gdf[gdf.geometry.intersects(bbox_poly)].copy()
        logger.info(f"BBox filter: {len(gdf)} label polygons within ROI")

    if len(gdf) == 0:
        raise ValueError("No label polygons found after bbox filtering.")

    raw_res = embedding_ds.res
    embedding_res = float(raw_res[0]) if hasattr(raw_res, "__len__") else float(raw_res)

    label_tmp_dir = Path(tempfile.mkdtemp(prefix="eo_fm_labels_"))
    label_tif_path = label_tmp_dir / "labels.tif"
    rasterize_gdf(gdf, label_column, label_tif_path, res=embedding_res)
    logger.info(f"Rasterised {len(gdf)} polygons → {label_tif_path}")

    label_ds = LCZLabelDataset(
        paths=label_tmp_dir, crs=embedding_ds.crs, res=embedding_ds.res
    )
    return label_ds, gdf, label_tmp_dir


# ── Spatial split ─────────────────────────────────────────────────────────────

def spatial_split(gdf: gpd.GeoDataFrame, val_fraction: float, test_fraction: float):
    """Split by centroid x-coordinate quantiles into non-overlapping ROIs.

    Args:
        gdf: Label GeoDataFrame in the target CRS.
        val_fraction: Fraction of x range for validation.
        test_fraction: Fraction of x range for test.

    Returns:
        (train_roi, val_roi, test_roi) as Shapely box geometries.
    """
    from shapely.geometry import box as shapely_box

    cx = gdf.geometry.centroid.x
    train_frac = 1.0 - val_fraction - test_fraction
    split_x1 = cx.quantile(train_frac)
    split_x2 = cx.quantile(train_frac + val_fraction)
    minx, miny, maxx, maxy = gdf.total_bounds

    train_roi = shapely_box(minx, miny, split_x1, maxy)
    val_roi   = shapely_box(split_x1, miny, split_x2, maxy)
    test_roi  = shapely_box(split_x2, miny, maxx, maxy)
    return train_roi, val_roi, test_roi


# ── DataLoaders ───────────────────────────────────────────────────────────────

def _collate(batch: list[dict]) -> dict:
    """stack_samples + shift mask from 1-indexed to 0-indexed (-1 = nodata)."""
    collated = stack_samples(batch)
    if "mask" in collated:
        collated["mask"] = collated["mask"].squeeze(1).long() - 1
    return collated


def _train_collate(batch: list[dict], do_augment: bool) -> dict:
    collated = _collate(batch)
    if do_augment:
        images, masks = augment_batch(collated["image"].float(), collated["mask"])
        collated["image"] = images
        collated["mask"] = masks
    return collated


def make_train_loader(
    combined_ds,
    embedding_ds: RasterDataset,
    train_roi,
    patch_size: int,
    batch_size: int,
    length: int,
    num_workers: int,
    augment: bool,
) -> DataLoader:
    sampler = RandomBatchGeoSampler(
        embedding_ds,
        size=patch_size,
        batch_size=batch_size,
        length=length,
        roi=train_roi,
        units=Units.PIXELS,
    )
    return DataLoader(
        combined_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda b: _train_collate(b, augment),
    )


def make_eval_loader(
    combined_ds,
    embedding_ds: RasterDataset,
    roi,
    patch_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    sampler = GridGeoSampler(
        embedding_ds,
        size=patch_size,
        stride=patch_size,
        roi=roi,
        units=Units.PIXELS,
    )
    return DataLoader(
        combined_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(
    task_module: LCZUNetModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    run_dir: Path,
    model_name: str,
) -> tuple[LCZUNetModule, Path | None]:
    """Pure-PyTorch training loop with early stopping on val_miou.

    Args:
        task_module: LCZUNetModule to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Compute device.
        max_epochs: Maximum epochs.
        patience: Early stopping patience (val_miou).
        run_dir: Checkpoint save directory.
        model_name: Stem for checkpoint filename.

    Returns:
        (task_module with best weights loaded, best_ckpt_path or None).
    """
    import wandb

    task_module = task_module.to(device)
    opt = torch.optim.Adam(
        task_module.parameters(), lr=task_module.lr, weight_decay=task_module.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_miou = -1.0
    patience_counter = 0
    best_ckpt_path: Path | None = None

    for epoch in range(max_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        task_module.train()
        train_loss = train_ce = train_dice = 0.0
        n_valid = 0
        for batch in train_loader:
            images = batch["image"].to(device).float()
            masks  = batch["mask"].to(device)
            if (masks != -1).sum() == 0:
                continue  # skip all-nodata batches (CE returns NaN)
            opt.zero_grad()
            logits = task_module.model(images)
            loss, ce, dice = task_module._loss(logits, masks)
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()
            train_loss += loss.item()
            train_ce   += ce.item() if not torch.isnan(ce) else 0.0
            train_dice += dice.item()
            n_valid += 1
        n = max(1, n_valid)
        train_loss /= n; train_ce /= n; train_dice /= n

        # ── Validate ───────────────────────────────────────────────────────
        task_module.eval()
        task_module.val_miou.reset()
        task_module.val_acc.reset()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device).float()
                masks  = batch["mask"].to(device)
                logits = task_module.model(images)
                loss, _, _ = task_module._loss(logits, masks)
                preds = logits.argmax(dim=1)
                task_module.val_miou(preds, masks)
                task_module.val_acc(preds, masks)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(1, n_val)
        val_miou = task_module.val_miou.compute().item()
        val_acc  = task_module.val_acc.compute().item()
        sched.step()

        if wandb.run:
            wandb.log({
                "train_loss": train_loss, "train_ce": train_ce, "train_dice": train_dice,
                "val_loss": val_loss, "val_miou": val_miou, "val_acc": val_acc,
                "epoch": epoch + 1,
            })
        logger.info(
            f"Epoch {epoch+1}/{max_epochs}  "
            f"loss={train_loss:.4f}  val_miou={val_miou:.4f}  val_acc={val_acc:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            best_ckpt_path = run_dir / f"{model_name}-best.pt"
            torch.save(
                {
                    "model_state_dict": task_module.model.state_dict(),
                    "epoch": epoch + 1,
                    "val_miou": val_miou,
                },
                best_ckpt_path,
            )
            logger.info(f"  → New best (val_miou={val_miou:.4f}), checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    if best_ckpt_path and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        task_module.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded best model (val_miou={ckpt['val_miou']:.4f}) from {best_ckpt_path}"
        )

    return task_module, best_ckpt_path


# ── Inference (notebook pattern) ──────────────────────────────────────────────

def predict_roi(
    model: torch.nn.Module,
    embedding_ds: RasterDataset,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    roi=None,
    stride: int | None = None,
    output_path: str | Path = "prediction.tif",
) -> Path:
    """Run inference over the full embedding ROI and write a GeoTIFF.

    Mirrors the notebook's Section 11 exactly:
    - Samples from *embedding_ds* (not the intersection dataset)
    - Uses batch['bounds'] tensor for patch coordinates
    - Stitches argmax predictions into a single raster
    - Exports 1-indexed classes (0 = nodata)

    Args:
        model: Trained model (UNet or any nn.Module with matching in/out).
        embedding_ds: Embedding GeoDataset (TIF or Zarr).
        patch_size: Square patch size in pixels (must match training).
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        device: Compute device.
        roi: Optional Shapely geometry to restrict prediction extent.
            When None, covers the full embedding tile index.
        stride: Grid stride in pixels. Defaults to patch_size (non-overlapping).
        output_path: Where to write the output GeoTIFF.

    Returns:
        Path to the saved GeoTIFF.
    """
    from rasterio.transform import from_bounds

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

    # Use dataset resolution to avoid floating-point drift from per-patch bounds
    raw_res = embedding_ds.res
    res = float(raw_res[0]) if hasattr(raw_res, "__len__") else float(raw_res)

    model.eval()

    patch_results: list[tuple] = []
    total_batches = len(sampler) // batch_size + int(len(sampler) % batch_size > 0)
    logger.info(f"Predicting {len(sampler)} patches ({total_batches} batches) …")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs   = batch["image"].to(device).float()
            logits = model(imgs)
            preds  = logits.argmax(dim=1).cpu().numpy()  # (N, H, W)
            # batch['bounds'] shape: (N, 9)  →  [minx, maxx, _, miny, maxy, ...]
            bounds = batch["bounds"]
            for j in range(preds.shape[0]):
                minx = float(bounds[j, 0])
                maxx = float(bounds[j, 1])
                miny = float(bounds[j, 3])
                maxy = float(bounds[j, 4])
                patch_results.append((preds[j], minx, miny, maxx, maxy))
            if (i + 1) % 200 == 0:
                logger.info(f"  {i + 1}/{total_batches} batches done")

    if not patch_results:
        raise RuntimeError("No predictions produced — check ROI and embedding paths.")

    all_minx = min(p[1] for p in patch_results)
    all_miny = min(p[2] for p in patch_results)
    all_maxx = max(p[3] for p in patch_results)
    all_maxy = max(p[4] for p in patch_results)

    out_w = int(round((all_maxx - all_minx) / res))
    out_h = int(round((all_maxy - all_miny) / res))
    raster = np.full((out_h, out_w), fill_value=-1, dtype=np.int16)

    for pred, minx, miny, maxx, maxy in patch_results:
        col = int(round((minx - all_minx) / res))
        row = int(round((all_maxy - maxy) / res))
        ph, pw = pred.shape
        raster[row:row + ph, col:col + pw] = pred

    logger.info(
        f"Raster: {out_h} × {out_w} px  "
        f"({out_h * res / 1000:.1f} × {out_w * res / 1000:.1f} km)"
    )

    # Export 1-indexed classes (0 = nodata), matching the notebook's Section 12
    export_raster = np.where(raster >= 0, raster + 1, 0).astype(np.uint8)
    crs = getattr(embedding_ds, "crs", None)
    transform = from_bounds(all_minx, all_miny, all_maxx, all_maxy, out_w, out_h)

    with rasterio.open(
        str(output_path), "w",
        driver="GTiff",
        height=out_h, width=out_w,
        count=1, dtype="uint8",
        crs=crs, transform=transform,
        nodata=0,
    ) as dst:
        dst.write(export_raster, 1)

    logger.info(f"Prediction saved: {output_path}")
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_bbox_to_shapely(bbox: str, src_crs, dst_crs):
    """Parse 'west,south,east,north' string and reproject to dst_crs."""
    from pyproj import CRS, Transformer
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    coords = [float(v) for v in bbox.split(",")]
    if len(coords) != 4:
        raise typer.BadParameter("--bbox must be 'west,south,east,north' (EPSG:4326)")
    west, south, east, north = coords
    poly = box(west, south, east, north)

    src = CRS.from_epsg(4326)
    dst = CRS.from_user_input(dst_crs)
    if src != dst:
        t = Transformer.from_crs(src, dst, always_xy=True)
        poly = shapely_transform(t.transform, poly)
    return poly


# ── CLI commands ──────────────────────────────────────────────────────────────

@app.command()
def train(
    # Data
    embedding_path: str = typer.Option(..., help="Directory with GeoTIFF (*.tif) or Zarr (*.zarr) embedding tiles"),
    embedding_name: str | None = typer.Option(None, help="Registry key for Zarr metadata: tessera, alpha_earth, seamless"),
    label_path: str = typer.Option(..., help="Vector label file (.gpkg / .geojson / .shp)"),
    label_column: str = typer.Option("LCZ_class", help="Column with 1-based integer class labels"),
    bbox: str | None = typer.Option(None, help="Label clip bbox 'west,south,east,north' (EPSG:4326). Inference always covers full tile set."),
    year: int | None = typer.Option(None, help="Metadata only — used for run directory naming"),
    # Model
    preset: str = typer.Option("large", help="U-Net preset: nano, small, base, medium, large"),
    depth: int | None = typer.Option(None, help="Override preset encoder depth"),
    base_features: int | None = typer.Option(None, help="Override preset base feature width"),
    bottleneck_dropout: float = typer.Option(0.3, help="Dropout2d at bottleneck"),
    num_classes: int = typer.Option(17, help="Number of segmentation classes"),
    in_channels: int | None = typer.Option(None, help="Embedding band count. Auto-detected if not set."),
    # Loss
    dice_weight: float = typer.Option(0.5, help="Dice loss weight (0=CE only, 1=Dice only)"),
    # Training
    patch_size: int = typer.Option(32, help="Square patch size in pixels"),
    batch_size: int = typer.Option(16, help="Batch size (train + eval)"),
    length: int = typer.Option(500, help="Training patches per epoch (RandomBatchGeoSampler)"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    augment: bool = typer.Option(True, help="Random flips + 90° rotation during training"),
    lr: float = typer.Option(1e-3, help="Adam learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Adam L2 regularisation"),
    max_epochs: int = typer.Option(50, help="Max training epochs"),
    # Splits
    val_size: float = typer.Option(0.15, help="Fraction of x range for validation (spatial split)"),
    test_size: float = typer.Option(0.15, help="Fraction of x range for test (spatial split)"),
    seed: int = typer.Option(411, help="Random seed"),
    # Callbacks
    early_stopping_patience: int = typer.Option(10, help="Early stopping patience (val_miou)"),
    accelerator: str = typer.Option("auto", help="Device: auto, gpu, cpu"),
    # Prediction
    pred_patch_size: int | None = typer.Option(None, help="Patch size for ROI prediction. Defaults to training patch_size."),
    pred_stride: int | None = typer.Option(None, help="Grid stride for ROI prediction. Defaults to pred_patch_size."),
    # Output
    output_dir: str | None = typer.Option(None, help="Directory for checkpoints and prediction GeoTIFF"),
    wandb_project: str = typer.Option("lcz-classification-dl", help="WandB project name"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Disable WandB logging"),
) -> None:
    """Train a U-Net for LCZ segmentation using the raster-label notebook pipeline."""
    import datetime
    import random

    import wandb

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── Embedding ─────────────────────────────────────────────────────────────
    bbox_tuple: tuple[float, float, float, float] | None = None
    if bbox is not None:
        parts = [float(v) for v in bbox.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter("--bbox must be 'west,south,east,north'")
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])

    logger.info(f"Loading embeddings from {embedding_path}")
    embedding_ds = load_embedding_dataset(embedding_path, bbox_tuple, embedding_name)
    logger.info(
        f"Tiles: {len(embedding_ds.index)}  CRS: {embedding_ds.crs}  res: {embedding_ds.res}"
    )

    n_channels = in_channels if in_channels is not None else detect_in_channels(embedding_ds, embedding_path)
    logger.info(f"in_channels: {n_channels}")

    # ── Labels → unified raster ───────────────────────────────────────────────
    bbox_poly = _parse_bbox_to_shapely(bbox, None, embedding_ds.crs) if bbox else None
    label_ds, gdf, _ = build_label_dataset(label_path, label_column, embedding_ds, bbox_poly)

    # ── Spatial split ─────────────────────────────────────────────────────────
    train_roi, val_roi, test_roi = spatial_split(gdf, val_size, test_size)
    cx = gdf.geometry.centroid.x
    train_frac = 1.0 - val_size - test_size
    n_tr = (cx <= cx.quantile(train_frac)).sum()
    n_va = ((cx > cx.quantile(train_frac)) & (cx <= cx.quantile(train_frac + val_size))).sum()
    n_te = len(gdf) - n_tr - n_va
    logger.info(f"Spatial split: ~{n_tr} train / ~{n_va} val / ~{n_te} test polygons")

    # ── Single combined dataset ───────────────────────────────────────────────
    combined_ds = embedding_ds & label_ds

    train_loader = make_train_loader(
        combined_ds, embedding_ds, train_roi,
        patch_size, batch_size, length, num_workers, augment,
    )
    val_loader = make_eval_loader(
        combined_ds, embedding_ds, val_roi,
        patch_size, batch_size, num_workers,
    )
    test_loader = make_eval_loader(
        combined_ds, embedding_ds, test_roi,
        patch_size, batch_size, num_workers,
    )
    logger.info(
        f"Train batches: {len(train_loader)}  "
        f"Val patches: {len(val_loader.sampler)}  "
        f"Test patches: {len(test_loader.sampler)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    d, bf = UNet.PRESETS.get(preset, (4, 48))
    if depth is not None:
        d = depth
    if base_features is not None:
        bf = base_features

    logger.info(f"Building U-Net: preset={preset}, depth={d}, base_features={bf}")
    model = UNet(
        in_channels=n_channels, num_classes=num_classes,
        depth=d, base_features=bf, bottleneck_dropout=bottleneck_dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    task_module = LCZUNetModule(
        model=model, num_classes=num_classes,
        lr=lr, weight_decay=weight_decay, dice_weight=dice_weight, max_epochs=max_epochs,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    # Use embedding_name if provided, else derive a short label from the path.
    embedding_label = embedding_name or Path(embedding_path).parent.name or Path(embedding_path).name
    run_config = {
        "embedding": embedding_label,
        "embedding_path": embedding_path,
        "preset": preset, "depth": d, "base_features": bf,
        "in_channels": n_channels, "num_classes": num_classes,
        "patch_size": patch_size, "batch_size": batch_size,
        "lr": lr, "weight_decay": weight_decay, "dice_weight": dice_weight,
        "max_epochs": max_epochs, "n_params": n_params,
    }
    # ── Output directory ──────────────────────────────────────────────────────
    year_str = str(year) if year is not None else "all"
    bbox_str = (
        f"W{bbox_tuple[0]:.1f}_S{bbox_tuple[1]:.1f}_E{bbox_tuple[2]:.1f}_N{bbox_tuple[3]:.1f}"
        if bbox_tuple else "global"
    )
    base_out = Path(output_dir) if output_dir else OUTPUT_DIR / "models"

    wandb_run = None
    if not no_wandb:
        wandb_run = init_wandb_run(
            WandbConfig(project=wandb_project),
            run_config=run_config,
            dir=str(base_out),
        )
        run_name = wandb_run.name
    else:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = base_out / f"{embedding_label}_{year_str}_{bbox_str}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Train ─────────────────────────────────────────────────────────────────
    task_module, best_ckpt_path = run_training(
        task_module=task_module,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=max_epochs,
        patience=early_stopping_patience,
        run_dir=run_dir,
        model_name=f"unet-{preset}",
    )

    # ── Test evaluation ────────────────────────────────────────────────────────
    logger.info("Running test evaluation …")
    task_module.eval()
    task_module.test_miou.reset()
    task_module.test_acc.reset()
    test_loss_total = 0.0
    n_test_batches = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device).float()
            masks  = batch["mask"].to(device)
            logits = task_module.model(images)
            loss, _, _ = task_module._loss(logits, masks)
            preds = logits.argmax(dim=1)
            task_module.test_miou(preds, masks)
            task_module.test_acc(preds, masks)
            test_loss_total += loss.item()
            n_test_batches += 1
            all_preds.append(preds.cpu())
            all_labels.append(masks.cpu())

    test_loss = test_loss_total / max(1, n_test_batches)
    test_miou = task_module.test_miou.compute().item()
    test_acc  = task_module.test_acc.compute().item()
    logger.info(
        f"Test: loss={test_loss:.4f}  miou={test_miou:.4f}  acc={test_acc:.4f}"
    )

    if not no_wandb:
        wandb.log({"test_loss": test_loss, "test_miou": test_miou, "test_acc": test_acc})
        if all_preds:
            y_pred_all = torch.cat(all_preds).numpy().ravel()
            y_true_all = torch.cat(all_labels).numpy().ravel()
            valid = y_true_all != -1
            if valid.sum() > 0:
                log_confusion_matrix(
                    y_true_all[valid] + 1,
                    y_pred_all[valid] + 1,
                    key="test_confusion_matrix",
                )

    # ── WandB artifact ────────────────────────────────────────────────────────
    if not no_wandb and best_ckpt_path:
        artifact = wandb.Artifact(
            name=f"unet-{run_name}", type="model", metadata=run_config
        )
        artifact.add_file(str(best_ckpt_path))
        wandb.log_artifact(artifact)

    # ── Full-ROI prediction ────────────────────────────────────────────────────
    # Covers the full embedding tile set (roi=None), matching the notebook's
    # tif_index.total_bounds approach. No restriction to label extent.
    p_size   = pred_patch_size if pred_patch_size is not None else patch_size
    p_stride = pred_stride if pred_stride is not None else p_size
    pred_output = run_dir / f"{run_dir.name}_unet-{preset}-prediction.tif"
    logger.info(
        f"Running full-ROI prediction (patch={p_size}, stride={p_stride}) → {pred_output}"
    )
    predict_roi(
        model=task_module.model,
        embedding_ds=embedding_ds,
        patch_size=p_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        output_path=pred_output,
        stride=p_stride,
    )

    if not no_wandb:
        log_prediction_raster(pred_output)
        wandb.finish()


@app.command()
def predict(
    checkpoint_path: str = typer.Option(..., help="Path to .pt checkpoint file"),
    embedding_path: str = typer.Option(..., help="Directory with GeoTIFF or Zarr embedding tiles"),
    embedding_name: str | None = typer.Option(None, help="Registry key for Zarr: tessera, alpha_earth, seamless"),
    preset: str = typer.Option("large", help="U-Net preset used at training time"),
    depth: int | None = typer.Option(None, help="Override preset depth (must match training)"),
    base_features: int | None = typer.Option(None, help="Override preset base features (must match training)"),
    bottleneck_dropout: float = typer.Option(0.3, help="Bottleneck dropout (must match training)"),
    num_classes: int = typer.Option(17, help="Number of classes (must match training)"),
    in_channels: int | None = typer.Option(None, help="Embedding band count. Auto-detected if not set."),
    bbox: str | None = typer.Option(None, help="Restrict prediction to 'west,south,east,north' (EPSG:4326)"),
    patch_size: int = typer.Option(32, help="Patch size in pixels (must match training)"),
    stride: int | None = typer.Option(None, help="Grid stride in pixels; defaults to patch_size"),
    batch_size: int = typer.Option(16, help="Inference batch size"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    output_path: str | None = typer.Option(None, help="Output GeoTIFF. Defaults to <checkpoint_dir>/unet-<preset>-prediction.tif"),
    accelerator: str = typer.Option("auto", help="Device: auto, gpu, cpu"),
    wandb_project: str = typer.Option("lcz-classification-dl", help="WandB project name"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Disable WandB logging"),
) -> None:
    """Load a checkpoint and run full-ROI prediction (notebook inference pattern)."""
    import wandb

    # ── Embedding ─────────────────────────────────────────────────────────────
    bbox_tuple: tuple[float, float, float, float] | None = None
    if bbox is not None:
        parts = [float(v) for v in bbox.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter("--bbox must be 'west,south,east,north'")
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])

    logger.info(f"Loading embeddings from {embedding_path}")
    embedding_ds = load_embedding_dataset(embedding_path, bbox_tuple, embedding_name)
    logger.info(
        f"Tiles: {len(embedding_ds.index)}  CRS: {embedding_ds.crs}  res: {embedding_ds.res}"
    )

    n_channels = in_channels if in_channels is not None else detect_in_channels(embedding_ds, embedding_path)
    logger.info(f"in_channels: {n_channels}")

    # ── Model ─────────────────────────────────────────────────────────────────
    d, bf = UNet.PRESETS.get(preset, (4, 48))
    if depth is not None:
        d = depth
    if base_features is not None:
        bf = base_features

    model = UNet(
        in_channels=n_channels, num_classes=num_classes,
        depth=d, base_features=bf, bottleneck_dropout=bottleneck_dropout,
    )
    task_module = LCZUNetModule(model=model, num_classes=num_classes)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    task_module.model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu"
    )
    task_module = task_module.to(device)

    # ── Optional ROI ──────────────────────────────────────────────────────────
    roi = _parse_bbox_to_shapely(bbox, None, embedding_ds.crs) if bbox else None

    # ── Output path ───────────────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_path).parent
    resolved_output = (
        Path(output_path)
        if output_path
        else ckpt_dir / f"{ckpt_dir.name}_unet-{preset}-prediction.tif"
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    pred_run_config = {"checkpoint": checkpoint_path, "preset": preset, "bbox": bbox}
    if not no_wandb:
        init_wandb_run(
            WandbConfig(project=wandb_project),
            run_config=pred_run_config,
            dir=str(resolved_output.parent),
        )

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_path = predict_roi(
        model=task_module.model,
        embedding_ds=embedding_ds,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        roi=roi,
        stride=stride,
        output_path=resolved_output,
    )

    if not no_wandb:
        log_prediction_raster(pred_path)
        wandb.finish()


if __name__ == "__main__":
    app()
