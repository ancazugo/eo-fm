"""Standalone U-Net segmentation trainer for LCZ classification.

Architecture: U-Net with configurable presets (nano/small/base/medium/large),
multiclass Dice + CrossEntropy loss, ignore_index=-1 for unlabeled pixels.

Designed for sparse vector labels (So2Sat GeoPackage): polygons are rasterized
per patch so the network sees full spatial masks, not just patch-level scalars.

Usage:
    python src/train_unet.py train \\
        --embedding tessera \\
        --embedding-path /maps/acz25/phd-thesis-data/input/GeoTessera/2017/ \\
        --label-path /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/patches_reference_Nairobi.gpkg \\
        --label-column LCZ_class \\
        --preset small --patch-size 64 --batch-size 8 --num-classes 17 \\
        --bbox "36.45,-1.54,37.16,-0.96" --year 2017
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from loguru import logger
from torch.utils.data import DataLoader
from torchgeo.datasets.geo import GeoDataset
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, Units
from torchmetrics import Accuracy, JaccardIndex

from conf import WandbConfig
from datasets.labels import LCZLabelDataset, rasterize_gdf
from datasets.registry import create_embedding_dataset, get_in_channels
from utils.wandb import init_wandb_run


app = typer.Typer(pretty_exceptions_enable=False)


# ─── U-Net Architecture ───────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv2d(3×3) → BatchNorm → ReLU blocks.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        dropout: Dropout2d probability applied after the second ReLU (0 = off).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net with configurable depth and feature width.

    Encoder: DoubleConv → MaxPool2d (×depth).
    Bottleneck: DoubleConv (with dropout).
    Decoder: ConvTranspose2d → skip-cat → DoubleConv (×depth).
    Head: Conv2d(1×1) → num_classes logits.

    Built-in presets (depth, base_features):
        nano:   (2,  8)  — fast experiments, minimal params
        small:  (3, 32)  — lightweight baseline
        base:   (3, 48)  — wider baseline (mirrors tessera-cnn-example "base")
        medium: (4, 32)  — deeper with moderate width
        large:  (4, 48)  — deepest + widest recommended preset

    Args:
        in_channels: Number of embedding input channels.
        num_classes: Number of segmentation output classes.
        depth: Number of encoder/decoder stages.
        base_features: Feature maps at the first encoder stage;
            doubles at each subsequent stage.
        bottleneck_dropout: Dropout2d probability at the bottleneck.
    """

    PRESETS: dict[str, tuple[int, int]] = {
        "nano":   (2,  8),
        "small":  (3, 32),
        "base":   (3, 48),
        "medium": (4, 32),
        "large":  (4, 48),
    }

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 3,
        base_features: int = 32,
        bottleneck_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.depth = depth

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        enc_channels: list[int] = []
        ch = in_channels
        for i in range(depth):
            out_ch = base_features * (2 ** i)
            self.encoders.append(DoubleConv(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            enc_channels.append(out_ch)
            ch = out_ch

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_ch = base_features * (2 ** depth)
        self.bottleneck = DoubleConv(ch, bottleneck_ch, dropout=bottleneck_dropout)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = bottleneck_ch
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            self.upsamples.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(skip_ch * 2, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            # Correct for odd-sized inputs (bilinear resize if needed)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


# ─── Loss Functions ───────────────────────────────────────────────────────────

class MulticlassDiceLoss(nn.Module):
    """Soft multiclass Dice loss, macro-averaged over present classes.

    For each class, computes the soft Dice coefficient over valid pixels
    (those not equal to ``ignore_index``), then averages over classes that
    have at least one positive target pixel. Returns 0 if no valid class exists.

    Args:
        num_classes: Number of segmentation classes.
        ignore_index: Label value to exclude from loss computation.
        smooth: Laplace smoothing term to avoid division by zero.
    """

    def __init__(
        self, num_classes: int, ignore_index: int = -1, smooth: float = 1.0
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C, H, W) — unnormalized class logits.
            targets: (N, H, W)    — integer class indices; ignore_index excluded.
        """
        probs = F.softmax(logits, dim=1)          # (N, C, H, W)
        valid = targets != self.ignore_index       # (N, H, W) bool

        dice_terms: list[torch.Tensor] = []
        for c in range(self.num_classes):
            target_c = ((targets == c) & valid).float()   # (N, H, W)
            pred_c = probs[:, c]                          # (N, H, W)

            p = pred_c[valid]
            t = target_c[valid]

            if t.numel() == 0 or t.sum() == 0:
                continue  # absent class — skip so macro average is fair

            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth
            )
            dice_terms.append(1.0 - dice)

        if not dice_terms:
            return logits.sum() * 0.0  # differentiable zero

        return torch.stack(dice_terms).mean()


# ─── Augmentation (notebook-style: exact pixel ops, no interpolation) ──────────

def augment_batch(
    images: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random flips and exact 90° rotations per sample (no interpolation).

    All geometric transforms use exact pixel operations (flip, rot90) so mask
    integer values (0-16 class indices, -1 nodata) are preserved exactly.
    Kornia's RandomRotation uses bilinear interpolation which corrupts integer
    masks and causes stitching artifacts at patch boundaries.

    Augmentations applied per sample:
    - Random horizontal flip (p=0.5)
    - Random vertical flip (p=0.5)
    - Random 90° rotation k ∈ {0,1,2,3} (uniform, exact pixel op)
    - Gaussian noise σ=0.05 on image only (p=0.5)

    Args:
        images: (N, C, H, W) float tensor on any device.
        masks:  (N, H, W) long tensor, values 0–16 or -1 (nodata).

    Returns:
        Augmented (images, masks) with the same shapes and dtypes.
    """
    aug_images, aug_masks = [], []
    for img, msk in zip(images, masks):
        if torch.rand(1) < 0.5:
            img = img.flip(-1)
            msk = msk.flip(-1)
        if torch.rand(1) < 0.5:
            img = img.flip(-2)
            msk = msk.flip(-2)
        k = torch.randint(0, 4, (1,)).item()
        if k:
            img = torch.rot90(img, k, dims=(-2, -1))
            msk = torch.rot90(msk, k, dims=(-2, -1))
        if torch.rand(1) < 0.5:
            img = img + torch.randn_like(img) * 0.05
        aug_images.append(img)
        aug_masks.append(msk)
    return torch.stack(aug_images), torch.stack(aug_masks)


# ─── Spatial split helper ──────────────────────────────────────────────────────

def _spatial_split(gdf, val_fraction: float = 0.15, test_fraction: float = 0.15):
    """Split GeoDataFrame spatially by centroid x-coordinate.

    Follows the torchgeo notebook pattern: split at quantile thresholds along x
    so the three ROIs are non-overlapping geographic regions. Eliminates spatial
    data leakage that polygon-level random splits cannot avoid.

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

class UNetDataModule:
    """DataModule following the torchgeo notebook pattern.

    Uses a single pre-rasterized unified label GeoTIFF with spatial ROI-based
    train/val/test splits. This avoids per-patch rasterization inconsistencies
    at polygon boundaries and the interpolation artifacts from Kornia's
    RandomRotation (which bilinearly interpolates integer mask values).

    Label convention:
    - LCZLabelDataset returns values 1-17 (1-indexed), 0 = nodata.
    - Collate shifts by -1: classes 0-16, nodata -1 (= ignore_index in losses).

    Args:
        embedding_ds: GeoDataset providing embedding tensors.
        label_ds: LCZLabelDataset (RasterDataset with is_image=False).
            Returns masks with values 1-17 (1-indexed) and 0 for nodata.
        train_roi: Shapely geometry restricting training sampler.
        val_roi: Shapely geometry restricting validation sampler.
        test_roi: Shapely geometry restricting test sampler.
        patch_size: Square patch size in pixels.
        batch_size: Batch size for all loaders.
        length: Training patches per epoch (RandomBatchGeoSampler).
        num_workers: DataLoader worker processes.
        augment: Apply random flips + exact 90° rotation during training.
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
        self.dataset: "IntersectionDataset | None" = None

    def setup(self) -> None:
        # Single combined dataset; ROI restriction is handled by the samplers.
        self.dataset = self.embedding_ds & self.label_ds

    @staticmethod
    def _collate(batch: list[dict]) -> dict:
        from torchgeo.datasets.utils import stack_samples
        collated = stack_samples(batch)
        if "mask" in collated:
            # LCZLabelDataset: 1-17 (1-indexed), nodata=0.
            # Shift: 1-17 → 0-16, nodata 0 → -1 (= ignore_index in losses).
            collated["mask"] = collated["mask"].squeeze(1).long() - 1
        return collated

    def _train_collate(self, batch: list[dict]) -> dict:
        collated = self._collate(batch)
        if not self.augment:
            return collated
        images, masks = augment_batch(
            collated["image"].float(), collated["mask"]
        )
        collated["image"] = images
        collated["mask"] = masks
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

class LCZUNetModule(nn.Module):
    """nn.Module wrapping UNet for multiclass LCZ segmentation.

    Loss: ``(1 − dice_weight) × CrossEntropy + dice_weight × MulticlassDice``
    Both losses use ``ignore_index=-1`` to skip unlabeled pixels.

    Metrics: val_miou (macro mIoU), val_acc (per-pixel accuracy).

    Args:
        model: UNet instance.
        num_classes: Number of segmentation classes.
        lr: Adam learning rate.
        weight_decay: Adam L2 regularization.
        dice_weight: Weighting of Dice loss (0 = CE only, 1 = Dice only).
        max_epochs: Total training epochs (used for CosineAnnealingLR T_max).
    """

    def __init__(
        self,
        model: UNet,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dice_weight: float = 0.5,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dice_weight = dice_weight
        self.max_epochs = max_epochs

        self.dice_loss = MulticlassDiceLoss(num_classes, ignore_index=-1)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        metric_kw = dict(task="multiclass", num_classes=num_classes, ignore_index=-1)
        self.val_miou = JaccardIndex(**metric_kw, average="macro")
        self.val_acc = Accuracy(**metric_kw)
        self.test_miou = JaccardIndex(**metric_kw, average="macro")
        self.test_acc = Accuracy(**metric_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.float())

    def _loss(
        self, logits: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        combined = (1.0 - self.dice_weight) * ce + self.dice_weight * dice
        return combined, ce, dice


# ─── Training Loop ────────────────────────────────────────────────────────────

def _run_unet_training_loop(
    task_module: LCZUNetModule,
    datamodule: UNetDataModule,
    device: torch.device,
    max_epochs: int,
    early_stopping_patience: int,
    run_dir: Path,
    model_name: str,
) -> tuple[LCZUNetModule, Path | None]:
    """Run the pure-PyTorch training loop for a LCZUNetModule.

    Args:
        task_module: The LCZUNetModule to train (moved to device inside).
        datamodule: UNetDataModule (setup() called inside).
        device: Device to train on.
        max_epochs: Maximum number of epochs.
        early_stopping_patience: Stop after this many epochs without val_miou improvement.
        run_dir: Directory to save checkpoints.
        model_name: Stem for checkpoint filename.

    Returns:
        (task_module_with_best_weights, best_ckpt_path)
    """
    import wandb

    task_module = task_module.to(device)
    opt = torch.optim.Adam(task_module.parameters(), lr=task_module.lr, weight_decay=task_module.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    best_miou = -1.0
    patience_counter = 0
    best_ckpt_path: Path | None = None

    for epoch in range(max_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        task_module.train()
        train_total_loss = train_ce = train_dice = 0.0
        n_valid = 0
        for batch in train_loader:
            images = batch["image"].to(device).float()
            masks = batch["mask"].to(device)
            # Skip all-nodata batches (CE returns NaN when every pixel is ignored)
            if (masks != -1).sum() == 0:
                continue
            opt.zero_grad()
            logits = task_module.model(images)
            loss, ce, dice = task_module._loss(logits, masks)
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()
            train_total_loss += loss.item()
            train_ce += ce.item() if not torch.isnan(ce) else 0.0
            train_dice += dice.item()
            n_valid += 1
        n = max(1, n_valid)
        train_total_loss /= n
        train_ce /= n
        train_dice /= n

        # ── Validate ───────────────────────────────────────────────────────
        task_module.eval()
        task_module.val_miou.reset()
        task_module.val_acc.reset()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device).float()
                masks = batch["mask"].to(device)
                logits = task_module.model(images)
                loss, _, _ = task_module._loss(logits, masks)
                preds = logits.argmax(dim=1)
                task_module.val_miou(preds, masks)
                task_module.val_acc(preds, masks)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(1, n_val)
        val_miou = task_module.val_miou.compute().item()
        val_acc = task_module.val_acc.compute().item()
        sched.step()

        if wandb.run:
            wandb.log({
                "train_loss": train_total_loss,
                "train_ce": train_ce,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            })
        logger.info(
            f"Epoch {epoch+1}/{max_epochs}  "
            f"loss={train_total_loss:.4f}  val_miou={val_miou:.4f}  val_acc={val_acc:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            best_ckpt_path = run_dir / f"{model_name}-best.pt"
            torch.save(
                {"model_state_dict": task_module.model.state_dict(), "epoch": epoch + 1, "val_miou": val_miou},
                best_ckpt_path,
            )
            logger.info(f"  → New best (val_miou={val_miou:.4f}), checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_ckpt_path and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        task_module.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model (val_miou={ckpt['val_miou']:.4f}) from {best_ckpt_path}")

    return task_module, best_ckpt_path


# ─── ROI Prediction ──────────────────────────────────────────────────────────

def predict_unet_roi(
    task_module: "LCZUNetModule",
    embedding_ds: GeoDataset,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    roi=None,
    stride: int | None = None,
    output_path: str | Path = "unet_prediction.tif",
    tile_border_trim: int | tuple[int, int, int, int] = 0,
) -> Path:
    """Run U-Net over the full embedding ROI and write a GeoTIFF.

    Mirrors the torchgeo notebook inference pattern exactly:
    - GridGeoSampler over embedding_ds (not intersection dataset)
    - Direct argmax placement (no softmax, no probability averaging)
    - embedding_ds.res[0] as the single canonical resolution

    Args:
        task_module: Trained LCZUNetModule (eval mode set internally).
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

    # Use dataset resolution directly — avoids floating-point drift from
    # deriving res from individual patch bounds across tile boundaries.
    res = embedding_ds.res[0]

    # Single-pass inference following the torchgeo notebook pattern exactly.
    # batch["bounds"] from stack_samples is a (N, 9) tensor:
    #   [minx, maxx, xstep, miny, maxy, ystep, mint, maxt, tstep]
    # Tuple stored as: (pred_hw, minx, miny, maxx, maxy)
    patch_results: list[tuple] = []

    total = len(sampler) // batch_size + int(len(sampler) % batch_size > 0)
    logger.info(f"Predicting {len(sampler)} patches ({total} batches) …")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch["image"].to(device).float()
            preds = task_module.model(imgs).argmax(dim=1).cpu().numpy()  # (N, H, W)
            bounds = batch["bounds"]  # (N, 9) tensor
            for j in range(preds.shape[0]):
                minx = float(bounds[j, 0])
                maxx = float(bounds[j, 1])
                miny = float(bounds[j, 3])
                maxy = float(bounds[j, 4])
                patch_results.append((preds[j], minx, miny, maxx, maxy))
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

    for pred, minx, miny, maxx, maxy in patch_results:
        col = int(round((minx - all_minx) / res))
        row = int(round((all_maxy - maxy) / res))
        ph, pw = pred.shape
        raster[row:row + ph, col:col + pw] = pred

    # Apply tile-border fill if requested — replaces contaminated tile-edge
    # predictions with nearest interior prediction via distance transform.
    # Same fix as _fill_tile_borders() in sklearn_pixel.py.
    if tile_border_trim:
        from models.sklearn_pixel import _fill_tile_borders
        trim_px = tile_border_trim if isinstance(tile_border_trim, tuple) else (tile_border_trim,) * 4
        # raster is int16 with -1=nodata; convert to uint8 temporarily for the fill
        # (values are 0-indexed class ids 0..num_classes-1)
        fill_raster = np.where(raster >= 0, raster, 0).astype(np.uint8)
        fill_raster = _fill_tile_borders(
            fill_raster, embedding_ds.index.geometry,
            all_minx, all_maxy, out_h, out_w, res, res, trim_px,
        )
        raster = np.where(raster >= 0, fill_raster.astype(np.int16), raster)

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
    label_path: str = typer.Option(..., help="Path to vector label file (.gpkg / .geojson / .shp)"),
    label_column: str = typer.Option("LCZ_class", help="Column with 1-based integer class labels"),
    bbox: str | None = typer.Option(None, help="ROI bounding box 'west,south,east,north' (EPSG:4326)"),
    year: int | None = typer.Option(None, help="Year for temporal filtering of embedding tiles"),
    # Model
    preset: str = typer.Option("small", help="U-Net size preset: nano, small, base, medium, large"),
    depth: int | None = typer.Option(None, help="Override preset encoder depth"),
    base_features: int | None = typer.Option(None, help="Override preset base feature count"),
    bottleneck_dropout: float = typer.Option(0.3, help="Dropout2d rate at the bottleneck block"),
    num_classes: int = typer.Option(17, help="Number of LCZ segmentation classes"),
    # Loss
    dice_weight: float = typer.Option(0.5, help="Dice loss weight (0=CE only, 1=Dice only)"),
    # Training
    patch_size: int = typer.Option(64, help="Square patch size in pixels"),
    batch_size: int = typer.Option(8, help="Training and evaluation batch size"),
    length: int = typer.Option(500, help="Training patches per epoch (RandomBatchGeoSampler)"),
    num_workers: int = typer.Option(4, help="DataLoader worker processes"),
    augment: bool = typer.Option(True, help="Random flips + 90° rotation during training"),
    lr: float = typer.Option(1e-3, help="Adam learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Adam weight decay (L2 regularization)"),
    max_epochs: int = typer.Option(50, help="Maximum training epochs"),
    # Splits
    val_size: float = typer.Option(0.15, help="Fraction of polygons per class for validation"),
    test_size: float = typer.Option(0.15, help="Fraction of polygons per class for test"),
    seed: int = typer.Option(411, help="Random seed for polygon splits"),
    # Callbacks
    early_stopping_patience: int = typer.Option(10, help="EarlyStopping patience (monitors val_miou)"),
    accelerator: str = typer.Option("auto", help="Device: auto, gpu, cpu"),
    output_dir: str | None = typer.Option(None, help="Directory for checkpoint files"),
    # Logging
    wandb_project: str = typer.Option("lcz-classification-dl", help="WandB project name"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Disable WandB logging"),
) -> None:
    """Train a U-Net segmentation model for LCZ classification."""
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

    # ── Labels: rasterize vector file into a unified GeoTIFF ──────────────────
    import geopandas as gpd
    import tempfile

    label_path_obj = Path(label_path)
    if label_path_obj.suffix not in (".gpkg", ".geojson", ".shp", ".json"):
        raise typer.BadParameter(
            f"train_unet.py only supports vector labels (.gpkg / .geojson / .shp). "
            f"For raster labels use: python src/cli.py train-lightning --task segmentation"
        )

    logger.info(f"Loading vector labels from {label_path}")
    gdf = gpd.read_file(label_path).to_crs(embedding_ds.crs)
    logger.info(f"Loaded {len(gdf)} label polygons, CRS={gdf.crs}")

    # Rasterize all polygons into one unified GeoTIFF before splitting.
    # This follows the torchgeo notebook approach: the same pixel always gets
    # the same label value regardless of which split it falls in, eliminating
    # the ambiguity that arises when overlapping polygons are rasterised
    # independently per split.
    label_tmp_dir = Path(tempfile.mkdtemp(prefix="eo_fm_labels_"))
    label_tif_path = label_tmp_dir / "labels.tif"
    rasterize_gdf(gdf, label_column, label_tif_path, res=embedding_res)
    logger.info(f"Rasterized {len(gdf)} polygons → {label_tif_path}")

    label_ds = LCZLabelDataset(
        paths=label_tmp_dir, crs=embedding_ds.crs, res=embedding_ds.res
    )

    # ── Spatial train/val/test split ──────────────────────────────────────────
    # Split by centroid x-coordinate quantiles so the three ROIs are
    # non-overlapping geographic regions (no spatial data leakage).
    train_roi, val_roi, test_roi = _spatial_split(
        gdf, val_fraction=val_size, test_fraction=test_size
    )
    cx = gdf.geometry.centroid.x
    train_frac = 1.0 - val_size - test_size
    n_train = (cx <= cx.quantile(train_frac)).sum()
    n_val   = ((cx > cx.quantile(train_frac)) & (cx <= cx.quantile(train_frac + val_size))).sum()
    n_test  = len(gdf) - n_train - n_val
    logger.info(f"Spatial split: ~{n_train} train / ~{n_val} val / ~{n_test} test polygons")

    datamodule = UNetDataModule(
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

    # ── U-Net model ───────────────────────────────────────────────────────────
    d, bf = UNet.PRESETS.get(preset, (3, 32))
    if depth is not None:
        d = depth
    if base_features is not None:
        bf = base_features

    logger.info(f"Building U-Net: preset={preset}, depth={d}, base_features={bf}")
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=d,
        base_features=bf,
        bottleneck_dropout=bottleneck_dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"U-Net trainable parameters: {n_params:,}")

    task_module = LCZUNetModule(
        model=model,
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
        dice_weight=dice_weight,
        max_epochs=max_epochs,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    run_config = {
        "embedding": embedding,
        "preset": preset,
        "depth": d,
        "base_features": bf,
        "num_classes": num_classes,
        "patch_size": patch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dice_weight": dice_weight,
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
    base_out = Path(output_dir) if output_dir else Path("/tmp/unet_runs")
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
    task_module, best_ckpt_path = _run_unet_training_loop(
        task_module=task_module,
        datamodule=datamodule,
        device=device,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        run_dir=run_dir,
        model_name=f"unet-{preset}",
    )

    # ── Test evaluation ────────────────────────────────────────────────────────
    logger.info("Running test evaluation …")
    # Ensure test_dataset is set up (setup() was called inside training loop)
    task_module.eval()
    task_module.test_miou.reset()
    task_module.test_acc.reset()
    test_preds: list[torch.Tensor] = []
    test_labels: list[torch.Tensor] = []
    test_loss_total = 0.0
    n_test = 0

    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            images = batch["image"].to(device).float()
            masks = batch["mask"].to(device)
            logits = task_module.model(images)
            loss, _, _ = task_module._loss(logits, masks)
            preds = logits.argmax(dim=1)
            task_module.test_miou(preds, masks)
            task_module.test_acc(preds, masks)
            test_loss_total += loss.item()
            n_test += 1
            test_preds.append(preds.cpu())
            test_labels.append(masks.cpu())

    test_loss = test_loss_total / max(1, n_test)
    test_miou = task_module.test_miou.compute().item()
    test_acc = task_module.test_acc.compute().item()
    logger.info(f"Test results: loss={test_loss:.4f}  miou={test_miou:.4f}  acc={test_acc:.4f}")

    if wandb.run:
        wandb.log({"test_loss": test_loss, "test_miou": test_miou, "test_acc": test_acc})

    # Log confusion matrix from test predictions
    if wandb.run and test_preds:
        y_pred_all = torch.cat(test_preds).numpy().ravel()
        y_true_all = torch.cat(test_labels).numpy().ravel()
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
            name=f"unet-{run_name}",
            type="model",
            metadata=run_config,
        )
        artifact.add_file(str(best_ckpt_path))
        wandb.log_artifact(artifact)
        logger.info(f"Logged model artifact: unet-{run_name}")

    # ── ROI prediction ────────────────────────────────────────────────────────
    pred_output = run_dir / f"{run_dir.name}_unet-{preset}-segmentation-prediction.tif"
    logger.info(f"Running ROI prediction → {pred_output}")
    predict_unet_roi(
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
    preset: str = typer.Option("small", help="U-Net preset used at training time"),
    depth: int | None = typer.Option(None, help="Override preset depth (must match training)"),
    base_features: int | None = typer.Option(None, help="Override preset base features (must match training)"),
    bottleneck_dropout: float = typer.Option(0.3, help="Bottleneck dropout (must match training)"),
    num_classes: int = typer.Option(17, help="Number of classes (must match training)"),
    bbox: str | None = typer.Option(None, help="ROI 'west,south,east,north' (EPSG:4326)"),
    year: int | None = typer.Option(None, help="Year for temporal filtering"),
    patch_size: int = typer.Option(64, help="Patch size in pixels (must match training)"),
    stride: int | None = typer.Option(None, help="Grid stride in pixels; defaults to patch_size"),
    batch_size: int = typer.Option(8, help="Inference batch size"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    output_path: str | None = typer.Option(None, help="Output GeoTIFF path. Defaults to <checkpoint_dir>/<checkpoint_dir.name>_unet-<preset>-segmentation-prediction.tif"),
    accelerator: str = typer.Option("auto", help="Device: auto, gpu, cpu"),
    wandb_project: str = typer.Option("lcz-classification-dl", help="WandB project name"),
    tile_border_trim: str = typer.Option("0", help="Pixels to trim near tessera tile edges before nearest-interior fill. Single int N or 'N,S,E,W'."),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Disable WandB logging"),
) -> None:
    """Load a U-Net checkpoint and predict over the full ROI."""
    import wandb

    # ── Embedding dataset ────────────────────────────────────────────────────
    bbox_tuple: tuple[float, float, float, float] | None = None
    if bbox:
        parts = [float(v) for v in bbox.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter("--bbox must be 'west,south,east,north'")
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])

    logger.info(f"Loading {embedding} embeddings from {embedding_path}")
    embedding_ds = create_embedding_dataset(embedding, embedding_path, bbox=bbox_tuple)

    # ── Reconstruct model (same architecture as training) ────────────────────
    in_channels = get_in_channels(embedding)
    d, bf = UNet.PRESETS.get(preset, (3, 32))
    if depth is not None:
        d = depth
    if base_features is not None:
        bf = base_features

    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=d,
        base_features=bf,
        bottleneck_dropout=bottleneck_dropout,
    )
    task_module = LCZUNetModule(model=model, num_classes=num_classes)

    # Load checkpoint — support both new (.pt) and legacy Lightning (.ckpt) formats
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    task_module.model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # Move to device
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
            config={"checkpoint": checkpoint_path, "embedding": embedding,
                    "preset": preset, "depth": d, "base_features": bf, "bbox": bbox},
        )

    # ── Resolve output path ───────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_path).parent
    pred_name = f"{ckpt_dir.name}_unet-{preset}-segmentation-prediction.tif"
    resolved_output = Path(output_path) if output_path else ckpt_dir / pred_name

    # Parse tile_border_trim
    _trim_parts = [int(v) for v in tile_border_trim.split(",")]
    trim: int | tuple[int, int, int, int] = (
        tuple(_trim_parts) if len(_trim_parts) == 4 else _trim_parts[0]  # type: ignore[assignment]
    )

    # ── Predict ──────────────────────────────────────────────────────────────
    pred_path = predict_unet_roi(
        task_module=task_module,
        embedding_ds=embedding_ds,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        roi=roi,
        stride=stride,
        output_path=resolved_output,
        tile_border_trim=trim,
    )

    if not no_wandb:
        from utils.wandb import log_prediction_raster
        log_prediction_raster(pred_path)
        wandb.finish()


if __name__ == "__main__":
    app()
