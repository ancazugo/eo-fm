"""Standalone ResNet-backbone U-Net segmentation trainer for LCZ classification.

Architecture: U-Net decoder with a timm ResNet encoder, configurable backbone
(resnet18 / resnet34 / resnet50).  The ResNet stem is adapted for small spatial
inputs — the standard 7×7 stride-2 conv and stride-2 maxpool are replaced with
a single 3×3 stride-1 conv, reducing the encoder total stride from 32 to 8.
This keeps feature maps large enough for patches as small as 32×32 px.

Skip connections come from the four ResNet layer-group outputs.  The decoder
uses the same DoubleConv blocks as train_unet.py.  All other components
(loss, dataset, datamodule, module, ROI prediction, CLI) are
imported unchanged from train_unet.py.

Usage:
    python src/train_resnet_unet.py train \\
        --embedding tessera \\
        --embedding-path /maps/acz25/phd-thesis-data/input/GeoTessera/2017/ \\
        --label-path /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/patches_reference_Nairobi.gpkg \\
        --label-column LCZ_class \\
        --backbone resnet50 --patch-size 64 --batch-size 8 --num-classes 17 \\
        --bbox "36.45,-1.54,37.16,-0.96" --year 2017
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from loguru import logger

# ── Shared components imported unchanged from train_unet ─────────────────────
from train_unet import (
    DoubleConv,
    MulticlassDiceLoss,
    UNetDataModule,
    LCZUNetModule,
    predict_unet_roi,
    _run_unet_training_loop,
    augment_batch,
    _spatial_split,
)

from conf import WandbConfig
from datasets.labels import LCZLabelDataset, rasterize_gdf
from datasets.registry import create_embedding_dataset, get_in_channels
from utils.wandb import init_wandb_run


app = typer.Typer(pretty_exceptions_enable=False)


# ─── ResNet-UNet Architecture ─────────────────────────────────────────────────

# Encoder output channels per backbone for out_indices=(0,1,2,3)
# i.e. after layer1, layer2, layer3, layer4 respectively.
_BACKBONE_CHANNELS: dict[str, list[int]] = {
    "resnet18": [64,  128,  256,  512],
    "resnet34": [64,  128,  256,  512],
    "resnet50": [256, 512, 1024, 2048],
}


class ResNetUNet(nn.Module):
    """U-Net with a timm ResNet encoder.

    The ResNet stem is adapted for small spatial inputs by replacing the
    standard 7×7 stride-2 conv and stride-2 maxpool with a 3×3 stride-1 conv.
    This reduces the encoder total stride from 32 to 8, making the architecture
    suitable for patches as small as 32×32 px.

    Skip connections are taken from the outputs of ResNet layer1–4.  The decoder
    mirrors the standard U-Net pattern: ConvTranspose2d upsample → concat skip
    → DoubleConv, repeated three times (layer4→layer3→layer2→layer1), ending at
    the full input resolution.

    Supported backbones (``PRESETS``):
        resnet18 — 11 M params (ImageNet baseline)
        resnet34 — 21 M params
        resnet50 — 25 M params (bottleneck blocks, wider feature maps)

    Args:
        in_channels: Embedding input channels (e.g. 128 for Tessera, 64 for AlphaEarth).
        num_classes: Number of segmentation output classes.
        backbone: timm model name; one of the keys in ``PRESETS``.
        bottleneck_dropout: Dropout2d probability applied to the deepest encoder
            features before decoding (0 = off).
        pretrained: Whether to load ImageNet-pretrained weights.  Only meaningful
            when ``in_channels == 3``; ignored (set to False) otherwise since
            embedding inputs differ fundamentally from RGB.
    """

    PRESETS: dict[str, str] = {
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
    }

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        backbone: str = "resnet18",
        bottleneck_dropout: float = 0.3,
        pretrained: bool = False,
    ) -> None:
        import timm

        super().__init__()

        if backbone not in _BACKBONE_CHANNELS:
            raise ValueError(
                f"backbone='{backbone}' not supported. "
                f"Choose from: {list(_BACKBONE_CHANNELS)}"
            )

        # Pretrained weights are only sensible for 3-channel RGB input.
        use_pretrained = pretrained and (in_channels == 3)
        if pretrained and not use_pretrained:
            logger.warning(
                f"pretrained=True ignored for in_channels={in_channels} "
                "(embedding inputs are not RGB — ImageNet weights are irrelevant)."
            )

        # ── Encoder (timm ResNet, feature extraction at each layer group) ────
        self.encoder = timm.create_model(
            backbone,
            pretrained=use_pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # after layer1, layer2, layer3, layer4
        )

        # Adapt stem for small spatial inputs.
        # Original stem: Conv2d(in_ch, 64, 7×7, stride=2) + MaxPool(stride=2) → stride 4.
        # Modified stem: Conv2d(in_ch, 64, 3×3, stride=1) + Identity → stride 1.
        # After this change the four encoder feature strides become ≈ [1, 2, 4, 8]
        # instead of [4, 8, 16, 32], so a 64-px patch yields an 8-px bottleneck.
        stem_out_ch = self.encoder.conv1.out_channels
        self.encoder.conv1 = nn.Conv2d(
            in_channels, stem_out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.encoder.maxpool = nn.Identity()

        enc_ch = _BACKBONE_CHANNELS[backbone]  # [c1, c2, c3, c4] deepest last

        # ── Bottleneck dropout ───────────────────────────────────────────────
        self.bottleneck_drop = (
            nn.Dropout2d(bottleneck_dropout) if bottleneck_dropout > 0 else nn.Identity()
        )

        # ── Decoder: one up-block per skip connection (layer3 → layer2 → layer1) ─
        # enc_ch[-1] is the bottleneck; enc_ch[:-1] are the skip sources (reversed).
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = enc_ch[-1]
        for skip_ch in reversed(enc_ch[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(skip_ch * 2, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)  # [layer1, layer2, layer3, layer4] finest→coarsest

        bottleneck = self.bottleneck_drop(features[-1])
        skips = features[:-1]  # [layer1, layer2, layer3]

        out = bottleneck
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            out = up(out)
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = dec(out)

        return self.head(out)


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
    backbone: str = typer.Option("resnet18", help="ResNet backbone: resnet18, resnet34, resnet50"),
    bottleneck_dropout: float = typer.Option(0.3, help="Dropout2d rate on the deepest encoder features"),
    pretrained: bool = typer.Option(False, help="Load ImageNet-pretrained weights (only for in_channels=3)"),
    num_classes: int = typer.Option(17, help="Number of LCZ segmentation classes"),
    # Loss
    dice_weight: float = typer.Option(0.5, help="Dice loss weight (0=CE only, 1=Dice only)"),
    # Training
    patch_size: int = typer.Option(64, help="Square patch size in pixels (minimum 32)"),
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
    """Train a ResNet-backbone U-Net segmentation model for LCZ classification."""
    import datetime
    import random

    import numpy as np
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
            f"train_resnet_unet.py only supports vector labels (.gpkg / .geojson / .shp). "
            f"For raster labels use: python src/cli.py train-lightning --task segmentation"
        )

    logger.info(f"Loading vector labels from {label_path}")
    gdf = gpd.read_file(label_path).to_crs(embedding_ds.crs)
    logger.info(f"Loaded {len(gdf)} label polygons, CRS={gdf.crs}")

    # Rasterize all polygons into one unified GeoTIFF before splitting.
    label_tmp_dir = Path(tempfile.mkdtemp(prefix="eo_fm_labels_"))
    label_tif_path = label_tmp_dir / "labels.tif"
    rasterize_gdf(gdf, label_column, label_tif_path, res=embedding_res)
    logger.info(f"Rasterized {len(gdf)} polygons → {label_tif_path}")

    label_ds = LCZLabelDataset(
        paths=label_tmp_dir, crs=embedding_ds.crs, res=embedding_ds.res
    )

    # ── Spatial train/val/test split ──────────────────────────────────────────
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

    # ── ResNet-UNet model ─────────────────────────────────────────────────────
    if backbone not in ResNetUNet.PRESETS:
        raise typer.BadParameter(
            f"--backbone must be one of {list(ResNetUNet.PRESETS)}. Got '{backbone}'."
        )

    logger.info(f"Building ResNet-UNet: backbone={backbone}, pretrained={pretrained}")
    model = ResNetUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=backbone,
        bottleneck_dropout=bottleneck_dropout,
        pretrained=pretrained,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ResNet-UNet trainable parameters: {n_params:,}")

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
        "backbone": backbone,
        "pretrained": pretrained,
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
    base_out = Path(output_dir) if output_dir else Path("/tmp/resnet_unet_runs")
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
        model_name=f"resnet_unet-{backbone}",
    )

    # ── Test evaluation ────────────────────────────────────────────────────────
    logger.info("Running test evaluation …")
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
            name=f"resnet_unet-{run_name}", type="model", metadata=run_config
        )
        artifact.add_file(str(best_ckpt_path))
        wandb.log_artifact(artifact)
        logger.info(f"Logged model artifact: resnet_unet-{run_name}")

    # ── ROI prediction ────────────────────────────────────────────────────────
    pred_output = run_dir / f"{run_dir.name}_resnet-unet-{backbone}-segmentation-prediction.tif"
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
    backbone: str = typer.Option("resnet18", help="Backbone used at training time"),
    bottleneck_dropout: float = typer.Option(0.3, help="Bottleneck dropout (must match training)"),
    num_classes: int = typer.Option(17, help="Number of classes (must match training)"),
    bbox: str | None = typer.Option(None, help="ROI 'west,south,east,north' (EPSG:4326)"),
    year: int | None = typer.Option(None, help="Year for temporal filtering"),
    patch_size: int = typer.Option(64, help="Patch size in pixels (must match training)"),
    stride: int | None = typer.Option(None, help="Grid stride in pixels; defaults to patch_size"),
    batch_size: int = typer.Option(8, help="Inference batch size"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    output_path: str | None = typer.Option(None, help="Output GeoTIFF path"),
    accelerator: str = typer.Option("auto", help="Device: auto, gpu, cpu"),
    wandb_project: str = typer.Option("lcz-classification-dl", help="WandB project name"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Disable WandB logging"),
) -> None:
    """Load a ResNet-UNet checkpoint and predict over the full ROI."""
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

    model = ResNetUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=backbone,
        bottleneck_dropout=bottleneck_dropout,
        pretrained=False,
    )
    task_module = LCZUNetModule(model=model, num_classes=num_classes)

    # Load checkpoint — support both new (.pt) and legacy Lightning (.ckpt) formats
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    task_module.model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and accelerator != "cpu" else "cpu"
    )
    task_module = task_module.to(device)

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

    if not no_wandb:
        wandb.init(
            project=wandb_project,
            config={"checkpoint": checkpoint_path, "embedding": embedding,
                    "backbone": backbone, "bbox": bbox},
        )

    ckpt_dir = Path(checkpoint_path).parent
    pred_name = f"{ckpt_dir.name}_resnet-unet-{backbone}-segmentation-prediction.tif"
    resolved_output = Path(output_path) if output_path else ckpt_dir / pred_name

    pred_path = predict_unet_roi(
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
