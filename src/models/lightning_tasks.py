"""Pure-PyTorch model builders for classification and segmentation."""

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn as nn
from loguru import logger
from rasterio.transform import from_bounds
from sklearn.metrics import confusion_matrix

from conf import TrainConfig
from datasets.registry import get_in_channels


def build_model(
    config: "TrainConfig",
    embedding_name: str,
    class_weights: "torch.Tensor | None" = None,
) -> tuple["torch.nn.Module", "torch.nn.Module"]:
    """Return (model, loss_fn) for the given task config.

    Args:
        config: Training configuration.
        embedding_name: Name of the embedding (used to look up in_channels).
        class_weights: Optional 1-D tensor of per-class loss weights (length =
            num_classes, 0-indexed).  Pass None to use uniform weighting.

    Returns:
        model: nn.Module (timm for classification, SMP for segmentation)
        loss_fn: CrossEntropyLoss with class_weights and ignore_index=-1
    """
    in_channels = get_in_channels(embedding_name)

    if class_weights is not None:
        logger.info(f"Class weights: {class_weights.tolist()}")

    if config.task == "classification":
        import timm
        pretrained = config.weights is not None and config.weights.lower() in ("true", "imagenet")
        model = timm.create_model(
            config.model,
            in_chans=in_channels,
            num_classes=config.num_classes,
            pretrained=pretrained,
        )
        if config.weights and config.weights.lower() not in ("true", "imagenet"):
            # torchgeo weight enum — load into the model
            from torchgeo.models import get_weight
            weights = get_weight(config.weights)
            model = weights.get_transform()(model)
            logger.info(f"Loaded pretrained weights: {config.weights}")
    elif config.task == "segmentation":
        import segmentation_models_pytorch as smp
        backbone = config.backbone or "resnet50"
        encoder_weights = None
        if config.weights and config.weights.lower() in ("true", "imagenet"):
            encoder_weights = "imagenet"
        model = smp.create_model(
            config.model,
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=config.num_classes,
        )
    else:
        raise ValueError(f"Unknown task: {config.task!r}. Choose 'classification' or 'segmentation'.")

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-1,
    )
    return model, loss_fn


def predict_dl_roi(
    task_module: "torch.nn.Module",
    datamodule,
    task_type: str,
    output_path: str | Path = "prediction.tif",
    pred_resolution: str = "pixel",
    device: "torch.device | None" = None,
) -> dict[str, Any]:
    """Predict over the full ROI using a plain nn.Module and write a GeoTIFF.

    Uses the predict dataloader (GridGeoSampler) for full-coverage prediction.

    Args:
        task_module: Trained nn.Module.
        datamodule: EmbeddingLabelDataModule with test_roi set.
        task_type: "classification" or "segmentation".
        output_path: Path to write the prediction GeoTIFF.
        pred_resolution: Controls output resolution for classification tasks.
            "pixel" — each patch is written as a filled patch_size×patch_size
            block at embedding resolution (e.g. 10 m).
            "patch" — one pixel per patch at patch resolution (e.g. 320 m),
            matching the spatial granularity of the classifier.
            Has no effect for segmentation tasks.
        device: Device to run inference on. Defaults to the model's current device.

    Returns:
        Dict with keys: y_true, y_pred, confusion_matrix, raster_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure datasets are initialised
    if datamodule.test_dataset is None:
        datamodule.setup()

    device = device or next(task_module.parameters()).device
    crs = getattr(datamodule.test_dataset, "crs", None)

    # Use dataset resolution directly — avoids floating-point drift from
    # deriving res from individual patch bounds across tile boundaries.
    res = getattr(datamodule.embedding_ds, "res", None)
    res = float(res[0]) if res is not None else None

    if task_type == "segmentation":
        # Mirrors the torchgeo notebook inference pattern exactly:
        # direct argmax, direct placement, embedding_ds.res[0] as canonical res.
        # batch["bounds"] from stack_samples is a (N, 9) tensor:
        #   [minx, maxx, xstep, miny, maxy, ystep, mint, maxt, tstep]
        # Tuple stored as: (pred_hw, minx, miny, maxx, maxy)
        patch_results: list[tuple] = []

        task_module.eval()
        with torch.no_grad():
            for batch in datamodule.predict_dataloader():
                imgs = batch["image"].to(device).float()
                preds = task_module(imgs).argmax(dim=1).cpu().numpy()  # (N, H, W)
                bounds = batch["bounds"]  # (N, 9) tensor
                for i in range(preds.shape[0]):
                    minx = float(bounds[i, 0])
                    maxx = float(bounds[i, 1])
                    miny = float(bounds[i, 3])
                    maxy = float(bounds[i, 4])
                    patch_results.append((preds[i], minx, miny, maxx, maxy))

        if not patch_results:
            raise RuntimeError("No predictions produced by the segmentation model.")

        all_minx = min(p[1] for p in patch_results)
        all_miny = min(p[2] for p in patch_results)
        all_maxx = max(p[3] for p in patch_results)
        all_maxy = max(p[4] for p in patch_results)

        if res is None:
            first = patch_results[0]
            ph, pw = first[0].shape
            res = (first[3] - first[1]) / pw  # fallback: derive from first patch

        out_w = int(round((all_maxx - all_minx) / res))
        out_h = int(round((all_maxy - all_miny) / res))
        raster = np.full((out_h, out_w), fill_value=-1, dtype=np.int16)

        for pred, minx, miny, maxx, maxy in patch_results:
            col = int(round((minx - all_minx) / res))
            row = int(round((all_maxy - maxy) / res))
            ph, pw = pred.shape
            raster[row:row + ph, col:col + pw] = pred

        output_raster = np.where(raster >= 0, raster + 1, 0).astype(np.uint8)
        y_pred = output_raster.ravel()

        transform = from_bounds(all_minx, all_miny, all_maxx, all_maxy, out_w, out_h)
        with rasterio.open(
            str(output_path), "w", driver="GTiff",
            height=out_h, width=out_w, count=1, dtype="uint8",
            crs=crs, transform=transform, nodata=0,
        ) as dst:
            dst.write(output_raster, 1)
        logger.info(f"Prediction raster saved to {output_path} ({out_h}×{out_w})")

    else:
        # Classification: single-pass inference, same notebook-style placement.
        patch_results: list[tuple] = []  # (cls_1based, minx, miny, maxx, maxy)
        patch_px: int | None = None

        task_module.eval()
        with torch.no_grad():
            for batch in datamodule.predict_dataloader():
                imgs = batch["image"].to(device).float()
                if patch_px is None:
                    patch_px = imgs.shape[-1]
                preds = task_module(imgs).argmax(dim=1).cpu().numpy() + 1  # 1-based
                bounds = batch["bounds"]  # (N, 9) tensor
                for i in range(preds.shape[0]):
                    minx = float(bounds[i, 0])
                    maxx = float(bounds[i, 1])
                    miny = float(bounds[i, 3])
                    maxy = float(bounds[i, 4])
                    patch_results.append((int(preds[i]), minx, miny, maxx, maxy))

        y_pred = np.array([r[0] for r in patch_results])

        if patch_results and patch_px:
            all_minx = min(r[1] for r in patch_results)
            all_miny = min(r[2] for r in patch_results)
            all_maxx = max(r[3] for r in patch_results)
            all_maxy = max(r[4] for r in patch_results)

            if res is None:
                first = patch_results[0]
                res = (first[3] - first[1]) / patch_px  # fallback

            if pred_resolution == "patch":
                # One pixel per patch — resolution matches label granularity
                patch_res = res * patch_px
                out_w = int(round((all_maxx - all_minx) / patch_res))
                out_h = int(round((all_maxy - all_miny) / patch_res))
                raster = np.full((out_h, out_w), fill_value=0, dtype=np.uint8)
                for cls, minx, miny, maxx, maxy in patch_results:
                    col = int(round((minx - all_minx) / patch_res))
                    row = int(round((all_maxy - maxy) / patch_res))
                    raster[row, col] = cls
                output_raster = raster
            else:
                # One pixel per embedding pixel — fill patch_px×patch_px block
                out_w = int(round((all_maxx - all_minx) / res))
                out_h = int(round((all_maxy - all_miny) / res))
                raster = np.full((out_h, out_w), fill_value=0, dtype=np.uint8)
                for cls, minx, miny, maxx, maxy in patch_results:
                    col = int(round((minx - all_minx) / res))
                    row = int(round((all_maxy - maxy) / res))
                    raster[row:row + patch_px, col:col + patch_px] = cls
                output_raster = raster

            transform = from_bounds(all_minx, all_miny, all_maxx, all_maxy,
                                    output_raster.shape[1], output_raster.shape[0])
        else:
            output_raster = y_pred.reshape(1, len(y_pred)).astype(np.uint8)
            transform = None

        with rasterio.open(
            str(output_path), "w", driver="GTiff",
            height=output_raster.shape[0], width=output_raster.shape[1],
            count=1, dtype="uint8", crs=crs, transform=transform, nodata=0,
        ) as dst:
            dst.write(output_raster, 1)
        logger.info(f"Prediction raster saved to {output_path} ({output_raster.shape[0]}×{output_raster.shape[1]})")

    # Confusion matrix — separate inference pass over test patches only.
    cm_truths: list[np.ndarray] = []
    cm_preds: list[np.ndarray] = []

    if task_type == "classification":
        task_module.eval()
        for batch in datamodule.test_dataloader():
            if "label" not in batch:
                continue
            imgs = batch["image"].to(device)
            with torch.no_grad():
                logits = task_module(imgs)
            pred_cls = logits.argmax(dim=1).cpu().numpy() + 1  # 1-based
            true_cls = (batch["label"].numpy() + 1).ravel()    # 1-based
            valid = true_cls > 0
            if valid.any():
                cm_truths.append(true_cls[valid])
                cm_preds.append(pred_cls.ravel()[valid])
    else:
        # Segmentation: collect test masks
        for batch in datamodule.test_dataloader():
            if "mask" not in batch:
                continue
            mask = (batch["mask"].numpy() + 1).ravel()
            valid = mask > 0
            if valid.any():
                cm_truths.append(mask[valid])
        y_pred_flat = y_pred.ravel()
        y_true_flat = np.concatenate(cm_truths) if cm_truths else np.array([], dtype=int)
        if len(y_true_flat) > 0 and len(y_true_flat) == len(y_pred_flat):
            valid = y_true_flat > 0
            cm_truths = [y_true_flat[valid]]
            cm_preds = [y_pred_flat[valid]]
        else:
            cm_truths, cm_preds = [], []

    y_true_valid = np.concatenate(cm_truths) if cm_truths else np.array([], dtype=int)
    y_pred_valid = np.concatenate(cm_preds) if cm_preds else np.array([], dtype=int)

    if len(y_true_valid) > 0:
        cm = confusion_matrix(y_true_valid, y_pred_valid)
        logger.info(f"Confusion matrix computed from {len(y_true_valid)} valid patches")
    else:
        cm = None

    return {
        "y_true": y_true_valid,
        "y_pred": y_pred_valid,
        "confusion_matrix": cm,
        "raster_path": output_path,
    }
