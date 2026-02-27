"""Lightning task builders for classification and segmentation."""

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from loguru import logger
from rasterio.transform import from_bounds
from sklearn.metrics import confusion_matrix
from torchgeo.trainers import ClassificationTask, SemanticSegmentationTask

from conf import LightningConfig
from datasets.registry import get_in_channels


def build_task(
    config: LightningConfig,
    embedding_name: str,
) -> ClassificationTask | SemanticSegmentationTask:
    """Build a torchgeo Lightning task from config.

    Args:
        config: Lightning training configuration.
        embedding_name: Name of the embedding (used to look up in_channels).

    Returns:
        A ClassificationTask or SemanticSegmentationTask instance.
    """
    in_channels = get_in_channels(embedding_name)

    if config.task == "classification":
        return ClassificationTask(
            model=config.model,
            in_channels=in_channels,
            num_classes=config.num_classes,
            lr=config.lr,
            ignore_index=-1,  # nodata patches (label=0 in raster) are shifted to -1
        )
    elif config.task == "segmentation":
        backbone = config.backbone or "resnet50"
        return SemanticSegmentationTask(
            model=config.model,
            backbone=backbone,
            in_channels=in_channels,
            num_classes=config.num_classes,
            lr=config.lr,
            ignore_index=-1,  # nodata pixels (label=0 in raster) are shifted to -1
        )
    else:
        raise ValueError(f"Unknown task: {config.task}. Choose 'classification' or 'segmentation'.")


def predict_lightning_roi(
    trainer,
    task_module: ClassificationTask | SemanticSegmentationTask,
    datamodule,
    task_type: str,
    output_path: str | Path = "prediction.tif",
) -> dict[str, Any]:
    """Predict over the full ROI using a Lightning model and write a GeoTIFF.

    Uses the test dataloader (GridGeoSampler) for full-coverage prediction.

    torchgeo 0.9 API:
      SemanticSegmentationTask.predict_step → {"probabilities": (N,C,H,W),
                                               "bounds": [BoundingBox, ...],
                                               "transform": ...}
      ClassificationTask.predict_step       → tensor (N, num_classes)

    Ground truth is collected in a separate pass through the test dataloader
    because predict_step does not include labels in its output.

    Args:
        trainer: Lightning Trainer instance.
        task_module: Trained ClassificationTask or SemanticSegmentationTask.
        datamodule: EmbeddingLabelDataModule with test_roi set.
        task_type: "classification" or "segmentation".
        output_path: Path to write the prediction GeoTIFF.

    Returns:
        Dict with keys: y_true, y_pred, confusion_matrix, raster_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = trainer.predict(task_module, datamodule=datamodule)

    from types import SimpleNamespace

    def _extract_bbox(b: Any) -> SimpleNamespace:
        """Extract spatial bbox from a bounds entry.

        Bounds tensor format per sample: [minx, maxx, xstep, miny, maxy, ystep, ...]
        (IntersectionDataset concatenates bounds of both datasets; first 9 values
        are the embedding.)
        """
        if isinstance(b, torch.Tensor):
            return SimpleNamespace(
                minx=float(b[0]), maxx=float(b[1]),
                miny=float(b[3]), maxy=float(b[4]),
            )
        return SimpleNamespace(
            minx=float(b.minx), maxx=float(b.maxx),
            miny=float(b.miny), maxy=float(b.maxy),
        )

    dataset = datamodule.dataset
    crs = getattr(dataset, "crs", None)

    if task_type == "segmentation":
        # ── Pass 1: collect bboxes to determine full output extent ──────────
        all_bboxes: list[SimpleNamespace] = []
        first_probs_shape: tuple | None = None
        for batch_output in predictions:
            if not isinstance(batch_output, dict):
                continue
            if first_probs_shape is None:
                first_probs_shape = batch_output["probabilities"].shape  # (N, C, H, W)
            bounds_list = batch_output.get("bounds")
            if bounds_list is None:
                continue
            n = batch_output["probabilities"].shape[0]
            for i in range(n):
                all_bboxes.append(_extract_bbox(bounds_list[i]))

        if not all_bboxes or first_probs_shape is None:
            raise RuntimeError("No predictions with bounds produced by the segmentation model.")

        all_minx = min(b.minx for b in all_bboxes)
        all_miny = min(b.miny for b in all_bboxes)
        all_maxx = max(b.maxx for b in all_bboxes)
        all_maxy = max(b.maxy for b in all_bboxes)

        _, num_classes, ph, pw = first_probs_shape
        first_bbox = all_bboxes[0]
        res_x = (first_bbox.maxx - first_bbox.minx) / pw
        res_y = (first_bbox.maxy - first_bbox.miny) / ph

        out_w = int(round((all_maxx - all_minx) / res_x))
        out_h = int(round((all_maxy - all_miny) / res_y))

        # ── Pass 2: accumulate probability sums over overlapping patches ────
        # Overlapping patches each contribute to prob_sum; dividing by count_map
        # gives the average probability, eliminating patch boundary artifacts.
        prob_sum = np.zeros((out_h, out_w, num_classes), dtype=np.float32)
        count_map = np.zeros((out_h, out_w), dtype=np.float32)

        bbox_idx = 0
        for batch_output in predictions:
            if not isinstance(batch_output, dict):
                continue
            probs_np = batch_output["probabilities"].cpu().numpy()  # (N, C, H, W)
            n = probs_np.shape[0]
            bounds_list = batch_output.get("bounds")
            for i in range(n):
                bbox = all_bboxes[bbox_idx]
                bbox_idx += 1
                col = int(round((bbox.minx - all_minx) / res_x))
                row = int(round((all_maxy - bbox.maxy) / res_y))
                row_end = min(row + ph, out_h)
                col_end = min(col + pw, out_w)
                dh, dw = row_end - row, col_end - col
                if dh > 0 and dw > 0:
                    # (C, H, W) → (H, W, C) then add to accumulator
                    patch = probs_np[i].transpose(1, 2, 0)[:dh, :dw]
                    prob_sum[row:row_end, col:col_end] += patch
                    count_map[row:row_end, col:col_end] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_probs = prob_sum / count_map[:, :, np.newaxis]
        np.nan_to_num(avg_probs, nan=0.0, copy=False)
        output_raster = (avg_probs.argmax(axis=2) + 1).astype(np.uint8)
        output_raster[count_map == 0] = 0  # nodata where no patch covered

        y_pred = output_raster.ravel()

        transform = from_bounds(all_minx, all_miny, all_maxx, all_maxy, out_w, out_h)
        with rasterio.open(
            str(output_path), "w", driver="GTiff",
            height=out_h, width=out_w, count=1, dtype="uint8",
            crs=crs, transform=transform,
        ) as dst:
            dst.write(output_raster, 1)
        logger.info(f"Prediction raster saved to {output_path} ({out_h}×{out_w})")

    else:
        # Classification: argmax per patch → flat array
        all_preds = []
        for batch_output in predictions:
            pred_batch = batch_output.argmax(dim=1).cpu().numpy() + 1  # (N,), 1-based
            all_preds.append(pred_batch)
        y_pred = np.concatenate(all_preds)

        bounds = getattr(dataset, "bounds", None)
        out_w = len(y_pred)
        output_raster = y_pred.astype(np.uint8).reshape(1, out_w)
        transform = None
        if bounds is not None:
            x_slice, y_slice, _ = bounds
            transform = from_bounds(x_slice.start, y_slice.start, x_slice.stop, y_slice.stop, out_w, 1)
        with rasterio.open(
            str(output_path), "w", driver="GTiff",
            height=1, width=out_w, count=1, dtype="uint8",
            crs=crs, transform=transform,
        ) as dst:
            dst.write(output_raster, 1)
        logger.info(f"Prediction raster saved to {output_path}")

    # Ground truth — separate pass through test dataloader (not in predict_step output)
    all_truths = []
    for batch in datamodule.test_dataloader():
        if task_type == "classification" and "label" in batch:
            all_truths.append(batch["label"].numpy() + 1)
        elif task_type == "segmentation" and "mask" in batch:
            all_truths.append(batch["mask"].numpy() + 1)

    y_true = np.concatenate(all_truths, axis=0) if all_truths else np.array([], dtype=int)

    # Confusion matrix from valid (non-nodata) pixels
    y_pred_flat = y_pred.ravel()
    y_true_flat = y_true.ravel() if len(y_true) > 0 else np.array([], dtype=int)

    if len(y_true_flat) > 0:
        valid = y_true_flat > 0
        y_true_valid = y_true_flat[valid]
        y_pred_valid = y_pred_flat[valid]
        cm = confusion_matrix(y_true_valid, y_pred_valid)
        logger.info(f"Confusion matrix computed from {len(y_true_valid)} valid pixels")
    else:
        y_true_valid = np.array([], dtype=int)
        y_pred_valid = y_pred_flat
        cm = None

    return {
        "y_true": y_true_valid if len(y_true_flat) > 0 else y_true_flat,
        "y_pred": y_pred_valid if len(y_true_flat) > 0 else y_pred_flat,
        "confusion_matrix": cm,
        "raster_path": output_path,
    }
