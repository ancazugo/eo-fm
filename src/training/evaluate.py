"""Test-set evaluation shared by both pipelines.

Both evaluators compute the same metric suite (OA, macro accuracy, macro/micro
F1, Cohen's kappa, per-class table, confusion matrix PNG); segmentation
additionally reports macro mIoU. WandB keys match the pre-refactor pipelines
(``test_acc``, ``test_f1``, … for classification; ``test_miou``, ``test_acc``,
``test_loss`` for segmentation, now extended with the full suite).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.classification import MulticlassCohenKappa, MulticlassF1Score

# Confusion-matrix pixel cap for segmentation (subsampled beyond this)
_MAX_CM_SAMPLES = 2_000_000


def style_lcz_ticklabels(ax, present_1idx) -> None:
    """Replace a confusion-matrix axes' tick labels with short LCZ codes
    (1-10, A-G) and highlight each with its class colour.

    ``present_1idx`` is the list of 1-indexed LCZ classes, in the same order
    used for the matrix axes. Only the tick labels are restyled — the matrix
    colour palette is left untouched.
    """
    from matplotlib.colors import to_rgb

    from utils.constants import lcz_dict

    short_labels = [lcz_dict.get(l, {}).get("alt_code", str(l)) for l in present_1idx]
    colors = [lcz_dict.get(l, {}).get("color", "#ffffff") for l in present_1idx]

    def _text_color(bg: str) -> str:
        r, g, b = to_rgb(bg)
        # perceived luminance → dark text on light backgrounds, light on dark
        return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"

    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=0)
    ax.set_yticklabels(short_labels, rotation=0)

    for axis_labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick, color in zip(axis_labels, colors):
            tick.set_color(_text_color(color))
            tick.set_fontweight("bold")
            tick.set_bbox(dict(facecolor=color, edgecolor="none",
                               boxstyle="round,pad=0.2"))


def save_confusion_matrix(
    y_true_1idx: np.ndarray,
    y_pred_1idx: np.ndarray,
    run_dir: Path,
    filename: str = "test_confusion_matrix.png",
    normalize: str | None = None,
) -> Path:
    """Plot + save a confusion matrix PNG for 1-indexed LCZ labels.

    ``normalize=None`` plots raw integer counts; ``normalize="true"`` plots the
    proportion of each true class (rows sum to 1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    present = sorted(set(y_true_1idx.tolist()) | set(y_pred_1idx.tolist()))
    cm = confusion_matrix(y_true_1idx, y_pred_1idx, labels=present,
                          normalize=normalize)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm).plot(
        ax=ax, colorbar=True, xticks_rotation=45,
        values_format=".2f" if normalize else "d",
    )
    style_lcz_ticklabels(ax, present)
    plt.tight_layout()
    cm_path = run_dir / filename
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {cm_path}")
    return cm_path


def _make_metrics(num_classes: int, device: torch.device, with_miou: bool) -> dict:
    metric_kw = dict(task="multiclass", num_classes=num_classes, ignore_index=-1)
    f1_kw = dict(num_classes=num_classes, ignore_index=-1)
    metrics = {
        "acc":           Accuracy(**metric_kw).to(device),
        "acc_macro":     Accuracy(**metric_kw, average="macro").to(device),
        "acc_per_class": Accuracy(**metric_kw, average="none").to(device),
        "f1":            MulticlassF1Score(**f1_kw, average="macro").to(device),
        "f1_micro":      MulticlassF1Score(**f1_kw, average="micro").to(device),
        "f1_per_class":  MulticlassF1Score(**f1_kw, average="none").to(device),
        "kappa":         MulticlassCohenKappa(num_classes=num_classes, ignore_index=-1).to(device),
    }
    if with_miou:
        metrics["miou"] = JaccardIndex(**metric_kw, average="macro").to(device)
    return metrics


def _evaluate(
    task,
    test_loader,
    device: torch.device,
    num_classes: int,
    run_dir: Path,
    run_label: str,
    use_wandb: bool,
    *,
    target_key: str,
    segmentation: bool,
    tta: bool = False,
) -> dict[str, float] | None:
    """Shared evaluation core. Returns the metrics dict or None if no batches."""
    import wandb

    task.eval()

    def _forward(imgs: torch.Tensor) -> torch.Tensor:
        if not tta:
            return task(imgs)
        # Average logits over the dihedral group (4 rotations × {id, hflip}),
        # the same transforms used for training augmentation.
        logits = None
        for k in range(4):
            r = torch.rot90(imgs, k, dims=(-2, -1)) if k else imgs
            out = task(r) + task(r.flip(-1))
            logits = out if logits is None else logits + out
        return logits / 8.0

    metrics = _make_metrics(num_classes, device, with_miou=segmentation)
    test_loss_total = 0.0
    n_test_batches = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch[target_key].to(device)
            if (labels != -1).sum() == 0:
                continue
            logits = _forward(imgs)
            if segmentation:
                loss, _, _ = task._loss(logits, labels)
            else:
                loss = task.ce_loss(logits, labels)
            preds = logits.argmax(dim=1)
            for m in metrics.values():
                m(preds, labels)
            test_loss_total += loss.item()
            n_test_batches += 1
            valid = labels != -1
            all_preds.append((preds[valid] + 1).cpu().numpy())
            all_labels.append((labels[valid] + 1).cpu().numpy())

    if n_test_batches == 0:
        logger.warning("No test batches with valid labels found.")
        return None

    results = {
        "test_acc":       metrics["acc"].compute().item(),
        "test_acc_macro": metrics["acc_macro"].compute().item(),
        "test_f1":        metrics["f1"].compute().item(),
        "test_f1_micro":  metrics["f1_micro"].compute().item(),
        "test_kappa":     metrics["kappa"].compute().item(),
        "test_loss":      test_loss_total / n_test_batches,
    }
    if segmentation:
        results["test_miou"] = metrics["miou"].compute().item()
    per_cls_acc = metrics["acc_per_class"].compute().cpu().numpy()
    per_cls_f1 = metrics["f1_per_class"].compute().cpu().numpy()

    miou_str = f"  mIoU: {results['test_miou']:.4f}" if segmentation else ""
    logger.info(
        f"Test — OA: {results['test_acc']:.4f}{miou_str}"
        f"  Acc_macro: {results['test_acc_macro']:.4f}"
        f"  F1_macro: {results['test_f1']:.4f}  F1_micro: {results['test_f1_micro']:.4f}"
        f"  Kappa: {results['test_kappa']:.4f}  Loss: {results['test_loss']:.4f}"
    )

    if use_wandb and wandb.run:
        wandb.log(results)
        from utils.wandb import log_per_class_metrics
        log_per_class_metrics(per_cls_acc, per_cls_f1, num_classes, prefix="test")

    if all_preds:
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        if len(y_true) > _MAX_CM_SAMPLES:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y_true), _MAX_CM_SAMPLES, replace=False)
            y_true, y_pred = y_true[idx], y_pred[idx]
        save_confusion_matrix(
            y_true, y_pred, run_dir, filename="test_confusion_matrix.png"
        )
        if not segmentation:
            save_confusion_matrix(
                y_true, y_pred, run_dir,
                filename="test_confusion_matrix_proportions.png",
                normalize="true",
            )

    return results


def evaluate_classification(
    task,
    test_loader,
    device: torch.device,
    num_classes: int,
    run_dir: Path,
    run_label: str,
    use_wandb: bool,
    tta: bool = False,
) -> dict[str, float] | None:
    """Patch-classification test evaluation (batch key: "label").

    With ``tta=True``, logits are averaged over flips/90° rotations.
    """
    return _evaluate(
        task, test_loader, device, num_classes, run_dir, run_label, use_wandb,
        target_key="label", segmentation=False, tta=tta,
    )


def evaluate_segmentation(
    task,
    test_loader,
    device: torch.device,
    num_classes: int,
    run_dir: Path,
    run_label: str,
    use_wandb: bool,
) -> dict[str, float] | None:
    """Segmentation test evaluation (batch key: "mask", per-pixel metrics)."""
    return _evaluate(
        task, test_loader, device, num_classes, run_dir, run_label, use_wandb,
        target_key="mask", segmentation=True,
    )
