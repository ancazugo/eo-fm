"""Test-set evaluation shared by both pipelines.

Both evaluators compute the same metric suite (OA, macro accuracy, macro/micro
F1, Cohen's kappa, per-class table, confusion matrix PNG); segmentation
additionally reports macro mIoU. WandB keys match the pre-refactor pipelines
(``test_acc``, ``test_f1``, … for classification; ``test_miou``, ``test_acc``,
``test_loss`` for segmentation, now extended with the full suite).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.classification import MulticlassCohenKappa, MulticlassF1Score

from training.lcz_metrics import lcz_metrics_from_cm, load_similarity_matrix

# Confusion-matrix pixel cap for the PLOT only (subsampled beyond this). The
# dense matrix the LCZ metrics consume is always built on the full stream:
# OAw and kappa_w are chance-corrected, so a subsample would move them.
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


def dense_confusion_matrix(
    y_true_0idx: np.ndarray, y_pred_0idx: np.ndarray, num_classes: int
) -> np.ndarray:
    """Confusion matrix over ALL ``num_classes``, whether or not each appears.

    ``sklearn``'s default (and :func:`save_confusion_matrix`, which plots only
    the classes present) returns a matrix whose axes are the observed labels.
    That is right for a plot and wrong for anything indexed by class: the LCZ
    similarity matrix is a fixed 17x17, so a matrix missing an absent class
    would silently mis-align against it. ``cm[i, j]`` = true ``i``, predicted
    ``j``.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(cm, (y_true_0idx.astype(np.int64), y_pred_0idx.astype(np.int64)), 1)
    return cm


def save_metrics_json(results: dict, run_dir: Path,
                      filename: str = "test_metrics.json") -> Path:
    """Persist a metrics dict next to the checkpoint.

    Until now the evaluators returned their metrics and both pipelines threw
    the return value away, so a finished run left no machine-readable record of
    its own test numbers -- only WandB, and only for runs that used it.
    """
    path = run_dir / filename
    path.write_text(json.dumps(
        {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
         for k, v in results.items()},
        indent=2, sort_keys=True,
    ) + "\n")
    logger.info(f"Metrics saved to {path}")
    return path


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


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    """Run classification inference and return (N, num_classes) softmax probs.

    With ``tta``, logits are averaged over the dihedral group (4 rotations ×
    {id, hflip}) before the softmax — the same scheme _evaluate uses.
    Batches must carry an "image" key; order follows the loader.
    """
    model.eval()
    out = []
    for batch in loader:
        imgs = batch["image"].to(device).float()
        if tta:
            logits = None
            for k in range(4):
                r = torch.rot90(imgs, k, dims=(-2, -1)) if k else imgs
                o = model(r) + model(r.flip(-1))
                logits = o if logits is None else logits + o
            logits = logits / 8.0
        else:
            logits = model(imgs)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


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


def _mode_pool(x: torch.Tensor, factor: int, num_classes: int) -> torch.Tensor:
    """Majority-pool a (B, H, W) long label tensor by ``factor``, ignoring -1.

    Trailing rows/cols that don't fill a block are dropped; blocks with no
    valid pixel become -1.
    """
    B, H, W = x.shape
    Hc, Wc = (H // factor) * factor, (W // factor) * factor
    if Hc == 0 or Wc == 0:
        return torch.full((B, 0, 0), -1, dtype=x.dtype, device=x.device)
    blocks = (
        x[:, :Hc, :Wc]
        .reshape(B, Hc // factor, factor, Wc // factor, factor)
        .permute(0, 1, 3, 2, 4)
        .reshape(B, Hc // factor, Wc // factor, factor * factor)
    )
    # Shift -1 → channel 0 so one_hot is valid, then count per class
    counts = torch.nn.functional.one_hot(blocks + 1, num_classes + 1).sum(dim=-2)
    pooled = counts[..., 1:].argmax(dim=-1)
    pooled[counts[..., 1:].sum(dim=-1) == 0] = -1
    return pooled


def _lcz_suite(cm: np.ndarray, num_classes: int, suffix: str = "") -> dict:
    """OAu / OAbu / OAw / kappa_w, or nothing if this is not a 17-class LCZ run.

    Guarded rather than assumed: the pipelines accept ``--num-classes``, and a
    similarity matrix defined for the 17 LCZ types means nothing against a
    different label set.
    """
    if num_classes != 17:
        return {}
    try:
        out = lcz_metrics_from_cm(cm, load_similarity_matrix(), suffix=suffix)
    except (FileNotFoundError, ValueError) as exc:
        # A missing or malformed matrix must not take down a finished training
        # run -- the generic suite is already computed and is the primary one.
        logger.warning(f"LCZ weighted metrics skipped: {exc}")
        return {}
    logger.info(
        f"Test LCZ suite{suffix} — OAu: {out[f'test_oau{suffix}']:.4f}"
        f"  OAbu: {out[f'test_oabu{suffix}']:.4f}"
        f"  OAw: {out[f'test_oaw{suffix}']:.4f}"
        f"  Kappa_w: {out[f'test_kappa_w{suffix}']:.4f}"
    )
    return out


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
    coarse_factor: int | None = None,
    coarse_label: str = "100m",
    metric_suffix: str = "",
) -> dict[str, float] | None:
    """Shared evaluation core. Returns the metrics dict or None if no batches.

    coarse_factor (segmentation only): additionally majority-pool predictions
    and labels by this factor and report the metric suite at that scale under
    ``test_*_{coarse_label}`` keys — LCZ is a ~100 m concept, so 10× pooling of
    10 m pixels gives the definition-scale numbers.
    """
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
    coarse_metrics = (
        _make_metrics(num_classes, device, with_miou=True)
        if segmentation and coarse_factor else None
    )
    n_coarse_batches = 0
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
            if coarse_metrics is not None:
                preds_c = _mode_pool(preds, coarse_factor, num_classes)
                labels_c = _mode_pool(labels, coarse_factor, num_classes)
                if (labels_c != -1).any():
                    for m in coarse_metrics.values():
                        m(preds_c, labels_c)
                    n_coarse_batches += 1
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
    if coarse_metrics is not None and n_coarse_batches > 0:
        results.update({
            f"test_acc_{coarse_label}":       coarse_metrics["acc"].compute().item(),
            f"test_acc_macro_{coarse_label}": coarse_metrics["acc_macro"].compute().item(),
            f"test_f1_{coarse_label}":        coarse_metrics["f1"].compute().item(),
            f"test_kappa_{coarse_label}":     coarse_metrics["kappa"].compute().item(),
            f"test_miou_{coarse_label}":      coarse_metrics["miou"].compute().item(),
        })
        logger.info(
            f"Test @{coarse_label} (mode-pooled ×{coarse_factor}) — "
            f"OA: {results[f'test_acc_{coarse_label}']:.4f}"
            f"  mIoU: {results[f'test_miou_{coarse_label}']:.4f}"
            f"  F1_macro: {results[f'test_f1_{coarse_label}']:.4f}"
            f"  Kappa: {results[f'test_kappa_{coarse_label}']:.4f}"
        )
    per_cls_acc = metrics["acc_per_class"].compute().cpu().numpy()
    per_cls_f1 = metrics["f1_per_class"].compute().cpu().numpy()

    miou_str = f"  mIoU: {results['test_miou']:.4f}" if segmentation else ""
    logger.info(
        f"Test — OA: {results['test_acc']:.4f}{miou_str}"
        f"  Acc_macro: {results['test_acc_macro']:.4f}"
        f"  F1_macro: {results['test_f1']:.4f}  F1_micro: {results['test_f1_micro']:.4f}"
        f"  Kappa: {results['test_kappa']:.4f}  Loss: {results['test_loss']:.4f}"
    )

    logged_keys = set(results)
    if use_wandb and wandb.run:
        wandb.log(results)
        from utils.wandb import log_per_class_metrics
        log_per_class_metrics(per_cls_acc, per_cls_f1, num_classes, prefix="test")

    if all_preds:
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        # Dense matrix on the FULL stream, before any subsampling, and always
        # saved: it is what the WUDAPT metrics are computed from and what any
        # later per-city or class-pair analysis needs.
        cm = dense_confusion_matrix(y_true - 1, y_pred - 1, num_classes)
        np.save(run_dir / f"test_confusion_matrix{metric_suffix}.npy", cm)
        results.update(_lcz_suite(cm, num_classes, metric_suffix))

        if len(y_true) > _MAX_CM_SAMPLES:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y_true), _MAX_CM_SAMPLES, replace=False)
            y_true, y_pred = y_true[idx], y_pred[idx]
        save_confusion_matrix(
            y_true, y_pred, run_dir,
            filename=f"test_confusion_matrix{metric_suffix}.png",
        )
        if not segmentation:
            save_confusion_matrix(
                y_true, y_pred, run_dir,
                filename=f"test_confusion_matrix_proportions{metric_suffix}.png",
                normalize="true",
            )

    if use_wandb and wandb.run:
        wandb.log({k: v for k, v in results.items() if k not in logged_keys})

    return results


@torch.no_grad()
def evaluate_segmentation_as_patches(
    task,
    test_loader,
    device: torch.device,
    num_classes: int,
    run_dir: Path,
    run_label: str,
    use_wandb: bool,
    *,
    uid_to_label: dict[int, int],
    uid_to_key: dict[int, tuple] | None = None,
    tta: bool = False,
    metric_suffix: str = "_patch",
    save_probs: bool = True,
    dataset_filter: str = "testing",
) -> dict[str, float] | None:
    """Score a dense segmentation model at So2Sat **patch** level.

    Mean softmax over each patch footprint -> argmax -> compared against the
    patch label. This is the number that goes on the same axis as the patch
    classification ladder (0.6497 single / 0.6871 LOCO-weighted ensemble /
    0.7055 with the aux corrector); per-pixel mIoU cannot be compared to
    anything in that table.

    Probabilities are averaged in probability space, not logit space, and
    weighted by pixel count implicitly (a running sum divided by the count),
    so a patch clipped by a tile edge is not over-weighted relative to a whole
    one.

    ``uid_to_label`` maps patch UID -> 0-indexed LCZ class. ``uid_to_key`` maps
    UID -> ``(city, dataset, patch_id)`` and, when given, lets the cached
    probabilities be written in the layout ``ensemble_eval.py`` produces and
    ``ensemble_stacking.py --city-holdout`` consumes.

    **Coverage caveat.** Under ``--split-mode global`` this scores a strict
    subset of the patch campaign's test set: tiles that fail split purity or
    the proximity buffer are dropped whole, and their patches go with them. The
    fraction is logged and stored as ``test_coverage*``. For an exact
    comparison against the 0.6497 / 0.6871 / 0.7055 ladder, intersect on
    ``patch_id`` and re-score both sides on the intersection.
    """
    import wandb

    task.eval()
    prob_sum: dict[int, np.ndarray] = {}
    pix_count: dict[int, int] = {}

    def _forward(imgs):
        if not tta:
            return task(imgs)
        logits = None
        for k in range(4):
            r = torch.rot90(imgs, k, dims=(-2, -1)) if k else imgs
            out = task(r) + task(r.flip(-1))
            logits = out if logits is None else logits + out
        return logits / 8.0

    seen_uid_key = False
    for batch in test_loader:
        if "patch_uid" not in batch:
            continue
        seen_uid_key = True
        imgs = batch["image"].to(device).float()
        uids = batch["patch_uid"].to(device)
        probs = torch.softmax(_forward(imgs), dim=1)      # (B, C, H, W)
        valid = uids >= 0
        if not valid.any():
            continue
        # Flatten to (N_valid, C) and scatter-add per UID.
        flat_probs = probs.permute(0, 2, 3, 1)[valid]      # (N, C)
        flat_uids = uids[valid]                            # (N,)
        uniq, inverse = torch.unique(flat_uids, return_inverse=True)
        summed = torch.zeros(
            len(uniq), num_classes, device=probs.device, dtype=flat_probs.dtype
        ).index_add_(0, inverse, flat_probs)
        counts = torch.zeros(len(uniq), device=probs.device, dtype=torch.long)
        counts.index_add_(0, inverse, torch.ones_like(flat_uids))
        for u, sm, ct in zip(uniq.tolist(), summed.cpu().numpy(),
                             counts.cpu().tolist()):
            if u in prob_sum:
                prob_sum[u] += sm
                pix_count[u] += ct
            else:
                prob_sum[u] = sm.copy()
                pix_count[u] = ct

    if not seen_uid_key:
        logger.warning(
            "Patch-level evaluation skipped: the loader emitted no `patch_uid`. "
            "Build the DataModule with emit_patch_uids=True and gpkg labels."
        )
        return None
    if not prob_sum:
        logger.warning("Patch-level evaluation found no labelled patch pixels.")
        return None

    uids = sorted(u for u in prob_sum if u in uid_to_label)
    if not uids:
        logger.warning("No evaluated patch UID carries a label.")
        return None
    probs = np.stack([prob_sum[u] / max(pix_count[u], 1) for u in uids])
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.array([uid_to_label[u] for u in uids], dtype=np.int64)
    preds = probs.argmax(axis=1)

    cm = dense_confusion_matrix(labels, preds, num_classes)
    correct = labels == preds
    results = {
        f"test_acc{metric_suffix}": float(correct.mean()),
        f"test_n{metric_suffix}": int(len(uids)),
    }
    # Reuse the torchmetrics implementations so patch-level and pixel-level
    # numbers are produced by exactly the same estimators.
    t_pred = torch.from_numpy(preds)
    t_true = torch.from_numpy(labels)
    m = _make_metrics(num_classes, torch.device("cpu"), with_miou=False)
    for fn in m.values():
        fn(t_pred, t_true)
    results.update({
        f"test_acc_macro{metric_suffix}": m["acc_macro"].compute().item(),
        f"test_f1{metric_suffix}":        m["f1"].compute().item(),
        f"test_f1_micro{metric_suffix}":  m["f1_micro"].compute().item(),
        f"test_kappa{metric_suffix}":     m["kappa"].compute().item(),
    })
    results.update(_lcz_suite(cm, num_classes, metric_suffix))

    # Coverage is a headline caveat, not a footnote. Making the split honest
    # costs patches: tiles failing purity or the proximity buffer are dropped
    # whole, taking their patches with them. So this kappa is computed over a
    # SUBSET of the patch campaign's test set, and the two are only exactly
    # comparable after intersecting on patch_id -- which is why probs*.npz
    # carries patch_ids and datasets.
    coverage_note = ""
    if uid_to_key is not None:
        n_available = sum(1 for k in uid_to_key.values() if k[1] == dataset_filter)
        if n_available:
            frac = len(uids) / n_available
            results[f"test_coverage{metric_suffix}"] = float(frac)
            coverage_note = (
                f" [{len(uids)}/{n_available} = {frac:.1%} of {dataset_filter} patches]"
            )
            if frac < 0.95:
                logger.warning(
                    f"Patch-level metrics cover {frac:.1%} of the available "
                    f"{dataset_filter} patches ({len(uids)}/{n_available}). "
                    "The rest sit in tiles "
                    "dropped for split purity or the proximity buffer. Do NOT "
                    "compare this kappa to a patch-model number scored on the "
                    "full test set without intersecting on patch_id first."
                )

    level = "Test" if metric_suffix in ("_patch", "") else metric_suffix.lstrip("_")
    logger.info(
        f"{level} @patch ({len(uids)} patches){coverage_note} — "
        f"OA: {results[f'test_acc{metric_suffix}']:.4f}"
        f"  F1_macro: {results[f'test_f1{metric_suffix}']:.4f}"
        f"  Kappa: {results[f'test_kappa{metric_suffix}']:.4f}"
    )

    np.save(run_dir / f"test_confusion_matrix{metric_suffix}.npy", cm)
    save_confusion_matrix(
        labels + 1, preds + 1, run_dir,
        filename=f"test_confusion_matrix{metric_suffix}.png",
    )

    if uid_to_key is not None:
        cities = np.array([
            (uid_to_key.get(u) or ("", "", ""))[0] for u in uids
        ])
        by_city = per_city_metrics(labels, preds, cities, num_classes)
        if by_city:
            (run_dir / f"per_city_metrics{metric_suffix}.json").write_text(
                json.dumps(by_city, indent=2, sort_keys=True) + "\n"
            )
            spread = by_city.get("_spread", {})
            named = {k: v for k, v in by_city.items() if k != "_spread"}
            worst = min(named, key=lambda c: named[c]["kappa"]) if named else None
            best = max(named, key=lambda c: named[c]["kappa"]) if named else None
            logger.info(
                f"Per-city kappa over {len(named)} cities — "
                f"mean {spread.get('kappa_mean', float('nan')):.4f} "
                f"± {spread.get('kappa_std', float('nan')):.4f}"
                + (f", worst {worst} {named[worst]['kappa']:.4f}, "
                   f"best {best} {named[best]['kappa']:.4f}" if worst else "")
            )
            results[f"test_kappa_city_mean{metric_suffix}"] = spread.get("kappa_mean", float("nan"))
            results[f"test_kappa_city_std{metric_suffix}"] = spread.get("kappa_std", float("nan"))

    if save_probs and uid_to_key is not None:
        keys = [uid_to_key.get(u) for u in uids]
        np.savez_compressed(
            run_dir / f"probs{metric_suffix}.npz",
            labels=labels,
            patch_ids=np.array([k[2] if k else "" for k in keys]),
            cities=np.array([k[0] if k else "" for k in keys]),
            datasets=np.array([k[1] if k else "" for k in keys]),
            **{run_label: probs.astype(np.float32)},
        )
        logger.info(f"Patch probs cached to {run_dir}/probs{metric_suffix}.npz")

    if use_wandb and wandb.run:
        wandb.log(results)

    return results


def per_city_metrics(
    labels: np.ndarray, preds: np.ndarray, cities: np.ndarray, num_classes: int
) -> dict[str, dict[str, float]]:
    """Kappa / OA / macro-F1 per city, plus the spread across cities.

    A pooled number hides exactly the signal that matters: on the patch task
    the LOCO per-city kappa ran from Munich at 0.90 down to Nairobi and
    Santiago at 0.47.
    """
    from sklearn.metrics import cohen_kappa_score, f1_score

    out: dict[str, dict[str, float]] = {}
    for city in sorted({c for c in cities.tolist() if c}):
        m = cities == city
        if m.sum() == 0:
            continue
        out[city] = {
            "n": int(m.sum()),
            "oa": float((labels[m] == preds[m]).mean()),
            "kappa": float(cohen_kappa_score(
                labels[m], preds[m], labels=list(range(num_classes)))),
            "f1_macro": float(f1_score(
                labels[m], preds[m], labels=list(range(num_classes)),
                average="macro", zero_division=0)),
        }
    if out:
        ks = [v["kappa"] for v in out.values()]
        out["_spread"] = {
            "kappa_mean": float(np.mean(ks)),
            "kappa_std": float(np.std(ks)),
            "kappa_min": float(np.min(ks)),
            "kappa_max": float(np.max(ks)),
        }
    return out


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
    coarse_factor: int | None = 10,
    coarse_label: str = "100m",
    tta: bool = False,
    metric_suffix: str = "",
) -> dict[str, float] | None:
    """Segmentation test evaluation (batch key: "mask", per-pixel metrics).

    Also reports the metric suite majority-pooled by ``coarse_factor``
    (default 10 → 100 m at 10 m/px, the scale LCZ is defined at) as
    ``test_*_{coarse_label}``. Pass ``coarse_factor=None`` to disable.

    This is a **masked** evaluation: every metric is computed over labelled
    pixels only. mIoU in particular is therefore not IoU against the true map,
    because the union is restricted to the labelled subset — it is closer to a
    macro-F1, and it is not comparable to papers scoring against dense
    reference maps. Use ``metric_suffix="_pixel"`` when reporting alongside the
    patch-level aggregation so the two levels stay distinguishable.
    """
    return _evaluate(
        task, test_loader, device, num_classes, run_dir, run_label, use_wandb,
        target_key="mask", segmentation=True, tta=tta,
        coarse_factor=coarse_factor, coarse_label=coarse_label,
        metric_suffix=metric_suffix,
    )
