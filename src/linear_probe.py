"""Linear probe + kNN baseline on So2Sat patch embeddings.

Features are extracted from pre-computed embedding .npy patches via global
average pooling (or mean+std concatenation), z-score normalised using
training-set statistics, then used to train a single linear layer and a
cosine-kNN classifier.

Data modes (--global-split vs per-city) and dequantization logic mirror
patch_classification.py.  Grid-tiles are not supported here.

Example (per-city, AlphaEarth):
    python src/linear_probe.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi \\
        --output-name AlphaEarth --year 2017 \\
        --embedding-name alpha_earth \\
        --pooling gap --class-weights sqrt_inv_freq \\
        --batch-size 1024 --max-epochs 100 \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl

Example (global split, seamless — dequantize applied automatically):
    python src/linear_probe.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --global-split \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --output-name Seamless --year 2017 \\
        --embedding-name seamless \\
        --pooling mean_std --class-weights effective_number \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassCohenKappa, MulticlassF1Score
from tqdm import tqdm

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from patch_classification import (
    _build_city_items,
    _build_global_items,
    _build_patch_index,
)
from utils.constants import lcz_dict


# ── Feature extraction ────────────────────────────────────────────────────────

def _pool(arr: np.ndarray, pooling: str) -> np.ndarray:
    """Apply spatial pooling to (C, H, W) float32 → 1-D feature vector."""
    mean = arr.mean(axis=(1, 2))
    if pooling == "gap":
        return mean
    return np.concatenate([mean, arr.std(axis=(1, 2))])


def extract_and_cache(
    items: list[tuple],
    pooling: str,
    dequantize_fn,
    cache_dir: Path,
    cache_key: str,
    no_cache: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Extract and cache pooled features for all splits.

    Returns:
        feats:  {"train": (N,D), "val": (N,D), "test": (N,D)}
        labels: {"train": (N,), "val": (N,), "test": (N,)}
        mean:   (D,) — training-set feature mean
        std:    (D,) — training-set feature std
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cache_dir / f"{cache_key}_stats.npz"
    splits = ("train", "val", "test")

    def _cache_paths(s: str):
        return (
            cache_dir / f"{cache_key}_{s}_feats.npy",
            cache_dir / f"{cache_key}_{s}_labels.npy",
        )

    all_cached = (
        not no_cache
        and stats_path.exists()
        and all(fp.exists() and lp.exists() for fp, lp in (_cache_paths(s) for s in splits))
    )

    if all_cached:
        logger.info(f"Loading feature cache ({cache_key}) from {cache_dir}")
        feats = {s: np.load(_cache_paths(s)[0]) for s in splits}
        labels = {s: np.load(_cache_paths(s)[1]) for s in splits}
        stats = np.load(stats_path)
        return feats, labels, stats["mean"], stats["std"]

    logger.info(f"Extracting features (pooling={pooling}) …")
    split_items: dict[str, list] = {s: [] for s in splits}
    for path, label, sp in items:
        if sp in split_items:
            split_items[sp].append((path, label))

    feats: dict[str, np.ndarray] = {}
    labels_out: dict[str, np.ndarray] = {}

    for s in splits:
        rows = split_items[s]
        fp, lp = _cache_paths(s)
        if not rows:
            feats[s] = np.empty((0, 1), dtype=np.float32)
            labels_out[s] = np.empty((0,), dtype=np.int64)
            np.save(fp, feats[s])
            np.save(lp, labels_out[s])
            continue

        feat_list, lbl_list = [], []
        for path, label in tqdm(rows, desc=f"  {s}", leave=False):
            arr = np.load(path).astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            if dequantize_fn is not None:
                arr = dequantize_fn(arr)
            feat_list.append(_pool(arr, pooling))
            lbl_list.append(label)

        feats[s] = np.stack(feat_list, axis=0).astype(np.float32)
        labels_out[s] = np.array(lbl_list, dtype=np.int64)
        np.save(fp, feats[s])
        np.save(lp, labels_out[s])
        logger.info(f"  {s}: {feats[s].shape[0]} patches, dim={feats[s].shape[1]}")

    mean = feats["train"].mean(axis=0)
    std = feats["train"].std(axis=0)
    np.savez(stats_path, mean=mean, std=std)
    logger.info(f"Stats saved → {stats_path.name}")

    return feats, labels_out, mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / (std + 1e-8)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    labels_train: np.ndarray,
    scheme: str,
    num_classes: int,
    beta: float = 0.999,
) -> torch.Tensor | None:
    if scheme == "none":
        return None
    counts = np.bincount(labels_train[labels_train >= 0], minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if scheme == "inv_freq":
        w = 1.0 / counts
    elif scheme == "sqrt_inv_freq":
        w = 1.0 / np.sqrt(counts)
    else:  # effective_number
        effective_n = (1.0 - beta ** counts) / (1.0 - beta)
        w = 1.0 / effective_n
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


# ── Linear probe ──────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    patience: int,
    class_weights: torch.Tensor | None,
    run_dir: Path,
) -> tuple[LinearProbe, Path | None]:
    feat_dim = X_train.shape[1]
    model = LinearProbe(feat_dim, num_classes).to(device)

    cw = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=cw, ignore_index=-1)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
        batch_size=batch_size, shuffle=False,
    )

    metric_kw = dict(task="multiclass", num_classes=num_classes, ignore_index=-1)
    f1_kw = dict(num_classes=num_classes, ignore_index=-1)

    train_acc_m       = Accuracy(**metric_kw).to(device)
    train_acc_macro_m = Accuracy(**metric_kw, average="macro").to(device)
    train_f1_macro_m  = MulticlassF1Score(**f1_kw, average="macro").to(device)
    train_f1_micro_m  = MulticlassF1Score(**f1_kw, average="micro").to(device)
    train_kappa_m     = MulticlassCohenKappa(num_classes=num_classes, ignore_index=-1).to(device)

    val_acc_m         = Accuracy(**metric_kw).to(device)
    val_acc_macro_m   = Accuracy(**metric_kw, average="macro").to(device)
    val_f1_macro_m    = MulticlassF1Score(**f1_kw, average="macro").to(device)
    val_f1_micro_m    = MulticlassF1Score(**f1_kw, average="micro").to(device)
    val_kappa_m       = MulticlassCohenKappa(num_classes=num_classes, ignore_index=-1).to(device)

    best_val_oa = -1.0
    patience_ctr = 0
    best_ckpt: Path | None = None

    for epoch in range(max_epochs):
        model.train()
        for m in (train_acc_m, train_acc_macro_m, train_f1_macro_m, train_f1_micro_m, train_kappa_m):
            m.reset()
        t_loss, n_t = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            if (yb != -1).sum() == 0:
                continue
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n_t += 1
            with torch.no_grad():
                preds = logits.argmax(1)
                train_acc_m(preds, yb)
                train_acc_macro_m(preds, yb)
                train_f1_macro_m(preds, yb)
                train_f1_micro_m(preds, yb)
                train_kappa_m(preds, yb)

        train_loss     = t_loss / max(1, n_t)
        train_oa       = train_acc_m.compute().item()
        train_acc_macro = train_acc_macro_m.compute().item()
        train_f1_macro  = train_f1_macro_m.compute().item()
        train_f1_micro  = train_f1_micro_m.compute().item()
        train_kappa     = train_kappa_m.compute().item()

        model.eval()
        for m in (val_acc_m, val_acc_macro_m, val_f1_macro_m, val_f1_micro_m, val_kappa_m):
            m.reset()
        v_loss, n_v = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                if (yb != -1).sum() == 0:
                    continue
                logits = model(Xb)
                v_loss += criterion(logits, yb).item()
                n_v += 1
                preds = logits.argmax(1)
                val_acc_m(preds, yb)
                val_acc_macro_m(preds, yb)
                val_f1_macro_m(preds, yb)
                val_f1_micro_m(preds, yb)
                val_kappa_m(preds, yb)

        scheduler.step()
        val_loss     = v_loss / max(1, n_v)
        val_oa       = val_acc_m.compute().item()
        val_acc_macro = val_acc_macro_m.compute().item()
        val_f1_macro  = val_f1_macro_m.compute().item()
        val_f1_micro  = val_f1_micro_m.compute().item()
        val_kappa     = val_kappa_m.compute().item()

        if wandb.run:
            wandb.log({
                "train_loss":      train_loss,
                "train_oa":        train_oa,
                "train_acc_macro": train_acc_macro,
                "train_f1_macro":  train_f1_macro,
                "train_f1_micro":  train_f1_micro,
                "train_kappa":     train_kappa,
                "val_loss":        val_loss,
                "val_oa":          val_oa,
                "val_acc_macro":   val_acc_macro,
                "val_f1":          val_f1_macro,
                "val_f1_micro":    val_f1_micro,
                "val_kappa":       val_kappa,
                "epoch":           epoch + 1,
            })
        logger.info(
            f"Epoch {epoch+1}/{max_epochs}  "
            f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_oa={val_oa:.4f}"
        )

        if val_oa > best_val_oa:
            best_val_oa = val_oa
            patience_ctr = 0
            best_ckpt = run_dir / "linear_probe_best.pt"
            torch.save({"model_state_dict": model.state_dict(), "val_oa": val_oa}, best_ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (val_oa={val_oa:.4f})")
                break

    if best_ckpt and best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best model (val_oa={ckpt['val_oa']:.4f}) from {best_ckpt.name}")

    return model, best_ckpt


# ── kNN ───────────────────────────────────────────────────────────────────────

def run_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """L2-normalise, cosine-kNN, majority vote over k neighbours."""
    def _l2(X: np.ndarray) -> np.ndarray:
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # n_jobs capped to avoid OpenBLAS thread-count overflow on large datasets
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=4)
    nbrs.fit(_l2(X_train))
    _, indices = nbrs.kneighbors(_l2(X_test))  # (N_test, k)

    neighbor_labels = y_train[indices]          # (N_test, k)
    preds = np.empty(len(X_test), dtype=np.int64)
    for i in range(len(X_test)):
        valid_nl = neighbor_labels[i][neighbor_labels[i] >= 0]
        if valid_nl.size == 0:
            preds[i] = -1
        else:
            counts = np.bincount(valid_nl.astype(np.intp))
            preds[i] = int(counts.argmax())
    return preds


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str,
    num_classes: int,
    run_dir: Path,
    use_wandb: bool,
    test_loss: float | None = None,
) -> dict[str, float]:
    valid = y_true != -1
    yt, yp = y_true[valid], y_pred[valid]

    oa = float(accuracy_score(yt, yp))
    f1_macro = float(f1_score(yt, yp, average="macro", zero_division=0))
    f1_micro = float(f1_score(yt, yp, average="micro", zero_division=0))
    try:
        kappa = float(cohen_kappa_score(yt, yp))
    except Exception:
        kappa = float("nan")

    prec, rec, f1_per, support = precision_recall_fscore_support(
        yt, yp, labels=list(range(num_classes)), average=None, zero_division=0,
    )
    acc_macro = float(np.mean(rec))  # mean per-class recall = macro accuracy

    logger.info(
        f"[{method}] OA={oa:.4f}  Acc_macro={acc_macro:.4f}"
        f"  F1_macro={f1_macro:.4f}  F1_micro={f1_micro:.4f}  Kappa={kappa:.4f}"
    )

    # Per-class metrics CSV
    cls_csv = run_dir / f"per_class_metrics_{method}.csv"
    with cls_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
        for i in range(num_classes):
            cid = i + 1
            cname = lcz_dict.get(cid, {}).get("name", str(cid))
            w.writerow([cid, cname, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1_per[i]:.4f}", int(support[i])])

    # Confusion matrix
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    display_labels = [lcz_dict.get(l + 1, {}).get("name", str(l + 1)) for l in present]
    cm = confusion_matrix(yt, yp, labels=present)

    cm_csv = run_dir / f"test_confusion_matrix_{method}.csv"
    with cm_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + display_labels)
        for row_label, row in zip(display_labels, cm):
            w.writerow([row_label] + list(row))

    fig, ax = plt.subplots(figsize=(13, 11))
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(
        ax=ax, colorbar=True, xticks_rotation=45,
    )
    ax.set_title(f"Test Confusion Matrix — {method}")
    plt.tight_layout()
    cm_png = run_dir / f"test_confusion_matrix_{method}.png"
    fig.savefig(cm_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Confusion matrix → {cm_png.name}")

    if use_wandb and wandb.run:
        per_cls_acc = np.array([
            float(accuracy_score(yt[yt == i], yp[yt == i])) if (yt == i).any() else 0.0
            for i in range(num_classes)
        ])
        log_dict = {
            "test_oa":        oa,
            "test_acc_macro": acc_macro,
            "test_f1_macro":  f1_macro,
            "test_f1_micro":  f1_micro,
            "test_kappa":     kappa,
            "confusion_matrix": wandb.Image(str(cm_png)),
        }
        if test_loss is not None:
            log_dict["test_loss"] = test_loss
        wandb.log(log_dict)
        from utils.wandb import log_per_class_metrics
        log_per_class_metrics(per_cls_acc, f1_per, num_classes, prefix="test")

    return {"oa": oa, "acc_macro": acc_macro, "f1_macro": f1_macro, "f1_micro": f1_micro, "kappa": kappa}


def _per_city_oa(
    test_items: list[tuple],
    y_pred: np.ndarray,
    cities_dir: Path | None,
    method: str,
    run_dir: Path,
) -> None:
    if cities_dir is None:
        return

    pid_to_idx: dict[str, int] = {}
    for idx, (path, _, _) in enumerate(test_items):
        stem = path.stem
        pid = stem[len("patch_"):] if stem.startswith("patch_") else stem
        pid_to_idx[pid] = idx

    rows = []
    for city_dir in sorted(cities_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        gpkg = city_dir / f"patches_reference_{city}_split.gpkg"
        if not gpkg.exists():
            continue
        try:
            sdf = gpd.read_file(gpkg)
        except Exception as e:
            logger.warning(f"  {city}: could not read split GPKG — {e}")
            continue
        test_pids = [
            str(row["patch_id"]) for _, row in sdf.iterrows()
            if str(row.get("split", "")) == "test"
        ]
        indices = [pid_to_idx[pid] for pid in test_pids if pid in pid_to_idx]
        if not indices:
            continue
        yt = np.array([test_items[i][1] for i in indices])
        yp = y_pred[np.array(indices)]
        valid = yt != -1
        if valid.sum() == 0:
            continue
        city_oa = float(accuracy_score(yt[valid], yp[valid]))
        rows.append((city, int(valid.sum()), city_oa))

    rows.sort(key=lambda r: r[1], reverse=True)
    city_csv = run_dir / f"per_city_oa_{method}.csv"
    with city_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "n_test_patches", "oa"])
        for city, n, oa in rows:
            w.writerow([city, n, f"{oa:.4f}"])
    logger.info(f"  Per-city OA ({len(rows)} cities) → {city_csv.name}")


def _append_summary(summary_csv: Path, row: dict) -> None:
    fieldnames = [
        "run_name", "date", "method", "embedding", "cities", "split_mode",
        "pooling", "class_weights", "feature_dim",
        "train_n", "val_n", "test_n",
        "test_oa", "test_f1_macro", "test_kappa",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
    logger.info(f"Summary appended → {summary_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linear probe + kNN baseline on So2Sat patch embeddings."
    )

    g = parser.add_argument_group("Data")
    g.add_argument("--so2sat-dir", required=True, type=Path,
                   help="Root So2Sat directory with training/validation/testing subfolders.")
    g.add_argument("--global-split", action="store_true",
                   help="Use patches_reference_rxr.gpkg global split instead of per-city GPKGs.")
    g.add_argument("--global-gpkg", type=Path, default=None,
                   help="Path to global GPKG (default: {so2sat_dir}/patches_reference_rxr.gpkg).")
    g.add_argument("--cities-dir", required=False, default=None, type=Path,
                   help="Directory of per-city subfolders (required unless --global-split).")
    g.add_argument("--cities", nargs="+", default=None,
                   help="City names to include.")
    g.add_argument("--output-name", required=True,
                   help="Embedding subfolder name (e.g. AlphaEarth, GeoTessera).")
    g.add_argument("--year", required=True, help="Year subfolder (e.g. 2017).")
    g.add_argument("--label-col", default="LCZ_class")
    g.add_argument("--embedding-name", required=True,
                   choices=["tessera", "tesserav1.1", "tesserav1.1_global", "alpha_earth", "alpha_earth_coop", "seamless"],
                   help="Embedding type — controls auto-dequantization.")
    g.add_argument("--dequantize", action="store_true",
                   help="Force dequantize (auto-applied for alpha_earth_coop and seamless).")

    g = parser.add_argument_group("Features")
    g.add_argument("--pooling", choices=["gap", "mean_std"], default="gap",
                   help="Spatial pooling: gap = global average; mean_std = mean+std concat (doubles dim).")
    g.add_argument("--cache-dir", type=Path, default=None,
                   help="Feature cache directory (default: {output_dir}/cache).")
    g.add_argument("--no-cache", action="store_true", help="Ignore and overwrite existing cache.")

    g = parser.add_argument_group("Linear probe")
    g.add_argument("--num-classes", type=int, default=17)
    g.add_argument("--batch-size", type=int, default=1024)
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--max-epochs", type=int, default=100)
    g.add_argument("--early-stopping-patience", type=int, default=10)
    g.add_argument("--class-weights", default="sqrt_inv_freq",
                   choices=["none", "inv_freq", "sqrt_inv_freq", "effective_number"])
    g.add_argument("--beta", type=float, default=0.999,
                   help="Beta for effective_number class weighting.")

    g = parser.add_argument_group("Method")
    g.add_argument("--method", choices=["linear_probe", "knn"], default="linear_probe",
                   help="Which method to run as its own WandB run.")
    g.add_argument("--knn-k", type=int, default=20)

    g = parser.add_argument_group("Output")
    g.add_argument("--output-dir", required=True, type=Path)
    g.add_argument("--run-name", default=None)
    g.add_argument("--wandb-project", default="lcz-classification-dl")
    g.add_argument("--wandb-entity", default="phd-thesis-team")
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.accelerator == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.accelerator)
    logger.info(f"Device: {device}")

    # ── Dequantize (auto-required for coop and seamless) ──────────────────────
    need_dequant = args.dequantize or args.embedding_name in {"alpha_earth_coop", "seamless"}
    dequantize_fn = None
    in_channels_override: int | None = None
    if need_dequant:
        if args.embedding_name == "seamless":
            from dequantize_embeddings import dequantize_esd
            dequantize_fn = dequantize_esd
            in_channels_override = 72
            logger.info("Seamless: dequantize_esd applied (13→72 channels)")
        else:
            from dequantize_embeddings import dequantize_alphaearth_embeddings
            dequantize_fn = dequantize_alphaearth_embeddings
            logger.info("AlphaEarth coop: dequantize_alphaearth_embeddings applied")

    # ── Patch index ───────────────────────────────────────────────────────────
    patch_index = _build_patch_index(args.so2sat_dir, args.output_name, args.year)
    if not patch_index:
        logger.error(
            f"No patch npy files found under {args.so2sat_dir} "
            f"for output_name={args.output_name!r}, year={args.year!r}"
        )
        raise SystemExit(1)

    # ── Item lists ────────────────────────────────────────────────────────────
    if args.global_split:
        gpkg = args.global_gpkg or (args.so2sat_dir / "patches_reference_rxr.gpkg")
        if not gpkg.exists():
            logger.error(f"Global GPKG not found: {gpkg}")
            raise SystemExit(1)
        all_items = _build_global_items(gpkg, patch_index, args.label_col)
        city_dirs: list[Path] = []
        if args.cities and args.cities_dir:
            city_dirs = [args.cities_dir / c for c in args.cities if (args.cities_dir / c).is_dir()]
        city_names = [d.name for d in city_dirs]
        split_mode = "global"
    else:
        if args.cities_dir is None:
            logger.error("--cities-dir is required when not using --global-split")
            raise SystemExit(1)
        city_dirs = sorted(d for d in args.cities_dir.iterdir() if d.is_dir())
        if args.cities:
            city_dirs = [d for d in city_dirs if d.name in args.cities]
        if not city_dirs:
            logger.error(f"No matching city directories found in {args.cities_dir}")
            raise SystemExit(1)
        all_items = []
        for city_dir in city_dirs:
            all_items.extend(
                _build_city_items(args.cities_dir, city_dir.name, patch_index, args.label_col)
            )
        city_names = [d.name for d in city_dirs]
        split_mode = "grid"

    if not all_items:
        logger.error("No items found. Check --so2sat-dir, --output-name, --year.")
        raise SystemExit(1)

    split_counts = {s: sum(1 for _, _, sp in all_items if sp == s) for s in ("train", "val", "test")}
    logger.info(f"Total patches: {len(all_items)}  splits: {split_counts}")

    # ── Channel count ─────────────────────────────────────────────────────────
    in_channels = int(np.load(all_items[0][0], mmap_mode="r").shape[0])
    if in_channels_override is not None:
        in_channels = in_channels_override
    feature_dim = in_channels * (2 if args.pooling == "mean_std" else 1)
    logger.info(f"in_channels={in_channels}  feature_dim={feature_dim}")

    # ── Feature extraction + cache ────────────────────────────────────────────
    _run_label = "global" if args.global_split else "_".join(city_names[:3])
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    cache_key = f"{_run_label}_{args.output_name}_{args.pooling}"

    feats, labels_dict, feat_mean, feat_std = extract_and_cache(
        all_items, args.pooling, dequantize_fn, cache_dir, cache_key, no_cache=args.no_cache,
    )

    # ── Normalisation ─────────────────────────────────────────────────────────
    X_train = normalize(feats["train"], feat_mean, feat_std)
    X_val   = normalize(feats["val"],   feat_mean, feat_std)
    X_test  = normalize(feats["test"],  feat_mean, feat_std)
    y_train = labels_dict["train"]
    y_val   = labels_dict["val"]
    y_test  = labels_dict["test"]

    # ── WandB ─────────────────────────────────────────────────────────────────
    run_cfg = dict(
        task=args.method,
        embedding=args.output_name,
        embedding_name=args.embedding_name,
        cities="all_so2sat" if args.global_split else city_names,
        year=args.year,
        split_mode=split_mode,
        pooling=args.pooling,
        feature_dim=feature_dim,
        **{f"{s}_patches": split_counts[s] for s in ("train", "val", "test")},
    )
    if args.method == "linear_probe":
        run_cfg.update(
            class_weights=args.class_weights,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience,
        )
    else:
        run_cfg["knn_k"] = args.knn_k

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=str(args.output_dir),
            config=run_cfg,
            name=args.run_name,
        )
        run_dir = args.output_dir / wandb.run.name
    else:
        run_name = args.run_name or f"{args.method}_{args.output_name}_{_run_label}"
        run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    test_items = [(p, l, s) for p, l, s in all_items if s == "test"]

    if args.method == "linear_probe":
        # ── Class weights ─────────────────────────────────────────────────────
        cw = compute_class_weights(y_train, args.class_weights, args.num_classes, args.beta)
        if cw is not None:
            logger.info(f"Class weights: min={cw.min():.3f}  max={cw.max():.3f}")

        # ── Linear probe ──────────────────────────────────────────────────────
        logger.info("Training linear probe …")
        probe, _ = train_linear_probe(
            X_train, y_train, X_val, y_val,
            num_classes=args.num_classes,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.early_stopping_patience,
            class_weights=cw,
            run_dir=run_dir,
        )

        probe.eval()
        test_criterion = nn.CrossEntropyLoss(
            weight=cw.to(device) if cw is not None else None, ignore_index=-1,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),
            batch_size=args.batch_size, shuffle=False,
        )
        lp_preds_list, t_loss_total, n_t = [], 0.0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = probe(Xb)
                if (yb != -1).sum() > 0:
                    t_loss_total += test_criterion(logits, yb).item()
                    n_t += 1
                lp_preds_list.append(logits.argmax(1).cpu())
        lp_preds = torch.cat(lp_preds_list).numpy()
        test_loss = t_loss_total / max(1, n_t)
        logger.info(f"[linear_probe] test_loss={test_loss:.4f}")

        metrics = _eval_and_save(
            y_test, lp_preds, "linear_probe", args.num_classes, run_dir, not args.no_wandb,
            test_loss=test_loss,
        )
        _per_city_oa(test_items, lp_preds, args.cities_dir, "linear_probe", run_dir)

    else:  # knn
        # ── kNN baseline ──────────────────────────────────────────────────────
        logger.info(f"Running kNN (k={args.knn_k}) …")
        knn_preds = run_knn(X_train, y_train, X_test, k=args.knn_k)
        metrics = _eval_and_save(
            y_test, knn_preds, "knn", args.num_classes, run_dir, not args.no_wandb,
        )
        _per_city_oa(test_items, knn_preds, args.cities_dir, "knn", run_dir)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    _append_summary(
        args.output_dir / "results" / "summary.csv",
        {
            "run_name": run_dir.name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "method": args.method,
            "embedding": args.output_name,
            "cities": "all_so2sat" if args.global_split else "_".join(city_names[:5]),
            "split_mode": split_mode,
            "pooling": args.pooling,
            "class_weights": args.class_weights if args.method == "linear_probe" else "none",
            "feature_dim": feature_dim,
            "train_n": split_counts["train"],
            "val_n": split_counts["val"],
            "test_n": split_counts["test"],
            "test_oa": f"{metrics['oa']:.4f}",
            "test_f1_macro": f"{metrics['f1_macro']:.4f}",
            "test_kappa": f"{metrics['kappa']:.4f}",
        },
    )

    if not args.no_wandb and wandb.run:
        wandb.finish()

    logger.info(f"Run complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
