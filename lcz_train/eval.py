"""T5 — the shared yardstick: block-level accuracy/macro-F1 on held-out cities.

Same blocks, same splits => A (dense, majority-voted per block) and B (direct)
are directly comparable. Coarse-labelled blocks score correct on set
membership and are also reported separately. Secondary: So2Sat patch-level
agreement via the Stage 8 patch transfer, for continuity with prior results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

N_LCZ = 17

# Explicit confusions the spec calls out: 7<->3 (informal vs formal compact
# lowrise) and compact<->open at matched height (2<->5 mid, 3<->6 low).
CONFUSION_PAIRS = [(7, 3), (3, 7), (2, 5), (5, 2), (3, 6), (6, 3)]


# ── A: dense inference -> per-block majority vote ─────────────────────────────

def dense_predict_full(
    model: torch.nn.Module, mosaic: np.ndarray, device: torch.device, *, chunk: int = 512,
) -> np.ndarray:
    """Chunked per-pixel argmax over a full mosaic. Returns (H, W) int64, 0-indexed class."""
    c, h, w = mosaic.shape
    pred = np.zeros((h, w), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for r0 in range(0, h, chunk):
            for c0 in range(0, w, chunk):
                r1, c1 = min(r0 + chunk, h), min(c0 + chunk, w)
                x = torch.from_numpy(
                    np.asarray(mosaic[:, r0:r1, c0:c1], dtype=np.float32)
                ).unsqueeze(0).to(device)
                logits = model(x)
                pred[r0:r1, c0:c1] = logits.argmax(dim=1)[0].cpu().numpy()
    return pred


def majority_vote_per_block(pred: np.ndarray, block_idx: np.ndarray, n_blocks: int) -> np.ndarray:
    """Per-block majority-vote prediction. Returns (n_blocks,) 1-indexed LCZ code, 0 = no votes."""
    flat_idx = block_idx.ravel().astype(np.int64)
    flat_pred = pred.ravel().astype(np.int64)
    valid = flat_idx > 0
    idx0 = flat_idx[valid] - 1
    classes = flat_pred[valid]
    combined = idx0 * N_LCZ + classes
    votes = np.bincount(combined, minlength=n_blocks * N_LCZ).reshape(n_blocks, N_LCZ)
    has_votes = votes.sum(axis=1) > 0
    out = np.zeros(n_blocks, dtype=np.int64)
    out[has_votes] = votes[has_votes].argmax(axis=1) + 1
    return out


# ── Shared block-level scoring (A and B feed this the same way) ──────────────

def _as_set(v) -> frozenset:
    if v is None:
        return frozenset()
    try:
        return frozenset(int(x) for x in v)
    except TypeError:
        return frozenset()


def evaluate_blocks(pred_lcz: np.ndarray, gt: pd.DataFrame) -> dict:
    """Block-level metrics: OA/macro-F1, per-class, explicit confusions,
    block_kind stratification, coarse reported separately.

    ``gt`` rows align positionally with ``pred_lcz`` (same block order) and
    must carry ``label_type, lcz, lcz_set`` and, for stratification,
    ``block_kind``. Only rows with an emitted label (hard/coarse) and a
    prediction (``pred_lcz > 0``) are scored.
    """
    df = gt.copy()
    df["pred"] = pred_lcz
    df["_set"] = df["lcz_set"].map(_as_set)
    scored = df[df["label_type"].isin(["hard", "coarse"]) & (df["pred"] > 0)].copy()
    if scored.empty:
        return {"n": 0}

    scored["correct"] = [p in s for p, s in zip(scored["pred"], scored["_set"])]
    is_hard = scored["label_type"] == "hard"

    result = {
        "n": len(scored),
        "oa": float(scored["correct"].mean()),
        "n_coarse": int((~is_hard).sum()),
        "coarse_frac": float((~is_hard).mean()),
        "coarse_oa": float(scored.loc[~is_hard, "correct"].mean()) if (~is_hard).any() else float("nan"),
    }

    # Macro-F1 + per-class: only meaningful against a definite hard GT class.
    hard = scored[is_hard].copy()
    hard["lcz"] = hard["lcz"].astype(int)
    per_class = {}
    f1s = []
    for c in sorted(hard["lcz"].unique()):
        tp = int(((hard["lcz"] == c) & (hard["pred"] == c)).sum())
        fp = int(((hard["lcz"] != c) & (hard["pred"] == c)).sum())
        fn = int(((hard["lcz"] == c) & (hard["pred"] != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": int((hard["lcz"] == c).sum())}
        f1s.append(f1)
    result["per_class"] = per_class
    result["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    result["hard_oa"] = float((hard["lcz"] == hard["pred"]).mean()) if len(hard) else float("nan")

    # Explicit confusions
    result["confusions"] = {
        f"{gt_c}->{pr_c}": int(((hard["lcz"] == gt_c) & (hard["pred"] == pr_c)).sum())
        for gt_c, pr_c in CONFUSION_PAIRS
    }

    # block_kind stratification (the fallback stratum must stay visible)
    if "block_kind" in scored.columns:
        result["by_block_kind"] = {
            kind: {"n": len(g), "oa": float(g["correct"].mean())}
            for kind, g in scored.groupby("block_kind")
        }
    return result


# ── Secondary: So2Sat patch-level agreement (via the Stage 8 patch transfer) ──

def patch_agreement(patch_labels: pd.DataFrame, *, min_dominant: float = 0.75) -> dict:
    """Continuity metric: block-derived dominant_lcz vs So2Sat ground truth."""
    if "so2sat_lcz" not in patch_labels.columns:
        return {}
    df = patch_labels[patch_labels["dominant_frac"] >= min_dominant].copy()
    df = df[df["so2sat_lcz"].notna()]
    if df.empty:
        return {}
    dom_sets = df["dominant_set"].map(_as_set)
    correct = [int(gt) in s for gt, s in zip(df["so2sat_lcz"], dom_sets)]
    return {"n": len(df), "oa": float(np.mean(correct))}
