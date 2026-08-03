"""Stage 9 — validation of pseudo-labels against So2Sat ground truth.

For AOIs that overlap So2Sat cities the grid carries the true ``LCZ_class``
(column ``so2sat_lcz`` in the label output). This module joins on it and reports
per-class confusion, overall/average accuracy, agreement at ``confidence >= 0.8``,
and — crucially — an accuracy-vs-confidence sweep that MUST be monotone if the
confidence model is meaningful. Results are written to a small markdown report.

LCZ 7 patches are excluded from the accuracy of the attempted classes (we never
emit 7), matching the acceptance criterion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from utils.constants import lcz_dict

from .config import LczLabelConfig

LCZ7 = 7
CONF_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]


def _agreement(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).mean()) if len(pred) else float("nan")


def _as_set(v) -> set[int]:
    """Coerce an ``lcz_set`` cell (list/array/None) to a set of ints."""
    if v is None:
        return set()
    try:
        return {int(x) for x in v}
    except TypeError:
        return set()


def _matched(labels: pd.DataFrame) -> pd.DataFrame:
    """Rows with an emitted label (hard or coarse) and So2Sat GT present.

    Adds a ``correct`` column scored by SET MEMBERSHIP: a coarse {3,7} label is
    correct when the So2Sat class is 3 or 7 (hard labels have a singleton set, so
    this reduces to exact match). GT 7 is now included — the router attempts it.
    """
    df = labels.copy()
    df = df[df["label_type"].isin(["hard", "coarse"]) & df["so2sat_lcz"].notna()].copy()
    df["so2sat_lcz"] = df["so2sat_lcz"].astype(int)
    df["_set"] = df["lcz_set"].map(_as_set)
    df["correct"] = [gt in s for gt, s in zip(df["so2sat_lcz"], df["_set"])]
    return df


def validate_labels(labels: pd.DataFrame, aoi_name: str) -> dict:
    """Compute set-aware agreement metrics + LCZ-7 audit for one AOI."""
    if "so2sat_lcz" not in labels.columns:
        logger.warning(f"[{aoi_name}] no so2sat_lcz column — not a So2Sat city, skipping")
        return {}
    m = _matched(labels)
    if m.empty:
        logger.warning(f"[{aoi_name}] no matched labelled patches for validation")
        return {}

    hi = m[m["confidence"] >= 0.8]
    is_coarse = m["label_type"] == "coarse"
    result = {
        "aoi": aoi_name,
        "n_matched": len(m),
        "oa_all": float(m["correct"].mean()),
        "n_high_conf": len(hi),
        "oa_conf80": float(hi["correct"].mean()) if len(hi) else float("nan"),
        "n_coarse": int(is_coarse.sum()),
        "coarse_frac": float(is_coarse.mean()),
        "coarse_sets": {str(sorted(s)): int(c) for s, c in
                        m.loc[is_coarse, "_set"].map(lambda s: tuple(sorted(s))).value_counts().items()},
    }

    # Average (macro) accuracy at conf>=0.8, scored set-aware per GT class
    per_class = {int(c): float(hi.loc[hi["so2sat_lcz"] == c, "correct"].mean())
                 for c in sorted(hi["so2sat_lcz"].unique())}
    result["aa_conf80"] = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    result["per_class_conf80"] = per_class

    # Accuracy-vs-confidence sweep (should be monotone non-decreasing)
    sweep = {}
    for thr in CONF_SWEEP:
        s = m[m["confidence"] >= thr]
        sweep[thr] = (float(s["correct"].mean()) if len(s) else float("nan"), len(s))
    result["sweep"] = sweep
    accs = [sweep[t][0] for t in CONF_SWEEP if not np.isnan(sweep[t][0])]
    result["monotone"] = all(b >= a - 1e-9 for a, b in zip(accs, accs[1:]))

    # Top-5 confusion pairs at conf>=0.8 (hard mispredictions only, for legibility)
    hard_wrong = hi[(hi["label_type"] == "hard") & ~hi["correct"] & hi["lcz"].notna()]
    pairs = (hard_wrong.groupby(["so2sat_lcz", "lcz"]).size()
             .sort_values(ascending=False).head(5))
    result["top_confusions"] = [((int(gt), int(pr)), int(n))
                                for (gt, pr), n in pairs.items()]

    result["lcz7_audit"] = lcz7_audit(m)
    return result


def lcz7_audit(m: pd.DataFrame) -> dict:
    """Router audit against So2Sat LCZ 7 (mandatory report section).

    ``m`` is the matched frame from :func:`_matched`. Reports hard-7
    precision/recall, 7↔3 confusion, and where So2Sat-7 patches were routed —
    the key number is ``contamination`` = fraction of So2Sat-7 emitted as HARD 3
    (the failure mode this design exists to prevent; acceptance bound ≤ 0.10).
    """
    gt7 = m[m["so2sat_lcz"] == LCZ7]
    hard = m["label_type"] == "hard"
    pred7 = m[hard & (m["lcz"] == LCZ7)]
    n_gt7 = len(gt7)
    if n_gt7 == 0:
        return {"n_gt7": 0}

    to_hard7 = int(((gt7["label_type"] == "hard") & (gt7["lcz"] == LCZ7)).sum())
    to_hard3 = int(((gt7["label_type"] == "hard") & (gt7["lcz"] == 3)).sum())
    to_coarse37 = int(gt7["_set"].map(lambda s: s == {3, 7}).sum())
    to_other = n_gt7 - to_hard7 - to_hard3 - to_coarse37

    gt3 = m[m["so2sat_lcz"] == 3]
    return {
        "n_gt7": n_gt7,
        "hard7_precision": float((pred7["so2sat_lcz"] == LCZ7).mean()) if len(pred7) else float("nan"),
        "hard7_recall": to_hard7 / n_gt7,
        "gt7_to_hard7": to_hard7 / n_gt7,
        "gt7_to_coarse37": to_coarse37 / n_gt7,
        "gt7_to_hard3": to_hard3 / n_gt7,      # contamination
        "gt7_to_other": to_other / n_gt7,
        "contamination": to_hard3 / n_gt7,
        "gt3_to_hard7": int(((gt3["label_type"] == "hard") & (gt3["lcz"] == LCZ7)).sum()),
        "n_pred_hard7": len(pred7),
    }


def _name(code: int) -> str:
    return lcz_dict.get(int(code), {}).get("name", f"LCZ{code}")


def write_report(results: list[dict], out_path: Path) -> Path:
    """Render a markdown validation report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# LCZ pseudo-label validation vs So2Sat",
             "(coarse labels scored as set membership; GT 7 included)", ""]
    lines.append("| AOI | n | OA(all) | OA(conf≥0.8) | AA(conf≥0.8) | coarse% | monotone |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        if not r:
            continue
        lines.append(
            f"| {r['aoi']} | {r['n_matched']} | {r['oa_all']:.3f} | "
            f"{r['oa_conf80']:.3f} ({r['n_high_conf']}) | {r['aa_conf80']:.3f} | "
            f"{r['coarse_frac']*100:.0f}% | {'yes' if r['monotone'] else 'NO'} |"
        )
    for r in results:
        if not r:
            continue
        lines += ["", f"## {r['aoi']}", "",
                  "Accuracy vs confidence threshold:", "",
                  "| conf≥ | accuracy | n |", "|---|---|---|"]
        for thr, (acc, n) in r["sweep"].items():
            lines.append(f"| {thr:.1f} | {acc:.3f} | {n} |")
        if r.get("coarse_sets"):
            lines += ["", f"Coarse labels ({r['n_coarse']}, {r['coarse_frac']*100:.0f}%): "
                      + ", ".join(f"{k}×{v}" for k, v in r["coarse_sets"].items())]
        lines += ["", "Top confusions (GT → hard pred) at conf≥0.8:", ""]
        for (gt, pr), n in r["top_confusions"]:
            lines.append(f"- {_name(gt)} → {_name(pr)}: {n}")

        a = r.get("lcz7_audit", {})
        if a.get("n_gt7"):
            lines += ["", "### LCZ 7 audit", "",
                      f"So2Sat-7 patches: {a['n_gt7']}", "",
                      f"- hard-7 precision: {a['hard7_precision']:.3f} "
                      f"(over {a['n_pred_hard7']} predicted hard-7)",
                      f"- hard-7 recall: {a['hard7_recall']:.3f}",
                      f"- So2Sat-7 routed → hard-7: {a['gt7_to_hard7']:.3f}, "
                      f"coarse {{3,7}}: {a['gt7_to_coarse37']:.3f}, "
                      f"**hard-3 (contamination): {a['contamination']:.3f}**, "
                      f"other: {a['gt7_to_other']:.3f}",
                      f"- So2Sat-3 → hard-7 (reverse leak): {a['gt3_to_hard7']}",
                      "",
                      ("✅ contamination ≤ 0.10" if a['contamination'] <= 0.10
                       else "⚠️ contamination > 0.10 — tighten router toward coarse")]
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Validation report -> {out_path}")
    return out_path
