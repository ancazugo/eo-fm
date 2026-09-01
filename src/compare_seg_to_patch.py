"""Score a segmentation run against the patch-classification ladder, honestly.

A segmentation model under ``--split-mode global`` cannot predict on every
So2Sat test patch: a tile is the unit of prediction, and tiles that fail split
purity or the proximity buffer are dropped whole, taking their patches with
them. Measured coverage on Nairobi in the first verification run was 41 %.

So the two numbers in

    seg   kappa = 0.6xx  over N_seg patches
    patch kappa = 0.6497 over 23,858 patches

are computed on **different test sets**, and comparing them directly flatters
whichever side kept more of the easy patches. This script removes that
objection the only way available: intersect on ``patch_id`` and re-score *both*
sides on the intersection.

Inputs are the ``probs*.npz`` files both pipelines already emit
(``labels`` / ``patch_ids`` + one ``(N, C)`` array per model), so nothing has
to be re-run.

    python src/compare_seg_to_patch.py \\
        --seg-npz  <seg_run>/probs_patch.npz \\
        --patch-npz <ensemble_dir>/ensemble_3models_test/probs.npz \\
        --output-json comparison.json

Per-city kappa is reported for both sides when the segmentation npz carries a
``cities`` column, because a pooled number hides the spread that matters --
Munich 0.90 against Nairobi 0.47 on the patch task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.lcz_metrics import lcz_metrics_from_cm, load_similarity_matrix  # noqa: E402

_RESERVED = {"labels", "patch_ids", "cities", "datasets"}


def load_probs(path: Path) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """Return (patch_ids, labels, {model: (N, C)}, cities_or_None)."""
    z = np.load(path, allow_pickle=True)
    models = {k: z[k] for k in z.files if k not in _RESERVED}
    if not models:
        raise ValueError(f"{path} carries no model probability arrays")
    cities = z["cities"] if "cities" in z.files else None
    return (z["patch_ids"].astype(str), z["labels"].astype(int), models, cities)


def score(labels: np.ndarray, preds: np.ndarray, num_classes: int = 17) -> dict:
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(cm, (labels, preds), 1)
    out = {
        "n": int(len(labels)),
        "oa": float((labels == preds).mean()),
        "kappa": float(cohen_kappa_score(labels, preds,
                                         labels=list(range(num_classes)))),
        "f1_macro": float(f1_score(labels, preds, labels=list(range(num_classes)),
                                   average="macro", zero_division=0)),
    }
    if num_classes == 17:
        out.update({k.replace("test_", ""): v for k, v in
                    lcz_metrics_from_cm(cm, load_similarity_matrix()).items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-npz", required=True, type=Path,
                    help="probs_patch.npz from a segmentation run.")
    ap.add_argument("--patch-npz", required=True, type=Path,
                    help="probs.npz from ensemble_eval.py / the patch pipeline.")
    ap.add_argument("--num-classes", type=int, default=17)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    seg_ids, seg_lab, seg_models, seg_cities = load_probs(args.seg_npz)
    pat_ids, pat_lab, pat_models, _ = load_probs(args.patch_npz)

    seg_pos = {pid: i for i, pid in enumerate(seg_ids)}
    pat_pos = {pid: i for i, pid in enumerate(pat_ids)}
    common = sorted(set(seg_pos) & set(pat_pos))
    if not common:
        raise SystemExit(
            "No patch_ids in common. Both files must cover the same split — "
            "patch_id restarts at 000000 in each So2Sat original split, so a "
            "val npz will not intersect a test npz."
        )
    si = np.array([seg_pos[p] for p in common])
    pi = np.array([pat_pos[p] for p in common])

    if not (seg_lab[si] == pat_lab[pi]).all():
        n_bad = int((seg_lab[si] != pat_lab[pi]).sum())
        raise SystemExit(
            f"{n_bad} of {len(common)} shared patch_ids carry different labels "
            "in the two files. They are not describing the same patches — "
            "check that both were produced for the same split and label column."
        )
    labels = seg_lab[si]

    logger.info(
        f"Intersection: {len(common)} patches "
        f"({len(common)/len(seg_ids):.1%} of the segmentation set, "
        f"{len(common)/len(pat_ids):.1%} of the patch set)"
    )
    if len(common) < len(pat_ids):
        logger.warning(
            f"The patch model is being re-scored on {len(common)} of its "
            f"{len(pat_ids)} patches. Its headline number from the full set is "
            "NOT the number below, and only the number below is comparable to "
            "the segmentation result."
        )

    report: dict = {
        "n_intersection": len(common),
        "n_seg_total": int(len(seg_ids)),
        "n_patch_total": int(len(pat_ids)),
        "seg": {}, "patch": {},
    }
    for name, probs in seg_models.items():
        report["seg"][name] = score(labels, probs[si].argmax(1), args.num_classes)
    for name, probs in pat_models.items():
        report["patch"][name] = score(labels, probs[pi].argmax(1), args.num_classes)

    if seg_cities is not None:
        cities = seg_cities.astype(str)[si]
        per_city: dict = {}
        for city in sorted({c for c in cities.tolist() if c}):
            m = cities == city
            per_city[city] = {
                "n": int(m.sum()),
                "seg": {n: score(labels[m], p[si][m].argmax(1), args.num_classes)["kappa"]
                        for n, p in seg_models.items()},
                "patch": {n: score(labels[m], p[pi][m].argmax(1), args.num_classes)["kappa"]
                          for n, p in pat_models.items()},
            }
        report["per_city"] = per_city

    print(json.dumps(report, indent=2, sort_keys=True))
    for side in ("seg", "patch"):
        for name, m in report[side].items():
            logger.info(
                f"{side:5s} {name:28s} n={m['n']:6d}  kappa={m['kappa']:.4f}  "
                f"OA={m['oa']:.4f}  F1={m['f1_macro']:.4f}  kappa_w={m['kappa_w']:.4f}"
            )

    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        logger.info(f"Written to {args.output_json}")


if __name__ == "__main__":
    main()
