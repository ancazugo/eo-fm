"""Seed ensembles of patch classifiers, on exactly the trainer's test set.

Scores every checkpoint of one or more "arms" (a set of seeds of one recipe) on
the split ``patch_classification.py --global-split`` evaluates, averages the
softmax within each arm, and reports single-model and ensemble metrics.

Why not ``ensemble_eval.py``: that tool aligns patches ACROSS embeddings and
enumerates every model subset (2^21 for three 7-seed arms), and it predates
checkpoint-stored normalisation. Here every model is rebuilt through
``infer_roi.load_model_and_normalize`` -- its own channel stats, nodata masking
and dihedral TTA -- and the item list comes from the trainer's own
``build_so2sat_items`` (global split + patch manifest), so a single model's
kappa here must equal the one its run logged. That is checked against the
run's ``test_confusion_matrix.npy`` and a mismatch aborts, because an ensemble
number is only worth reporting if its members reproduce.

Ensembling adds points to ANY arm, so compare arms ensemble-to-ensemble, not an
ensemble against a single model. ``--k`` also reports the mean over all k-seed
sub-ensembles, the fair comparison when arms have different seed counts.

Example:
    python src/seed_ensemble.py \\
        --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 --year 2017 \\
        --output-name GeoTessera_v2 --embedding-name tesserav2 \\
        --patch-manifest diagnostics/patch_manifest_v1.parquet \\
        --arm "base=<dir_s0>/*-best.pt,<dir_s1>/*-best.pt" \\
        --arm "E3=<dir>/wudapt-e3-mobilenet-small-s*/*-best.pt" \\
        --family mobilenet --preset small --cache-dir <dir> --out results.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))


def metrics(probs: np.ndarray, labels: np.ndarray, num_classes: int = 17) -> dict:
    pred = probs.argmax(1)
    cm = np.zeros((num_classes, num_classes))
    np.add.at(cm, (labels, pred), 1)
    return cm_metrics(cm)


def cm_metrics(cm: np.ndarray) -> dict:
    cm = cm.astype(float)
    tp = np.diag(cm)
    n = cm.sum()
    po = tp.sum() / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    # Macro over classes that occur in the labels or the predictions -- the
    # torchmetrics convention the trainer logs, so a single model reproduces
    # its run's test_f1 (an absent class would otherwise count as F1 = 0).
    present = (cm.sum(0) + cm.sum(1)) > 0
    return {"kappa": float((po - pe) / (1 - pe)), "oa": float(po),
            "f1_macro": float(f1[present].mean()), "f1": [round(float(x), 4) for x in f1]}


def score_checkpoint(ckpt: Path, items, args, cache_dir: Path) -> np.ndarray:
    """(N, C) TTA softmax for one checkpoint, cached on (checkpoint, items)."""
    key = hashlib.sha1(
        (str(ckpt.resolve()) + str(ckpt.stat().st_mtime) + args.split
         + str(len(items)) + str(items[0].path) + str(items[-1].path)
         + ("|notta" if args.no_tta else "")).encode()
    ).hexdigest()[:16]
    cache = cache_dir / f"probs_{ckpt.parent.name}_{key}.npy"
    if cache.exists():
        return np.load(cache)

    import torch
    from torch.utils.data import DataLoader

    from datasets.registry import get_nodata_predicate
    from datasets.so2sat import PatchDataset
    from infer_roi import load_model_and_normalize
    from training.evaluate import predict_probs
    from utils.runtime import resolve_dequantize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalize = load_model_and_normalize(
        ckpt, args.family, args.embedding_name, device,
        preset=args.preset, patch_size=args.patch_size,
    )
    dequantize_fn, _ = resolve_dequantize(args.embedding_name)
    mean, std = normalize if normalize is not None else (None, None)
    ds = PatchDataset(
        items, args.patch_size, dequantize_fn=dequantize_fn,
        nodata_mode=args.nodata_mode,
        nodata_predicate=get_nodata_predicate(args.embedding_name),
        normalize="channel" if normalize is not None else "none",
        channel_mean=mean, channel_std=std,
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=args.num_workers)
    with torch.no_grad():
        probs = predict_probs(model, loader, device, tta=not args.no_tta).astype(np.float32)
    np.save(cache, probs)
    return probs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--so2sat-dir", type=Path, required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--output-name", required=True)
    p.add_argument("--embedding-name", required=True)
    p.add_argument("--patch-manifest", type=Path, default=None)
    p.add_argument("--arm", action="append", required=True,
                   help="NAME=glob[,glob...] of best checkpoints; repeatable")
    p.add_argument("--family", default="mobilenet")
    p.add_argument("--preset", default="small")
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--nodata-mode", default="mask", choices=["zero", "mask"])
    p.add_argument("--no-tta", action="store_true",
                   help="score without dihedral TTA -- for arms whose runs were "
                        "evaluated without --tta, so the reproduction check can pass")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--k", type=int, nargs="*", default=[3],
                   help="also report the mean over all k-seed sub-ensembles")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-verify", action="store_true",
                   help="skip the single-model reproduction check (e.g. --split val)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    from datasets.so2sat import build_so2sat_items

    items, _ = build_so2sat_items(
        args.so2sat_dir, args.output_name, args.year, global_split=True,
        embedding_names=[args.embedding_name], patch_manifest=args.patch_manifest,
    )
    items = [it for it in items if it.split == args.split]
    labels = np.array([it.label for it in items])
    logger.info(f"{args.split}: {len(items):,} patches")

    arms: dict[str, list[Path]] = {}
    for spec in args.arm:
        name, globs = spec.split("=", 1)
        paths = sorted({Path(f) for g in globs.split(",") for f in glob.glob(g)})
        if not paths:
            raise SystemExit(f"arm {name}: no checkpoints match {globs}")
        arms[name] = paths

    out: dict = {"split": args.split, "n": len(items), "arms": {}}
    for name, paths in arms.items():
        probs, singles = [], []
        for ck in paths:
            pr = score_checkpoint(ck, items, args, args.cache_dir)
            m = metrics(pr, labels)
            cm_file = ck.parent / "test_confusion_matrix.npy"
            if args.split == "test" and not args.no_verify and cm_file.exists():
                logged = cm_metrics(np.load(cm_file))["kappa"]
                if abs(logged - m["kappa"]) > 2e-3:
                    raise SystemExit(
                        f"{ck.parent.name}: rescored kappa {m['kappa']:.4f} != logged "
                        f"{logged:.4f} -- the item list or input path differs from "
                        "training; no ensemble number would be trustworthy."
                    )
                m["logged_kappa"] = logged
            logger.info(f"  {name} {ck.parent.name}: kappa {m['kappa']:.4f}"
                        + (f" (logged {m['logged_kappa']:.4f})" if "logged_kappa" in m else ""))
            probs.append(pr)
            singles.append({"run": ck.parent.name, **m})
        stack = np.stack(probs)
        ens = metrics(stack.mean(0), labels)
        ks = np.array([s["kappa"] for s in singles])
        res = {"runs": singles, "single_mean": float(ks.mean()),
               "single_sd": float(ks.std(ddof=1)) if len(ks) > 1 else 0.0,
               "ensemble": ens}
        for k in args.k:
            if 1 < k < len(paths):
                sub = [metrics(stack[list(c)].mean(0), labels)["kappa"]
                       for c in itertools.combinations(range(len(paths)), k)]
                res[f"ensemble_k{k}"] = {"mean": float(np.mean(sub)),
                                         "sd": float(np.std(sub)), "n_subsets": len(sub)}
        out["arms"][name] = res
        logger.info(f"{name}: {len(paths)} seeds, single {ks.mean():.4f} +/- "
                    f"{res['single_sd']:.4f} -> ensemble kappa {ens['kappa']:.4f} "
                    f"OA {ens['oa']:.4f} F1 {ens['f1_macro']:.4f}"
                    + "".join(f" | k={k} mean {res[f'ensemble_k{k}']['mean']:.4f}"
                              for k in args.k if f"ensemble_k{k}" in res))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
