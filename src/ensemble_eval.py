"""Softmax-average ensemble evaluation of trained patch classifiers.

Evaluates any number of trained checkpoints — typically the same architecture
trained on different embeddings (e.g. Tessera v1.1 + AlphaEarth coop +
Seamless) — on the global So2Sat test split, and reports metrics for every
model subset (singles, pairs, ..., full ensemble).

Patches are aligned across embeddings by (dataset, patch_id); only patches
present in ALL sources are evaluated, so single-model numbers here can differ
slightly from their original runs (which used each embedding's full test set).

Example:
    python src/ensemble_eval.py \\
        --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 --year 2017 \\
        --model GeoTessera_v1.1_global,tesserav1.1_global,<ckpt>.pt \\
        --model AlphaEarthCoop,alpha_earth_coop,<ckpt>.pt \\
        --model EmbeddedSeamless,seamless,<ckpt>.pt \\
        --tta --output-dir ${DATA_DIR}/output/lcz-classification/dl

Model spec: OUTPUT_NAME,EMBEDDING_NAME,CHECKPOINT[,FAMILY,PRESET]
(family/preset default to resnet,small).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

# ── src/ must be on sys.path (run from repo root) ────────────────────────────

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from datasets.so2sat import PatchDataset, build_patch_index
from models import build_model
from training.evaluate import predict_probs, save_confusion_matrix
from utils.cli import parse_model_spec
from utils.runtime import detect_in_channels, resolve_dequantize, resolve_device



def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    preds = probs.argmax(axis=1)
    return {
        "test_acc":   float(accuracy_score(labels, preds)),
        "test_f1":    float(f1_score(labels, preds, average="macro")),
        "test_kappa": float(cohen_kappa_score(labels, preds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Softmax-average ensemble evaluation on the global So2Sat split."
    )
    parser.add_argument("--so2sat-dir", required=True, type=Path)
    parser.add_argument("--year", required=True)
    parser.add_argument("--global-gpkg", type=Path, default=None,
                        help="Default: {so2sat_dir}/patches_reference_rxr.gpkg")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--label-col", default="LCZ_class")
    parser.add_argument("--model", action="append", required=True, type=parse_model_spec,
                        help="OUTPUT_NAME,EMBEDDING_NAME,CHECKPOINT[,FAMILY,PRESET]; repeatable.")
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    # ── Align patches across all sources ─────────────────────────────────────
    gpkg = args.global_gpkg or (args.so2sat_dir / "patches_reference_rxr.gpkg")
    orig_split = "testing" if args.split == "test" else "validation"
    gdf = gpd.read_file(gpkg)
    rows = gdf[gdf["dataset"] == orig_split]
    logger.info(f"{orig_split} rows in GPKG: {len(rows)}")

    indexes = {}
    for m in args.model:
        for on in m["output_names"]:
            if on not in indexes:
                indexes[on] = build_patch_index(args.so2sat_dir, on, args.year)

    aligned: list[tuple[str, int]] = []   # (patch_id, label 0-16)
    for _, row in rows.iterrows():
        pid = str(row["patch_id"])
        if all(pid in ix.get(orig_split, {}) for ix in indexes.values()):
            aligned.append((pid, int(row[args.label_col]) - 1))
    labels = np.array([lab for _, lab in aligned])
    logger.info(f"Aligned {args.split} patches present in all "
                f"{len(indexes)} sources: {len(aligned)}")

    # ── Per-model inference ───────────────────────────────────────────────────
    names, all_probs = [], {}
    for m in args.model:
        name = f"{m['family']}-{m['preset']}-{m['name']}"
        names.append(name)
        fused = len(m["output_names"]) > 1
        deq = [resolve_dequantize(e) for e in m["embedding_names"]]
        if fused:
            # tuple item paths (one npy per source) → PatchDataset fusion path
            sub = [indexes[on][orig_split] for on in m["output_names"]]
            items = [(tuple(ix[pid] for ix in sub), lab, args.split)
                     for pid, lab in aligned]
            dequantize_fn = [fn for fn, _ in deq]
            in_channels = sum(
                detect_in_channels(p, override)
                for p, (_, override) in zip(items[0][0], deq)
            )
        else:
            idx = indexes[m["output_names"][0]][orig_split]
            items = [(idx[pid], lab, args.split) for pid, lab in aligned]
            dequantize_fn, override = deq[0]
            in_channels = detect_in_channels(items[0][0], override)
        model = build_model(
            m["family"], m["preset"], None,
            in_channels=in_channels, num_classes=args.num_classes,
        )
        ckpt = torch.load(m["checkpoint"], map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model = model.to(device)
        logger.info(f"{name}: loaded {m['checkpoint']} (in_channels={in_channels})")

        ds = PatchDataset(items, args.patch_size, dequantize_fn=dequantize_fn)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        all_probs[name] = predict_probs(model, loader, device, tta=args.tta)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Metrics for every model subset ────────────────────────────────────────
    out_dir = args.output_dir / f"ensemble_{len(names)}models_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            probs = np.mean([all_probs[n] for n in combo], axis=0)
            key = " + ".join(combo)
            results[key] = compute_metrics(probs, labels)
            if r == len(names):
                save_confusion_matrix(
                    labels + 1, probs.argmax(axis=1) + 1, out_dir,
                    filename="ensemble_confusion_matrix.png",
                )

    logger.info(f"── Results on {len(labels)} aligned {args.split} patches "
                f"(TTA={args.tta}) ──")
    for key, met in sorted(results.items(), key=lambda kv: kv[1]["test_kappa"]):
        logger.info(f"  kappa={met['test_kappa']:.4f}  OA={met['test_acc']:.4f}  "
                    f"F1={met['test_f1']:.4f}  | {key}")

    np.savez_compressed(
        out_dir / "probs.npz", labels=labels,
        patch_ids=np.array([pid for pid, _ in aligned]),
        **{n: p for n, p in all_probs.items()},
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved probs + results to {out_dir}")


if __name__ == "__main__":
    main()
