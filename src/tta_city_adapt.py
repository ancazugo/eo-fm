"""Per-city test-time adaptation (AdaBN / TENT) of trained patch classifiers.

The global-split models carry train-city BatchNorm statistics that are wrong
for the 10 unseen val/test cities (per-city kappa spans 0.47-0.90). This
script adapts each model to each city on unlabeled patches only, then writes
val/test probs in the exact npz format ensemble_eval.py produces, so
ensemble_stacking.py (--city-holdout etc.) runs on the output unchanged.

Methods:
  none  — plain inference (regression check against cached ensemble_eval runs)
  adabn — reset BN running stats, recompute them on the city's adapt-split
          patches (forward passes only, no gradients)
  tent  — AdaBN reset + entropy minimisation updating only BN affine params

Honesty: with --adapt-split val (default) adaptation sees only val-split
inputs of the city (never labels, never test patches); --adapt-split test is
the standard transductive variant — flag it when reporting.

Example:
    python src/tta_city_adapt.py \\
        --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 --year 2017 \\
        --model GeoTessera_v1.1_global,tesserav1.1_global,<ckpt>.pt \\
        --method adabn --tta --output-dir ${DL}/tta_adabn_val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from torch.nn.modules.batchnorm import _BatchNorm

# ── src/ must be on sys.path (run from repo root) ────────────────────────────

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from datasets.so2sat import PatchDataset, build_patch_index
from utils.cli import parse_model_spec
from utils.geo_lookup import assign_cities
from models import build_model
from training.evaluate import predict_probs
from utils.runtime import detect_in_channels, resolve_dequantize, resolve_device


def align_split(gdf, split: str, indexes: dict, label_col: str):
    """(patch_id, label 0-16) pairs present in every source, ensemble_eval-style."""
    dataset = "validation" if split == "val" else "testing"
    rows = gdf[gdf["dataset"] == dataset]
    aligned = [
        (str(row["patch_id"]), int(row[label_col]) - 1)
        for _, row in rows.iterrows()
        if all(str(row["patch_id"]) in ix.get(dataset, {}) for ix in indexes.values())
    ]
    logger.info(f"Aligned {split} patches present in all {len(indexes)} sources: {len(aligned)}")
    return aligned, dataset


def reset_bn(model: nn.Module) -> None:
    """AdaBN prep: zero the BN running stats; momentum=None → exact cumulative mean."""
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.reset_running_stats()
            m.momentum = None


@torch.no_grad()
def adabn_pass(model: nn.Module, loader, device: torch.device) -> None:
    """Recompute BN running stats on the adapt loader (train-mode forwards)."""
    model.train()
    for batch in loader:
        model(batch["image"].to(device).float())


def tent_pass(model: nn.Module, loader, device: torch.device,
              lr: float, epochs: int) -> float:
    """Entropy minimisation on BN affine params only. Returns final mean entropy."""
    model.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            for p in (m.weight, m.bias):
                if p is not None:
                    p.requires_grad_(True)
                    params.append(p)
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    entropy = float("nan")
    for _ in range(epochs):
        for batch in loader:
            logits = model(batch["image"].to(device).float())
            probs = torch.softmax(logits, dim=1)
            loss = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            entropy = loss.item()
    return entropy


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-city AdaBN/TENT adaptation + inference.")
    parser.add_argument("--so2sat-dir", required=True, type=Path)
    parser.add_argument("--year", required=True)
    parser.add_argument("--global-gpkg", type=Path, default=None,
                        help="Default: {so2sat_dir}/patches_reference_rxr.gpkg")
    parser.add_argument("--label-col", default="LCZ_class")
    parser.add_argument("--model", action="append", required=True, type=parse_model_spec,
                        help="OUTPUT_NAME,EMBEDDING_NAME,CHECKPOINT[,FAMILY,PRESET]; repeatable.")
    parser.add_argument("--method", choices=["none", "adabn", "tent"], default="adabn")
    parser.add_argument("--adapt-split", choices=["val", "test"], default="val",
                        help="val = honest (adapt on val inputs, eval on test); test = transductive.")
    parser.add_argument("--tent-lr", type=float, default=1e-4)
    parser.add_argument("--tent-epochs", type=int, default=1)
    parser.add_argument("--cities", nargs="+", default=None,
                        help="Restrict to these cities (smoke tests); default all.")
    parser.add_argument("--city-bounds", type=Path, default=Path("data/so2sat_guppd_bounds.csv"))
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    for m in args.model:
        if len(m["output_names"]) != 1:
            raise SystemExit(
                "tta_city_adapt.py supports single-source models only "
                f"(got fused spec '{m['name']}')"
            )

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}  method={args.method}  adapt-split={args.adapt_split}")

    # ── Align patches across all sources on BOTH splits ──────────────────────
    gpkg = args.global_gpkg or (args.so2sat_dir / "patches_reference_rxr.gpkg")
    gdf = gpd.read_file(gpkg)
    indexes = {}
    for m in args.model:
        on = m["output_names"][0]
        if on not in indexes:
            indexes[on] = build_patch_index(args.so2sat_dir, on, args.year)
    aligned, datasets, ids, labels, cities = {}, {}, {}, {}, {}
    for split in ("val", "test"):
        aligned[split], datasets[split] = align_split(gdf, split, indexes, args.label_col)
        ids[split] = np.array([pid for pid, _ in aligned[split]])
        labels[split] = np.array([lab for _, lab in aligned[split]])
        cities[split] = assign_cities(ids[split], split, gpkg, args.city_bounds)
    city_list = args.cities or sorted(set(cities["val"]) | set(cities["test"]))
    logger.info(f"Cities: {city_list}")

    # ── Per-model, per-city adaptation + inference ────────────────────────────
    def make_loader(items, dequantize_fn, shuffle=False):
        ds = PatchDataset(items, args.patch_size, dequantize_fn=dequantize_fn)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers)

    names, all_probs, report = [], {}, {}
    for m in args.model:
        name = f"{m['family']}-{m['preset']}-{m['name']}"
        names.append(name)
        dequantize_fn, override = resolve_dequantize(m["embedding_names"][0])
        items = {
            split: [(indexes[m["output_names"][0]][datasets[split]][pid], lab, split)
                    for pid, lab in aligned[split]]
            for split in ("val", "test")
        }
        in_channels = detect_in_channels(items["val"][0][0], override)
        ckpt = torch.load(m["checkpoint"], map_location=device)
        state = ckpt.get("model_state_dict", ckpt)

        def fresh_model() -> nn.Module:
            model = build_model(m["family"], m["preset"], None,
                                in_channels=in_channels, num_classes=args.num_classes)
            model.load_state_dict(state)
            return model.to(device)

        probs = {split: np.full((len(ids[split]), args.num_classes), np.nan,
                                dtype=np.float32) for split in ("val", "test")}

        if args.method == "none":
            model = fresh_model()
            for split in ("val", "test"):
                probs[split][:] = predict_probs(
                    model, make_loader(items[split], dequantize_fn), device, tta=args.tta)
            del model
        else:
            for city in city_list:
                masks = {s: cities[s] == city for s in ("val", "test")}
                adapt_items = [it for it, keep in
                               zip(items[args.adapt_split], masks[args.adapt_split]) if keep]
                if not adapt_items:
                    logger.warning(f"{name} / {city}: no adapt patches, skipping")
                    continue
                model = fresh_model()
                bn0 = next(mm for mm in model.modules() if isinstance(mm, _BatchNorm))
                mean_before = bn0.running_mean.detach().clone()
                reset_bn(model)
                adapt_loader = make_loader(adapt_items, dequantize_fn,
                                           shuffle=(args.method == "tent"))
                adabn_pass(model, adapt_loader, device)
                if args.method == "tent":
                    ent = tent_pass(model, adapt_loader, device,
                                    args.tent_lr, args.tent_epochs)
                    logger.info(f"{name} / {city}: TENT final entropy {ent:.4f}")
                delta = (bn0.running_mean - mean_before).abs().mean().item()
                model.eval()
                for split in ("val", "test"):
                    sub = [it for it, keep in zip(items[split], masks[split]) if keep]
                    if sub:
                        probs[split][masks[split]] = predict_probs(
                            model, make_loader(sub, dequantize_fn), device, tta=args.tta)
                k_city = cohen_kappa_score(
                    labels["test"][masks["test"]],
                    probs["test"][masks["test"]].argmax(axis=1))
                logger.info(f"{name} / {city}: n_adapt={len(adapt_items)}  "
                            f"BN1 |Δmean|={delta:.4f}  test kappa={k_city:.4f}")
                report.setdefault(name, {}).setdefault("per_city_test_kappa", {})[city] = float(k_city)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        all_probs[name] = probs
        done = ~np.isnan(probs["test"]).any(axis=1)
        pooled = cohen_kappa_score(labels["test"][done], probs["test"][done].argmax(axis=1))
        report.setdefault(name, {})["pooled_test_kappa"] = float(pooled)
        report[name]["n_test_covered"] = int(done.sum())
        logger.info(f"{name}: pooled test kappa={pooled:.4f} on {done.sum()} patches")

    # ── Save in ensemble_eval npz format (stacking script runs unchanged) ─────
    for split in ("val", "test"):
        out_dir = args.output_dir / f"ensemble_{len(names)}models_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # drop rows any model failed to cover (e.g. --cities smoke runs)
        covered = np.all([~np.isnan(all_probs[n][split]).any(axis=1) for n in names], axis=0)
        np.savez_compressed(
            out_dir / "probs.npz",
            labels=labels[split][covered], patch_ids=ids[split][covered],
            **{n: all_probs[n][split][covered] for n in names},
        )
    report["_config"] = {k: str(v) for k, v in vars(args).items() if k != "model"}
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved probs + results to {args.output_dir}")


if __name__ == "__main__":
    main()
