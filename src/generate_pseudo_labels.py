"""Generate pseudo-labels for the unlabeled patch pool (noisy-student step).

Runs a trained teacher checkpoint over the extracted unlabeled patches and
fuses its predictions with the Demuzere weak labels carried in the unlabeled
GeoPackage (from sample_unlabeled_patches.py). Selection rules:

  agree:      teacher top-1 == Demuzere label AND teacher confidence >= --min-conf
              -> keep that label, weight = --agree-weight
  rare-relax: Demuzere label in --rare-classes AND purity >= --rare-min-purity
              AND teacher prob of that class >= --rare-min-prob
              -> keep the Demuzere label, weight = --rare-weight
              (rare classes are where the teacher is weakest, so agreement
              cannot be required — Demuzere is the only usable signal)
  otherwise:  dropped.

Outputs (to --output-dir):
  pseudo_labels.parquet         every unlabeled patch: teacher top-1/conf,
                                Demuzere label/purity/prob, rule, kept flag
  patches_reference_pseudo.gpkg kept rows only, ready for
                                patch_classification.py --pseudo-gpkg
                                (patch_id, dataset='unlabeled', LCZ_class,
                                weight, geometry)

Example:
    python src/generate_pseudo_labels.py \\
        --checkpoint <run_dir>/resnet_small_GeoTessera_v1.1_global_global-best.pt \\
        --unlabeled-gpkg data/patches_reference_unlabeled.gpkg \\
        --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \\
        --output-name GeoTessera_v1.1_global --year 2017 \\
        --embedding-name tesserav1.1_global --tta \\
        --output-dir data/pseudo_labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from datasets.registry import EMBEDDING_REGISTRY
from datasets.so2sat import PatchDataset, build_patch_index
from models import build_model
from training.evaluate import predict_probs
from utils.runtime import detect_in_channels, resolve_dequantize, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher inference + Demuzere fusion over the unlabeled pool."
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Teacher checkpoint (.pt with model_state_dict).")
    parser.add_argument("--family", default="resnet")
    parser.add_argument("--preset", default="small")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--unlabeled-gpkg", required=True, type=Path,
                        help="GeoPackage from sample_unlabeled_patches.py.")
    parser.add_argument("--so2sat-dir", required=True, type=Path,
                        help="Root dir whose unlabeled/ subfolder holds the extracted npys.")
    parser.add_argument("--output-name", required=True,
                        help="Embedding subfolder (e.g. GeoTessera_v1.1_global).")
    parser.add_argument("--year", required=True)
    parser.add_argument("--embedding-name", required=True, choices=sorted(EMBEDDING_REGISTRY))
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    # Selection thresholds
    parser.add_argument("--min-conf", type=float, default=0.70,
                        help="Teacher confidence for the agree rule.")
    parser.add_argument("--agree-weight", type=float, default=0.5)
    parser.add_argument("--rare-classes", type=int, nargs="+",
                        default=[1, 4, 7, 10, 15, 16])
    parser.add_argument("--rare-min-purity", type=float, default=0.75)
    parser.add_argument("--rare-min-prob", type=float, default=0.20,
                        help="Teacher prob of the Demuzere class for the rare-relax rule.")
    parser.add_argument("--rare-weight", type=float, default=0.3)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    gdf = gpd.read_file(args.unlabeled_gpkg)
    logger.info(f"Unlabeled pool: {len(gdf)} patches from {args.unlabeled_gpkg}")

    index = build_patch_index(args.so2sat_dir, args.output_name, args.year)
    unlab = index.get("unlabeled", {})
    has_npy = gdf["patch_id"].astype(str).isin(unlab)
    if (~has_npy).any():
        logger.warning(f"{(~has_npy).sum()} patches have no extracted npy — skipped")
        gdf = gdf[has_npy].reset_index(drop=True)

    # Teacher inference (labels unused during prediction; keep loader order = gdf order)
    items = [(unlab[str(pid)], 0, "unlabeled") for pid in gdf["patch_id"]]
    dequantize_fn, override = resolve_dequantize(args.embedding_name)
    in_channels = detect_in_channels(items[0][0], override)
    model = build_model(args.family, args.preset, args.arch,
                        in_channels=in_channels, num_classes=args.num_classes)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)
    logger.info(f"Teacher: {args.family}/{args.preset} from {args.checkpoint}")

    ds = PatchDataset(items, args.patch_size, dequantize_fn=dequantize_fn)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    probs = predict_probs(model, loader, device, tta=args.tta)   # (N, C)

    teacher_top1 = probs.argmax(axis=1) + 1                       # 1-17
    teacher_conf = probs.max(axis=1)
    demuzere = gdf["LCZ_class"].to_numpy().astype(int)
    purity = gdf["demuzere_purity"].to_numpy()
    prob_of_demuzere = probs[np.arange(len(gdf)), demuzere - 1]

    rare = np.isin(demuzere, args.rare_classes)
    agree = (teacher_top1 == demuzere) & (teacher_conf >= args.min_conf)
    rare_relax = (~agree & rare
                  & (purity >= args.rare_min_purity)
                  & (prob_of_demuzere >= args.rare_min_prob))

    rule = np.where(agree, "agree", np.where(rare_relax, "rare_relax", "dropped"))
    kept = rule != "dropped"
    weight = np.where(agree, args.agree_weight,
                      np.where(rare_relax, args.rare_weight, 0.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = pd.DataFrame(dict(
        patch_id=gdf["patch_id"].astype(str),
        teacher_top1=teacher_top1,
        teacher_conf=teacher_conf,
        demuzere_class=demuzere,
        demuzere_purity=purity,
        demuzere_prob=gdf["demuzere_prob"].to_numpy(),
        prob_of_demuzere=prob_of_demuzere,
        rule=rule,
        kept=kept,
        weight=weight,
    ))
    full.to_parquet(args.output_dir / "pseudo_labels.parquet")

    out = gdf.loc[kept, ["patch_id", "dataset", "tile_name", "geometry"]].copy()
    out["LCZ_class"] = demuzere[kept]           # both rules keep the agreed/Demuzere label
    out["weight"] = weight[kept]
    out_path = args.output_dir / "patches_reference_pseudo.gpkg"
    out.to_file(out_path, driver="GPKG")

    n_agree, n_rare = int(agree.sum()), int(rare_relax.sum())
    logger.info(f"Rules — agree: {n_agree} ({n_agree / len(gdf):.1%}), "
                f"rare_relax: {n_rare} ({n_rare / len(gdf):.1%}), "
                f"dropped: {int((~kept).sum())} ({(~kept).mean():.1%})")
    per_cls = pd.Series(demuzere[kept]).value_counts().sort_index()
    logger.info("Kept per class: "
                + ", ".join(f"LCZ{c}: {n}" for c, n in per_cls.items()))
    logger.info(f"Wrote {args.output_dir / 'pseudo_labels.parquet'} and {out_path} "
                f"({int(kept.sum())} kept)")


if __name__ == "__main__":
    main()
