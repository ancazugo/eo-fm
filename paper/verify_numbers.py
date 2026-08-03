#!/usr/bin/env python
"""Re-derive every headline number in paper/main.tex from cached artifacts.

Usage:
    python paper/verify_numbers.py            # local JSON artifacts only
    python paper/verify_numbers.py --wandb    # also verify run metrics via WandB API

Exit code 0 = all numbers match; non-zero = at least one FAIL (printed).
Numbers verified here may be unwrapped from the red \\uv{} draft marker in main.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

D = Path(os.environ.get("DATA_DIR", "/maps/acz25/phd-thesis-data"))
DL = D / "output/lcz-classification/dl"
ENS = DL / "ensemble_coopv1/ensemble_3models_test"

TOL = 5e-5  # values are asserted at 4-decimal precision

checks: list[tuple[str, float, float]] = []  # (label, expected, actual)


def expect(label: str, expected: float, actual: float) -> None:
    checks.append((label, expected, actual))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true", help="also verify WandB run summaries")
    ap.add_argument("--splits", action="store_true", help="also verify split sizes from the gpkg (slow)")
    args = ap.parse_args()

    # --- solo members on the aligned 23,858-patch universe -------------------
    r = json.load(open(ENS / "results.json"))
    m = {k.replace("resnet-small-", ""): v for k, v in r.items()}
    expect("v3 solo kappa (aligned)", 0.6497, m["GeoTessera_v1.1_global"]["test_kappa"])
    expect("v3 solo OA", 0.6794, m["GeoTessera_v1.1_global"]["test_acc"])
    expect("v3 solo macro-F1", 0.5656, m["GeoTessera_v1.1_global"]["test_f1"])
    expect("coop-v1 solo kappa (aligned)", 0.5153, m["AlphaEarthCoop"]["test_kappa"])
    expect("seamless solo kappa (aligned)", 0.5059, m["EmbeddedSeamless"]["test_kappa"])

    # --- ensembles / stacking / calibration / LOCO ---------------------------
    s = json.load(open(ENS / "stacking_results_calibrated_loco.json"))
    expect("equal-weight kappa", 0.6691, s["equal_weight"]["test"]["kappa"])
    expect("equal-weight OA", 0.6983, s["equal_weight"]["test"]["acc"])
    expect("equal-weight F1", 0.5978, s["equal_weight"]["test"]["f1"])
    expect("weighted [.2/.7/.1] kappa", 0.6923, s["weighted"]["test"]["kappa"])
    expect("weighted OA", 0.7202, s["weighted"]["test"]["acc"])
    expect("weighted F1", 0.6065, s["weighted"]["test"]["f1"])
    expect("equal-weight calibrated kappa", 0.6824, s["equal_weight_calibrated"]["test"]["kappa"])
    expect("weighted calibrated kappa", 0.6903, s["weighted_calibrated"]["test"]["kappa"])
    expect("stacked-LR (leaked) kappa", 0.7756, s["stacked_lr"]["test"]["kappa"])
    expect("weighted LOCO kappa", 0.6871, s["weighted_loco"]["test"]["kappa"])
    expect("weighted LOCO OA", 0.7154, s["weighted_loco"]["test"]["acc"])
    expect("weighted LOCO F1", 0.6038, s["weighted_loco"]["test"]["f1"])
    expect("stacked-LR LOCO kappa", 0.6121, s["stacked_lr_loco"]["test"]["kappa"])
    expect("stacked-LR LOCO OA", 0.6461, s["stacked_lr_loco"]["test"]["acc"])
    expect("stacked-LR LOCO F1", 0.5167, s["stacked_lr_loco"]["test"]["f1"])
    for i, (name, t) in enumerate(zip(["tessera", "coop", "seamless"], [0.9189, 0.4581, 0.9786])):
        expect(f"temperature {name}", t, s["weighted_calibrated"]["temperatures"][i])
    pc = s["weighted_loco"]["per_city"]
    expect("LOCO Munich kappa", 0.9007, pc["Munich"]["kappa_weighted"])
    expect("LOCO Nairobi kappa", 0.4748, pc["Nairobi"]["kappa_weighted"])
    expect("LOCO Santiago kappa", 0.4748, pc["Santiago"]["kappa_weighted"])

    # --- aux structural corrector (grid versions v1/v2/v3) --------------------
    for ver, suffix, kappa, acc, f1 in [
        ("v2 (mid grid)", "_v2", 0.7081, 0.7343, 0.6129),
        ("v3 (fine grid)", "_v3", 0.7055, 0.7320, 0.6227),
    ]:
        a = json.load(open(ENS / f"stacking_results_auxgate_full_confgated{suffix}.json"))
        expect(f"aux corrector {ver} kappa", kappa, a["probs_aux_offset_loco"]["test"]["kappa"])
        expect(f"aux corrector {ver} OA", acc, a["probs_aux_offset_loco"]["test"]["acc"])
        expect(f"aux corrector {ver} F1", f1, a["probs_aux_offset_loco"]["test"]["f1"])
    a = json.load(open(ENS / "stacking_results_auxgate_full_confgated.json"))
    expect("aux-only LR kappa", 0.5785, a["aux_only_lr_loco"]["test"]["kappa"])
    expect("aux-only LR OA", 0.6166, a["aux_only_lr_loco"]["test"]["acc"])
    expect("aux-only LR F1", 0.4521, a["aux_only_lr_loco"]["test"]["f1"])
    expect("free LR [logp+aux] kappa", 0.6584, a["probs_aux_lr_loco"]["test"]["kappa"])
    expect("no-aux LR control kappa", 0.6167, a["probs_ctrl_lr_loco"]["test"]["kappa"])

    # --- aligned-universe size -------------------------------------------------
    import numpy as np
    probs = np.load(ENS / "probs.npz")
    n = probs[list(probs.keys())[0]].shape[0]
    expect("aligned test universe", 23858, n)

    # --- intrinsic diagnostics (tab:diag) -------------------------------------
    # Silhouettes from the projection run dirs; domain AUC / frac-sparse from the
    # 2026-07-14 recompute (paper/artifacts/, protocol = embedding_metrics.
    # train_test_shift on scaler+PCA-50 GAP feats, seed 42). NOTE: the June memory
    # note had coop/seamless AUC values TRANSPOSED; these are the correct ones.
    import csv

    viz = D / "output/lcz-classification/embedding_viz"
    sil = {}
    for emb in ("GeoTessera_v1.1_global", "AlphaEarthCoop", "EmbeddedSeamless"):
        with open(viz / f"proj_{emb}_global_gap/separability.csv") as fh:
            sil[emb] = {row["facet"]: float(row["silhouette"]) for row in csv.DictReader(fh)}
    expect("tessera continent silhouette", 0.0392, sil["GeoTessera_v1.1_global"]["continent"])
    expect("tessera lcz silhouette", -0.0063, sil["GeoTessera_v1.1_global"]["lcz_name"])
    expect("coop continent silhouette", 0.1529, sil["AlphaEarthCoop"]["continent"])
    expect("coop lcz silhouette", 0.0753, sil["AlphaEarthCoop"]["lcz_name"])
    expect("seamless continent silhouette", -0.0124, sil["EmbeddedSeamless"]["continent"])
    expect("seamless lcz silhouette", 0.0908, sil["EmbeddedSeamless"]["lcz_name"])
    auc = json.load(open(Path(__file__).parent / "artifacts/domain_auc_recompute_2026-07-14.json"))
    expect("tessera domain AUC", 0.8845, auc["GeoTessera_v1.1_global"]["domain_auc"])
    expect("tessera frac sparse", 0.5537, auc["GeoTessera_v1.1_global"]["frac_sparse"])
    expect("coop domain AUC", 0.9575, auc["AlphaEarthCoop"]["domain_auc"])
    expect("coop frac sparse", 0.7659, auc["AlphaEarthCoop"]["frac_sparse"])
    expect("seamless domain AUC", 0.7856, auc["EmbeddedSeamless"]["domain_auc"])
    expect("seamless frac sparse", 0.3750, auc["EmbeddedSeamless"]["frac_sparse"])

    # --- TTA (negatives) + 4-model aux ensemble --------------------------------
    for t, solo, ens in [("tta_adabn_val", 0.5982, 0.6090), ("tta_tent_val", 0.6084, 0.6164)]:
        r2 = json.load(open(DL / t / "results.json"))
        expect(f"{t} tessera pooled", solo,
               r2["resnet-small-GeoTessera_v1.1_global"]["pooled_test_kappa"])
        s2 = json.load(open(DL / t / "stacking_results.json"))
        expect(f"{t} LOCO ensemble", ens, s2["weighted_loco"]["test"]["kappa"])
    for f, wl, corr in [("stacking_3sub.json", 0.6857, 0.7018), ("stacking_4model.json", 0.6918, 0.7023)]:
        d4 = json.load(open(DL / "ensemble_aux4" / f))
        expect(f"aux4 {f} weighted LOCO", wl, d4["weighted_loco"]["test"]["kappa"])
        expect(f"aux4 {f} corrector", corr, d4["probs_aux_offset_loco"]["test"]["kappa"])

    # --- WandB-verified run metrics (teachers, students, negatives) ----------
    if args.wandb:
        import wandb

        api = wandb.Api()

        def summary(name: str) -> dict:
            # Prefer runs that actually logged test metrics: crashed launches
            # (e.g. the segfaulted 2026-07-14 GMM attempts) leave zombie
            # "running" runs with empty summaries under the same display name.
            runs = list(api.runs("phd-thesis-team/lcz-classification-dl", {"display_name": name}))
            for run in runs:
                if "test_kappa" in run.summary:
                    return dict(run.summary)
            if runs:
                return dict(runs[0].summary)
            raise KeyError(name)

        for name, kappa in [
            ("opt3-lr5e-4-warmup3", 0.6190),          # tessera teacher
            ("wandering-firefly-267", 0.5195),        # coop teacher
            ("good-flower-268", 0.5125),              # seamless teacher
            ("student-noisy-v1", 0.6419),
            ("student-noisy-v3", 0.6497),
            ("student-coop-v1", 0.5218),              # full coop universe
            ("clean-universe-260", 0.6047),
            ("fusion-tessera-coop", 0.5974),          # phase-0 negative
            # sampler-sqrt 0.6099: wandb run deleted + log truncated at ep30 —
            # UNVERIFIED; re-evaluate dl/sampler-sqrt checkpoint before citing.
            ("logit-adj-tau1", 0.5541),
            ("opt1-resnet101-medium", 0.6033),        # capacity negative
            ("effortless-sponge-266", 0.5908),        # resnet101 + opt3 recipe
            ("aux-fusion-v1", 0.6310),                # stage-2a fusion
            ("wise-lake-259", 0.4985),                # tessera kNN k=20 GAP
            ("paper-knn-coop", 0.4636),               # coop kNN (2026-07-14)
            ("paper-knn-seamless", 0.4576),           # seamless kNN (2026-07-14)
            ("paper-gmm-tessera", 0.4208),            # GMM density (2026-07-14)
            ("paper-gmm-coop", 0.3746),
            ("paper-gmm-seamless", 0.4601),           # GMM: seamless ranks FIRST
            ("paper-lp-tessera", 0.5961),             # linear probes (2026-07-15)
            ("paper-lp-coop", 0.5337),
            ("paper-lp-seamless", 0.4710),
            ("opt3-tessera-seed1", 0.6268),           # seed variance (2026-07-15)
            ("opt3-tessera-seed2", 0.6171),
            ("opt3-coop-seed1", 0.5288),
            ("opt3-coop-seed2", 0.5177),
            ("opt3-seamless-seed1", 0.5124),
            ("opt3-seamless-seed2", 0.4937),
            ("student-seamless-v1", 0.5135),          # seamless SSL round (+0.1 only)
            # London per-city grid rows (quarantined from culture-10 numbers):
            ("percity-london-tv11global-small", 0.9639),
            ("percity-london-tv11global-medium", 0.9682),
            ("percity-london-tv11global-large", 0.9589),
            ("percity-london-coop-small", 0.9624),
            ("percity-london-tv11-small", 0.9742),
        ]:
            expect(f"wandb {name} kappa", kappa, summary(name).get("test_kappa", float("nan")))
        expect("wandb opt3 OA", 0.6514, summary("opt3-lr5e-4-warmup3").get("test_acc", float("nan")))

    # --- split sizes (culture-10 identity) ------------------------------------
    if args.splits:
        import geopandas as gpd

        g = gpd.read_file(D / "input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg")
        vc = g["dataset"].value_counts()
        expect("total patches", 400673, len(g))
        expect("train (culture-10)", 352366, vc["training"])
        expect("val (culture-10)", 24119, vc["validation"])
        expect("test (culture-10)", 24188, vc["testing"])

    # --- report ----------------------------------------------------------------
    fails = 0
    for label, exp, act in checks:
        ok = abs(exp - act) <= (TOL if exp < 1 else 0.5)
        status = "PASS" if ok else "FAIL"
        fails += not ok
        print(f"[{status}] {label:42s} expected {exp:<10.4f} got {act:.4f}")
    print(f"\n{len(checks) - fails}/{len(checks)} numbers verified.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
