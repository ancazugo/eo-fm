"""Task 1.5.1 — are `tesserav1.1` and `tesserav1.1_global` the same product?

`datasets/tiles.py` routes both through `load_and_dequantize_tessera_representation`
on an int8 + scales pair: same version label, same decode. The 0.80 vs 1.14
median-std gap GATE 0 reported therefore should not exist, and the two Phase 0
measurements came from different sample pools (51-city subset vs global), so
that comparison was never controlled.

This script controls it. It intersects the patch_id sets, samples N of those
**same** ids, and compares the two extractions pixel by pixel on the native grid
— no resize is introduced, because a resize would blur exactly the disagreement
being measured.

Three outcomes were pre-registered in PLAN-V2, each with an action:

    values match, aggregate std differs   sampling artifact -> one product
    near-constant factor between them     scale bug         -> fix before Phase 2
    genuinely uncorrelated                distinct products -> keep both, pick one

The discriminating statistics are the per-channel correlation (are channel k of
each the same feature?), the spread of the per-channel std ratio (a scale bug is
one number, not a distribution), and the R^2 of a full C->C linear map (do they
carry the same information in a different basis?).

Example:

    python src/diagnostics/tessera_product_check.py --n-sample 2000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.embedding_stats import FAMILIES, build_id_index, load_raw  # noqa: E402
from utils.constants import DATA_DIR                                        # noqa: E402


def load_pair(
    ids: list[str],
    a_index: dict[str, Path],
    b_index: dict[str, Path],
    a_name: str,
    b_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Stack both extractions of the same patches into two (C, N_pixels) arrays.

    Also returns a per-column patch index, so the linear-map fit below can hold
    out whole patches rather than pixels.

    Patches whose native shapes disagree are skipped and counted: with different
    crop windows there is no pixel correspondence to compare, so including them
    would manufacture a disagreement that is about geometry, not values.
    """
    a_kind = FAMILIES[a_name]["kind"]
    b_kind = FAMILIES[b_name]["kind"]
    A: list[np.ndarray] = []
    B: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    shape_mismatch = 0
    shapes: Counter = Counter()

    for pid in tqdm(ids, desc=f"{a_name} vs {b_name}", unit="patch"):
        a = load_raw(a_index[pid], a_kind).astype(np.float64)
        b = load_raw(b_index[pid], b_kind).astype(np.float64)
        if a.shape != b.shape:
            shape_mismatch += 1
            continue
        shapes[f"{a.shape[1]}x{a.shape[2]}"] += 1
        A.append(a.reshape(a.shape[0], -1))
        B.append(b.reshape(b.shape[0], -1))
        groups.append(np.full(A[-1].shape[1], len(A) - 1, dtype=np.int32))

    if not A:
        raise SystemExit("No comparable patches: every sampled pair had a shape mismatch.")
    return (
        np.concatenate(A, axis=1),
        np.concatenate(B, axis=1),
        np.concatenate(groups),
        {
            "n_compared": len(A),
            "n_shape_mismatch": shape_mismatch,
            "native_shapes": dict(shapes.most_common(5)),
        },
    )


def _fit_linear_r2(
    Ac: np.ndarray, Bc: np.ndarray, train: np.ndarray, test: np.ndarray
) -> float:
    """R² of a least-squares C->C map fitted on *train* columns, scored on *test*.

    Held out **by patch**, not by pixel: neighbouring pixels of one patch are
    strongly autocorrelated, so a pixel-level split would score the map on
    near-copies of its own training data and report a good fit for nothing.
    """
    coef, *_ = np.linalg.lstsq(Ac[:, train].T, Bc[:, train].T, rcond=None)
    resid = Bc[:, test].T - Ac[:, test].T @ coef
    return float(1.0 - (resid ** 2).sum() / max((Bc[:, test].T ** 2).sum(), 1e-12))


def compare(
    A: np.ndarray, B: np.ndarray, groups: np.ndarray, max_pixels_for_map: int = 200_000
) -> dict:
    """Every statistic that separates the three pre-registered outcomes."""
    C, N = A.shape
    a_mean, b_mean = A.mean(axis=1, keepdims=True), B.mean(axis=1, keepdims=True)
    a_std, b_std = A.std(axis=1), B.std(axis=1)
    Ac, Bc = A - a_mean, B - b_mean

    with np.errstate(divide="ignore", invalid="ignore"):
        Az = Ac / np.maximum(a_std, 1e-12)[:, None]
        Bz = Bc / np.maximum(b_std, 1e-12)[:, None]

    # Matched-channel correlation: is channel k the same feature in both?
    r_diag = (Az * Bz).mean(axis=1)
    # Cross-channel: is it the same feature under a different index/basis?
    X = (Az @ Bz.T) / N
    best_match = np.abs(X).max(axis=1)

    ratio = b_std / np.maximum(a_std, 1e-12)
    rmse = float(np.sqrt(((A - B) ** 2).mean()))

    # R^2 of the best linear C->C map: high means same information in a
    # different basis; low means the two extractions do not describe the same
    # scene. Held out by patch, and reported against a shuffled-patch null so
    # the number cannot be read as evidence when it is only C free parameters
    # fitting per-channel marginals.
    rng = np.random.default_rng(0)
    patches = np.unique(groups)
    rng.shuffle(patches)
    half = len(patches) // 2
    train_p, test_p = set(patches[:half].tolist()), set(patches[half:].tolist())
    tr = np.flatnonzero(np.isin(groups, list(train_p)))
    te = np.flatnonzero(np.isin(groups, list(test_p)))
    if len(tr) > max_pixels_for_map:
        tr = rng.choice(tr, max_pixels_for_map, replace=False)
    if len(te) > max_pixels_for_map:
        te = rng.choice(te, max_pixels_for_map, replace=False)

    r2 = _fit_linear_r2(Ac, Bc, tr, te)
    # Null: roll B by half the sample so every A pixel is paired with a B pixel
    # from a different patch. Marginals and spatial structure are untouched; only
    # the correspondence is destroyed, so whatever R^2 survives is what the map
    # buys from C free parameters alone.
    r2_null = _fit_linear_r2(Ac, np.roll(Bc, N // 2, axis=1), tr, te)

    na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
    return {
        "n_channels": int(C),
        "n_pixels": int(N),
        "pearson_r_matched_channels": {
            "median": float(np.median(r_diag)), "min": float(r_diag.min()),
            "max": float(r_diag.max()),
            "frac_above_0.9": float((r_diag > 0.9).mean()),
        },
        "pearson_r_best_cross_channel": {
            "median": float(np.median(best_match)), "min": float(best_match.min()),
            "max": float(best_match.max()),
        },
        "pearson_r_pooled": float(np.corrcoef(A.ravel(), B.ravel())[0, 1]),
        "rmse": rmse,
        "rms_a": float(np.sqrt((A ** 2).mean())),
        "rms_b": float(np.sqrt((B ** 2).mean())),
        "std_ratio_b_over_a": {
            "median": float(np.median(ratio)), "min": float(ratio.min()),
            "max": float(ratio.max()),
            "cv": float(ratio.std() / max(ratio.mean(), 1e-12)),
        },
        "median_std_a": float(np.median(a_std)),
        "median_std_b": float(np.median(b_std)),
        "linear_map_r2_heldout": r2,
        "linear_map_r2_shuffled_null": r2_null,
        "l2_norm_correlation": float(np.corrcoef(na, nb)[0, 1]),
        "l2_norm_median_a": float(np.median(na)),
        "l2_norm_median_b": float(np.median(nb)),
    }


def verdict(stats: dict) -> tuple[str, str]:
    """Map the statistics onto PLAN-V2's three pre-registered outcomes."""
    r = stats["pearson_r_matched_channels"]["median"]
    cv = stats["std_ratio_b_over_a"]["cv"]
    r2 = stats["linear_map_r2_heldout"]
    r2_null = stats["linear_map_r2_shuffled_null"]

    if r > 0.99:
        return ("same_product",
                "Matched channels agree pixel for pixel; any aggregate difference "
                "is a sampling artifact of the two coverage sets.")
    if r > 0.9 and cv < 0.05:
        return ("scale_bug",
                "Matched channels are perfectly correlated but differ by a "
                "near-constant factor — one decode path is mis-scaled. STOP and "
                "fix before Phase 2; every existing Tessera number is affected.")
    if r2 > 0.5 and r2 - r2_null > 0.3:
        return ("distinct_products_shared_information",
                "Matched channels are uncorrelated but a linear map recovers most "
                "of one from the other: same scene, different feature basis, i.e. "
                "different inference passes sharing a version label. Treat as "
                "distinct products, keep both registry keys, pick one canonical.")
    return ("unrelated",
            "Neither matched channels nor any linear map relates the two. They do "
            "not describe the same data — check the extraction geometry and year.")


def markdown_report(results: dict, meta: dict) -> str:
    lines = [
        "### Task 1.5.1 — Tessera product identity",
        "",
        f"{meta['n_sample']} patch_ids sampled from the intersection of each pair "
        f"(seed {meta['seed']}), cultural-split **{meta['split']}** set, compared "
        "on the **native** grid with no resize.",
        "",
        "| pair | ids in common | compared | shape mismatches |",
        "|---|---|---|---|",
    ]
    for key, r in results.items():
        a, b = key.split(" vs ")
        lines.append(
            f"| `{a}` vs `{b}` | {r['n_common_ids']:,} | {r['n_compared']:,} | "
            f"{r['n_shape_mismatch']} |"
        )

    lines += [
        "",
        "| pair | matched-channel r (med) | best cross-channel r (med) | "
        "std ratio (med / CV) | linear-map R² (held out / null) | L2-norm r | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        a, b = key.split(" vs ")
        s = r["stats"]
        lines.append(
            f"| `{a}` vs `{b}` | {s['pearson_r_matched_channels']['median']:.4f} | "
            f"{s['pearson_r_best_cross_channel']['median']:.4f} | "
            f"{s['std_ratio_b_over_a']['median']:.3f} / "
            f"{s['std_ratio_b_over_a']['cv']:.3f} | "
            f"{s['linear_map_r2_heldout']:.4f} / "
            f"{s['linear_map_r2_shuffled_null']:.4f} | "
            f"{s['l2_norm_correlation']:.4f} | "
            f"**{r['verdict']}** |"
        )

    lines.append("")
    for key, r in results.items():
        lines.append(f"- **{key}** — {r['verdict_reason']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Task 1.5.1 — Tessera product identity.")
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--split", default="training")
    p.add_argument("--year", default="2017")
    p.add_argument("--pairs", nargs="+",
                   default=["tesserav1.1:tesserav1.1_global",
                            "tesserav1.1_global:tesserav2",
                            "tesserav1.1:tesserav2"],
                   help="Colon-separated family pairs to compare.")
    p.add_argument("--n-sample", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path,
                   default=Path("diagnostics/tessera_product_check.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/tessera_product_check.md"))
    args = p.parse_args()

    names = sorted({n for pair in args.pairs for n in pair.split(":")})
    indexes = {n: build_id_index(args.so2sat_dir, n, args.split, args.year) for n in names}
    for n, ix in indexes.items():
        logger.info(f"  {n:20s} {len(ix):>7d} patches")

    rng = np.random.default_rng(args.seed)
    results: dict[str, dict] = {}
    for pair in args.pairs:
        a_name, b_name = pair.split(":")
        common = sorted(set(indexes[a_name]) & set(indexes[b_name]))
        logger.info(f"{a_name} ∩ {b_name}: {len(common):,} shared patch_ids")
        ids = common
        if len(ids) > args.n_sample:
            ids = sorted(ids[i] for i in rng.choice(len(ids), args.n_sample, replace=False))

        A, B, groups, geom = load_pair(
            ids, indexes[a_name], indexes[b_name], a_name, b_name
        )
        stats = compare(A, B, groups)
        v, reason = verdict(stats)
        results[f"{a_name} vs {b_name}"] = {
            "n_common_ids": len(common), **geom,
            "stats": stats, "verdict": v, "verdict_reason": reason,
        }
        logger.info(
            f"  matched-channel r={stats['pearson_r_matched_channels']['median']:.4f}  "
            f"best cross-channel r={stats['pearson_r_best_cross_channel']['median']:.4f}  "
            f"std ratio={stats['std_ratio_b_over_a']['median']:.3f} "
            f"(CV {stats['std_ratio_b_over_a']['cv']:.3f})  "
            f"linear R²={stats['linear_map_r2_heldout']:.4f} "
            f"(null {stats['linear_map_r2_shuffled_null']:.4f})  →  {v}"
        )

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split, "year": args.year,
        "n_sample": args.n_sample, "seed": args.seed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"meta": meta, "pairs": results}, indent=2))
    args.output_md.write_text(markdown_report(results, meta))
    logger.info(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
