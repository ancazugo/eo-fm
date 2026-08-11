"""Task 1.5.0 — what the bilinear resize to 32x32 actually does, per family.

Two things are measured here, and they turn out to be unrelated.

**1. How much resampling is happening.** ``PatchDataset._resize`` sends every
patch through ``F.interpolate(mode="bilinear")`` to ``--patch-size``. Nobody had
checked what the native crop shapes are. If 10 m embeddings over a 320 m patch
were 32x32 natively the call would be a no-op; if they are not, an information
loss sits underneath every result in the chapter and belongs in the methods
section regardless of what any later experiment shows.

**2. Whether the resize explains the sentinel-variance discrepancy.** GATE 0
measured AlphaEarth's -128 sentinel at ~28% of per-channel variance on the
native grid; GATE 1 measured <1% post-resize and attributed the drop to bilinear
dilution. That explanation requires spreading one input pixel over ~30 output
pixels, which a ~1.05x resize cannot do. This script recomputes the
decomposition **four ways** with the native shapes attached, and splits it by
whether the patch is in the paired cross-family intersection, which is where the
real explanation lives.

The four variants, in pipeline order:

    native, unmasked          the raw array as stored, sentinel included
    native, masked            sentinel pixels excluded  -> the GATE 0 native number
    resized, unmasked, unfilled   Phase 0's path: nan_to_num -> dequantize -> resize
    resized, masked, unfilled     same, sentinel pixels excluded post-resize
    resized, masked, FILLED       Phase 1's path: mean-fill before the resize

Measure-only: nothing here is imported by the training path.

Example (the Phase 1.5 run):

    python src/diagnostics/resize_audit.py \\
        --n-sample 2000 --seed 0 --patch-size 32 \\
        --output-json diagnostics/resize_audit.json \\
        --output-md diagnostics/resize_audit.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.registry import get_nodata_predicate          # noqa: E402
from diagnostics.embedding_stats import (                   # noqa: E402
    FAMILIES,
    build_id_index,
    load_raw,
)
from utils.constants import DATA_DIR                        # noqa: E402
from utils.runtime import resolve_dequantize               # noqa: E402

# The families the GATE 0 audit sampled paired. Membership of this intersection
# is the variable that actually explains the 28%-vs-<1% gap, so it has to be
# computed from exactly this set.
PAIRED_FAMILIES = [
    "alpha_earth_coop", "tesserav1.1_global", "seamless", "sentinel1", "sentinel2",
]

AUDIT_FAMILIES = [
    "alpha_earth_coop", "tesserav1.1_global", "tesserav1.1", "tesserav2",
    "seamless", "sentinel1", "sentinel2",
]


# ── Accumulation ─────────────────────────────────────────────────────────────

class WeightedMoments:
    """Streaming per-channel mean/std over an explicit per-pixel weight mask.

    Weights are 0/1 validity flags, so ``n`` counts valid pixel-samples and the
    masked and unmasked variants differ only in which pixels are summed. float64
    throughout: the point of the exercise is a difference of a few percent
    between two large sums.
    """

    def __init__(self, n_channels: int) -> None:
        self.C = n_channels
        self.n = 0.0
        self.s1 = np.zeros(n_channels, dtype=np.float64)
        self.s2 = np.zeros(n_channels, dtype=np.float64)

    def update(self, arr: np.ndarray, valid: np.ndarray | None = None) -> None:
        """arr: (C, H, W) float; valid: (H, W) bool or None for all-valid."""
        flat = arr.reshape(self.C, -1).astype(np.float64)
        if valid is None:
            self.s1 += flat.sum(axis=1)
            self.s2 += (flat * flat).sum(axis=1)
            self.n += flat.shape[1]
        else:
            w = valid.reshape(-1).astype(np.float64)
            self.s1 += (flat * w).sum(axis=1)
            self.s2 += ((flat * flat) * w).sum(axis=1)
            self.n += w.sum()

    @property
    def mean(self) -> np.ndarray:
        return self.s1 / max(self.n, 1.0)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.s2 / max(self.n, 1.0) - self.mean ** 2, 0.0))

    def summary(self) -> dict:
        std = self.std
        return {
            "median_std": float(np.median(std)),
            "min_std": float(std.min()),
            "max_std": float(std.max()),
            "n_pixel_samples": float(self.n),
        }


def variance_share(unmasked: WeightedMoments, masked: WeightedMoments) -> float:
    """Fraction of the unmasked per-channel variance contributed by nodata.

    Computed on the medians so it lines up with the median-std numbers reported
    everywhere else; returns 0.0 when there is no nodata to attribute.
    """
    su, sm = np.median(unmasked.std), np.median(masked.std)
    if su <= 0:
        return 0.0
    return float(max(0.0, 1.0 - (sm / su) ** 2))


# ── Pipeline replication ─────────────────────────────────────────────────────

def resize_image(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """PatchDataset._resize, on a numpy (C, H, W) array."""
    if arr.shape[-2:] == (patch_size, patch_size):
        return arr
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)
    out = F.interpolate(t, size=(patch_size, patch_size),
                        mode="bilinear", align_corners=False)
    return out.squeeze(0).numpy()


def resize_valid(valid: np.ndarray, patch_size: int) -> np.ndarray:
    """PatchDataset._resize_valid: bilinear, then demand a full weight of 1.

    Conservative by construction — any output pixel whose interpolation touched
    an invalid input pixel comes back invalid.
    """
    if valid.shape[-2:] == (patch_size, patch_size):
        return valid
    t = torch.from_numpy(valid.astype(np.float32))[None, None]
    out = F.interpolate(t, size=(patch_size, patch_size),
                        mode="bilinear", align_corners=False)[0, 0].numpy()
    return out >= 1.0 - 1e-6


def audit_family(
    family: str,
    ids: list[str],
    index: dict[str, Path],
    paired_ids: set[str],
    patch_size: int,
    seed: int,
    norm_pixels_per_patch: int = 16,
) -> dict:
    """One family, one pass for the geometry + a second pass for the fill variant.

    The mean-fill variant needs the masked channel mean, which is only known
    after a full pass, so families that actually have a sentinel are read twice.
    Families with no nodata skip the second pass entirely — with nothing to fill,
    all five variants collapse onto the two unmasked ones.
    """
    spec = FAMILIES[family]
    dequantize_fn, _ = resolve_dequantize(spec["embedding_name"])
    predicate = get_nodata_predicate(spec["embedding_name"])
    rng = np.random.default_rng(seed)

    shapes: Counter = Counter()
    n_noop = 0
    factors_h: list[float] = []
    factors_w: list[float] = []
    n_invalid = n_pixels = 0
    n_patches_with_invalid = 0
    norms_native: list[np.ndarray] = []
    norms_resized: list[np.ndarray] = []

    acc: dict[str, WeightedMoments] = {}
    # Same decomposition restricted to each side of the paired intersection.
    sub: dict[str, dict] = {
        "paired": {"n": 0, "invalid": 0, "pixels": 0, "acc": {}},
        "unpaired": {"n": 0, "invalid": 0, "pixels": 0, "acc": {}},
    }

    def _acc(store: dict, key: str, C: int) -> WeightedMoments:
        if key not in store:
            store[key] = WeightedMoments(C)
        return store[key]

    for pid in tqdm(ids, desc=f"{family} (pass 1)", unit="patch"):
        raw = load_raw(index[pid], spec["kind"])
        H, W = raw.shape[-2:]
        shapes[f"{H}x{W}"] += 1
        if (H, W) == (patch_size, patch_size):
            n_noop += 1
        factors_h.append(H / patch_size)
        factors_w.append(W / patch_size)

        invalid = predicate(raw)                      # (H, W), stored units
        valid = ~invalid
        n_invalid += int(invalid.sum())
        n_pixels += invalid.size
        if invalid.any():
            n_patches_with_invalid += 1

        arr = np.nan_to_num(raw, nan=0.0)
        if dequantize_fn is not None:
            arr = dequantize_fn(arr)
        C = arr.shape[0]

        rs = resize_image(arr, patch_size)            # no fill: the Phase 0 path
        rs_valid = resize_valid(valid, patch_size)

        _acc(acc, "native_unmasked", C).update(arr)
        _acc(acc, "native_masked", C).update(arr, valid)
        _acc(acc, "resized_unmasked_unfilled", C).update(rs)
        _acc(acc, "resized_masked_unfilled", C).update(rs, rs_valid)

        side = "paired" if pid in paired_ids else "unpaired"
        s = sub[side]
        s["n"] += 1
        s["invalid"] += int(invalid.sum())
        s["pixels"] += invalid.size
        _acc(s["acc"], "native_unmasked", C).update(arr)
        _acc(s["acc"], "native_masked", C).update(arr, valid)
        _acc(s["acc"], "resized_unmasked_unfilled", C).update(rs)

        # L2 norms on valid pixels only, native vs resized — the norm-shrinkage
        # note (bilinear interpolation of unit-norm vectors does not preserve
        # the norm).
        for src, store in ((arr, norms_native), (rs, norms_resized)):
            flat = src.reshape(src.shape[0], -1)
            vflat = (valid if src is arr else rs_valid).reshape(-1)
            keep = np.flatnonzero(vflat)
            if keep.size:
                pick = rng.choice(keep, min(norm_pixels_per_patch, keep.size),
                                  replace=False)
                store.append(np.linalg.norm(flat[:, pick].astype(np.float64), axis=0))

    has_nodata = n_invalid > 0
    filled_summary = None
    if has_nodata:
        # Second pass: fill invalid pixels with the masked channel mean BEFORE
        # the resize, exactly as PatchDataset._load_source does.
        fill = acc["native_masked"].mean.astype(np.float32)
        filled = WeightedMoments(fill.shape[0])
        for pid in tqdm(ids, desc=f"{family} (pass 2, fill)", unit="patch"):
            raw = load_raw(index[pid], spec["kind"])
            invalid = predicate(raw)
            arr = np.nan_to_num(raw, nan=0.0)
            if dequantize_fn is not None:
                arr = dequantize_fn(arr)
            if invalid.any():
                arr[:, invalid] = fill[:, None]
            filled.update(resize_image(arr, patch_size),
                          resize_valid(~invalid, patch_size))
        filled_summary = filled.summary()

    def _norm_stats(chunks: list[np.ndarray]) -> dict:
        v = np.concatenate(chunks) if chunks else np.array([0.0])
        return {"p1": float(np.percentile(v, 1)), "p50": float(np.percentile(v, 50)),
                "p99": float(np.percentile(v, 99)), "mean": float(v.mean())}

    out = {
        "family": family,
        "embedding_name": spec["embedding_name"],
        "n_patches": len(ids),
        "patch_size": patch_size,
        "geometry": {
            "native_shapes": dict(shapes.most_common()),
            "frac_exact_noop": float(n_noop / max(len(ids), 1)),
            "resample_factor_h": {
                "median": float(np.median(factors_h)),
                "min": float(np.min(factors_h)), "max": float(np.max(factors_h)),
            },
            "resample_factor_w": {
                "median": float(np.median(factors_w)),
                "min": float(np.min(factors_w)), "max": float(np.max(factors_w)),
            },
        },
        "nodata": {
            "frac_invalid_pixels": float(n_invalid / max(n_pixels, 1)),
            "frac_patches_with_invalid": float(n_patches_with_invalid / max(len(ids), 1)),
        },
        "variants": {k: v.summary() for k, v in acc.items()},
        "sentinel_variance_share": {
            "native": variance_share(acc["native_unmasked"], acc["native_masked"]),
            "resized_unfilled": variance_share(
                acc["resized_unmasked_unfilled"], acc["resized_masked_unfilled"]
            ),
        },
        "l2_norm": {"native": _norm_stats(norms_native),
                    "resized": _norm_stats(norms_resized)},
        "paired_split": {
            side: {
                "n_patches": s["n"],
                "frac_invalid_pixels": float(s["invalid"] / max(s["pixels"], 1)),
                "variants": {k: v.summary() for k, v in s["acc"].items()},
                "sentinel_variance_share_native": (
                    variance_share(s["acc"]["native_unmasked"], s["acc"]["native_masked"])
                    if s["n"] else None
                ),
            }
            for side, s in sub.items()
        },
    }
    if filled_summary is not None:
        out["variants"]["resized_masked_filled"] = filled_summary
    return out


# ── Reporting ────────────────────────────────────────────────────────────────

def markdown_report(results: dict[str, dict], meta: dict) -> str:
    lines = [
        "### Task 1.5.0 — Resize audit",
        "",
        f"{meta['n_sample']} patches per family from the cultural-split "
        f"**{meta['split']}** set (each family sampled over its own ids, seed "
        f"{meta['seed']}), resized to {meta['patch_size']}x{meta['patch_size']}.",
        "",
        "#### Geometry — the resize is essentially never a no-op",
        "",
        "| family | native shapes (top 3) | exact no-op | H factor (med) | W factor (med) |",
        "|---|---|---|---|---|",
    ]
    for f, r in results.items():
        g = r["geometry"]
        top = ", ".join(f"{k} ({v})" for k, v in list(g["native_shapes"].items())[:3])
        lines.append(
            f"| `{f}` | {top} | {g['frac_exact_noop']:.1%} | "
            f"{g['resample_factor_h']['median']:.3f} | "
            f"{g['resample_factor_w']['median']:.3f} |"
        )

    lines += [
        "",
        "A factor > 1 is a downsample, < 1 an upsample.",
        "",
        "#### Sentinel variance decomposition — five variants",
        "",
        "| family | invalid px | native unmasked | native masked | resized unmasked | "
        "resized masked | resized masked+filled |",
        "|---|---|---|---|---|---|---|",
    ]
    for f, r in results.items():
        v = r["variants"]
        filled = v.get("resized_masked_filled")
        filled_cell = f"{filled['median_std']:.4f}" if filled else "—"
        lines.append(
            f"| `{f}` | {r['nodata']['frac_invalid_pixels']:.4%} | "
            f"{v['native_unmasked']['median_std']:.4f} | "
            f"{v['native_masked']['median_std']:.4f} | "
            f"{v['resized_unmasked_unfilled']['median_std']:.4f} | "
            f"{v['resized_masked_unfilled']['median_std']:.4f} | {filled_cell} |"
        )

    lines += [
        "",
        "Sentinel share of per-channel variance, native vs post-resize (unfilled):",
        "",
        "| family | native | post-resize | change |",
        "|---|---|---|---|",
    ]
    for f, r in results.items():
        s = r["sentinel_variance_share"]
        lines.append(
            f"| `{f}` | {s['native']:.2%} | {s['resized_unfilled']:.2%} | "
            f"{s['resized_unfilled'] - s['native']:+.2%} |"
        )

    lines += [
        "",
        "#### Paired vs unpaired — where the nodata actually lives",
        "",
        "`paired` = the patch_id is present in every one of "
        f"{', '.join('`' + x + '`' for x in PAIRED_FAMILIES)}, i.e. it would have "
        "been eligible for the GATE 0 sample.",
        "",
        "| family | paired n | paired invalid px | unpaired n | unpaired invalid px |",
        "|---|---|---|---|---|",
    ]
    for f, r in results.items():
        p, u = r["paired_split"]["paired"], r["paired_split"]["unpaired"]
        lines.append(
            f"| `{f}` | {p['n_patches']} | {p['frac_invalid_pixels']:.4%} | "
            f"{u['n_patches']} | {u['frac_invalid_pixels']:.4%} |"
        )

    lines += [
        "",
        "#### L2 norm, native vs post-resize (valid pixels)",
        "",
        "| family | native p1 / p50 / p99 | resized p1 / p50 / p99 |",
        "|---|---|---|",
    ]
    for f, r in results.items():
        n, s = r["l2_norm"]["native"], r["l2_norm"]["resized"]
        lines.append(
            f"| `{f}` | {n['p1']:.3f} / {n['p50']:.3f} / {n['p99']:.3f} | "
            f"{s['p1']:.3f} / {s['p50']:.3f} / {s['p99']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Task 1.5.0 — resize audit.")
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--split", default="training")
    p.add_argument("--year", default="2017")
    p.add_argument("--families", nargs="+", default=AUDIT_FAMILIES)
    p.add_argument("--n-sample", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--output-json", type=Path, default=Path("diagnostics/resize_audit.json"))
    p.add_argument("--output-md", type=Path, default=Path("diagnostics/resize_audit.md"))
    args = p.parse_args()

    indexes = {
        f: build_id_index(args.so2sat_dir, f, args.split, args.year)
        for f in sorted(set(args.families) | set(PAIRED_FAMILIES))
    }
    for f, ix in indexes.items():
        logger.info(f"  {f:20s} {len(ix):>7d} patches")

    paired_ids = set.intersection(*(set(indexes[f]) for f in PAIRED_FAMILIES))
    logger.info(f"Paired intersection over {PAIRED_FAMILIES}: {len(paired_ids)}")

    rng = np.random.default_rng(args.seed)
    results: dict[str, dict] = {}
    for f in args.families:
        ids = sorted(indexes[f])
        if len(ids) > args.n_sample:
            ids = sorted(ids[i] for i in rng.choice(len(ids), args.n_sample, replace=False))
        results[f] = audit_family(f, ids, indexes[f], paired_ids,
                                  args.patch_size, args.seed)
        g, v = results[f]["geometry"], results[f]["variants"]
        logger.info(
            f"{f}: no-op {g['frac_exact_noop']:.1%}, factor "
            f"{g['resample_factor_h']['median']:.3f}x{g['resample_factor_w']['median']:.3f}, "
            f"sentinel share {results[f]['sentinel_variance_share']['native']:.2%} native "
            f"-> {results[f]['sentinel_variance_share']['resized_unfilled']:.2%} resized "
            f"(std {v['native_unmasked']['median_std']:.4f} -> "
            f"{v['resized_unmasked_unfilled']['median_std']:.4f})"
        )

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split, "year": args.year, "n_sample": args.n_sample,
        "seed": args.seed, "patch_size": args.patch_size,
        "paired_families": PAIRED_FAMILIES,
        "paired_intersection_size": len(paired_ids),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"meta": meta, "families": results}, indent=2))
    args.output_md.write_text(markdown_report(results, meta))
    logger.info(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
