"""Task 0.1 — embedding scale audit across the So2Sat cultural-split training set.

Measures the numerical scale of every input family the classification pipeline
can consume, *exactly as the training path sees it*: the per-patch load,
``np.nan_to_num``, the dequantize function chosen by
``utils.runtime.resolve_dequantize``, and the bilinear resize to ``--patch-size``
are the same four steps as ``datasets.so2sat.PatchDataset._load_source`` +
``_resize``. Nothing is written back into the pipeline — this module is
measure-only.

Why it exists: ``training.augment.augment_images`` adds Gaussian noise at a
FIXED ABSOLUTE sigma of 0.05 to unnormalized inputs. The relative size of that
perturbation is ``0.05 / median_per_channel_std``, which differs per family, so
the cross-family comparison is confounded until inputs are normalized. This
script reports that ratio explicitly.

Patches are sampled PAIRED across families (the intersection of patch_ids) so
the numbers are comparable patch-for-patch.

Example (the Phase 0 run):

    python src/diagnostics/embedding_stats.py \\
        --n-sample 5000 --seed 0 --patch-size 32 \\
        --output-json diagnostics/embedding_stats.json \\
        --output-md diagnostics/embedding_stats.md

Example (the 51-city Tessera subset, sampled on its own):

    python src/diagnostics/embedding_stats.py \\
        --families tesserav1.1 --n-sample 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.constants import DATA_DIR          # noqa: E402
from utils.runtime import resolve_dequantize  # noqa: E402

# The absolute sigma hardcoded in training.augment.augment_images.
AUGMENT_NOISE_SIGMA = 0.05

# Where each family's patches live under {so2sat_dir}/{split}/ and how they are
# named. ``embedding_name`` is fed to resolve_dequantize, which is the single
# source of truth for whether a dequantization is applied (a genuine no-op for
# the tessera and sentinel families).
FAMILIES: dict[str, dict] = {
    "alpha_earth_coop": {
        "subdir": "AlphaEarthCoop", "year": True, "kind": "npy",
        "prefix": "patch_", "suffix": ".npy", "embedding_name": "alpha_earth_coop",
        "note": "int8 values stored as float32; dequantized at train time",
    },
    "tesserav1.1_global": {
        "subdir": "GeoTessera_v1.1_global", "year": True, "kind": "npy",
        "prefix": "patch_", "suffix": ".npy", "embedding_name": "tesserav1.1_global",
        "note": "already dequantized at extraction time",
    },
    "tesserav1.1": {
        "subdir": "GeoTessera_v1.1", "year": True, "kind": "npy",
        "prefix": "patch_", "suffix": ".npy", "embedding_name": "tesserav1.1",
        "note": "51-city subset only — sample it on its own, not paired",
    },
    "tesserav2": {
        "subdir": "GeoTessera_v2", "year": True, "kind": "npy",
        "prefix": "patch_", "suffix": ".npy", "embedding_name": "tesserav2",
        "note": "already dequantized at extraction time; covers ~36.6% of So2Sat patches",
    },
    "seamless": {
        "subdir": "EmbeddedSeamless", "year": True, "kind": "npy",
        "prefix": "patch_", "suffix": ".npy", "embedding_name": "seamless",
        "note": "uint16 VQ indices stored as float32; 13 -> 72 channels at train time",
    },
    "sentinel1": {
        "subdir": "sentinel1", "year": False, "kind": "tif",
        "prefix": "sen1_patch_", "suffix": ".tif", "embedding_name": "sentinel1",
        "note": "raw So2Sat S1, 8 bands float64",
    },
    "sentinel2": {
        "subdir": "sentinel2", "year": False, "kind": "tif",
        "prefix": "sen2_patch_", "suffix": ".tif", "embedding_name": "sentinel2",
        "note": "raw So2Sat S2, 10 bands float64",
    },
}

DEFAULT_FAMILIES = [
    "alpha_earth_coop", "tesserav1.1_global", "seamless", "sentinel1", "sentinel2",
]


# ── Sampling (shared with verify_alphaearth_dequant.py) ───────────────────────

def build_id_index(so2sat_dir: Path, family: str, split: str, year: str) -> dict[str, Path]:
    """Return ``{patch_id: path}`` for one family in one original So2Sat split."""
    spec = FAMILIES[family]
    d = so2sat_dir / split / spec["subdir"]
    if spec["year"]:
        d = d / year
    if not d.exists():
        raise SystemExit(f"Family '{family}': directory not found: {d}")
    n = len(spec["prefix"])
    index = {p.stem[n:]: p for p in d.glob(f"{spec['prefix']}*{spec['suffix']}")}
    if not index:
        raise SystemExit(f"Family '{family}': no {spec['prefix']}*{spec['suffix']} under {d}")
    return index


def sample_paired_ids(
    so2sat_dir: Path,
    families: list[str],
    split: str,
    year: str,
    n_sample: int,
    seed: int,
) -> tuple[list[str], dict[str, dict[str, Path]]]:
    """Sample ``n_sample`` patch_ids present in EVERY requested family.

    Returns ``(ids, {family: {patch_id: path}})``. Pairing keeps the per-family
    statistics comparable patch-for-patch instead of over different subsets of
    cities.
    """
    indexes = {f: build_id_index(so2sat_dir, f, split, year) for f in families}
    for f, ix in indexes.items():
        logger.info(f"  {f:20s} {len(ix):>7d} patches")

    common = set.intersection(*(set(ix) for ix in indexes.values()))
    logger.info(
        f"Paired id intersection: {len(common)} "
        f"(largest family has {max(len(ix) for ix in indexes.values())})"
    )
    if not common:
        raise SystemExit("No patch_id is present in all requested families.")

    ids = sorted(common)
    rng = np.random.default_rng(seed)
    if len(ids) > n_sample:
        ids = [ids[i] for i in rng.choice(len(ids), n_sample, replace=False)]
    else:
        logger.warning(f"Only {len(ids)} paired patches available (< {n_sample})")
    return sorted(ids), indexes


# ── Training-path replication ────────────────────────────────────────────────

def load_raw(path: Path, kind: str) -> np.ndarray:
    """Load one patch as ``(C, H, W)`` float32, before any cleaning."""
    if kind == "npy":
        return np.load(path).astype(np.float32)
    import rasterio
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def to_training_tensor(
    raw: np.ndarray, dequantize_fn, patch_size: int
) -> np.ndarray:
    """nan_to_num -> dequantize -> bilinear resize, mirroring PatchDataset."""
    arr = np.nan_to_num(raw, nan=0.0)
    if dequantize_fn is not None:
        arr = dequantize_fn(arr)
    img = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if img.shape[-2:] != (patch_size, patch_size):
        img = F.interpolate(
            img.unsqueeze(0), size=(patch_size, patch_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return img.numpy()


# ── Accumulators ─────────────────────────────────────────────────────────────

class ChannelAccumulator:
    """Exact streaming per-channel moments + a pixel reservoir for quantiles.

    Raw moments are summed in float64 (values are O(1), so this is stable);
    everything quantile-shaped is computed from a fixed-size random subsample of
    pixels, so peak memory stays at a few hundred MB rather than the ~2.6 GB a
    full 5000x128x32x32 tensor would need.
    """

    def __init__(self, n_channels: int, keep_per_patch: int, seed: int) -> None:
        self.C = n_channels
        self.keep = keep_per_patch
        self.rng = np.random.default_rng(seed)
        self.n_values = 0                              # per channel
        self.s1 = np.zeros(n_channels, dtype=np.float64)
        self.s2 = np.zeros(n_channels, dtype=np.float64)
        self.s3 = np.zeros(n_channels, dtype=np.float64)
        self.n_pixels = 0
        self.n_allzero_pixels = 0
        self._reservoir: list[np.ndarray] = []

    def update(self, img: np.ndarray) -> None:
        flat = img.reshape(self.C, -1)                  # (C, HW) float32
        x = flat.astype(np.float64)
        self.s1 += x.sum(axis=1)
        self.s2 += (x * x).sum(axis=1)
        self.s3 += (x * x * x).sum(axis=1)
        self.n_values += flat.shape[1]

        self.n_pixels += flat.shape[1]
        self.n_allzero_pixels += int((flat == 0).all(axis=0).sum())

        k = min(self.keep, flat.shape[1])
        idx = self.rng.choice(flat.shape[1], k, replace=False)
        self._reservoir.append(flat[:, idx].copy())

    def finalize(self) -> dict:
        n = self.n_values
        mean = self.s1 / n
        var = np.maximum(self.s2 / n - mean ** 2, 0.0)
        std = np.sqrt(var)
        # third central moment from raw moments
        m3 = self.s3 / n - 3.0 * mean * (self.s2 / n) + 2.0 * mean ** 3
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.where(std > 0, m3 / np.maximum(std, 1e-30) ** 3, 0.0)

        pix = np.concatenate(self._reservoir, axis=1)   # (C, N)
        p001 = np.percentile(pix, 0.1, axis=1)
        p999 = np.percentile(pix, 99.9, axis=1)
        norms = np.sqrt((pix.astype(np.float64) ** 2).sum(axis=0))

        return {
            "n_channels": int(self.C),
            "n_values_per_channel": int(n),
            "n_reservoir_pixels": int(pix.shape[1]),
            "per_channel": {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "skewness": skew.tolist(),
                "p0.1": p001.tolist(),
                "p99.9": p999.tolist(),
            },
            "std_summary": {
                "median": float(np.median(std)),
                "min": float(std.min()),
                "max": float(std.max()),
            },
            "mean_summary": {
                "median": float(np.median(mean)),
                "min": float(mean.min()),
                "max": float(mean.max()),
            },
            "abs_skew_summary": {
                "median": float(np.median(np.abs(skew))),
                "max": float(np.abs(skew).max()),
            },
            "l2_norm": {
                "mean": float(norms.mean()),
                "std": float(norms.std()),
                "p1": float(np.percentile(norms, 1)),
                "p50": float(np.percentile(norms, 50)),
                "p99": float(np.percentile(norms, 99)),
            },
            "frac_allzero_pixels": float(self.n_allzero_pixels / max(self.n_pixels, 1)),
        }


# ── Per-family audit ─────────────────────────────────────────────────────────

def audit_family(
    family: str,
    ids: list[str],
    index: dict[str, Path],
    patch_size: int,
    keep_per_patch: int,
    seed: int,
) -> dict:
    spec = FAMILIES[family]
    dequantize_fn, ch_override = resolve_dequantize(spec["embedding_name"])

    acc: ChannelAccumulator | None = None
    n_raw_values = 0
    n_raw_zero = 0
    n_raw_nan = 0
    raw_shape = None
    train_shape = None

    for pid in tqdm(ids, desc=family, unit="patch"):
        raw = load_raw(index[pid], spec["kind"])
        # zero / NaN fractions are measured on the raw array, before nan_to_num
        n_raw_values += raw.size
        n_raw_zero += int((raw == 0).sum())
        n_raw_nan += int(np.isnan(raw).sum())

        img = to_training_tensor(raw, dequantize_fn, patch_size)
        if acc is None:
            raw_shape = list(raw.shape)
            train_shape = list(img.shape)
            acc = ChannelAccumulator(img.shape[0], keep_per_patch, seed)
        acc.update(img)

    assert acc is not None
    out = acc.finalize()
    out.update({
        "family": family,
        "embedding_name": spec["embedding_name"],
        "note": spec["note"],
        "dequantize_applied": dequantize_fn is not None,
        "dequantize_channel_override": ch_override,
        "raw_shape": raw_shape,
        "train_shape": train_shape,
        "n_patches": len(ids),
        "raw_frac_zero": float(n_raw_zero / max(n_raw_values, 1)),
        "raw_frac_nan": float(n_raw_nan / max(n_raw_values, 1)),
        "effective_noise_ratio": float(
            AUGMENT_NOISE_SIGMA / out["std_summary"]["median"]
        ),
    })
    return out


# ── Reporting ────────────────────────────────────────────────────────────────

def markdown_report(results: dict[str, dict], meta: dict) -> str:
    lines = [
        "### Task 0.1 — Embedding scale audit",
        "",
        f"Cultural-split **{meta['split']}** set, {meta['n_sample']} patches "
        f"(paired across families, seed {meta['seed']}), measured after the exact "
        f"training-path transform (nan_to_num -> dequantize -> bilinear resize to "
        f"{meta['patch_size']}x{meta['patch_size']}).",
        "",
        "| family | C | per-channel std (median) | std min | std max | "
        "L2 norm p1 / p50 / p99 | frac zero (raw) | frac NaN (raw) | "
        "all-zero px | `0.05 / median_std` |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        s = r["std_summary"]
        n = r["l2_norm"]
        lines.append(
            f"| `{name}` | {r['n_channels']} | **{s['median']:.4f}** | "
            f"{s['min']:.4f} | {s['max']:.4f} | "
            f"{n['p1']:.3f} / {n['p50']:.3f} / {n['p99']:.3f} | "
            f"{r['raw_frac_zero']:.2%} | {r['raw_frac_nan']:.2%} | "
            f"{r['frac_allzero_pixels']:.2%} | **{r['effective_noise_ratio']:.3f}** |"
        )

    lines += [
        "",
        "Tail / asymmetry summary (per-channel, pooled across channels):",
        "",
        "| family | mean of per-channel means | median \\|skew\\| | max \\|skew\\| | "
        "min p0.1 | max p99.9 |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        pc = r["per_channel"]
        lines.append(
            f"| `{name}` | {np.mean(pc['mean']):+.4f} | "
            f"{r['abs_skew_summary']['median']:.2f} | "
            f"{r['abs_skew_summary']['max']:.2f} | "
            f"{min(pc['p0.1']):.4g} | {max(pc['p99.9']):.4g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 0.1 — per-family embedding scale audit (measure only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--split", default="training",
                   choices=["training", "validation", "testing"],
                   help="Original So2Sat split dir; 'training' IS the cultural-split "
                        "training set.")
    p.add_argument("--year", default="2017")
    p.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES,
                   choices=list(FAMILIES))
    p.add_argument("--n-sample", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--keep-per-patch", type=int, default=64,
                   help="Pixels kept per patch for quantile/L2-norm statistics.")
    p.add_argument("--output-json", type=Path,
                   default=Path("diagnostics/embedding_stats.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/embedding_stats.md"))
    args = p.parse_args()

    logger.info(f"So2Sat dir: {args.so2sat_dir}  split={args.split}  year={args.year}")
    ids, indexes = sample_paired_ids(
        args.so2sat_dir, args.families, args.split, args.year,
        args.n_sample, args.seed,
    )
    logger.info(f"Sampled {len(ids)} paired patch_ids")

    results = {
        f: audit_family(f, ids, indexes[f], args.patch_size,
                        args.keep_per_patch, args.seed)
        for f in args.families
    }

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "so2sat_dir": str(args.so2sat_dir),
        "split": args.split,
        "year": args.year,
        "n_sample": len(ids),
        "seed": args.seed,
        "patch_size": args.patch_size,
        "keep_per_patch": args.keep_per_patch,
        "paired_families": args.families,
        "augment_noise_sigma": AUGMENT_NOISE_SIGMA,
    }

    # Merge into any existing JSON so separate invocations (e.g. the 51-city
    # tessera subset) accumulate into one file.
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": [], "families": {}}
    if args.output_json.exists():
        payload = json.loads(args.output_json.read_text())
        payload.setdefault("meta", [])
        payload.setdefault("families", {})
    payload["meta"].append(meta)
    for name, r in results.items():
        r["meta_index"] = len(payload["meta"]) - 1
        payload["families"][name] = r
    args.output_json.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {args.output_json}")

    md = markdown_report(results, meta)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md + "\n")
    logger.info(f"Wrote {args.output_md}")
    print()
    print(md)


if __name__ == "__main__":
    main()
