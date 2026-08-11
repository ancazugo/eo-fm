"""Per-channel input statistics for `--normalize channel` (Task 1.2).

The CNN training path historically fed raw embedding values straight into the
network, and `training.augment.augment_images` then added Gaussian noise at a
FIXED ABSOLUTE sigma. Phase 0 measured what that means in practice: the noise
lands at 4.4% of a channel std for Tessera but 47.4% for AlphaEarth, a 14x
spread across families, so the cross-family comparison was confounded by scale
alone. Normalizing inputs per channel removes that confound and makes
`--noise-sigma` mean the same thing everywhere.

Two properties matter for the numbers to be trustworthy:

* **Train split only.** Statistics are estimated from the training patches and
  then applied unchanged to val/test, so no test-set information leaks in.
* **Valid pixels only.** Nodata sentinels are excluded via the Task 1.5
  predicates. This is not cosmetic: AlphaEarth's all-channel -128 sentinel is
  0.4% of pixels but contributes ~28% of the measured per-channel variance,
  inflating std by ~18%.

Statistics are accumulated on the **post-resize** tensor — exactly the array
the model consumes — so normalized inputs have unit variance as the model sees
them. (Bilinear resize smooths, so native-grid stds run ~15% higher; the two
are not interchangeable.)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from loguru import logger

from utils.constants import DATA_DIR

STATS_CACHE_DIR = DATA_DIR / "cache" / "channel_stats"


def _cache_key(items: list, n_sample: int, seed: int, patch_size: int,
               nodata_mode: str) -> str:
    """Short digest of the inputs that change the statistics."""
    h = hashlib.sha256()
    h.update(f"{len(items)}|{n_sample}|{seed}|{patch_size}|{nodata_mode}".encode())
    # A few paths pin the identity of the item list without hashing all 350k.
    for it in items[:64]:
        h.update(str(it.path).encode())
    return h.hexdigest()[:16]


def compute_channel_stats(
    items: list,
    dequantize_fn=None,
    nodata_predicate=None,
    *,
    patch_size: int = 32,
    n_sample: int = 20000,
    seed: int = 0,
    nodata_mode: str = "mask",
    cache_path: Path | None = None,
    recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel ``(mean, std)`` over a random sample of *items*.

    Args:
        items: Item tuples/PatchItems for the TRAIN split only. Passing val or
            test items here would leak evaluation data into the normalizer.
        dequantize_fn: Family dequantization, or a sequence for fused sources.
        nodata_predicate: From ``datasets.registry.get_nodata_predicate``; a
            sequence for fused sources. Invalid pixels are excluded.
        patch_size: Resize target — statistics describe the post-resize tensor.
        n_sample: Patches to draw. 20000 rather than 5000 because the sentinel
            fraction itself varies (0.41-0.68% across Phase 0 samples).
        seed: Sampling seed.
        nodata_mode: "mask" excludes invalid pixels; "zero" reproduces the
            unmasked estimate for ablations.
        cache_path: ``.npz`` to read/write. ``None`` disables caching.
        recompute: Ignore an existing cache and overwrite it.

    Returns:
        ``(mean, std)``, each ``(C,)`` float32. std is floored at 1e-6 so a
        constant channel cannot produce a division by zero downstream.

    A single streaming pass in float64 — never materializes the sample.
    """
    if cache_path is not None and cache_path.exists() and not recompute:
        cached = np.load(cache_path)
        logger.info(f"Channel stats: loaded {cache_path}")
        return cached["mean"], cached["std"]

    # Reuse the dataset itself so the statistics describe the exact tensor the
    # model consumes: same load, same nodata fill, same dequantize, same resize.
    from datasets.so2sat import PatchDataset

    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(items), n_sample, replace=False)
           if len(items) > n_sample else np.arange(len(items)))
    sample = [items[int(i)] for i in idx]
    logger.info(
        f"Channel stats: {len(sample)} train patches "
        f"(of {len(items)}), nodata_mode={nodata_mode}, patch_size={patch_size}"
    )

    ds = PatchDataset(
        sample, patch_size, dequantize_fn=dequantize_fn,
        nodata_mode=nodata_mode, nodata_predicate=nodata_predicate,
    )

    n = s1 = s2 = None
    for i in range(len(ds)):
        item = ds[i]
        x = item["image"].numpy().astype(np.float64)          # (C, H, W)
        flat = x.reshape(x.shape[0], -1)
        v = item.get("valid")
        if v is None:
            w = np.ones(flat.shape[1], dtype=np.float64)
        else:
            w = v.numpy().reshape(-1).astype(np.float64)
        if s1 is None:
            s1 = np.zeros(flat.shape[0])
            s2 = np.zeros(flat.shape[0])
            n = 0.0
        s1 += (flat * w).sum(axis=1)
        s2 += ((flat ** 2) * w).sum(axis=1)
        n += w.sum()

    if n is None or n <= 0:
        raise ValueError("No valid pixels found while computing channel stats")

    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
    mean = mean.astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)

    logger.info(
        f"Channel stats: median std={np.median(std):.4f} "
        f"(min {std.min():.4f}, max {std.max():.4f}), {n:,.0f} valid pixel-samples"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, mean=mean, std=std)
        logger.info(f"Channel stats: cached to {cache_path}")
    return mean, std


def stats_cache_path(
    output_name: str, year: str, split_source: str, *,
    items: list | None = None, n_sample: int = 20000, seed: int = 0,
    patch_size: int = 32, nodata_mode: str = "mask",
) -> Path:
    """Cache path for one (embedding, year, split) combination.

    The digest keeps runs that differ in sample size, seed, patch size or
    nodata mode from silently sharing a normalizer.
    """
    name = f"{output_name}_{year}_{split_source}"
    if items is not None:
        name += f"_{_cache_key(items, n_sample, seed, patch_size, nodata_mode)}"
    return STATS_CACHE_DIR / f"{name}.npz"
