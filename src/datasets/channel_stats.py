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

from datasets.registry import provenance
from utils.constants import DATA_DIR

STATS_CACHE_DIR = DATA_DIR / "cache" / "channel_stats"


def _cache_key(items: list, n_sample: int, seed: int, patch_size: int,
               nodata_mode: str, prov: str = "") -> str:
    """Short digest of the inputs that change the statistics."""
    h = hashlib.sha256()
    h.update(f"{len(items)}|{n_sample}|{seed}|{patch_size}|{nodata_mode}|{prov}".encode())
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


def grid_stats_cache_path(
    run_root: Path, output_names, year: str, embedding_name: str,
    items: list, n_sample: int, seed: int,
) -> Path:
    """Cache path for one segmentation (embedding, year, tile-set) combination.

    Keyed on the things that change the statistics -- the embedding, the year,
    the sample size and seed, and the identity of the train tile set -- so a run
    on a different city list, or one narrowed by ``--require-embeddings``, gets
    its own entry instead of silently reusing another's normalizer.
    """
    h = hashlib.sha256()
    names = "+".join(output_names) if not isinstance(output_names, str) else output_names
    h.update(f"{names}|{year}|{embedding_name}|{len(items)}|{n_sample}|{seed}".encode())
    for it in items[:64]:
        p0 = it[0][0] if isinstance(it[0], tuple) else it[0]
        h.update(str(p0).encode())
    return Path(run_root) / "_channel_stats" / f"grid_{names}_{year}_{h.hexdigest()[:16]}.npz"


def compute_grid_channel_stats(
    items: list,
    dequantize_fn=None,
    nodata_predicate=None,
    *,
    n_sample: int = 400,
    seed: int = 0,
    cache_path: Path | None = None,
    recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel ``(mean, std)`` over a sample of segmentation grid tiles.

    The segmentation analogue of :func:`compute_channel_stats`. It exists
    separately rather than as a flag because the two consume different
    datasets, and the whole point of both is to describe *the exact tensor the
    model consumes* -- so the statistics must be produced by the same class
    that produces the training batches.

    That principle is what makes this correct across embedding families without
    any per-family branching here:

    * dequantization is applied by ``GridSegDataset`` exactly as in training,
      so coop and seamless statistics describe dequantized values, not int8
      codes;
    * ``seamless`` expands 13 stored bands to 72 channels during dequantize, so
      the returned arrays are 72 long, matching the model's input;
    * for fused items dequantization applies to source 0 only, again matching
      training, and the returned arrays span the full concatenated stack;
    * nodata pixels are excluded via the family predicate, which runs on the
      RAW array before dequantization -- coop's all-channels ``-128`` sentinel
      becomes a plausible-looking vector of L2 norm 8.06 afterwards, so
      measuring it post-dequantize would quietly bias every channel.

    ``items`` must be the TRAIN split only; val or test items here would leak
    evaluation data into the normalizer.

    ``n_sample`` defaults to 400 tiles rather than the patch pipeline's 20000
    patches because a 128x128 tile carries 16384 pixels against a 32x32 patch's
    1024 -- 400 tiles is already ~6.5M pixel samples.

    Returns ``(mean, std)``, each ``(C,)`` float32, std floored at 1e-6.
    """
    if cache_path is not None and cache_path.exists() and not recompute:
        cached = np.load(cache_path)
        logger.info(f"Channel stats: loaded {cache_path}")
        return cached["mean"], cached["std"]

    from datasets.grid_tiles import GridSegDataset

    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(items), n_sample, replace=False)
           if len(items) > n_sample else np.arange(len(items)))
    sample = [items[int(i)] for i in idx]
    logger.info(
        f"Channel stats: {len(sample)} train tiles (of {len(items)})"
    )

    ds = GridSegDataset(
        sample, "gpkg", dequantize_fn=dequantize_fn,
        nodata_predicate=nodata_predicate, emit_valid=True,
    )

    n = s1 = s2 = None
    for i in range(len(ds)):
        item = ds[i]
        x = item["image"].numpy().astype(np.float64)          # (C, H, W)
        flat = x.reshape(x.shape[0], -1)
        v = item.get("valid")
        w = (np.ones(flat.shape[1], dtype=np.float64) if v is None
             else v.numpy().reshape(-1).astype(np.float64))
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
        f"Channel stats: {len(mean)} channels, median std={np.median(std):.4f} "
        f"(min {std.min():.4f}, max {std.max():.4f}), {n:,.0f} valid pixels"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, mean=mean, std=std)
        logger.info(f"Channel stats: cached to {cache_path}")
    return mean, std


def stats_cache_path(
    output_name: str, year: str, split_source: str, *,
    embedding_name: str | list[str] | tuple[str, ...],
    items: list | None = None, n_sample: int = 20000, seed: int = 0,
    patch_size: int = 32, nodata_mode: str = "mask",
) -> Path:
    """Cache path for one (embedding, year, split) combination.

    Keyed on the full provenance triple (Task 1.5.3, guard 1), not just
    ``output_name``: two extractions of the same product from different tile
    sources are different feature spaces — `tesserav1.1` and
    `tesserav1.1_global` have matched-channel correlation ≈ 0 — so sharing a
    normalizer between them would z-score one product by the other's statistics
    and produce a plausible wrong number. ``output_name`` is a free-text CLI
    label and cannot be relied on to differ.

    The digest additionally keeps runs that differ in sample size, seed, patch
    size or nodata mode from silently sharing a normalizer.
    """
    prov = provenance(embedding_name)
    triple = f"{prov['product']}-{prov['version']}-{prov['source']}"
    name = f"{triple}_{output_name}_{year}_{split_source}"
    if items is not None:
        name += f"_{_cache_key(items, n_sample, seed, patch_size, nodata_mode, triple)}"
    return STATS_CACHE_DIR / f"{name}.npz"
