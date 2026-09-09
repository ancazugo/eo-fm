"""Block-wise spatial pooling of patch embeddings, cached one block at a time.

A So2Sat patch is 320 m of ground — roughly 33x33 px at 10 m — collapsed to a
single vector before any pooled-feature model sees it. Which collapse is right is
an empirical question (see ``src/pooling_bakeoff.py``), so this module caches
**blocks** rather than finished feature vectors:

    mean std center q10 q50 q90 ring_in ring_out

A *recipe* names an ordered tuple of blocks; ``compose_features`` concatenates
them. Caching blocks rather than recipes means a new recipe costs a column
slice, not another pass over 226 GB of npy.

Two properties every block here holds to, because the data does not cooperate
otherwise:

* **Shape-safety.** Only ~18% of extracted patches are square — 33x34, 34x33,
  35x33 and 33x35 are all common — so ``H // 2`` picks half a pixel off centre
  and the bias flips direction between 33x34 and 34x33. ``center`` therefore
  averages the central 2x2 whenever a side is even, and the ring radii are
  fractions of the patch, never pixel counts.
* **Nodata awareness.** The predicate runs on the *raw* array, before
  ``nan_to_num`` and before dequantization, per the contract documented in
  ``datasets.registry.get_nodata_predicate``. For ``alpha_earth_coop`` the
  all-channel -128 sentinel dequantizes to a vector of L2 norm 8.06 against a
  normal pixel's 1.0, so averaging it in biases every patch that touches a
  coastline or a city edge. ``center`` is worse still: one sentinel pixel at the
  centre *is* the whole feature.

Masked statistics follow ``models.pooling.pool_mean_std``: a patch with no valid
pixel at all falls back to the unmasked value rather than emitting NaN.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

# Block vocabulary. Each block reduces (C, H, W) -> (C,).
POOL_BLOCKS: tuple[str, ...] = (
    "mean", "std", "center", "q10", "q50", "q90", "ring_in", "ring_out",
)

# Named feature sets. Order is significant: it fixes the column layout, and
# `feature_dim` is `in_channels * len(recipe)`.
POOLING_RECIPES: dict[str, tuple[str, ...]] = {
    # The two historical modes. Their layouts must never change: existing caches
    # and every linear-probe checkpoint depend on them.
    "gap": ("mean",),
    "mean_std": ("mean", "std"),
    # Single focal pixel — the "one pixel" representation, for reference.
    "center": ("center",),
    # Distribution shape. Permutation-invariant: shuffling the patch's pixels
    # leaves this bit-identical, so it measures heterogeneity, not layout.
    "quantile": ("q10", "q50", "q90"),
    # Centre-vs-surround. The cheapest block here that is NOT
    # permutation-invariant, so it is the one that actually tests whether the
    # within-patch arrangement carries signal.
    "ring": ("mean", "ring_in", "ring_out"),
    "rich": POOL_BLOCKS,
}

# Fraction of the patch half-extent counted as "inner" for the ring blocks.
# 0.5 puts about a quarter of the area inside the ring and the rest outside.
RING_INNER_FRAC = 0.5

# How far `center` may widen its window looking for a valid pixel before giving
# up and using the masked patch mean.
CENTER_MAX_PAD = 3


def recipe_blocks(recipe: str) -> tuple[str, ...]:
    """Blocks for a named recipe, in concatenation order."""
    try:
        return POOLING_RECIPES[recipe]
    except KeyError:
        raise ValueError(
            f"unknown pooling recipe {recipe!r}; choose from {sorted(POOLING_RECIPES)}"
        ) from None


def feature_dim(recipe: str, in_channels: int) -> int:
    """Width of the composed feature vector for a recipe."""
    return in_channels * len(recipe_blocks(recipe))


# ── Blocks ────────────────────────────────────────────────────────────────────

def _masked_mean(arr: np.ndarray, m: np.ndarray | None) -> np.ndarray:
    """Channel mean over valid pixels, falling back when nothing is valid."""
    if m is None:
        return arr.mean(axis=(1, 2))
    n = m.sum()
    if n <= 0:
        return arr.mean(axis=(1, 2))
    return (arr * m).sum(axis=(1, 2)) / n


def _center_slice(n: int) -> slice:
    """Central 1 px for an odd extent, central 2 px for an even one.

    An even side has no single central pixel; ``n // 2`` sits half a pixel past
    centre, and which way it leans flips between 33x34 and 34x33 patches. Taking
    both central pixels puts the sampled point at the true geometric centre for
    every shape.
    """
    return slice(n // 2, n // 2 + 1) if n % 2 else slice(n // 2 - 1, n // 2 + 1)


def _ring_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(inner, outer) boolean masks, defined on *relative* radius.

    Chebyshev distance from the patch centre normalised to the half-extent, so
    the split is the same fraction of the patch on a 12x12 seamless patch as on
    a 35x33 tessera one.
    """
    yy = np.abs(np.arange(h) - (h - 1) / 2.0) / max((h - 1) / 2.0, 1e-9)
    xx = np.abs(np.arange(w) - (w - 1) / 2.0) / max((w - 1) / 2.0, 1e-9)
    r = np.maximum(yy[:, None], xx[None, :])
    inner = r <= RING_INNER_FRAC
    return inner, ~inner


def pool_blocks(
    arr: np.ndarray,
    blocks: Sequence[str] = POOL_BLOCKS,
    valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Reduce ``(C, H, W)`` to one ``(C,)`` vector per requested block.

    Args:
        arr: ``(C, H, W)`` float array, already dequantized.
        blocks: Which blocks to compute. Unknown names raise.
        valid: ``(H, W)`` bool mask, True = usable. ``None`` pools everything.

    Returns:
        ``{block_name: (C,) float32}``.
    """
    unknown = set(blocks) - set(POOL_BLOCKS)
    if unknown:
        raise ValueError(f"unknown pool blocks {sorted(unknown)}; known: {list(POOL_BLOCKS)}")
    if arr.ndim != 3:
        raise ValueError(f"expected (C, H, W), got shape {arr.shape}")

    c, h, w = arr.shape
    want = set(blocks)
    # A mask that excludes everything is no more informative than no mask, and
    # the fallback path is what the torch pooling does too.
    m = None
    if valid is not None:
        m = valid.astype(arr.dtype, copy=False)[None, :, :]
        if m.sum() <= 0:
            m = None

    out: dict[str, np.ndarray] = {}

    if {"mean", "std"} & want:
        mean = _masked_mean(arr, m)
        if "mean" in want:
            out["mean"] = mean
        if "std" in want:
            if m is None:
                out["std"] = arr.std(axis=(1, 2))
            else:
                n = m.sum()
                var = (((arr - mean[:, None, None]) ** 2) * m).sum(axis=(1, 2)) / n
                out["std"] = np.sqrt(np.maximum(var, 0.0))

    if "center" in want:
        cy, cx = _center_slice(h), _center_slice(w)
        if m is None:
            out["center"] = arr[:, cy, cx].mean(axis=(1, 2))
        else:
            # If the centre itself is nodata, grow the window rather than
            # returning the sentinel — for `center` a single invalid pixel would
            # otherwise BE the entire feature. Falls back to the masked patch
            # mean if even a widened neighbourhood is empty.
            got = None
            for pad in range(CENTER_MAX_PAD + 1):
                ys = slice(max(cy.start - pad, 0), min(cy.stop + pad, h))
                xs = slice(max(cx.start - pad, 0), min(cx.stop + pad, w))
                sm = m[:, ys, xs]
                if sm.sum() > 0:
                    got = _masked_mean(arr[:, ys, xs], sm)
                    break
            out["center"] = got if got is not None else _masked_mean(arr, m)

    quant = [q for q in ("q10", "q50", "q90") if q in want]
    if quant:
        pcts = [int(q[1:]) for q in quant]
        if m is None:
            vals = np.percentile(arr.reshape(c, -1), pcts, axis=1)
        else:
            keep = m[0].astype(bool).reshape(-1)
            flat = arr.reshape(c, -1)[:, keep]
            vals = np.percentile(flat, pcts, axis=1)
        for q, v in zip(quant, vals):
            out[q] = v

    if {"ring_in", "ring_out"} & want:
        inner, outer = _ring_masks(h, w)
        for name, ring in (("ring_in", inner), ("ring_out", outer)):
            if name not in want:
                continue
            rm = ring[None, :, :].astype(arr.dtype)
            if m is not None:
                rm = rm * m
            # A degenerate patch (e.g. 1 px wide) can leave a ring empty.
            out[name] = _masked_mean(arr, rm if rm.sum() > 0 else m)

    return {k: np.asarray(out[k], dtype=np.float32) for k in blocks}


# ── Extraction ────────────────────────────────────────────────────────────────

def _block_path(cache_dir: Path, cache_key: str, block: str, split: str) -> Path:
    return cache_dir / f"{cache_key}_{block}_{split}.npy"


def _labels_path(cache_dir: Path, cache_key: str, split: str) -> Path:
    return cache_dir / f"{cache_key}_{split}_labels.npy"


def _shapes_path(cache_dir: Path, cache_key: str, split: str) -> Path:
    return cache_dir / f"{cache_key}_{split}_shapes.parquet"


def masked_cache_key(cache_key: str, masked: bool) -> str:
    """Cache key suffix so masked and unmasked blocks never collide."""
    return f"{cache_key}_masked" if masked else cache_key


def _load_one(
    path, dequantize_fn, nodata_predicate, blocks: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], tuple[int, int, int]]:
    """Load one patch and reduce it. Also returns (h, w, n_invalid)."""
    raw = np.load(path)
    # The predicate reads the sentinel in the units it is stored in, so it has
    # to run before nan_to_num and before dequantization.
    valid = None
    n_invalid = 0
    if nodata_predicate is not None:
        invalid = nodata_predicate(raw)
        n_invalid = int(invalid.sum())
        valid = ~invalid
    arr = np.nan_to_num(raw.astype(np.float32), nan=0.0)
    if dequantize_fn is not None:
        arr = dequantize_fn(arr)
    h, w = arr.shape[1], arr.shape[2]
    return pool_blocks(arr, blocks, valid), (h, w, n_invalid)


# Worker state, populated in the parent and inherited through fork. Both the
# dequantize function and the nodata predicate are closures, so neither can be
# pickled as a map() argument — the fork context is what makes them reachable.
_WORKER_STATE: dict = {}


def _worker(paths):
    dequantize_fn = _WORKER_STATE["dequantize_fn"]
    nodata_predicate = _WORKER_STATE["nodata_predicate"]
    blocks = _WORKER_STATE["blocks"]
    cap_blas_threads(1)
    feats = {b: [] for b in blocks}
    shapes = []
    for p in paths:
        vec, hw = _load_one(p, dequantize_fn, nodata_predicate, blocks)
        for b in blocks:
            feats[b].append(vec[b])
        shapes.append(hw)
    return {b: np.stack(v) for b, v in feats.items()}, shapes


def _chunks(seq: list, n: int) -> list[list]:
    if n <= 1:
        return [seq]
    size = max(1, (len(seq) + n - 1) // n)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def extract_blocks_and_cache(
    items: list,
    blocks: Sequence[str],
    dequantize_fn,
    cache_dir: Path,
    cache_key: str,
    *,
    nodata_predicate=None,
    workers: int = 1,
    no_cache: bool = False,
    splits: Sequence[str] = ("train", "val", "test"),
    write_shapes: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Pool every item into each requested block, caching one array per block.

    Row order matches ``knn_baseline.extract_and_cache``: splits concatenated in
    ``("train", "val", "test")`` order, original item order preserved within
    each split. ``embedding_projection.build_metadata`` relies on exactly this.

    Args:
        items: ``PatchItem``s (``.path``, ``.label``, ``.split``).
        blocks: Block names to compute and cache.
        dequantize_fn: Family dequantization, or None.
        cache_dir: Where the ``.npy`` blocks live.
        cache_key: Already suffixed for masking by the caller, if applicable.
        nodata_predicate: From ``datasets.registry.get_nodata_predicate``.
        workers: Processes. Pooling is CPU-bound (measured 261 patches/s
            single-threaded against 1,835 patches/s for the bare reads), so this
            is where the wall-clock actually goes.
        no_cache: Recompute and overwrite.
        write_shapes: Also record per-patch (h, w, n_invalid, invalid_frac),
            which is the schema ``diagnostics/nodata_population.py`` writes and
            which tesserav2 has no entry in.

    Returns:
        ``(blocks_by_split, labels_by_split)`` where ``blocks_by_split[split][block]``
        is ``(N, C)``.
    """
    blocks = tuple(blocks)
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_items: dict[str, list] = {s: [] for s in splits}
    for it in items:
        if it.split in split_items:
            split_items[it.split].append((it.path, it.label))

    out: dict[str, dict[str, np.ndarray]] = {}
    labels_out: dict[str, np.ndarray] = {}

    for s in splits:
        rows = split_items[s]
        lp = _labels_path(cache_dir, cache_key, s)
        cached = (
            not no_cache
            and lp.exists()
            and all(_block_path(cache_dir, cache_key, b, s).exists() for b in blocks)
        )
        if cached:
            out[s] = {b: np.load(_block_path(cache_dir, cache_key, b, s)) for b in blocks}
            labels_out[s] = np.load(lp)
            logger.info(f"  {s}: cached {labels_out[s].shape[0]} patches")
            continue

        if not rows:
            out[s] = {b: np.empty((0, 1), dtype=np.float32) for b in blocks}
            labels_out[s] = np.empty((0,), dtype=np.int64)
            for b in blocks:
                np.save(_block_path(cache_dir, cache_key, b, s), out[s][b])
            np.save(lp, labels_out[s])
            continue

        paths = [r[0] for r in rows]
        labels = np.array([r[1] for r in rows], dtype=np.int64)
        logger.info(
            f"  {s}: pooling {len(paths)} patches -> {list(blocks)} "
            f"(workers={workers}, masked={nodata_predicate is not None})"
        )

        if workers > 1:
            parts = _chunks(paths, workers * 4)
            _WORKER_STATE.update(
                dequantize_fn=dequantize_fn,
                nodata_predicate=nodata_predicate,
                blocks=blocks,
            )
            feats_parts, shape_parts = [], []
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                for f, sh in tqdm(ex.map(_worker, parts), total=len(parts),
                                  desc=f"  {s}", leave=False):
                    feats_parts.append(f)
                    shape_parts.extend(sh)
            feats = {b: np.concatenate([p[b] for p in feats_parts], axis=0) for b in blocks}
            shapes = shape_parts
        else:
            acc = {b: [] for b in blocks}
            shapes = []
            for p in tqdm(paths, desc=f"  {s}", leave=False):
                vec, hw = _load_one(p, dequantize_fn, nodata_predicate, blocks)
                for b in blocks:
                    acc[b].append(vec[b])
                shapes.append(hw)
            feats = {b: np.stack(v).astype(np.float32) for b, v in acc.items()}

        for b in blocks:
            np.save(_block_path(cache_dir, cache_key, b, s), feats[b].astype(np.float32))
        np.save(lp, labels)
        out[s] = {b: feats[b].astype(np.float32) for b in blocks}
        labels_out[s] = labels

        if write_shapes and shapes:
            _write_shapes(cache_dir, cache_key, s, rows, shapes)

    return out, labels_out


def _write_shapes(cache_dir: Path, cache_key: str, split: str, rows, shapes) -> None:
    """Per-patch geometry + nodata census, in the nodata_population schema."""
    import pandas as pd

    from datasets.so2sat import patch_key

    keys = [patch_key(Path(p)) for p, _ in rows]
    h, w, n_inv = zip(*shapes)
    df = pd.DataFrame({
        "dataset": [k[0] for k in keys],
        "patch_id": [k[1] for k in keys],
        "h": np.asarray(h, dtype=np.int32),
        "w": np.asarray(w, dtype=np.int32),
        "n_invalid": np.asarray(n_inv, dtype=np.int64),
    })
    df["invalid_frac"] = df["n_invalid"] / (df["h"] * df["w"])
    df.to_parquet(_shapes_path(cache_dir, cache_key, split), index=False)


def compose_features(
    cache_dir: Path, cache_key: str, recipe: str, split: str,
) -> np.ndarray:
    """Concatenate a recipe's cached blocks into ``(N, C * len(recipe))``."""
    blocks = recipe_blocks(recipe)
    arrs = []
    for b in blocks:
        p = _block_path(cache_dir, cache_key, b, split)
        if not p.exists():
            raise FileNotFoundError(
                f"block {b!r} not cached for split {split!r} at {p}. "
                f"Run extract_blocks_and_cache with blocks including {b!r} first."
            )
        arrs.append(np.load(p))
    return np.concatenate(arrs, axis=1) if len(arrs) > 1 else arrs[0]


def cap_blas_threads(n: int = 1) -> None:
    """Pin BLAS to ``n`` threads. Call before forking a process pool.

    ``embedding_projection`` caps the parent at 16 because OpenBLAS on this
    256-core host aborts with "tried to allocate too many memory regions". A
    pool of workers each spawning their own 16-thread pool reproduces exactly
    that, so workers get 1.
    """
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(n)
