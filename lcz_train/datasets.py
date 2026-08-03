"""T2 — datasets: A (dense pixel windows) and B (block samples).

Erosion is a *training-time* transform (the exported rasters are un-eroded):
each block's rasterized region is eroded inward by ``erosion_px`` using the
``block_id`` index raster, so boundary-mixed pixels leave the loss. Because
the index raster burns *all* blocks (unlabelled too, 0 = no block), erosion
also fires on labelled/unlabelled boundaries.

Both A and B honor the stability contract from ``lcz_labels.export.
to_training_pairs``: a stable block/pixel may pair with any requested
embedding year; an unstable one only with its own label year. Callers pass
in the already-year-resolved mosaic — these classes don't re-derive years.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import grey_dilation, grey_erosion
from torch.utils.data import Dataset

N_LCZ = 17


def erosion_valid_mask(
    block_idx: np.ndarray,
    bitmask: np.ndarray,
    conf: np.ndarray | None = None,
    *,
    min_conf: float = 0.0,
    erosion_px: int = 1,
) -> np.ndarray:
    """Boolean valid-mask for the dense loss.

    valid = (bitmask != 0) AND (conf >= min_conf) AND block-interior, where
    interior means every pixel within Chebyshev radius ``erosion_px`` belongs
    to the same block (grey erosion == grey dilation of the index raster).

    Args:
        block_idx: (H, W) uint32 block index raster (0 = no block).
        bitmask:   (H, W) uint32 LCZ set bitmask (0 = unlabelled).
        conf:      (H, W) confidence in [0, 1] (pass raster uint8 / 100).
        min_conf:  threshold on ``conf`` (ignored when ``conf`` is None).
        erosion_px: Chebyshev erosion radius; 0 disables erosion.
    """
    if block_idx.shape != bitmask.shape:
        raise ValueError(f"block_idx {block_idx.shape} vs bitmask {bitmask.shape}")
    valid = bitmask != 0
    if conf is not None and min_conf > 0.0:
        valid &= np.asarray(conf, dtype=np.float64) >= min_conf
    if erosion_px > 0:
        size = 2 * int(erosion_px) + 1
        interior = grey_erosion(block_idx, size=size) == grey_dilation(block_idx, size=size)
        valid &= interior
    return valid


# ── A (dense): class-balanced pixel sampling + windows ────────────────────────

def _popcount17(bitmask: np.ndarray) -> np.ndarray:
    bits = np.arange(N_LCZ, dtype=np.uint32)
    return (((bitmask.astype(np.uint32)[..., None] >> bits) & 1)).sum(axis=-1)


def class_balanced_weights(bitmask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel sampling weight: inverse-frequency over the HARD distribution.

    Hard pixels (a single bit set) weigh ``1 / count(class)``. Coarse pixels
    (>=2 bits) weigh the MEAN of their member classes' inverse frequencies —
    "sampled at the mean rate of their set" — so LCZ 6 in well-mapped suburbs
    doesn't drown the rarer classes. Invalid pixels get weight 0.
    """
    w = np.zeros(bitmask.shape, dtype=np.float64)
    if not valid.any():
        return w
    pop = _popcount17(bitmask)
    hard = valid & (pop == 1)
    class_counts = np.zeros(N_LCZ, dtype=np.int64)
    for c in range(N_LCZ):
        class_counts[c] = int((hard & ((bitmask >> c) & 1).astype(bool)).sum())
    inv_freq = np.divide(1.0, class_counts, out=np.zeros(N_LCZ), where=class_counts > 0)

    for c in range(N_LCZ):
        member = valid & ((bitmask >> c) & 1).astype(bool)
        w[member] += inv_freq[c]
    # coarse pixels: mean over their |S| members, not the sum
    coarse_mask = valid & (pop > 1)
    w[coarse_mask] /= np.maximum(pop[coarse_mask], 1)
    return w


class PixelWindowDataset(Dataset):
    """A: (embedding window, bitmask window, valid window) class-balanced samples.

    Windows may cross block boundaries freely (context is the point); only
    ``valid`` masks the loss. Sampling draws an anchor pixel per item with
    probability proportional to :func:`class_balanced_weights`, seeded so an
    epoch is reproducible; the window is centred on the anchor and clipped/
    padded at raster edges (zero-padding — those pixels are marked invalid).

    ``__getitem__``'s ``index`` is intentionally ignored: this is a streaming
    sampler over one seeded RNG, not an index-addressed map. Use with a
    ``DataLoader(shuffle=False, num_workers=0)`` (or one dataset instance per
    worker with a derived seed) — under ``shuffle=True`` the RNG draw order
    would depend on the sampler's index order, which defeats the point.
    """

    def __init__(
        self,
        mosaic: np.ndarray,
        bitmask: np.ndarray,
        conf: np.ndarray,
        block_idx: np.ndarray,
        *,
        window_px: int = 128,
        min_conf: float = 0.0,
        erosion_px: int = 1,
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ):
        if not (96 <= window_px <= 256):
            raise ValueError(f"window_px={window_px} outside the spec range 96-256")
        self.mosaic = mosaic
        self.bitmask = bitmask
        self.window_px = window_px
        self.valid = erosion_valid_mask(block_idx, bitmask, conf, min_conf=min_conf,
                                        erosion_px=erosion_px)
        weights = class_balanced_weights(bitmask, self.valid)
        total = weights.sum()
        if total <= 0:
            raise ValueError("no valid, class-balanceable pixels in this AOI/year")
        self._flat_weights = (weights / total).ravel()
        self._shape = bitmask.shape
        self._rng = np.random.default_rng(seed)
        self._n = samples_per_epoch or int(self.valid.sum())

    def __len__(self) -> int:
        return self._n

    def _sample_anchor(self) -> tuple[int, int]:
        idx = self._rng.choice(self._flat_weights.size, p=self._flat_weights)
        return np.unravel_index(idx, self._shape)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        r, c = self._sample_anchor()
        w = self.window_px
        h_img, w_img = self._shape
        r0, c0 = r - w // 2, c - w // 2

        emb = np.zeros((self.mosaic.shape[0], w, w), dtype=np.float32)
        bm = np.zeros((w, w), dtype=np.int64)
        valid = np.zeros((w, w), dtype=bool)

        sr0, sc0 = max(r0, 0), max(c0, 0)
        sr1, sc1 = min(r0 + w, h_img), min(c0 + w, w_img)
        if sr1 > sr0 and sc1 > sc0:
            dr0, dc0 = sr0 - r0, sc0 - c0
            dr1, dc1 = dr0 + (sr1 - sr0), dc0 + (sc1 - sc0)
            emb[:, dr0:dr1, dc0:dc1] = np.asarray(
                self.mosaic[:, sr0:sr1, sc0:sc1], dtype=np.float32
            )
            bm[dr0:dr1, dc0:dc1] = self.bitmask[sr0:sr1, sc0:sc1]
            valid[dr0:dr1, dc0:dc1] = self.valid[sr0:sr1, sc0:sc1]

        return {
            "image": torch.from_numpy(emb),
            "bitmask": torch.from_numpy(bm),
            "valid": torch.from_numpy(valid),
        }


# ── B (block): pooled embeddings + geometric features ─────────────────────────

def pool_blocks_mean(mosaic: np.ndarray, block_idx: np.ndarray, n_blocks: int) -> np.ndarray:
    """Mean-pool embedding pixels per block via the ``block_id`` index raster.

    Returns ``(n_blocks, C)``, row ``i`` for ``block_idx`` value ``i+1``
    (1-based in the raster). Zero-valued pixels (no tile coverage — the
    mosaic's nodata convention) are excluded from the mean; blocks with zero
    covered pixels get an all-zero row.
    """
    c, h, w = mosaic.shape
    flat_idx = block_idx.ravel().astype(np.int64)
    covered = np.any(mosaic != 0, axis=0).ravel()
    valid_px = (flat_idx > 0) & covered
    idx0 = flat_idx[valid_px] - 1

    out = np.zeros((n_blocks, c), dtype=np.float64)
    counts = np.bincount(idx0, minlength=n_blocks).astype(np.float64)
    flat_emb = mosaic.reshape(c, -1)[:, valid_px]
    for ch in range(c):
        out[:, ch] = np.bincount(idx0, weights=flat_emb[ch], minlength=n_blocks)
    nonzero = counts > 0
    out[nonzero] /= counts[nonzero, None]
    return out.astype(np.float32)


def sample_blocks_attention_set(
    mosaic: np.ndarray, block_idx: np.ndarray, n_blocks: int, *, k: int = 256, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``k`` sampled pixel embeddings per block, for :class:`AttentionPool`.

    Returns ``(features [n_blocks, k, C], mask [n_blocks, k])``; blocks with
    fewer than ``k`` covered pixels are zero-padded with ``mask=False``.
    """
    c = mosaic.shape[0]
    rng = np.random.default_rng(seed)
    flat_idx = block_idx.ravel().astype(np.int64)
    covered = np.any(mosaic != 0, axis=0).ravel()
    flat_emb = mosaic.reshape(c, -1)

    features = np.zeros((n_blocks, k, c), dtype=np.float32)
    mask = np.zeros((n_blocks, k), dtype=bool)
    order = np.argsort(flat_idx, kind="stable")
    sorted_idx = flat_idx[order]
    starts = np.searchsorted(sorted_idx, np.arange(1, n_blocks + 1))
    ends = np.searchsorted(sorted_idx, np.arange(1, n_blocks + 1), side="right")
    for b in range(n_blocks):
        px = order[starts[b]:ends[b]]
        px = px[covered[px]]
        if px.size == 0:
            continue
        take = px if px.size <= k else rng.choice(px, size=k, replace=False)
        n = take.size
        features[b, :n] = flat_emb[:, take].T
        mask[b, :n] = True
    return features, mask


class BlockDataset(Dataset):
    """B: pooled embedding (+ geometric/UCP features) per block, for B1/B2/B3.

    Node feature = pooled embedding + geometric descriptors (area, compactness,
    elongation), always; + UCP columns only when ``use_ucp_features=True``
    (default False — the circularity guard: labels are derived from these
    UCPs, so headline models must not see them, design principle 8).
    """

    GEOMETRIC_COLS = ("area_m2", "compactness", "elongation")

    def __init__(
        self,
        pooled: np.ndarray,
        blocks_df,
        *,
        ucp_df=None,
        use_ucp_features: bool = False,
    ):
        self.pooled = pooled
        self.blocks_df = blocks_df.reset_index(drop=True)
        geo = self.blocks_df.reindex(columns=list(self.GEOMETRIC_COLS)).fillna(0.0)
        extra = [geo.to_numpy(dtype=np.float32)]
        if use_ucp_features:
            if ucp_df is None:
                raise ValueError("use_ucp_features=True requires ucp_df")
            ucp_cols = [c for c in ucp_df.columns if c not in ("block_id",)]
            extra.append(
                ucp_df.set_index("block_id").loc[self.blocks_df["block_id"], ucp_cols]
                .fillna(0.0).to_numpy(dtype=np.float32)
            )
        self.extra = np.concatenate(extra, axis=1) if extra else np.zeros((len(pooled), 0))

        bm = self.blocks_df.get("bitmask")
        self.bitmask = (bm.to_numpy() if bm is not None else np.zeros(len(pooled), dtype=np.uint32))
        self.confidence = self.blocks_df.get(
            "confidence", pd_series_zeros(len(pooled))
        ).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.pooled)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "embedding": torch.from_numpy(self.pooled[index]),
            "extra": torch.from_numpy(self.extra[index]),
            "bitmask": torch.tensor(int(self.bitmask[index]), dtype=torch.int64),
            "confidence": torch.tensor(float(self.confidence[index]), dtype=torch.float32),
        }


def pd_series_zeros(n: int):
    import pandas as pd
    return pd.Series(np.zeros(n))


def block_graph_edges(adjacency_df, block_id_to_pos: dict) -> np.ndarray:
    """Adjacency parquet -> a PyG-style ``edge_index`` ``(2, 2E)`` (undirected).

    Remaps ``block_id`` strings to dense 0-based positions matching a
    :class:`BlockDataset`'s row order; edges outside the AOI's block set
    (shouldn't occur — adjacency is built per-AOI) are dropped defensively.
    """
    a = adjacency_df["block_a"].map(block_id_to_pos)
    b = adjacency_df["block_b"].map(block_id_to_pos)
    keep = a.notna() & b.notna()
    a, b = a[keep].to_numpy(dtype=np.int64), b[keep].to_numpy(dtype=np.int64)
    return np.concatenate([np.stack([a, b]), np.stack([b, a])], axis=1)
