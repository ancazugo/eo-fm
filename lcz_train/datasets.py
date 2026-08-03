"""T2 — datasets: valid-mask construction (A/dense), block samples (B).

This module starts with the training-time mask transform shared by all dense
models; the window/block dataset classes land with the harness (M5).

Erosion is a *training-time* transform (the exported rasters are un-eroded):
each block's rasterized region is eroded inward by ``erosion_px`` using the
``block_id`` index raster, so boundary-mixed pixels leave the loss. Because
the index raster burns *all* blocks (unlabelled too, 0 = no block), erosion
also fires on labelled/unlabelled boundaries.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion


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
