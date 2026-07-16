from __future__ import annotations

from typing import Sequence

import numpy as np


def load_and_dequantize_tessera_representation(representation_file_path, scales_file_path):
    """
    Load and dequantize int8 representations back to float32.

    Args:
        representation_file_path: Path to the int8 representation file (H,W,C)
        scales_file_path: Path to the float32 scales file (H,W) or (H,W,1)

    Returns:
        representation_f32: float32 ndarray of shape (H,W,C)
    """
    representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
    scales = np.load(scales_file_path)  # (H, W) or (H, W, 1), dtype=float32

    representation_f32 = representation_int8.astype(np.float32)
    if scales.ndim == 3:  # (H, W, 1) → (H, W)
        scales = scales.squeeze(-1)
    scales_expanded = scales[..., np.newaxis]  # (H, W, 1)
    representation_f32 = representation_f32 * scales_expanded

    return representation_f32


def dequantize_alphaearth_embeddings(values):
    return ((values / 127.5) ** 2) * np.sign(values)


def dequantize_esd(arr: np.ndarray, levels: Sequence[int] = (8, 8, 8, 5, 5, 5)) -> np.ndarray:
    """Dequantize ESD uint16 indices to float32 embeddings.

    Numpy port of the factorized vector quantizer shipped with the Embedded
    Seamless Data release (ESD ``Quantizer.indices_to_codes``); the ``levels``
    default is the published codebook. Applies it to the first 12 bands
    (temporal months) and skips band 13 (QA band). Each index is factorized
    into ``len(levels)`` continuous values in [-1, 1].

    Args:
        arr: ``(13, H, W)`` or ``(12, H, W)`` array of uint16/int32 indices.
        levels: VQ codebook levels per dimension (default: ``(8, 8, 8, 5, 5, 5)``).

    Returns:
        ``(72, H, W)`` float32 array with values in [-1, 1].
    """
    if arr.shape[0] == 13:
        arr = arr[:12]  # drop QA band

    lv = np.array(levels, dtype=np.int32)                           # (L,)
    basis = np.cumprod(np.array([1, *list(levels[:-1])], dtype=np.int32))  # (L,)
    half = lv // 2

    indices = arr.astype(np.int32)[..., np.newaxis]                  # (12, H, W, 1)
    level_indices = (indices // basis) % lv                          # (12, H, W, L)
    codes = (level_indices - half) / half                            # (12, H, W, L) in [-1, 1]
    codes = codes.transpose(0, 3, 1, 2)                              # (12, L, H, W)
    T, L, H, W = codes.shape
    return codes.reshape(T * L, H, W).astype(np.float32)             # (72, H, W)
