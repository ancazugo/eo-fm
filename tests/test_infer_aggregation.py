"""The sliding windows must hand back real probability volumes, and the three
--aggregate modes must pool them into an output cell the way they claim to.

Offline: synthetic arrays and stub nn.Modules, no data mounts, no GPU.

This is the seam where a silent error is invisible: every mode produces a
plausible-looking LCZ map, so a volume that is not actually normalised, a
'majority' that quietly weighs confidence, or a coverage rule that keeps
near-empty edge cells all yield maps nobody would look at twice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infer_roi import (  # noqa: E402
    _sliding_window_cls,
    _sliding_window_seg,
    build_aggregate_volume,
)
from utils.lcz_smoothing import (  # noqa: E402
    gaussian_likelihood_filter,
    smooth_class_volume,
)

NUM_CLASSES = 5
DEVICE = torch.device("cpu")


class _ConstSeg(nn.Module):
    """Segmentation stub: every pixel gets the same fixed logit vector."""

    def __init__(self, logits: np.ndarray):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        B, _, H, W = x.shape
        return self.logits[None, :, None, None].expand(B, -1, H, W).clone()


class _ChannelMeanCls(nn.Module):
    """Classification stub: logits are the patch's own per-channel means, so
    the prediction depends on where the window sits."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return x.mean(dim=(2, 3))[:, : self.num_classes]


def test_seg_window_returns_normalised_probabilities():
    logits = np.array([0.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32)
    arr = np.zeros((2, 24, 24), dtype=np.float32)
    probs = _sliding_window_seg(
        _ConstSeg(logits), arr, patch_size=8, stride=4,
        device=DEVICE, num_classes=NUM_CLASSES, batch_size=4,
    )
    assert probs.shape == (NUM_CLASSES, 24, 24)
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-5)
    assert (probs.argmax(axis=0) == 1).all()
    # Hanning blending must not distort a spatially constant field.
    expected = np.exp(logits) / np.exp(logits).sum()
    assert np.allclose(probs[:, 5, 5], expected, atol=1e-5)


def test_cls_window_returns_normalised_probabilities():
    arr = np.zeros((NUM_CLASSES, 20, 20), dtype=np.float32)
    arr[3] = 10.0  # channel 3 dominates -> class 3 everywhere
    probs = _sliding_window_cls(
        _ChannelMeanCls(NUM_CLASSES), arr, patch_size=8, stride=4,
        device=DEVICE, num_classes=NUM_CLASSES, batch_size=4,
    )
    assert probs.shape == (NUM_CLASSES, 20, 20)
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-5)
    assert (probs.argmax(axis=0) == 3).all()


# ── how the three --aggregate modes pool a cell ──────────────────────────────
# infer_roi hands build_aggregate_volume's output to
# reproject(Resampling.sum); _pool is that accumulation on an aligned grid, so
# the pooling semantics can be asserted without a CRS, a tile or a GPU.

def _volume(probs: np.ndarray, aggregate: str, native_res_m: float = 10.0) -> np.ndarray:
    return build_aggregate_volume(probs, aggregate, native_res_m)


def _pool(probs: np.ndarray, aggregate: str, factor: int, min_coverage: float = 0.5):
    """Sum the volume and its validity mask over factor x factor blocks, then
    divide — the numpy equivalent of the Resampling.sum accumulation."""
    vol = _volume(probs, aggregate)
    valid = (probs.sum(axis=0) > 0).astype(np.float32)
    C, H, W = vol.shape
    b = vol.reshape(C, H // factor, factor, W // factor, factor).sum(axis=(2, 4))
    w = valid.reshape(H // factor, factor, W // factor, factor).sum(axis=(1, 3))
    covered = w >= min_coverage * factor * factor
    out = np.zeros(w.shape, dtype=np.uint8)
    if covered.any():
        out[covered] = (b[:, covered] / w[None, covered]).argmax(axis=0) + 1
    return out, w


def _cell(assignments: list[tuple[int, float]], size: int = 3) -> np.ndarray:
    """One output cell of ``size``x``size`` fine pixels, each pixel given a
    (winning class, winning probability) pair; the remaining mass is spread
    evenly over the other classes."""
    probs = np.zeros((NUM_CLASSES, size, size), dtype=np.float32)
    for i, (cls, p) in enumerate(assignments):
        r, c = divmod(i, size)
        probs[:, r, c] = (1.0 - p) / (NUM_CLASSES - 1)
        probs[cls, r, c] = p
    return probs


def test_soft_and_majority_agree_on_a_uniform_cell():
    probs = _cell([(2, 0.7)] * 9)
    soft, _ = _pool(probs, "soft", factor=3)
    maj, _ = _pool(probs, "majority", factor=3)
    assert soft.item() == maj.item() == 3  # 0-indexed class 2 -> LCZ 3


def test_soft_uses_confidence_where_majority_only_counts():
    # 5 barely-confident pixels of class 0 against 4 near-certain pixels of
    # class 1. Counting votes gives class 0; averaging probability gives
    # class 1. If both modes returned the same answer here, 'soft' would not
    # actually be reading confidence.
    probs = _cell([(0, 0.30)] * 5 + [(1, 0.99)] * 4)
    maj, _ = _pool(probs, "majority", factor=3)
    soft, _ = _pool(probs, "soft", factor=3)
    assert maj.item() == 1, "majority must follow the count"
    assert soft.item() == 2, "soft must follow the accumulated probability"


def test_uncovered_cells_fall_below_min_coverage():
    probs = np.zeros((NUM_CLASSES, 3, 6), dtype=np.float32)
    probs[1, :, :3] = 1.0            # left cell fully covered
    probs[1, 0, 3] = 1.0             # right cell: 1 of 9 pixels
    pooled, weight = _pool(probs, "soft", factor=3)
    assert pooled.tolist() == [[2, 0]]
    assert weight.tolist() == [[9.0, 1.0]]


def test_gaussian_volume_matches_the_hard_label_filter_on_one_hot_input():
    # gaussian_likelihood_filter is now implemented as one-hot ->
    # smooth_class_volume, so feeding smooth_class_volume the same one-hot
    # volume must reproduce it exactly. Locks the refactor.
    rng = np.random.default_rng(0)
    labels = rng.integers(1, NUM_CLASSES + 1, size=(24, 24)).astype(np.uint8)
    onehot = np.stack([labels == c for c in range(1, NUM_CLASSES + 1)]).astype(np.float32)
    smoothed = smooth_class_volume(onehot, native_res_m=10.0)
    expected = gaussian_likelihood_filter(
        labels, native_res_m=10.0, out_res_m=None, num_classes=NUM_CLASSES
    )
    assert np.array_equal((smoothed.argmax(axis=0) + 1).astype(np.uint8), expected)


def test_gaussian_aggregate_removes_an_isolated_fine_misclassification():
    probs = np.zeros((NUM_CLASSES, 30, 30), dtype=np.float32)
    probs[1] = 1.0
    probs[:, 15, 15] = 0.0
    probs[4, 15, 15] = 1.0           # one confident stray pixel
    vol = _volume(probs, "gaussian")
    assert vol.argmax(axis=0)[15, 15] == 1, "a class-appropriate kernel should absorb it"


@pytest.mark.parametrize("aggregate", ["soft", "majority", "gaussian"])
def test_every_mode_leaves_uncovered_pixels_empty(aggregate):
    probs = np.zeros((NUM_CLASSES, 12, 12), dtype=np.float32)
    probs[2, :6, :6] = 1.0
    vol = _volume(probs, aggregate)
    assert (vol[:, 6:, 6:] == 0).all()
