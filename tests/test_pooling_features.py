"""Block-wise pooling: shape safety, masking, and the legacy layout contract.

Offline: every fixture is synthetic, so these run without the data mounts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.pooling_features import (  # noqa: E402
    POOL_BLOCKS,
    POOLING_RECIPES,
    _center_slice,
    _ring_masks,
    feature_dim,
    pool_blocks,
    recipe_blocks,
)

# Only ~18% of extracted So2Sat patches are square; these are the shapes that
# actually occur across tesserav2 and alpha_earth_coop.
REAL_SHAPES = [(33, 33), (33, 34), (34, 33), (35, 33), (33, 35), (36, 33), (12, 12)]


def _patch(c=8, h=33, w=33, seed=0):
    return np.random.default_rng(seed).normal(size=(c, h, w)).astype(np.float32)


# ── The historical modes must not move ────────────────────────────────────────

def test_gap_matches_plain_mean():
    """`gap` has to stay bit-identical to the old _pool, or caches are void."""
    arr = _patch()
    out = pool_blocks(arr, ("mean",))["mean"]
    assert np.array_equal(out, arr.mean(axis=(1, 2)).astype(np.float32))


def test_mean_std_matches_legacy_concat():
    arr = _patch()
    got = pool_blocks(arr, ("mean", "std"))
    legacy = np.concatenate([arr.mean(axis=(1, 2)), arr.std(axis=(1, 2))])
    assert np.allclose(np.concatenate([got["mean"], got["std"]]), legacy, atol=1e-6)


def test_recipe_widths():
    assert feature_dim("gap", 128) == 128
    assert feature_dim("mean_std", 128) == 256
    assert feature_dim("quantile", 64) == 192
    assert feature_dim("rich", 64) == 64 * len(POOL_BLOCKS)


def test_unknown_recipe_and_block_raise():
    with pytest.raises(ValueError, match="unknown pooling recipe"):
        recipe_blocks("nope")
    with pytest.raises(ValueError, match="unknown pool blocks"):
        pool_blocks(_patch(), ("nope",))


# ── Shape safety ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("h,w", REAL_SHAPES)
def test_all_blocks_survive_every_real_shape(h, w):
    out = pool_blocks(_patch(c=6, h=h, w=w), POOL_BLOCKS)
    assert set(out) == set(POOL_BLOCKS)
    for name, v in out.items():
        assert v.shape == (6,), name
        assert np.isfinite(v).all(), name


def test_center_slice_is_geometric_centre():
    """Odd -> the one central pixel; even -> both, so the point is the centre."""
    assert _center_slice(33) == slice(16, 17)
    assert _center_slice(34) == slice(16, 18)


def test_center_is_symmetric_under_transpose():
    """33x34 and 34x33 must not lean opposite ways.

    With a naive H // 2 the sampled point sits half a pixel past centre and
    which way it leans flips between the two shapes, so a gradient patch and its
    transpose would disagree. Averaging the central 2x2 removes that.
    """
    c = 4
    grad = np.tile(np.linspace(0, 1, 34, dtype=np.float32), (c, 33, 1))  # (c,33,34)
    a = pool_blocks(grad, ("center",))["center"]
    b = pool_blocks(np.transpose(grad, (0, 2, 1)).copy(), ("center",))["center"]
    assert np.allclose(a, b, atol=1e-6)
    # And it really is the middle of the ramp.
    assert np.allclose(a, 0.5, atol=1e-6)


def test_center_of_constant_patch_is_that_constant():
    arr = np.full((5, 34, 33), 3.25, dtype=np.float32)
    assert np.allclose(pool_blocks(arr, ("center",))["center"], 3.25)


def test_ring_masks_are_relative_and_partition():
    for h, w in REAL_SHAPES:
        inner, outer = _ring_masks(h, w)
        assert inner.shape == (h, w)
        assert not (inner & outer).any()          # disjoint
        assert (inner | outer).all()              # and exhaustive
        assert inner.any() and outer.any()        # neither degenerates
        assert inner[h // 2, w // 2]              # the centre is inside


def test_ring_detects_centre_surround_difference():
    """ring_in - ring_out is the point of the block: a centre unlike its surround."""
    arr = np.zeros((3, 33, 33), dtype=np.float32)
    inner, _ = _ring_masks(33, 33)
    arr[:, inner] = 1.0
    out = pool_blocks(arr, ("ring_in", "ring_out", "mean"))
    assert np.allclose(out["ring_in"], 1.0)
    assert np.allclose(out["ring_out"], 0.0)
    # A plain mean cannot tell this patch from a uniform one of the same mean.
    assert np.allclose(out["mean"], inner.mean(), atol=1e-6)


def test_quantiles_are_ordered():
    out = pool_blocks(_patch(c=16, seed=3), ("q10", "q50", "q90"))
    assert (out["q10"] <= out["q50"]).all()
    assert (out["q50"] <= out["q90"]).all()


def test_quantiles_are_permutation_invariant():
    """Documents *why* quantiles alone cannot answer the texture question."""
    arr = _patch(c=4, seed=7)
    rng = np.random.default_rng(0)
    flat = arr.reshape(4, -1)
    shuffled = flat[:, rng.permutation(flat.shape[1])].reshape(arr.shape)
    a = pool_blocks(arr, ("q10", "q50", "q90", "mean", "std"))
    b = pool_blocks(shuffled, ("q10", "q50", "q90", "mean", "std"))
    for k in a:
        assert np.allclose(a[k], b[k], atol=1e-5), k
    # ...whereas ring pooling does move, which is the whole point.
    ra = pool_blocks(arr, ("ring_in",))["ring_in"]
    rb = pool_blocks(shuffled, ("ring_in",))["ring_in"]
    assert not np.allclose(ra, rb, atol=1e-3)


# ── Masking ───────────────────────────────────────────────────────────────────

def test_mask_of_all_valid_equals_unmasked():
    arr = _patch(c=6, h=34, w=33)
    valid = np.ones((34, 33), dtype=bool)
    a = pool_blocks(arr, POOL_BLOCKS)
    b = pool_blocks(arr, POOL_BLOCKS, valid=valid)
    for k in a:
        assert np.allclose(a[k], b[k], atol=1e-5), k


def test_mask_excludes_sentinel_from_mean():
    """A coop-shaped case: a few huge sentinel pixels dragging the mean."""
    arr = np.ones((4, 33, 33), dtype=np.float32)
    valid = np.ones((33, 33), dtype=bool)
    arr[:, :2, :] = 8.06          # the dequantized -128 sentinel's L2 norm
    valid[:2, :] = False
    assert np.allclose(pool_blocks(arr, ("mean",), valid=valid)["mean"], 1.0)
    assert pool_blocks(arr, ("mean",))["mean"][0] > 1.4   # unmasked is biased


def test_all_invalid_patch_falls_back_rather_than_nan():
    """Matches models.pooling.pool_mean_std: fall back, never propagate NaN."""
    arr = _patch(c=5)
    valid = np.zeros(arr.shape[1:], dtype=bool)
    out = pool_blocks(arr, POOL_BLOCKS, valid=valid)
    for k, v in out.items():
        assert np.isfinite(v).all(), k
    assert np.allclose(out["mean"], arr.mean(axis=(1, 2)), atol=1e-5)


def test_masked_center_falls_back_when_centre_is_invalid():
    """One sentinel at the centre would otherwise *be* the whole feature."""
    arr = np.ones((3, 33, 33), dtype=np.float32)
    arr[:, 16, 16] = 99.0
    valid = np.ones((33, 33), dtype=bool)
    valid[16, 16] = False
    assert np.allclose(pool_blocks(arr, ("center",), valid=valid)["center"], 1.0)


def test_masked_quantiles_ignore_invalid_pixels():
    arr = np.zeros((2, 20, 20), dtype=np.float32)
    valid = np.ones((20, 20), dtype=bool)
    arr[:, :10, :] = 100.0        # half the patch is garbage
    valid[:10, :] = False
    out = pool_blocks(arr, ("q10", "q50", "q90"), valid=valid)
    for v in out.values():
        assert np.allclose(v, 0.0)


def test_output_dtype_and_order():
    blocks = ("std", "mean", "q50")
    out = pool_blocks(_patch(), blocks)
    assert list(out) == list(blocks)      # concatenation order is the recipe's
    assert all(v.dtype == np.float32 for v in out.values())


def test_rejects_non_chw_input():
    with pytest.raises(ValueError, match=r"expected \(C, H, W\)"):
        pool_blocks(np.zeros((33, 33), dtype=np.float32), ("mean",))


def test_every_recipe_uses_known_blocks():
    for name, blocks in POOLING_RECIPES.items():
        assert set(blocks) <= set(POOL_BLOCKS), name
