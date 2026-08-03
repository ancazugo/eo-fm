"""T2 — erosion valid-mask correctness at block edges."""

import numpy as np
import pytest

from lcz_train.datasets import erosion_valid_mask


def _two_blocks(h=8, w=8, split=4):
    """Blocks 1|2 side by side, both labelled hard 3 / hard 6."""
    block_idx = np.zeros((h, w), dtype=np.uint32)
    block_idx[:, :split] = 1
    block_idx[:, split:] = 2
    bitmask = np.zeros((h, w), dtype=np.uint32)
    bitmask[:, :split] = 1 << 2
    bitmask[:, split:] = 1 << 5
    return block_idx, bitmask


def test_erosion_removes_block_boundary_band():
    block_idx, bitmask = _two_blocks()
    valid = erosion_valid_mask(block_idx, bitmask, erosion_px=1)
    assert not valid[:, 3].any() and not valid[:, 4].any()  # 1 px each side
    assert valid[:, :3].all() and valid[:, 5:].all()


def test_erosion_radius_two():
    block_idx, bitmask = _two_blocks(w=12, split=6)
    valid = erosion_valid_mask(block_idx, bitmask, erosion_px=2)
    assert not valid[:, 4:8].any()
    assert valid[:, :4].all() and valid[:, 8:].all()


def test_erosion_zero_keeps_all_labelled():
    block_idx, bitmask = _two_blocks()
    valid = erosion_valid_mask(block_idx, bitmask, erosion_px=0)
    assert valid.all()


def test_unlabelled_block_is_invalid_and_erodes_neighbours():
    block_idx, bitmask = _two_blocks()
    bitmask[:, 4:] = 0  # block 2 unlabelled
    valid = erosion_valid_mask(block_idx, bitmask, erosion_px=1)
    assert not valid[:, 4:].any()          # unlabelled itself
    assert not valid[:, 3].any()           # labelled ring next to it erodes too
    assert valid[:, :3].all()


def test_no_block_background_erodes_labelled_ring():
    block_idx = np.zeros((8, 8), dtype=np.uint32)
    block_idx[2:6, 2:6] = 7
    bitmask = np.where(block_idx > 0, np.uint32(1 << 8), np.uint32(0))
    valid = erosion_valid_mask(block_idx, bitmask, erosion_px=1)
    assert valid[3:5, 3:5].all()           # interior 2x2 survives
    assert valid.sum() == 4                # the 1-px ring is gone


def test_confidence_threshold():
    block_idx, bitmask = _two_blocks()
    conf = np.full((8, 8), 0.9)
    conf[0] = 0.2
    valid = erosion_valid_mask(block_idx, bitmask, conf, min_conf=0.5, erosion_px=0)
    assert not valid[0].any() and valid[1:].all()


def test_image_edge_is_not_a_block_edge():
    # A single block filling the raster: nothing erodes at the image border.
    block_idx = np.full((6, 6), 3, dtype=np.uint32)
    bitmask = np.full((6, 6), np.uint32(1 << 0))
    valid = erosion_valid_mask(block_idx, bitmask, erosion_px=1)
    assert valid.all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        erosion_valid_mask(np.zeros((4, 4), np.uint32), np.zeros((4, 5), np.uint32))
