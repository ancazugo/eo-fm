"""T2 — class-balanced weights, PixelWindowDataset, block pooling, BlockDataset."""

import numpy as np
import pandas as pd
import pytest
import torch

from lcz_train.datasets import (
    BlockDataset,
    PixelWindowDataset,
    block_graph_edges,
    class_balanced_weights,
    pool_blocks_mean,
    sample_blocks_attention_set,
)

N_LCZ = 17


def test_class_balanced_weights_hard_inverse_frequency():
    bitmask = np.array([[1 << 0, 1 << 0, 1 << 5]])   # two class-1, one class-6
    valid = np.ones_like(bitmask, dtype=bool)
    w = class_balanced_weights(bitmask, valid)
    assert w[0, 0] == w[0, 1] == pytest.approx(0.5)   # 1/2
    assert w[0, 2] == pytest.approx(1.0)               # 1/1


def test_class_balanced_weights_coarse_is_mean_of_members():
    # class 1 appears twice (hard), class 6 appears once (hard) elsewhere,
    # and one coarse pixel is {1,6} -> weight = mean(1/2, 1/1) = 0.75
    bitmask = np.array([[1 << 0, 1 << 0, 1 << 5, (1 << 0) | (1 << 5)]])
    valid = np.ones_like(bitmask, dtype=bool)
    w = class_balanced_weights(bitmask, valid)
    assert w[0, 3] == pytest.approx((0.5 + 1.0) / 2)


def test_class_balanced_weights_invalid_pixels_are_zero():
    bitmask = np.array([[1, 1, 1]])
    valid = np.array([[True, False, True]])
    w = class_balanced_weights(bitmask, valid)
    assert w[0, 1] == 0.0
    assert w[0, 0] > 0 and w[0, 2] > 0


def test_class_balanced_weights_all_invalid():
    bitmask = np.array([[1, 2]])
    valid = np.zeros_like(bitmask, dtype=bool)
    w = class_balanced_weights(bitmask, valid)
    assert (w == 0).all()


def _synthetic_raster_set(h=200, w=200, seed=0):
    rng = np.random.default_rng(seed)
    mid_h, mid_w = h // 2, w // 2
    bitmask = np.zeros((h, w), dtype=np.uint32)
    bitmask[:mid_h, :mid_w] = 1 << 2      # hard 3
    bitmask[:mid_h, mid_w:] = 1 << 5      # hard 6
    bitmask[mid_h:, :] = 0                # unlabelled
    conf = np.where(bitmask != 0, 90, 0).astype(np.uint8)
    block_idx = np.zeros((h, w), dtype=np.uint32)
    block_idx[:mid_h, :mid_w] = 1
    block_idx[:mid_h, mid_w:] = 2
    block_idx[mid_h:, :mid_w] = 3
    block_idx[mid_h:, mid_w:] = 4
    mosaic = rng.normal(size=(8, h, w)).astype(np.float32)
    mosaic[np.abs(mosaic) < 1e-6] = 1e-3  # avoid the mosaic's zero-as-nodata sentinel
    return mosaic, bitmask, conf, block_idx


def test_pixel_window_dataset_window_range_validation():
    mosaic, bitmask, conf, block_idx = _synthetic_raster_set()
    with pytest.raises(ValueError):
        PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=64)
    with pytest.raises(ValueError):
        PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=300)


def test_pixel_window_dataset_shapes_and_determinism():
    mosaic, bitmask, conf, block_idx = _synthetic_raster_set()
    ds = PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=96,
                           erosion_px=1, samples_per_epoch=5, seed=7)
    assert len(ds) == 5
    item = ds[0]
    assert item["image"].shape == (8, 96, 96)
    assert item["bitmask"].shape == (96, 96)
    assert item["valid"].shape == (96, 96)
    assert item["valid"].dtype == torch.bool

    ds2 = PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=96,
                             erosion_px=1, samples_per_epoch=5, seed=7)
    # Same seed -> same anchor-draw stream: ds's first item matches ds2's first.
    torch.testing.assert_close(item["image"], ds2[0]["image"])


def test_pixel_window_dataset_only_samples_valid_anchors():
    mosaic, bitmask, conf, block_idx = _synthetic_raster_set()
    ds = PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=96,
                           erosion_px=1, samples_per_epoch=50, seed=3)
    for i in range(len(ds)):
        r, c = ds._sample_anchor()
        assert ds.valid[r, c]


def test_pixel_window_dataset_edge_padding_marks_invalid():
    mosaic, bitmask, conf, block_idx = _synthetic_raster_set()
    ds = PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=96,
                           erosion_px=0, samples_per_epoch=1, seed=1)
    item = ds[0]
    # 40x40 raster, 96 window: most of the window pads beyond the raster edge
    assert (~item["valid"]).any()


def test_pixel_window_dataset_raises_when_nothing_valid():
    mosaic, bitmask, conf, block_idx = _synthetic_raster_set()
    bitmask[:] = 0
    with pytest.raises(ValueError, match="no valid"):
        PixelWindowDataset(mosaic, bitmask, conf, block_idx, window_px=96)


def test_pool_blocks_mean():
    mosaic = np.zeros((2, 4, 4), dtype=np.float32)
    mosaic[0, :2, :2] = 3.0
    mosaic[0, :2, 2:] = 5.0
    block_idx = np.zeros((4, 4), dtype=np.uint32)
    block_idx[:2, :2] = 1
    block_idx[:2, 2:] = 2
    pooled = pool_blocks_mean(mosaic, block_idx, n_blocks=2)
    assert pooled.shape == (2, 2)
    np.testing.assert_allclose(pooled[0, 0], 3.0)
    np.testing.assert_allclose(pooled[1, 0], 5.0)


def test_pool_blocks_mean_uncovered_block_is_zero():
    mosaic = np.zeros((2, 4, 4), dtype=np.float32)
    block_idx = np.zeros((4, 4), dtype=np.uint32)
    block_idx[:2, :2] = 1     # block 1 has no embedding coverage (all zero)
    pooled = pool_blocks_mean(mosaic, block_idx, n_blocks=1)
    np.testing.assert_array_equal(pooled[0], 0.0)


def test_pool_blocks_mean_excludes_nodata_pixels():
    mosaic = np.zeros((1, 2, 4), dtype=np.float32)
    mosaic[0, 0, :2] = 10.0   # covered half
    # mosaic[0, 0, 2:] stays 0 -> nodata, excluded from the block-1 mean
    block_idx = np.ones((2, 4), dtype=np.uint32)
    pooled = pool_blocks_mean(mosaic, block_idx, n_blocks=1)
    np.testing.assert_allclose(pooled[0, 0], 10.0)  # not (10+10+0+0+...)/8


def test_sample_blocks_attention_set_padding_and_mask():
    mosaic = np.ones((3, 4, 4), dtype=np.float32)
    block_idx = np.zeros((4, 4), dtype=np.uint32)
    block_idx[:2, :2] = 1   # 4 pixels
    feats, mask = sample_blocks_attention_set(mosaic, block_idx, n_blocks=2, k=10, seed=0)
    assert feats.shape == (2, 10, 3) and mask.shape == (2, 10)
    assert mask[0].sum() == 4         # only 4 real pixels, rest padded
    assert not mask[1].any()          # block 2 has no pixels at all
    assert (feats[0, mask[0]] == 1.0).all()
    assert (feats[0, ~mask[0]] == 0.0).all()


def test_sample_blocks_attention_set_subsamples_when_over_k():
    mosaic = np.ones((2, 10, 10), dtype=np.float32)
    block_idx = np.ones((10, 10), dtype=np.uint32)   # 100 pixels, one block
    feats, mask = sample_blocks_attention_set(mosaic, block_idx, n_blocks=1, k=5, seed=0)
    assert mask[0].sum() == 5


def _blocks_df():
    return pd.DataFrame({
        "block_id": ["a", "b", "c"],
        "area_m2": [100.0, 200.0, 300.0],
        "compactness": [0.5, 0.6, 0.7],
        "elongation": [0.1, 0.2, 0.3],
        "bitmask": [1 << 2, (1 << 2) | (1 << 6), 0],
        "confidence": [0.9, 0.5, 0.0],
    })


def test_block_dataset_basic_shapes():
    pooled = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    ds = BlockDataset(pooled, _blocks_df())
    assert len(ds) == 3
    item = ds[1]
    assert item["embedding"].shape == (8,)
    assert item["extra"].shape == (3,)   # area, compactness, elongation
    assert int(item["bitmask"]) == ((1 << 2) | (1 << 6))
    assert item["confidence"].item() == pytest.approx(0.5)


def test_block_dataset_ucp_features_gated_by_flag():
    pooled = np.zeros((3, 4), dtype=np.float32)
    blocks_df = _blocks_df()
    ucp_df = pd.DataFrame({"block_id": ["a", "b", "c"], "bsf": [0.1, 0.2, 0.3],
                          "h_mean": [5.0, 10.0, 15.0]})

    ds_off = BlockDataset(pooled, blocks_df, ucp_df=ucp_df, use_ucp_features=False)
    assert ds_off.extra.shape[1] == 3   # geometric only

    ds_on = BlockDataset(pooled, blocks_df, ucp_df=ucp_df, use_ucp_features=True)
    assert ds_on.extra.shape[1] == 3 + 2   # + bsf, h_mean


def test_block_dataset_ucp_features_require_ucp_df():
    pooled = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        BlockDataset(pooled, _blocks_df(), use_ucp_features=True)


def test_block_graph_edges_undirected_and_remapped():
    adjacency = pd.DataFrame({"block_a": ["x", "y"], "block_b": ["y", "z"]})
    pos = {"x": 0, "y": 1, "z": 2}
    edges = block_graph_edges(adjacency, pos)
    assert edges.shape == (2, 4)
    pairs = set(map(tuple, edges.T.tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_block_graph_edges_drops_out_of_aoi_blocks():
    adjacency = pd.DataFrame({"block_a": ["x", "x"], "block_b": ["y", "outside"]})
    pos = {"x": 0, "y": 1}
    edges = block_graph_edges(adjacency, pos)
    assert edges.shape == (2, 2)
