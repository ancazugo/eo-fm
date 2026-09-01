"""Segmentation input normalisation, and its interaction with dequantization.

Offline: synthetic arrays on disk, no data mounts.

The property that carries the weight is **ordering**. Channel statistics must
describe the tensor the model actually consumes, which means they are taken
*after* dequantization, while the nodata sentinel must be detected *before* it.
Both halves matter and they pull in opposite directions:

- coop stores invalid pixels as all 64 channels == -128. Dequantization turns
  that into an ordinary-looking vector (L2 norm 8.06 rather than 1.0), so a
  predicate applied post-dequantize cannot find it and those pixels silently
  bias every channel's mean and std.
- seamless expands 13 stored bands to 72 channels during dequantization, so
  statistics taken pre-dequantize would have the wrong length entirely and
  could not be applied to the model's input.

The third property is that normalisation must not leak: statistics come from
train tiles only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.channel_stats import compute_grid_channel_stats  # noqa: E402
from datasets.grid_tiles import GridSegDataset  # noqa: E402

TILE = box(0, 0, 1280, 1280)
POLYS = [(box(100, 100, 420, 420), 3, 0)]
C, H, W = 6, 16, 16


def _write(tmp_path, name, arr):
    p = tmp_path / f"{name}.npy"
    np.save(p, arr.astype(np.float32))
    return p


def _items(tmp_path, arrays):
    return [(_write(tmp_path, f"t{i}", a), TILE, "EPSG:32637", POLYS, None)
            for i, a in enumerate(arrays)]


def test_normalised_output_is_standardised(tmp_path):
    rng = np.random.default_rng(0)
    arrays = [rng.normal(5.0, 3.0, (C, H, W)) for _ in range(12)]
    items = _items(tmp_path, arrays)
    mean, std = compute_grid_channel_stats(items, n_sample=12)

    ds = GridSegDataset(items, "gpkg", normalize="channel",
                        channel_mean=mean, channel_std=std)
    stacked = torch.stack([ds[i]["image"] for i in range(len(ds))])
    per_channel = stacked.permute(1, 0, 2, 3).reshape(C, -1)
    assert torch.allclose(per_channel.mean(1), torch.zeros(C), atol=1e-3)
    assert torch.allclose(per_channel.std(1), torch.ones(C), atol=1e-2)


def test_normalize_none_leaves_the_tensor_untouched(tmp_path):
    arr = np.full((C, H, W), 7.0)
    items = _items(tmp_path, [arr])
    ds = GridSegDataset(items, "gpkg", normalize="none")
    assert torch.allclose(ds[0]["image"], torch.full((C, H, W), 7.0))


def test_channel_normalize_demands_statistics(tmp_path):
    items = _items(tmp_path, [np.zeros((C, H, W))])
    with pytest.raises(ValueError, match="requires channel_mean"):
        GridSegDataset(items, "gpkg", normalize="channel")


def test_an_unknown_normalize_mode_is_rejected(tmp_path):
    items = _items(tmp_path, [np.zeros((C, H, W))])
    with pytest.raises(ValueError, match="normalize must be"):
        GridSegDataset(items, "gpkg", normalize="zscore")


def test_statistics_describe_post_dequantize_values(tmp_path):
    """Stats must match the tensor the model sees, not the stored codes."""
    arr = np.full((C, H, W), 2.0)
    items = _items(tmp_path, [arr])
    plain, _ = compute_grid_channel_stats(items, n_sample=1)
    scaled, _ = compute_grid_channel_stats(
        items, dequantize_fn=lambda a: a * 10.0, n_sample=1)
    assert np.allclose(plain, 2.0)
    assert np.allclose(scaled, 20.0)


def test_a_channel_expanding_dequantize_sets_the_statistic_length(tmp_path):
    """seamless turns 13 stored bands into 72 model channels."""
    items = _items(tmp_path, [np.ones((13, H, W))])
    mean, std = compute_grid_channel_stats(
        items, dequantize_fn=lambda a: np.repeat(a, 4, axis=0), n_sample=1)
    assert len(mean) == 52 and len(std) == 52


def test_the_nodata_predicate_runs_before_dequantization(tmp_path):
    """The coop case: the sentinel is only visible in the stored units."""
    arr = np.ones((C, H, W))
    arr[:, 0, :] = -128.0                      # one row of sentinel pixels
    items = _items(tmp_path, [arr])

    def coop_like(raw):                        # all channels == -128, RAW units
        return (raw == -128.0).all(axis=0)

    # Dequantize maps -128 -> a large-but-plausible value; a predicate applied
    # afterwards would not recognise it.
    deq = lambda a: a * 0.5  # noqa: E731

    masked, _ = compute_grid_channel_stats(
        items, dequantize_fn=deq, nodata_predicate=coop_like, n_sample=1)
    unmasked, _ = compute_grid_channel_stats(
        items, dequantize_fn=deq, n_sample=1)

    # Valid pixels are all 1.0 -> 0.5 after dequantize.
    assert np.allclose(masked, 0.5)
    # Without masking the sentinel row drags the mean far below it.
    assert (unmasked < 0.0).all()


def test_a_fused_tile_is_invalid_where_any_source_is(tmp_path):
    base = np.ones((4, H, W))
    aux = np.ones((2, H, W))
    aux[:, 5, :] = -128.0
    p0, p1 = _write(tmp_path, "s0", base), _write(tmp_path, "s1", aux)
    items = [((p0, p1), TILE, "EPSG:32637", POLYS, None)]

    def pred(raw):
        return (raw == -128.0).all(axis=0)

    ds = GridSegDataset(items, "gpkg", nodata_predicate=[None, pred],
                        emit_valid=True)
    out = ds[0]
    assert out["image"].shape[0] == 6            # channels concatenated
    assert not out["valid"][5, :].any()          # aux sentinel row masked
    assert out["valid"][0, :].all()


def test_valid_is_all_true_without_a_predicate(tmp_path):
    items = _items(tmp_path, [np.ones((C, H, W))])
    ds = GridSegDataset(items, "gpkg", emit_valid=True)
    assert ds[0]["valid"].all()


def test_statistics_are_cached_and_reused(tmp_path):
    items = _items(tmp_path, [np.full((C, H, W), 3.0)])
    cache = tmp_path / "stats.npz"
    a, _ = compute_grid_channel_stats(items, n_sample=1, cache_path=cache)
    assert cache.exists()
    # Corrupt the items; a cached read must not touch them.
    b, _ = compute_grid_channel_stats([], n_sample=1, cache_path=cache)
    assert np.allclose(a, b)


def test_a_constant_channel_cannot_divide_by_zero(tmp_path):
    items = _items(tmp_path, [np.full((C, H, W), 4.0)])
    mean, std = compute_grid_channel_stats(items, n_sample=1)
    assert (std >= 1e-6).all()
    ds = GridSegDataset(items, "gpkg", normalize="channel",
                        channel_mean=mean, channel_std=std)
    assert torch.isfinite(ds[0]["image"]).all()
