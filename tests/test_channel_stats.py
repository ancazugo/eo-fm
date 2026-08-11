"""Task 1.2 — channel statistics and `--normalize channel`.

Offline: synthetic patches written to tmp_path, no data mounts needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.channel_stats import compute_channel_stats  # noqa: E402
from datasets.registry import get_nodata_predicate  # noqa: E402
from datasets.so2sat import PatchDataset  # noqa: E402


def _items(tmp_path: Path, arrays: list[np.ndarray]) -> list[tuple]:
    items = []
    for i, a in enumerate(arrays):
        p = tmp_path / f"patch_{i:06d}.npy"
        np.save(p, a)
        items.append((p, 0, "train"))
    return items


def test_stats_recover_the_generating_distribution(tmp_path):
    rng = np.random.default_rng(0)
    true_mean = np.array([1.0, -2.0, 10.0], dtype=np.float32)
    true_std = np.array([0.5, 2.0, 0.1], dtype=np.float32)
    arrays = [
        (rng.standard_normal((3, 32, 32)).astype(np.float32) * true_std[:, None, None]
         + true_mean[:, None, None])
        for _ in range(40)
    ]

    mean, std = compute_channel_stats(_items(tmp_path, arrays), nodata_mode="zero")

    assert np.allclose(mean, true_mean, atol=0.05)
    assert np.allclose(std, true_std, rtol=0.1)


def test_normalized_output_is_zero_mean_unit_variance(tmp_path):
    """The point of the exercise: what the model sees is standardised."""
    rng = np.random.default_rng(1)
    arrays = [
        (rng.standard_normal((3, 32, 32)).astype(np.float32) * 7.0 + 4.0)
        for _ in range(40)
    ]
    items = _items(tmp_path, arrays)
    mean, std = compute_channel_stats(items, nodata_mode="zero")

    ds = PatchDataset(items, patch_size=32, normalize="channel",
                      channel_mean=mean, channel_std=std)
    stacked = torch.stack([ds[i]["image"] for i in range(len(ds))])

    assert stacked.mean().abs() < 0.05
    assert abs(float(stacked.std()) - 1.0) < 0.05


def test_stats_exclude_nodata_pixels(tmp_path):
    """AlphaEarth's sentinel inflates std by ~18% if it is not masked out."""
    rng = np.random.default_rng(2)
    arrays = []
    for _ in range(30):
        a = (rng.standard_normal((64, 32, 32)).astype(np.float32) * 0.1)
        a[:, :2, :] = -128.0                      # a strip of sentinel pixels
        arrays.append(a)
    items = _items(tmp_path, arrays)
    pred = get_nodata_predicate("alpha_earth_coop")

    _, std_masked = compute_channel_stats(
        items, nodata_predicate=pred, nodata_mode="mask")
    _, std_raw = compute_channel_stats(items, nodata_mode="zero")

    assert np.median(std_masked) == pytest.approx(0.1, rel=0.1)
    assert np.median(std_raw) > 5 * np.median(std_masked)


def test_normalize_channel_requires_stats():
    with pytest.raises(ValueError, match="requires channel_mean"):
        PatchDataset([], patch_size=32, normalize="channel")


def test_masked_pixels_normalise_to_zero(tmp_path):
    """Invalid pixels are filled with the channel mean, so they land on 0."""
    arr = np.zeros((64, 8, 8), dtype=np.float32)
    arr[:, 0, 0] = -128.0
    items = _items(tmp_path, [arr])
    mean = np.full(64, 3.0, dtype=np.float32)
    std = np.ones(64, dtype=np.float32)

    ds = PatchDataset(
        items, patch_size=8, normalize="channel",
        channel_mean=mean, channel_std=std,
        nodata_mode="mask", nodata_predicate=get_nodata_predicate("alpha_earth_coop"),
    )
    item = ds[0]

    assert item["valid"][0, 0, 0] == 0.0
    assert abs(float(item["image"][:, 0, 0].abs().max())) < 1e-5


def test_cache_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    items = _items(tmp_path, [rng.standard_normal((3, 8, 8)).astype(np.float32)
                              for _ in range(5)])
    cache = tmp_path / "stats.npz"

    m1, s1 = compute_channel_stats(items, patch_size=8, nodata_mode="zero",
                                   cache_path=cache)
    assert cache.exists()
    m2, s2 = compute_channel_stats(items, patch_size=8, nodata_mode="zero",
                                   cache_path=cache)
    assert np.array_equal(m1, m2) and np.array_equal(s1, s2)
