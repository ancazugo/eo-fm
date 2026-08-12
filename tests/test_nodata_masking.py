"""Task 1.5 — nodata predicates, mask propagation and masked pooling.

Offline: every fixture is synthetic, so these run without the data mounts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.registry import get_nodata_predicate  # noqa: E402
from datasets.so2sat import PatchItem  # noqa: E402
from models.pooling import pool_mean_std  # noqa: E402


# ── Predicates ───────────────────────────────────────────────────────────────

def test_alphaearth_predicate_needs_all_channels():
    """-128 in one channel is data; -128 in all 64 is the nodata sentinel."""
    pred = get_nodata_predicate("alpha_earth_coop")
    arr = np.zeros((64, 4, 4), dtype=np.float32)
    arr[:, 1, 1] = -128.0          # full sentinel pixel
    arr[7, 2, 2] = -128.0          # single channel only — real data

    invalid = pred(arr)
    assert invalid[1, 1]
    assert not invalid[2, 2]
    assert invalid.sum() == 1


def test_tessera_predicate_ignores_per_channel_quantization_zeros():
    """Tessera has ~0.9% per-channel zeros; only all-zero pixels are nodata."""
    pred = get_nodata_predicate("tesserav1.1_global")
    arr = np.ones((128, 4, 4), dtype=np.float32)
    arr[:, 0, 0] = 0.0             # all-channel zero — nodata
    arr[3, 3, 3] = 0.0             # quantization zero — data

    invalid = pred(arr)
    assert invalid[0, 0]
    assert not invalid[3, 3]


@pytest.mark.parametrize("name", ["seamless", "sentinel1", "sentinel2"])
def test_families_without_a_sentinel_mark_nothing(name):
    pred = get_nodata_predicate(name)
    arr = np.zeros((8, 4, 4), dtype=np.float32)
    assert not pred(arr).any()


def test_nan_is_invalid_for_every_family():
    arr = np.zeros((8, 2, 2), dtype=np.float32)
    arr[3, 0, 1] = np.nan
    assert get_nodata_predicate("sentinel1")(arr)[0, 1]


# ── Masked pooling ───────────────────────────────────────────────────────────

def test_pool_matches_plain_mean_when_all_valid():
    """The no-nodata path must be unchanged (up to float associativity)."""
    x = torch.randn(4, 6, 8, 8)
    valid = torch.ones(4, 1, 8, 8)

    mean, std = pool_mean_std(x, valid)
    assert torch.allclose(mean, x.mean(dim=(-2, -1)), atol=1e-6)
    assert torch.allclose(std, x.std(dim=(-2, -1), unbiased=False), atol=1e-6)


def test_pool_ignores_invalid_pixels():
    """A masked-out corner must not move the mean, whatever is written into it."""
    x = torch.ones(1, 3, 4, 4)
    valid = torch.ones(1, 1, 4, 4)
    x[:, :, 0, 0] = 999.0
    valid[:, :, 0, 0] = 0.0

    mean, std = pool_mean_std(x, valid)
    assert torch.allclose(mean, torch.ones(1, 3), atol=1e-6)
    assert torch.allclose(std, torch.zeros(1, 3), atol=1e-6)


def test_pool_falls_back_when_nothing_is_valid():
    x = torch.full((1, 2, 3, 3), 5.0)
    mean, _ = pool_mean_std(x, torch.zeros(1, 1, 3, 3))
    assert torch.allclose(mean, torch.full((1, 2), 5.0))


# ── Dataset plumbing ─────────────────────────────────────────────────────────

def _write_patch(tmp_path: Path, arr: np.ndarray) -> Path:
    p = tmp_path / "patch_000000.npy"
    np.save(p, arr)
    return p


def test_dataset_emits_valid_channel_and_fills_sentinel(tmp_path):
    """mask mode: sentinel pixels are flagged and overwritten, not dequantized."""
    from datasets.so2sat import PatchDataset

    arr = np.zeros((64, 8, 8), dtype=np.float32)
    arr[:, 0, 0] = -128.0
    path = _write_patch(tmp_path, arr)

    ds = PatchDataset(
        [PatchItem(path, 0, "train")], patch_size=8,
        nodata_mode="mask", nodata_predicate=get_nodata_predicate("alpha_earth_coop"),
    )
    item = ds[0]

    assert item["valid"].shape == (1, 8, 8)
    assert item["valid"][0, 0, 0] == 0.0
    assert item["valid"].sum() == 63
    assert item["image"][:, 0, 0].abs().max() == 0.0    # filled, not left at -128


def test_zero_mode_preserves_previous_behaviour(tmp_path):
    """'zero' emits no mask and leaves the sentinel untouched — the old path."""
    from datasets.so2sat import PatchDataset

    arr = np.zeros((64, 8, 8), dtype=np.float32)
    arr[:, 0, 0] = -128.0
    path = _write_patch(tmp_path, arr)

    ds = PatchDataset(
        [PatchItem(path, 0, "train")], patch_size=8,
        nodata_mode="zero", nodata_predicate=get_nodata_predicate("alpha_earth_coop"),
    )
    item = ds[0]

    assert "valid" not in item
    assert item["image"][0, 0, 0] == -128.0


def test_task_logs_invalid_fraction_and_passes_the_mask():
    """The task reports the masked fraction and routes the mask to pooling models."""
    from models.linear_probe import LinearProbeModel
    from training.tasks import LCZResNetModule

    task = LCZResNetModule(LinearProbeModel(4, 3), num_classes=3)
    task.train()
    valid = torch.ones(2, 1, 4, 4)
    valid[:, :, 0, :] = 0.0                        # a quarter of every patch
    batch = {
        "image": torch.randn(2, 4, 4, 4),
        "label": torch.tensor([0, 1]),
        "valid": valid,
    }
    task.reset_train_metrics()
    assert task.train_step(batch, torch.device("cpu")) is not None

    logs = task.compute_train_logs()
    assert logs["train_invalid_frac"] == pytest.approx(0.25)


def test_no_mask_means_no_invalid_metric():
    """Without a 'valid' key the extra metric must not appear at all."""
    from models.linear_probe import LinearProbeModel
    from training.tasks import LCZResNetModule

    task = LCZResNetModule(LinearProbeModel(4, 3), num_classes=3)
    task.train()
    task.reset_train_metrics()
    task.train_step(
        {"image": torch.randn(2, 4, 4, 4), "label": torch.tensor([0, 1])},
        torch.device("cpu"),
    )
    assert "train_invalid_frac" not in task.compute_train_logs()


def test_resized_mask_rejects_pixels_blended_with_nodata(tmp_path):
    """Bilinear resize spreads a sentinel, so the mask must spread with it."""
    from datasets.so2sat import PatchDataset

    arr = np.zeros((64, 8, 8), dtype=np.float32)
    arr[:, 0, 0] = -128.0
    path = _write_patch(tmp_path, arr)

    ds = PatchDataset(
        [PatchItem(path, 0, "train")], patch_size=4,          # forces a resize
        nodata_mode="mask", nodata_predicate=get_nodata_predicate("alpha_earth_coop"),
    )
    item = ds[0]

    assert item["image"].shape == (64, 4, 4)
    assert item["valid"].shape == (1, 4, 4)
    # The invalid input pixel must invalidate every output pixel it touched.
    assert item["valid"][0, 0, 0] == 0.0
    assert item["valid"].sum() < 16


def test_mask_is_built_on_the_native_grid_not_after_the_resize(tmp_path):
    """Task 1.5.0 follow-on: ordering, at the real ~1.03x resampling factor.

    Every family is resized (33x33 -> 32x32 for the 10 m embeddings; nothing is
    32x32 natively except the raw Sentinel patches), so a mask computed *after*
    the resize would have to recover the sentinel from an already-blended array.
    An interior invalid pixel at that mild a factor must still come back invalid.
    """
    from datasets.so2sat import PatchDataset

    arr = np.zeros((64, 33, 33), dtype=np.float32)
    arr[:, 17, 17] = -128.0
    path = _write_patch(tmp_path, arr)

    ds = PatchDataset(
        [PatchItem(path, 0, "train")], patch_size=32,
        nodata_mode="mask", nodata_predicate=get_nodata_predicate("alpha_earth_coop"),
    )
    valid = ds[0]["valid"]

    assert valid.shape == (1, 32, 32)
    assert valid.sum() < 32 * 32          # the sentinel survived the resample
    assert valid[0, 16:18, 16:18].min() == 0.0


def test_invalid_pixels_are_filled_with_the_channel_mean(tmp_path):
    """The fill is the channel mean in decoded units, so it normalises to 0.

    Filling with 0 instead would push masked pixels to -mean/std after
    normalisation, which is a real value the resize would then blend into
    neighbouring valid pixels.
    """
    from datasets.so2sat import PatchDataset

    arr = np.zeros((4, 6, 6), dtype=np.float32)
    arr[:, 2, 2] = np.nan                       # NaN is invalid for every family
    path = _write_patch(tmp_path, arr)

    mean = np.array([0.5, -0.25, 2.0, 10.0], dtype=np.float32)
    std = np.ones(4, dtype=np.float32)
    ds = PatchDataset(
        [PatchItem(path, 0, "train")], patch_size=6,
        nodata_mode="mask", nodata_predicate=get_nodata_predicate("sentinel1"),
        normalize="channel", channel_mean=mean, channel_std=std,
    )
    item = ds[0]

    assert item["valid"][0, 2, 2] == 0.0
    # Normalised, the filled pixel is exactly 0 — not -mean/std.
    assert torch.allclose(item["image"][:, 2, 2], torch.zeros(4), atol=1e-5)
