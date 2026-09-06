"""Resolution-coarsening and seam-repair filters must behave correctly on
synthetic label grids.

Offline: synthetic numpy arrays only, no data mounts, no GPU.

`majority_pool` and `gaussian_likelihood_filter` are the only thing standing
between a raw 10 m sliding-window prediction and a physically meaningful LCZ
map (LCZ is a ~100 m urban-climate concept). A silently wrong nodata
convention, an off-by-one block size, or a sigma table that isn't actually
wired per-class would corrupt every coarsened map without an obvious symptom.
`repair_seams` sits right next to the multi-tile mosaic step, where an
unbounded fill would silently smear real "no tile covers this" regions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.lcz_smoothing import (  # noqa: E402
    gaussian_likelihood_filter,
    majority_pool,
    repair_seams,
)


def test_majority_pool_recovers_block_mode():
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[0:4, 0:4] = 3
    labels[0:4, 4:8] = 7
    labels[4:8, 0:4] = 7
    labels[4:8, 4:8] = 3
    # One stray pixel per block shouldn't flip the block's majority.
    labels[0, 0] = 9
    pooled = majority_pool(labels, factor=4)
    assert pooled.shape == (2, 2)
    assert pooled.tolist() == [[3, 7], [7, 3]]


def test_majority_pool_allnodata_block_stays_nodata():
    labels = np.full((4, 4), 5, dtype=np.uint8)
    labels[0:2, 0:2] = 0
    pooled = majority_pool(labels, factor=2)
    assert pooled[0, 0] == 0
    assert pooled[0, 1] == 5 and pooled[1, 0] == 5 and pooled[1, 1] == 5


def test_majority_pool_output_shape_drops_remainder():
    labels = np.ones((7, 7), dtype=np.uint8)
    pooled = majority_pool(labels, factor=2)
    assert pooled.shape == (3, 3)


def test_gaussian_filter_removes_isolated_misclassification():
    labels = np.full((41, 41), 6, dtype=np.uint8)
    labels[20, 20] = 12
    denoised = gaussian_likelihood_filter(labels, native_res_m=10.0, out_res_m=None)
    assert denoised.shape == labels.shape
    assert denoised[20, 20] == 6


def _island(size: int, island_cls: int, sea_cls: int, island_side: int) -> np.ndarray:
    """A square ``island_side``-px island of ``island_cls`` centred in a sea
    of ``sea_cls``, on a ``size``x``size`` grid."""
    labels = np.full((size, size), sea_cls, dtype=np.uint8)
    c = size // 2
    r = island_side // 2
    labels[c - r:c + r + 1, c - r:c + r + 1] = island_cls
    return labels


def test_gaussian_filter_respects_per_class_sigma():
    # A fixed 7px island of class 2 in a sea of class 1 (sigma held at 100m
    # throughout). With a small sigma for class 2, its own kernel concentrates
    # enough locally to survive; with a large sigma, the same island's mass
    # gets diluted below the sea's likelihood at its own centre. Only class
    # 2's sigma changes between the two calls, so this isolates whether
    # sigma_by_class is actually consulted per class rather than one global
    # value (which could not produce different outcomes here).
    size, island_side = 41, 7
    labels = _island(size, island_cls=2, sea_cls=1, island_side=island_side)
    small_sigma = {1: 100.0, 2: 10.0}
    large_sigma = {1: 100.0, 2: 200.0}

    out_small = gaussian_likelihood_filter(
        labels, native_res_m=10.0, out_res_m=None, sigma_by_class=small_sigma)
    out_large = gaussian_likelihood_filter(
        labels, native_res_m=10.0, out_res_m=None, sigma_by_class=large_sigma)

    c = size // 2
    assert out_small[c, c] == 2, "a tight sigma should hold the island's own peak"
    assert out_large[c, c] == 1, "a very wide sigma should dilute the island away"


def test_gaussian_filter_nodata_edges_stay_nodata():
    labels = np.zeros((30, 30), dtype=np.uint8)
    labels[10:20, 10:20] = 6
    out = gaussian_likelihood_filter(
        labels, native_res_m=10.0, out_res_m=None, min_coverage=0.5
    )
    assert out[0, 0] == 0
    assert out[15, 15] == 6


def test_gaussian_filter_downsample_shape():
    labels = np.full((100, 100), 6, dtype=np.uint8)
    out = gaussian_likelihood_filter(labels, native_res_m=10.0, out_res_m=100.0)
    assert out.shape == (10, 10)
    assert (out == 6).all()


def test_gaussian_filter_rejects_upsampling():
    labels = np.full((10, 10), 6, dtype=np.uint8)
    try:
        gaussian_likelihood_filter(labels, native_res_m=100.0, out_res_m=10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out_res_m < native_res_m")


def test_repair_seams_fills_only_thin_gaps():
    raster = np.full((20, 20), 4, dtype=np.uint8)
    raster[15, 15:17] = 0          # 2px-wide seam, flanked by valid pixels
    raster[0:12, 0:12] = 0         # a genuinely large, unrelated nodata region
    repaired, _ = repair_seams(raster, max_dist_px=2)
    assert repaired[15, 15] != 0 and repaired[15, 16] != 0
    assert repaired[5, 5] == 0, "a 12x12 nodata region's interior must stay nodata"
    # A per-pixel distance threshold (rather than a per-component one) would
    # also erode the *edge* of the big block, since any region's own rim is
    # always close to valid data by definition -- that must not happen either.
    assert repaired[11, 11] == 0, "a 12x12 nodata region's own boundary must not be eroded"
    assert repaired[0, 0] == 0
