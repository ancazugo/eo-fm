"""Task 2.1 — the Phase 1 augmentation changed its RNG stream, not its semantics.

`f45fd80` replaced a per-sample Python loop with batched tensor ops, and its
docstring asserts "the distribution is unchanged". Task 2.1 reproduces a
pre-Phase-1 result and can therefore only ask for **seed-level** agreement, not
bit-level. That relaxation is only legitimate if the assertion is true, so these
tests pin it rather than trust it.

The old implementation is vendored below from `e12c7c2:src/training/augment.py`,
verbatim apart from the noise constants being lifted to arguments so both sides
can be driven at the same settings. It is compared against the current one at the
anchor's configuration — sigma 0.05 applied with probability 0.5, which under
`--normalize none` is the absolute-units noise the original runs used.

Three properties are separable and are tested separately, because a single
pooled test cannot see all of them:

- **values** — Kolmogorov-Smirnov, pooled and per channel. This is where noise lives.
- **geometry** — the frequency of each of the 8 dihedral transforms. A pooled KS
  is *blind* to this: flips and rotations permute pixel positions, and the
  distribution over all positions is invariant to permutation.
- **noise gating** — the rate at which noise is applied, and its magnitude.

None of those three can see whether the draws are independent *across samples
within a batch*, which is a fourth property and the one Task 2.1b was opened to
settle; the tests for it are in their own section at the foot of this file.

Every test seeds `torch` explicitly, so the p-values are fixed rather than
resampled per run — a test that fails 1% of the time is worse than no test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.augment import augment_images  # noqa: E402

ALPHA = 0.01


# ── The pre-Phase-1 implementation, vendored from e12c7c2 ────────────────────

def _augment_images_pre_phase1(
    images: torch.Tensor, noise_sigma: float = 0.05, noise_prob: float = 0.5
) -> torch.Tensor:
    """`augment_images` as it stood before f45fd80: one Python-level pass per
    sample, drawing its own scalars. The constants were literals in the original
    (0.05 and 0.5); they are arguments here only so the comparison can be driven
    at matched settings."""
    aug = []
    for img in images:
        if torch.rand(1) < 0.5:
            img = img.flip(-1)
        if torch.rand(1) < 0.5:
            img = img.flip(-2)
        k = torch.randint(0, 4, (1,)).item()
        if k:
            img = torch.rot90(img, k, dims=(-2, -1))
        if torch.rand(1) < noise_prob:
            img = img + torch.randn_like(img) * noise_sigma
        aug.append(img)
    return torch.stack(aug)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _distinct_patches(n: int, c: int = 2, s: int = 8) -> torch.Tensor:
    """Patches whose every pixel is distinct, so the dihedral transform applied
    to one is recoverable by exact match against the 8 candidates."""
    base = torch.arange(c * s * s, dtype=torch.float32).reshape(1, c, s, s)
    return base + torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1) * 1e5


def _dihedral_elements(x: torch.Tensor) -> list[torch.Tensor]:
    """The 8 elements of the dihedral group applied to one (C,H,W) patch."""
    out = []
    for flip in (False, True):
        b = x.flip(-1) if flip else x
        out.extend(b if k == 0 else torch.rot90(b, k, dims=(-2, -1)) for k in range(4))
    return out


def _dihedral_ids(inp: torch.Tensor, out: torch.Tensor) -> list[int]:
    ids = []
    for x, y in zip(inp, out):
        matches = [i for i, e in enumerate(_dihedral_elements(x)) if torch.equal(e, y)]
        assert len(matches) == 1, f"transform not uniquely identified: {matches}"
        ids.append(matches[0])
    return ids


def _counts(ids: list[int]) -> torch.Tensor:
    return torch.bincount(torch.tensor(ids), minlength=8).double()


# ── Values ───────────────────────────────────────────────────────────────────

def test_pooled_value_distributions_are_indistinguishable():
    """The headline: every value either implementation emits, pooled."""
    torch.manual_seed(0)
    images = torch.randn(3000, 4, 8, 8)

    torch.manual_seed(1)
    old = _augment_images_pre_phase1(images).flatten().numpy()
    torch.manual_seed(1)
    new = augment_images(images.clone()).flatten().numpy()

    assert ks_2samp(old, new).pvalue > ALPHA


def test_per_channel_value_distributions_are_indistinguishable():
    """Pooling across channels could hide a per-channel shift that cancels."""
    torch.manual_seed(0)
    images = torch.randn(2000, 4, 8, 8) * torch.tensor([1.0, 5.0, 0.2, 50.0]).view(1, 4, 1, 1)

    torch.manual_seed(2)
    old = _augment_images_pre_phase1(images)
    torch.manual_seed(2)
    new = augment_images(images.clone())

    for c in range(images.shape[1]):
        p = ks_2samp(old[:, c].flatten().numpy(), new[:, c].flatten().numpy()).pvalue
        assert p > ALPHA, f"channel {c}: p={p}"


# ── Geometry ─────────────────────────────────────────────────────────────────

def test_every_dihedral_transform_is_reachable_and_equally_likely():
    """Both implementations compose flip_h, flip_v and rot90(k) in that order,
    which covers the 8-element dihedral group uniformly. Noise is off so the
    transform can be identified by exact match."""
    torch.manual_seed(0)
    images = _distinct_patches(4000)

    torch.manual_seed(3)
    old_ids = _dihedral_ids(images, _augment_images_pre_phase1(images, noise_sigma=0.0))
    torch.manual_seed(3)
    new_ids = _dihedral_ids(images, augment_images(images.clone(), noise_sigma=0.0))

    old_c, new_c = _counts(old_ids), _counts(new_ids)
    assert (old_c > 0).all() and (new_c > 0).all()

    expected = len(images) / 8.0
    for name, c in (("old", old_c), ("new", new_c)):
        chi2 = (((c - expected) ** 2) / expected).sum().item()
        assert chi2 < 24.32, f"{name} transforms not uniform: chi2={chi2}"  # df=7, p=0.001


def test_geometry_is_exact_with_no_interpolation():
    """The pixel set must survive untouched — a resize or a rounded rotation
    would leave the KS tests happy while quietly changing the data."""
    torch.manual_seed(4)
    images = _distinct_patches(64)
    out = augment_images(images.clone(), noise_sigma=0.0)
    for x, y in zip(images, out):
        assert torch.equal(x.flatten().sort().values, y.flatten().sort().values)


# ── Noise gating ─────────────────────────────────────────────────────────────

def test_noise_is_applied_at_the_same_rate():
    """A zero input makes the gate directly observable: an untouched sample is
    exactly zero, a noised one is not."""
    n = 6000
    images = torch.zeros(n, 2, 8, 8)

    torch.manual_seed(5)
    old_hit = (_augment_images_pre_phase1(images) != 0).any(dim=(1, 2, 3)).double().mean()
    torch.manual_seed(5)
    new_hit = (augment_images(images.clone()) != 0).any(dim=(1, 2, 3)).double().mean()

    for name, rate in (("old", old_hit), ("new", new_hit)):
        assert abs(rate.item() - 0.5) < 0.02, f"{name} noise rate {rate.item()}"


def test_noise_magnitude_distribution_matches():
    """Given the gate fires, the perturbation itself must be the same N(0, sigma)."""
    images = torch.zeros(4000, 4, 8, 8)

    torch.manual_seed(6)
    old = _augment_images_pre_phase1(images)
    torch.manual_seed(6)
    new = augment_images(images.clone())

    old_v = old[(old != 0).any(dim=(1, 2, 3))].flatten().numpy()
    new_v = new[(new != 0).any(dim=(1, 2, 3))].flatten().numpy()

    assert ks_2samp(old_v, new_v).pvalue > ALPHA
    assert abs(float(old_v.std()) - 0.05) < 0.002
    assert abs(float(new_v.std()) - 0.05) < 0.002


def test_sigma_zero_disables_noise_in_both():
    images = torch.zeros(256, 2, 8, 8)
    torch.manual_seed(7)
    assert torch.equal(_augment_images_pre_phase1(images, noise_sigma=0.0), images)
    torch.manual_seed(7)
    assert torch.equal(augment_images(images.clone(), noise_sigma=0.0), images)


# ── The negative control ─────────────────────────────────────────────────────

def test_a_fixed_seed_does_not_reproduce_the_old_sequence():
    """This is why Task 2.1 asks for seed-level rather than bit-level agreement.
    The batched version draws n values at once where the loop drew scalars per
    sample, so the streams diverge even from an identical seed. If this ever
    starts passing, the two implementations have converged and the anchor could
    be compared bit for bit instead."""
    torch.manual_seed(0)
    images = torch.randn(128, 2, 8, 8)

    torch.manual_seed(8)
    old = _augment_images_pre_phase1(images)
    torch.manual_seed(8)
    new = augment_images(images.clone())

    assert not torch.equal(old, new)


# ── Contract preserved from the current implementation ───────────────────────

def test_the_validity_mask_gets_the_same_geometry_and_no_noise():
    """Only the new implementation takes a mask; flipping the image without the
    mask would misalign it from the data it describes."""
    torch.manual_seed(9)
    images = _distinct_patches(256)
    valid = (images[:, :1] % 2 == 0).float()

    out, out_valid = augment_images(images.clone(), valid=valid, noise_sigma=0.0)
    for i in range(len(images)):
        # The mask is 0/1 and often symmetric, so its own transform is not
        # uniquely identifiable. Identify the transform from the image, where
        # every pixel is distinct, then require the mask to have taken that one.
        tid = _dihedral_ids(images[i : i + 1], out[i : i + 1])[0]
        assert torch.equal(_dihedral_elements(valid[i])[tid], out_valid[i])
    assert set(out_valid.unique().tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.2])
def test_shape_and_dtype_survive(sigma):
    images = torch.randn(32, 3, 8, 8)
    out = augment_images(images.clone(), noise_sigma=sigma)
    assert out.shape == images.shape and out.dtype == images.dtype


# ── Task 2.1b — independence ACROSS samples within one batch ─────────────────
#
# Every test above passes under a mechanism they cannot see. "One batched draw
# instead of N scalar draws" has two readings: `torch.rand(n)`, which keeps
# per-sample randomness, and `torch.rand(1)` broadcast, which hands every sample
# in a batch the SAME transform. The marginal distribution of augmented values
# is identical either way, and so is the marginal distribution of transforms
# pooled over batches — so KS and the pooled chi-square are both blind to it.
#
# The difference is real: under broadcast, augmentation diversity per epoch
# collapses from 8^B states to 8, a uniform handicap on every training run and
# invisible to any marginal test. What separates the two readings is
# independence across samples *within* one batch, which is what these measure.
#
# Batches of 64 here, per the Rev C specification; the anchor trains at 256, so
# the real margin against a collapsed batch is wider still.

BATCH_2_1B = 64
N_BATCHES_2_1B = 200


def _per_batch_dihedral_ids(fn, n_batches: int = N_BATCHES_2_1B) -> list[list[int]]:
    """The transform each sample actually received, batch by batch.

    Recovered by exact match against the 8 candidates rather than read off the
    draws, so it tests the applied result and would also catch a correct draw
    that is then broadcast during application.
    """
    out = []
    for _ in range(n_batches):
        images = _distinct_patches(BATCH_2_1B)
        out.append(_dihedral_ids(images, fn(images.clone(), noise_sigma=0.0)))
    return out


def test_every_sample_in_a_batch_draws_its_own_transform():
    """The test that fails under broadcast: it would realise exactly 1 distinct
    transform in every batch, against 8 here.

    The bound is 6, not 8. With 64 samples over 8 equiprobable states a batch
    misses one about 0.16% of the time — seed 0 of the old path does it once in
    200 batches — so requiring all 8 every time would be seed-luck rather than a
    property. Six still sits five states clear of the failure being excluded.
    """
    for name, fn in (("old", _augment_images_pre_phase1), ("new", augment_images)):
        torch.manual_seed(11)
        distinct = [len(set(ids)) for ids in _per_batch_dihedral_ids(fn)]
        mean = sum(distinct) / len(distinct)
        assert min(distinct) >= 6, f"{name}: a batch realised only {min(distinct)} transforms"
        assert mean > 7.9, f"{name}: mean {mean:.3f} distinct transforms per batch"


def test_two_samples_in_a_batch_agree_no_more_often_than_chance():
    """Independence stated as the quantity that actually separates the two
    readings: how often do two samples in the same batch receive the same
    transform? Under per-sample draws, 1/8. Under broadcast, 1.

    This replaces the chi-square of sample index against transform that Rev C
    specified. That test was written first and does not work — see
    `test_batch_position_does_not_predict_the_transform` below. Broadcast makes
    samples perfectly dependent on *each other*, not on their position, so a
    test keyed on position is blind to it. Pairwise agreement is keyed on the
    dependence itself and catches partial sharing as well as total.
    """
    chance = 1.0 / 8.0
    for name, fn in (("old", _augment_images_pre_phase1), ("new", augment_images)):
        torch.manual_seed(12)
        agree = total = 0
        for ids in _per_batch_dihedral_ids(fn):
            counts = _counts(ids)
            # pairs sharing a transform, over all pairs in the batch
            agree += int(((counts * (counts - 1)) / 2).sum().item())
            total += BATCH_2_1B * (BATCH_2_1B - 1) // 2
        rate = agree / total
        assert rate < 2 * chance, f"{name}: {rate:.4f} of within-batch pairs share a transform"
        assert abs(rate - chance) < 0.02, f"{name}: agreement {rate:.4f}, chance {chance:.4f}"


# Rev C specified this as "a chi-square test of independence between sample
# index and augmentation state". That test was written, measured and dropped,
# for two independent reasons:
#
#   1. It does not detect the mechanism it was proposed for. Verified against a
#      deliberately broadcast implementation: it PASSES, because broadcast makes
#      samples perfectly dependent on each other, not on their position — every
#      position keeps the same marginal distribution over transforms.
#   2. It is flaky. A 64x8 table over 200 batches leaves ~25 counts per cell,
#      and the old path returned p=0.0035 at one of four seeds tried — a failure
#      at ALPHA on correct code, which this file's own preamble rules out.
#
# The pairwise-agreement test above is the statistic that separates the two
# readings, and it is stable across seeds.


def test_the_noise_gate_is_drawn_per_sample():
    """The third draw, checked the same way. A broadcast gate would noise whole
    batches or none of them, so the per-batch noised fraction would only ever be
    0.0 or 1.0 instead of scattering around 0.5.

    Zeros in, so any non-zero output pixel came from the noise and nowhere else.
    """
    torch.manual_seed(13)
    fractions = []
    for _ in range(N_BATCHES_2_1B):
        out = augment_images(torch.zeros(BATCH_2_1B, 2, 6, 6), noise_sigma=0.05)
        fractions.append((out.reshape(BATCH_2_1B, -1).abs().sum(1) > 0).float().mean().item())

    assert not any(f in (0.0, 1.0) for f in fractions), "a whole batch shared the noise gate"
    mean = sum(fractions) / len(fractions)
    assert 0.45 < mean < 0.55, f"noise applied to {mean:.3f} of samples, expected ~0.5"
