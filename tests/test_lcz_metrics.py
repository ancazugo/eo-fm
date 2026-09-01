"""G3 -- the WUDAPT metric suite (OAu / OAbu / OAw / kappa_w).

Offline: every metric is a pure function of a 17x17 confusion matrix, so none
of this needs the data mounts.

Two properties carry the weight. The **identity reductions** must hold exactly:
OAw with W = I is plain OA, and kappa_w with W = I is Cohen's kappa. If they
did not, the weighted numbers would be reported alongside the unweighted ones
on a different scale and the kappa/kappa_w gap -- which the campaign treats as
a result in its own right -- would be meaningless.

And the **class ordering** must be pinned. A transposed or alphabetically
sorted similarity CSV still produces plausible-looking OAw values, so nothing
downstream would catch it. LCZ 1 (compact high-rise) must score more similar to
LCZ 4 (open high-rise) than to LCZ B (scattered trees).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.lcz_metrics import (  # noqa: E402
    DEFAULT_SIMILARITY_CSV,
    LCZ_ORDER,
    kappa_weighted,
    lcz_metrics_from_cm,
    load_similarity_matrix,
    oa_built_natural,
    oa_urban,
    oa_weighted,
)

RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def w():
    return load_similarity_matrix()


@pytest.fixture(scope="module")
def cm_and_labels():
    """A random but reproducible 17-class prediction stream and its dense cm."""
    y_true = RNG.integers(0, 17, size=4000)
    # Correlate predictions with truth so kappa is not ~0 and the chance
    # correction is actually exercised.
    flip = RNG.random(4000) < 0.35
    y_pred = np.where(flip, RNG.integers(0, 17, size=4000), y_true)
    cm = np.zeros((17, 17), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm, y_true, y_pred


def test_the_vendored_matrix_loads_and_passes_every_contract(w):
    assert w.shape == (17, 17)
    assert np.allclose(w, w.T)
    assert np.allclose(np.diag(w), 1.0)
    assert w.min() >= 0.0 and w.max() <= 1.0


def test_lcz_1_is_more_similar_to_lcz_4_than_to_lcz_b(w):
    # The permutation check: index 3 is LCZ 4 (open high-rise, same height
    # family), index 11 is LCZ B (scattered trees, different land cover and
    # surface objects).
    assert LCZ_ORDER[0] == "1" and LCZ_ORDER[3] == "4" and LCZ_ORDER[11] == "B"
    assert w[0, 3] > w[0, 11]


def test_oa_weighted_with_identity_is_plain_overall_accuracy(cm_and_labels):
    cm, y_true, y_pred = cm_and_labels
    assert oa_weighted(cm, np.eye(17)) == pytest.approx((y_true == y_pred).mean())


def test_kappa_weighted_with_identity_is_cohens_kappa(cm_and_labels):
    cm, y_true, y_pred = cm_and_labels
    expected = cohen_kappa_score(y_true, y_pred, labels=list(range(17)))
    assert kappa_weighted(cm, np.eye(17)) == pytest.approx(expected, abs=1e-9)


def test_weighted_scores_are_kinder_than_unweighted_ones(cm_and_labels, w):
    """Partial credit can only add, so OAw >= OA on any confusion matrix."""
    cm, _, _ = cm_and_labels
    assert oa_weighted(cm, w) > oa_weighted(cm, np.eye(17))


def test_a_perfect_matrix_scores_one_everywhere(w):
    cm = np.diag(np.full(17, 10.0))
    assert oa_weighted(cm, w) == pytest.approx(1.0)
    assert kappa_weighted(cm, w) == pytest.approx(1.0)
    assert oa_urban(cm) == pytest.approx(1.0)
    assert oa_built_natural(cm) == pytest.approx(1.0)


def test_oa_urban_restricts_both_axes_not_just_rows():
    """Built pixels predicted as natural leave the OAu denominator entirely."""
    cm = np.zeros((17, 17))
    cm[0, 0] = 8.0     # LCZ 1 correct
    cm[0, 16] = 92.0   # LCZ 1 predicted as G (water) -- outside the 10x10 block
    # Row-restricted OA would be 8/100 = 0.08; the block reduction is 8/8 = 1.0.
    assert oa_urban(cm) == pytest.approx(1.0)
    # ...and the built/natural collapse is where that error actually lands.
    assert oa_built_natural(cm) == pytest.approx(0.08)


def test_oa_built_natural_forgives_confusion_inside_a_family():
    cm = np.zeros((17, 17))
    cm[0, 9] = 50.0    # LCZ 1 -> LCZ 10, both built
    cm[10, 16] = 50.0  # LCZ A -> LCZ G, both natural
    assert oa_built_natural(cm) == pytest.approx(1.0)


def test_the_suite_returns_prefixed_keys(cm_and_labels, w):
    cm, _, _ = cm_and_labels
    out = lcz_metrics_from_cm(cm, w, prefix="test")
    assert set(out) == {"test_oau", "test_oabu", "test_oaw", "test_kappa_w"}
    out_patch = lcz_metrics_from_cm(cm, w, prefix="test_patch")
    assert "test_patch_kappa_w" in out_patch


def test_a_sparse_confusion_matrix_is_rejected(w):
    """labels=present matrices mis-index against W, so they must not load."""
    with pytest.raises(ValueError, match="dense"):
        lcz_metrics_from_cm(np.eye(5), w)


def test_a_transposed_similarity_csv_is_rejected(tmp_path):
    """The ordering guard is the only thing standing between us and a silent
    permutation bug, so it must actually fire."""
    good = DEFAULT_SIMILARITY_CSV.read_text().splitlines()
    rows = [ln for ln in good if ln.strip() and not ln.startswith("#")]
    header, body = rows[0], rows[1:]
    # Reverse the class order in both axes -- still symmetric, still unit
    # diagonal, still in [0,1]. Only the ordering check can catch it.
    labels = header.split(",")[1:]
    mat = [ln.split(",")[1:] for ln in body]
    rev = list(reversed(range(17)))
    bad_header = "LCZ," + ",".join(labels[i] for i in rev)
    bad_rows = [
        labels[rev[i]] + "," + ",".join(mat[rev[i]][rev[j]] for j in range(17))
        for i in range(17)
    ]
    p = tmp_path / "bad.csv"
    p.write_text("\n".join([bad_header, *bad_rows]) + "\n")
    with pytest.raises(ValueError, match="order"):
        load_similarity_matrix(p)


def test_comment_lines_and_a_missing_file_are_handled(tmp_path):
    p = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_similarity_matrix(p)
    # The vendored file carries a '#' provenance header; loading it proves
    # comments are skipped rather than parsed as data.
    assert DEFAULT_SIMILARITY_CSV.read_text().startswith("#")
    assert load_similarity_matrix(DEFAULT_SIMILARITY_CSV).shape == (17, 17)


def test_an_empty_matrix_yields_nan_not_a_zero_division(w):
    cm = np.zeros((17, 17))
    assert np.isnan(oa_weighted(cm, w))
    assert np.isnan(kappa_weighted(cm, w))
    assert np.isnan(oa_urban(cm))
    assert np.isnan(oa_built_natural(cm))
