"""Colour models: determinism, persistence, and the stretch contract.

Offline: synthetic pixels only, so these run without the data mounts. The UMAP
path is skipped when umap-learn is unavailable; the t-SNE and PCA paths always
run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_rgb import (  # noqa: E402
    STRETCH_PCT,
    ColourModel,
    MLPColour,
    PCAColour,
    build_model,
    subspace_angles_check,
)


def _pixels(n=4000, c=16, seed=0):
    """Structured pixels: a few blobs, so 3 components are actually meaningful."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(5, c)) * 3
    which = rng.integers(0, len(centres), n)
    return (centres[which] + rng.normal(size=(n, c))).astype(np.float32)


# ── PCA ───────────────────────────────────────────────────────────────────────

def test_pca_transform_is_in_unit_range():
    m = PCAColour().fit(_pixels())
    out = m.transform(_pixels(seed=1))
    assert out.shape[1] == 3
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_pca_is_deterministic():
    X = _pixels()
    a = PCAColour().fit(X).transform(X)
    b = PCAColour().fit(X).transform(X)
    assert np.allclose(a, b)


def test_pca_component_signs_are_pinned():
    """Without a sign convention a refit silently inverts the map's colours."""
    X = _pixels()
    a = PCAColour().fit(X)
    b = PCAColour().fit(X)
    assert np.allclose(a.components_, b.components_)
    # The convention itself: largest-magnitude loading is positive.
    lead = a.components_[np.arange(3), np.argmax(np.abs(a.components_), axis=1)]
    assert (lead > 0).all()


def test_pca_uses_one_common_divisor():
    """PC3 must not be inflated to full range by an independent stretch.

    An independent per-PC stretch would give every axis the same span and make
    a low-variance PC3 look as structured as PC1.
    """
    m = PCAColour().fit(_pixels())
    spans = m.hi_ - m.lo_
    assert np.allclose(spans, spans[0]), "PCA axes should share one divisor"


def test_pca_preserves_relative_variance_in_output():
    """With a shared divisor, a low-variance PC3 really does vary less on screen.

    Built on data with a deliberate variance hierarchy, because that is the
    situation the shared divisor exists for: an independent per-axis stretch
    would give all three channels the same on-screen spread and make PC3 look
    as structured as PC1.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8000, 6)) * np.array([10.0, 3.0, 0.5, 0.2, 0.1, 0.05])
    m = PCAColour().fit(X)
    spread = m.transform(X).std(axis=0)
    assert spread[0] > spread[1] > spread[2]
    # And PC3 is genuinely faint, not merely third.
    assert spread[2] < 0.2 * spread[0]


def test_scale_flag_changes_the_fit():
    """Documents that --scale is correlation PCA, a genuinely different model."""
    X = _pixels()
    X[:, 0] *= 50.0                       # one dominant channel
    a = PCAColour().fit(X, scale=False)
    b = PCAColour().fit(X, scale=True)
    assert not np.allclose(np.abs(a.components_[0]), np.abs(b.components_[0]))
    assert a.meta["scaled"] is False and b.meta["scaled"] is True


# ── Persistence ───────────────────────────────────────────────────────────────

def test_pca_roundtrip(tmp_path):
    X = _pixels()
    m = PCAColour()
    m.meta["embedding_name"] = "tesserav2"
    m.fit(X)
    p = tmp_path / "colour.npz"
    m.save(p)

    loaded = ColourModel.load(p)
    assert loaded.method == "pca"
    assert loaded.meta["embedding_name"] == "tesserav2"
    assert np.array_equal(m.transform(X), loaded.transform(X))


def test_stretch_comes_from_the_model_not_the_image(tmp_path):
    """The whole point of persisting the model: two ROIs share a stretch.

    A per-image stretch would map each subset onto the full 0-1 range and the
    two would be incomparable.
    """
    X = _pixels(n=6000)
    m = PCAColour().fit(X)
    m.save(tmp_path / "c.npz")
    loaded = ColourModel.load(tmp_path / "c.npz")

    # Two disjoint "ROIs" drawn from the same space.
    a, b = X[:1500], X[3000:4500]
    assert np.array_equal(loaded.lo_, m.lo_) and np.array_equal(loaded.hi_, m.hi_)
    # Neither subset is individually renormalised to fill the range.
    assert not np.isclose(loaded.transform(a).min(), 0.0, atol=1e-6) or \
           not np.isclose(loaded.transform(a).max(), 1.0, atol=1e-6)
    # And a point present in both maps to the same colour.
    shared = X[100:200]
    assert np.array_equal(loaded.to_uint8(shared), m.to_uint8(shared))


def test_to_uint8_range():
    m = PCAColour().fit(_pixels())
    out = m.to_uint8(_pixels(seed=5))
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255


# ── Distilled manifold models ─────────────────────────────────────────────────

def test_tsne_model_fits_and_roundtrips(tmp_path):
    X = _pixels(n=1200, c=12)
    m = build_model("tsne", manifold_max=600, pre_pca=8, perplexity=15.0,
                    hidden=(64, 32))
    m.fit(X, seed=0)
    assert m.method == "tsne"
    # Fidelity is reported, not assumed.
    assert "holdout_r2" in m.meta and "holdout_median_err" in m.meta

    p = tmp_path / "tsne.npz"
    m.save(p)
    loaded = ColourModel.load(p)
    assert np.allclose(m.transform(X), loaded.transform(X), atol=1e-9)


def test_distilled_head_is_smooth():
    """The property that makes a manifold usable on a raster.

    Raw umap.transform() is not Lipschitz: near-identical pixels can land far
    apart, which on a map reads as spatial texture that is not in the data. An
    MLP cannot do that -- a small input perturbation gives a small output one.
    """
    X = _pixels(n=1000, c=12)
    m = build_model("tsne", manifold_max=500, pre_pca=8, perplexity=10.0,
                    hidden=(64, 32))
    m.fit(X, seed=0)

    probe = X[:200]
    nudged = probe + np.random.default_rng(0).normal(scale=1e-4, size=probe.shape)
    delta = np.abs(m.transform(probe) - m.transform(nudged)).max()
    assert delta < 0.01, f"tiny input change moved the colour by {delta}"


def test_mlp_forward_matches_sklearn():
    """The saved weights are replayed by hand, so applying never depends on the
    sklearn version that fitted the model."""
    from sklearn.neural_network import MLPRegressor

    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(300, 8))
    Y = rng.normal(size=(300, 3))
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8), random_state=0,
                       max_iter=50).fit(Xr, Y)

    m = MLPColour(method="tsne")
    m.W_ = [np.asarray(w) for w in mlp.coefs_]
    m.b_ = [np.asarray(b) for b in mlp.intercepts_]
    assert np.allclose(m._head(Xr), mlp.predict(Xr), atol=1e-8)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("umap") is None,
    reason="umap-learn not installed",
)
def test_umap_model_fits_and_roundtrips(tmp_path):
    X = _pixels(n=800, c=12)
    m = build_model("umap", manifold_max=400, pre_pca=8, n_neighbors=10,
                    min_dist=0.1, hidden=(64, 32))
    m.fit(X, seed=0)
    m.save(tmp_path / "umap.npz")
    loaded = ColourModel.load(tmp_path / "umap.npz")
    assert loaded.method == "umap"
    assert np.allclose(m.transform(X), loaded.transform(X), atol=1e-9)


def test_manifold_models_use_per_axis_stretch():
    """Unlike PCA, these axes carry no variance ordering, so per-axis is right."""
    X = _pixels(n=1000, c=12)
    m = build_model("tsne", manifold_max=500, pre_pca=8, perplexity=10.0,
                    hidden=(32,))
    m.fit(X, seed=0)
    spans = m.hi_ - m.lo_
    assert spans.shape == (3,)
    assert not np.allclose(spans, spans[0])   # independent, by design


# ── Sample adequacy ───────────────────────────────────────────────────────────

def test_subspace_angles_small_for_a_large_sample():
    """Turns 'I picked N pixels' into a number."""
    ang = subspace_angles_check(_pixels(n=20000, c=16), seed=0)
    assert len(ang) == 3
    assert max(ang) < 10.0, f"basis unstable across halves: {ang}"


def test_stretch_percentiles_are_symmetric():
    assert STRETCH_PCT[0] + STRETCH_PCT[1] == 100.0
