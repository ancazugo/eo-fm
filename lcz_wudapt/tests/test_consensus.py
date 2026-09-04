"""H2/H3 — author collapse, weights, and the per-pixel consensus posterior."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from lcz_labels.export import decode_bitmask
from lcz_wudapt.config import WudaptConfig
from lcz_wudapt.consensus import _accumulate, _decide, burn_order, consensus_for_aoi, footprint_grid
from lcz_wudapt.ingest import N_LCZ
from lcz_wudapt.quality import apply_gates, polygon_weights, submission_accuracy

# A ~1 km square in UTM terms, at the equator so degrees are easy to reason about.
_SQUARE = box(0.0, 0.0, 0.01, 0.01)


def _polys(specs: list[dict]) -> gpd.GeoDataFrame:
    """specs: [{annotator, class, [date], [geom], [oa], [oau], [f1]}]"""
    rows = []
    for i, s in enumerate(specs):
        row = {
            "annotator_id": s["annotator"],
            "submission_id": s.get("submission", f"sub{i}"),
            "submission_date": pd.Timestamp(s.get("date", "2022-01-01"), tz="UTC"),
            "aoi": "A__30_1",
            "class": s["class"],
            "area_km2": 1.0,
            "vertices": 5,
            "label_year": 2022,
            "qc_step1": True, "qc_step2": True, "qc_step3": True,
            "oa": s.get("oa", 0.8), "oau": s.get("oau", 0.8),
        }
        for c in range(1, N_LCZ + 1):
            row[f"f1_{c}"] = s.get("f1", 0.5)
        rows.append({**row, "geometry": s.get("geom", _SQUARE)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _run(gdf, cfg=None, res_m=100.0):
    cfg = cfg or WudaptConfig()
    w = polygon_weights(gdf, cfg)["weight"].to_numpy()
    return consensus_for_aoi(gdf, w, cfg, res_m=res_m)


def test_unanimous_annotators_give_a_hard_label():
    res = _run(_polys([{"annotator": f"a{i}", "class": 3} for i in range(4)]))
    assert len(res) > 0
    assert (res.set_size == 1).all()
    assert (res.top_class == 3).all()
    assert decode_bitmask(res.bitmask[:1]) == [[3]]


def test_split_annotators_give_a_coarse_set_not_a_forced_guess():
    """The whole point of the bitmask contract: never guess which annotator won."""
    specs = [{"annotator": f"a{i}", "class": 2} for i in range(3)]
    specs += [{"annotator": f"b{i}", "class": 5} for i in range(3)]
    res = _run(_polys(specs))
    assert (res.set_size == 2).all()
    assert decode_bitmask(res.bitmask[:1]) == [[2, 5]]


def test_fully_diffuse_pixels_are_rejected_not_labelled():
    """Five-way disagreement exceeds max_set_size and must emit nothing."""
    res = _run(_polys([{"annotator": f"a{i}", "class": c}
                       for i, c in enumerate([1, 4, 8, 12, 16])]))
    assert len(res) == 0


def test_more_agreeing_annotators_raise_confidence():
    lone = _run(_polys([{"annotator": "a0", "class": 3}]))
    many = _run(_polys([{"annotator": f"a{i}", "class": 3} for i in range(6)]))
    assert many.confidence.mean() > lone.confidence.mean()
    assert many.n_eff.mean() > lone.n_eff.mean()


def test_depth_is_continuous_in_n_eff():
    """A branch at n_eff == 1 made confidence depend on float32-vs-64 rounding.

    Confidence must vary smoothly as a second annotator's weight goes to zero.
    """
    cfg = WudaptConfig()
    confs = []
    for oa in (0.40, 0.4001, 0.41, 0.5):        # second annotator's weight ~0 -> small
        res = _run(_polys([{"annotator": "a", "class": 3},
                           {"annotator": "b", "class": 3, "oa": oa, "oau": oa}]), cfg)
        confs.append(float(res.confidence.mean()))
    assert all(np.isfinite(confs))
    # No cliff: consecutive steps stay small and the trend is monotone up.
    assert max(abs(np.diff(confs))) < 0.15
    assert confs == sorted(confs)


def test_partial_selection_matches_a_full_argsort_reference():
    """_decide only ranks the top (max_set_size + 1) classes. Pin that shortcut."""
    rng = np.random.default_rng(0)
    cfg = WudaptConfig()
    h = w = 40
    S = rng.random((N_LCZ, h, w)).astype(np.float32)
    W1 = S.sum(axis=0)
    W2 = (S ** 2).sum(axis=0)
    grid = footprint_grid(_polys([{"annotator": "a", "class": 1}]), cfg, res_m=100.0)
    grid = type(grid)(grid.transform, grid.crs, (h, w), grid.res_m)

    got = _decide(S, W1, W2, cfg, grid)

    fv = np.flatnonzero(W1.ravel() > 0)
    Sv = S.reshape(N_LCZ, -1)[:, fv].astype(np.float64)
    tot = Sv.sum(0)
    prior = Sv.sum(1) / Sv.sum()
    P = (Sv + cfg.consensus.prior_alpha * prior[:, None]) / (tot + cfg.consensus.prior_alpha)
    order = np.argsort(-P, axis=0)
    cum = np.cumsum(np.take_along_axis(P, order, axis=0), axis=0)
    ss = (cum < cfg.consensus.tau_mass).sum(0) + 1
    rej = (ss > cfg.consensus.max_set_size) | (np.take_along_axis(P, order, axis=0)[0]
                                               < cfg.consensus.min_p_top)

    assert np.array_equal(got.index, fv[~rej])
    assert np.array_equal(got.set_size, ss[~rej].astype(np.int8))


def test_one_author_resubmitting_does_not_count_as_many_votes():
    """8,827 submissions collapse to ~1,500 authors; votes are per author."""
    one_author = _run(_polys([{"annotator": "a", "class": 3, "submission": f"s{i}",
                               "date": f"202{i}-01-01"} for i in range(4)]))
    four_authors = _run(_polys([{"annotator": f"a{i}", "class": 3} for i in range(4)]))
    assert one_author.n_eff.mean() == pytest.approx(1.0, abs=1e-5)
    assert four_authors.n_eff.mean() > 3.0


def test_an_authors_latest_revision_wins_on_self_overlap():
    """Burn order is oldest-first, so a revision overwrites its own earlier version."""
    res = _run(_polys([
        {"annotator": "a", "class": 3, "submission": "old", "date": "2021-01-01"},
        {"annotator": "a", "class": 6, "submission": "new", "date": "2024-01-01"},
    ]))
    assert (res.top_class == 6).all()
    assert (res.set_size == 1).all()


def test_burn_order_is_deterministic():
    gdf = _polys([{"annotator": "b", "class": 1, "date": "2024-01-01"},
                  {"annotator": "a", "class": 2, "date": "2021-01-01"}])
    a = burn_order(gdf)["submission_id"].tolist()
    b = burn_order(gdf.iloc[::-1].reset_index(drop=True))["submission_id"].tolist()
    assert a == b


def test_disjoint_ground_from_the_same_author_all_survives():
    """Collapse must not delete an author's non-overlapping earlier work."""
    far = box(0.02, 0.0, 0.03, 0.01)
    res = _run(_polys([
        {"annotator": "a", "class": 3, "submission": "old", "date": "2021-01-01"},
        {"annotator": "a", "class": 6, "submission": "new", "date": "2024-01-01", "geom": far},
    ]))
    assert set(np.unique(res.top_class)) == {3, 6}


def test_submission_accuracy_uses_oau_for_built_and_oa_for_natural():
    gdf = _polys([{"annotator": "a", "class": 3, "oa": 0.9, "oau": 0.4},
                  {"annotator": "b", "class": 14, "oa": 0.9, "oau": 0.4}])
    got = submission_accuracy(gdf)
    assert got[0] == pytest.approx(0.4)     # built -> urban accuracy
    assert got[1] == pytest.approx(0.9)     # natural -> overall accuracy


def test_failed_qc_step1_is_gated_out_entirely():
    cfg = WudaptConfig()
    gdf = _polys([{"annotator": "a", "class": 3}, {"annotator": "b", "class": 3}])
    gdf.loc[0, "qc_step1"] = False
    assert len(apply_gates(gdf, cfg)) == 1


def test_tiny_polygons_are_gated_out():
    """5% of polygons are under 0.0012 km2 — a dozen 10 m pixels, no LCZ signal."""
    cfg = WudaptConfig()
    gdf = _polys([{"annotator": "a", "class": 3}, {"annotator": "b", "class": 3}])
    gdf.loc[0, "area_km2"] = 1e-5
    assert len(apply_gates(gdf, cfg)) == 1


def test_empty_input_raises_rather_than_returning_a_silent_empty_result():
    with pytest.raises(ValueError):
        consensus_for_aoi(_polys([]).iloc[:0], np.zeros(0), WudaptConfig())


def test_pinned_grid_makes_indices_comparable_across_subsets():
    """Leave-one-author-out cross-indexes results; without a pinned grid those
    indices refer to different rasters and silently compare the wrong pixels."""
    cfg = WudaptConfig()
    near = box(0.0, 0.0, 0.01, 0.01)
    far = box(0.05, 0.05, 0.06, 0.06)
    full = _polys([{"annotator": "a", "class": 3, "geom": near},
                   {"annotator": "b", "class": 3, "geom": near},
                   {"annotator": "c", "class": 6, "geom": far}])
    grid = footprint_grid(full, cfg, res_m=100.0)

    subset = full[full["annotator_id"] != "c"]
    w = polygon_weights(subset, cfg)["weight"].to_numpy()

    pinned = consensus_for_aoi(subset, w, cfg, grid=grid)
    own = consensus_for_aoi(subset, w, cfg, res_m=100.0)

    # The subset's own footprint excludes the far polygon, so its grid differs...
    assert own.grid.shape != grid.shape
    # ...but the pinned run stays addressable in the full AOI's raster.
    assert pinned.grid.shape == grid.shape
    assert pinned.index.max() < grid.shape[0] * grid.shape[1]
    # Same labels either way — only the addressing changes.
    assert set(np.unique(pinned.top_class)) == set(np.unique(own.top_class))
