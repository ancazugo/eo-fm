"""LCZ-Generator QC rules, as published. Offline — no gpkg, no network.

The rules under test come from the Zenodo release notes, Demuzere et al. 2021
(LCZ Generator) and 2022 (ESSD global map), and the WUDAPT digitizing guide.
Where a test encodes a number, that number is from a source, not from taste.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from lcz_wudapt.config import WudaptConfig
from lcz_wudapt.qc import (
    apply_qc,
    fits_square,
    neighbour_relations,
    polygon_weights,
    qc_step1,
    reduce_oversize,
    resolve_duplicates,
    shape_index,
    submission_gates,
    temporal_alignment,
)

# A metric CRS so side lengths in the fixtures are literally metres.
_UTM = "EPSG:32633"


def _sq(cx: float, cy: float, side: float):
    h = side / 2.0
    return box(cx - h, cy - h, cx + h, cy + h)


# ── QC step 1: area and shape ────────────────────────────────────────────────

def test_shape_index_of_a_square_is_4_over_pi():
    """P^2/(4 pi A) for a square = 16 s^2 / (4 pi s^2) = 4/pi = 1.2732."""
    assert shape_index(np.array([100.0**2]), np.array([4 * 100.0]))[0] == pytest.approx(
        4 / np.pi, rel=1e-9
    )


def test_shape_index_of_a_circle_is_one():
    r = 50.0
    got = shape_index(np.array([np.pi * r**2]), np.array([2 * np.pi * r]))[0]
    assert got == pytest.approx(1.0, rel=1e-9)


def test_elongated_polygons_exceed_the_generator_threshold_of_3():
    """The Generator flags shape >= 3; a 10:1 strip must trip it, a square must not."""
    long_side, short_side = 1000.0, 100.0
    area = long_side * short_side
    per = 2 * (long_side + short_side)
    assert shape_index(np.array([area]), np.array([per]))[0] > 3.0
    assert shape_index(np.array([100.0**2]), np.array([400.0]))[0] < 3.0


def test_qc_step1_applies_both_area_and_shape():
    rules = WudaptConfig().qc
    df = pd.DataFrame(
        {
            "area_km2_utm": [0.05, 0.01, 0.05],   # ok, too small, ok
            "shape_utm": [1.3, 1.3, 4.0],         # ok, ok, too complex
        }
    )
    assert list(qc_step1(df, rules)) == [True, False, False]


def test_shipped_flag_is_lenient_at_high_latitude():
    """The released qc_step1 is computed on Web-Mercator area, inflated by
    1/cos^2(lat). A polygon that is genuinely below 0.04 km2 at 60N passes the
    shipped test and must fail ours -- the whole reason area is recomputed.
    """
    true_km2 = 0.03
    inflation = 1.0 / np.cos(np.radians(60.0)) ** 2   # ~4.0
    assert true_km2 * inflation > 0.04                # shipped flag would pass it
    rules = WudaptConfig().qc
    df = pd.DataFrame({"area_km2_utm": [true_km2], "shape_utm": [1.3]})
    assert not bool(qc_step1(df, rules).iloc[0])      # ours rejects it


# ── The So2Sat geometry target ───────────────────────────────────────────────

def test_fits_square_prefilter_rejects_too_small_and_accepts_large_enough():
    g = gpd.GeoSeries([_sq(0, 0, 300.0), _sq(2000, 0, 400.0)], crs=_UTM)
    assert list(fits_square(g, 320.0)) == [False, True]


def test_fits_square_rejects_a_thin_strip_with_ample_area():
    """Area is not width. A 2 km x 100 m strip is 0.2 km2 -- five times the
    Generator's minimum -- and contains no 320 m square at all. This is the
    WUDAPT '>200 m at the narrowest point' rule doing real work.
    """
    strip = gpd.GeoSeries([box(0, 0, 2000, 100)], crs=_UTM)
    assert strip.area.iloc[0] / 1e6 > 0.04
    assert not fits_square(strip, 320.0)[0]


# ── Relationship to other labels ─────────────────────────────────────────────

def test_neighbour_distance_and_conflict_flag():
    """WUDAPT asks for >100 m between different LCZs. A 50 m gap violates it."""
    rules = WudaptConfig().qc
    gdf = gpd.GeoDataFrame(
        {"class": [1, 2]},
        geometry=[box(0, 0, 400, 400), box(450, 0, 850, 400)],  # 50 m apart
        crs=_UTM,
    ).to_crs("EPSG:4326")
    out = neighbour_relations(gdf, rules)
    assert out["nbr_dist_m"].max() == pytest.approx(50.0, abs=2.0)
    assert bool(out["nbr_conflict"].all())


def test_same_class_neighbours_are_not_a_conflict():
    """Two annotators agreeing on adjacent ground is not contested."""
    rules = WudaptConfig().qc
    gdf = gpd.GeoDataFrame(
        {"class": [5, 5]},
        geometry=[box(0, 0, 400, 400), box(410, 0, 810, 400)],
        crs=_UTM,
    ).to_crs("EPSG:4326")
    out = neighbour_relations(gdf, rules)
    assert np.isinf(out["nbr_dist_m"]).all()
    assert not bool(out["nbr_conflict"].any())


def test_overlap_fraction_with_a_different_class():
    rules = WudaptConfig().qc
    gdf = gpd.GeoDataFrame(
        {"class": [1, 2]},
        geometry=[box(0, 0, 400, 400), box(200, 0, 600, 400)],  # half overlap
        crs=_UTM,
    ).to_crs("EPSG:4326")
    out = neighbour_relations(gdf, rules)
    assert out["overlap_frac_diff_class"].iloc[0] == pytest.approx(0.5, abs=0.02)


def test_essd_priority_keeps_the_highest_accuracy_submission():
    """ESSD: 'only the submission with the highest overall accuracy' is kept."""
    gdf = gpd.GeoDataFrame(
        {
            "class": [1, 2],
            "acc": [0.60, 0.90],
            "submission_date": pd.to_datetime(["2022-01-01", "2023-01-01"], utc=True),
        },
        geometry=[box(0, 0, 400, 400), box(200, 0, 600, 400)],
        crs=_UTM,
    )
    assert list(resolve_duplicates(gdf)) == [False, True]


def test_priority_leaves_uncontested_and_same_class_overlaps_alone():
    gdf = gpd.GeoDataFrame(
        {
            "class": [3, 3, 7],
            "acc": [0.5, 0.9, 0.4],
            "submission_date": pd.to_datetime(["2022-01-01"] * 3, utc=True),
        },
        geometry=[box(0, 0, 400, 400), box(200, 0, 600, 400), box(5000, 0, 5400, 400)],
        crs=_UTM,
    )
    assert list(resolve_duplicates(gdf)) == [True, True, True]


def test_priority_yields_a_conflict_free_survivor_set():
    """The rank order is total, so exactly one member of every different-class
    overlapping pair survives. build_label_raster relies on this to skip its
    contested-pixel scan, so it is pinned here rather than left to luck.
    """
    rng = np.random.default_rng(0)
    n = 40
    xs = rng.uniform(0, 2000, n)
    gdf = gpd.GeoDataFrame(
        {
            "class": rng.integers(1, 6, n),
            "acc": rng.uniform(0.4, 0.95, n),
            "submission_date": pd.to_datetime(["2022-01-01"] * n, utc=True),
        },
        geometry=[box(x, 0, x + 400, 400) for x in xs],
        crs=_UTM,
    )
    won = resolve_duplicates(gdf)
    kept = gdf[won].reset_index(drop=True)
    pairs = gpd.sjoin(kept, kept.rename(columns={"class": "class_r"}), predicate="intersects")
    assert not bool((pairs["class"] != pairs["class_r"]).any())


# ── Oversize reduction ───────────────────────────────────────────────────────

def test_oversize_polygons_are_reduced_to_a_350m_core():
    """Generator: >1.5 km2 polygons are reduced to a ~350 m radius."""
    rules = WudaptConfig().qc
    big = _sq(0, 0, 3000.0)          # 9 km2
    small = _sq(20000, 0, 500.0)     # 0.25 km2
    g = gpd.GeoSeries([big, small], crs=_UTM)
    out = reduce_oversize(g, rules)
    assert out.iloc[0].area < np.pi * rules.oversize_core_radius_m**2 * 1.01
    assert out.iloc[1].equals(small)


# ── Temporal alignment ───────────────────────────────────────────────────────

def test_embedding_year_is_the_nearest_available():
    rules = WudaptConfig().qc            # embedding_years = (2017, 2025)
    out = temporal_alignment(pd.Series([2016, 2019, 2022, 2024]), rules)
    assert list(out["embedding_year"]) == [2017, 2017, 2025, 2025]
    assert list(out["year_lag"]) == [1.0, 2.0, 3.0, 1.0]


def test_year_matching_beats_a_fixed_epoch():
    """The measured median lag against a fixed 2017 is 4 years. Year-matching to
    {2017, 2025} must never do worse and usually does better.
    """
    rules = WudaptConfig().qc
    years = pd.Series(range(2015, 2025))
    out = temporal_alignment(years, rules)
    fixed = (years - 2017).abs()
    assert (out["year_lag"].to_numpy() <= fixed.to_numpy()).all()
    assert out["year_lag"].mean() < fixed.mean()


def test_soft_decay_keeps_most_of_the_corpus_unlike_tau_3():
    """tau=8 is soft; tau=3 is a hard filter in disguise. Measured on the real
    release, tau=3 puts 80.1% of polygons below weight 0.5 and tau=8 only 29.6%.
    A four-year lag -- the measured median -- must stay above 0.5 at tau=8.
    """
    cfg = WudaptConfig()
    out8 = temporal_alignment(pd.Series([2021]), cfg.qc)   # lag 4 either way
    assert float(out8["w_time"].iloc[0]) > 0.5

    strict = cfg.model_copy(deep=True)
    strict.qc.time_decay_years = 3.0
    out3 = temporal_alignment(pd.Series([2021]), strict.qc)
    assert float(out3["w_time"].iloc[0]) < 0.5


def test_tau_none_disables_temporal_weighting():
    cfg = WudaptConfig().model_copy(deep=True)
    cfg.qc.time_decay_years = None
    out = temporal_alignment(pd.Series([1995, 2024]), cfg.qc)
    assert list(out["w_time"]) == [1.0, 1.0]


def test_missing_dates_do_not_get_a_free_pass():
    """Unknown provenance is not evidence of freshness."""
    rules = WudaptConfig().qc
    out = temporal_alignment(pd.Series([2017, 2024, None]), rules)
    assert float(out["w_time"].iloc[2]) < 1.0


# ── Weights ──────────────────────────────────────────────────────────────────

def test_weights_are_bounded_and_never_zero():
    """Gates are booleans, weights are soft: a weight of 0 would be a silent drop."""
    rules = WudaptConfig().qc
    df = pd.DataFrame(
        {
            "acc": [0.0, 0.5, 1.0, np.nan],
            "area_km2_utm": [0.0001, 0.05, 10.0, 0.5],
            "nbr_dist_m": [0.0, 50.0, np.inf, 500.0],
            "w_time": np.float32([0.1, 0.5, 1.0, 0.3]),
        }
    )
    out = polygon_weights(df, rules)
    assert (out["weight"] > 0).all()
    assert (out["weight"] <= 1.0).all()


def test_proximity_discounts_a_label_rather_than_deleting_it():
    """Tehran has 96% of polygons within 100 m of another class. A hard cut would
    delete the city; the weight must merely discount it.
    """
    rules = WudaptConfig().qc
    df = pd.DataFrame(
        {"acc": [0.8, 0.8], "area_km2_utm": [0.2, 0.2],
         "nbr_dist_m": [0.0, 1000.0], "w_time": np.float32([1.0, 1.0])}
    )
    out = polygon_weights(df, rules)
    assert out["weight"].iloc[0] < out["weight"].iloc[1]
    assert out["weight"].iloc[0] >= 0.5 * out["weight"].iloc[1]


# ── Submission gates ─────────────────────────────────────────────────────────

def test_bechtel_oa_floor_of_half():
    rules = WudaptConfig().qc
    df = pd.DataFrame({"class": [1, 1], "oa": [0.45, 0.55], "oau": [0.9, 0.9], "f1_1": [0.8, 0.8]})
    assert list(submission_gates(df, rules)["gate_oa"]) == [False, True]


def test_built_classes_score_on_oau_and_natural_on_oa():
    rules = WudaptConfig().qc
    df = pd.DataFrame({"class": [3, 14], "oa": [0.7, 0.7], "oau": [0.2, 0.2]})
    acc = submission_gates(df, rules)["acc"]
    assert acc.iloc[0] == pytest.approx(0.2)   # built -> oau
    assert acc.iloc[1] == pytest.approx(0.7)   # natural -> oa


def test_missing_metrics_pass_rather_than_silently_failing():
    rules = WudaptConfig().qc
    df = pd.DataFrame({"class": [5], "oa": [np.nan], "oau": [np.nan]})
    g = submission_gates(df, rules)
    assert bool(g["gate_oa"].iloc[0]) and bool(g["gate_oau"].iloc[0])


# ── End to end ───────────────────────────────────────────────────────────────

def _aoi_frame() -> gpd.GeoDataFrame:
    geoms = [_sq(0, 0, 600.0), _sq(1000, 0, 600.0), _sq(3000, 0, 150.0)]
    gdf = gpd.GeoDataFrame(
        {
            "class": [1, 14, 5],
            "oa": [0.8, 0.8, 0.8],
            "oau": [0.75, 0.75, 0.75],
            "f1_1": [0.7] * 3, "f1_14": [0.7] * 3, "f1_5": [0.7] * 3,
            "qc_step1": [True, True, False],
            "label_year": pd.array([2018, 2022, 2020], dtype="Int16"),
            "submission_id": ["a", "b", "c"],
            "submission_date": pd.to_datetime(["2022-01-01"] * 3, utc=True),
            "aoi": ["x"] * 3,
        },
        geometry=geoms, crs=_UTM,
    ).to_crs("EPSG:4326")
    return gdf


def test_apply_qc_emits_the_full_column_contract():
    out = apply_qc(_aoi_frame(), WudaptConfig())
    for col in (
        "area_km2_utm", "perimeter_m", "shape_utm", "qc1", "qc1_shipped", "fits_patch",
        "acc", "class_f1", "gate_oa", "gate_oau", "gate_f1", "wins_conflict",
        "nbr_dist_m", "nbr_conflict", "overlap_frac_diff_class",
        "embedding_year", "year_lag", "w_time", "w_acc", "w_area", "w_nbr",
        "weight", "qc_pass",
    ):
        assert col in out.columns, col


def test_apply_qc_drops_nothing_so_any_operating_point_is_recoverable():
    src = _aoi_frame()
    out = apply_qc(src, WudaptConfig())
    assert len(out) == len(src)
    assert not bool(out["qc_pass"].all())   # the 150 m polygon fails step 1


def test_conflict_priority_is_switchable():
    cfg = WudaptConfig().model_copy(deep=True)
    gdf = gpd.GeoDataFrame(
        {
            "class": [1, 2],
            "oa": [0.9, 0.6], "oau": [0.9, 0.6], "f1_1": [0.8, 0.8], "f1_2": [0.8, 0.8],
            "qc_step1": [True, True],
            "label_year": pd.array([2020, 2020], dtype="Int16"),
            "submission_id": ["a", "b"],
            "submission_date": pd.to_datetime(["2022-01-01"] * 2, utc=True),
            "aoi": ["x", "x"],
        },
        geometry=[_sq(0, 0, 600.0), _sq(300, 0, 600.0)], crs=_UTM,
    ).to_crs("EPSG:4326")

    cfg.qc.use_conflict_priority = True
    assert int(apply_qc(gdf, cfg)["qc_pass"].sum()) == 1
    cfg.qc.use_conflict_priority = False
    assert int(apply_qc(gdf, cfg)["qc_pass"].sum()) == 2


def test_priority_does_not_depend_on_the_callers_index():
    """The rank bookkeeping is positional; a non-RangeIndex caller must still get
    correct answers, not answers that happen to line up.
    """
    gdf = gpd.GeoDataFrame(
        {
            "class": [1, 2],
            "acc": [0.60, 0.90],
            "submission_date": pd.to_datetime(["2022-01-01", "2023-01-01"], utc=True),
        },
        geometry=[box(0, 0, 400, 400), box(200, 0, 600, 400)],
        crs=_UTM,
        index=["poly-b", "poly-a"],
    )
    got = resolve_duplicates(gdf)
    assert list(got.index) == ["poly-b", "poly-a"]
    assert list(got) == [False, True]
