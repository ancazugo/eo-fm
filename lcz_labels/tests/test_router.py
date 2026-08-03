"""Stage 5b compact-lowrise router (LCZ 3 vs 7 vs coarse {3,7}) tests."""

import pytest

from lcz_labels.classify import classify_patch
from lcz_labels.config import LczLabelConfig

CFG = LczLabelConfig()


def _compact_low(**kw):
    """A compact + low-rise + built base row that enters the router."""
    base = dict(
        bsf=0.5, h_mean=6, ghs_built_s=0.5, height_evidence_frac=0.4,
        height_none_frac=0.0, built_area_ml_frac=0.0,
        # morphology / road defaults (overridden per test)
        median_footprint_area=90.0, building_count_density=3000.0,
        footprint_area_cv=0.5, buildings_per_road_km=100.0,
        road_length_density=10.0, f_google_source=0.8,
    )
    base.update(kw)
    return classify_patch(base, CFG)


def test_informal_to_hard7():
    d = _compact_low(median_footprint_area=40, building_count_density=8000,
                     footprint_area_cv=0.9, buildings_per_road_km=400, road_length_density=3)
    assert d["label_type"] == "hard" and d["lcz"] == 7 and d["lcz_set"] == [7]
    assert d["confidence"] <= CFG.router.lcz7_confidence_cap + 1e-9
    assert d["informal_morphology"] and d["road_deficit"]


def test_formal_terrace_to_hard3():
    d = _compact_low(median_footprint_area=140, building_count_density=1500,
                     footprint_area_cv=0.3, buildings_per_road_km=80,
                     road_length_density=12, f_google_source=0.9)
    assert d["label_type"] == "hard" and d["lcz"] == 3
    assert d["formal_morphology"] and not d["road_deficit"]


def test_blob_suspect_to_coarse():
    # Non-Google footprints, low count density, high bsf -> morphology unreadable
    d = _compact_low(f_google_source=0.2, building_count_density=1000,
                     median_footprint_area=300, bsf=0.5)
    assert d["label_type"] == "coarse" and d["lcz"] is None and d["lcz_set"] == [3, 7]
    assert d["blob_suspect"]


def test_ambiguous_to_coarse():
    d = _compact_low(median_footprint_area=80, building_count_density=3000,
                     footprint_area_cv=0.5, buildings_per_road_km=100,
                     road_length_density=8, f_google_source=0.8)
    assert d["label_type"] == "coarse" and d["lcz_set"] == [3, 7]
    assert not d["informal_morphology"] and not d["formal_morphology"]


def test_mn_corroboration_raises_cap():
    # Formal-looking morphology but Million Neighborhoods flags informal -> hard 7
    # at the higher MN cap.
    d = _compact_low(median_footprint_area=300, building_count_density=1000,
                     footprint_area_cv=0.2, f_google_source=0.9,
                     buildings_per_road_km=50, road_length_density=10, mn_informal_frac=0.8)
    assert d["label_type"] == "hard" and d["lcz"] == 7 and d["mn_informal"]
    assert d["confidence"] == pytest.approx(CFG.router.lcz7_confidence_cap_mn)


def test_missing_mn_degrades():
    # No MN column at all -> mn_informal False, router still works (here: informal).
    base = dict(bsf=0.5, h_mean=6, ghs_built_s=0.5, height_evidence_frac=0.4,
                median_footprint_area=40, building_count_density=8000,
                footprint_area_cv=0.9, buildings_per_road_km=400,
                road_length_density=3, f_google_source=0.8)
    d = classify_patch(base, CFG)
    assert d["mn_informal"] is False and d["lcz"] == 7


def test_height_evidence_waived_for_router():
    # Zero height evidence must NOT drop a router verdict (informal areas lack tags)
    d = _compact_low(median_footprint_area=40, building_count_density=8000,
                     footprint_area_cv=0.9, buildings_per_road_km=400,
                     road_length_density=3, height_evidence_frac=0.0, height_none_frac=0.9)
    assert d["lcz"] == 7 and d["reject_reason"] is None


def test_no_compact_low_bypasses_router():
    # Every compact+low+built patch is hard 3, hard 7, or coarse {3,7} — never
    # unlabelled for a height-evidence reason, never any other class.
    import itertools
    for medfp, cd, cv, gfrac in itertools.product(
        [30, 90, 200], [500, 3000, 9000], [0.2, 0.8], [0.2, 0.9]
    ):
        d = _compact_low(median_footprint_area=medfp, building_count_density=cd,
                         footprint_area_cv=cv, f_google_source=gfrac)
        assert d["label_type"] in ("hard", "coarse")
        assert set(d["lcz_set"]) <= {3, 7}


def test_coarse_8_10():
    # Industrial land-use, large lowrise geometry, but no heavy-industry POI
    d = classify_patch(dict(bsf=0.3, h_mean=8, mean_footprint_area=1200,
                            large_lowrise_frac=0.8, f_industrial_lu=0.6,
                            n_heavy_industry_poi=0, ghs_built_s=0.5,
                            height_evidence_frac=0.9), CFG)
    assert d["label_type"] == "coarse" and d["lcz_set"] == [8, 10]
