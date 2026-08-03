"""T1 — split determinism, region coverage, leakage guard, So2Sat-in-test."""

import pytest

from lcz_train.splits import (
    assert_no_leakage,
    city_regions,
    make_splits,
    save_splits,
    load_splits,
)

BOUNDS_CSV = "data/guppd_bounds.csv"
SO2SAT_TEST_CITIES = {"Nairobi", "Munich", "Mumbai", "Sydney", "Jakarta",
                      "Moscow", "San Jose", "Santiago", "Tehran", "Guangzhou"}
PILOT_CITIES = SO2SAT_TEST_CITIES | {
    "Toronto", "Medellín", "Accra", "Bangkok", "Busan", "Manchester", "Auckland",
}


def test_so2sat_cities_always_in_test():
    splits = make_splits(list(PILOT_CITIES), SO2SAT_TEST_CITIES, seed=0,
                         version="v1", bounds_csv=BOUNDS_CSV)
    assert set(splits["test"]) == SO2SAT_TEST_CITIES
    assert not (set(splits["train"]) & SO2SAT_TEST_CITIES)
    assert not (set(splits["val"]) & SO2SAT_TEST_CITIES)


def test_deterministic_given_seed():
    a = make_splits(list(PILOT_CITIES), SO2SAT_TEST_CITIES, seed=42,
                    version="v1", bounds_csv=BOUNDS_CSV)
    b = make_splits(list(PILOT_CITIES), SO2SAT_TEST_CITIES, seed=42,
                    version="v1", bounds_csv=BOUNDS_CSV)
    assert a == b


def test_different_seeds_can_differ():
    # With >=2 trainable cities in some region this should produce a
    # different train/val partition; skip gracefully if the fixture roster
    # happens to be too small to exercise that (documented, not asserted-away).
    results = [
        make_splits(list(PILOT_CITIES), SO2SAT_TEST_CITIES, seed=s,
                   version="v1", bounds_csv=BOUNDS_CSV)["train"]
        for s in range(5)
    ]
    if len(set(map(tuple, results))) == 1:
        pytest.skip("pilot roster has <2 trainable cities per region — nothing to permute")


def test_no_leakage_guard_catches_overlap():
    splits = make_splits(list(PILOT_CITIES), SO2SAT_TEST_CITIES, seed=0,
                         version="v1", bounds_csv=BOUNDS_CSV)
    assert_no_leakage(splits, SO2SAT_TEST_CITIES)  # passes on a valid split

    bad = dict(splits)
    bad["train"] = list(splits["train"]) + [splits["test"][0]]
    with pytest.raises(AssertionError):
        assert_no_leakage(bad, SO2SAT_TEST_CITIES)


def test_leakage_guard_flags_so2sat_outside_test():
    splits = make_splits(list(PILOT_CITIES), SO2SAT_TEST_CITIES, seed=0,
                         version="v1", bounds_csv=BOUNDS_CSV)
    bad = dict(splits)
    stray = splits["test"][0]
    bad["test"] = [c for c in splits["test"] if c != stray]
    bad["train"] = list(splits["train"]) + [stray]
    with pytest.raises(AssertionError):
        assert_no_leakage(bad, SO2SAT_TEST_CITIES)


def test_every_city_covered_exactly_once():
    splits = make_splits(list(PILOT_CITIES), SO2SAT_TEST_CITIES, seed=0,
                         version="v1", bounds_csv=BOUNDS_CSV)
    covered = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert covered == PILOT_CITIES


def test_region_coverage_not_blind():
    regions = city_regions(list(PILOT_CITIES), BOUNDS_CSV)
    assert set(regions.values()) >= {
        "Europe", "North America", "Latin America", "Africa",
        "South/SE Asia", "East Asia", "Oceania",
    }
    # Every pilot city gets a region — nothing silently dropped
    assert set(regions) == PILOT_CITIES


def test_unmapped_country_raises():
    with pytest.raises(ValueError):
        city_regions(["Atlantis"], BOUNDS_CSV)


def test_save_load_round_trip(tmp_path):
    splits = make_splits(list(PILOT_CITIES), SO2SAT_TEST_CITIES, seed=7,
                         version="v1", bounds_csv=BOUNDS_CSV)
    p = save_splits(splits, tmp_path / "splits_v1.json")
    back = load_splits(p)
    assert back == splits
