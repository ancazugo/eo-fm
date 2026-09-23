"""WUDAPT split assignment. The leakage guards are the point. Offline."""

from __future__ import annotations

import pandas as pd
import pytest

from lcz_wudapt.leakage import SO2SAT_TEST_CITIES
from lcz_wudapt.splits import (
    ISO_TO_REGION,
    assert_split_integrity,
    assign_splits,
    region_for,
)


def _index(n: int = 120) -> pd.DataFrame:
    isos = ["CHN", "IND", "KEN", "BRA", "DEU", "USA", "IDN", "TUR", "AUS", "MEX"]
    return pd.DataFrame(
        {
            "aoi": [f"city{i}__{i}" for i in range(n)],
            "jrc_name": [f"City{i}" for i in range(n)],
            "iso": [isos[i % len(isos)] for i in range(n)],
            "n_polys": [100 + i for i in range(n)],
        }
    )


def test_every_guppd_iso_code_maps_to_a_region():
    """lcz_train.splits.city_regions() raises on unmapped countries, and the
    roster here is 173 countries rather than So2Sat's handful. An unmapped code
    would silently collapse the stratification.
    """
    b = pd.read_csv("data/guppd_bounds.csv")
    unmapped = sorted({c for c in b.ISO.dropna().astype(str).unique() if c not in ISO_TO_REGION})
    assert unmapped == []


def test_unknown_iso_is_bucketed_not_silently_merged():
    assert region_for("ZZZ") == "Unknown"
    assert region_for(None) == "Unknown"


def test_splits_are_city_disjoint():
    s = assign_splits(_index())
    by_split = s.groupby("wudapt_split")["aoi"].apply(set)
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["train"] & by_split["val"])
    assert not (by_split["val"] & by_split["test"])


def test_every_split_is_non_empty():
    """--split-col raises ValueError if any of train/val/test ends up empty."""
    s = assign_splits(_index())
    assert set(s.wudapt_split) == {"train", "val", "test"}


def test_each_region_is_represented_in_test():
    """Unstratified, China's dominance would put most of the test set on one
    continent. Every region with enough AOIs must contribute.
    """
    s = assign_splits(_index())
    big = s.groupby("region").filter(lambda g: len(g) >= 3)
    for region, grp in big.groupby("region"):
        assert (grp.wudapt_split == "test").any(), region


def test_culture_cities_are_forced_to_test_via_the_so2sat_mapping():
    idx = _index(60)
    idx.loc[0, "aoi"] = "tehran__x"
    idx.loc[1, "aoi"] = "guangzhou__y"
    mapping = {"tehran__x": "Tehran", "guangzhou__y": "Guangzhou"}
    s = assign_splits(idx, so2sat_aoi_map=mapping)
    forced = s[s.aoi.isin(mapping)]
    assert bool(forced.forced_test.all())
    assert set(forced.wudapt_split) == {"test"}


def test_culture_cities_are_caught_by_name_when_unmapped():
    """GUPPD names and So2Sat directory names disagree, so the fallback matters."""
    idx = _index(60)
    idx.loc[0, "jrc_name"] = "Nairobi"
    idx.loc[1, "jrc_name"] = "San_Jose"
    s = assign_splits(idx)
    assert s.loc[0, "wudapt_split"] == "test"
    assert s.loc[1, "wudapt_split"] == "test"


def test_all_ten_culture_cities_are_recognised():
    n = len(SO2SAT_TEST_CITIES)
    idx = _index(80)
    for i, city in enumerate(SO2SAT_TEST_CITIES):
        idx.loc[i, "jrc_name"] = city
    s = assign_splits(idx)
    assert int(s.forced_test.sum()) == n
    assert (s.loc[s.forced_test, "wudapt_split"] == "test").all()


def test_integrity_guard_rejects_a_leaked_culture_city():
    idx = _index(60)
    idx.loc[0, "jrc_name"] = "Mumbai"
    s = assign_splits(idx)
    s.loc[0, "wudapt_split"] = "train"        # simulate a downstream mistake
    with pytest.raises(AssertionError, match="leaked into WUDAPT"):
        assert_split_integrity(s)


def test_integrity_guard_passes_a_clean_assignment():
    assert_split_integrity(assign_splits(_index()))


def test_assignment_is_stable_when_the_roster_grows():
    """New Tessera tiles will add AOIs later. If that reshuffled the existing
    split, no two experiments would be comparable.
    """
    small = _index(60)
    big = _index(120)
    a = assign_splits(small).set_index("aoi")["wudapt_split"]
    b = assign_splits(big).set_index("aoi")["wudapt_split"]
    shared = a.index.intersection(b.index)
    # Region membership is unchanged, so the per-region hash ordering is too;
    # only the cut points move, and they move monotonically.
    agreement = (a.loc[shared] == b.loc[shared]).mean()
    assert agreement > 0.8


def test_assignment_is_deterministic():
    assert assign_splits(_index()).wudapt_split.tolist() == assign_splits(_index()).wudapt_split.tolist()
