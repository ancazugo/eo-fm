"""G1 -- the city-level (global) split for segmentation.

Offline: role assignment and per-tile split resolution are pure functions over
plain dicts and shapely geometries, so none of this needs the data mounts.

Three properties carry the weight, and each one is a leak if it breaks.

**Tile purity.** A 1.28 km tile is one training example. If it contains both a
train and a test patch, majority-assigning it puts test labels into training.
The rule is to drop it, and dropping has to be the behaviour under test because
"mostly train" is exactly the plausible-looking wrong answer.

**Culture cities contribute no training tiles.** Guangzhou is the one So2Sat
city carrying both `training` and `testing`/`validation` patches, so "the 42
training cities" and "the 10 culture cities" genuinely overlap by one. Left
alone it would appear in both sides of the split.

**Stratified, size-aware val-inner selection.** Held-out cities chosen for
early stopping have to span regions and have to be big enough for the signal to
mean anything -- picking Salvador, which has one grid tile, would look correct
and be useless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.city_split import (  # noqa: E402
    DATASET_TO_ROLE,
    SO2SAT_CULTURE_CITIES,
    assign_city_roles,
    continent_of,
    tile_splits_for_city,
)

# A 1280 m tile at 10 m/px, and patches sized like So2Sat's 320 m footprints.
TILE = box(0, 0, 1280, 1280)
FAR_TILE = box(100_000, 0, 101_280, 1280)


def _patch(x, y, cls=3, dataset="training", pid="000001", size=320):
    return (box(x, y, x + size, y + size), cls, dataset, pid)


def test_a_tile_with_train_and_test_patches_is_dropped():
    """The segmentation-specific leak: one test patch contaminates the tile."""
    id2patches = {0: [_patch(100, 100, dataset="validation"),
                      _patch(600, 600, dataset="testing")]}
    out, drops = tile_splits_for_city(
        "Nairobi", "culture", id2patches, {0: TILE}, buffer_km=0.0,
    )
    assert out == {}
    assert drops["mixed_split"] == 1


def test_a_pure_tile_survives_and_keeps_its_split():
    id2patches = {0: [_patch(100, 100, dataset="testing"),
                      _patch(600, 600, dataset="testing")]}
    out, _ = tile_splits_for_city(
        "Nairobi", "culture", id2patches, {0: TILE}, buffer_km=0.0,
    )
    assert out == {0: "test"}


def test_culture_city_training_patches_are_dropped():
    """Guangzhou is in both the 42 train cities and the 10 culture cities."""
    id2patches = {0: [_patch(100, 100, dataset="training")]}
    out, drops = tile_splits_for_city(
        "Guangzhou", "culture", id2patches, {0: TILE}, buffer_km=0.0,
    )
    assert out == {}
    assert drops["culture_train"] == 1


def test_the_same_tile_would_be_kept_for_a_training_pool_city():
    """Proves the drop above is about the city's role, not the patch data."""
    id2patches = {0: [_patch(100, 100, dataset="training")]}
    out, _ = tile_splits_for_city(
        "Berlin", "train", id2patches, {0: TILE}, buffer_km=0.0,
    )
    assert out == {0: "train"}


def test_a_val_inner_city_routes_its_training_patches_to_val():
    """Early stopping runs on held-out training-pool cities, not on the culture
    cities' own validation patches."""
    id2patches = {0: [_patch(100, 100, dataset="training")]}
    out, _ = tile_splits_for_city(
        "London", "val_inner", id2patches, {0: TILE}, buffer_km=0.0,
    )
    assert out == {0: "val"}


def test_culture_validation_tiles_get_their_own_split_label():
    """They are kept addressable because ensemble_stacking --city-holdout fits
    its combiner weights on exactly these patches."""
    id2patches = {0: [_patch(100, 100, dataset="validation")]}
    out, _ = tile_splits_for_city(
        "Munich", "culture", id2patches, {0: TILE}, buffer_km=0.0,
    )
    assert out == {0: "culture_val"}


def test_a_test_tile_near_a_val_patch_is_dropped_by_the_buffer():
    id2patches = {
        0: [_patch(100, 100, dataset="testing")],
        1: [_patch(100_100, 100, dataset="validation")],
    }
    geoms = {0: TILE, 1: FAR_TILE}
    near, _ = tile_splits_for_city(
        "Nairobi", "culture", id2patches, geoms, buffer_km=1000.0,
    )
    # With a 1000 km buffer both tiles see each other and both are dropped.
    assert near == {}
    far, _ = tile_splits_for_city(
        "Nairobi", "culture", id2patches, geoms, buffer_km=1.3,
    )
    # 100 km apart, so a 1.3 km buffer leaves both standing.
    assert far == {0: "test", 1: "culture_val"}


def test_a_sparsely_labelled_tile_is_dropped():
    """Otherwise batches are mostly ignore_index."""
    tiny = (box(0, 0, 32, 32), 3, "training", "000001")
    out, drops = tile_splits_for_city(
        "Berlin", "train", {0: [tiny]}, {0: TILE},
        buffer_km=0.0, min_labelled_frac=0.01,
    )
    assert out == {} and drops["sparse"] == 1
    kept, _ = tile_splits_for_city(
        "Berlin", "train", {0: [tiny]}, {0: TILE},
        buffer_km=0.0, min_labelled_frac=0.0,
    )
    assert kept == {0: "train"}


def test_a_tile_with_no_patches_is_dropped():
    out, drops = tile_splits_for_city(
        "Berlin", "train", {}, {0: TILE}, buffer_km=0.0,
    )
    assert out == {} and drops["no_labels"] == 1


def test_every_culture_city_resolves_a_continent():
    """City directories are ASCII-ified and underscored while geo_lookup uses
    the JRC spellings; comparing them raw silently drops cities."""
    for city in SO2SAT_CULTURE_CITIES:
        assert continent_of(city) != "Unknown", city
    for tricky in ("Sao_Paulo", "Osaka_[Kyoto]", "Dongying",
                   "Rawalpindi_[Islamabad]", "Washington_D.C."):
        assert continent_of(tricky) != "Unknown", tricky


def test_val_inner_cities_span_continents_and_exclude_culture_cities():
    cities = ["Berlin", "London", "Madrid", "Cairo", "Lima", "Melbourne",
              "Chicago", "Beijing", "Nairobi", "Munich", "Sydney"]
    roles = assign_city_roles(cities, n_val_inner=3, seed=42)
    val_inner = [c for c, r in roles.items() if r == "val_inner"]
    assert len(val_inner) == 3
    assert not set(val_inner) & set(SO2SAT_CULTURE_CITIES)
    # Round-robin over continents means 3 slots take 3 distinct regions.
    assert len({continent_of(c) for c in val_inner}) == 3
    assert all(roles[c] == "culture" for c in ("Nairobi", "Munich", "Sydney"))


def test_val_inner_selection_prefers_larger_cities():
    """Salvador has a single grid tile; stratification alone would still pick
    it, and the early-stopping signal would be noise."""
    cities = ["Salvador", "Sao_Paulo", "Berlin"]
    weights = {"Salvador": 1, "Sao_Paulo": 900, "Berlin": 500}
    roles = assign_city_roles(cities, n_val_inner=1, seed=42,
                              city_weights=weights)
    assert [c for c, r in roles.items() if r == "val_inner"] == ["Sao_Paulo"]


def test_role_assignment_is_deterministic():
    cities = ["Berlin", "London", "Madrid", "Cairo", "Lima", "Melbourne"]
    a = assign_city_roles(cities, n_val_inner=2, seed=7)
    b = assign_city_roles(cities, n_val_inner=2, seed=7)
    assert a == b


def test_asking_for_more_val_inner_cities_than_exist_is_an_error():
    with pytest.raises(ValueError, match="exceeds"):
        assign_city_roles(["Berlin", "Nairobi"], n_val_inner=5)


def test_the_dataset_role_map_matches_so2sat_split_names():
    assert DATASET_TO_ROLE == {
        "training": "train", "validation": "val", "testing": "test"
    }


def test_the_culture_city_list_agrees_with_lcz_wudapt():
    """utils.city_split keeps its own copy because `python src/...` puts src/
    on sys.path, not the repo root. The copies must not drift."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lcz_wudapt.leakage import SO2SAT_TEST_CITIES

    norm = lambda s: s.replace("_", " ")  # noqa: E731
    assert {norm(c) for c in SO2SAT_CULTURE_CITIES} == set(SO2SAT_TEST_CITIES)
