"""Restricting two embedding arms to the ground they share.

Offline: the filter is a pure function over item tuples and file existence, so
none of this needs the data mounts.

The ablation ladder's rows 4 and 5 ask one question -- does the embedding
matter -- and answer it by training the same U-Net on AlphaEarth and on
Tessera. That only works if both arms see the same tiles. In 2017 they do not:
AlphaEarth coop covers every valid tile in all 51 cities, while Tessera v1.1
global is short 993 across 14 coastal cities, 37 % of Qingdao and 32 % of
Istanbul. Those tiles are absent from the archive rather than unextracted, so
the intersection is the only common ground available.

The property under test is symmetry: after restriction, both arms must hold the
*same* cells, not merely similar counts. A filter that kept the right number
from the wrong cells would pass a count check and still confound the ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "semseg", Path(__file__).resolve().parents[1] / "src" / "semantic_segmentation.py"
)
semseg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(semseg)

TILE = box(0, 0, 1280, 1280)
YEAR = "2017"


def _make(root: Path, city: str, name: str, split: str, gid: int) -> Path:
    """Create {city}/{name}/{year}/{split}/{city}_{gid}.npy, matching the layout
    _restrict_to_common_tiles navigates by parents[3]."""
    p = root / city / name / YEAR / split / f"{city}_{gid:02d}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def _items(paths):
    return [(p, TILE, "EPSG:32637", [], None) for p in paths]


@pytest.fixture
def world(tmp_path):
    """coop holds cells 0-4; tessera holds only 0, 1, 4."""
    coop = [_make(tmp_path, "Qingdao", "AlphaEarthCoop", "train", g) for g in range(5)]
    for g in (0, 1, 4):
        _make(tmp_path, "Qingdao", "GeoTessera_v1.1_global", "train", g)
    tess = [tmp_path / "Qingdao" / "GeoTessera_v1.1_global" / YEAR / "train"
            / f"Qingdao_{g:02d}.npy" for g in (0, 1, 4)]
    return coop, tess


def test_the_richer_arm_is_cut_down_to_the_intersection(world):
    coop, _ = world
    kept, dropped, per_city = semseg._restrict_to_common_tiles(
        _items(coop), ["GeoTessera_v1.1_global"], YEAR)
    assert len(kept) == 3 and dropped == 2
    assert per_city == {"Qingdao": 2}


def test_the_poorer_arm_is_left_alone(world):
    _, tess = world
    kept, dropped, _ = semseg._restrict_to_common_tiles(
        _items(tess), ["AlphaEarthCoop"], YEAR)
    assert len(kept) == 3 and dropped == 0


def test_both_arms_end_up_on_the_same_cells_not_just_the_same_count(world):
    """The property that actually protects the ablation."""
    coop, tess = world
    k_coop, _, _ = semseg._restrict_to_common_tiles(
        _items(coop), ["GeoTessera_v1.1_global"], YEAR)
    k_tess, _, _ = semseg._restrict_to_common_tiles(
        _items(tess), ["AlphaEarthCoop"], YEAR)
    ids = lambda ks: {p.name for p, *_ in ks}  # noqa: E731
    assert ids(k_coop) == ids(k_tess) == {
        "Qingdao_00.npy", "Qingdao_01.npy", "Qingdao_04.npy"
    }


def test_restriction_is_idempotent(world):
    coop, _ = world
    once, _, _ = semseg._restrict_to_common_tiles(
        _items(coop), ["GeoTessera_v1.1_global"], YEAR)
    twice, dropped, _ = semseg._restrict_to_common_tiles(
        once, ["GeoTessera_v1.1_global"], YEAR)
    assert len(twice) == len(once) and dropped == 0


def test_requiring_several_embeddings_takes_the_full_intersection(tmp_path):
    coop = [_make(tmp_path, "Lisbon", "AlphaEarthCoop", "train", g) for g in range(4)]
    for g in (0, 1, 2):
        _make(tmp_path, "Lisbon", "GeoTessera_v1.1_global", "train", g)
    for g in (0, 2, 3):
        _make(tmp_path, "Lisbon", "EmbeddedSeamless", "train", g)
    kept, dropped, _ = semseg._restrict_to_common_tiles(
        _items(coop), ["GeoTessera_v1.1_global", "EmbeddedSeamless"], YEAR)
    assert {p.name for p, *_ in kept} == {"Lisbon_00.npy", "Lisbon_02.npy"}
    assert dropped == 2


def test_the_split_folder_is_part_of_the_identity(tmp_path):
    """A cell present in another arm's *val* folder must not satisfy a *train*
    requirement -- the grid split lives in the path."""
    train = _make(tmp_path, "Lisbon", "AlphaEarthCoop", "train", 7)
    _make(tmp_path, "Lisbon", "GeoTessera_v1.1_global", "val", 7)
    kept, dropped, _ = semseg._restrict_to_common_tiles(
        _items([train]), ["GeoTessera_v1.1_global"], YEAR)
    assert kept == [] and dropped == 1


def test_fused_items_are_keyed_on_their_first_source(tmp_path):
    p0 = _make(tmp_path, "Lisbon", "AlphaEarthCoop", "train", 3)
    p1 = _make(tmp_path, "Lisbon", "aux_struct", "train", 3)
    _make(tmp_path, "Lisbon", "GeoTessera_v1.1_global", "train", 3)
    items = [((p0, p1), TILE, "EPSG:32637", [], None)]
    kept, dropped, _ = semseg._restrict_to_common_tiles(
        items, ["GeoTessera_v1.1_global"], YEAR)
    assert len(kept) == 1 and dropped == 0


def test_per_city_counts_name_every_affected_city(tmp_path):
    paths = []
    for city, n_coop, n_tess in (("Qingdao", 4, 1), ("Istanbul", 3, 3)):
        paths += [_make(tmp_path, city, "AlphaEarthCoop", "train", g)
                  for g in range(n_coop)]
        for g in range(n_tess):
            _make(tmp_path, city, "GeoTessera_v1.1_global", "train", g)
    _kept, dropped, per_city = semseg._restrict_to_common_tiles(
        _items(paths), ["GeoTessera_v1.1_global"], YEAR)
    assert dropped == 3
    assert per_city == {"Qingdao": 3}          # Istanbul complete, so absent
