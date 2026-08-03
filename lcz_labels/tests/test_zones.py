"""Stage 5c — zone formation: dissolve, zone_grade, coarse-set identity."""

import pandas as pd

from lcz_labels.classify import form_zones
from lcz_labels.config import LczLabelConfig


def _labels():
    return pd.DataFrame({
        "block_id": list("abcdefgh") + ["u"],
        "label_type": ["hard"] * 4 + ["coarse", "coarse", "hard", "coarse", "unlabelled"],
        "lcz": [6, 6, 6, 6, None, None, 2, None, None],
        "lcz_set": [[6], [6], [6], [6], [3, 7], [3, 7], [2], [8, 10], []],
    })


def _blocks():
    # a+b+c = 18 ha (zone-grade); d alone = 5 ha; e+f = 16 ha; g = 3 ha; h = 20 ha
    areas = {"a": 6e4, "b": 6e4, "c": 6e4, "d": 5e4, "e": 8e4,
             "f": 8e4, "g": 3e4, "h": 20e4, "u": 1e4}
    return pd.DataFrame({"block_id": list(areas), "area_m2": list(areas.values())})


def _adjacency():
    # a-b-c chain; d touches g (different label); e-f same coarse set;
    # e-h coarse but DIFFERENT sets; a-g different labels; u touches a.
    return pd.DataFrame({
        "block_a": ["a", "b", "d", "e", "e", "a", "a"],
        "block_b": ["b", "c", "g", "f", "h", "g", "u"],
        "shared_len_m": [100.0] * 7,
    })


def test_zone_dissolve_and_grade():
    cfg = LczLabelConfig()
    z = form_zones(_labels(), _blocks(), _adjacency(), cfg).set_index("block_id")

    # a,b,c dissolve into one 18 ha zone -> zone-grade
    assert z.loc["a", "zone_id"] == z.loc["b", "zone_id"] == z.loc["c", "zone_id"] == "a"
    assert abs(z.loc["a", "zone_area_ha"] - 18.0) < 1e-9
    assert z.loc[["a", "b", "c"], "zone_grade"].all()

    # d: same hard 6 but not adjacent to the chain -> own 5 ha zone, not graded
    assert z.loc["d", "zone_id"] == "d"
    assert not z.loc["d", "zone_grade"]

    # e,f: identical coarse {3,7} -> one 16 ha zone, graded
    assert z.loc["e", "zone_id"] == z.loc["f", "zone_id"] == "e"
    assert z.loc[["e", "f"], "zone_grade"].all()

    # h: coarse {8,10} adjacent to e but a DIFFERENT set -> not dissolved with e,
    # and 20 ha alone -> zone-grade on its own
    assert z.loc["h", "zone_id"] == "h"
    assert z.loc["h", "zone_grade"]

    # g: hard 2, no same-label neighbour, 3 ha -> not graded
    assert z.loc["g", "zone_id"] == "g"
    assert not z.loc["g", "zone_grade"]

    # unlabelled: null zone columns (pandas' string dtype uses its own NA
    # marker, not Python None, so check with pd.isna())
    assert pd.isna(z.loc["u", "zone_id"])
    assert pd.isna(z.loc["u", "zone_area_ha"])
    assert not z.loc["u", "zone_grade"]


def test_min_zone_area_is_config_driven():
    cfg = LczLabelConfig()
    cfg.zones.min_zone_area_ha = 4.0
    z = form_zones(_labels(), _blocks(), _adjacency(), cfg).set_index("block_id")
    assert z.loc["d", "zone_grade"]        # 5 ha clears a 4 ha bar
    assert not z.loc["g", "zone_grade"]    # 3 ha still below


def test_all_unlabelled_degrades():
    cfg = LczLabelConfig()
    labels = _labels()
    labels["label_type"] = "unlabelled"
    labels["lcz"] = None
    labels["lcz_set"] = [[] for _ in range(len(labels))]
    z = form_zones(labels, _blocks(), _adjacency(), cfg)
    assert z["zone_id"].isna().all()
    assert not z["zone_grade"].any()
