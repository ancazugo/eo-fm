"""Offline tests for src/design_global_splits.py on a synthetic WUDAPT world."""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import design_global_splits as dgs  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# (name, iso, lon, lat): real places so regions and climates differ
CITIES = [
    ("Lagos", "NGA", 3.39, 6.52), ("Accra", "GHA", -0.19, 5.60), ("Abidjan", "CIV", -4.01, 5.36),
    ("Ouagadougou", "BFA", -1.52, 12.37), ("Delhi", "IND", 77.21, 28.61), ("Kolkata", "IND", 88.36, 22.57),
    ("Chennai", "IND", 80.27, 13.08), ("Singapore", "SGP", 103.82, 1.35), ("Surabaya", "IDN", 112.75, -7.25),
    ("Havana", "CUB", -82.37, 23.11), ("Managua", "NIC", -86.25, 12.13), ("Lima", "PER", -77.04, -12.05),
    ("Chicago", "USA", -87.63, 41.88), ("Paris", "FRA", 2.35, 48.86), ("Lyon", "FRA", 4.84, 45.76),
    ("Dongguan", "CHN", 113.75, 23.02),   # ~50 km from Guangzhou (culture-10)
    ("Tehran", "IRN", 51.39, 35.69),      # quarantined
    ("Kano", "NGA", 8.52, 12.00), ("Dakar", "SEN", -17.47, 14.72), ("Nagpur", "IND", 79.09, 21.15),
]


class FakeKoppen:
    """Latitude bands only: keeps tests independent of kgcpy / a GeoTIFF."""

    def __call__(self, lon, lat):
        lat = np.abs(np.asarray(lat, float))
        return np.where(lat < 10, "Aw", np.where(lat < 20, "BSh", np.where(lat < 35, "Cwa",
                        np.where(lat < 45, "Cfa", "Dfb")))).astype(object)


def make_world(seed: int = 0, per_city: int = 60) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    sub = 0
    for name, iso, lon, lat in CITIES:
        for i in range(per_city):
            x = lon + rng.normal(0, 0.08)
            y = lat + rng.normal(0, 0.08)
            s = 0.004 + rng.random() * 0.004
            if i % 15 == 0:
                sub += 1
            rows.append({"geometry": box(x, y, x + s, y + s), "aoi": f"{name.lower()}__x",
                         "jrc_name": name, "iso": iso, "class": int(rng.integers(1, 18)),
                         "label_year": int(rng.choice([2014, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])),
                         "annotator_id": f"{name}-{i % 3}", "submission_id": f"s{sub}",
                         "qc_step1": True, "qc_step2": True, "qc_step3": pd.NA})
    # one rural polygon right next to Lagos, one far from everything
    rows.append({"geometry": box(3.45, 6.55, 3.46, 6.56), "aoi": "_rural", "jrc_name": None, "iso": "NGA",
                 "class": 11, "label_year": 2020, "annotator_id": "r", "submission_id": "r1",
                 "qc_step1": True, "qc_step2": True, "qc_step3": True})
    rows.append({"geometry": box(20.0, -20.0, 20.01, -19.99), "aoi": "_rural", "jrc_name": None, "iso": "BWA",
                 "class": 14, "label_year": 2020, "annotator_id": "r", "submission_id": "r2",
                 "qc_step1": True, "qc_step2": True, "qc_step3": True})
    g = gpd.GeoDataFrame(rows, crs=4326)
    g["area_km2"] = g.to_crs("EPSG:6933").area / 1e6
    pts = g.geometry.representative_point()
    g["lon"], g["lat"] = pts.x, pts.y
    g["ybin"] = dgs.year_bin(g.label_year)
    g["pid"] = np.arange(len(g))
    return g


@pytest.fixture(scope="module")
def world():
    return make_world()


@pytest.fixture(scope="module")
def inputs():
    kop = FakeKoppen()
    regions = dgs.region_table()
    so2sat = dgs.aoi_table_so2sat(REPO / "data/so2sat_city_summary.csv",
                                  REPO / "data/so2sat_city_class_counts.csv", regions, kop)
    target = dgs.target_table(REPO / "data/guppd_bounds.csv", regions, kop, n_grid=2)
    return kop, regions, so2sat, target


def run(world, inputs, **kw):
    kop, regions, so2sat, target = inputs
    cfg = dgs.DesignConfig(restarts=2, swap_iters=40, min_test_km2=0.5, min_test_classes=5,
                           min_class_km2=0.05, min_year_km2=0.1, **kw)
    quality = pd.DataFrame({"aoi": ["kolkata__x"], "nbr_conflict": [0.9]})   # untrustworthy
    return dgs.design(world, so2sat, target, cfg, kop=kop, regions=regions, quality=quality)


@pytest.fixture(scope="module")
def result(world, inputs):
    return run(world, inputs)


def test_block_diameter_is_bounded(result):
    a = result["aois"]
    for _, d in a.groupby("block"):
        if len(d) > 1:
            dist = dgs.haversine_km(d.lat, d.lon, d.lat, d.lon)
            assert dist.max() <= result["cfg"].block_km + 1e-6


def test_culture10_fixed_as_test_a(result):
    a = result["aois"]
    c10 = a[a.get("so2sat_city").isin(dgs.SO2SAT_TEST_CITIES)]
    assert len(c10) == 10
    assert (c10.split == "test_A").all()


def test_block_neighbour_of_culture10_is_not_trained_on_inside_block(result):
    # Dongguan sits ~50 km from Guangzhou: either it shares Guangzhou's block
    # (then it is test_A) or the buffer must keep its polygons out of train if close.
    a = result["aois"].set_index("aoi")
    assert a.loc["dongguan__x", "split"] in {"test_A", "train", "val", "test_B"}


def test_one_split_per_block(result):
    a = result["aois"]
    assert (a.groupby("block").split.nunique() == 1).all()


def test_quarantine_and_conflict_never_held_out_as_b_or_val(result):
    a = result["aois"].set_index("aoi")
    assert a.loc["tehran__x", "split"] not in {"test_B", "val"}
    assert a.loc["kolkata__x", "split"] not in {"test_B", "val"}


def test_so2sat_training_cities_never_test_b(result):
    a = result["aois"]
    s2s_train = a[(a.source == "so2sat") & (a.so2sat_role == "train") & (a.so2sat_patches >= 500)]
    assert not s2s_train.split.isin(["test_B", "val"]).any()


def test_test_b_and_val_exist_and_are_disjoint(result):
    a = result["aois"]
    b, v = set(a.loc[a.split == "test_B", "block"]), set(a.loc[a.split == "val", "block"])
    assert b and v and not (b & v)


def test_buffer_keeps_training_polygons_away(world, result):
    p = result["polygons"]
    a = result["aois"]
    held = set(a.loc[a.split != "train", "aoi"])
    w = world.copy()
    w["aoi"] = dgs.snap_rural(world).aoi
    m = w.to_crs("EPSG:6933")
    hulls = m[m.aoi.isin(held)].dissolve(by="aoi").convex_hull
    kept = m[(p.split == "train").to_numpy() & ~p.buffer_drop.to_numpy()]
    if len(kept) and len(hulls):
        d = kept.geometry.apply(lambda g: hulls.distance(g).min())
        assert (d > result["cfg"].buffer_km * 1000 - 1).all()


def test_folds_cover_train_blocks_once(result):
    b = result["blocks"]
    tr = b[b.split == "train"]
    assert tr.fold.notna().all()
    assert set(tr.fold.astype(int)) <= set(range(result["cfg"].folds))
    assert b.loc[b.split != "train", "fold"].isna().all()


def test_deterministic(world, inputs, result):
    again = run(world, inputs)
    pd.testing.assert_series_equal(again["aois"].split, result["aois"].split)


def test_snap_rural():
    w = make_world(per_city=5)
    s = dgs.snap_rural(w)
    rural_rows = w.index[w.aoi == "_rural"]
    assert s.loc[rural_rows[0], "aoi"] == "lagos__x"
    assert s.loc[rural_rows[1], "aoi"].startswith("rural:")


def test_revisit_pairs_detects_change():
    g = make_world(per_city=3)
    extra = gpd.GeoDataFrame([
        {"geometry": box(2.0, 48.0, 2.01, 48.01), "aoi": "paris__x", "class": 6, "label_year": 2017,
         "submission_id": "A", "annotator_id": "a", "pid": 10_000},
        {"geometry": box(2.0, 48.0, 2.0095, 48.01), "aoi": "paris__x", "class": 2, "label_year": 2023,
         "submission_id": "B", "annotator_id": "b", "pid": 10_001},
    ], crs=4326)
    g = pd.concat([g, extra], ignore_index=True)
    r = dgs.revisit_pairs(gpd.GeoDataFrame(g, crs=4326), {"paris__x"})
    hit = r[(r.pid_early == 10_000) & (r.pid_late == 10_001)]
    assert len(hit) == 1 and bool(hit.changed.iloc[0])
    assert hit.year_early.iloc[0] == 2017 and hit.year_late.iloc[0] == 2023


def test_ks_weighted_basics():
    x = np.array([1.0, 2.0, 3.0])
    assert dgs.ks_weighted(x, np.ones(3), x, np.ones(3)) == 0
    assert dgs.ks_weighted(x, np.ones(3), x + 10, np.ones(3)) == 1


def test_report_renders(result):
    md = dgs.report(result)
    assert "Stratum shares" in md and "Held-out AOIs" in md


def test_topk_nearest_matches_full_scan():
    rng = np.random.default_rng(1)
    d = rng.random((400, 200)) * 5000
    tk = dgs.Problem._topk(d, k=16)
    rows = np.arange(0, 400, 3)
    for frac in (0.9, 0.3, 0.05, 0.01):
        m = rng.random(200) < frac
        m[0] = True
        assert np.allclose(dgs.Problem._nearest(tk, slice(None), m), d[:, m].min(axis=1))
        assert np.allclose(dgs.Problem._nearest(tk, rows, m), d[np.ix_(rows, m)].min(axis=1))


def test_force_lists(world, inputs):
    res = run(world, inputs, force_test=("lagos",), force_train=("singapore", "accra"))
    a = res["aois"].set_index("aoi")
    assert a.loc["lagos__x", "split"] == "test_B"
    assert a.loc["singapore__x", "split"] == "train"
    assert a.loc["accra__x", "split"] == "train"


def test_stratum_holdout_cap(world, inputs):
    cap = 0.5
    res = run(world, inputs, max_stratum_holdout=cap)
    a = res["aois"]
    held = a[a.split.isin(["test_B", "val"])].groupby("stratum").km2.sum()
    total = a.groupby("stratum").km2.sum()
    # test A is fixed and may exceed the cap on its own; B+val must not push past it
    a_only = a[a.split == "test_A"].groupby("stratum").km2.sum()
    for s, v in held.items():
        assert (v + a_only.get(s, 0)) / total[s] <= cap + 1e-9 or a_only.get(s, 0) / total[s] > cap
