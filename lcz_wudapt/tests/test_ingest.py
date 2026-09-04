"""H1 ingest/cleaning invariants. Offline — no gpkg, no network."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Polygon, box

from lcz_wudapt.config import WudaptConfig
from lcz_wudapt.ingest import (
    EQUAL_AREA_CRS,
    RURAL_AOI,
    _equal_area_km2,
    aoi_key,
    assign_aoi,
    clean,
    normalise_qc,
    resolve_annotators,
    slug,
)


def _raw(n_extra_class: int = 0) -> gpd.GeoDataFrame:
    """Minimal frame with the columns clean() touches."""
    geoms = [box(0.0 + i * 0.01, 0.0, 0.005 + i * 0.01, 0.005) for i in range(4)]
    rows = {
        "submission_id": ["s1", "s1", "s2", "s3"],
        "submission_date": pd.to_datetime(
            ["2022-01-01", "2022-01-01", "2023-05-05", "2021-02-02"], utc=True
        ),
        "representative_date": ["2021-06-01", "2021-06-01", "2323-01-01", None],
        "city": ["wuhan", "武汉", "..", "x"],
        "reference": ["ref-a", "ref-a", "", ""],
        "cite_as": ["", "", "", ""],
        "firstname": ["Ada", "Ada", "", ""],
        "lastname": ["Lovelace", "Lovelace", "", ""],
        "version": ["1.0.0", "1.0.0", "2.1.5", "1.2.2"],
        "class": [1, 11, 3, 5],
        "area": [1.0, 1.0, 1.0, 1.0],
        "perimeter": [1.0] * 4,
        "shape": [1.0] * 4,
        "vertices": [5, 5, 5, 5],
        "qc_step1": ["True", "T", "False", "F"],
        "qc_step2": ["T", "True", "T", "True"],
        "qc_step3": ["True", "T", "True", "T"],
        "oa": [0.8] * 4,
        "oau": [0.7] * 4,
        "oabu": [0.7] * 4,
        "oaw": [0.7] * 4,
    }
    for c in range(1, 18):
        rows[f"f1_{c}"] = [0.5] * 4
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    if n_extra_class:
        extra = gdf.iloc[[0]].copy()
        extra["class"] = 18
        gdf = pd.concat([gdf, extra], ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    return gdf


def test_qc_mixed_encodings_normalise():
    """The shipped column mixes True/False with T/F; both must map identically."""
    s = pd.Series(["True", "T", "true", "False", "F", "f", "wat", None])
    out = normalise_qc(s)
    assert out.tolist()[:3] == [True, True, True]
    assert out.tolist()[3:6] == [False, False, False]
    # Unrecognised tokens become NA rather than a silent False.
    assert pd.isna(out.iloc[6]) and pd.isna(out.iloc[7])


def test_qc_normalise_is_idempotent_on_bool():
    s = pd.Series([True, False], dtype="bool")
    assert normalise_qc(s).tolist() == [True, False]


def test_classes_outside_1_17_are_dropped():
    """The 2024-10-01 release carries 580 rows of class 18 and 53 of 19."""
    out = clean(_raw(n_extra_class=1), WudaptConfig())
    assert out["class"].between(1, 17).all()
    assert len(out) == 4


def test_equal_area_matches_geodesic():
    """Area must not come from the shipped Mercator column (1.35x inflated)."""
    # A polygon at 60N, where Mercator inflation is ~4x.
    poly = Polygon([(10.0, 60.0), (10.05, 60.0), (10.05, 60.03), (10.0, 60.03)])
    g = gpd.GeoSeries([poly], crs="EPSG:4326")
    got = _equal_area_km2(g)[0]
    want = abs(pyproj.Geod(ellps="WGS84").geometry_area_perimeter(poly)[0]) / 1e6
    assert got == pytest.approx(want, rel=1e-3)
    # And it must differ sharply from the Web Mercator figure at this latitude.
    merc = g.to_crs("EPSG:3857").area.iloc[0] / 1e6
    assert merc > 3 * got


def test_clean_preserves_raw_area_under_a_different_name():
    out = clean(_raw(), WudaptConfig())
    assert "area_km2_mercator_raw" in out.columns
    assert "area_km2" in out.columns
    assert "area" not in out.columns


def test_implausible_label_years_fall_back_to_submission_year():
    """2323 is a typo; it must not become a real-looking epoch."""
    out = clean(_raw(), WudaptConfig())
    row = out[out["submission_id"] == "s2"].iloc[0]
    assert pd.isna(row["rep_year"])
    assert row["label_year"] == 2023          # from submission_date
    named = out[out["submission_id"] == "s1"].iloc[0]
    assert named["label_year"] == 2021        # representative_date wins over submission


def test_annotator_id_prefers_name_then_study_then_blank():
    out = clean(_raw(), WudaptConfig())
    by_sub = out.set_index("submission_id")
    assert by_sub.loc["s1", "annotator_src"].iloc[0] == "name"
    assert by_sub.loc["s2", "annotator_src"] == "blank"     # no name, no reference
    assert by_sub.loc["s1", "annotator_id"].iloc[0].startswith("name:lovelace|ada")


def test_blank_name_policy_collapses_per_aoi():
    """Per-submission keying invents more annotators than there are real authors."""
    cfg_collapse = WudaptConfig(blank_name_policy="collapse_per_aoi")
    cfg_per_sub = WudaptConfig(blank_name_policy="per_submission")
    cleaned = clean(_raw(), cfg_collapse)
    cleaned = cleaned.assign(aoi="CityA__30_1")

    collapsed = resolve_annotators(cleaned, cfg_collapse)
    per_sub = resolve_annotators(cleaned, cfg_per_sub)
    blank = cleaned["annotator_src"] == "blank"
    assert collapsed.loc[blank, "annotator_id"].nunique() == 1
    assert per_sub.loc[blank, "annotator_id"].nunique() == 2
    # Named authors are untouched by the policy either way.
    assert collapsed.loc[~blank, "annotator_id"].equals(per_sub.loc[~blank, "annotator_id"])


def test_blank_name_policy_is_part_of_the_ingest_cache_key():
    """Changing the policy changes the cached parquet, so it must change the hash."""
    a = WudaptConfig(blank_name_policy="collapse_per_aoi")
    b = WudaptConfig(blank_name_policy="per_submission")
    assert a.ingest_hash != b.ingest_hash


def test_aoi_key_is_smod_qualified_so_duplicate_names_stay_distinct():
    """JRC_NAME_MAIN repeats in guppd_bounds.csv (Leon x3, Aurangabad x3)."""
    assert aoi_key("León", "30_1") != aoi_key("León", "30_2")
    assert aoi_key("León", "30_1").endswith("__30_1")


def test_slug_keeps_non_ascii_names_distinct():
    """东营区 must not collapse to an empty string and collide with everything."""
    assert slug("东营区") != slug("Dongying")
    assert slug("东营区")
    assert "/" not in slug("a/b") and " " not in slug("San Cristóbal")


def test_assign_aoi_retains_rural_polygons(tmp_path):
    """27.4% of polygons fall outside every GUPPD bbox and are 61.8% natural class."""
    bounds = tmp_path / "bounds.csv"
    pd.DataFrame(
        {
            "SMOD_ID": ["30_1"],
            "JRC_NAME_MAIN": ["CityA"],
            "ISO": ["CHN"],
            "CNTRY_NAME": ["China"],
            "minx": [-0.001], "miny": [-0.001], "maxx": [0.006], "maxy": [0.006],
        }
    ).to_csv(bounds, index=False)
    cfg = WudaptConfig(bounds_csv=bounds)
    out = assign_aoi(clean(_raw(), cfg), cfg)
    assert (out["aoi"] == RURAL_AOI).sum() == 3      # only the first box is inside
    assert (out["aoi"] == aoi_key("CityA", "30_1")).sum() == 1
    assert len(out) == 4                              # nothing dropped


def test_assign_aoi_prefers_the_smaller_bbox(tmp_path):
    """GUPPD boxes nest; the smaller box is the more specific city."""
    bounds = tmp_path / "bounds.csv"
    pd.DataFrame(
        {
            "SMOD_ID": ["30_big", "30_small"],
            "JRC_NAME_MAIN": ["Big", "Small"],
            "ISO": ["CHN", "CHN"],
            "CNTRY_NAME": ["China", "China"],
            "minx": [-1.0, -0.001], "miny": [-1.0, -0.001],
            "maxx": [1.0, 0.006], "maxy": [1.0, 0.006],
        }
    ).to_csv(bounds, index=False)
    cfg = WudaptConfig(bounds_csv=bounds)
    out = assign_aoi(clean(_raw(), cfg), cfg)
    assert out.iloc[0]["aoi"] == aoi_key("Small", "30_small")
