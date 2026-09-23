"""Suitability assessment helpers. Offline — synthetic rasters and frames."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from lcz_wudapt.leakage import SO2SAT_TEST_CITIES
from lcz_wudapt.suitability import (
    REGION_GROUPS,
    TIER1_CULTURE,
    TIER2_SPARSE,
    WEST_AFRICA_ISO,
    _cells_with_label,
    labelled_km2,
    resolve_city_aois,
    tile_footprint,
)


def _write_raster(path, arr, west=0.0, north=0.0, res=1e-4):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="uint8", crs="EPSG:4326", transform=from_origin(west, north, res, res),
        nodata=0,
    ) as dst:
        dst.write(arr, 1)
    return path


# ── Consistency with the leakage guard ───────────────────────────────────────

def test_tier1_is_exactly_the_culture_ten():
    """If these drift apart, the assessment silently stops covering the cities
    every headline kappa is measured on.
    """
    assert len(TIER1_CULTURE) == 10
    norm = {c.replace("_", " ") for c in TIER1_CULTURE}
    assert norm == set(SO2SAT_TEST_CITIES)


def test_tier2_does_not_intersect_the_culture_ten():
    """The sparse tier must be So2Sat *training* cities only — a held-out city
    appearing here would mean the sparsity rule could admit one.
    """
    assert not ({c.replace("_", " ") for c in TIER2_SPARSE} & set(SO2SAT_TEST_CITIES))


# ── Raster measures ──────────────────────────────────────────────────────────

def test_labelled_km2_counts_only_labelled_pixels(tmp_path):
    arr = np.zeros((100, 100), dtype="uint8")
    arr[:50, :] = 6          # half the raster, one class
    km2, ncls = labelled_km2(_write_raster(tmp_path / "a.tif", arr))
    assert ncls == 1
    # 5000 px at ~11.13 m on a side near the equator
    assert km2 == pytest.approx(5000 * (111_320 * 1e-4) ** 2 / 1e6, rel=1e-3)


def test_labelled_km2_reports_every_class_present(tmp_path):
    arr = np.zeros((10, 10), dtype="uint8")
    arr[0] = 1; arr[1] = 7; arr[2] = 17
    _, ncls = labelled_km2(_write_raster(tmp_path / "b.tif", arr))
    assert ncls == 3


def test_labelled_km2_is_zero_for_an_empty_raster(tmp_path):
    km2, ncls = labelled_km2(_write_raster(tmp_path / "c.tif", np.zeros((10, 10), "uint8")))
    assert km2 == 0.0 and ncls == 0


def test_cells_with_label_computes_the_right_fraction(tmp_path):
    arr = np.zeros((100, 100), dtype="uint8")
    arr[:25, :] = 3          # top quarter labelled
    tif = _write_raster(tmp_path / "d.tif", arr, west=0.0, north=0.0, res=1e-4)
    grid = gpd.GeoDataFrame(
        {"split": ["train"]},
        geometry=[box(0.0, -100 * 1e-4, 100 * 1e-4, 0.0)],   # the whole raster
        crs="EPSG:4326",
    )
    per = _cells_with_label(tif, grid)
    assert len(per) == 1
    assert per["frac"].iloc[0] == pytest.approx(0.25, abs=0.02)


def test_cells_with_label_separates_grid_splits(tmp_path):
    arr = np.full((100, 100), 6, dtype="uint8")
    tif = _write_raster(tmp_path / "e.tif", arr, res=1e-4)
    grid = gpd.GeoDataFrame(
        {"split": ["train", "test"]},
        geometry=[box(0.0, -50e-4, 100e-4, 0.0), box(0.0, -100e-4, 100e-4, -50e-4)],
        crs="EPSG:4326",
    )
    per = _cells_with_label(tif, grid)
    assert set(per["split"]) == {"train", "test"}
    assert (per["frac"] > 0.9).all()


# ── Tile footprint (sizing an embedding request) ─────────────────────────────

def test_tile_footprint_of_a_small_patch_is_one_tile():
    g = gpd.GeoDataFrame(geometry=[box(3.31, 6.51, 3.32, 6.52)], crs="EPSG:4326")
    assert len(tile_footprint(g)) == 1


def test_tile_footprint_centres_are_on_the_half_cell():
    """Tessera names tiles grid_{lon}_{lat} at the CENTRE of a 0.1 deg cell, so
    a footprint of cell corners would request tiles that do not exist.
    """
    g = gpd.GeoDataFrame(geometry=[box(3.31, 6.51, 3.32, 6.52)], crs="EPSG:4326")
    (lon, lat), = tile_footprint(g)
    assert lon == pytest.approx(3.35) and lat == pytest.approx(6.55)


def test_tile_footprint_grows_with_extent():
    small = gpd.GeoDataFrame(geometry=[box(0.01, 0.01, 0.02, 0.02)], crs="EPSG:4326")
    big = gpd.GeoDataFrame(geometry=[box(0.01, 0.01, 0.45, 0.45)], crs="EPSG:4326")
    assert len(tile_footprint(big)) > len(tile_footprint(small))


# ── Region selectors ─────────────────────────────────────────────────────────

def test_region_groups_select_disjoint_isos_where_expected():
    df = pd.DataFrame({
        "iso": ["NGA", "IND", "IDN", "CUB", "DEU"],
        "region": ["Africa", "Asia-South", "Asia-Southeast", "America-Central", "Europe"],
    })
    got = {name: set(df[sel(df)].iso) for name, sel in REGION_GROUPS.items()}
    assert got["West Africa"] == {"NGA"}
    assert got["India"] == {"IND"}
    assert got["Southeast Asia"] == {"IDN"}
    assert got["Central America"] == {"CUB"}


def test_west_africa_is_an_iso_set_not_a_coarse_region():
    """It sits inside splits.ISO_TO_REGION's 'Africa' bucket but behaves nothing
    like North or Southern Africa here, so it is carved out explicitly.
    """
    from lcz_wudapt.splits import ISO_TO_REGION

    assert {ISO_TO_REGION[i] for i in WEST_AFRICA_ISO if i in ISO_TO_REGION} == {"Africa"}
    assert "ZAF" not in WEST_AFRICA_ISO and "EGY" not in WEST_AFRICA_ISO


# ── City -> AOI resolution ───────────────────────────────────────────────────

def _make_roots(tmp_path, names):
    root = tmp_path / "cities"
    for n in names:
        (root / n).mkdir(parents=True)
    return root


def test_resolve_city_aois_folds_underscores_to_spaces(tmp_path, monkeypatch):
    from lcz_wudapt.config import WudaptConfig

    root = _make_roots(tmp_path, ["San_Jose__30_12257", "Buenos_Aires__30_207"])
    got = resolve_city_aois(WudaptConfig(), ("San_Jose", "Buenos_Aires"), cities_root=root)
    assert got == {"San_Jose": "San_Jose__30_12257", "Buenos_Aires": "Buenos_Aires__30_207"}


def test_resolve_city_aois_skips_unknown_cities(tmp_path):
    from lcz_wudapt.config import WudaptConfig

    root = _make_roots(tmp_path, ["Lagos__30_9893"])
    got = resolve_city_aois(WudaptConfig(), ("Lagos", "Atlantis"), cities_root=root)
    assert got == {"Lagos": "Lagos__30_9893"}


def test_resolve_city_aois_is_deterministic_when_ambiguous(tmp_path):
    from lcz_wudapt.config import WudaptConfig

    root = _make_roots(tmp_path, ["Santiago__30_1187", "Santiago__30_9999"])
    a = resolve_city_aois(WudaptConfig(), ("Santiago",), cities_root=root)
    b = resolve_city_aois(WudaptConfig(), ("Santiago",), cities_root=root)
    assert a == b and a["Santiago"] == "Santiago__30_1187"
