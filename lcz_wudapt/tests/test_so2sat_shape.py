"""So2Sat-shaped artefacts from QC'd WUDAPT polygons. Offline."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import box

from lcz_wudapt.config import WudaptConfig
from lcz_wudapt.qc import apply_qc
from lcz_wudapt.so2sat_shape import (
    _SO2SAT_COLS,
    build_label_raster,
    build_patches,
    place_patches,
    write_city_dir,
)

_UTM = "EPSG:32633"


def _sq(cx: float, cy: float, side: float):
    h = side / 2.0
    return box(cx - h, cy - h, cx + h, cy + h)


# ── Patch placement geometry ─────────────────────────────────────────────────

def test_a_320m_square_does_not_fit_in_a_300m_polygon():
    assert place_patches(_sq(0, 0, 300.0), 320.0) == []


def test_erosion_alone_would_wrongly_admit_a_square():
    """buffer(-160) non-empty means a 320 m *disc* fits, not a 320 m *square*
    (which needs radius 226 m). The erosion is only a necessary prefilter, so
    containment is verified exactly -- a disc of diameter 320 m admits no
    axis-aligned 320 m square.
    """
    disc = _sq(0, 0, 1.0).buffer(160.0)              # radius 160 m
    assert not disc.buffer(-160.0 + 1e-6).is_empty   # the prefilter admits it
    assert place_patches(disc, 320.0) == []          # exact containment rejects it


def test_every_emitted_patch_lies_inside_its_source_polygon():
    poly = _sq(0, 0, 1500.0)
    for sq in place_patches(poly, 320.0):
        assert sq.within(poly)


def test_patches_are_exactly_the_requested_side():
    for sq in place_patches(_sq(0, 0, 1000.0), 320.0):
        minx, miny, maxx, maxy = sq.bounds
        assert (maxx - minx) == pytest.approx(320.0)
        assert (maxy - miny) == pytest.approx(320.0)


def test_patch_centres_are_lattice_aligned_so_neighbours_tile():
    """Two adjacent polygons must produce patches on the same lattice, otherwise
    they half-overlap at arbitrary offsets and dominant_frac loses meaning.
    """
    a = place_patches(_sq(0, 0, 1300.0), 320.0)
    b = place_patches(_sq(2000, 0, 1300.0), 320.0)
    cx = [s.bounds[0] for s in a + b]
    offsets = {round(x % 320.0, 6) for x in cx}
    assert len(offsets) == 1


def test_per_polygon_cap_is_enforced():
    """The anti-water-domination lever: one huge lake must not supply hundreds
    of patches. Uncapped a 3 km square yields many; capped it yields 4.
    """
    big = _sq(0, 0, 3000.0)
    assert len(place_patches(big, 320.0)) > 4
    assert len(place_patches(big, 320.0, max_patches=4)) == 4


def test_cap_keeps_the_most_interior_patches():
    poly = _sq(0, 0, 3000.0)
    kept = place_patches(poly, 320.0, max_patches=2)
    rp = poly.representative_point()
    far = max(abs(s.centroid.x - rp.x) for s in kept)
    assert far < 500.0


def test_a_thin_core_still_yields_a_patch_via_the_fallback():
    """A polygon can admit a 320 m square while its eroded core misses every
    lattice point. Losing it to lattice phase would be an arbitrary data loss.
    """
    poly = box(5.0, 5.0, 5.0 + 330.0, 5.0 + 330.0)
    assert len(place_patches(poly, 320.0)) >= 1


# ── The So2Sat contract ──────────────────────────────────────────────────────

def _qc_frame(classes=(1, 14, 17), side=900.0, spacing=8000.0, sides=None) -> gpd.GeoDataFrame:
    sides = list(sides) if sides is not None else [side] * len(classes)
    geoms = [_sq(i * spacing, 0, sd) for i, sd in enumerate(sides)]
    cols = {f"f1_{c}": [0.8] * len(classes) for c in range(1, 18)}
    gdf = gpd.GeoDataFrame(
        {
            "class": list(classes),
            "oa": [0.85] * len(classes),
            "oau": [0.8] * len(classes),
            "qc_step1": [True] * len(classes),
            "label_year": pd.array([2020] * len(classes), dtype="Int16"),
            "submission_id": [f"s{i}" for i in range(len(classes))],
            "submission_date": pd.to_datetime(["2022-01-01"] * len(classes), utc=True),
            "aoi": ["testcity"] * len(classes),
            **cols,
        },
        geometry=geoms, crs=_UTM,
    ).to_crs("EPSG:4326")
    return apply_qc(gdf, WudaptConfig())


def test_patch_gpkg_carries_the_so2sat_columns_in_order():
    out = build_patches(_qc_frame(), WudaptConfig())
    assert list(out.columns)[: len(_SO2SAT_COLS)] == list(_SO2SAT_COLS)


def test_patch_gpkg_is_epsg_4326_like_so2sat():
    assert build_patches(_qc_frame(), WudaptConfig()).crs.to_epsg() == 4326


def test_patch_ids_are_unique_and_aoi_qualified():
    """patch_id must be globally unique, not per-AOI: every patch is extracted to
    {root}/{dataset}/{output_name}/{year}/patch_{id}.npy, one directory shared by
    all cities. A per-AOI counter would have one city overwrite another -- the
    global-split patch_id collision this repo already fixed once, in 2026-06.
    """
    a = build_patches(_qc_frame(), WudaptConfig())
    assert a.patch_id.is_unique
    assert all(p.startswith("testcity_") for p in a.patch_id)

    other = _qc_frame()
    other = other.assign(aoi="othercity")
    b = build_patches(other, WudaptConfig())
    assert not (set(a.patch_id) & set(b.patch_id))


def test_dataset_column_names_a_real_split_directory():
    """`dataset` selects the embedding sub-directory in build_patch_index, so it
    has to be one of the four So2Sat split folders, not a free-text tag.
    """
    out = build_patches(_qc_frame(), WudaptConfig())
    assert set(out.dataset) <= {"training", "validation", "testing", "unlabeled"}


def test_patches_measure_320m_on_the_ground_after_reprojection():
    out = build_patches(_qc_frame(), WudaptConfig())
    m = out.to_crs(out.estimate_utm_crs()).geometry.bounds
    assert (m.maxx - m.minx).mean() == pytest.approx(320.0, abs=1.0)
    assert (m.maxy - m.miny).mean() == pytest.approx(320.0, abs=1.0)


def test_provenance_columns_survive_for_post_hoc_sweeps():
    out = build_patches(_qc_frame(), WudaptConfig())
    for col in ("weight", "nbr_dist_m", "nbr_conflict", "label_year",
                "embedding_year", "w_time", "src_area_km2", "oa"):
        assert col in out.columns, col


def test_polygons_failing_qc_contribute_no_patches():
    cfg = WudaptConfig()
    qc = _qc_frame()
    qc = qc.copy()
    qc["qc_pass"] = False
    assert len(build_patches(qc, cfg)) == 0


def test_oversize_reduction_caps_a_huge_natural_polygon():
    """A 3 km lake must not out-supply a 900 m built polygon. This is the
    mechanism that took water from 42% of an earlier pool down to single digits:
    uncapped, the lake alone yields dozens of patches.
    """
    cfg = WudaptConfig()
    out = build_patches(_qc_frame(classes=(17, 1), sides=(3000.0, 900.0)), cfg)
    water = int((out.LCZ_class == 17).sum())
    built = int((out.LCZ_class == 1).sum())
    assert len(place_patches(_sq(0, 0, 3000.0), 320.0)) > 20   # uncapped yield
    assert water <= cfg.qc.max_patches_per_polygon
    assert built >= 1
    assert water <= 4 * built


# ── The segmentation raster ──────────────────────────────────────────────────

def test_label_raster_matches_the_so2sat_convention(tmp_path):
    path = build_label_raster(_qc_frame(), WudaptConfig(), tmp_path / "t.tif")
    assert path is not None
    with rasterio.open(path) as r:
        assert r.crs.to_epsg() == 4326
        assert r.dtypes[0] == "uint8"
        assert r.nodata == 0
        arr = r.read(1)
    assert set(np.unique(arr)) <= set(range(0, 18))
    assert (arr > 0).any()


def test_label_raster_keeps_only_qc_passing_classes(tmp_path):
    qc = _qc_frame(classes=(1, 14, 17))
    path = build_label_raster(qc, WudaptConfig(), tmp_path / "t.tif")
    with rasterio.open(path) as r:
        present = set(np.unique(r.read(1)).tolist()) - {0}
    assert present <= {1, 14, 17}


def test_contested_pixels_become_nodata_when_priority_is_off(tmp_path):
    """With the ESSD priority rule disabled, two classes can claim one pixel.
    The raster must say 'unknown' rather than pick a winner.
    """
    cfg = WudaptConfig().model_copy(deep=True)
    cfg.qc.use_conflict_priority = False
    cols = {f"f1_{c}": [0.8, 0.8] for c in range(1, 18)}
    gdf = gpd.GeoDataFrame(
        {
            "class": [1, 2], "oa": [0.85, 0.85], "oau": [0.8, 0.8],
            "qc_step1": [True, True],
            "label_year": pd.array([2020, 2020], dtype="Int16"),
            "submission_id": ["a", "b"],
            "submission_date": pd.to_datetime(["2022-01-01"] * 2, utc=True),
            "aoi": ["x", "x"], **cols,
        },
        geometry=[_sq(0, 0, 900.0), _sq(300, 0, 900.0)], crs=_UTM,
    ).to_crs("EPSG:4326")
    qc = apply_qc(gdf, cfg)
    assert int(qc.qc_pass.sum()) == 2
    path = build_label_raster(qc, cfg, tmp_path / "t.tif")
    with rasterio.open(path) as r:
        arr = r.read(1)
    # The overlap strip is ~600 m wide; at 10 m that is thousands of pixels.
    assert (arr == 0).sum() > 0
    assert {1, 2} <= set(np.unique(arr).tolist())


def test_raster_erosion_is_20m_not_the_100m_buffer_distance(tmp_path):
    """The median WUDAPT polygon is ~219 m across, so eroding by the 100 m
    inter-LCZ buffer would erase most of the dataset. A 250 m polygon must
    survive the raster path.
    """
    cfg = WudaptConfig()
    assert cfg.qc.raster_erode_m < cfg.qc.neighbour_buffer_m
    qc = _qc_frame(classes=(6,), side=400.0)
    path = build_label_raster(qc, cfg, tmp_path / "t.tif")
    assert path is not None
    with rasterio.open(path) as r:
        assert (r.read(1) == 6).any()


def test_write_city_dir_produces_the_pair_create_city_grids_needs(tmp_path):
    """src/create_city_grids.py requires exactly patches_reference_{city}.gpkg
    and .tif in the city directory. Anything else and the whole downstream
    stack stops working unchanged.
    """
    written = write_city_dir("testcity", _qc_frame(), WudaptConfig(), tmp_path)
    d = tmp_path / "testcity"
    assert (d / "patches_reference_testcity.gpkg").exists()
    assert (d / "patches_reference_testcity.tif").exists()
    assert set(written) == {"gpkg", "tif"}
