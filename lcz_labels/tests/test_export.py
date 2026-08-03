"""Stage 8 — bitmask contract round-trip and canonical raster grid."""

import numpy as np
import pandas as pd
import pytest

from lcz_labels.config import AOI, LczLabelConfig
from lcz_labels.export import decode_bitmask, encode_lcz_set, raster_grid


def test_encode_hard_coarse_unlabelled():
    bm = encode_lcz_set([[3], [3, 7], [8, 10], [17], None, []])
    assert bm.dtype == np.uint32
    assert bm.tolist() == [
        1 << 2, (1 << 2) | (1 << 6), (1 << 7) | (1 << 9), 1 << 16, 0, 0,
    ]


def test_bitmask_round_trip():
    rng = np.random.default_rng(0)
    sets = [sorted(rng.choice(np.arange(1, 18), size=k, replace=False).tolist())
            for k in (1, 1, 2, 2, 3, 17) ] + [[]]
    assert decode_bitmask(encode_lcz_set(sets)) == sets


def test_encode_rejects_out_of_range():
    with pytest.raises(ValueError):
        encode_lcz_set([[0]])
    with pytest.raises(ValueError):
        encode_lcz_set([[18]])


def test_encode_accepts_numpy_arrays_from_parquet_round_trip():
    # lcz_set comes back from a parquet read as a numpy array, not a list;
    # `if not s` on a multi-element array raises ValueError — regression check.
    sets = [np.array([3, 7]), np.array([3]), np.array([], dtype=int), None]
    assert encode_lcz_set(sets).tolist() == [(1 << 2) | (1 << 6), 1 << 2, 0, 0]


@pytest.fixture()
def cfg(tmp_path):
    c = LczLabelConfig(cache_dir=tmp_path)
    c.aoi_list = [AOI(name="Box", bbox=(36.70, -1.35, 36.75, -1.30))]
    return c


def test_raster_grid_snapped_and_deterministic(cfg):
    transform, utm, (h, w) = raster_grid("Box", cfg)
    res = cfg.export.raster_res_m
    assert transform.a == res and transform.e == -res
    assert transform.c % res == 0 and transform.f % res == 0  # snapped origin
    assert h > 0 and w > 0
    t2, utm2, shape2 = raster_grid("Box", cfg)
    assert t2 == transform and utm2 == utm and shape2 == (h, w)
    # ~0.05 deg at the equator is ~5.5 km -> hundreds of 10 m pixels
    assert 400 < w < 800 and 400 < h < 800


def test_raster_grid_covers_the_bbox(cfg):
    import geopandas as gpd
    from shapely.geometry import box

    transform, utm, (h, w) = raster_grid("Box", cfg)
    ux0, uy0, ux1, uy1 = (
        gpd.GeoSeries([box(36.70, -1.35, 36.75, -1.30)], crs="EPSG:4326")
        .to_crs(utm).total_bounds
    )
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    assert left <= ux0 and bottom <= uy0 and right >= ux1 and top >= uy1


def test_raster_grid_honours_res_config(cfg):
    cfg.export.raster_res_m = 20.0
    t20, _, (h20, w20) = raster_grid("Box", cfg)
    cfg.export.raster_res_m = 10.0
    _, _, (h10, w10) = raster_grid("Box", cfg)
    assert t20.a == 20.0
    assert abs(w10 - 2 * w20) <= 2 and abs(h10 - 2 * h20) <= 2


# ── write_rasters + patch_transfer ────────────────────────────────────────────

@pytest.fixture()
def small_cfg(tmp_path):
    import geopandas as gpd  # noqa: F401 — ensures geo stack importable
    c = LczLabelConfig(cache_dir=tmp_path)
    c.aoi_list = [AOI(name="Tiny", bbox=(36.700, -1.350, 36.710, -1.340))]
    return c


def _tiny_labels(small_cfg):
    """Three blocks in the Tiny AOI's UTM grid: hard 3, coarse {3,7}, unlabelled."""
    import geopandas as gpd
    from shapely.geometry import box as sbox

    from lcz_labels.export import raster_grid as rg

    transform, utm, (h, w) = rg("Tiny", small_cfg)
    x0, y1 = transform.c, transform.f          # top-left corner
    res = transform.a
    def cell_box(c0, r0, c1, r1):              # pixel coords -> UTM box
        return sbox(x0 + c0 * res, y1 - r1 * res, x0 + c1 * res, y1 - r0 * res)

    return gpd.GeoDataFrame({
        "block_id": ["b1", "b2", "b3"],
        "block_idx": np.array([1, 2, 3], dtype=np.uint32),
        "block_kind": ["enclosure"] * 3,
        "label_type": ["hard", "coarse", "unlabelled"],
        "lcz": pd.array([3, None, None], dtype="Int64"),
        "lcz_set": [[3], [3, 7], []],
        "confidence": [0.9, 0.5, 0.0],
        "area_m2": [1.0] * 3,
    }, geometry=[cell_box(10, 10, 30, 30), cell_box(30, 10, 50, 30), cell_box(10, 30, 30, 50)],
        crs=utm)


def test_write_rasters_round_trip(small_cfg):
    import rasterio

    from lcz_labels.export import raster_grid as rg
    from lcz_labels.export import write_rasters

    labels = _tiny_labels(small_cfg)
    paths = write_rasters(labels, "Tiny", small_cfg)
    transform, utm, (h, w) = rg("Tiny", small_cfg)

    with rasterio.open(paths["bitmask"]) as src:
        assert src.transform == transform and str(src.crs) == utm
        bm = src.read(1)
    with rasterio.open(paths["confidence"]) as src:
        cf = src.read(1)
    with rasterio.open(paths["block_id"]) as src:
        bi = src.read(1)

    assert bm.shape == (h, w) and bm.dtype == np.uint32
    # hard 3 -> bit 2; coarse {3,7} -> bits 2|6; unlabelled block absent
    assert set(np.unique(bm)) == {0, 1 << 2, (1 << 2) | (1 << 6)}
    assert bm[20, 20] == 1 << 2                 # inside b1
    assert bm[20, 40] == (1 << 2) | (1 << 6)    # inside b2
    assert bm[40, 20] == 0                      # inside b3 (unlabelled)
    assert decode_bitmask(np.array([bm[20, 40]]))[0] == [3, 7]
    # confidence x100
    assert cf[20, 20] == 90 and cf[20, 40] == 50 and cf[40, 20] == 0
    # block index raster burns ALL blocks incl. unlabelled
    assert bi[20, 20] == 1 and bi[20, 40] == 2 and bi[40, 20] == 3
    assert bi[5, 5] == 0                        # outside any block


def test_patch_transfer_fractions_and_keys(small_cfg):
    import geopandas as gpd
    from shapely.geometry import box as sbox

    from lcz_labels.export import patch_transfer

    labels = _tiny_labels(small_cfg)
    utm = labels.crs
    # Patch 1 sits exactly on b1(80%)+b2(20%); patch 2 half on b2, half outside.
    t = labels.geometry.iloc[0].bounds  # b1: 200x200 m starting at its minx/miny
    x0, y0 = t[0], t[1]
    # p1: 200 m tall, 250 m wide -> 200 m over b1 + 50 m over b2
    p1 = sbox(x0, y0, x0 + 250, y0 + 200)
    # p2: 400 m wide, only its first 100 m over b2 (b2 spans x0+200..x0+400)
    p2 = sbox(x0 + 300, y0, x0 + 700, y0 + 200)
    grid = gpd.GeoDataFrame({
        "patch_id": ["0000001", "0000001"],
        "dataset": ["training", "validation"],
        "LCZ_class": [3.0, 7.0],
    }, geometry=[p1, p2], crs=utm)

    out = patch_transfer(labels, grid, "Tiny", small_cfg).set_index(["dataset", "patch_id"])
    assert len(out) == 2                        # (dataset, patch_id) keying keeps both

    r1 = out.loc[("training", "0000001")]
    a1 = 250 * 200
    f_b1, f_b2 = (200 * 200) / a1, (50 * 200) / a1
    assert abs(r1["f_lcz_3"] - (f_b1 + f_b2 / 2)) < 1e-6
    assert abs(r1["f_lcz_7"] - f_b2 / 2) < 1e-6
    assert abs(r1["coarse_frac"] - f_b2) < 1e-6
    assert r1["dominant_lcz"] == 3 and r1["dominant_frac"] > 0.75
    assert not r1["boundary_flag"]
    assert abs(r1["mean_confidence"] - (0.9 * f_b1 + 0.5 * f_b2)) < 1e-6
    assert r1["so2sat_lcz"] == 3.0

    r2 = out.loc[("validation", "0000001")]
    assert r2["dominant_lcz"] is None or pd.isna(r2["dominant_lcz"])  # coarse dominant
    assert list(r2["dominant_set"]) == [3, 7]
    assert r2["boundary_flag"]                  # dominant covers < 0.75
    assert r2["unlabelled_frac"] > 0.4
