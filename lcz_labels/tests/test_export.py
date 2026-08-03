"""Stage 8 — bitmask contract round-trip and canonical raster grid."""

import numpy as np
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
