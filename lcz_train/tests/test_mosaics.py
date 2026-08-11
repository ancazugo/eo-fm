"""T2 mosaic step — warp math + full build_mosaic orchestration (mocked tiles)."""

import json

import numpy as np
import pytest
import rioxarray  # noqa: F401 — registers the .rio DataArray accessor
import xarray as xr
from rasterio.transform import Affine, from_origin

from lcz_train.mosaics import (
    _tile_to_northup_array,
    build_mosaic,
    load_mosaic,
    warp_tile_onto_grid,
)


def _synthetic_da(value: float, *, south_up: bool = False, res: float = 10.0,
                  origin=(500000.0, 5500000.0), size=20, crs="EPSG:32633"):
    """A constant-value (1, size, size) north- or south-up DataArray."""
    x0, y0 = origin
    xs = x0 + res * (np.arange(size) + 0.5)
    ys = (y0 - res * (np.arange(size) + 0.5)) if not south_up else \
         (y0 - res * size + res * (np.arange(size) + 0.5))
    da = xr.DataArray(
        np.full((1, size, size), value, dtype=np.float32),
        dims=("band", "y", "x"), coords={"band": [0], "y": ys, "x": xs},
    )
    da = da.rio.write_crs(crs)
    return da


def test_tile_to_northup_flips_south_up():
    da = _synthetic_da(1.0, south_up=True)
    arr, transform, crs = _tile_to_northup_array(da)
    assert transform.e < 0          # north-up: negative y-step
    assert crs == "EPSG:32633"
    assert arr.shape == (1, 20, 20)


def test_tile_to_northup_is_noop_for_north_up():
    da = _synthetic_da(2.0, south_up=False)
    arr, transform, _ = _tile_to_northup_array(da)
    assert transform.e < 0
    np.testing.assert_array_equal(arr[0], 2.0)


def test_warp_tile_identity_same_grid():
    da = _synthetic_da(5.0)
    arr, src_t, src_crs = _tile_to_northup_array(da)
    out = warp_tile_onto_grid(arr, src_t, src_crs, src_t, src_crs, (20, 20))
    np.testing.assert_allclose(out[0], 5.0)


def test_warp_tile_onto_larger_grid_partial_coverage():
    # Tile spans x:[500000,500100] y:[5499900,5500000]; dst canvas
    # x:[499950,500250] y:[5499900,5500200] overlaps only the tile's lower-left.
    da = _synthetic_da(3.0, origin=(500000.0, 5500000.0), size=10)
    arr, src_t, src_crs = _tile_to_northup_array(da)
    dst_t = from_origin(499950.0, 5500200.0, 10.0, 10.0)
    out = warp_tile_onto_grid(arr, src_t, src_crs, dst_t, src_crs, (30, 30))
    assert (out != 0).any()
    assert (out == 0).any()          # some of the canvas is outside the tile


def test_warp_accumulates_disjoint_tiles_without_erasing_earlier_ones():
    # Regression: rasterio.warp.reproject recomputes the WHOLE destination
    # from the current source alone, writing dst_nodata everywhere outside
    # its footprint — passing the same array as `destination` across calls
    # used to silently erase every earlier tile once a later, non-overlapping
    # tile was warped onto the same accumulator (only the last tile survived).
    # Two tiles at disjoint UTM locations must BOTH show up in a shared dst.
    da_left = _synthetic_da(3.0, origin=(500000.0, 5500000.0), size=10)
    da_right = _synthetic_da(7.0, origin=(500200.0, 5500000.0), size=10)
    a1, t1, c1 = _tile_to_northup_array(da_left)
    a2, t2, c2 = _tile_to_northup_array(da_right)
    dst_transform = from_origin(500000.0, 5500000.0, 10.0, 10.0)
    dst = np.zeros((1, 10, 30), dtype=np.float32)
    warp_tile_onto_grid(a1, t1, c1, dst_transform, c1, (10, 30), dst=dst)
    warp_tile_onto_grid(a2, t2, c2, dst_transform, c1, (10, 30), dst=dst)
    np.testing.assert_allclose(dst[0, :, :10], 3.0)     # left tile still present
    np.testing.assert_allclose(dst[0, :, 20:], 7.0)     # right tile placed correctly
    assert (dst[0, :, 10:20] == 0).all()                # untouched gap stays zero


def test_warp_last_writer_wins_on_shared_dst():
    da1 = _synthetic_da(1.0, size=10)
    da2 = _synthetic_da(9.0, size=10)
    a1, t1, c1 = _tile_to_northup_array(da1)
    a2, t2, c2 = _tile_to_northup_array(da2)
    dst = np.zeros((1, 10, 10), dtype=np.float32)
    warp_tile_onto_grid(a1, t1, c1, t1, c1, (10, 10), dst=dst)
    warp_tile_onto_grid(a2, t2, c2, t1, c1, (10, 10), dst=dst)
    np.testing.assert_allclose(dst[0], 9.0)


class _FakeAOI:
    def __init__(self, name):
        self.name = name
        self.bbox = (10.0, 45.0, 10.01, 45.01)
        self.equal_area_crs = None


class _FakeConfig:
    def aoi(self, name):
        return _FakeAOI(name)


def test_build_mosaic_orchestration(tmp_path, monkeypatch):
    """Mocks tiles.py + raster_grid so the full cache-write path runs offline."""
    grid_transform = from_origin(600000.0, 5000100.0, 10.0, 10.0)
    grid_shape = (10, 10)

    monkeypatch.setattr(
        "lcz_train.mosaics.raster_grid",
        lambda aoi, cfg: (grid_transform, "EPSG:32633", grid_shape),
    )
    monkeypatch.setattr(
        "lcz_labels.grid.resolve_aoi_bbox", lambda aoi, cfg: (10.0, 45.0, 10.01, 45.01)
    )

    tile_da = _synthetic_da(7.0, origin=(600000.0, 5000100.0), size=10,
                            res=10.0, crs="EPSG:32633")

    import sys
    fake_tiles = type(sys)("datasets.tiles")
    fake_tiles.build_tile_index = lambda d, name, year: (["tile0"], _FakeTree())
    fake_tiles.open_tile = lambda p: tile_da
    # Via monkeypatch so sys.modules is restored afterwards: a plain assignment
    # leaves the fake installed for the rest of the session, and any later test
    # that imports datasets.tiles for real gets this stub instead.
    monkeypatch.setitem(sys.modules, "datasets.tiles", fake_tiles)
    if "datasets" not in sys.modules:
        monkeypatch.setitem(sys.modules, "datasets", type(sys)("datasets"))

    cfg = _FakeConfig()
    path = build_mosaic("TestAOI", 2025, "tessera", tmp_path / "emb", cfg, tmp_path / "mosaics")
    assert path.exists()
    arr, meta = load_mosaic(path)
    assert arr.shape == (1, 10, 10)
    np.testing.assert_allclose(np.asarray(arr[0], dtype=np.float32), 7.0, atol=0.01)  # fp16 round-trip
    assert meta["crs"] == "EPSG:32633"
    assert meta["shape"] == [1, 10, 10]

    # cache hit path doesn't re-warp (would raise if it tried — tile is a string)
    fake_tiles.open_tile = lambda p: (_ for _ in ()).throw(AssertionError("should not reopen"))
    path2 = build_mosaic("TestAOI", 2025, "tessera", tmp_path / "emb", cfg, tmp_path / "mosaics")
    assert path2 == path


class _FakeTree:
    def query(self, geom):
        return [0]
