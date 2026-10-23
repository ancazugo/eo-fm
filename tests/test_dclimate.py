"""Offline tests for the tesserav1.1_dclimate source (datasets/dclimate.py).

A small synthetic Zarr v3 store under tmp_path mimics the published layout —
one ``utm37`` group with int8 ``embeddings [time, band, y, x]``, float32
``scales [time, y, x]``, ``time`` and the ``proj:code``/``spatial:transform``
attrs — and is selected through ``TESSERA_DCLIMATE_STORE``. Four bands rather
than 128 keep it small; nothing in the reader assumes the count.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import dclimate  # noqa: E402
from datasets.registry import (  # noqa: E402
    EMBEDDING_REGISTRY, SOURCES, available_embeddings, is_comparable,
)
from datasets.tiles import build_tile_index, crop_patch, open_tile, tile_index_name  # noqa: E402

CRS = "EPSG:32637"
PX = 10.0
N_BANDS = 4
YEARS = [2017, 2018]
# Region lon 36.8-37.0, lat -1.3--1.2: cell grid_36.85_-1.25 and its east
# neighbour grid_36.95_-1.25, both in UTM zone 37.
REGION = (36.78, -1.32, 37.02, -1.18)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    import zarr
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    minx, miny, maxx, maxy = t.transform_bounds(*REGION, densify_pts=21)
    x0, y0 = math.floor(minx / PX) * PX, math.ceil(maxy / PX) * PX
    w, h = math.ceil((maxx - x0) / PX), math.ceil((y0 - miny) / PX)

    rng = np.random.default_rng(0)
    emb = rng.integers(-127, 128, size=(len(YEARS), N_BANDS, h, w), dtype=np.int8)
    scales = rng.uniform(0.01, 0.1, size=(len(YEARS), h, w)).astype(np.float32)
    scales[:, 100:120, 200:230] = np.nan  # an unembedded gap

    path = tmp_path_factory.mktemp("dclimate") / "store.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update({"geotessera:mask_source": "source_nodata"})
    zone = root.create_group("utm37")
    zone.attrs.update({
        "proj:code": CRS,
        "spatial:transform": [PX, 0, x0, 0, -PX, y0],
    })
    zone.create_array("embeddings", data=emb, chunks=(1, N_BANDS, 256, 256))
    zone.create_array("scales", data=scales, chunks=(1, 256, 256))
    zone.create_array("time", data=np.array(YEARS, dtype=np.int32))
    return {"path": path, "emb": emb, "scales": scales, "x0": x0, "y0": y0}


@pytest.fixture
def env(store, monkeypatch, tmp_path):
    monkeypatch.setenv(dclimate.STORE_ENV, str(store["path"]))
    dclimate._root.cache_clear()
    dclimate._zone.cache_clear()
    yield tmp_path / "cache"
    dclimate._root.cache_clear()
    dclimate._zone.cache_clear()


def _dequantised(store, ti):
    s = store["scales"][ti]
    out = store["emb"][ti].astype(np.float32) * np.where(np.isfinite(s), s, 0)
    out[:, ~np.isfinite(s)] = np.nan
    return out


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_entry():
    meta = EMBEDDING_REGISTRY["tesserav1.1_dclimate"]
    assert meta["source"] in SOURCES
    assert meta["status"] == "supported"
    assert meta["in_channels"] == 128
    assert meta["nodata_all_channels_eq"] == 0.0
    assert "tesserav1.1_dclimate" in available_embeddings()


def test_not_comparable_to_global_archive():
    # Same product/version/channels, different inference run.
    assert not is_comparable("tesserav1.1_dclimate", "tesserav1.1_global")
    assert is_comparable("tesserav1.1_dclimate", "tesserav1.1_dclimate")


# ── Index ────────────────────────────────────────────────────────────────────

def test_build_tile_index_needs_year(tmp_path):
    with pytest.raises(ValueError, match="--year"):
        build_tile_index(tmp_path, "tesserav1.1_dclimate", None)


def test_cell_paths_round_trip(tmp_path):
    paths, tree = build_tile_index(tmp_path, "tesserav1.1_dclimate", "2017")
    idx = tree.query(box(36.86, -1.24, 36.87, -1.23))
    assert len(idx) == 1
    p = paths[idx[0]]
    assert p.name == "grid_36.85_-1.25"
    assert tile_index_name(p) == "grid_36.85_-1.25"  # Path.stem would truncate
    assert dclimate.parse_cell_path(p) == (tmp_path, 2017, 36.85, -1.25)
    assert tree.geometries[idx[0]].equals(box(36.8, -1.3, 36.9, -1.2))


def test_query_matches_brute_force():
    tree = dclimate.CellGrid()
    geom = box(36.83, -1.27, 37.01, -1.11).buffer(0.02)
    got = set(tree.query(geom, predicate="intersects").tolist())
    cand = tree.query(geom)  # bounds candidates, as STRtree returns
    brute = {i for i in cand if geom.intersects(dclimate._cell_box(i))}
    assert got == brute and len(got) >= 6
    # Batched form: (2, n) [input, cell] pairs.
    pairs = tree.query(np.array([geom, box(0.01, 0.01, 0.02, 0.02)]), predicate="intersects")
    assert pairs.shape[0] == 2
    assert set(pairs[1][pairs[0] == 0].tolist()) == brute
    assert len(pairs[1][pairs[0] == 1]) == 1
    geoms = tree.geometries[np.array(sorted(brute))]
    assert all(g.intersects(geom) for g in geoms)


# ── Lazy cell reads ──────────────────────────────────────────────────────────

def test_open_tile_is_lazy_and_exact(store, env, monkeypatch):
    paths, tree = build_tile_index(env, "tesserav1.1_dclimate", "2018")
    cell = paths[tree.query(box(36.86, -1.24, 36.87, -1.23))[0]]

    reads = []
    real_read = dclimate._CellBackend._read
    monkeypatch.setattr(dclimate._CellBackend, "_read",
                        lambda self, key: reads.append(key) or real_read(self, key))

    da = open_tile(cell)
    assert da.dims == ("band", "y", "x") and da.rio.crs.to_string() == CRS
    assert float(da.y[0]) > float(da.y[-1])  # north-up
    assert not reads

    clipped = da.rio.clip_box(float(da.x[10]), float(da.y[60]), float(da.x[50]), float(da.y[20]))
    assert not reads
    vals = clipped.values
    assert len(reads) == 1

    full = _dequantised(store, ti=1)
    c = round((float(clipped.x[0]) - store["x0"]) / PX - 0.5)
    r = round((store["y0"] - float(clipped.y[0])) / PX - 0.5)
    np.testing.assert_array_equal(vals, full[:, r:r + vals.shape[1], c:c + vals.shape[2]])


def test_nan_scales_read_as_nan(store, env):
    from pyproj import Transformer

    # Pixel-centre coords of the synthetic gap (rows 100-120, cols 200-230).
    gx = store["x0"] + (np.array([200, 229]) + 0.5) * PX
    gy = store["y0"] - (np.array([100, 119]) + 0.5) * PX
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(gx.mean(), gy.mean())
    paths, tree = build_tile_index(env, "tesserav1.1_dclimate", "2017")
    da = open_tile(paths[tree.query(box(lon, lat, lon, lat))[0]])
    gap = da.sel(x=slice(gx[0], gx[1]), y=slice(gy[0], gy[1])).values
    assert gap.size and np.isnan(gap).all()
    assert np.isfinite(da.isel(x=slice(0, 5), y=slice(0, 5)).values).all()


def test_unknown_year_raises(store, env):
    paths, tree = build_tile_index(env, "tesserav1.1_dclimate", "2024")
    with pytest.raises(ValueError, match="2024"):
        open_tile(paths[tree.query(box(36.86, -1.24, 36.87, -1.23))[0]])


def test_crop_patch_across_two_cells(store, env, tmp_path):
    """A patch straddling lon 36.9 matches one direct window read."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    e, n = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(36.9, -1.25)
    c = round((e - store["x0"]) / PX) - 16
    r = round((store["y0"] - n) / PX) - 16
    left, top = store["x0"] + c * PX, store["y0"] - r * PX
    patch = box(left, top - 32 * PX, left + 32 * PX, top)

    paths, tree = build_tile_index(env, "tesserav1.1_dclimate", "2017")
    to_4326 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    idx = tree.query(shp_transform(to_4326.transform, patch), predicate="intersects")
    matched = [paths[i] for i in idx]
    assert {p.name for p in matched} >= {"grid_36.85_-1.25", "grid_36.95_-1.25"}

    out = tmp_path / "patch.npy"
    assert crop_patch(patch, CRS, matched, out)
    arr = np.load(out)
    np.testing.assert_array_equal(arr, _dequantised(store, ti=0)[:, r:r + 32, c:c + 32])
