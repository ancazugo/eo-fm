"""Offline tests for the streamed per-UTM-zone zarr sources (datasets/zone_zarr.py).

Small synthetic Zarr v3 stores under tmp_path mimic the published layouts —
one ``utm37`` group with int8 ``embeddings [time, band, y, x]``, ``time`` and
the ``proj:code``/``spatial:transform`` attrs — selected through each source's
store env var:

* dClimate (``TESSERA_DCLIMATE_STORE``): plus float32 ``scales [time, y, x]``.
  Four bands keep it small; the dClimate reader is given a 4-band source.
* OlmoEarth (``OLMOEARTH_STORE``): no scales, -128 = nodata, 128 bands over a
  small area; coverage from a synthetic ``windows/{year}.parquet`` selected
  through ``OLMOEARTH_PROVENANCE``.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import zone_zarr  # noqa: E402
from datasets.registry import (  # noqa: E402
    EMBEDDING_REGISTRY, PRODUCTS, SOURCES, VERSIONS, available_embeddings, is_comparable,
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


def _clear_caches():
    zone_zarr._root.cache_clear()
    zone_zarr._zone.cache_clear()


@pytest.fixture
def env(store, monkeypatch, tmp_path):
    monkeypatch.setenv("TESSERA_DCLIMATE_STORE", str(store["path"]))
    # The real dClimate source reads 128 bands; the synthetic store has 4.
    src = zone_zarr.SOURCES["tesserav1.1_dclimate"]
    monkeypatch.setitem(zone_zarr.SOURCES, src.key, dataclasses.replace(src, n_bands=N_BANDS))
    monkeypatch.setitem(zone_zarr._BY_DIRNAME, src.cell_dirname, zone_zarr.SOURCES[src.key])
    _clear_caches()
    yield tmp_path / "cache"
    _clear_caches()


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
    src, root, year, lon, lat = zone_zarr.parse_cell_path(p)
    assert (src.key, root, year, lon, lat) == ("tesserav1.1_dclimate", tmp_path, 2017, 36.85, -1.25)
    assert tree.geometries[idx[0]].equals(box(36.8, -1.3, 36.9, -1.2))


def test_query_matches_brute_force():
    tree = zone_zarr.CellGrid()
    geom = box(36.83, -1.27, 37.01, -1.11).buffer(0.02)
    got = set(tree.query(geom, predicate="intersects").tolist())
    cand = tree.query(geom)  # bounds candidates, as STRtree returns
    brute = {i for i in cand if geom.intersects(zone_zarr._cell_box(i))}
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
    real_read = zone_zarr._CellBackend._read
    monkeypatch.setattr(zone_zarr._CellBackend, "_read",
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


# ── OlmoEarth ────────────────────────────────────────────────────────────────

OLMO_BANDS = 128
# A small grid at the west edge of cell grid_36.85_-1.25 (lon 36.80-36.90).
OLMO_LONLAT = (36.805, -1.245)
OLMO_SIZE = 200


@pytest.fixture(scope="module")
def olmo_store(tmp_path_factory):
    import zarr
    from pyproj import Transformer

    e, n = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(*OLMO_LONLAT)
    x0, y0 = math.floor(e / PX) * PX, math.ceil(n / PX) * PX
    rng = np.random.default_rng(1)
    emb = rng.integers(-127, 128, size=(len(YEARS), OLMO_BANDS, OLMO_SIZE, OLMO_SIZE), dtype=np.int8)
    emb[:, :, 50:60, 70:90] = -128  # nodata

    path = tmp_path_factory.mktemp("olmo") / "embeddings.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    zone = root.create_group("utm37")
    zone.attrs.update({"proj:code": CRS, "spatial:transform": [PX, 0, x0, 0, -PX, y0]})
    zone.create_array("embeddings", data=emb, chunks=(1, 64, 128, 128), fill_value=-128)
    zone.create_array("time", data=np.array(YEARS, dtype=np.int32))
    return {"path": path, "emb": emb, "x0": x0, "y0": y0}


def _windows(path: Path, boxes_written: list, boxes_unwritten: list) -> None:
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"written": [True] * len(boxes_written) + [False] * len(boxes_unwritten)},
        geometry=list(boxes_written) + list(boxes_unwritten), crs="OGC:CRS84",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)


@pytest.fixture
def olmo_env(olmo_store, monkeypatch, tmp_path):
    prov = tmp_path / "prov"
    # Written: a window over the synthetic grid. Unwritten: one over Paris.
    _windows(prov / "windows" / "2017.parquet", [box(36.80, -1.30, 36.95, -1.15)], [box(2.2, 48.8, 2.5, 49.0)])
    monkeypatch.setenv("OLMOEARTH_STORE", str(olmo_store["path"]))
    monkeypatch.setenv("OLMOEARTH_PROVENANCE", str(prov))
    _clear_caches()
    yield tmp_path / "cache", prov
    _clear_caches()


def test_olmoearth_registry():
    a, b = EMBEDDING_REGISTRY["olmoearth_v1.3"], EMBEDDING_REGISTRY["olmoearth_v1.3_d64"]
    assert (a["in_channels"], b["in_channels"]) == (128, 64)
    for m in (a, b):
        assert m["product"] in PRODUCTS and m["version"] in VERSIONS and m["source"] in SOURCES
        assert m["nodata_all_channels_eq"] == 0.0
    assert {"olmoearth_v1.3", "olmoearth_v1.3_d64"} <= set(available_embeddings())
    assert not is_comparable("olmoearth_v1.3", "tesserav1.1_dclimate")


def test_olmoearth_coverage_masks_index(olmo_env):
    cache, _ = olmo_env
    paths, tree = build_tile_index(cache, "olmoearth_v1.3", "2017")
    assert len(tree.query(box(36.86, -1.24, 36.87, -1.23))) == 1   # covered
    assert len(tree.query(box(2.30, 48.85, 2.31, 48.86))) == 0     # unwritten window
    assert len(tree.query(box(10.0, 10.0, 10.01, 10.01))) == 0     # never computed
    # Batched predicate form honours the mask too.
    pairs = tree.query(np.array([box(36.86, -1.24, 36.87, -1.23), box(2.3, 48.85, 2.31, 48.86)]),
                       predicate="intersects")
    assert set(pairs[0].tolist()) == {0}
    # Compact index: paths/geometries/len only span covered cells, and agree.
    i = tree.query(box(36.86, -1.24, 36.87, -1.23))[0]
    assert paths[i].name == "grid_36.85_-1.25"
    assert tree.geometries[i].equals(box(36.8, -1.3, 36.9, -1.2))
    assert len(paths) == int(zone_zarr._olmoearth_coverage(cache, 2017).sum()) < 6_480_000
    assert all(paths[j].name.startswith("grid_36.") for j in range(len(paths)))


def test_olmoearth_coverage_refreshes_when_provenance_changes(olmo_env):
    import os
    import time

    cache, prov = olmo_env
    mask = zone_zarr._olmoearth_coverage(cache, 2017)
    assert not zone_zarr.CellGrid(mask).query(box(2.30, 48.85, 2.31, 48.86)).size
    # The "backfill" lands: Paris is now written. The stand-in for the ETag
    # (mtime for a local file) changes, so the cached mask must be rebuilt.
    time.sleep(0.01)
    _windows(prov / "windows" / "2017.parquet",
             [box(36.80, -1.30, 36.95, -1.15), box(2.2, 48.8, 2.5, 49.0)], [])
    os.utime(prov / "windows" / "2017.parquet")
    mask = zone_zarr._olmoearth_coverage(cache, 2017)
    assert zone_zarr.CellGrid(mask).query(box(2.30, 48.85, 2.31, 48.86)).size == 1


def _olmo_cell(cache, key):
    paths, tree = build_tile_index(cache, key, "2017")
    return open_tile(paths[tree.query(box(36.86, -1.24, 36.87, -1.23))[0]])


def test_olmoearth_reads_dequantised_with_nodata(olmo_store, olmo_env):
    from dequantize_embeddings import dequantize_alphaearth_embeddings

    cache, _ = olmo_env
    da = _olmo_cell(cache, "olmoearth_v1.3")
    assert da.shape[0] == 128 and da.rio.crs.to_string() == CRS
    # The synthetic grid starts at its own origin inside the cell window.
    gx = olmo_store["x0"] + (np.arange(OLMO_SIZE) + 0.5) * PX
    gy = olmo_store["y0"] - (np.arange(OLMO_SIZE) + 0.5) * PX
    got = da.sel(x=gx, y=gy).values
    q = olmo_store["emb"][0]
    want = dequantize_alphaearth_embeddings(q.astype(np.float32))
    want[q == -128] = np.nan
    np.testing.assert_array_equal(got, want)
    assert np.isnan(got[:, 50:60, 70:90]).all()


def test_olmoearth_d64_is_first_64_bands(olmo_store, olmo_env):
    cache, _ = olmo_env
    full, d64 = _olmo_cell(cache, "olmoearth_v1.3"), _olmo_cell(cache, "olmoearth_v1.3_d64")
    assert d64.shape[0] == 64
    gx = olmo_store["x0"] + (np.arange(0, OLMO_SIZE, 7) + 0.5) * PX
    gy = olmo_store["y0"] - (np.arange(0, OLMO_SIZE, 7) + 0.5) * PX
    np.testing.assert_array_equal(d64.sel(x=gx, y=gy).values, full.sel(x=gx, y=gy).values[:64])


def test_extract_chunksize():
    assert zone_zarr.extract_chunksize("tesserav1.1_dclimate") == 16
    assert zone_zarr.extract_chunksize("olmoearth_v1.3") == 64
    assert zone_zarr.extract_chunksize("tesserav2", default=256) == 256


def test_olmoearth_block_cache_reuses_and_matches(olmo_store, olmo_env, monkeypatch):
    """A second read inside the same blocks makes no zarr call, and windows
    spanning several blocks (128² in the synthetic store) assemble exactly."""
    cache, _ = olmo_env
    zone_zarr._blocks.clear()
    da = _olmo_cell(cache, "olmoearth_v1.3")
    gx = olmo_store["x0"] + (np.arange(OLMO_SIZE) + 0.5) * PX
    gy = olmo_store["y0"] - (np.arange(OLMO_SIZE) + 0.5) * PX
    sub = da.sel(x=gx, y=gy)
    first = sub.isel(x=slice(100, 160), y=slice(110, 150)).values  # crosses the 128 block edge

    calls = []
    import zarr.core.array as za
    orig = za.Array.__getitem__
    monkeypatch.setattr(za.Array, "__getitem__", lambda self, k: calls.append(k) or orig(self, k))
    again = sub.isel(x=slice(105, 150), y=slice(115, 140)).values
    assert not calls
    np.testing.assert_array_equal(again, first[:, 5:30, 5:50])
