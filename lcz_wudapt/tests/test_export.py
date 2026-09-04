"""H5/H6 — the Stage 8 contract. These are the labels lcz_train actually reads."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from lcz_labels.export import decode_bitmask, encode_lcz_set
from lcz_wudapt.config import RegionParams, WudaptConfig
from lcz_wudapt.consensus import consensus_for_aoi, footprint_grid
from lcz_wudapt.export import _dense, consensus_regions, merge_so2sat, write_stage8
from lcz_wudapt.quality import polygon_weights
from lcz_wudapt.tests.test_consensus import _polys


def _result(specs, cfg, res_m=50.0):
    gdf = _polys(specs)
    w = polygon_weights(gdf, cfg)["weight"].to_numpy()
    return gdf, consensus_for_aoi(gdf, w, cfg, res_m=res_m)


def test_so2sat_always_wins_on_contested_ground():
    """The user's rule: So2Sat is authoritative wherever it exists."""
    cfg = WudaptConfig()
    _, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(4)], cfg)
    assert (res.top_class == 3).all()

    so2sat = np.zeros(res.grid.shape, dtype=np.uint8)
    so2sat[res.grid.shape[0] // 2, res.grid.shape[1] // 2] = 6      # disagrees
    bitmask, conf, source = merge_so2sat(res, so2sat)

    px = so2sat > 0
    assert decode_bitmask(bitmask[px]) == [[6]]          # So2Sat class, not WUDAPT's 3
    assert conf[px] == pytest.approx(1.0)
    assert source[px] == 2


def test_so2sat_does_not_bleed_into_unlabelled_ground():
    cfg = WudaptConfig()
    _, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(3)], cfg)
    bitmask, conf, source = merge_so2sat(res, np.zeros(res.grid.shape, dtype=np.uint8))
    assert not (source == 2).any()
    assert (source == 1).sum() == len(res)


def test_regions_split_on_label_set_and_on_source():
    """A region must never span two different label sets or two provenances."""
    cfg = WudaptConfig(regions=RegionParams(min_region_area_m2=1.0))
    gdf, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(3)], cfg)
    so2sat = np.zeros(res.grid.shape, dtype=np.uint8)
    so2sat[: res.grid.shape[0] // 2] = 6
    bitmask, conf, source = merge_so2sat(res, so2sat)
    regions = consensus_regions(bitmask, conf, source, res.grid, "A__30_1", cfg)

    assert set(regions["source"]) == {"wudapt", "so2sat"}
    for _, r in regions.iterrows():
        assert len(set(map(tuple, [r.lcz_set]))) == 1
    assert regions.loc[regions.source == "so2sat", "lcz"].eq(6).all()
    assert regions.loc[regions.source == "wudapt", "lcz"].eq(3).all()


def test_regions_below_the_area_floor_are_dropped():
    cfg_small = WudaptConfig(regions=RegionParams(min_region_area_m2=1.0))
    cfg_big = WudaptConfig(regions=RegionParams(min_region_area_m2=1e12))
    gdf, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(3)], cfg_small)
    bm, cf, sr = merge_so2sat(res, None)
    assert len(consensus_regions(bm, cf, sr, res.grid, "A__30_1", cfg_small)) > 0
    assert len(consensus_regions(bm, cf, sr, res.grid, "A__30_1", cfg_big)) == 0


def test_written_rasters_agree_with_each_other_and_with_the_parquet(tmp_path):
    """block_id, bitmask and confidence must mark exactly the same pixels."""
    cfg = WudaptConfig(cache_dir=tmp_path, regions=RegionParams(min_region_area_m2=1.0))
    gdf, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(3)]
                       + [{"annotator": f"b{i}", "class": 5} for i in range(3)], cfg)
    bm, cf, sr = merge_so2sat(res, None)
    regions = consensus_regions(bm, cf, sr, res.grid, "A__30_1", cfg,
                                extra={"n_eff": _dense(res, res.n_eff, "float32")})
    paths = write_stage8(regions, bm, cf, res.grid, "A__30_1", cfg)

    B = rasterio.open(paths["bitmask"]).read(1)
    C = rasterio.open(paths["confidence"]).read(1)
    I = rasterio.open(paths["block_id"]).read(1)
    assert B.dtype == np.uint32 and C.dtype == np.uint8 and I.dtype == np.uint32
    assert ((B > 0) == (I > 0)).all()
    assert C.max() <= 100
    tbl = gpd.read_parquet(paths["blocks"])
    assert len(tbl) == len(np.unique(I)) - 1
    assert set(tbl["block_idx"]) == set(np.unique(I)[1:])


def test_every_written_class_is_a_valid_lcz_code(tmp_path):
    cfg = WudaptConfig(cache_dir=tmp_path, regions=RegionParams(min_region_area_m2=1.0))
    _, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(3)], cfg)
    bm, cf, sr = merge_so2sat(res, None)
    for s in decode_bitmask(np.unique(bm[bm > 0])):
        assert s and all(1 <= c <= 17 for c in s)


def test_rasters_are_written_self_describing(tmp_path):
    """Alignment is by geo-reference, not convention: all three carry CRS+transform."""
    cfg = WudaptConfig(cache_dir=tmp_path, regions=RegionParams(min_region_area_m2=1.0))
    _, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(3)], cfg)
    bm, cf, sr = merge_so2sat(res, None)
    regions = consensus_regions(bm, cf, sr, res.grid, "A__30_1", cfg)
    paths = write_stage8(regions, bm, cf, res.grid, "A__30_1", cfg)
    opened = [rasterio.open(paths[k]) for k in ("bitmask", "confidence", "block_id")]
    assert len({str(o.crs) for o in opened}) == 1
    assert len({o.transform for o in opened}) == 1
    assert all(o.crs is not None for o in opened)


def test_adjacency_is_empty_but_schema_correct(tmp_path):
    """Empty-with-schema makes lcz_train report B3 as *not run*, not crashed."""
    import pandas as pd

    cfg = WudaptConfig(cache_dir=tmp_path, regions=RegionParams(min_region_area_m2=1.0))
    _, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(3)], cfg)
    bm, cf, sr = merge_so2sat(res, None)
    regions = consensus_regions(bm, cf, sr, res.grid, "A__30_1", cfg)
    paths = write_stage8(regions, bm, cf, res.grid, "A__30_1", cfg)
    adj = pd.read_parquet(paths["adjacency"])
    assert list(adj.columns) == ["block_a", "block_b", "shared_len_m"]
    assert len(adj) == 0


def test_block_kind_records_whether_the_label_had_corroboration(tmp_path):
    """lcz_train's by_block_kind then answers 'is consensus worth anything?' free."""
    cfg = WudaptConfig(cache_dir=tmp_path, regions=RegionParams(min_region_area_m2=1.0))
    _, res = _result([{"annotator": f"a{i}", "class": 3} for i in range(6)], cfg)
    bm, cf, sr = merge_so2sat(res, None)
    regions = consensus_regions(bm, cf, sr, res.grid, "A__30_1", cfg,
                                extra={"n_eff": _dense(res, res.n_eff, "float32")})
    assert regions["block_kind"].str.startswith("wudapt_consensus_").all()
    assert (regions["n_eff"] > 1).any()


# ── Region-centred patch sampling (H7) ───────────────────────────────────────

def _build_aoi(tmp_path, specs, res_m=10.0):
    """Write a real Stage 8 export so the sampler can be tested end to end."""
    from lcz_wudapt.consensus import consensus_for_aoi
    from lcz_wudapt.export import _dense
    cfg = WudaptConfig(cache_dir=tmp_path, regions=RegionParams(min_region_area_m2=1.0))
    gdf = _polys(specs)
    w = polygon_weights(gdf, cfg)["weight"].to_numpy()
    res = consensus_for_aoi(gdf, w, cfg, res_m=res_m)
    bm, cf, sr = merge_so2sat(res, None)
    regions = consensus_regions(bm, cf, sr, res.grid, "A__30_1", cfg,
                                extra={"n_eff": _dense(res, res.n_eff, "float32")})
    write_stage8(regions, bm, cf, res.grid, "A__30_1", cfg)
    return cfg


def test_region_centred_patches_land_on_the_label(tmp_path):
    """A patch centred on a region should be that region's class, near-pure."""
    from lcz_wudapt.patch_bridge import region_centred_patches
    cfg = _build_aoi(tmp_path, [{"annotator": f"a{i}", "class": 3} for i in range(3)])
    p = region_centred_patches("A__30_1", cfg)
    assert len(p) > 0
    assert (p["LCZ_class"] == 3).all()
    assert p["dominant_frac"].min() >= 0.75
    assert str(p.crs) == "EPSG:4326"


def test_patch_geometry_is_a_320m_square(tmp_path):
    """The pool must be So2Sat-shaped or it cannot feed patch_classification."""
    from lcz_wudapt.patch_bridge import region_centred_patches
    cfg = _build_aoi(tmp_path, [{"annotator": f"a{i}", "class": 3} for i in range(3)])
    p = region_centred_patches("A__30_1", cfg, patch_px=32)
    utm = p.to_crs(p.estimate_utm_crs())
    w = utm.geometry.bounds.maxx - utm.geometry.bounds.minx
    h = utm.geometry.bounds.maxy - utm.geometry.bounds.miny
    assert w.between(315, 325).all() and h.between(315, 325).all()


def test_max_per_region_caps_large_regions(tmp_path):
    """The cap is what keeps water from re-dominating: big regions are mostly
    water and natural classes, so an uncapped tiling reintroduces the bias."""
    from shapely.geometry import box as _box
    from lcz_wudapt.patch_bridge import region_centred_patches
    big = _box(0.0, 0.0, 0.03, 0.03)          # comfortably many 320 m cells
    cfg = _build_aoi(tmp_path, [{"annotator": f"a{i}", "class": 17, "geom": big}
                                for i in range(3)])
    few = region_centred_patches("A__30_1", cfg, max_per_region=2)
    many = region_centred_patches("A__30_1", cfg, max_per_region=16)
    assert len(few) <= 2
    assert len(many) > len(few)


def test_coarse_regions_never_become_patches(tmp_path):
    """A patch CE loss needs one class; an ambiguous set has none to give."""
    from lcz_wudapt.patch_bridge import region_centred_patches
    specs = [{"annotator": f"a{i}", "class": 2} for i in range(3)]
    specs += [{"annotator": f"b{i}", "class": 5} for i in range(3)]
    cfg = _build_aoi(tmp_path, specs)
    tbl = gpd.read_parquet(tmp_path / "A__30_1" / "blocks_labelled_A__30_1.parquet")
    assert (tbl["label_type"] == "coarse").any()          # the export IS coarse here
    assert len(region_centred_patches("A__30_1", cfg)) == 0


def test_patches_carry_confidence_as_weight(tmp_path):
    """build_pseudo_items feeds `weight` straight into the weighted CE loss."""
    from lcz_wudapt.patch_bridge import region_centred_patches
    cfg = _build_aoi(tmp_path, [{"annotator": f"a{i}", "class": 3} for i in range(4)])
    p = region_centred_patches("A__30_1", cfg)
    assert p["weight"].between(0, 1).all()
    assert p["weight"].gt(0).all()


def test_missing_export_raises_rather_than_returning_empty(tmp_path):
    from lcz_wudapt.patch_bridge import region_centred_patches
    with pytest.raises(FileNotFoundError):
        region_centred_patches("Nope__30_9", WudaptConfig(cache_dir=tmp_path))
