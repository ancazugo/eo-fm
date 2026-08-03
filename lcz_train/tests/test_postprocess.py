"""T6 — min-zone dissolve + map product export."""

import numpy as np
import pandas as pd
import pytest

from lcz_labels.config import AOI, LczLabelConfig
from lcz_train.postprocess import dissolve_small_zones, write_map_product


def test_small_isolated_prediction_reassigned_to_neighbour():
    # 3 blocks in a row, a-b-c; a,c predicted class 3, each part of a large
    # (>=15 ha) zone; b predicted class 6 alone as a 0.1 ha sliver -> swallowed.
    block_ids = np.array(["a", "b", "c"])
    areas = np.array([2.0e5, 1000.0, 2.0e5])   # a,c = 20 ha each; b = 0.1 ha
    pred = np.array([3, 6, 3])
    adjacency = pd.DataFrame({"block_a": ["a", "b"], "block_b": ["b", "c"]})
    out = dissolve_small_zones(pred, block_ids, areas, adjacency, min_zone_area_ha=15.0)
    assert out[1] == 3    # b's isolated 6-zone reassigned to the majority neighbour (3)
    assert out[0] == 3 and out[2] == 3


def test_large_zone_is_not_dissolved():
    block_ids = np.array(["a", "b"])
    areas = np.array([2.0e5, 2.0e5])   # 20 ha each -> combined 40 ha zone
    pred = np.array([6, 6])
    adjacency = pd.DataFrame({"block_a": ["a"], "block_b": ["b"]})
    out = dissolve_small_zones(pred, block_ids, areas, adjacency, min_zone_area_ha=15.0)
    np.testing.assert_array_equal(out, pred)


def test_unpredicted_blocks_untouched():
    block_ids = np.array(["a", "b"])
    areas = np.array([100.0, 100.0])
    pred = np.array([0, 3])   # a has no prediction
    adjacency = pd.DataFrame({"block_a": ["a"], "block_b": ["b"]})
    out = dissolve_small_zones(pred, block_ids, areas, adjacency, min_zone_area_ha=15.0)
    assert out[0] == 0


def test_small_zone_with_no_valid_neighbour_stays_unchanged():
    block_ids = np.array(["a", "b"])
    areas = np.array([100.0, 100.0])
    pred = np.array([3, 0])    # b has no prediction, so a has no valid neighbour to borrow from
    adjacency = pd.DataFrame({"block_a": ["a"], "block_b": ["b"]})
    out = dissolve_small_zones(pred, block_ids, areas, adjacency, min_zone_area_ha=15.0)
    assert out[0] == 3   # unchanged — nothing to reassign to


def test_no_adjacency_edges_returns_copy_unchanged():
    block_ids = np.array(["a", "b"])
    areas = np.array([100.0, 100.0])
    pred = np.array([3, 6])
    adjacency = pd.DataFrame({"block_a": [], "block_b": []})
    out = dissolve_small_zones(pred, block_ids, areas, adjacency, min_zone_area_ha=15.0)
    np.testing.assert_array_equal(out, pred)
    assert out is not pred


def test_majority_among_multiple_neighbours():
    # star: centre 'x' (small, class 9) surrounded by three class-2 and one class-4
    block_ids = np.array(["x", "n1", "n2", "n3", "n4"])
    areas = np.array([100.0, 2.0e5, 2.0e5, 2.0e5, 2.0e5])
    pred = np.array([9, 2, 2, 2, 4])
    adjacency = pd.DataFrame({
        "block_a": ["x", "x", "x", "x"], "block_b": ["n1", "n2", "n3", "n4"],
    })
    out = dissolve_small_zones(pred, block_ids, areas, adjacency, min_zone_area_ha=15.0)
    assert out[0] == 2   # majority (3 of 4) neighbours are class 2


@pytest.fixture()
def small_cfg(tmp_path):
    c = LczLabelConfig(cache_dir=tmp_path)
    c.aoi_list = [AOI(name="Tiny", bbox=(36.700, -1.350, 36.710, -1.340))]
    return c


def test_write_map_product(small_cfg, tmp_path):
    import geopandas as gpd
    from shapely.geometry import box as sbox

    from lcz_labels.export import raster_grid

    transform, utm, (h, w) = raster_grid("Tiny", small_cfg)
    x0, y1, res = transform.c, transform.f, transform.a
    b1 = sbox(x0 + 10 * res, y1 - 30 * res, x0 + 30 * res, y1 - 10 * res)
    b2 = sbox(x0 + 30 * res, y1 - 30 * res, x0 + 50 * res, y1 - 10 * res)
    blocks_gdf = gpd.GeoDataFrame({"block_id": ["a", "b"]}, geometry=[b1, b2], crs=utm)

    paths = write_map_product(np.array([3, 0]), blocks_gdf, "Tiny", small_cfg, tmp_path / "map")
    assert paths["raster"].exists() and paths["vector"].exists()

    import rasterio
    with rasterio.open(paths["raster"]) as src:
        arr = src.read(1)
        assert arr[20, 20] == 3    # inside block a
        assert arr[20, 40] == 0    # block b unpredicted -> not burned

    out_gdf = gpd.read_parquet(paths["vector"])
    assert list(out_gdf["pred_lcz"]) == [3, 0]
