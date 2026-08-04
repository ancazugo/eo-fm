"""Stage 4a — block delineation, sliver merge, mega-block fallback, adjacency.

All offline: synthetic road/rail/water layers in a metric CRS, plus the
committed ~1.5 km² Nairobi Overture fixture (EPSG:32737).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, box

from lcz_labels.blocks import (
    _fallback_cells,
    _merge_small_blocks,
    assemble_barriers,
    block_id_hash,
    build_adjacency,
    build_blocks,
    delineate,
)
from lcz_labels.config import AOI, LczLabelConfig
from lcz_labels.overture import OvertureExtract

UTM = "EPSG:32737"
FIXTURES = Path(__file__).parent / "fixtures"


def _gdf(geoms, crs=UTM, **cols):
    return gpd.GeoDataFrame(cols, geometry=list(geoms), crs=crs)


def _road_grid(step=100, extent=300):
    """Orthogonal residential-road grid: (extent/step)^2 enclosures of step^2 m2."""
    lines = [LineString([(x, 0), (x, extent)]) for x in range(0, extent + 1, step)]
    lines += [LineString([(0, y), (extent, y)]) for y in range(0, extent + 1, step)]
    return _gdf(lines, **{"class": ["residential"] * len(lines)})


def _extract(roads=None, rail=None, landcover=None, mn=None):
    empty = _gdf([])
    return OvertureExtract(
        buildings=empty, landcover=landcover if landcover is not None else empty,
        infrastructure=empty, utm_crs=UTM, roads=roads, rail=rail, mn_blocks=mn,
    )


@pytest.fixture()
def cfg(tmp_path):
    c = LczLabelConfig(cache_dir=tmp_path)
    return c


LIMIT = box(0, 0, 300, 300)


def test_enclosures_from_road_grid(cfg):
    blocks = delineate(_extract(roads=_road_grid()), LIMIT, cfg, utm=UTM)
    assert len(blocks) == 9
    assert (blocks["block_kind"] == "enclosure").all()
    np.testing.assert_allclose(blocks.geometry.area, 10_000, rtol=1e-6)


def test_nonmotorized_roads_are_not_barriers(cfg):
    roads = _road_grid(step=400, extent=1200)
    roads.loc[:, "class"] = "footway"
    limit = box(0, 0, 1200, 1200)
    blocks = delineate(_extract(roads=roads), limit, cfg, utm=UTM)
    # No barriers -> the AOI is one mega-block -> 320 m grid fallback.
    assert (blocks["block_kind"] == "grid_fallback").all()
    assert abs(blocks.geometry.area.sum() - limit.area) < 1.0


def test_sliver_merged_into_neighbor(cfg):
    roads = _road_grid()
    # Diagonal clip of one corner of cell (0-100, 0-100): ~50 m2 triangle < 200 m2.
    roads = gpd.GeoDataFrame(
        {"class": list(roads["class"]) + ["residential"]},
        geometry=list(roads.geometry) + [LineString([(0, 10), (10, 0)])], crs=UTM,
    )
    blocks = delineate(_extract(roads=roads), LIMIT, cfg, utm=UTM)
    assert len(blocks) == 9  # triangle absorbed, not a 10th block
    assert abs(blocks.geometry.area.sum() - LIMIT.area) < 1.0


def test_mega_block_grid_fallback(cfg):
    # Only the outer frame is roads -> one 1200x1200 m enclosure > 0.5 km2.
    frame = box(0, 0, 1200, 1200).boundary
    roads = _gdf([frame], **{"class": ["primary"]})
    blocks = delineate(_extract(roads=roads), box(0, 0, 1200, 1200), cfg, utm=UTM)
    assert (blocks["block_kind"] == "grid_fallback").all()
    # 1200/320 -> 4 cells per axis (snapped origin at 0), 16 pieces
    assert len(blocks) == 16
    assert abs(blocks.geometry.area.sum() - 1200 * 1200) < 1.0


def test_water_barrier_subdivides(cfg):
    # 200 m cells so the ring around the pond stays wider than the corridor bar.
    cfg.blocks.barrier_water_min_m2 = 1000.0  # pond must fit inside one cell
    limit = box(0, 0, 600, 600)
    water = _gdf([box(270, 270, 330, 330)], **{
        "class": ["water"], "base_type": ["water"],
    })
    blocks = delineate(
        _extract(roads=_road_grid(step=200, extent=600), landcover=water),
        limit, cfg, utm=UTM,
    )
    assert len(blocks) == 10  # the middle cell splits into pond + remainder
    assert abs(blocks.geometry.area.sum() - limit.area) < 1.0


def test_small_water_is_not_a_barrier(cfg):
    # 60x60 = 3600 m2 < barrier_water_min_m2 (10 000) when polygon is small
    water = _gdf([box(140, 140, 160, 160)], **{"class": ["water"], "base_type": ["water"]})
    blocks = delineate(_extract(roads=_road_grid(), landcover=water), LIMIT, cfg, utm=UTM)
    assert len(blocks) == 9


def test_rail_barrier_and_class_filter(cfg):
    rail = _gdf(
        [LineString([(150, 0), (150, 300)]), LineString([(0, 150), (300, 150)])],
        **{"class": ["rail", "subway"]},
    )
    blocks = delineate(_extract(roads=_gdf([LIMIT.boundary], **{"class": ["primary"]}),
                                rail=rail), LIMIT, cfg, utm=UTM)
    # Surface rail splits the frame in two; the subway line is ignored.
    assert len(blocks) == 2
    assert (blocks["block_kind"] == "enclosure").all()


def test_large_landcover_boundary_is_barrier(cfg):
    forest = _gdf([box(0, 0, 300, 260)], **{"class": ["forest"], "base_type": ["land"]})
    primary = _gdf([LIMIT.boundary], **{"class": ["primary"]})
    blocks = delineate(_extract(roads=primary, landcover=forest), LIMIT, cfg, utm=UTM)
    assert len(blocks) == 2  # forest vs the 40 m strip above it


def test_fallback_cells_snapped():
    cells = _fallback_cells(box(10, 10, 650, 330), 320.0)
    # snapped origin at 0: x tiles [0,320,640), y tiles [0,320) -> 3x2 pieces
    assert len(cells) == 6
    assert abs(sum(shapely.area(c) for c in cells) - 640 * 320) < 1.0


def test_merge_small_drops_isolated():
    blocks = _gdf([box(0, 0, 100, 100), box(500, 500, 505, 505)])
    out = _merge_small_blocks(blocks, min_area=200.0, corridor_width=20.0, drop_area=200.0)
    assert len(out) == 1


def test_dual_carriageway_median_is_merged(cfg):
    # Paired centerlines 15 m apart create a 300 x 15 m median: 4500 m2 (above
    # min_block_area) but thinner than the corridor bar -> merged, not kept.
    roads = _road_grid()
    extra = gpd.GeoDataFrame(
        {"class": ["primary"]}, geometry=[LineString([(0, 115), (300, 115)])], crs=UTM,
    )
    roads = gpd.GeoDataFrame(pd.concat([roads, extra], ignore_index=True), crs=UTM)
    blocks = delineate(_extract(roads=roads), LIMIT, cfg, utm=UTM)
    assert len(blocks) == 9  # median absorbed into an adjacent cell
    assert abs(blocks.geometry.area.sum() - LIMIT.area) < 1.0
    # nothing thinner than the corridor bar survives
    assert not shapely.is_empty(shapely.buffer(blocks.geometry.values, -10.0)).any()


def test_small_block_above_sliver_floor_is_merged(cfg):
    # A 40 x 40 m pocket (1600 m2: > sliver floor, < min_block_area) merges.
    roads = _road_grid()
    extra = gpd.GeoDataFrame(
        {"class": ["primary", "primary"]},
        geometry=[LineString([(0, 40), (40, 40), (40, 0)])] * 1
        + [LineString([(40, 40), (40, 0)])],
        crs=UTM,
    )
    roads = gpd.GeoDataFrame(pd.concat([roads, extra], ignore_index=True), crs=UTM)
    blocks = delineate(_extract(roads=roads), LIMIT, cfg, utm=UTM)
    assert len(blocks) == 9
    assert abs(blocks.geometry.area.sum() - LIMIT.area) < 1.0


def test_block_id_stable_and_per_aoi():
    g = box(0, 0, 100, 100)
    assert block_id_hash("Nairobi", g) == block_id_hash("Nairobi", g)
    assert block_id_hash("Nairobi", g) != block_id_hash("Paris", g)
    assert len(block_id_hash("Nairobi", g)) == 16


def test_mn_substitution_inside_mega_block(cfg):
    cfg.blocks.use_mn_blocks = True
    mn = _gdf([box(0, 0, 600, 1200)], informal=[True])
    roads = _gdf([box(0, 0, 1200, 1200).boundary], **{"class": ["primary"]})
    blocks = delineate(_extract(roads=roads, mn=mn), box(0, 0, 1200, 1200), cfg, utm=UTM)
    kinds = set(blocks["block_kind"])
    assert kinds == {"mn", "grid_fallback"}
    mn_area = blocks[blocks["block_kind"] == "mn"].geometry.area.sum()
    assert abs(mn_area - 600 * 1200) < 1.0
    assert abs(blocks.geometry.area.sum() - 1200 * 1200) < 1.0


def test_adjacency_on_road_grid(cfg):
    ex = _extract(roads=_road_grid())
    aoi = AOI(name="Synthetic", bbox=None)
    blocks = delineate(ex, LIMIT, cfg, utm=UTM)
    blocks["block_id"] = [block_id_hash("Synthetic", g) for g in blocks.geometry.values]
    adj = build_adjacency(blocks, cfg, "Synthetic")
    # 3x3 rook grid: 12 undirected edges, each sharing a 100 m edge
    assert len(adj) == 12
    np.testing.assert_allclose(adj["shared_len_m"], 100.0, rtol=1e-6)
    assert (adj["block_a"] < adj["block_b"]).all()
    deg = (adj["block_a"].value_counts().add(adj["block_b"].value_counts(), fill_value=0))
    assert sorted(deg.astype(int)) == [2, 2, 2, 2, 3, 3, 3, 3, 4]


def test_adjacency_strict_catches_vertex_mismatch(cfg):
    # B and C share only part of A's right edge, with no common vertices on A.
    A = shapely.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    B = shapely.Polygon([(100, 10), (200, 10), (200, 60), (100, 60)])
    C = shapely.Polygon([(100, 60), (200, 60), (200, 90), (100, 90)])
    blocks = _gdf([A, B, C], block_id=["a", "b", "c"])
    adj = build_adjacency(blocks, cfg, "VertexMismatch")
    pairs = set(map(tuple, adj[["block_a", "block_b"]].to_numpy()))
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


def test_build_blocks_nairobi_fixture(cfg):
    roads = gpd.read_parquet(FIXTURES / "overture_roads.parquet")
    landcover = gpd.read_parquet(FIXTURES / "overture_landcover.parquet")
    grid = gpd.read_parquet(FIXTURES / "grid_fixture.parquet")
    bbox = tuple(grid.total_bounds)
    cfg.aoi_list = [AOI(name="NairobiFixture", bbox=bbox)]
    ex = OvertureExtract(
        buildings=_gdf([]), landcover=landcover, infrastructure=_gdf([]),
        utm_crs=str(roads.crs), roads=roads, rail=None,
    )
    blocks = build_blocks("NairobiFixture", ex, cfg)
    assert len(blocks) > 20
    assert set(blocks.columns) == {"block_id", "block_idx", "block_kind", "area_m2", "aoi", "geometry"}
    assert blocks["block_id"].is_unique
    np.testing.assert_array_equal(blocks["block_idx"], np.arange(1, len(blocks) + 1))
    assert set(blocks["block_kind"]) <= {"enclosure", "grid_fallback", "mn"}
    # Enclosures tile the limit: total block area ~ AOI bbox area in UTM
    limit_utm = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(roads.crs).iloc[0].envelope
    assert abs(blocks["area_m2"].sum() - limit_utm.area) / limit_utm.area < 0.02
    # cache round-trip
    again = build_blocks("NairobiFixture", ex, cfg)
    assert (again["block_id"] == blocks["block_id"]).all()
    # adjacency runs on the real fixture too
    adj = build_adjacency(blocks, cfg, "NairobiFixture")
    assert len(adj) >= len(blocks)  # planar tiling: edges >= nodes for 20+ blocks
    assert (adj["shared_len_m"] > 0).all()


def test_leaked_barrier_does_not_produce_enclosure_outside_limit(cfg):
    # Regression: momepy.enclosures polygonizes primary+additional barriers
    # together with `limit`'s own boundary line. Real road/rail segments
    # extend beyond the AOI bbox, so a barrier that starts on the limit
    # boundary, loops far outside, and closes back on the boundary forms one
    # huge enclosed face OUTSIDE the true AOI — momepy returns it unfiltered
    # unless called with clip=True (and even then only filters by
    # representative-point containment, not a geometric clip). Without both
    # clip=True and the explicit intersection, this "leak" becomes a single
    # ~100 km2 grid_fallback mega-block far outside Manchester's real extent
    # (observed on real data: 4.79M leaked cells spanning ~914,000 km2).
    roads = _road_grid()
    leak = LineString([(150, 300), (150, 10300), (10450, 10300), (10450, 300), (250, 300)])
    roads = gpd.GeoDataFrame(
        {"class": list(roads["class"]) + ["residential"]},
        geometry=list(roads.geometry) + [leak], crs=UTM,
    )
    blocks = delineate(_extract(roads=roads), LIMIT, cfg, utm=UTM)
    minx, miny, maxx, maxy = blocks.total_bounds
    assert minx >= -1e-6 and miny >= -1e-6
    assert maxx <= 300 + 1e-6 and maxy <= 300 + 1e-6
    assert blocks.geometry.area.sum() <= LIMIT.area + 1e-6


def test_assemble_barriers_prefers_config_classes(cfg):
    roads = _road_grid()
    rail = _gdf([LineString([(50, 0), (50, 300)])], **{"class": ["subway"]})
    primary, additional = assemble_barriers(_extract(roads=roads, rail=rail), cfg)
    # subway excluded -> only the 8 road lines (exploded) remain
    assert len(primary) == 8
    assert additional == []
