"""Height-tiering tests, including OSM/Esri trust gating and raster backfill."""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from lcz_labels.config import LczLabelConfig
from lcz_labels.heights import compute_heights, sample_ghs_built_h

UTM = "EPSG:32737"  # Nairobi


def _buildings(records):
    geoms = [box(0, 0, 10, 10)] * len(records)  # 100 m^2 footprints, placement irrelevant
    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=UTM)
    return gdf


def test_height_tiers(tmp_path):
    cfg = LczLabelConfig()
    cfg.rasters.ghs_built_h_dir = tmp_path  # empty -> raster tier unavailable
    b = _buildings([
        dict(height=50.0, num_floors=None, roof_height=None, is_osm_or_esri=True, **{"class": None}),   # explicit
        dict(height=50.0, num_floors=None, roof_height=None, is_osm_or_esri=False, **{"class": None}),  # untrusted -> none
        dict(height=None, num_floors=5.0, roof_height=None, is_osm_or_esri=True, **{"class": None}),     # levels 16m
        dict(height=1500.0, num_floors=3.0, roof_height=None, is_osm_or_esri=True, **{"class": None}),   # insane -> levels
        dict(height=None, num_floors=None, roof_height=None, is_osm_or_esri=True, **{"class": None}),    # none
    ])
    out = compute_heights(b, cfg)
    assert list(out["height_tier"]) == ["explicit", "none", "levels", "levels", "none"]
    assert out["height_m"].iloc[0] == 50.0
    assert out["height_m"].iloc[2] == 5 * cfg.metres_per_floor
    assert np.isnan(out["height_m"].iloc[1])
    assert out["footprint_area_m2"].iloc[0] == 100.0


def test_type_flags():
    cfg = LczLabelConfig()
    cfg.rasters.ghs_built_h_dir = "/nonexistent"
    b = _buildings([
        dict(height=8.0, num_floors=None, roof_height=None, is_osm_or_esri=True, **{"class": "warehouse"}),
        dict(height=80.0, num_floors=None, roof_height=None, is_osm_or_esri=True, **{"class": "skyscraper"}),
    ])
    out = compute_heights(b, cfg)
    assert bool(out["is_large_lowrise_type"].iloc[0]) is True
    assert bool(out["is_tower_type"].iloc[1]) is True


def test_sample_ghs_built_h_point(tmp_path):
    """A synthetic north-up GHS-BUILT-H tile is sampled at the right point."""
    cfg = LczLabelConfig()
    cfg.rasters.ghs_built_h_dir = tmp_path
    # Tile covering lon in [0,0.5], lat in [0,0.5]; key = (0.0, 0.0)
    data = np.full((50, 50), 12.0, dtype="float32")
    transform = from_origin(0.0, 0.5, 0.01, 0.01)  # north-up (0.5 top -> 0.0 bottom)
    with rasterio.open(
        tmp_path / "builth_0.0_0.0.tif", "w", driver="GTiff",
        height=50, width=50, count=1, dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=-999,
    ) as dst:
        dst.write(data, 1)
    pts = np.array([[0.25, 0.25], [10.0, 10.0]])  # first in tile, second off-grid
    vals = sample_ghs_built_h(pts, cfg)
    assert vals[0] == 12.0
    assert np.isnan(vals[1])
