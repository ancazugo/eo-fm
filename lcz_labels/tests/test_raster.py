"""GHS-BUILT-S zonal test — proves the exactextract path works on a north-up tile."""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from lcz_labels.config import LczLabelConfig
from lcz_labels.ucp import ghs_built_s_fraction


def test_ghs_built_s_fraction(tmp_path):
    cfg = LczLabelConfig()
    cfg.rasters.ghs_built_s_dir = tmp_path
    # North-up tile keyed (0.0, 0.0); built_surface = 5000 m^2 per cell -> frac 0.5
    data = np.full((100, 100), 5000.0, dtype="float32")
    transform = from_origin(0.0, 0.5, 0.005, 0.005)
    with rasterio.open(
        tmp_path / "builts_0.0_0.0.tif", "w", driver="GTiff",
        height=100, width=100, count=1, dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=-999,
    ) as dst:
        dst.write(data, 1)

    # Two small patches inside the tile (EPSG:4326 grid, as compute_ucp passes)
    patches = gpd.GeoDataFrame(
        {"patch_id": ["0", "1"]},
        geometry=[box(0.10, 0.10, 0.11, 0.11), box(0.30, 0.30, 0.31, 0.31)],
        crs="EPSG:4326",
    )
    fracs = ghs_built_s_fraction(patches, cfg)
    assert np.allclose(fracs, 0.5, atol=1e-6)
