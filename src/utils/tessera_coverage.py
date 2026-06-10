#!/usr/bin/env python3
"""Generate a GeoPackage of tessera v1.1 global tile footprints from directory names."""

import re
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

TILES_DIR = Path("/tessera/v1.1/global_0.1_degree_representation/2017")
OUT_GPKG = Path("data/tessera_v1.1_global_2017_tiles.gpkg")

pattern = re.compile(r"^grid_(-?\d+\.\d+)_(-?\d+\.\d+)$")

records = []
for d in sorted(TILES_DIR.iterdir()):
    if not d.is_dir():
        continue
    m = pattern.match(d.name)
    if m:
        lon, lat = float(m.group(1)), float(m.group(2))
        geom = box(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
        records.append({"lon_center": lon, "lat_center": lat, "tile_name": d.name, "geometry": geom})

gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
gdf.to_file(OUT_GPKG, driver="GPKG")
print(f"Wrote {len(gdf)} tiles to {OUT_GPKG}")
