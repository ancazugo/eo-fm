"""Rasterize OSM evidence layers into 15-band 10 m tiles for input fusion.

For each city (bbox from {city}_grid.gpkg) this enumerates the 0.5-degree
WGS84 tiles intersecting the bbox and writes osm_{lon}_{lat}.tif (uint8,
one binary band per evidence layer, local UTM, 10 m) — the embedding-dir
for `--embedding-name osm_evidence` in the extraction scripts.

Band order (see docs/osm_lcz_tag_mapping.md for the tag rationale):
  0 building     all building=* footprints
  1 building_low footprints with height 0-10 m / levels 1-3 (tagged only)
  2 building_mid height 10-25 m / levels 4-9
  3 building_high height >25 m / levels >=10
  4 roads        highway=motorway..residential, buffered by width/lanes tags
  5 railway      railway=rail|light_rail|tram|yard|station, buffered 6 m
  6 industrial   landuse=industrial|quarry|..., man_made works, power plants, aeroways
  7 commercial   landuse=commercial|retail, large shops
  8 residential  landuse=residential
  9 wood         natural=wood, landuse=forest
 10 scrub        natural=scrub|heath (+deprecated variants)
 11 grass        farmland/meadow/grass/parks/pitches (LCZ D evidence)
 12 water        natural=water, water=*, buffered waterways
 13 bare         natural=sand|beach|bare_rock|... (LCZ E/F evidence)
 14 completeness union of bands 0-13 ("OSM mapped anything here")

Fetches are cached per (tile, tag-group) as GeoParquet, so re-runs and
Overpass hiccups are cheap. `--date` queries the OSM database as of a past
instant (attic data) for temporal ablations.

CAVEAT (campaign doc / extract_osm_features.py): OSM completeness varies by
city — models consuming these bands as input must be gated on HELD-OUT
cities, and training should use aux-channel dropout.

Example:
    python src/build_osm_rasters.py \\
        --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi London \\
        --out-dir ${DATA_DIR}/input/osm_evidence
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

# ── Band / fetch-group spec ───────────────────────────────────────────────────

# Tag queries per fetch group (osmnx convention: keys OR'd, True = any value).
FETCH_GROUPS: dict[str, dict] = {
    "building": {"building": True},
    "roads": {"highway": [
        "motorway", "trunk", "primary", "secondary", "tertiary", "residential",
        "unclassified", "service", "living_street", "pedestrian",
        "motorway_link", "trunk_link", "primary_link", "secondary_link",
        "tertiary_link", "road", "raceway",
    ]},
    "railway": {"railway": ["rail", "light_rail", "tram", "narrow_gauge",
                            "yard", "station", "monorail", "funicular"]},
    "industrial": {
        "landuse": ["industrial", "quarry", "landfill", "railway", "port",
                    "brownfield", "construction"],
        "man_made": ["works", "storage_tank", "silo", "chimney", "gasometer",
                     "wastewater_plant", "water_works", "kiln"],
        "power": ["plant", "substation"],
        "aeroway": ["aerodrome", "terminal", "hangar", "apron", "taxiway", "runway"],
        "industrial": True,
    },
    "commercial": {"landuse": ["commercial", "retail"],
                   "shop": ["mall", "supermarket", "department_store", "wholesale"]},
    "residential": {"landuse": ["residential"]},
    "wood": {"natural": ["wood", "trees"], "landuse": ["forest"]},
    "scrub": {"natural": ["scrub", "heath", "shrub", "moor"],
              "landuse": ["heath", "scrub", "scrubs"]},
    "grass": {
        "landuse": ["farmland", "meadow", "grass", "allotments",
                    "recreation_ground", "greenfield", "village_green",
                    "farm", "pasture", "field", "farmyard"],
        "natural": ["grassland", "fell", "grass", "meadow"],
        "leisure": ["park", "pitch", "golf_course", "garden", "common",
                    "playground"],
        "landcover": ["grass"],
    },
    "water": {"natural": ["water", "bay", "strait"], "water": True,
              "waterway": ["river", "canal", "stream", "drain", "ditch",
                           "riverbank", "dock"],
              "leisure": ["marina", "swimming_pool"],
              "landuse": ["reservoir", "basin", "salt_pond"]},
    "bare": {"natural": ["sand", "beach", "dune", "bare_rock", "scree", "rock",
                         "mud", "shingle", "desert"]},
}

# Non-geometry columns to keep per group (used by filters / width buffering).
KEEP_COLS: dict[str, tuple] = {
    "building": ("height", "building:levels"),
    "roads": ("width", "lanes"),
    "railway": ("width",),
    "water": ("width",),
}

BAND_NAMES = [
    "building", "building_low", "building_mid", "building_high",
    "roads", "railway", "industrial", "commercial", "residential",
    "wood", "scrub", "grass", "water", "bare",
]
N_BANDS = len(BAND_NAMES) + 1   # + completeness


# ── Height helpers (building bucket filters) ──────────────────────────────────

def _parse_num(v) -> float:
    """First numeric value of an OSM tag cell ('12', '12 m', '3;4') → float."""
    try:
        return float(str(v).split(";")[0].strip().rstrip("m").strip())
    except (ValueError, TypeError):
        return float("nan")


def _height_m(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Building height in metres from `height`, else `building:levels` × 3."""
    idx = gdf.index
    h = (gdf["height"].map(_parse_num) if "height" in gdf
         else pd.Series(np.nan, index=idx))
    lv = (gdf["building:levels"].map(_parse_num) if "building:levels" in gdf
          else pd.Series(np.nan, index=idx))
    return h.fillna(lv * 3.0)


def _bucket(lo: float, hi: float):
    def _f(gdf: gpd.GeoDataFrame):
        h = _height_m(gdf)
        return (h > lo) & (h <= hi)
    return _f


# ── Tile grid / CRS helpers ───────────────────────────────────────────────────

def utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def tiles_for_bbox(bbox: tuple, tile_size: float) -> list[tuple[float, float]]:
    """Bottom-left corners of the tile grid intersecting bbox (WGS84)."""
    w, s, e, n = bbox
    lon0 = math.floor(w / tile_size) * tile_size
    lat0 = math.floor(s / tile_size) * tile_size
    out = []
    lat = lat0
    while lat < n:
        lon = lon0
        while lon < e:
            out.append((round(lon, 4), round(lat, 4)))
            lon += tile_size
        lat += tile_size
    return out


# ── Fetch (cached) ────────────────────────────────────────────────────────────

def fetch_group(group: str, bbox: tuple, cache_dir: Path,
                date: str | None) -> gpd.GeoDataFrame:
    """Fetch one tag group for one tile, cached as GeoParquet."""
    lon, lat = bbox[0], bbox[1]
    tag = f"{date.replace(':', '').replace('-', '')}_" if date else ""
    cache = cache_dir / f"{group}_{tag}{lon}_{lat}.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)

    # osm-rasterizer fetches via osmnx, whose response cache defaults to
    # ./cache in the CWD — redirect it next to our parquet cache (/maps;
    # /home is small and fills up).
    import osmnx as ox
    ox.settings.cache_folder = str(cache_dir / "_overpass")

    from osm_rasterizer import fetch_features
    empty = gpd.GeoDataFrame({}, geometry=[], crs="EPSG:4326")
    gdf = None
    for attempt, backoff in enumerate((60, 300, 900), start=1):
        try:
            gdf = fetch_features(bbox, FETCH_GROUPS[group], date=date)
            break
        except Exception as e:  # noqa: BLE001 — empty Overpass responses raise
            if "InsufficientResponse" in type(e).__name__:
                gdf = empty
                break
            # Connection refused / rate limiting: wait it out instead of
            # cascading fast failures through the whole tile queue.
            if attempt == 3:
                raise
            logger.warning(f"{group} ({lon},{lat}) fetch failed "
                           f"({type(e).__name__}) — retry in {backoff}s")
            import time
            time.sleep(backoff)
    if len(gdf):
        keep = [c for c in KEEP_COLS.get(group, ()) if c in gdf.columns]
        gdf = gdf[keep + ["geometry"]].reset_index(drop=True)
    else:
        gdf = empty
    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache)
    return gdf


# ── Rasterize one tile ────────────────────────────────────────────────────────

def build_tile(lon: float, lat: float, tile_size: float, resolution: float,
               cache_dir: Path, out_path: Path, date: str | None) -> dict:
    """Fetch + rasterize one 0.5° tile → 15-band uint8 GeoTIFF."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from osm_rasterizer import rasterize

    bbox = (lon, lat, round(lon + tile_size, 4), round(lat + tile_size, 4))
    gdfs = {g: fetch_group(g, bbox, cache_dir, date) for g in FETCH_GROUPS}

    # Explicit UTM grid so every group (even all-empty ocean tiles) yields
    # the same deterministic raster footprint.
    crs = utm_epsg(lon + tile_size / 2, lat + tile_size / 2)
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = t.transform([bbox[0], bbox[2], bbox[0], bbox[2]],
                         [bbox[1], bbox[1], bbox[3], bbox[3]])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    W = max(1, int(round((maxx - minx) / resolution)))
    H = max(1, int(round((maxy - miny) / resolution)))
    transform = from_origin(minx, maxy, resolution, resolution)

    bld = gdfs["building"]
    features = [
        ("building", bld, {}),
        ("building_low", bld, {"filter": _bucket(0.0, 10.0)}),
        ("building_mid", bld, {"filter": _bucket(10.0, 25.0)}),
        ("building_high", bld, {"filter": _bucket(25.0, 1000.0)}),
        ("roads", gdfs["roads"], {"width_from_tags": True, "line_width": 8.0}),
        ("railway", gdfs["railway"], {"line_width": 6.0}),
        ("industrial", gdfs["industrial"], {"line_width": 15.0}),
        ("commercial", gdfs["commercial"], {}),
        ("residential", gdfs["residential"], {}),
        ("wood", gdfs["wood"], {}),
        ("scrub", gdfs["scrub"], {}),
        ("grass", gdfs["grass"], {}),
        ("water", gdfs["water"], {"width_from_tags": True, "line_width": 10.0}),
        ("bare", gdfs["bare"], {}),
    ]
    present = [(n, g, o) for n, g, o in features if len(g)]

    arr = np.zeros((N_BANDS, H, W), dtype=np.uint8)
    if present:
        res = rasterize(bbox, present, transform=transform, crs=crs)
        for i, name in enumerate(res.band_names):
            band = (res.array[i] > 0).astype(np.uint8)
            arr[BAND_NAMES.index(name)] = band[:H, :W]
    arr[-1] = arr[:-1].max(axis=0)   # completeness = union

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(out_path), "w", driver="GTiff", height=H, width=W, count=N_BANDS,
        dtype="uint8", crs=crs, transform=transform, nodata=None,
        tiled=True, blockxsize=256, blockysize=256, compress="DEFLATE",
    ) as dst:
        dst.write(arr)
        dst.descriptions = tuple(BAND_NAMES + ["completeness"])

    counts = {n: int(arr[i].sum()) for i, n in enumerate(BAND_NAMES)}
    return counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize OSM evidence layers into 10 m tiles per city."
    )
    parser.add_argument("--cities-dir", required=True, type=Path)
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Root output dir; tiles go to {out-dir}/tiles, "
                             "fetch caches to {out-dir}/cache.")
    parser.add_argument("--tile-size", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--date", default=None,
                        help="Optional ISO date — query OSM as of this instant "
                             "(temporal ablation).")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    tile_dir = args.out_dir / "tiles"
    cache_dir = args.out_dir / "cache"

    # Union of tiles across cities (cities share tiles when bboxes overlap)
    todo: dict[tuple, str] = {}
    for city in args.cities:
        grid_gpkg = args.cities_dir / city / f"{city}_grid.gpkg"
        if not grid_gpkg.exists():
            logger.warning(f"{city}: {grid_gpkg.name} missing — skipping")
            continue
        bbox = tuple(gpd.read_file(grid_gpkg).to_crs("EPSG:4326").total_bounds)
        for lonlat in tiles_for_bbox(bbox, args.tile_size):
            todo.setdefault(lonlat, city)
    logger.info(f"{len(todo)} unique tiles across {len(args.cities)} cities")

    n_ok = n_skip = n_fail = 0
    for i, ((lon, lat), city) in enumerate(sorted(todo.items())):
        out_path = tile_dir / f"osm_{lon}_{lat}.tif"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue
        logger.info(f"[{i + 1}/{len(todo)}] tile ({lon}, {lat}) [{city}] …")
        try:
            counts = build_tile(lon, lat, args.tile_size, args.resolution,
                                cache_dir, out_path, args.date)
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
            logger.info("  wrote " + out_path.name + "  top bands: "
                        + ", ".join(f"{n}={c}" for n, c in top))
            n_ok += 1
        except Exception:  # noqa: BLE001
            logger.exception(f"  tile ({lon}, {lat}) FAILED — continuing")
            n_fail += 1
    logger.info(f"Done: {n_ok} written, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
