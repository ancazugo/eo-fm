"""Fetch OSM features and burn them into a LCZ raster label file.

Implements the unambiguous OSM → LCZ tag mappings from docs/osm_lcz_tag_mapping.md:
only land-cover classes that can be derived from a single OSM tag without building
height or density analysis.  Built types (LCZ 1–9) are excluded for that reason.

LCZ class assignments (see utils/constants.py lcz_dict):
    10  Heavy Industry   — landuse=industrial, man_made=works/*, power=plant
    11  Dense Trees      — natural=wood, landuse=forest
    13  Bush / Scrub     — natural=scrub, natural=heath
    14  Low Plants       — natural=grassland, landuse=farmland/meadow/grass
    15  Bare Rock/Paved  — natural=bare_rock/scree, aeroway=runway/taxiway/apron
    16  Bare Soil / Sand — natural=sand/beach/dune/mud
    17  Water            — natural=water, waterway=*(buffered), landuse=reservoir/basin

Burn priority (last burn wins, water always wins):
    10 → 14 → 13 → 11 → 16 → 15 → 17

Usage
-----
    python src/datasets/osm_lcz_labels.py generate \\
        --ref-path /data/So2Sat-LCZ42/v4/cities/Nairobi/patches_reference_Nairobi.tif \\
        --date "2017-01-01" \\
        --classes "11,17" \\
        --output-path /data/So2Sat-LCZ42/v4/cities/Nairobi/osm_labels/osm_lcz_2017.tif

    # Output defaults to <ref_path.parent>/osm_labels/osm_lcz_<date>.tif
    python src/datasets/osm_lcz_labels.py generate \\
        --ref-path /data/So2Sat-LCZ42/v4/cities/Nairobi/patches_reference_Nairobi.tif \\
        --date "2017-01-01"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import numpy as np
import typer
from loguru import logger
from shapely.geometry import box as shapely_box

# ── OSM tag rules ─────────────────────────────────────────────────────────────
# Structure: {lcz_class: [(tags_query, buffer_m), ...]}
#   - tags_query : dict passed to osm_rasterizer.fetch_features
#   - buffer_m   : buffer radius in CRS metres applied to non-polygon geometries
#                  (None → skip non-polygons entirely, 0 → do not buffer)

LCZ_OSM_RULES: dict[int, list[tuple[dict, float | None]]] = {
    # ── Built type ─────────────────────────────────────────────────────────────
    10: [  # Heavy Industry
        ({"landuse": "industrial"}, None),
        ({"man_made": ["works", "wastewater_plant", "water_works"]}, None),
        ({"power": "plant"}, None),
    ],
    # ── Land cover types ───────────────────────────────────────────────────────
    11: [  # Dense Trees (A)
        ({"natural": "wood"}, None),
        ({"landuse": "forest"}, None),
        ({"natural": "trees"}, None),          # deprecated but present in historic data
    ],
    13: [  # Bush / Scrub (C)
        ({"natural": ["scrub", "heath"]}, None),
        ({"landuse": ["heath", "scrub"]}, None),  # deprecated
    ],
    14: [  # Low Plants (D)
        ({"natural": "grassland"}, None),
        ({"landuse": ["farmland", "meadow", "grass"]}, None),
        ({"natural": ["grass", "meadow"]}, None),  # deprecated
    ],
    15: [  # Bare Rock / Paved (E)
        ({"natural": ["bare_rock", "scree"]}, None),
        ({"aeroway": ["runway", "taxiway", "apron"]}, 15.0),  # can be area or centreline
    ],
    16: [  # Bare Soil / Sand (F)
        ({"natural": ["sand", "beach", "dune", "mud", "shingle"]}, None),
    ],
    17: [  # Water (G)
        ({"natural": "water"}, None),
        ({"natural": "bay"}, None),
        # Deprecated area tags, commonly found in pre-2015 data
        ({"landuse": ["reservoir", "basin"]}, None),
        ({"waterway": ["riverbank"]}, None),      # deprecated; river polygon
        # Linear waterways — buffered to create area
        (
            {"waterway": ["river", "canal", "stream", "drain", "ditch", "dock"]},
            30.0,
        ),
    ],
}

# Burn order: lowest priority first (last entry in list wins when shapes overlap)
_BURN_ORDER = [10, 14, 13, 11, 16, 15, 17]

app = typer.Typer(pretty_exceptions_enable=False)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_polygon_like(geom) -> bool:
    return geom.geom_type in ("Polygon", "MultiPolygon")


def _fetch_one(
    bbox_wgs84: tuple[float, float, float, float],
    tags: dict,
    target_crs,
    date: str | None,
    buffer_m: float | None,
    lcz_class: int,
    label_col: str,
) -> gpd.GeoDataFrame | None:
    """Fetch a single tag set and return a GeoDataFrame in target_crs, or None."""
    from osm_rasterizer import fetch_features

    try:
        gdf = fetch_features(bbox_wgs84, tags, date=date)
    except Exception as exc:
        logger.warning(f"  fetch_features({tags}) failed: {exc}")
        return None

    if gdf is None or gdf.empty:
        return None

    gdf = gdf.to_crs(target_crs)[["geometry"]].copy()

    if buffer_m is not None and buffer_m > 0:
        # Buffer non-polygon geometries (lines); keep polygons as-is
        poly_mask = gdf.geometry.apply(_is_polygon_like)
        lines = gdf[~poly_mask].copy()
        polys = gdf[poly_mask].copy()
        if not lines.empty:
            lines["geometry"] = lines.geometry.buffer(buffer_m)
            gdf = gpd.GeoDataFrame(
                gpd.pd.concat([polys, lines], ignore_index=True),
                geometry="geometry",
                crs=target_crs,
            )
        else:
            gdf = polys
    else:
        # No buffer requested — keep only polygon geometries
        gdf = gdf[gdf.geometry.apply(_is_polygon_like)].copy()

    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return None

    gdf[label_col] = lcz_class
    return gdf[["geometry", label_col]]


def fetch_lcz_class_features(
    bbox_wgs84: tuple[float, float, float, float],
    lcz_class: int,
    target_crs,
    date: str | None = None,
    label_col: str = "lcz_class",
    rules: dict | None = None,
) -> gpd.GeoDataFrame:
    """Fetch all OSM features for a single LCZ class.

    Args:
        bbox_wgs84: (minx, miny, maxx, maxy) in EPSG:4326.
        lcz_class: Integer LCZ class (e.g. 17 for Water).
        target_crs: Projected CRS for reprojection and buffering.
        date: Optional ISO 8601 date string for OSM historical snapshot.
        label_col: Output column name for the integer class value.
        rules: Tag rules dict. Defaults to LCZ_OSM_RULES.

    Returns:
        GeoDataFrame with [geometry, label_col] in target_crs.  May be empty.
    """
    import pandas as pd

    if rules is None:
        rules = LCZ_OSM_RULES

    if lcz_class not in rules:
        raise ValueError(
            f"LCZ class {lcz_class} not in rules. Available: {sorted(rules)}"
        )

    gdfs = []
    for tags, buffer_m in rules[lcz_class]:
        result = _fetch_one(bbox_wgs84, tags, target_crs, date, buffer_m, lcz_class, label_col)
        if result is not None and not result.empty:
            gdfs.append(result)
            logger.info(f"  LCZ {lcz_class:>2} | tags={tags} → {len(result)} features")

    if not gdfs:
        return gpd.GeoDataFrame(columns=["geometry", label_col], crs=target_crs)

    merged = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=target_crs
    )
    # Drop duplicate geometries that may appear across tag variants
    merged = merged.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
    return merged


def fetch_all_lcz_features(
    bbox_wgs84: tuple[float, float, float, float],
    lcz_classes: list[int],
    target_crs,
    date: str | None = None,
    label_col: str = "lcz_class",
    rules: dict | None = None,
) -> gpd.GeoDataFrame:
    """Fetch OSM features for all requested LCZ classes.

    Features are concatenated in burn priority order: lower-confidence classes
    first so that water (class 17) overwrites earlier burns when shapes overlap.

    Args:
        bbox_wgs84: (minx, miny, maxx, maxy) in EPSG:4326.
        lcz_classes: LCZ class integers to fetch.  Must be present in rules.
        target_crs: Projected CRS for reprojection and buffering.
        date: Optional ISO 8601 date string for OSM historical snapshot.
        label_col: Output column name for the integer class value.
        rules: Tag rules dict. Defaults to LCZ_OSM_RULES.

    Returns:
        GeoDataFrame with [geometry, label_col] in target_crs, ordered so that
        higher-confidence classes appear last (water wins on overlap).
    """
    import pandas as pd

    if rules is None:
        rules = LCZ_OSM_RULES

    # Process in burn priority order; only include requested classes
    ordered = [c for c in _BURN_ORDER if c in lcz_classes]
    # Any requested classes not in _BURN_ORDER appended at end
    ordered += [c for c in lcz_classes if c not in _BURN_ORDER]

    all_gdfs = []
    for cls in ordered:
        logger.info(f"Fetching LCZ {cls} ({_lcz_name(cls)}) …")
        gdf = fetch_lcz_class_features(bbox_wgs84, cls, target_crs, date, label_col, rules)
        if not gdf.empty:
            all_gdfs.append(gdf)
            logger.info(f"  → {len(gdf)} features total for LCZ {cls}")
        else:
            logger.info(f"  → no features found for LCZ {cls}")

    if not all_gdfs:
        return gpd.GeoDataFrame(columns=["geometry", label_col], crs=target_crs)

    return gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True), geometry="geometry", crs=target_crs
    )


def burn_osm_lcz_raster(
    features_gdf: gpd.GeoDataFrame,
    label_col: str,
    output_path: str | Path,
    res: float,
    snap_bounds: tuple[float, float, float, float] | None = None,
    nodata: int = 0,
) -> Path:
    """Burn OSM LCZ features into a GeoTIFF raster.

    Polygons are burned in GeoDataFrame row order (last row wins on overlap).
    Call fetch_all_lcz_features to get features already sorted in priority order.

    Args:
        features_gdf: GeoDataFrame with [geometry, label_col]. Must be in a
            projected CRS (metres) for correct pixel sizing.
        label_col: Column holding integer LCZ class values.
        output_path: Output GeoTIFF path (.tif).
        res: Pixel size in CRS units (metres).
        snap_bounds: Optional (minx, miny, maxx, maxy) to snap the raster grid
            to an existing reference raster.  If None, uses features_gdf.total_bounds.
        nodata: Fill value for uncovered pixels (0 = no label).

    Returns:
        Path to the written GeoTIFF.
    """
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_origin

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if snap_bounds is not None:
        minx, miny, maxx, maxy = snap_bounds
    else:
        minx, miny, maxx, maxy = features_gdf.total_bounds

    width  = max(1, int(np.ceil((maxx - minx) / res)))
    height = max(1, int(np.ceil((maxy - miny) / res)))
    maxy_snap = miny + height * res
    transform = from_origin(minx, maxy_snap, res, res)

    shapes = (
        (geom.__geo_interface__, int(val))
        for geom, val in zip(features_gdf.geometry, features_gdf[label_col])
        if geom is not None and not geom.is_empty
    )
    burned = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=np.uint8,
    )

    crs = features_gdf.crs
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(burned, 1)

    logger.info(
        f"Wrote OSM LCZ raster: {output_path}  "
        f"({width}×{height} px, res={res}m, "
        f"unique classes: {np.unique(burned[burned != nodata]).tolist()})"
    )
    return output_path


def _lcz_name(lcz_class: int) -> str:
    try:
        from utils.constants import lcz_dict
        return lcz_dict[lcz_class]["name"]
    except Exception:
        return f"Class {lcz_class}"


def _read_ref_info(
    ref_path: str | Path,
) -> tuple[tuple[float, float, float, float], object, float]:
    """Read (bbox_projected, crs, res) from a .tif or .gpkg reference file.

    Returns:
        (bounds_in_crs, crs, pixel_res_m)
    """
    ref_path = Path(ref_path)

    if ref_path.suffix.lower() in (".tif", ".tiff"):
        import rasterio

        with rasterio.open(ref_path) as src:
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            crs = src.crs
            res = src.res[0]
        return bounds, crs, res

    # Vector file (.gpkg, .geojson, .shp)
    gdf = gpd.read_file(ref_path)
    bounds = tuple(gdf.total_bounds)  # type: ignore[assignment]
    crs = gdf.crs
    # No native resolution — return None; caller must supply --res
    return bounds, crs, None  # type: ignore[return-value]


def _bounds_to_wgs84(
    bounds: tuple[float, float, float, float],
    crs,
) -> tuple[float, float, float, float]:
    """Convert projected bounds to WGS84 (minx, miny, maxx, maxy)."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(bounds[0], bounds[1])
    maxx, maxy = transformer.transform(bounds[2], bounds[3])
    return (minx, miny, maxx, maxy)


def _reproject_bounds(
    bounds: tuple[float, float, float, float],
    src_crs,
    dst_crs,
) -> tuple[float, float, float, float]:
    """Reproject (minx, miny, maxx, maxy) from src_crs to dst_crs."""
    from pyproj import Transformer

    tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny = tf.transform(bounds[0], bounds[1])
    maxx, maxy = tf.transform(bounds[2], bounds[3])
    return (minx, miny, maxx, maxy)


def _utm_crs_for_point(lon: float, lat: float):
    """Return the UTM CRS (pyproj CRS) for a given WGS84 lon/lat."""
    from pyproj import CRS

    zone = int((lon + 180) / 6) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def generate(
    ref_path: str = typer.Option(
        ...,
        help=(
            "Path to the city's reference label file (.tif or .gpkg). "
            "Used to derive spatial extent, CRS, and default resolution. "
            "The osm_labels/ output folder is created next to this file."
        ),
    ),
    date: str | None = typer.Option(
        None,
        help="ISO 8601 date for OSM historical snapshot, e.g. '2017-01-01'. "
             "Omit for current OSM data.",
    ),
    classes: str = typer.Option(
        ",".join(str(c) for c in _BURN_ORDER),
        help=(
            "Comma-separated LCZ class integers to fetch. "
            f"Available: {sorted(LCZ_OSM_RULES)}. "
            "Default: all implemented classes."
        ),
    ),
    waterway_buffer_m: float = typer.Option(
        30.0,
        help="Buffer radius in metres applied to linear waterway features (class 17).",
    ),
    res: float | None = typer.Option(
        None,
        help=(
            "Pixel resolution in metres. "
            "Defaults to the resolution of the reference .tif. "
            "Required when ref-path is a vector file."
        ),
    ),
    output_path: str | None = typer.Option(
        None,
        help=(
            "Output GeoTIFF path. "
            "Defaults to <ref_path.parent>/osm_labels/osm_lcz_<date>.tif "
            "(or osm_lcz_current.tif if no date is given)."
        ),
    ),
    label_col: str = typer.Option(
        "lcz_class",
        help="Column / band label name for the integer LCZ class values.",
    ),
) -> None:
    """Fetch OSM features and burn them into a LCZ raster label file.

    Output is written to <ref_path.parent>/osm_labels/ by default.  The raster
    is aligned to the spatial extent and CRS of the reference file.  Classes
    are burned in priority order: water (17) wins over all others on overlap.
    """
    import copy

    # ── Parse classes ─────────────────────────────────────────────────────────
    lcz_classes = [int(c.strip()) for c in classes.split(",") if c.strip()]
    missing = [c for c in lcz_classes if c not in LCZ_OSM_RULES]
    if missing:
        raise typer.BadParameter(
            f"LCZ classes not in rules: {missing}. "
            f"Available: {sorted(LCZ_OSM_RULES)}"
        )

    # ── Apply waterway buffer override ────────────────────────────────────────
    # Clone rules so we can override the waterway buffer without mutating global
    rules = copy.deepcopy(LCZ_OSM_RULES)
    if 17 in rules:
        updated = []
        for tags, buf in rules[17]:
            if "waterway" in tags:
                buf = waterway_buffer_m
            updated.append((tags, buf))
        rules[17] = updated

    # ── Reference file ────────────────────────────────────────────────────────
    ref_p = Path(ref_path)
    if not ref_p.exists():
        raise typer.BadParameter(f"ref-path does not exist: {ref_p}")

    bounds_crs, crs, ref_res = _read_ref_info(ref_p)
    logger.info(f"Reference: {ref_p.name}  CRS: {crs}  bounds: {bounds_crs}")

    # ── BBox in WGS84 for OSM fetch ───────────────────────────────────────────
    bbox_wgs84 = _bounds_to_wgs84(bounds_crs, crs)

    # ── Always work in a projected (metre) CRS ────────────────────────────────
    if crs.is_geographic:
        lon_c = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
        lat_c = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
        working_crs = _utm_crs_for_point(lon_c, lat_c)
        logger.info(
            f"Ref CRS is geographic ({crs}) — reprojecting to {working_crs} "
            f"for metre-accurate buffering and rasterization"
        )
        snap_bounds = _reproject_bounds(bounds_crs, crs, working_crs)
        if res is None:
            res = 100.0
            logger.info("No --res given; defaulting to 100 m")
    else:
        working_crs = crs
        snap_bounds = bounds_crs
        if res is None:
            if ref_res is None:
                raise typer.BadParameter(
                    "--res is required when ref-path is a vector file (no native resolution)."
                )
            res = ref_res

    logger.info(f"Output resolution: {res} m  |  working CRS: {working_crs}")

    # ── Output path ───────────────────────────────────────────────────────────
    if output_path is None:
        date_tag = date.replace("-", "") if date else "current"
        out_p = ref_p.parent / "osm_labels" / f"osm_lcz_{date_tag}.tif"
    else:
        out_p = Path(output_path)

    logger.info(
        f"Fetching OSM features for bbox (WGS84): "
        f"W={bbox_wgs84[0]:.4f} S={bbox_wgs84[1]:.4f} "
        f"E={bbox_wgs84[2]:.4f} N={bbox_wgs84[3]:.4f}"
    )
    logger.info(
        "Classes requested: "
        + ", ".join(f"{c} ({_lcz_name(c)})" for c in lcz_classes)
    )

    # ── Fetch ─────────────────────────────────────────────────────────────────
    features_gdf = fetch_all_lcz_features(
        bbox_wgs84=bbox_wgs84,
        lcz_classes=lcz_classes,
        target_crs=working_crs,
        date=date,
        label_col=label_col,
        rules=rules,
    )

    if features_gdf.empty:
        logger.warning("No OSM features found for any requested class — output not written.")
        raise typer.Exit(0)

    logger.info(
        f"Total features: {len(features_gdf)}  "
        f"classes present: {sorted(features_gdf[label_col].unique().tolist())}"
    )

    # ── Burn ──────────────────────────────────────────────────────────────────
    burn_osm_lcz_raster(
        features_gdf=features_gdf,
        label_col=label_col,
        output_path=out_p,
        res=res,
        snap_bounds=snap_bounds,
        nodata=0,
    )


if __name__ == "__main__":
    app()
