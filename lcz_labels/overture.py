"""Stage 2 — Overture Maps extraction via DuckDB on S3.

Queries the pinned Overture GeoParquet release directly from
``s3://overturemaps-us-west-2/release/{release}/...`` with DuckDB (spatial +
httpfs), pushing bbox filters down onto Overture's ``bbox`` struct columns so
only the relevant row groups are scanned. Results are cached per AOI as local
GeoParquet and reprojected to the AOI's local UTM; extraction is idempotent
(cache is keyed on the config hash).

Provenance handling (design principle 2): every Overture feature carries a
``sources`` array of ``{dataset, ...}`` structs fusing OSM + Microsoft + Google
+ Esri. We keep ``primary_source`` (first dataset) and a boolean
``is_osm_or_esri`` (any source dataset in ``trusted_source_datasets``). Building
footprints from ALL sources feed density; only OSM/Esri features are trusted for
heights and semantics downstream.

Only ``theme=buildings/type=building`` is read, so ``building_part`` (a separate
``type``) is excluded as required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import shapely
from loguru import logger

from .config import LczLabelConfig
from .grid import local_utm_crs, resolve_aoi_bbox

S3_BASE = "s3://overturemaps-us-west-2/release"


@dataclass
class OvertureExtract:
    """Raw Overture features for one AOI, reprojected to local UTM."""

    buildings: gpd.GeoDataFrame
    landcover: gpd.GeoDataFrame      # base/land + land_use + water
    infrastructure: gpd.GeoDataFrame
    utm_crs: str
    roads: gpd.GeoDataFrame | None = None       # transportation/segment subtype=road
    mn_blocks: gpd.GeoDataFrame | None = None   # optional Million Neighborhoods blocks


def connect() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with spatial + httpfs loaded for anonymous S3."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    return con


def _bbox_predicate(bbox: tuple[float, float, float, float]) -> str:
    """Intersect (not centroid) predicate on Overture's bbox struct."""
    minx, miny, maxx, maxy = bbox
    return (
        f"bbox.xmin <= {maxx} AND bbox.xmax >= {minx} "
        f"AND bbox.ymin <= {maxy} AND bbox.ymax >= {miny}"
    )


def _read_theme(
    con: duckdb.DuckDBPyConnection,
    release: str,
    theme: str,
    typ: str,
    bbox: tuple[float, float, float, float],
    select_cols: str,
) -> gpd.GeoDataFrame:
    """Run one theme/type query and return an EPSG:4326 GeoDataFrame."""
    src = f"'{S3_BASE}/{release}/theme={theme}/type={typ}/*'"
    q = f"""
        SELECT {select_cols}, ST_AsWKB(geometry) AS _wkb
        FROM read_parquet({src}, hive_partitioning=1)
        WHERE {_bbox_predicate(bbox)}
    """
    df = con.execute(q).fetch_df()
    # DuckDB returns WKB as bytearray; shapely.from_wkb needs bytes.
    geom = shapely.from_wkb([bytes(b) for b in df.pop("_wkb").values])
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    return gdf


def _trusted_in_list(config: LczLabelConfig) -> str:
    vals = ", ".join(f"'{d}'" for d in config.trusted_source_datasets)
    return f"({vals})"


def _finalise(gdf: gpd.GeoDataFrame, utm_crs: str) -> gpd.GeoDataFrame:
    """Reproject to local UTM and repair invalid geometries.

    Overture occasionally carries geometries with non-finite coordinates; these
    crash GEOS (make_valid / overlay: "orientationIndex encountered NaN/Inf"), so
    they are dropped up front (detected via non-finite bounds).
    """
    if gdf.empty:
        return gdf.to_crs(utm_crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf = gdf.to_crs(utm_crs)
    # Reprojecting a wide AOI to a single UTM zone can push far-out geometries to
    # non-finite coordinates, which crash GEOS make_valid/overlay
    # ("orientationIndex encountered NaN/Inf"); drop them post-reprojection.
    finite = np.isfinite(gdf.geometry.bounds.to_numpy()).all(axis=1)
    if not finite.all():
        logger.warning(f"dropping {int((~finite).sum())} non-finite geometries after reprojection")
        gdf = gdf[finite]
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf.reset_index(drop=True)


def extract_overture(
    aoi_name: str, config: LczLabelConfig, *, force: bool = False
) -> OvertureExtract:
    """Extract + cache buildings / land-cover / infrastructure for one AOI."""
    bbox = resolve_aoi_bbox(config.aoi(aoi_name), config)
    utm = local_utm_crs(bbox, config.aoi(aoi_name).equal_area_crs)
    cache_dir = config.cache_dir / aoi_name / "overture"
    h = config.config_hash
    paths = {
        "buildings": cache_dir / f"buildings_{h}.parquet",
        "landcover": cache_dir / f"landcover_{h}.parquet",
        "infrastructure": cache_dir / f"infrastructure_{h}.parquet",
        "roads": cache_dir / f"roads_{h}.parquet",
    }
    if all(p.exists() for p in paths.values()) and not force:
        logger.info(f"[{aoi_name}] Overture cache hit ({h})")
        return OvertureExtract(
            buildings=gpd.read_parquet(paths["buildings"]),
            landcover=gpd.read_parquet(paths["landcover"]),
            infrastructure=gpd.read_parquet(paths["infrastructure"]),
            utm_crs=utm,
            roads=gpd.read_parquet(paths["roads"]),
            mn_blocks=load_million_neighborhoods(bbox, config, utm),
        )

    con = connect()
    trusted = _trusted_in_list(config)
    google = config.google_source_dataset
    logger.info(f"[{aoi_name}] querying Overture {config.overture_release} over {bbox}")

    # Buildings — footprint density (all sources) + height evidence (OSM/Esri) +
    # Google-source flag (Google Open Buildings segments informal fabric well).
    b_cols = (
        "id, height, num_floors, roof_height, subtype, class, "
        "CAST(sources[1].dataset AS VARCHAR) AS primary_source, "
        f"len(list_filter(sources, x -> x.dataset IN {trusted})) > 0 AS is_osm_or_esri, "
        f"len(list_filter(sources, x -> x.dataset = '{google}')) > 0 AS is_google"
    )
    buildings = _finalise(
        _read_theme(con, config.overture_release, "buildings", "building", bbox, b_cols),
        utm,
    )

    # Land-cover semantics from base themes.
    lc_frames = []
    lc_cols = "CAST(subtype AS VARCHAR) AS subtype, CAST(class AS VARCHAR) AS class"
    for typ in ("land", "land_use", "water"):
        g = _read_theme(con, config.overture_release, "base", typ, bbox, lc_cols)
        g["base_type"] = typ
        lc_frames.append(g)
    landcover = _finalise(
        gpd.GeoDataFrame(gpd.pd.concat(lc_frames, ignore_index=True), crs="EPSG:4326"),
        utm,
    )

    # Infrastructure: aeroway paved surfaces + heavy-industry point evidence.
    infra_cols = "CAST(subtype AS VARCHAR) AS subtype, CAST(class AS VARCHAR) AS class"
    infrastructure = _finalise(
        _read_theme(con, config.overture_release, "base", "infrastructure", bbox, infra_cols),
        utm,
    )

    # Roads: transportation/segment centerlines (road-deficit signal for LCZ 7).
    road_cols = "CAST(subtype AS VARCHAR) AS subtype, CAST(class AS VARCHAR) AS class"
    roads = _read_theme(con, config.overture_release, "transportation", "segment", bbox, road_cols)
    roads = _finalise(roads[roads["subtype"] == "road"].copy() if not roads.empty else roads, utm)
    con.close()

    cache_dir.mkdir(parents=True, exist_ok=True)
    buildings.to_parquet(paths["buildings"])
    landcover.to_parquet(paths["landcover"])
    infrastructure.to_parquet(paths["infrastructure"])
    roads.to_parquet(paths["roads"])
    logger.info(
        f"[{aoi_name}] Overture: {len(buildings)} buildings, {len(landcover)} land-cover, "
        f"{len(infrastructure)} infrastructure, {len(roads)} roads"
    )
    return OvertureExtract(buildings, landcover, infrastructure, utm, roads,
                           load_million_neighborhoods(bbox, config, utm))


def load_million_neighborhoods(
    bbox: tuple[float, float, float, float], config: LczLabelConfig, utm_crs: str
) -> gpd.GeoDataFrame | None:
    """Optional Million Neighborhoods (Mansueto) block layer for the AOI.

    Loads block polygons carrying an informality/low-access signal, clips to the
    AOI bbox and reprojects to local UTM. Returns None when the layer is not
    configured or the file is missing — the router then degrades gracefully
    (``mn_informal_frac`` -> NaN, ``mn_informal`` -> False).

    Expected schema (flexible): a polygon layer where an ``informal`` boolean can
    be derived — either a column literally named ``informal``, or a low block-level
    street-access / informality score. We look for the first of ``informal``,
    ``is_informal``, ``k_complexity`` (>=1 informal per Mansueto block-complexity),
    or ``access`` (low access = informal); adapt as needed for the actual file.
    """
    path = config.rasters.million_neighborhoods_path
    if path is None or not Path(path).exists():
        return None
    minx, miny, maxx, maxy = bbox
    gdf = gpd.read_file(path, bbox=(minx, miny, maxx, maxy))
    if gdf.empty:
        return None
    cols = {c.lower(): c for c in gdf.columns}
    if "informal" in cols:
        informal = gdf[cols["informal"]].astype(bool)
    elif "is_informal" in cols:
        informal = gdf[cols["is_informal"]].astype(bool)
    elif "k_complexity" in cols:
        informal = gdf[cols["k_complexity"]].astype(float) >= 1
    elif "access" in cols:
        informal = gdf[cols["access"]].astype(float) <= gdf[cols["access"]].astype(float).median()
    else:
        logger.warning(f"Million Neighborhoods {path}: no recognised informality column — ignoring")
        return None
    gdf = gdf.assign(informal=informal.to_numpy())
    gdf = gdf.to_crs(utm_crs)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
