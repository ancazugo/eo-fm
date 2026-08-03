"""Stage 4a — block delineation from the barrier network.

The unit of classification is the urban block: a polygon delimited by the
street network, rail, water and large land-cover boundaries. Blocks come from
``momepy.enclosures`` over the barrier network; where roads are unmapped the
enclosures degenerate into mega-blocks, which fall back to an aligned 320 m
tiling (``block_kind="grid_fallback"``) or, when available and enabled, Million
Neighborhoods block polygons (``block_kind="mn"``). A high grid-fallback share
is a data-quality signal, not an error — it is logged per AOI.

Every block gets a stable ``block_id`` (hash of AOI + geometry WKB) plus a
dense ``block_idx`` (uint32, 1..N in sorted-``block_id`` order) that is what
the Stage 8 ``block_id_{aoi}.tif`` raster stores — a raw hash does not fit in
uint32; the parquet is the idx <-> id lookup.

Adjacency (shared-edge rook neighbours + shared edge length) feeds Stage 5c
zone formation and the Part 2 B3 GNN. ``strict=True`` is required: the default
vertex-sharing heuristic misses edges between polygons with mismatched vertex
densities, which is exactly the grid-fallback-cell vs enclosure boundary case.
"""

from __future__ import annotations

import hashlib

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import shapely
from libpysal import graph
from loguru import logger
from shapely.geometry import box

from .config import LczLabelConfig
from .grid import local_utm_crs, resolve_aoi_bbox
from .overture import OvertureExtract


def block_id_hash(aoi_name: str, geom) -> str:
    """Stable 16-hex block id from the AOI name + geometry WKB."""
    return hashlib.sha256(aoi_name.encode() + shapely.to_wkb(geom)).hexdigest()[:16]


def _class_isin(gdf: gpd.GeoDataFrame, col: str, values: set[str]) -> np.ndarray:
    if gdf.empty or col not in gdf.columns:
        return np.zeros(len(gdf), dtype=bool)
    return gdf[col].astype("string").isin(values).fillna(False).to_numpy()


def _as_barrier_lines(geoms: np.ndarray, simplify_m: float, crs) -> gpd.GeoDataFrame:
    """Normalise a mix of lines/polygon-boundaries into an exploded line layer."""
    if len(geoms) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    gs = gpd.GeoSeries(geoms, crs=crs)
    poly = gs.geom_type.isin(("Polygon", "MultiPolygon"))
    if poly.any():
        gs = pd.concat([gs[~poly], gs[poly].boundary])
    gs = gs.simplify(simplify_m)
    gs = gs.explode(ignore_index=True)
    gs = gs[~gs.is_empty & gs.notna()]
    gs = gs[gs.geom_type.isin(("LineString", "LinearRing"))]
    return gpd.GeoDataFrame(geometry=gs.reset_index(drop=True), crs=crs)


def assemble_barriers(
    extract: OvertureExtract, config: LczLabelConfig
) -> tuple[gpd.GeoDataFrame, list[gpd.GeoDataFrame]]:
    """(primary, additional) barrier line layers in local UTM.

    Primary = motorized road + surface-rail centerlines (they define the
    enclosures). Additional = water boundaries/waterways and boundaries of
    large land-cover polygons (forest/farmland); they only subdivide.
    """
    bp = config.blocks
    utm = extract.utm_crs

    primary_parts: list[np.ndarray] = []
    if extract.roads is not None and not extract.roads.empty:
        m = _class_isin(extract.roads, "class", set(config.road_motorized_classes))
        primary_parts.append(extract.roads.geometry.values[m])
    if extract.rail is not None and not extract.rail.empty:
        m = _class_isin(extract.rail, "class", set(bp.rail_barrier_classes))
        primary_parts.append(extract.rail.geometry.values[m])
    primary = _as_barrier_lines(
        np.concatenate(primary_parts) if primary_parts else np.array([]),
        bp.barrier_simplify_m, utm,
    )

    additional: list[gpd.GeoDataFrame] = []
    lc = extract.landcover
    if lc is not None and not lc.empty:
        base_type = lc.get("base_type", pd.Series([""] * len(lc))).astype("string")
        is_water = (base_type == "water").fillna(False).to_numpy()
        wgeom = lc.geometry.values[is_water]
        if len(wgeom):
            is_poly = shapely.get_type_id(wgeom) >= 3  # (Multi)Polygon
            big = shapely.area(wgeom) >= bp.barrier_water_min_m2
            water = np.concatenate([wgeom[is_poly & big], wgeom[~is_poly]])
            wl = _as_barrier_lines(water, bp.barrier_simplify_m, utm)
            if not wl.empty:
                additional.append(wl)
        lcm = _class_isin(lc, "class", set(bp.barrier_landcover_classes)) & ~is_water
        lgeom = lc.geometry.values[lcm]
        if len(lgeom):
            lgeom = lgeom[shapely.area(lgeom) >= bp.barrier_landcover_min_m2]
            ll = _as_barrier_lines(lgeom, bp.barrier_simplify_m, utm)
            if not ll.empty:
                additional.append(ll)
    return primary, additional


def _flag_small_or_thin(geoms: np.ndarray, min_area: float, corridor_width: float) -> np.ndarray:
    """Blocks below the viable-block bar: tiny area OR no interior at width.

    A polygon that vanishes under a ``-corridor_width/2`` buffer is a road
    corridor (median, traffic island, interchange pocket) regardless of its
    area — a 20 m x 1 km median is 20 000 m² but still has no interior at the
    10 m raster + 1 px erosion scale.
    """
    small = shapely.area(geoms) < min_area
    thin = shapely.is_empty(shapely.buffer(geoms, -corridor_width / 2.0))
    return small | thin


def _merge_small_blocks(
    blocks: gpd.GeoDataFrame, min_area: float, corridor_width: float, drop_area: float
) -> gpd.GeoDataFrame:
    """Merge small/thin blocks into the neighbour sharing the longest edge.

    Iterates a few passes (a median may only touch other medians until those
    merge); whatever remains flagged and unmergeable afterwards is dropped
    (logged; by construction it is junk surrounded by junk, ~0.1% of area).
    ``block_kind`` (when present) is inherited from the absorbing neighbour.
    ``drop_area`` is the hard sliver floor used only for logging granularity.
    """
    for _ in range(3):
        geoms_all = blocks.geometry.values
        flag = _flag_small_or_thin(geoms_all, min_area, corridor_width)
        if not flag.any():
            return blocks.reset_index(drop=True)
        if flag.all():
            logger.warning(
                f"all {len(blocks)} blocks below the viable-block bar — keeping as-is"
            )
            return blocks.reset_index(drop=True)
        keep = blocks[~flag].reset_index(drop=True)
        geoms = keep.geometry.values.copy()
        tree = shapely.STRtree(geoms)
        unmerged_mask = np.zeros(int(flag.sum()), dtype=bool)
        n_merged = 0
        for k, g in enumerate(blocks.geometry.values[flag]):
            idx = tree.query(g, predicate="intersects")
            if len(idx):
                shared = shapely.length(
                    shapely.intersection(np.repeat(g, len(idx)), geoms[idx])
                )
                if shared.max() > 0:
                    j = idx[int(np.argmax(shared))]
                    geoms[j] = shapely.make_valid(shapely.union(geoms[j], g))
                    n_merged += 1
                    continue
            unmerged_mask[k] = True
        keep = keep.set_geometry(gpd.GeoSeries(geoms, crs=blocks.crs))
        leftovers = blocks[flag][unmerged_mask]
        blocks = (
            gpd.GeoDataFrame(
                pd.concat([keep, leftovers], ignore_index=True), crs=blocks.crs
            )
            if len(leftovers)
            else keep
        )
        if n_merged == 0:
            break
    geoms_all = blocks.geometry.values
    flag = _flag_small_or_thin(geoms_all, min_area, corridor_width)
    if flag.any():
        lost = float(shapely.area(geoms_all[flag]).sum())
        logger.debug(
            f"dropped {int(flag.sum())} unmergeable small/thin blocks "
            f"({lost / 1e4:.2f} ha; sliver floor {drop_area} m²)"
        )
        blocks = blocks[~flag]
    return blocks.reset_index(drop=True)


def _fallback_cells(geom, cell_m: float) -> np.ndarray:
    """Aligned ``cell_m`` tiling of ``geom`` (origin snapped to cell multiples)."""
    minx, miny, maxx, maxy = shapely.bounds(geom)
    x0 = np.floor(minx / cell_m) * cell_m
    y0 = np.floor(miny / cell_m) * cell_m
    xs = np.arange(x0, maxx, cell_m)
    ys = np.arange(y0, maxy, cell_m)
    cells = np.array([box(x, y, x + cell_m, y + cell_m) for y in ys for x in xs])
    pieces = shapely.intersection(cells, geom)
    out: list = []
    for p in pieces:
        if p is None or p.is_empty:
            continue
        for part in getattr(p, "geoms", [p]):
            if part.geom_type == "Polygon" and part.area > 0:
                out.append(part)
    return np.array(out)


def _split_mega_block(
    geom, extract: OvertureExtract, config: LczLabelConfig
) -> tuple[list, list[str]]:
    """Subdivide one mega-block: MN polygons where enabled/covered, grid cells else."""
    bp = config.blocks
    geoms: list = []
    kinds: list[str] = []
    residual = geom
    if bp.use_mn_blocks and extract.mn_blocks is not None and not extract.mn_blocks.empty:
        mn_geoms = extract.mn_blocks.geometry.values
        hits = shapely.STRtree(mn_geoms).query(geom, predicate="intersects")
        if len(hits):
            pieces = shapely.intersection(mn_geoms[hits], geom)
            kept = [p for p in pieces if p is not None and not p.is_empty
                    and shapely.area(p) >= bp.sliver_area_m2]
            if kept:
                geoms.extend(kept)
                kinds.extend(["mn"] * len(kept))
                residual = shapely.difference(geom, shapely.union_all(kept))
    if residual is not None and not residual.is_empty:
        cells = _fallback_cells(residual, config.patch_size_m)
        geoms.extend(cells)
        kinds.extend(["grid_fallback"] * len(cells))
    return geoms, kinds


def delineate(
    extract: OvertureExtract, limit_poly, config: LczLabelConfig, *, utm: str | None = None
) -> gpd.GeoDataFrame:
    """Core delineation: barriers -> enclosures -> sliver merge -> mega split.

    ``limit_poly`` is the AOI extent in local UTM. Returns ``block_kind`` +
    ``geometry`` (local UTM), un-identified — :func:`build_blocks` adds ids.
    """
    bp = config.blocks
    utm = utm or extract.utm_crs
    primary, additional = assemble_barriers(extract, config)
    if primary.empty:
        logger.warning("no primary barriers — whole AOI is one mega-block")
        enc = gpd.GeoDataFrame(geometry=[limit_poly], crs=utm)
    else:
        enc = momepy.enclosures(
            primary,
            limit=gpd.GeoSeries([limit_poly], crs=utm),
            additional_barriers=additional or None,
        )

    enc = enc.set_geometry(enc.geometry.make_valid())
    enc = enc.explode(ignore_index=True)
    enc = enc[enc.geom_type == "Polygon"]
    enc = enc[enc.geometry.area > 0].reset_index(drop=True)[["geometry"]]
    enc = _merge_small_blocks(enc, bp.min_block_area_m2, bp.corridor_min_width_m,
                              bp.sliver_area_m2)

    areas = enc.geometry.area.to_numpy()
    mega = areas > bp.max_block_area_km2 * 1.0e6
    geoms = list(enc.geometry.values[~mega])
    kinds = ["enclosure"] * len(geoms)
    for g in enc.geometry.values[mega]:
        sub_geoms, sub_kinds = _split_mega_block(g, extract, config)
        geoms.extend(sub_geoms)
        kinds.extend(sub_kinds)

    blocks = gpd.GeoDataFrame({"block_kind": kinds}, geometry=geoms, crs=utm)
    return _merge_small_blocks(blocks, bp.min_block_area_m2, bp.corridor_min_width_m,
                               bp.sliver_area_m2)


def build_blocks(
    aoi_name: str, extract: OvertureExtract, config: LczLabelConfig, *, force: bool = False
) -> gpd.GeoDataFrame:
    """Delineate blocks for one AOI (cached GeoParquet, local UTM).

    Returns columns ``block_id, block_idx, block_kind, area_m2, aoi, geometry``.
    """
    cache = config.cache_dir / aoi_name / f"blocks_{config.config_hash}.parquet"
    if cache.exists() and not force:
        logger.info(f"[{aoi_name}] blocks cache hit: {cache.name}")
        return gpd.read_parquet(cache)

    bbox = resolve_aoi_bbox(config.aoi(aoi_name), config)
    utm = extract.utm_crs or local_utm_crs(bbox, config.aoi(aoi_name).equal_area_crs)
    limit_poly = (
        gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(utm).iloc[0].envelope
    )
    blocks = delineate(extract, limit_poly, config, utm=utm)

    blocks["block_id"] = [block_id_hash(aoi_name, g) for g in blocks.geometry.values]
    if blocks["block_id"].duplicated().any():
        dup = blocks[blocks["block_id"].duplicated(keep=False)]
        raise ValueError(f"[{aoi_name}] duplicate block geometries -> id collision: {dup['block_id'].tolist()[:5]}")
    blocks = blocks.sort_values("block_id", ignore_index=True)
    blocks["block_idx"] = np.arange(1, len(blocks) + 1, dtype=np.uint32)
    blocks["area_m2"] = blocks.geometry.area
    blocks["aoi"] = aoi_name
    blocks = blocks[["block_id", "block_idx", "block_kind", "area_m2", "aoi", "geometry"]]

    comp = blocks.groupby("block_kind")["area_m2"].agg(["size", "sum"])
    total = float(blocks["area_m2"].sum())
    comp_str = ", ".join(
        f"{k}: {int(r['size'])} blocks ({r['sum'] / total:.1%} area)" for k, r in comp.iterrows()
    )
    logger.info(f"[{aoi_name}] {len(blocks)} blocks — {comp_str}")
    fallback_share = comp.loc["grid_fallback", "sum"] / total if "grid_fallback" in comp.index else 0.0
    if fallback_share > 0.5:
        logger.warning(
            f"[{aoi_name}] grid-fallback covers {fallback_share:.0%} of the AOI — "
            "road mapping is sparse; block geometry is degraded there (not an error)"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    blocks.to_parquet(cache)
    return blocks


def build_adjacency(
    blocks: gpd.GeoDataFrame, config: LczLabelConfig, aoi_name: str, *, force: bool = False
) -> pd.DataFrame:
    """Shared-edge (rook) block adjacency with shared edge length, cached.

    Returns columns ``block_a, block_b, shared_len_m`` with ``block_a < block_b``
    (each undirected edge once). Consumed by Stage 5c zone formation and the
    Part 2 B3 GNN (edge weights).
    """
    cache = config.cache_dir / aoi_name / f"adjacency_{config.config_hash}.parquet"
    if cache.exists() and not force:
        logger.info(f"[{aoi_name}] adjacency cache hit: {cache.name}")
        return pd.read_parquet(cache)

    # strict=True: vertex-sharing rook misses edges between polygons with
    # mismatched vertex densities (grid-fallback cells vs enclosures).
    g = graph.Graph.build_contiguity(blocks.set_index("block_id"), rook=True, strict=True)
    adj = g.adjacency.reset_index()
    adj.columns = ["block_a", "block_b", "weight"]
    adj = adj[(adj["weight"] > 0) & (adj["block_a"] < adj["block_b"])].reset_index(drop=True)

    geom_of = dict(zip(blocks["block_id"], blocks.geometry.values))
    ga = np.array([geom_of[b] for b in adj["block_a"]])
    gb = np.array([geom_of[b] for b in adj["block_b"]])
    adj["shared_len_m"] = shapely.length(shapely.intersection(ga, gb)) if len(adj) else []
    out = adj[["block_a", "block_b", "shared_len_m"]]

    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    logger.info(f"[{aoi_name}] adjacency: {len(out)} edges over {len(blocks)} blocks")
    return out
