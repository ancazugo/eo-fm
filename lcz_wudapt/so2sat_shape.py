"""Reshape QC-passing WUDAPT polygons into So2Sat-shaped city directories.

The load-bearing idea: ``src/create_city_grids.py`` needs exactly two files per
city directory -- ``patches_reference_{city}.gpkg`` and
``patches_reference_{city}.tif``. If this module writes that pair for each WUDAPT
AOI, then **the entire downstream stack runs unchanged**: ``create_city_grids``,
``extract_grid_embeddings``, ``extract_so2sat_embeddings --patches-file``, both
trainers' per-city mode, and ``--global-split`` / ``--split-col``. No dataset
layer, no loss, and no split code has to learn that WUDAPT exists.

Two artefacts, deliberately at different granularities:

``patches_reference_{aoi}.gpkg``
    320 m squares of one pure class, EPSG:4326, columns
    ``patch_id, dataset, LCZ_class`` -- byte-for-byte the So2Sat schema, plus
    provenance columns the So2Sat loader ignores. Drives patch classification.

``patches_reference_{aoi}.tif``
    A 10 m label raster burned from the full QC-passing polygons, EPSG:4326,
    uint8 1-17 with 0 = nodata, matching So2Sat's convention (whose own tif is
    coarse, ~320 m, because So2Sat has nothing finer to offer). Drives
    segmentation, where sparse labels with ``ignore_index=-1`` are the norm and
    the polygon interiors are worth far more than the patches carved from them.

That split matters because the two pathways have opposite constraints. Only
**39.8%** of QC-passing polygons can contain a 320 m square at all, and just
**3.1%** could fill a 1280 m segmentation tile -- but a tile does not need
filling. Forcing segmentation through patch containment would throw away most of
the hand-drawn label area for nothing.

Geometry note, easy to get wrong: ``buffer(-s/2)`` non-empty means the polygon
contains a *disc* of diameter ``s``, not a *square* of side ``s`` (which needs
radius ``s/sqrt(2)``). Since a square of side ``s`` does contain that disc, the
erosion is a valid **necessary** prefilter, and every candidate is then checked
exactly with ``square.within(polygon)``. The 39.8% figure above is therefore an
upper bound on patch yield.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from loguru import logger
from rasterio import features
from rasterio.transform import from_origin

from .config import WudaptConfig
from .ingest import N_LCZ
from .qc import project_valid, reduce_oversize, utm_crs

__all__ = [
    "build_label_raster",
    "combine_patches",
    "build_patches",
    "build_relaxed_patches",
    "place_patches",
    "run_relaxed",
    "run_shape",
    "write_city_dir",
]

# So2Sat's own per-city gpkg carries exactly these three columns. Anything extra
# we emit is provenance that `build_pseudo_items` / `build_city_items` ignore.
_SO2SAT_COLS = ("patch_id", "dataset", "LCZ_class")

# Metres per degree. Latitude is near-constant (WGS84 mean); longitude is scaled
# by cos(lat) at the AOI centre. Exact enough for choosing a raster resolution --
# the raster is a label mask, not a measurement.
_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON = 111_320.0


def place_patches(
    polygon,
    side_m: float,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    max_patches: int | None = None,
) -> list:
    """Lay So2Sat-shaped squares inside one polygon, in a metric CRS.

    Centres sit on a lattice of pitch ``side_m`` anchored at ``origin``, so
    patches from neighbouring polygons tile consistently instead of overlapping
    at arbitrary offsets -- the same property So2Sat's grid has, and what makes
    ``dominant_frac`` meaningful when two patches abut.

    The erosion is only a prefilter (see the module docstring); containment is
    then verified exactly. When more candidates survive than ``max_patches``, the
    ones nearest the polygon's representative point are kept: they are the most
    interior, hence the least likely to straddle an unlabelled edge.
    """
    core = polygon.buffer(-side_m / 2.0)
    if core.is_empty:
        return []

    minx, miny, maxx, maxy = core.bounds
    ox, oy = origin
    # Lattice centres are at origin + (k + 0.5) * side, so squares are
    # origin-aligned and never half-overlap a neighbour's square.
    kx0 = int(np.floor((minx - ox) / side_m))
    kx1 = int(np.ceil((maxx - ox) / side_m))
    ky0 = int(np.floor((miny - oy) / side_m))
    ky1 = int(np.ceil((maxy - oy) / side_m))
    if (kx1 - kx0) * (ky1 - ky0) > 100_000:  # pathological bbox, e.g. a ring
        return []

    xs = ox + (np.arange(kx0, kx1 + 1) + 0.5) * side_m
    ys = oy + (np.arange(ky0, ky1 + 1) + 0.5) * side_m
    gx, gy = np.meshgrid(xs, ys)
    cx, cy = gx.ravel(), gy.ravel()
    if cx.size == 0:
        return []

    # Necessary condition first (cheap), exact containment second.
    inside_core = shapely.contains_xy(core, cx, cy)
    cx, cy = cx[inside_core], cy[inside_core]
    if cx.size == 0:
        # A thin core can miss every lattice point while still admitting a
        # square; fall back to its representative point rather than losing the
        # polygon to lattice phase.
        rp = core.representative_point()
        cx, cy = np.array([rp.x]), np.array([rp.y])

    h = side_m / 2.0
    squares = shapely.box(cx - h, cy - h, cx + h, cy + h)
    keep = shapely.within(squares, polygon)
    squares, cx, cy = squares[keep], cx[keep], cy[keep]
    if squares.size == 0:
        return []

    if max_patches is not None and squares.size > max_patches:
        rp = polygon.representative_point()
        d = (cx - rp.x) ** 2 + (cy - rp.y) ** 2
        squares = squares[np.argsort(d)[:max_patches]]
    return list(squares)


def build_patches(gdf: gpd.GeoDataFrame, config: WudaptConfig) -> gpd.GeoDataFrame:
    """So2Sat-shaped patches for one AOI, from its QC'd polygons.

    Expects the output of :func:`lcz_wudapt.qc.apply_qc`. Only ``qc_pass``
    polygons contribute. The Generator's oversize reduction (>1.5 km2 -> ~350 m
    core) is applied **here and not to the raster**: it exists to stop a few huge
    natural polygons dominating a classification training set, which is a patch
    problem, not a dense-supervision one.
    """
    rules = config.qc
    src = gdf[gdf["qc_pass"]].copy()
    if src.empty:
        return gpd.GeoDataFrame(
            columns=[*_SO2SAT_COLS, "geometry"], geometry="geometry", crs="EPSG:4326"
        )

    proj = project_valid(src)
    crs = proj.crs
    proj["geometry"] = reduce_oversize(proj.geometry, rules)
    proj = proj[~proj.geometry.is_empty]

    rows = []
    side = float(rules.patch_size_m)
    for i, rec in enumerate(proj.itertuples(index=False)):
        for sq in place_patches(rec.geometry, side, max_patches=rules.max_patches_per_polygon):
            rows.append((i, sq))
    if not rows:
        return gpd.GeoDataFrame(
            columns=[*_SO2SAT_COLS, "geometry"], geometry="geometry", crs="EPSG:4326"
        )

    src_idx = np.array([r[0] for r in rows])
    geoms = [r[1] for r in rows]
    meta = proj.iloc[src_idx].reset_index(drop=True)

    out = gpd.GeoDataFrame(
        {
            "LCZ_class": meta["class"].astype("int16").to_numpy(),
            "weight": meta["weight"].astype("float32").to_numpy(),
            "aoi": meta["aoi"].to_numpy(),
            "src_area_km2": meta["area_km2_utm"].astype("float32").to_numpy(),
            "src_shape": meta["shape_utm"].astype("float32").to_numpy(),
            "nbr_dist_m": meta["nbr_dist_m"].astype("float32").to_numpy(),
            "nbr_conflict": meta["nbr_conflict"].to_numpy(),
            "overlap_frac_diff_class": meta["overlap_frac_diff_class"].astype("float32").to_numpy(),
            "oa": pd.to_numeric(meta["acc"], errors="coerce").astype("float32").to_numpy(),
            "label_year": meta["label_year"].to_numpy(),
            "embedding_year": meta["embedding_year"].to_numpy(),
            "w_time": meta["w_time"].astype("float32").to_numpy(),
            "submission_id": meta["submission_id"].astype(str).to_numpy(),
        },
        geometry=geoms,
        crs=crs,
    )

    if rules.max_patches_per_class is not None:
        out = (
            out.sort_values("weight", ascending=False)
            .groupby("LCZ_class", group_keys=False)
            .head(rules.max_patches_per_class)
        )

    out = out.to_crs("EPSG:4326").reset_index(drop=True)
    # `dataset` selects the embedding sub-directory in build_patch_index, so it
    # must be a real on-disk split name; "unlabeled" is the one So2Sat reserves
    # for material that carries no official split.
    out.insert(0, "dataset", "unlabeled")
    # patch_id must be globally unique, NOT per-AOI. extract_so2sat_embeddings
    # writes every patch to {root}/{dataset}/{output_name}/{year}/patch_{id}.npy,
    # a directory shared by all cities, so a per-AOI counter starting at 0 would
    # have Berlin's patch 0 silently overwrite Beijing's. That is precisely the
    # global-split patch_id collision fixed in this repo in 2026-06; do not
    # reintroduce it. The AOI key is already slug'd and filesystem-safe.
    aoi_key = str(meta["aoi"].iloc[0])
    out.insert(0, "patch_id", [f"{aoi_key}_{i:06d}" for i in range(len(out))])
    return out


def build_relaxed_patches(
    gdf: gpd.GeoDataFrame,
    config: WudaptConfig,
    *,
    min_frac: float = 0.6,
    max_other_frac: float = 0.1,
) -> gpd.GeoDataFrame:
    """Lattice squares that irregular polygons cover mostly, not entirely.

    Strict placement (:func:`build_patches`) needs a whole 320 m square inside
    one polygon, which only 39.8% of QC-passing polygons admit -- and it is
    class-biased against exactly the compact built classes (LCZ 1 22.4%, water
    55.6%). This is the relaxed arm, reported separately rather than mixed in.

    Candidates come only from QC-passing polygons that could NOT hold a strict
    square (``fits_patch`` False), so the arm adds new ground instead of
    re-cutting what strict placement already used. A candidate is a square of
    the same origin-aligned lattice :func:`place_patches` uses, so a relaxed
    square is either identical to a strict one or disjoint from it. It is kept
    when

    * at least ``min_frac`` of it lies in QC-passing polygons of its class
      (``dominant_frac`` -- the union, so adjacent same-class polygons count
      together), and
    * at most ``max_other_frac`` lies in polygons of any other class
      (``other_frac``; overlapping class unions are summed, which errs strict).
      "Other" counts EVERY polygon in the AOI, not only QC-passing ones: a
      polygon that lost the duplicate-priority rule or failed a gate is still
      an annotator saying this ground is something else.

    The label is therefore a majority, not a certainty: the remaining
    ``1 - dominant_frac - other_frac`` of the square is unlabelled ground. That
    is why these patches are meant to pass a teacher-agreement gate before
    training, and why ``weight`` is the source polygon's weight scaled by
    ``dominant_frac``. At most ``max_patches_per_polygon`` per source polygon,
    as in strict placement.
    """
    rules = config.qc
    empty = gpd.GeoDataFrame(columns=[*_SO2SAT_COLS, "geometry"],
                             geometry="geometry", crs="EPSG:4326")
    src = gdf[gdf["qc_pass"]].copy()
    if src.empty or "fits_patch" not in src or bool(src["fits_patch"].all()):
        return empty

    proj = project_valid(src)
    proj["geometry"] = reduce_oversize(proj.geometry, rules)
    proj = proj[~proj.geometry.is_empty].reset_index(drop=True)
    side = float(rules.patch_size_m)
    h = side / 2.0
    area = side * side

    unions = {int(c): shapely.union_all(g.geometry.values)
              for c, g in proj.groupby("class")}
    everyone = project_valid(gdf, crs=proj.crs)
    claims = {int(c): shapely.union_all(g.geometry.values)
              for c, g in everyone.groupby("class")}

    rows = []
    for i, rec in enumerate(proj.itertuples(index=False)):
        if bool(rec.fits_patch):
            continue
        minx, miny, maxx, maxy = rec.geometry.bounds
        kx = np.arange(int(np.floor(minx / side)), int(np.ceil(maxx / side)) + 1)
        ky = np.arange(int(np.floor(miny / side)), int(np.ceil(maxy / side)) + 1)
        gx, gy = np.meshgrid((kx + 0.5) * side, (ky + 0.5) * side)
        cx, cy = gx.ravel(), gy.ravel()
        squares = shapely.box(cx - h, cy - h, cx + h, cy + h)
        touches = shapely.intersects(squares, rec.geometry)
        squares, cx, cy = squares[touches], cx[touches], cy[touches]
        if squares.size == 0:
            continue
        cls = int(rec[proj.columns.get_loc("class")])
        own = shapely.area(shapely.intersection(squares, unions[cls])) / area
        other = np.zeros(squares.size)
        for c, u in claims.items():
            if c != cls:
                other += shapely.area(shapely.intersection(squares, u)) / area
        ok = (own >= min_frac) & (other <= max_other_frac)
        if not ok.any():
            continue
        # Most of THIS polygon first, so the cap keeps the squares it anchors.
        mine = shapely.area(shapely.intersection(squares[ok], rec.geometry)) / area
        order = np.argsort(-mine)[: rules.max_patches_per_polygon]
        for j in order:
            k = np.flatnonzero(ok)[j]
            rows.append((i, cx[k], cy[k], squares[k], own[k], other[k]))
    if not rows:
        return empty

    # One square can qualify through several same-class polygons; keep it once,
    # attributed to the first polygon that claimed it.
    seen, uniq = set(), []
    for r in rows:
        key = (round(r[1], 3), round(r[2], 3))
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    meta = proj.iloc[[r[0] for r in uniq]].reset_index(drop=True)
    dom = np.array([r[4] for r in uniq], dtype="float32")
    out = gpd.GeoDataFrame(
        {
            "LCZ_class": meta["class"].astype("int16").to_numpy(),
            "weight": (meta["weight"].astype("float32").to_numpy() * dom),
            "aoi": meta["aoi"].to_numpy(),
            "dominant_frac": dom,
            "other_frac": np.array([r[5] for r in uniq], dtype="float32"),
            "src_area_km2": meta["area_km2_utm"].astype("float32").to_numpy(),
            "src_shape": meta["shape_utm"].astype("float32").to_numpy(),
            "nbr_dist_m": meta["nbr_dist_m"].astype("float32").to_numpy(),
            "nbr_conflict": meta["nbr_conflict"].to_numpy(),
            "oa": pd.to_numeric(meta["acc"], errors="coerce").astype("float32").to_numpy(),
            "label_year": meta["label_year"].to_numpy(),
            "w_time": meta["w_time"].astype("float32").to_numpy(),
            "submission_id": meta["submission_id"].astype(str).to_numpy(),
        },
        geometry=[r[3] for r in uniq],
        crs=proj.crs,
    ).to_crs("EPSG:4326")
    out.insert(0, "dataset", "unlabeled")
    # "_r" keeps these ids disjoint from strict ones ({aoi}_{n:06d}): both are
    # extracted into the same unlabeled/ directory.
    out.insert(0, "patch_id", [f"{meta['aoi'].iloc[0]}_r{i:06d}" for i in range(len(out))])
    return out


def run_relaxed(
    config: WudaptConfig,
    out_path: Path | None = None,
    *,
    strict_gpkg: Path | None = None,
    min_frac: float = 0.6,
    max_other_frac: float = 0.1,
) -> tuple[Path, pd.DataFrame]:
    """Relaxed-containment patches for every AOI, as one gpkg beside the strict one.

    Carries the same ``wudapt_split``/``region``/``forced_test`` join as
    :func:`combine_patches`, and drops any square that shares ground with a
    strict patch, so the two pools can be combined without double counting.
    """
    from shapely.strtree import STRtree

    from .qc import run_qc
    from .splits import assert_split_integrity

    root = Path(config.gpkg_path).parent
    out_path = Path(out_path) if out_path else root / "patches_wudapt_relaxed.gpkg"
    strict_gpkg = Path(strict_gpkg) if strict_gpkg else root / "patches_wudapt_rxr.gpkg"

    split_path = Path(config.cache_dir) / f"wudapt_splits_{config.config_hash}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"run `lcz_wudapt splits` first ({split_path} missing)")
    splits = pd.read_parquet(split_path)
    assert_split_integrity(splits)

    gdf = gpd.read_parquet(run_qc(config))
    frames = []
    groups = list(gdf.groupby("aoi", sort=False))
    for n, (aoi, sub) in enumerate(groups, 1):
        try:
            p = build_relaxed_patches(sub, config, min_frac=min_frac,
                                      max_other_frac=max_other_frac)
        except Exception as exc:
            logger.warning(f"relaxed failed for {aoi}: {exc}")
            continue
        if len(p):
            frames.append(p)
        if n % 100 == 0:
            logger.info(f"  relaxed {n:,}/{len(groups):,} AOIs, "
                        f"{sum(len(f) for f in frames):,} patches so far")
    if not frames:
        raise ValueError("no relaxed patches produced")
    allp = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    if not allp["patch_id"].is_unique:
        raise ValueError("relaxed patch_id is not unique across AOIs")

    strict = gpd.read_file(strict_gpkg, columns=["patch_id"])
    tree = STRtree(strict.geometry.values)
    ia, ib = tree.query(allp.geometry.values, predicate="intersects")
    inter = shapely.area(shapely.intersection(allp.geometry.values[ia], strict.geometry.values[ib]))
    clash = np.zeros(len(allp), dtype=bool)
    clash[np.unique(ia[inter > 0])] = True
    if clash.any():
        logger.info(f"dropping {int(clash.sum()):,} relaxed squares that share ground with strict ones")
    allp = allp[~clash]

    allp = allp.merge(splits[["aoi", "wudapt_split", "region", "forced_test"]],
                      on="aoi", how="left", validate="many_to_one")
    allp = allp[allp["wudapt_split"].notna()]
    allp.to_file(out_path, driver="GPKG")
    review = (allp.groupby(["wudapt_split", "LCZ_class"]).size()
              .unstack(fill_value=0))
    logger.info(f"wrote {out_path} -- {len(allp):,} relaxed patches, "
                f"{allp.aoi.nunique():,} AOIs, splits {allp.wudapt_split.value_counts().to_dict()}")
    return out_path, review


def build_label_raster(
    gdf: gpd.GeoDataFrame,
    config: WudaptConfig,
    path: Path,
) -> Path | None:
    """Burn the QC-passing polygons to a 10 m label raster for segmentation.

    Convention matches So2Sat exactly: EPSG:4326, uint8, classes 1-17, **0 =
    nodata**, so ``grid_tiles.py``'s ``raw - 1 -> ignore_index=-1`` mapping is
    unchanged.

    Two rules from the sources are enforced here rather than at patch level:

    * **Contested pixels become nodata.** Where two different classes claim the
      same pixel the annotators disagree, and the >100 m inter-LCZ buffer rule
      has plainly been violated. Writing nodata says "we do not know" instead of
      picking a winner -- which is what keeps Tehran (96% of polygons within
      100 m of another class) usable rather than excluded outright.
    * **A small erosion**, ``raster_erode_m`` = 20 m, to keep boundary mixing out
      of the interior. Note this is *not* the 100 m buffer distance: the median
      polygon is only ~219 m across, so a 100 m erosion would erase most of the
      dataset.
    """
    rules = config.qc
    src = gdf[gdf["qc_pass"]]
    if src.empty:
        return None

    lat = float(src.geometry.representative_point().y.mean())
    dy = rules.raster_res_m / _M_PER_DEG_LAT
    dx = rules.raster_res_m / (_M_PER_DEG_LON * max(np.cos(np.radians(lat)), 0.05))

    minx, miny, maxx, maxy = src.total_bounds
    minx, miny = minx - dx, miny - dy
    maxx, maxy = maxx + dx, maxy + dy
    width = int(np.ceil((maxx - minx) / dx))
    height = int(np.ceil((maxy - miny) / dy))
    if width <= 0 or height <= 0:
        return None

    # Degrade resolution rather than fail on a continent-sized AOI bbox.
    if width * height > rules.raster_max_pixels:
        scale = np.sqrt(width * height / rules.raster_max_pixels)
        dx, dy = dx * scale, dy * scale
        width = int(np.ceil((maxx - minx) / dx))
        height = int(np.ceil((maxy - miny) / dy))
        logger.warning(
            f"{path.name}: raster capped at {rules.raster_max_pixels:,} px, "
            f"resolution coarsened {scale:.1f}x"
        )

    transform = from_origin(minx, maxy, dx, dy)
    shape = (height, width)

    # Erode in metres, which means projecting: a degree-space buffer is
    # anisotropic and would erode ~1.6x more in x than y at Berlin's latitude.
    proj = project_valid(src)
    eroded = proj.geometry.buffer(-rules.raster_erode_m)
    keep = ~eroded.is_empty
    if not bool(keep.any()):
        return None
    burn = gpd.GeoSeries(eroded[keep], crs=proj.crs).to_crs("EPSG:4326")
    classes = src["class"].to_numpy()[keep.to_numpy()]
    order = np.argsort(pd.to_numeric(src["acc"], errors="coerce").fillna(0.0).to_numpy()[keep.to_numpy()])

    # Highest-accuracy submission burns last, so it wins same-class overlaps.
    label = features.rasterize(
        ((g, int(c)) for g, c in zip(burn.to_numpy()[order], classes[order])),
        out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=False,
    )

    # Contested pixels -> nodata. Detected with one cheap sjoin rather than 17
    # rasterize passes, because the answer is usually "none": with
    # `use_conflict_priority` on, the ESSD rule imposes a TOTAL rank order, so
    # exactly one member of every different-class overlapping pair has already
    # been removed and no contested pixel can survive. Measured on Berlin,
    # Nairobi and Tehran: 0 contested pixels in all three. The loop below is the
    # safety net for when that rule is switched off.
    n_contested = 0
    kept = gpd.GeoDataFrame({"class": classes}, geometry=burn.to_numpy(), crs="EPSG:4326")
    pairs = gpd.sjoin(kept, kept.rename(columns={"class": "class_r"}), predicate="intersects")
    if bool((pairs["class"] != pairs["class_r"]).any()):
        claims = np.zeros(shape, dtype="uint8")
        for c in np.unique(classes):
            m = classes == c
            claims += features.rasterize(
                ((g, 1) for g in burn.to_numpy()[m]),
                out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=False,
            )
        contested = claims > 1
        n_contested = int(contested.sum())
        label[contested] = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=0,
        compress="LZW", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(label, 1)

    frac = float((label > 0).mean())
    logger.debug(
        f"{path.name}: {width}x{height} px, {frac:.3%} labelled, "
        f"{n_contested:,} contested px zeroed"
    )
    return path


def write_city_dir(
    aoi: str,
    gdf: gpd.GeoDataFrame,
    config: WudaptConfig,
    out_root: Path,
) -> dict[str, Path]:
    """Write one So2Sat-shaped city directory. Returns the paths written."""
    city_dir = Path(out_root) / aoi
    city_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    patches = build_patches(gdf, config)
    if len(patches):
        gpkg = city_dir / f"patches_reference_{aoi}.gpkg"
        patches.to_file(gpkg, driver="GPKG")
        written["gpkg"] = gpkg

    tif = build_label_raster(gdf, config, city_dir / f"patches_reference_{aoi}.tif")
    if tif is not None:
        written["tif"] = tif
    return written


def run_shape(
    config: WudaptConfig,
    *,
    aois: list[str] | None = None,
    out_root: Path | None = None,
    min_patches: int = 1,
    force: bool = False,
) -> pd.DataFrame:
    """Build So2Sat-shaped city directories for every AOI. Returns a review table.

    ``out_root`` defaults to ``$DATA_DIR/input/WUDAPT/cities``, mirroring
    ``$DATA_DIR/input/So2Sat-LCZ42/v4/cities`` so ``--cities-dir`` points at it
    directly.
    """
    from .qc import run_qc

    root = Path(out_root) if out_root else Path(config.gpkg_path).parent / "cities"
    root.mkdir(parents=True, exist_ok=True)

    qc_path = run_qc(config, aois=aois)
    gdf = gpd.read_parquet(qc_path)
    if aois:
        gdf = gdf[gdf["aoi"].isin(aois)]

    rows = []
    groups = list(gdf.groupby("aoi", sort=False))
    for n, (aoi, sub) in enumerate(groups, 1):
        gpkg = root / aoi / f"patches_reference_{aoi}.gpkg"
        if gpkg.exists() and not force:
            continue
        try:
            written = write_city_dir(aoi, sub, config, root)
        except Exception as exc:
            logger.warning(f"shape failed for {aoi}: {exc}")
            continue
        if "gpkg" not in written:
            continue
        p = gpd.read_file(written["gpkg"])
        if len(p) < min_patches:
            continue
        rows.append(
            {
                "aoi": aoi,
                "patches": len(p),
                "classes": int(p.LCZ_class.nunique()),
                "src_polys": int(sub["qc_pass"].sum()),
                "mean_weight": float(p.weight.mean()),
                "nbr_conflict_frac": float(p.nbr_conflict.mean()),
                "has_tif": "tif" in written,
            }
        )
        if n % 50 == 0:
            logger.info(f"  shaped {n:,}/{len(groups):,} AOIs")

    review = pd.DataFrame(rows).sort_values("patches", ascending=False)
    if len(review):
        review.to_parquet(Path(config.cache_dir) / f"shape_review_{config.config_hash}.parquet",
                          index=False)
        logger.info(
            f"shaped {len(review):,} AOIs | {int(review.patches.sum()):,} patches | "
            f"root {root}"
        )
    return review


def combine_patches(
    config: WudaptConfig,
    out_path: Path | None = None,
    *,
    out_root: Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Concatenate the per-AOI patch gpkgs into one global-arm GeoPackage.

    Joins the city-disjoint ``wudapt_split`` column from
    :mod:`lcz_wudapt.splits`, producing exactly what
    ``patch_classification.py --global-split --global-gpkg ... --split-col
    wudapt_split`` consumes. No new split code, and no new dataset code: the
    ``aoi`` column doubles as the city, which ``build_global_items`` now prefers
    over its So2Sat-only spatial join.

    ``patch_id`` is left exactly as written per AOI -- it is already AOI-prefixed
    and globally unique, and re-issuing it here would break the correspondence
    with any embeddings already extracted.
    """
    from .splits import assert_split_integrity

    root = Path(out_root) if out_root else Path(config.gpkg_path).parent / "cities"
    out_path = Path(out_path) if out_path else root.parent / "patches_wudapt_rxr.gpkg"

    frames = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        gpkg = d / f"patches_reference_{d.name}.gpkg"
        if gpkg.exists():
            frames.append(gpd.read_file(gpkg))
    if not frames:
        raise FileNotFoundError(f"no per-AOI patch gpkgs under {root}; run `shape` first")

    allp = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    if not allp["patch_id"].is_unique:
        dupes = allp.loc[allp["patch_id"].duplicated(), "patch_id"].head(5).tolist()
        raise ValueError(
            f"patch_id is not unique across AOIs (e.g. {dupes}). Every patch is "
            "extracted into one shared directory, so duplicates would overwrite "
            "each other's npy files."
        )

    split_path = Path(config.cache_dir) / f"wudapt_splits_{config.config_hash}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"run `lcz_wudapt splits` first ({split_path} missing)")
    splits = pd.read_parquet(split_path)
    assert_split_integrity(splits)

    allp = allp.merge(splits[["aoi", "wudapt_split", "region", "forced_test"]],
                      on="aoi", how="left", validate="many_to_one")
    n_unsplit = int(allp["wudapt_split"].isna().sum())
    if n_unsplit:
        logger.warning(f"{n_unsplit:,} patches have no split (AOI missing from the split table); dropping")
        allp = allp[allp["wudapt_split"].notna()]

    allp.to_file(out_path, driver="GPKG")
    review = (
        allp.groupby("aoi")
        .agg(patches=("patch_id", "size"), classes=("LCZ_class", "nunique"),
             split=("wudapt_split", "first"), region=("region", "first"),
             mean_weight=("weight", "mean"), nbr_conflict=("nbr_conflict", "mean"))
        .reset_index().sort_values("patches", ascending=False)
    )
    review.to_parquet(Path(config.cache_dir) / f"patch_review_{config.config_hash}.parquet", index=False)

    logger.info(
        f"wrote {out_path} — {len(allp):,} patches, {allp.aoi.nunique():,} AOIs, "
        f"{allp.LCZ_class.nunique()} classes, splits "
        f"{allp.wudapt_split.value_counts().to_dict()}"
    )
    return out_path, review
