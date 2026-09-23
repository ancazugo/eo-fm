"""LCZ-Generator / WUDAPT training-area quality control.

This module implements the quality-control regime the LCZ Generator and WUDAPT
actually publish, rather than inventing one. Sources:

* Zenodo 13869766 — the release notes documenting every column, including what
  each ``qc_step`` flags.
* Demuzere et al. 2021, *The LCZ Generator* (Front. Environ. Sci.) — QC steps
  1-3, the bootstrap accuracy protocol, the oversize reduction.
* Demuzere et al. 2022, *A global map of Local Climate Zones* (ESSD) — the
  submission-priority rule used to resolve duplicates for the global map.
* https://www.wudapt.org/digitize-training-areas/ — the digitizing guidance
  (>200 m narrowest width, >100 m buffer between LCZs, 5-15 polygons per class).

Two findings from measuring the 2024-10-01 release shape everything here.

**QC step 1 is exactly reconstructible, and as shipped it is latitude-biased.**
``qc_step1 == True`` (which means *passed*, not *flagged*) is precisely
``area >= 0.04 km2 AND shape < 3``, with zero exceptions in 630,311 rows. But it
was computed on the shipped ``area`` column, which is **Web Mercator** and
therefore inflated by 1/cos^2(lat). Measured against true UTM area on a
30,000-polygon sample, the ratio tracks 1/cos^2(lat) to four decimals
(correlation 1.0 across latitude bands):

======================  =========  ==============  ===========  ==============
|latitude|              ratio      1/cos^2(lat)    agreement    shipped pass
======================  =========  ==============  ===========  ==============
0-10 deg                1.0122     1.0124          0.998        0.561
30-40 deg               1.3837     1.3857          0.914        0.615
50-60 deg               2.6646     2.6745          0.832        0.859
======================  =========  ==============  ===========  ==============

**Every one of the 2,013 disagreements is in the same direction** -- shipped
passes, recomputed fails -- so the released QC is progressively *more lenient*
away from the equator: at 55 deg N a 0.04 km2 threshold admits polygons that are
really 0.015 km2. The apparent rise in pass rate with latitude (0.56 -> 0.86) is
almost entirely a projection artefact; corrected, it is far flatter
(0.56 -> 0.69).

That matters directly here. Trusting the flag would admit small, unreliable
polygons preferentially in Europe, Russia and Canada while holding tropical
cities to a stricter standard -- biasing against precisely the African, Asian and
Latin American coverage this label set exists to add. So the flag is not trusted:
:func:`geometry_metrics` recomputes area and perimeter in the AOI's local UTM and
:func:`qc_step1` re-derives the rule there.

**The neighbour rule is the sharpest quality signal available.** The fraction of
candidate patches *not* within 100 m of a differently-labelled polygon tracks
So2Sat agreement better than ``oa``, ``oau`` or any other shipped column --
Tehran 0.19 and Delhi 0.23 against Berlin 0.80 and Bogota 0.97, matching their
measured agreements (0.49, 0.38 vs 0.94). It is therefore computed and emitted
per polygon, but **not applied as a filter by default**: at a hard 100 m cut it
deletes most of Tehran and Delhi, and the whole point of covering those cities is
that they are hard. It enters as a soft weight and a sweepable column.

Nothing is dropped silently. Every gate writes a boolean column, so any operating
point can be recovered from the emitted parquet without re-running the stage.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger

from .config import QcRules, WudaptConfig
from .ingest import BUILT_CLASSES, N_LCZ, RURAL_AOI

__all__ = [
    "apply_qc",
    "geometry_metrics",
    "neighbour_relations",
    "polygon_weights",
    "project_valid",
    "qc_step1",
    "reduce_oversize",
    "resolve_duplicates",
    "run_qc",
    "shape_index",
    "submission_gates",
    "temporal_alignment",
    "utm_crs",
]

# Search radius for the nearest different-class polygon. Beyond this the exact
# distance is irrelevant -- every weight and flag has long since saturated -- and
# bounding it keeps sjoin_nearest linear on dense AOIs like Wuhan (44k polygons).
_NBR_SEARCH_M = 2000.0


def utm_crs(gdf: gpd.GeoDataFrame):
    """Local UTM zone for an AOI.

    Areas, perimeters, buffers and distances must all be metric and locally
    conformal. EPSG:6933 (used by :mod:`~lcz_wudapt.ingest` for the global area
    column) is equal-area but anisotropic away from the equator, so a negative
    buffer on it is not a circle and the 160 m erosion test would be wrong by
    ~30% at Berlin's latitude.
    """
    return gdf.estimate_utm_crs()


# GEOS overlay precision. Reprojecting a valid EPSG:4326 polygon into UTM can
# re-invalidate it (Tehran raises "side location conflict" without this), so
# every projection is revalidated and every overlay runs on a 1 mm grid. At
# 1 mm the area error is far below the 10 m pixels these labels feed.
_GRID_SIZE_M = 0.001


def project_valid(gdf: gpd.GeoDataFrame, crs=None) -> gpd.GeoDataFrame:
    """Reproject to a metric CRS and repair what the reprojection broke.

    ``ingest.clean`` already runs ``make_valid`` in EPSG:4326, but validity is
    not preserved by reprojection: coordinates are recomputed in a new plane and
    near-degenerate rings can cross. Skipping this makes overlay failures look
    like data problems in specific cities rather than a coordinate-system
    artefact affecting all of them.
    """
    out = gdf.to_crs(crs if crs is not None else utm_crs(gdf))
    bad = ~out.geometry.is_valid
    if bool(bad.any()):
        out = out.copy()
        out.loc[bad, "geometry"] = shapely.make_valid(out.loc[bad, "geometry"].values)
    return out


def shape_index(area_m2: np.ndarray, perimeter_m: np.ndarray) -> np.ndarray:
    """Generator ``shape`` column: ``P^2 / (4*pi*A)``.

    1.0 for a circle, 1.273 for a square, growing without bound as a polygon
    becomes elongated or crenellated. The Generator flags ``>= 3``.
    """
    area = np.asarray(area_m2, dtype="float64")
    per = np.asarray(perimeter_m, dtype="float64")
    out = np.full(area.shape, np.inf, dtype="float64")
    ok = area > 0
    out[ok] = per[ok] ** 2 / (4.0 * np.pi * area[ok])
    return out


def geometry_metrics(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Recompute area, perimeter and shape in local UTM.

    Returns a frame indexed like ``gdf`` with ``area_km2_utm``, ``perimeter_m``,
    ``shape_utm`` and ``fits_patch_m`` -- the largest inscribed-square side the
    polygon admits, probed by negative buffering (see :func:`fits_square`).
    """
    projected = project_valid(gdf)
    area = projected.geometry.area.to_numpy()
    per = projected.geometry.length.to_numpy()
    return pd.DataFrame(
        {
            "area_km2_utm": area / 1e6,
            "perimeter_m": per,
            "shape_utm": shape_index(area, per),
        },
        index=gdf.index,
    )


def fits_square(geometry: gpd.GeoSeries, side_m: float) -> np.ndarray:
    """Can each polygon contain an axis-free square of ``side_m``?

    Tested as ``buffer(-side/2)`` being non-empty. This is the WUDAPT ">200 m
    wide at the narrowest point" rule retargeted to the So2Sat patch size, and it
    is a genuine morphological test rather than an area proxy: a 2 km x 100 m
    strip has ample area and fits nothing.

    ``geometry`` must already be in a metric CRS.
    """
    eroded = geometry.buffer(-float(side_m) / 2.0)
    return (~eroded.is_empty).to_numpy()


def qc_step1(df: pd.DataFrame, rules: QcRules) -> pd.Series:
    """Generator QC step 1, re-derived on true metric geometry.

    ``area >= min_area_km2 AND shape < max_shape``. Uses the recomputed UTM
    columns, never the shipped Web-Mercator ones.
    """
    return (df["area_km2_utm"] >= rules.min_area_km2) & (df["shape_utm"] < rules.max_shape)


def submission_gates(df: pd.DataFrame, rules: QcRules) -> pd.DataFrame:
    """Submission-level accuracy gates.

    The 0.50 overall-accuracy floor is Bechtel et al. (2019a), adopted by both
    the Generator's automated QC and the ESSD global map ("only TA samples mapped
    to LCZs with an overall accuracy greater than 50% are kept").

    ``oau`` (urban-only accuracy) is the right metric for built classes 1-10 and
    ``oa`` for natural 11-17, mirroring how the Generator reports them; a
    submission can be strong overall and useless for LCZ 7 specifically, which is
    what the per-class ``f1_{c}`` gate catches.
    """
    built = df["class"].isin(BUILT_CLASSES)
    oa = pd.to_numeric(df["oa"], errors="coerce")
    oau = pd.to_numeric(df.get("oau"), errors="coerce") if "oau" in df else pd.Series(np.nan, index=df.index)

    # Per-polygon accuracy: the metric appropriate to the class it carries.
    acc = oa.where(~built, oau.fillna(oa))

    f1 = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in range(1, N_LCZ + 1):
        col = f"f1_{c}"
        if col in df.columns:
            m = df["class"] == c
            f1[m] = pd.to_numeric(df.loc[m, col], errors="coerce")

    out = pd.DataFrame(index=df.index)
    out["acc"] = acc
    out["class_f1"] = f1
    # NaN accuracy passes: a missing metric is not evidence of a bad submission,
    # and ~0 rows in the release lack `oa`. It is recorded so it can be swept.
    out["gate_oa"] = (oa >= rules.min_oa) | oa.isna()
    out["gate_oau"] = (~built) | (oau >= rules.min_oau) | oau.isna()
    out["gate_f1"] = (f1 >= rules.min_class_f1) | f1.isna()
    return out


def resolve_duplicates(gdf: gpd.GeoDataFrame) -> pd.Series:
    """ESSD duplicate rule: on contested ground, the best submission wins.

    The global-map paper resolves geographic duplicates by source priority
    (RUB > ARC > GEN) and otherwise keeps "the submission with the highest
    overall accuracy". This release is all GEN, so only the accuracy rule
    applies. Ties break on the earliest ``submission_date``, matching the Zenodo
    note that duplicate geometries keep the first submitted version.

    This deliberately replaces the multi-annotator consensus machinery of the
    superseded path: the papers specify a priority rule, not a posterior, and a
    priority rule cannot manufacture a class that no annotator drew.

    Returns a boolean Series: True where the polygon **wins** its contested
    group (or is uncontested).
    """
    if gdf.empty:
        return pd.Series(dtype=bool)

    # The rank/loss bookkeeping below is positional (sjoin's _i/_j are positions
    # into this frame), so work on a positional copy and restore the caller's
    # index at the end. Relying on the caller having reset its index would make
    # this correct only by coincidence.
    original_index = gdf.index
    gdf = gdf.reset_index(drop=True)

    acc = pd.to_numeric(gdf.get("acc"), errors="coerce").fillna(0.0)
    sub_date = pd.to_datetime(gdf.get("submission_date"), errors="coerce", utc=True)

    # Rank every polygon once: higher accuracy first, then earlier submission.
    order = pd.DataFrame({"acc": acc, "date": sub_date}, index=gdf.index)
    order["rank"] = order.sort_values(
        ["acc", "date"], ascending=[False, True], na_position="last"
    ).assign(r=np.arange(len(order))).sort_index()["r"]

    left = gdf[["class", "geometry"]].copy()
    left["_i"] = np.arange(len(left))
    pairs = gpd.sjoin(
        left, left.rename(columns={"class": "class_r", "_i": "_j"}),
        predicate="intersects", how="inner",
    )
    # Overlaps between *different* classes are the only contested ones; two
    # polygons of the same class agreeing is not a conflict to resolve.
    pairs = pairs[(pairs["_i"] != pairs["_j"]) & (pairs["class"] != pairs["class_r"])]
    if pairs.empty:
        return pd.Series(True, index=original_index)

    rank = order["rank"].to_numpy()
    i = pairs["_i"].to_numpy()
    j = pairs["_j"].to_numpy()
    # A polygon loses if any differently-classed polygon overlapping it ranks
    # better. Rank is a total order, so exactly one member of each contested
    # cluster survives against any given rival.
    loses = np.zeros(len(gdf), dtype=bool)
    np.logical_or.at(loses, i, rank[j] < rank[i])
    return pd.Series(~loses, index=original_index)


def neighbour_relations(gdf: gpd.GeoDataFrame, rules: QcRules) -> pd.DataFrame:
    """The "relationship to other labels" rules, per AOI, in local UTM.

    WUDAPT asks annotators to "leave a buffer of > 100 m between LCZs, if there
    is a clear boundary". Polygons that violate it sit on contested or
    transitional ground, and measuring the violation turns out to be the best
    quality signal in the dataset (see the module docstring).

    Emits three columns:

    ``nbr_dist_m``
        Distance to the nearest polygon of a *different* class, capped at
        :data:`_NBR_SEARCH_M`. ``inf`` when the AOI has only one class.
    ``nbr_conflict``
        ``nbr_dist_m < neighbour_buffer_m``.
    ``overlap_frac_diff_class``
        Fraction of the polygon's area overlapped by a different class. Distinct
        from proximity: a polygon can be well separated from most neighbours and
        still be half-covered by one disagreeing annotator.

    Implemented with per-class ``sjoin_nearest``/``sjoin``. Note the obvious
    ``dissolve(by="class").union_all()`` formulation is not used: it raises on
    real AOIs (Berlin, 549 polygons) and is quadratic on dense ones.
    """
    idx = gdf.index
    dist = pd.Series(np.inf, index=idx, dtype="float64")
    ov = pd.Series(0.0, index=idx, dtype="float64")
    if len(gdf) < 2 or gdf["class"].nunique() < 2:
        return pd.DataFrame(
            {"nbr_dist_m": dist, "nbr_conflict": dist < rules.neighbour_buffer_m,
             "overlap_frac_diff_class": ov}
        )

    proj = project_valid(gdf)
    areas = proj.geometry.area.to_numpy()

    for cls, sub in proj.groupby("class"):
        others = proj[proj["class"] != cls]
        if others.empty:
            continue
        left = sub[["geometry"]].copy()
        right = others[["geometry"]].copy()

        near = gpd.sjoin_nearest(
            left, right, how="left", distance_col="_d", max_distance=_NBR_SEARCH_M
        )
        if not near.empty:
            dist.loc[sub.index] = near.groupby(level=0)["_d"].min().reindex(sub.index).fillna(np.inf)

        hit = gpd.sjoin(left, right, predicate="intersects", how="inner")
        if not hit.empty:
            # Sum intersection area per left polygon. index_right indexes `right`
            # positionally after reset, so map back through its index.
            # shapely-level, not GeoSeries.intersection: the pairs are a
            # many-to-many join, so the two sides share no index to align on.
            inter = shapely.area(
                shapely.intersection(
                    hit.geometry.to_numpy(),
                    right.geometry.loc[hit["index_right"]].to_numpy(),
                    grid_size=_GRID_SIZE_M,
                )
            )
            agg = pd.Series(inter, index=hit.index).groupby(level=0).sum()
            ov.loc[agg.index] = (agg / pd.Series(areas, index=proj.index).loc[agg.index]).clip(0, 1)

    return pd.DataFrame(
        {
            "nbr_dist_m": dist,
            "nbr_conflict": dist < rules.neighbour_buffer_m,
            "overlap_frac_diff_class": ov,
        }
    )


def reduce_oversize(geometry: gpd.GeoSeries, rules: QcRules) -> gpd.GeoSeries:
    """Generator oversize reduction: >1.5 km2 polygons shrink to a ~350 m core.

    "large polygons (>1.5 km2) have surface area reduced to ~350 m radius before
    classification". 4.9% of the release exceeds the threshold, and they are
    disproportionately water and dense trees -- which is exactly how a handful of
    lakes came to supply 42% of an earlier patch pool.

    ``geometry`` must be in a metric CRS. Applied to the **patch** path only, not
    to the segmentation raster: dense per-pixel supervision has no equivalent
    failure mode and throwing away labelled area there would be pure loss.
    """
    big = geometry.area > rules.oversize_km2 * 1e6
    if not bool(big.any()):
        return geometry
    out = geometry.copy()
    cores = geometry[big].representative_point().buffer(rules.oversize_core_radius_m)
    out[big] = geometry[big].intersection(cores)
    return out


def temporal_alignment(label_year: pd.Series, rules: QcRules) -> pd.DataFrame:
    """Year-match the embedding, then softly weight the residual lag.

    So2Sat's imagery is 2016-18; WUDAPT's is not. Measured over the whole
    release, the median lag from 2017 is **4 years** and only 2.0% of polygons
    are from 2017 itself, so a fixed-2017 embedding mismatches almost everything.

    Two corrections, in order of importance:

    1. **Pick the nearest available embedding year.** With coop's {2017, 2025}
       the residual lag never exceeds 4 years, against a median of 4 and a
       maximum of 27 for a fixed 2017. This is the correction that removes real
       mismatch rather than merely discounting it.
    2. **Weight the residual softly**: ``exp(-lag / time_decay_years)``.

    ``time_decay_years`` defaults to 8.0, not the 3.0 of the superseded consensus
    path. At tau=3 fully **80.1%** of the corpus falls below weight 0.5 -- a hard
    filter wearing a soft filter's clothes -- against 29.6% at tau=8. Cities
    change slowly; age should tilt the weighting, not gut the dataset. Set
    ``time_decay_years=None`` for the tau=inf arm.

    Caution when tuning: ``oa`` *declines* with recency in this release (2019:
    0.768, 2022: 0.638, 2023: 0.615), so down-weighting old labels systematically
    up-weights less accurate ones. ``w_time`` and ``w_acc`` partly cancel and
    must be fitted and reported together, never one at a time.
    """
    years = pd.to_numeric(label_year, errors="coerce")
    available = np.asarray(sorted(rules.embedding_years), dtype="float64")

    if len(available) == 0:
        raise ValueError("QcRules.embedding_years is empty; cannot year-match")

    yv = years.to_numpy(dtype="float64")
    # Nearest available year per polygon; ties go to the earlier year.
    lag_matrix = np.abs(yv[:, None] - available[None, :])
    pick = np.nanargmin(np.where(np.isnan(lag_matrix), np.inf, lag_matrix), axis=1)
    emb_year = available[pick]
    lag = np.abs(yv - emb_year)

    # A polygon with no usable date gets the median lag rather than a free pass:
    # unknown provenance is not evidence of freshness.
    unknown = np.isnan(yv)
    if unknown.any():
        emb_year[unknown] = available[len(available) // 2]
        lag[unknown] = np.nanmedian(lag[~unknown]) if (~unknown).any() else 0.0

    if rules.time_decay_years is None:
        w = np.ones_like(lag)
    else:
        w = np.exp(-lag / float(rules.time_decay_years))

    return pd.DataFrame(
        {
            "embedding_year": pd.Series(emb_year, index=label_year.index).astype("Int16"),
            "year_lag": pd.Series(lag, index=label_year.index).astype("float32"),
            "w_time": pd.Series(w, index=label_year.index).astype("float32"),
        }
    )


def polygon_weights(df: pd.DataFrame, rules: QcRules) -> pd.DataFrame:
    """Per-polygon training weight, ``w = w_acc * w_time * w_area * w_nbr``.

    Each factor is in (0, 1] and none can zero the weight out, because a zero
    weight is a silent drop and this module drops nothing silently -- gates are
    booleans, weights are soft.

    * ``w_acc`` — linear ramp from ``acc_floor`` at ``acc_oa_min`` to 1.0 at
      ``acc_oa_min + acc_oa_span``, on the accuracy metric appropriate to the
      polygon's class.
    * ``w_time`` — from :func:`temporal_alignment`.
    * ``w_area`` — saturating in area, encoding the WUDAPT ">1 km2 is optimal"
      guidance without hard-thresholding on it.
    * ``w_nbr`` — soft ramp on ``nbr_dist_m``, floored at 0.5. Deliberately soft:
      the cities where it bites hardest (Tehran 0.19 clean, Delhi 0.23) are
      precisely the cities worth covering, so proximity discounts a label rather
      than deleting it.
    """
    acc = pd.to_numeric(df["acc"], errors="coerce")
    w_acc = ((acc - rules.acc_oa_min) / rules.acc_oa_span).clip(rules.acc_floor, 1.0)
    w_acc = w_acc.fillna(1.0)

    area = df["area_km2_utm"].to_numpy()
    w_area = np.clip(0.5 + 0.5 * np.sqrt(np.clip(area, 0, None) / 1.0), 0.5, 1.0)

    nbr = df["nbr_dist_m"].to_numpy()
    w_nbr = np.clip(0.5 + 0.5 * np.minimum(nbr, rules.neighbour_buffer_m) / rules.neighbour_buffer_m, 0.5, 1.0)

    out = pd.DataFrame(index=df.index)
    out["w_acc"] = w_acc.astype("float32")
    out["w_area"] = pd.Series(w_area, index=df.index).astype("float32")
    out["w_nbr"] = pd.Series(w_nbr, index=df.index).astype("float32")
    out["weight"] = (
        out["w_acc"] * df["w_time"].astype("float32") * out["w_area"] * out["w_nbr"]
    ).astype("float32")
    return out


def apply_qc(gdf: gpd.GeoDataFrame, config: WudaptConfig) -> gpd.GeoDataFrame:
    """Run the whole QC regime for one AOI. Returns every flag, drops nothing.

    Column contract added to the cleaned ingest frame:

    ==========================  =========================================
    ``area_km2_utm``            true area, local UTM
    ``perimeter_m``             true perimeter
    ``shape_utm``               ``P^2/(4 pi A)``, the Generator's index
    ``qc1``                     re-derived step 1 (area + shape)
    ``qc1_shipped``             the released flag, for the V1 agreement check
    ``fits_patch``             contains a ``patch_size_m`` square
    ``acc`` / ``class_f1``      accuracy metrics for this polygon's class
    ``gate_oa/oau/f1``          submission-level gates
    ``wins_conflict``           survives the ESSD duplicate-priority rule
    ``nbr_dist_m``              distance to nearest different-class polygon
    ``nbr_conflict``            within ``neighbour_buffer_m`` of one
    ``overlap_frac_diff_class`` area fraction overlapped by another class
    ``embedding_year``          nearest available embedding epoch
    ``year_lag`` / ``w_time``   residual temporal mismatch and its weight
    ``w_acc/w_area/w_nbr``      weight factors
    ``weight``                  the product, in (0, 1]
    ``qc_pass``                 all hard gates, the recommended default subset
    ==========================  =========================================
    """
    rules = config.qc
    if gdf.empty:
        return gdf.copy()

    out = gdf.copy().reset_index(drop=True)
    out = pd.concat([out, geometry_metrics(out)], axis=1)

    out["qc1"] = qc_step1(out, rules)
    out["qc1_shipped"] = out["qc_step1"].astype("boolean")

    proj = project_valid(out)
    out["fits_patch"] = fits_square(proj.geometry, rules.patch_size_m)

    out = pd.concat([out, submission_gates(out, rules)], axis=1)
    out = pd.concat([out, neighbour_relations(out, rules)], axis=1)
    out = pd.concat([out, temporal_alignment(out["label_year"], rules)], axis=1)
    out["wins_conflict"] = resolve_duplicates(out).to_numpy()
    out = pd.concat([out, polygon_weights(out, rules)], axis=1)

    out["qc_pass"] = (
        out["qc1"]
        & out["gate_oa"].fillna(True)
        & out["gate_oau"].fillna(True)
        & out["gate_f1"].fillna(True)
    )
    if rules.use_conflict_priority:
        out["qc_pass"] &= out["wins_conflict"]
    if rules.drop_neighbour_conflicts:
        out["qc_pass"] &= ~out["nbr_conflict"]

    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def run_qc(
    config: WudaptConfig,
    *,
    aois: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Apply :func:`apply_qc` per AOI and cache one parquet for the whole corpus.

    Keyed on ``config_hash`` (not ``ingest_hash``): changing a QC threshold must
    invalidate this, but must not force a re-read of all 630k polygons upstream.
    """
    from .ingest import ingest

    cache = Path(config.cache_dir)
    out_path = cache / f"wudapt_qc_{config.config_hash}.parquet"
    if out_path.exists() and not force:
        logger.info(f"qc cache hit: {out_path.name}")
        return out_path

    clean_path, _ = ingest(config)
    gdf = gpd.read_parquet(clean_path)
    gdf = gdf[gdf["aoi"] != RURAL_AOI]
    if aois:
        gdf = gdf[gdf["aoi"].isin(aois)]

    frames = []
    groups = list(gdf.groupby("aoi", sort=False))
    for n, (aoi, sub) in enumerate(groups, 1):
        try:
            frames.append(apply_qc(sub, config))
        except Exception as exc:  # one bad AOI must not lose the other 1,250
            logger.warning(f"QC failed for {aoi} ({len(sub):,} polys): {exc}")
        if n % 100 == 0:
            logger.info(f"  QC {n:,}/{len(groups):,} AOIs")

    res = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=gdf.crs)
    res.to_parquet(out_path)

    logger.info(
        f"wrote {out_path.name}: {len(res):,} polygons, "
        f"qc1 {res.qc1.mean():.3f}, fits_patch {res.fits_patch.mean():.3f}, "
        f"qc_pass {res.qc_pass.mean():.3f}, mean weight {res.weight.mean():.3f}"
    )
    return out_path
