"""Stages 5 + 6 — UCP -> LCZ decision rules + confidence / purity filters.

``classify_patch`` is a pure function over one UCP row (a dict) and a config; it
returns ``label_type`` (hard/coarse/unlabelled), the LCZ integer ``lcz`` (So2Sat
1-17 coding; null for coarse), an ``lcz_set`` (singleton for hard, e.g. [3,7] for
coarse), a name, a confidence in [0, 1], and a bag of diagnostics explaining every
decision. ``classify_patches`` applies it across a UCP DataFrame.

Rules are first-match-wins in the task's order: natural/non-built classes ->
special built (heavy industry, large low-rise) -> height x density matrix. Every
threshold comes from ``config`` — nothing here is hard-coded.

LCZ 7 (informal / lightweight low-rise): the compact+low matrix cell is routed by
footprint/road morphology (Stage 5b, ``_route_compact_low``) into hard 3, hard 7,
or **coarse {3,7}** — never guessed. When morphology is unreadable we emit coarse
(trained under a marginalised loss over ``lcz_set``), never a wrong hard class.
The same coarse mechanism covers the large-lowrise/heavy-industry {8,10} ambiguity.

LCZ integer coding (matches src/utils/constants.py ``lcz_dict``): built 1-10 map
to themselves; natural letters map A=11, B=12, C=13, D=14, E=15, F=16, G=17.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from loguru import logger

from utils.constants import lcz_dict  # src/utils/constants.py (on path via package __init__)

from .config import LczLabelConfig

# Letter -> So2Sat integer code
LCZ_A, LCZ_B, LCZ_C, LCZ_D, LCZ_E, LCZ_F, LCZ_G = 11, 12, 13, 14, 15, 16, 17
NATURAL_CODES = {LCZ_A, LCZ_B, LCZ_C, LCZ_D, LCZ_E, LCZ_F, LCZ_G}
MATRIX_CODES = {1, 2, 3, 4, 5, 6, 9}


def lcz_name(code: int | None) -> str | None:
    if code is None:
        return None
    return lcz_dict.get(int(code), {}).get("name")


def _f(row: dict, key: str, default: float = 0.0) -> float:
    """Fetch a fraction/number, treating missing/NaN as ``default``."""
    v = row.get(key, default)
    if v is None:
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def _natural_code(row: dict, t) -> tuple[int, float] | tuple[None, float]:
    """First matching natural class + its dominant evidence fraction (for
    confidence), or (None, 0.0). Rules require dominant evidence."""
    f_water = _f(row, "f_water")
    f_trees = _f(row, "f_trees")
    f_low = _f(row, "f_lowplants")
    f_shrub = _f(row, "f_shrub")
    f_sand = _f(row, "f_sand")
    f_rock = _f(row, "f_bare_rock")
    f_paved = _f(row, "f_paved_infra")

    if f_water >= t.water_min:
        return LCZ_G, f_water
    if f_trees >= t.trees_dense_min:
        return LCZ_A, f_trees
    if (t.trees_scatter_min <= f_trees < t.trees_dense_min
            and f_low + f_trees >= t.trees_scatter_veg_min):
        return LCZ_B, f_low + f_trees
    if f_shrub >= t.shrub_min:
        return LCZ_C, f_shrub
    if f_low >= t.lowplants_min:
        return LCZ_D, f_low
    if f_rock >= t.bare_rock_min or f_paved >= t.paved_infra_min:
        return LCZ_E, max(f_rock, f_paved)
    if f_sand >= t.sand_min:
        return LCZ_F, f_sand
    return None, 0.0


def _height_class(row: dict, t) -> str | None:
    h_mean = row.get("h_mean")
    h_max = _f(row, "h_max", default=0.0)
    n_tower = _f(row, "n_tower", default=0.0)
    if h_mean is None or (isinstance(h_mean, float) and math.isnan(h_mean)):
        # No height at all: high only if towers dominate, else undetermined.
        if h_max >= t.height_high_min and n_tower >= t.tower_count_min:
            return "high"
        return None
    h_mean = float(h_mean)
    if h_mean >= t.height_high_min or (h_max >= t.height_high_min and n_tower >= t.tower_count_min):
        return "high"
    if h_mean >= t.height_mid_min:
        return "mid"
    return "low"


def _density_class(bsf: float, t) -> str | None:
    if bsf >= t.bsf_compact_min:
        return "compact"
    if bsf >= t.bsf_open_min:
        return "open"
    if bsf >= t.bsf_sparse_min:
        return "sparse"
    return None


_MATRIX = {
    ("high", "compact"): 1, ("high", "open"): 4, ("high", "sparse"): None,
    ("mid", "compact"): 2, ("mid", "open"): 5, ("mid", "sparse"): None,
    ("low", "compact"): 3, ("low", "open"): 6,  # ("low","sparse") handled specially (LCZ 9)
}


def classify_patch(row: dict, config: LczLabelConfig) -> dict:
    """Classify one UCP row. Returns lcz/lcz_name/confidence + diagnostics."""
    t = config.classification
    cf = config.confidence
    bsf = _f(row, "bsf")
    ghs = row.get("ghs_built_s")
    ghs_known = ghs is not None and not (isinstance(ghs, float) and math.isnan(ghs))
    ghs_val = float(ghs) if ghs_known else float("nan")

    diag = {"lcz": None, "lcz_name": None, "lcz_set": [], "label_type": "unlabelled",
            "confidence": 0.0, "height_class": None, "density_class": None,
            "reject_reason": None, "suspect_ml": False, "boundary_flag": False,
            # Stage 5b router diagnostics
            "informal_morphology": False, "road_deficit": False,
            "formal_morphology": False, "mn_informal": False, "blob_suspect": False,
            "lcz7_strong_bsf": False}

    def reject(reason: str) -> dict:
        diag["reject_reason"] = reason
        return diag

    def hard(code: int, conf: float) -> dict:
        diag.update(lcz=int(code), lcz_name=lcz_name(code), lcz_set=[int(code)],
                    label_type="hard", confidence=conf,
                    suspect_ml=row.get("_suspect_ml", False),
                    boundary_flag=row.get("_boundary", False))
        return diag

    def coarse(codes: list[int], conf: float) -> dict:
        diag.update(lcz=None, lcz_name=None, lcz_set=[int(c) for c in codes],
                    label_type="coarse", confidence=conf,
                    suspect_ml=row.get("_suspect_ml", False),
                    boundary_flag=row.get("_boundary", False))
        return diag

    # ── Natural branch (low footprint density) ────────────────────────────────
    # Below the built floor the patch is either natural or under-mapped; it never
    # becomes a built class. Absence of features alone is NOT evidence (principle
    # 3): a natural label requires positive land-cover evidence.
    if bsf < t.natural_bsf_max:
        nat, dom_frac = _natural_code(row, t)
        if nat is None:
            return reject("no_positive_evidence")
        # Completeness trap: GHS-BUILT-S says built-up but Overture is empty ->
        # under-mapped (the informal-settlement trap), so drop it.
        if ghs_known and ghs_val > cf.completeness_ghs_min:
            return reject("completeness_trap")
        # Milder built signal (between the two thresholds) also disqualifies.
        if ghs_known and ghs_val >= t.natural_ghs_built_s_max:
            return reject("built_signal_present")
        # Confidence tracks purity: a patch that is 99% forest is more certain
        # than one barely over the 0.75 gate (gives a monotone confidence signal).
        return hard(nat, min(1.0, dom_frac))

    # ── Special built classes (before the matrix) ─────────────────────────────
    f_ind = _f(row, "f_industrial_lu")
    n_poi = _f(row, "n_heavy_industry_poi")
    if f_ind >= t.heavy_industry_lu_min and n_poi >= t.heavy_industry_poi_min:
        return hard(10, _built_confidence(row, 10, bsf, ghs_val, ghs_known, config))

    h_mean = row.get("h_mean")
    h_mean_low = (h_mean is None or (isinstance(h_mean, float) and math.isnan(h_mean))
                  or float(h_mean) < t.large_lowrise_h_max)
    if (bsf >= t.large_lowrise_bsf_min and h_mean_low
            and _f(row, "mean_footprint_area") >= t.large_lowrise_footprint_min
            and _f(row, "large_lowrise_frac") >= t.large_lowrise_type_frac):
        conf8 = _built_confidence(row, 8, bsf, ghs_val, ghs_known, config)
        # 8 vs 10 ambiguity: industrial land-use but no heavy-industry POI to
        # confirm 10 (poi>=1 was caught above) -> coarse {8,10}, don't guess.
        if f_ind >= t.heavy_industry_lu_min:
            return coarse([8, 10], conf8)
        return hard(8, conf8)

    # ── Height x density matrix (requires bsf >= sparse floor) ────────────────
    dens = _density_class(bsf, t)
    if dens is None:
        return reject("below_built_floor")
    hcls = _height_class(row, t)
    diag["density_class"] = dens
    diag["height_class"] = hcls

    if hcls is None:
        return reject("no_height_class")

    # Compact + low: OSM tags can't separate formal 3 from informal 7 — route by
    # footprint morphology + road topology (Stage 5b). Every such patch goes
    # through the router; none bypasses it.
    if hcls == "low" and dens == "compact":
        return _route_compact_low(row, config, bsf, ghs_val, ghs_known, diag, hard, coarse)

    if hcls == "low" and dens == "sparse":
        veg = _f(row, "f_lowplants") + _f(row, "f_trees")
        code = 9 if veg >= t.lcz9_veg_min else None
    else:
        code = _MATRIX.get((hcls, dens))
    if code is None:
        return reject("matrix_unlabelled")

    conf = _built_confidence(row, code, bsf, ghs_val, ghs_known, config)
    if conf is None:
        return reject("insufficient_height_evidence")
    return hard(code, conf)


def _route_compact_low(row, config, bsf, ghs_val, ghs_known, diag, hard, coarse) -> dict:
    """Stage 5b — LCZ 3 vs 7 vs coarse {3,7} from footprint/road morphology.

    Informal fabric (7) shows tiny footprints, extreme count density, high size
    irregularity, and many buildings per mapped road. When morphology is
    unreadable (ML footprints merged into blobs, or no clear signal) we emit
    coarse {3,7} rather than guess — the safe direction is always coarse.
    """
    r = config.router
    median_fp = _f(row, "median_footprint_area")
    count_density = _f(row, "building_count_density")
    cv = _f(row, "footprint_area_cv")
    bpr = _f(row, "buildings_per_road_km")
    road_density = _f(row, "road_length_density")
    f_google = _f(row, "f_google_source")
    height_evid = _f(row, "height_evidence_frac")
    mn = row.get("mn_informal_frac")
    mn_known = mn is not None and not (isinstance(mn, float) and math.isnan(mn))
    mn_frac = float(mn) if mn_known else float("nan")

    informal_morphology = (
        0 < median_fp <= r.median_footprint_informal_max
        and count_density >= r.count_density_informal_min
        and cv >= r.footprint_cv_informal_min
    )
    road_deficit = (
        bpr >= r.buildings_per_road_km_informal_min
        or (road_density < r.road_density_deficit_max
            and count_density >= r.count_density_informal_min)
    )
    formal_morphology = (
        median_fp >= r.median_footprint_formal_min
        and height_evid >= r.formal_height_evidence_min
        and not road_deficit
    )
    mn_informal = mn_known and mn_frac >= r.mn_informal_frac_min
    blob_suspect = (
        f_google < r.blob_google_frac_max
        and count_density < r.blob_count_density_max
        and bsf >= r.blob_bsf_min
    )
    diag.update(informal_morphology=informal_morphology, road_deficit=road_deficit,
                formal_morphology=formal_morphology, mn_informal=mn_informal,
                blob_suspect=blob_suspect, lcz7_strong_bsf=bsf >= r.lcz7_strong_bsf)

    # Shared confidence: compact+low+built is fully supported; height evidence is
    # WAIVED here (informal areas legitimately lack height tags).
    base_conf = _built_confidence(row, 3, bsf, ghs_val, ghs_known, config, waive_height=True)

    # Routing, first match wins.
    if blob_suspect:
        return coarse([3, 7], base_conf)                     # morphology unreadable
    if (informal_morphology and road_deficit) or mn_informal:
        cap = r.lcz7_confidence_cap_mn if mn_informal else r.lcz7_confidence_cap
        return hard(7, min(base_conf, cap))
    if formal_morphology:
        return hard(3, base_conf)
    return coarse([3, 7], base_conf)


def _built_confidence(row, code, bsf, ghs_val, ghs_known, config,
                      waive_height: bool = False) -> float | None:
    """Confidence for a built verdict; None means 'drop to unlabelled'.

    ``waive_height`` skips the height-evidence gate — used for the LCZ 7 / coarse
    {3,7} router outputs, where missing/raster-tier heights are expected in
    informal fabric and must not be penalized.
    """
    t = config.classification
    cf = config.confidence
    conf = 1.0

    # Height evidence (matrix classes 1-6 and 9; waived for the 3/7 router)
    if code in MATRIX_CODES and not waive_height:
        none_frac = _f(row, "height_none_frac")
        evid = _f(row, "height_evidence_frac")
        if none_frac > cf.height_none_max_area:
            return None
        if evid < cf.height_evidence_full:
            conf = min(conf, cf.raster_only_cap)   # heights mostly raster-tier

    # Purity: boundary patches straddling a density threshold
    for thr in (t.bsf_compact_min, t.bsf_open_min, t.bsf_sparse_min):
        if abs(bsf - thr) <= cf.boundary_band:
            conf *= cf.boundary_penalty
            row["_boundary"] = True
            break

    # ML-footprint sanity: built footprints with ~zero GHS-BUILT-S, all ML-sourced
    if (bsf >= t.bsf_sparse_min and ghs_known and ghs_val < cf.suspect_ml_ghs_max
            and _f(row, "built_area_ml_frac", default=0.0) > 0.99):
        conf *= cf.suspect_ml_penalty
        row["_suspect_ml"] = True

    conf = max(0.0, conf)
    return conf if conf > cf.min_emit or cf.min_emit == 0.0 else None


def classify_patches(ucp_df: pd.DataFrame, config: LczLabelConfig) -> pd.DataFrame:
    """Apply ``classify_patch`` across a UCP table; returns aligned diagnostics."""
    records = [classify_patch(dict(r), config) for r in ucp_df.to_dict("records")]
    out = pd.DataFrame.from_records(records)
    out["lcz"] = out["lcz"].astype("Int64")
    n_hard = int((out["label_type"] == "hard").sum())
    n_coarse = int((out["label_type"] == "coarse").sum())
    n7 = int((out["lcz"] == 7).sum())
    logger.info(f"Classified: {n_hard} hard ({n7} LCZ-7), {n_coarse} coarse, "
                f"{len(out) - n_hard - n_coarse} unlabelled / {len(out)}")
    return out


# Blocks classify through the identical rules — the row dict doesn't care what
# geometry produced its UCPs.
classify_blocks = classify_patches


# ── Stage 5c — zone formation ─────────────────────────────────────────────────

def _label_key(label_type: str, lcz, lcz_set) -> str | None:
    """Dissolve key: identical hard class, or identical coarse set."""
    if label_type == "hard":
        return f"h{int(lcz)}"
    if label_type == "coarse":
        return "c" + ",".join(str(int(c)) for c in sorted(lcz_set))
    return None


def form_zones(
    labels: pd.DataFrame,
    blocks: pd.DataFrame,
    adjacency: pd.DataFrame,
    config: LczLabelConfig,
) -> pd.DataFrame:
    """Dissolve adjacent same-label blocks into zones; grade them by area.

    A block is smaller than an LCZ ("zone of uniform structure", hundreds of m
    to km): adjacent blocks with an identical hard ``lcz`` (or identical coarse
    ``lcz_set``) form a candidate zone, and a block is **zone-grade** iff its
    zone's contiguous area >= ``zones.min_zone_area_ha``. Smaller islands keep
    their block label with ``zone_grade=False`` — a single tower block inside
    lowrise fabric is an anomaly flag, not "an LCZ 1 zone". The Stage 6
    confidence penalty for non-zone-grade labels is applied by the caller.

    Args:
        labels: per-block classification with ``block_id, label_type, lcz,
            lcz_set`` (unlabelled rows get null zone columns).
        blocks: block table with ``block_id, area_m2``.
        adjacency: shared-edge edge list ``block_a, block_b``.

    Returns columns ``block_id, zone_id, zone_area_ha, zone_grade``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    lab = labels[["block_id", "label_type", "lcz", "lcz_set"]].merge(
        blocks[["block_id", "area_m2"]], on="block_id", validate="1:1"
    )
    lab["_key"] = [
        _label_key(t, c, s)
        for t, c, s in zip(lab["label_type"], lab["lcz"], lab["lcz_set"])
    ]
    labelled = lab[lab["_key"].notna()].reset_index(drop=True)
    out = pd.DataFrame({
        "block_id": lab["block_id"],
        "zone_id": pd.Series([None] * len(lab), dtype=object),
        "zone_area_ha": np.nan,
        "zone_grade": False,
    })
    if labelled.empty:
        return out

    pos = {b: i for i, b in enumerate(labelled["block_id"])}
    key_of = dict(zip(labelled["block_id"], labelled["_key"]))
    edges = adjacency[
        adjacency["block_a"].isin(pos) & adjacency["block_b"].isin(pos)
    ]
    same = edges[
        edges["block_a"].map(key_of).to_numpy() == edges["block_b"].map(key_of).to_numpy()
    ]
    n = len(labelled)
    ii = same["block_a"].map(pos).to_numpy()
    jj = same["block_b"].map(pos).to_numpy()
    graph = coo_matrix((np.ones(len(same)), (ii, jj)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)

    comp_df = pd.DataFrame({
        "block_id": labelled["block_id"], "area_m2": labelled["area_m2"], "comp": comp,
    })
    agg = comp_df.groupby("comp").agg(
        zone_area_m2=("area_m2", "sum"), zone_id=("block_id", "min")
    )
    comp_df = comp_df.merge(agg, left_on="comp", right_index=True)
    comp_df["zone_area_ha"] = comp_df["zone_area_m2"] / 1.0e4
    comp_df["zone_grade"] = comp_df["zone_area_ha"] >= config.zones.min_zone_area_ha

    zoned = out.drop(columns=["zone_id", "zone_area_ha", "zone_grade"]).merge(
        comp_df[["block_id", "zone_id", "zone_area_ha", "zone_grade"]],
        on="block_id", how="left",
    )
    zoned["zone_grade"] = zoned["zone_grade"].fillna(False).astype(bool)
    # zone_id/zone_area_ha are NA (not necessarily Python None — pandas' string
    # dtype uses its own NA marker) for unlabelled blocks; use pd.isna(), not
    # `is None`, downstream.
    n_zones = int(agg.shape[0])
    n_graded = int(comp_df["zone_grade"].sum())
    logger.info(
        f"Zones: {n_zones} from {n} labelled blocks; "
        f"{n_graded} blocks zone-grade ({n_graded / max(n, 1):.0%})"
    )
    return zoned
