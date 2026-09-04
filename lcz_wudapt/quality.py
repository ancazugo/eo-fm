"""H2 — author collapse, quality gates and the annotator weight model.

Runs *before* any consensus is formed, because the whole harmonisation argument
rests on cleaning first: pooled class agreement between overlapping polygons is
only ~0.71 by area, and as low as 0.38 (Delhi) and 0.44 (Guangzhou).

Three things happen here, in order:

1. **Author collapse.** The unit of a vote is the author, not the submission.
   8,827 submissions collapse to ~1,500 named authors (Wuhan: 388 -> 41), and a
   single author averages 3.02 versions per city. Counting submissions as
   independent votes would let prolific resubmitters dominate consensus.
   Collapse is expressed as a deterministic *burn order* rather than a geometric
   merge: within one author, polygons are rasterised oldest-submission-first so
   a later revision overwrites its own earlier version on overlapping ground
   while disjoint earlier work survives. This is exact, and costs no geometry.
2. **Quality gates** on submissions and polygons.
3. **The weight model** ``w = w_qc * w_acc * w_time * w_size``, one weight per
   polygon, consumed by :mod:`lcz_wudapt.consensus`.

Every threshold comes from :class:`~lcz_wudapt.config.QualityGates` /
``ConsensusParams``; the operating point is calibrated against So2Sat agreement
in :mod:`lcz_wudapt.audit` (gate G0), never hard-coded here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .config import WudaptConfig
from .ingest import BUILT_CLASSES, N_LCZ

__all__ = [
    "apply_gates",
    "burn_order",
    "polygon_weights",
    "submission_accuracy",
]

# Sort keys defining the painter's-algorithm burn order (see module docstring).
_BURN_KEYS = ["aoi", "annotator_id", "submission_date", "submission_id"]


def burn_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort into the canonical rasterisation order (author collapse).

    Oldest submission first within each (aoi, author), so the newest revision is
    burned last and wins wherever an author overlaps their own earlier work.
    ``submission_id`` breaks date ties so the order is deterministic.
    """
    missing = [c for c in _BURN_KEYS if c not in df.columns]
    if missing:
        raise KeyError(f"burn_order needs {missing}")
    return df.sort_values(_BURN_KEYS, kind="mergesort").reset_index(drop=True)


def submission_accuracy(df: pd.DataFrame) -> np.ndarray:
    """The accuracy figure appropriate to each polygon's class.

    LCZ-Generator reports several cross-validation accuracies per submission.
    ``oau`` (urban) is the discriminating one for built classes 1-10; ``oa`` is
    used for the natural classes 11-17 (A-G), where the urban figure says
    nothing. Falls back to ``oa`` wherever ``oau`` is missing.
    """
    oa = pd.to_numeric(df["oa"], errors="coerce").to_numpy(dtype="float64")
    oau = pd.to_numeric(df["oau"], errors="coerce").to_numpy(dtype="float64")
    built = df["class"].isin(BUILT_CLASSES).to_numpy()
    out = np.where(built & np.isfinite(oau), oau, oa)
    return np.nan_to_num(out, nan=0.0)


def _class_f1(df: pd.DataFrame) -> np.ndarray:
    """Per-submission F1 for the class each polygon actually carries.

    A submission with a strong overall score can still be useless for one class
    — LCZ 7 especially — so the weight model reads the class-specific column
    rather than the headline accuracy.
    """
    cls = df["class"].to_numpy()
    out = np.full(len(df), np.nan)
    for c in range(1, N_LCZ + 1):
        col = f"f1_{c}"
        if col not in df.columns:
            continue
        sel = cls == c
        if sel.any():
            out[sel] = pd.to_numeric(df.loc[sel, col], errors="coerce").to_numpy()
    return out


def apply_gates(df: pd.DataFrame, config: WudaptConfig) -> pd.DataFrame:
    """Drop polygons failing the H2 submission- and polygon-level gates.

    Returns the surviving rows with a ``gate_drop_reason`` column removed; the
    per-reason counts are logged so a filter sweep is auditable.
    """
    q = config.quality
    n0 = len(df)
    reasons: dict[str, np.ndarray] = {}

    # Submission-level. NA QC values are treated as "not a pass" only when the
    # gate is switched on, so an unparsed token cannot sneak through.
    for step, required in (
        ("qc_step1", q.require_qc_step1),
        ("qc_step2", q.require_qc_step2),
        ("qc_step3", q.require_qc_step3),
    ):
        if required:
            reasons[step] = ~(df[step].fillna(False).to_numpy(dtype=bool))

    oa = pd.to_numeric(df["oa"], errors="coerce").to_numpy()
    if q.min_oa > 0:
        reasons["min_oa"] = ~(oa >= q.min_oa)
    if q.min_oau > 0:
        oau = pd.to_numeric(df["oau"], errors="coerce").to_numpy()
        built = df["class"].isin(BUILT_CLASSES).to_numpy()
        reasons["min_oau"] = built & ~(oau >= q.min_oau)
    if q.min_class_f1 > 0:
        reasons["min_class_f1"] = ~(_class_f1(df) >= q.min_class_f1)

    # Polygon-level.
    area = df["area_km2"].to_numpy(dtype="float64")
    reasons["min_area"] = ~(area >= q.min_area_km2)
    reasons["max_area"] = ~(area <= q.max_area_km2)
    if "vertices" in df.columns:
        v = pd.to_numeric(df["vertices"], errors="coerce").to_numpy()
        reasons["min_vertices"] = ~(v >= q.min_vertices)

    drop = np.zeros(len(df), dtype=bool)
    for name, mask in reasons.items():
        mask = np.asarray(mask, dtype=bool)
        logger.debug(f"gate {name}: drops {int(mask.sum()):,}")
        drop |= mask

    out = df.loc[~drop].reset_index(drop=True)
    logger.info(
        f"quality gates: {n0:,} -> {len(out):,} polygons "
        f"({100 * len(out) / max(n0, 1):.1f}% retained); "
        + ", ".join(f"{k}={int(np.asarray(v).sum()):,}" for k, v in reasons.items())
    )
    return out


def _trapezoid(x: np.ndarray, lo_off: float, lo_on: float, hi_on: float, hi_off: float,
               low: float, high: float) -> np.ndarray:
    """Ramp `low`->1 across [lo_off, lo_on], flat to hi_on, 1->`high` to hi_off."""
    out = np.ones_like(x, dtype="float64")
    rise = (x < lo_on)
    out[rise] = low + (1.0 - low) * np.clip((x[rise] - lo_off) / (lo_on - lo_off), 0.0, 1.0)
    fall = (x > hi_on)
    out[fall] = 1.0 - (1.0 - high) * np.clip((x[fall] - hi_on) / (hi_off - hi_on), 0.0, 1.0)
    return out


def polygon_weights(df: pd.DataFrame, config: WudaptConfig, *,
                    target_year: int | None = None) -> pd.DataFrame:
    """Per-polygon vote weight, plus its factors for auditing.

    ``w = w_qc * w_acc * w_time * w_size``. Factors are returned alongside the
    product so the audit can show which one is actually doing the work — if none
    of them predicts So2Sat agreement (gate G0.4) the weight model degenerates
    to uniform and that must be visible, not buried in a single number.

    ``target_year`` is the embedding epoch the labels will be paired with; when
    ``None`` the time factor is 1 everywhere (used for epoch-free audits).
    """
    c = config.consensus
    n = len(df)

    # QC. A failed step 1 is fatal (weight 0); steps 2 and 3 are soft.
    step1 = df["qc_step1"].fillna(False).to_numpy(dtype=bool)
    soft = (~df["qc_step2"].fillna(True).to_numpy(dtype=bool)) | (
        ~df["qc_step3"].fillna(True).to_numpy(dtype=bool)
    )
    w_qc = np.where(step1, 1.0, 0.0) * np.where(soft, c.qc_fail_soft_penalty, 1.0)

    # Accuracy: headline (class-appropriate) scaled by class-specific F1 relative
    # to the median submission for that class, so "good at this class" is scored
    # against peers rather than against an absolute that varies by class rarity.
    acc = submission_accuracy(df)
    w_acc = np.clip((acc - c.acc_oa_min) / c.acc_oa_span, c.acc_floor, 1.0)
    f1 = _class_f1(df)
    med = pd.Series(f1).groupby(df["class"].to_numpy()).transform("median").to_numpy()
    ratio = np.divide(f1, med, out=np.ones(n), where=np.isfinite(f1) & (med > 0))
    w_acc = w_acc * np.clip(np.nan_to_num(ratio, nan=1.0), 0.25, 2.0)

    # Recency relative to the embedding epoch.
    if target_year is None:
        w_time = np.ones(n)
    else:
        ly = pd.to_numeric(df["label_year"], errors="coerce").to_numpy(dtype="float64")
        gap = np.abs(np.nan_to_num(ly, nan=target_year) - float(target_year))
        w_time = np.exp(-gap / c.time_decay_years)

    # Size. Tiny polygons carry no LCZ information; oversized ones are sloppy
    # boxes rather than delineated zones.
    area = df["area_km2"].to_numpy(dtype="float64")
    w_size = _trapezoid(area, 0.005, 0.02, 1.0, 5.0, 0.4, 0.6)

    w = w_qc * w_acc * w_time * w_size
    out = pd.DataFrame(
        {"w_qc": w_qc, "w_acc": w_acc, "w_time": w_time, "w_size": w_size, "weight": w},
        index=df.index,
    )
    logger.info(
        f"weights: mean {w.mean():.3f}, zero-weight {int((w <= 0).sum()):,}/{n:,} "
        f"| factor means qc={w_qc.mean():.2f} acc={w_acc.mean():.2f} "
        f"time={w_time.mean():.2f} size={w_size.mean():.2f}"
    )
    return out
