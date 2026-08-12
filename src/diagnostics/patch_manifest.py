"""Task 2.0 — the common patch manifest, and an audit of what it removes.

The three PLAN-v3 families do not train or evaluate on the same patches. On the
training split `alpha_earth_coop` has 352,366 (100%), `seamless` 351,719
(99.82%) and `tesserav1.1_global` 342,944 (97.33%); Tessera also evaluates on
330 fewer test patches. A cross-family table built on that compares numbers
computed on different data, and no nodata threshold touches it — the patches are
simply not there.

This script freezes one manifest of patches all three families hold and can use,
so Task 2.2's Arm A is a controlled comparison. Membership requires, for **every**
family:

1. present and readable on disk
2. ``invalid_frac <= 0.25`` (the GATE 1.75 recommendation)
3. ``native_frac >= 0.5``

**The native-fraction rule is relative on purpose.** ``seamless`` is 30 m data
and its native crop is legitimately 11-13 px where the 10 m families sit near 33;
an absolute pixel threshold would erase ESD entirely. The denominator is each
family's own median native crop area on the training split, frozen into the
sidecar so the rule is reproducible rather than recomputed per run.

**Applying an intersection deliberately is fine. Applying one without reporting
what it removes is the GATE 1.5 mistake with better intentions** — there, pairing
across families acted as an accidental nodata filter at an 855x density ratio and
nobody noticed until Phase 1.75 measured the unpaired population. So the audit is
not an optional extra here: it is the reason the manifest is trustworthy.

Measure-and-freeze. No patch data is read — Task 1.75.1's parquet already carries
``h``, ``w`` and ``invalid_frac`` for all 1,191,379 (family, split, patch)
combinations.

Example (the Task 2.0 run):

    python src/diagnostics/patch_manifest.py \\
        --invalid-frac-parquet diagnostics/invalid_fraction.parquet \\
        --output-parquet diagnostics/patch_manifest_v1.parquet \\
        --output-json diagnostics/patch_manifest_v1.json \\
        --output-md diagnostics/patch_manifest.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.nodata_population import (                        # noqa: E402
    HELDOUT_CITIES,
    SCAN_FAMILIES,
    SPLITS,
    TRUNCATION_RATIO,
)
from utils.constants import DATA_DIR                               # noqa: E402

# Membership thresholds. Both are GATE 1.75 / PLAN-v3 Rev A decisions, spelled
# out here so the manifest sidecar can record what it was built with.
MAX_INVALID_FRAC = 0.25
MIN_NATIVE_FRAC = 0.5

# The split that defines each family's expected native crop area. Training is
# the largest and most geographically diverse, and using one split for all three
# keeps the denominator from drifting between them.
DENOMINATOR_SPLIT = "training"

# Cities where the two failure modes concentrate (Rev A asks for these by name):
# the first four are AlphaEarth's nodata cities, the last four Tessera's
# truncation cities.
WATCH_CITIES = [
    "Cape Town", "Lisbon", "Mumbai", "New York",
    "London", "Guangzhou", "Melbourne", "Shanghai",
]

# A patch counts as severely invalid at this fraction. It is the 75-100% bucket
# of the Task 1.75.1 histogram — AlphaEarth's 1,277 near-total-loss training
# patches, the population the intersection is checked against below.
SEVERE_INVALID_FRAC = 0.75


# ── Manifest construction ────────────────────────────────────────────────────

def native_frac(df: pd.DataFrame, denominators: dict[str, float]) -> pd.Series:
    """Native crop area as a fraction of the family's own expected area.

    Relative, never absolute: `seamless` crops to ~12 px per side at 30 m and the
    10 m families to ~33, so a shared pixel threshold would call every ESD patch
    truncated. Area rather than per-side because a crop can lose extent on either
    axis and the product is what determines how much ground actually survives.
    """
    area = df["h"].astype("int32") * df["w"].astype("int32")
    return area / df["family"].map(denominators).astype("float64")


def build_manifest(
    scan: pd.DataFrame,
    families: list[str],
    max_invalid_frac: float,
    min_native_frac: float,
) -> tuple[pd.DataFrame, dict]:
    """One row per reference patch, with per-family flags and the verdict.

    Non-members are kept in the table rather than filtered out. The loader has to
    tell "this patch was excluded" from "this patch is outside the manifest's
    universe" — an unlabeled or pseudo-labelled patch is not a member and must
    not be silently dropped — and only an explicit ``in_manifest = False`` row
    supports that distinction.
    """
    denominators = {
        f: float((g["h"].astype("int32") * g["w"].astype("int32")).median())
        for f, g in scan[scan["dataset"] == DENOMINATOR_SPLIT].groupby(
            "family", observed=True)
    }
    missing_denominator = [f for f in families if f not in denominators]
    if missing_denominator:
        logger.error(
            f"No {DENOMINATOR_SPLIT} rows for {missing_denominator} in the scan "
            "parquet — cannot derive an expected native area for them."
        )
        raise SystemExit(1)

    scan = scan.assign(native_frac=native_frac(scan, denominators))
    scan["ok"] = ((scan["invalid_frac"] <= max_invalid_frac)
                  & (scan["native_frac"] >= min_native_frac))

    # The reference universe is every patch in patches_reference_rxr.gpkg. The
    # family with complete coverage supplies it; if none is complete, the union
    # across families does, so a patch missing from all of them is still visible
    # as a row rather than vanishing from the denominator.
    universe = (
        scan[["dataset", "patch_id", "city", "lcz"]]
        .drop_duplicates(subset=["dataset", "patch_id"])
        .sort_values(["dataset", "patch_id"])
        .reset_index(drop=True)
    )

    man = universe
    for family in families:
        sub = scan[scan["family"] == family][
            ["dataset", "patch_id", "invalid_frac", "native_frac", "ok"]
        ].rename(columns={
            "invalid_frac": f"invalid_frac_{family}",
            "native_frac": f"native_frac_{family}",
            "ok": f"ok_{family}",
        })
        man = man.merge(sub, on=["dataset", "patch_id"], how="left")
        # A NaN after the merge means the family has no npy for this patch.
        man[f"present_{family}"] = man[f"ok_{family}"].notna()
        man[f"ok_{family}"] = man[f"ok_{family}"].fillna(False).astype(bool)

    man["in_manifest"] = np.logical_and.reduce(
        [man[f"present_{f}"] & man[f"ok_{f}"] for f in families]
    )

    meta = {
        "families": families,
        "max_invalid_frac": max_invalid_frac,
        "min_native_frac": min_native_frac,
        "denominator_split": DENOMINATOR_SPLIT,
        "expected_native_area": denominators,
        "n_reference": int(len(man)),
        "n_members": int(man["in_manifest"].sum()),
        "per_split": {
            s: {
                "n_reference": int((man["dataset"] == s).sum()),
                "n_members": int(man.loc[man["dataset"] == s, "in_manifest"].sum()),
            }
            for s in SPLITS
        },
    }
    return man, meta


def write_manifest(man: pd.DataFrame, path: Path) -> str:
    """Write the parquet and return its SHA-256.

    The hash goes into every run's W&B config, so it has to be the hash of the
    bytes on disk rather than of the frame in memory — that is the artefact a
    later reader would check against.
    """
    out = man.copy()
    for col in ("dataset", "city"):
        out[col] = out[col].astype("category")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, compression="zstd")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── 2.0a — native-area reporting ─────────────────────────────────────────────

def native_area_report(scan: pd.DataFrame, denominators: dict[str, float],
                       min_native_frac: float) -> dict:
    """Native-area distribution and the two truncation rules side by side.

    Task 1.75.1 counted truncation at 0.85 x the family median *per side*; the
    manifest thresholds 0.5 of the median *area*. Those disagree — a 33x17 crop
    is 0.51 of the area but 0.5 of one side — so both are reported. A rule the
    reader cannot compare against the previous one is a rule they cannot check.
    """
    out: dict[str, dict] = {}
    for family in sorted(scan["family"].unique()):
        fam = scan[scan["family"] == family]
        train = fam[fam["dataset"] == DENOMINATOR_SPLIT]
        med_h = float(train["h"].median())
        med_w = float(train["w"].median())
        area = (fam["h"].astype("int32") * fam["w"].astype("int32"))
        frac = area / denominators[family]
        per_side = (fam["h"] < TRUNCATION_RATIO * med_h) | (fam["w"] < TRUNCATION_RATIO * med_w)
        out[family] = {
            "expected_native_area": denominators[family],
            "median_shape": [med_h, med_w],
            "area_quantiles": {
                q: float(np.percentile(area, p))
                for q, p in (("min", 0), ("p1", 1), ("p50", 50), ("p99", 99), ("max", 100))
            },
            "modal_shapes": {
                f"{int(h)}x{int(w)}": int(n) for (h, w), n in
                train.groupby(["h", "w"]).size().sort_values(ascending=False).head(5).items()
            },
            "native_frac_histogram": {
                label: int(((frac >= lo) & (frac < hi)).sum())
                for label, lo, hi in (
                    ("<0.25", 0.0, 0.25), ("0.25-0.5", 0.25, 0.5),
                    ("0.5-0.75", 0.5, 0.75), ("0.75-0.95", 0.75, 0.95),
                    (">=0.95", 0.95, np.inf),
                )
            },
            "below_min_native_frac": {
                s: int(((frac < min_native_frac) & (fam["dataset"] == s)).sum())
                for s in SPLITS
            },
            "below_per_side_rule": {
                s: int((per_side & (fam["dataset"] == s)).sum()) for s in SPLITS
            },
        }
    return out


# ── 2.0c — what the intersection removes ─────────────────────────────────────

def intersection_audit(scan: pd.DataFrame, man: pd.DataFrame,
                       families: list[str]) -> dict:
    """Whether the intersection and the nodata policy are independent filters.

    GATE 1.5's lesson: an intersection can silently *be* a nodata filter. If the
    AlphaEarth patches the drop policy targets are the same patches Tessera is
    missing, then Arm A's coverage restriction has already applied the drop
    policy and the two levers cannot be reasoned about separately.
    """
    nodata_family = "alpha_earth_coop"
    out: dict = {"severe_invalid_frac": SEVERE_INVALID_FRAC, "by_split": {}}
    for split in SPLITS:
        base = scan[(scan["family"] == nodata_family) & (scan["dataset"] == split)]
        severe = base[base["invalid_frac"] > SEVERE_INVALID_FRAC]
        row: dict = {
            "n_severe": int(len(severe)),
            "n_above_max_invalid": int((base["invalid_frac"] > MAX_INVALID_FRAC).sum()),
            "others": {},
        }
        for other in families:
            if other == nodata_family:
                continue
            present = set(
                scan.loc[(scan["family"] == other) & (scan["dataset"] == split),
                         "patch_id"]
            )
            absent_severe = int((~severe["patch_id"].isin(present)).sum())
            base_rate = float((~base["patch_id"].isin(present)).mean())
            row["others"][other] = {
                "severe_also_absent": absent_severe,
                "severe_also_absent_frac": (
                    absent_severe / len(severe) if len(severe) else 0.0),
                "base_absence_rate": base_rate,
            }
        # How much work is left for the drop policy once the intersection has
        # run. Asking how many high-invalid patches are in the finished manifest
        # would be tautological — membership already requires passing the
        # threshold. The question is how many survive a *coverage-only*
        # intersection, because that is the count the nodata criterion is still
        # doing independent work on.
        sub = man[man["dataset"] == split]
        covered = np.logical_and.reduce([sub[f"present_{f}"] for f in families])
        high = sub[f"invalid_frac_{nodata_family}"] > MAX_INVALID_FRAC
        row["n_above_max_invalid_after_coverage_only"] = int((covered & high).sum())
        out["by_split"][split] = row
    return out


def _loss_table(man: pd.DataFrame, key: str, split: str) -> pd.DataFrame:
    sub = man[man["dataset"] == split]
    g = sub.groupby(key, observed=True)["in_manifest"].agg(["size", "sum"])
    g["lost"] = g["size"] - g["sum"]
    g["loss_pct"] = 100.0 * g["lost"] / g["size"]
    return g.sort_values("loss_pct", ascending=False).reset_index()


def composition_audit(man: pd.DataFrame, families: list[str]) -> dict:
    """Per-city and per-LCZ composition against the full reference population."""
    out: dict = {"by_split": {}}
    for split in SPLITS:
        by_city = _loss_table(man, "city", split)
        by_city["held_out"] = by_city["city"].isin(HELDOUT_CITIES)
        by_lcz = _loss_table(man, "lcz", split)

        sub = man[man["dataset"] == split]
        before = sub.groupby("lcz", observed=True).size() / max(len(sub), 1)
        kept = sub[sub["in_manifest"]]
        after = kept.groupby("lcz", observed=True).size() / max(len(kept), 1)
        shift = pd.DataFrame({"before": before, "after": after}).fillna(0.0)
        shift["delta_pp"] = 100.0 * (shift["after"] - shift["before"])

        # Which family's failure is responsible for each exclusion, so the city
        # numbers can be read as "coverage" or "nodata" rather than just "loss".
        lost = sub[~sub["in_manifest"]]
        blame = {
            f: {
                "absent": int((~lost[f"present_{f}"]).sum()),
                "failed_criteria": int((lost[f"present_{f}"] & ~lost[f"ok_{f}"]).sum()),
            }
            for f in families
        }

        out["by_split"][split] = {
            "n_reference": int(len(sub)),
            "n_members": int(sub["in_manifest"].sum()),
            "cities_over_5pct": by_city[by_city["loss_pct"] > 5.0].to_dict("records"),
            "by_city": by_city.to_dict("records"),
            "classes_over_5pct": by_lcz[by_lcz["loss_pct"] > 5.0].to_dict("records"),
            "by_lcz": by_lcz.to_dict("records"),
            "class_share_shift": [
                {"lcz": float(k), "before_pct": 100.0 * r.before,
                 "after_pct": 100.0 * r.after, "delta_pp": r.delta_pp}
                for k, r in shift.sort_values("delta_pp").iterrows()
            ],
            "lost_by_family": blame,
            "lost_lcz_counts": {
                str(int(k)): int(v)
                for k, v in lost["lcz"].value_counts().sort_index().items()
            },
            "watch_cities": [
                r for r in by_city.to_dict("records") if r["city"] in WATCH_CITIES
            ],
        }
    return out


def family_loss(scan: pd.DataFrame, man: pd.DataFrame,
                families: list[str]) -> dict:
    """Manifest size against each family's own filtered native coverage.

    The relevant cost of the manifest is not "how much of the reference set is
    gone" but "how much does *this* family give up relative to what it could have
    used on its own" — that is the number Arm B keeps and Arm A pays.
    """
    out: dict = {}
    for split in SPLITS:
        members = int(man.loc[man["dataset"] == split, "in_manifest"].sum())
        per_family = {}
        for f in families:
            fam = scan[(scan["family"] == f) & (scan["dataset"] == split)]
            on_disk = int(len(fam))
            own = int(((fam["invalid_frac"] <= MAX_INVALID_FRAC)
                       & (fam["native_frac"] >= MIN_NATIVE_FRAC)).sum())
            per_family[f] = {
                "on_disk": on_disk,
                "own_filtered": own,
                "manifest": members,
                "loss_vs_own_filtered": (own - members) / own if own else 0.0,
            }
        out[split] = per_family
    return out


# ── Reporting ────────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def markdown_report(payload: dict) -> str:
    meta = payload["meta"]
    man_meta = payload["manifest"]
    families = man_meta["families"]

    lines = [
        "### Task 2.0 — The common patch manifest",
        "",
        f"Built from `{meta['source_parquet']}` "
        f"(sha256 `{meta['source_sha256'][:16]}…`), no patch data read. "
        f"Generated {meta['generated']}.",
        "",
        f"Membership requires, for **all** of {', '.join(f'`{f}`' for f in families)}: "
        f"present on disk, `invalid_frac <= {man_meta['max_invalid_frac']}`, and "
        f"`native_frac >= {man_meta['min_native_frac']}`.",
        "",
        f"**Manifest** `{meta['manifest_path']}` — sha256 "
        f"`{man_meta['sha256']}`",
        "",
        "| split | reference | manifest | retained |",
        "|---|---|---|---|",
    ]
    for split, s in man_meta["per_split"].items():
        lines.append(
            f"| {split} | {s['n_reference']:,} | {s['n_members']:,} | "
            f"{s['n_members'] / max(s['n_reference'], 1):.2%} |"
        )

    # ── 2.0a ────────────────────────────────────────────────────────────────
    lines += [
        "",
        "#### 2.0a — Native crop area, and why the filter is relative",
        "",
        "| family | expected native area | median shape | modal shape | "
        "area min | area p1 | area max |",
        "|---|---|---|---|---|---|---|",
    ]
    for family, r in payload["native_area"].items():
        modal = next(iter(r["modal_shapes"]), "-")
        q = r["area_quantiles"]
        lines.append(
            f"| `{family}` | {r['expected_native_area']:.0f} | "
            f"{r['median_shape'][0]:.0f}x{r['median_shape'][1]:.0f} | {modal} | "
            f"{q['min']:.0f} | {q['p1']:.0f} | {q['max']:.0f} |"
        )
    lines += [
        "",
        f"`seamless` is 30 m data and crops natively to ~12 px per side against "
        f"~33 for the 10 m families, so an absolute pixel threshold would erase it "
        f"entirely. The denominator is each family's own median native crop area "
        f"on the `{man_meta['denominator_split']}` split.",
        "",
        "| family | native_frac <0.25 | 0.25-0.5 | 0.5-0.75 | 0.75-0.95 | >=0.95 |",
        "|---|---|---|---|---|---|",
    ]
    for family, r in payload["native_area"].items():
        h = r["native_frac_histogram"]
        lines.append(
            f"| `{family}` | " + " | ".join(f"{h[k]:,}" for k in
            ("<0.25", "0.25-0.5", "0.5-0.75", "0.75-0.95", ">=0.95")) + " |"
        )

    lines += [
        "",
        "The `seamless` mass in the 0.75-0.95 bucket is not damage: an 11x12 crop "
        "is 0.92 of a 12x12 median, so ordinary +/-1 px reprojection drift moves a "
        "large share of ESD patches a full bucket that the same drift barely "
        "registers for a 33 px crop. Only the `<0.5` buckets are truncation.",
        "",
        f"**Two truncation rules, side by side.** Task 1.75.1 counted a crop "
        f"truncated below {TRUNCATION_RATIO:.2f} x the family median *per side*; "
        f"the manifest thresholds {man_meta['min_native_frac']} of the median "
        f"*area*. They disagree — a 33x17 crop is 0.52 of the area but 0.5 of one "
        f"side — so the area rule is the more permissive of the two and the gap "
        f"is reported rather than left implicit.",
        "",
        "| family | split | below native_frac | below per-side rule |",
        "|---|---|---|---|",
    ]
    for family, r in payload["native_area"].items():
        for split in SPLITS:
            a, b = r["below_min_native_frac"][split], r["below_per_side_rule"][split]
            if a or b:
                lines.append(f"| `{family}` | {split} | {a:,} | {b:,} |")

    # ── 2.0b ────────────────────────────────────────────────────────────────
    lines += [
        "",
        "#### 2.0b — What the manifest costs each family",
        "",
        "`own_filtered` = the patches that family has on disk and that pass both "
        "criteria on its own — what Arm B would use. `loss` is what Arm A gives up "
        "relative to that.",
        "",
        "| split | family | on disk | own filtered | manifest | loss vs own |",
        "|---|---|---|---|---|---|",
    ]
    for split, per_family in payload["family_loss"].items():
        for family, r in per_family.items():
            lines.append(
                f"| {split} | `{family}` | {r['on_disk']:,} | {r['own_filtered']:,} | "
                f"{r['manifest']:,} | {_pct(r['loss_vs_own_filtered'])} |"
            )

    # ── 2.0c ────────────────────────────────────────────────────────────────
    audit = payload["intersection"]
    lines += [
        "",
        "#### 2.0c — What the intersection is silently doing",
        "",
        "**The intersection and the nodata policy are not independent filters.** "
        f"Of `alpha_earth_coop`'s severely invalid patches (>{SEVERE_INVALID_FRAC:.0%} "
        "of pixels), this many are *also* absent from each other family — against "
        "that family's base rate of absence over the whole split:",
        "",
        "| split | severe patches | other family | also absent | base absence rate |",
        "|---|---|---|---|---|",
    ]
    for split, r in audit["by_split"].items():
        for other, o in r["others"].items():
            lines.append(
                f"| {split} | {r['n_severe']:,} | `{other}` | "
                f"{o['severe_also_absent']:,} ({o['severe_also_absent_frac']:.1%}) | "
                f"{o['base_absence_rate']:.2%} |"
            )
    lines += [
        "",
        "So the coverage intersection already performs most of the drop policy. "
        "How much independent work the nodata criterion is left with — the count "
        "of high-invalid patches that survive a **coverage-only** intersection:",
        "",
        "| split | coop patches >max_invalid | survive coverage-only intersection |",
        "|---|---|---|",
    ]
    for split, r in audit["by_split"].items():
        lines.append(
            f"| {split} | {r['n_above_max_invalid']:,} | "
            f"{r['n_above_max_invalid_after_coverage_only']:,} |"
        )

    comp = payload["composition"]["by_split"]
    for split in SPLITS:
        c = comp[split]
        lines += [
            "",
            f"##### {split} — composition against the full reference population",
            "",
            f"{c['n_members']:,} of {c['n_reference']:,} retained. Attribution of "
            "the exclusions:",
            "",
            "| family | absent from disk | present but failed a criterion |",
            "|---|---|---|",
        ]
        for family, b in c["lost_by_family"].items():
            lines.append(f"| `{family}` | {b['absent']:,} | {b['failed_criteria']:,} |")

        over = c["cities_over_5pct"]
        lines += [
            "",
            f"**Cities losing more than 5%** ({len(over)}). **Bold** = one of the "
            "10 held-out cultural-split cities.",
            "",
        ]
        if over:
            lines += ["| city | held out | n | retained | loss |", "|---|---|---|---|---|"]
            for r in over:
                name = f"**{r['city']}**" if r["held_out"] else r["city"]
                lines.append(
                    f"| {name} | {'yes' if r['held_out'] else ''} | {r['size']:,} | "
                    f"{r['sum']:,} | {r['loss_pct']:.2f}% |"
                )
        else:
            lines.append("None.")

        over_c = c["classes_over_5pct"]
        lines += [
            "",
            f"**LCZ classes losing more than 5%** ({len(over_c)}).",
            "",
        ]
        if over_c:
            lines += ["| LCZ | n | retained | loss |", "|---|---|---|---|"]
            for r in over_c:
                lines.append(
                    f"| {int(r['lcz'])} | {r['size']:,} | {r['sum']:,} | "
                    f"{r['loss_pct']:.2f}% |"
                )
        else:
            lines.append("None.")

        lost = c["lost_lcz_counts"]
        if lost:
            total = sum(lost.values())
            top = sorted(lost.items(), key=lambda kv: -kv[1])[:3]
            share = ", ".join(f"LCZ {k} {v:,} ({v / total:.1%})" for k, v in top)
            lines += ["", f"Excluded patches by class: {share}."]

        lines += [
            "",
            "Watch cities (Rev A):",
            "",
            "| city | n | retained | loss |",
            "|---|---|---|---|",
        ]
        for r in c["watch_cities"]:
            lines.append(
                f"| {r['city']} | {r['size']:,} | {r['sum']:,} | {r['loss_pct']:.2f}% |"
            )

    lines += [
        "",
        "##### Class-share shift (training)",
        "",
        "| LCZ | before | after | delta |",
        "|---|---|---|---|",
    ]
    for r in comp["training"]["class_share_shift"]:
        lines.append(
            f"| {int(r['lcz'])} | {r['before_pct']:.2f}% | {r['after_pct']:.2f}% | "
            f"{r['delta_pp']:+.2f} pp |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 2.0 — build and freeze the common patch manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--invalid-frac-parquet", type=Path,
                   default=Path("diagnostics/invalid_fraction.parquet"),
                   help="Task 1.75.1 output: per-patch invalid_frac, h, w, city, lcz.")
    p.add_argument("--families", nargs="+", default=SCAN_FAMILIES,
                   help="Families that must all hold a patch for it to be a member.")
    p.add_argument("--max-invalid-frac", type=float, default=MAX_INVALID_FRAC)
    p.add_argument("--min-native-frac", type=float, default=MIN_NATIVE_FRAC)
    p.add_argument("--output-parquet", type=Path,
                   default=Path("diagnostics/patch_manifest_v1.parquet"))
    p.add_argument("--output-json", type=Path,
                   default=Path("diagnostics/patch_manifest_v1.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/patch_manifest.md"))
    args = p.parse_args()

    if not args.invalid_frac_parquet.exists():
        logger.error(
            f"{args.invalid_frac_parquet} not found. Write it with:\n"
            "    python src/diagnostics/nodata_population.py"
        )
        raise SystemExit(1)

    scan = pd.read_parquet(args.invalid_frac_parquet)
    scan["family"] = scan["family"].astype(str)
    scan["dataset"] = scan["dataset"].astype(str)
    known = set(scan["family"].unique())
    unknown = [f for f in args.families if f not in known]
    if unknown:
        logger.error(f"{args.invalid_frac_parquet.name} has no rows for {unknown}; "
                     f"it covers {sorted(known)}.")
        raise SystemExit(1)
    scan = scan[scan["family"].isin(args.families)].reset_index(drop=True)
    logger.info(f"Scan rows: {len(scan):,} over {len(args.families)} families")

    man, man_meta = build_manifest(
        scan, args.families, args.max_invalid_frac, args.min_native_frac
    )
    scan = scan.assign(
        native_frac=native_frac(scan, man_meta["expected_native_area"])
    )

    sha = write_manifest(man, args.output_parquet)
    man_meta["sha256"] = sha
    logger.info(
        f"Manifest: {man_meta['n_members']:,} of {man_meta['n_reference']:,} "
        f"patches → {args.output_parquet} (sha256 {sha[:16]}…)"
    )
    for split, s in man_meta["per_split"].items():
        logger.info(f"  {split}: {s['n_members']:,} / {s['n_reference']:,}")

    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_parquet": str(args.invalid_frac_parquet),
            "source_sha256": hashlib.sha256(
                args.invalid_frac_parquet.read_bytes()).hexdigest(),
            "manifest_path": str(args.output_parquet),
            "data_dir": str(DATA_DIR),
        },
        "manifest": man_meta,
        "native_area": native_area_report(
            scan, man_meta["expected_native_area"], args.min_native_frac),
        "family_loss": family_loss(scan, man, args.families),
        "intersection": intersection_audit(scan, man, args.families),
        "composition": composition_audit(man, args.families),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # default=str: the per-city and per-class rows come straight out of pandas
    # and carry numpy scalars, which json cannot serialise on its own.
    args.output_json.write_text(json.dumps(payload, indent=2, default=str))
    args.output_md.write_text(markdown_report(payload))
    logger.info(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
