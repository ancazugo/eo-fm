"""Task 2.0d — what a corrected Tessera re-extraction would recover. Measure only.

`tesserav1.1_global` is missing 9,422 training and 330 test patches, and 560 of
the training patches it does have came back truncated. Commit a958a3f fixed the
cause: `_fully_covered` tested a patch against the *bounding box* of each tile's
WGS84 footprint, and a UTM rectangle's bounding box over-claims 0.15% of its area
at the equator rising to 10.7% at 78°N. Patches falling in that sliver were
reported as covered and cropped short; patches just outside a tile were harder to
judge either way.

This replays the fixed coverage test over exactly those patches and counts how
many a re-extraction would **recover** (the exact footprint says fully covered,
so the crop would come back whole) against how many it would **correctly
reject** (genuinely not covered — the data does not exist and never did).

**Nothing is re-extracted, and nothing outside `diagnostics/` is written.** Task
2.1's anchor is tied to the current extraction and re-extracting would break
comparability with every existing Tessera number; per PLAN-v3 Rev A the decision
is deferred to GATE 3, where it costs one anchor re-run rather than the phase.
The point of this script is to make that decision on a number instead of a guess.

Example (the Task 2.0 run):

    python src/diagnostics/reextraction_scope.py \\
        --output-json diagnostics/reextraction_scope.json \\
        --output-md diagnostics/reextraction_scope.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.nodata_population import (                        # noqa: E402
    HELDOUT_CITIES,
    SPLITS,
    load_patch_metadata,
    truncation_mask,
)
from utils.constants import DATA_DIR                               # noqa: E402

FAMILY = "tesserav1.1_global"

# The one tile the reader skips with a warning; patches that depend on it cannot
# be recovered by a coverage fix, so they are called out rather than counted as
# recoverable.
CORRUPT_TILES = {"grid_121.35_31.25"}


def classify_patches(
    gdf,
    tile_paths: list[Path],
    tree,
) -> pd.DataFrame:
    """Run the fixed coverage test over patch geometries.

    This is `extract_so2sat_embeddings._fully_covered` verbatim — imported rather
    than reimplemented, so what is measured here is what a re-extraction would
    actually do.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from extract_so2sat_embeddings import _fully_covered

    from datasets.tiles import tessera_grid_geometry, tile_index_name

    # Not p.stem: build_tile_index returns the NPY *directory* and Path reads the
    # fractional latitude's ".25" as a suffix, so stem returns grid_121.35_31 and
    # the CORRUPT_TILES membership test below could never match. See
    # tiles.tile_index_name.
    names = [tile_index_name(p) for p in tile_paths]

    rows = []
    for row in tqdm(gdf.itertuples(index=False), total=len(gdf), unit="patch",
                    desc="coverage test"):
        idxs = tree.query(row.geometry)
        if len(idxs) == 0:
            rows.append((row.patch_id, row.dataset, 0, False, False, 1))
            continue
        covered = bool(_fully_covered(row.geometry, tree.geometries[idxs]))
        touches_corrupt = any(names[i] in CORRUPT_TILES for i in idxs)
        # How many UTM zones the matched tiles span. A patch crossing a zone
        # seam goes through crop_patch's multi-tile mosaic, which is a separate
        # failure path from coverage and has to be told apart from it.
        zones = {str(tessera_grid_geometry(names[i])[0]) for i in idxs}
        rows.append((row.patch_id, row.dataset, len(idxs), covered,
                     touches_corrupt, len(zones)))

    return pd.DataFrame(
        rows, columns=["patch_id", "dataset", "n_tiles", "fully_covered",
                       "touches_corrupt_tile", "n_utm_zones"]
    )


def summarize(df: pd.DataFrame, label: str) -> dict:
    """Recovery counts for one category of patch, with the caveats attached."""
    recoverable = df["fully_covered"] & ~df["touches_corrupt_tile"]
    return {
        "category": label,
        "n": int(len(df)),
        "n_recoverable": int(recoverable.sum()),
        "n_correctly_rejected": int((~df["fully_covered"]).sum()),
        "n_blocked_by_corrupt_tile": int(
            (df["fully_covered"] & df["touches_corrupt_tile"]).sum()),
        "n_no_tile_at_all": int((df["n_tiles"] == 0).sum()),
        "frac_recoverable": float(recoverable.mean()) if len(df) else 0.0,
    }


def by_group(df: pd.DataFrame, key: str) -> list[dict]:
    df = df.assign(recoverable=df["fully_covered"] & ~df["touches_corrupt_tile"])
    g = (df.groupby(key, observed=True)
           .agg(n=("patch_id", "size"), recoverable=("recoverable", "sum"))
           .reset_index())
    g["frac"] = g["recoverable"] / g["n"]
    return g.sort_values("recoverable", ascending=False).to_dict("records")


# ── Reporting ────────────────────────────────────────────────────────────────

def markdown_report(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        "### Task 2.0d — Re-extraction scoping (measure only)",
        "",
        f"The fixed `_fully_covered` from a958a3f replayed over "
        f"`{FAMILY}`'s absent and truncated patches, against a tile index of "
        f"{meta['n_tiles']:,} exact footprints. Generated {meta['generated']}.",
        "",
        "**Nothing was re-extracted.** Task 2.1's anchor is tied to the current "
        "extraction; per PLAN-v3 Rev A the decision is deferred to GATE 3 with "
        "these counts on the record.",
        "",
        "Patches **absent** from disk — how many a re-extraction would bring back. "
        "`no tile at all` means the STRtree returns no candidate whatsoever: there "
        "is no Tessera tile over that ground, so no coverage fix can help.",
        "",
        "| split | absent | would recover | correctly rejected | no tile at all |",
        "|---|---|---|---|---|",
    ]
    for split, cats in payload["by_split"].items():
        r = cats.get("absent")
        if r is None:
            continue
        lines.append(
            f"| {split} | {r['n']:,} | {r['n_recoverable']:,} "
            f"({r['frac_recoverable']:.1%}) | {r['n_correctly_rejected']:,} | "
            f"{r['n_no_tile_at_all']:,} |"
        )

    blocked = sum(
        r["n_blocked_by_corrupt_tile"]
        for cats in payload["by_split"].values() for r in cats.values()
    )
    lines += [
        "",
        f"`would recover` excludes patches whose only coverage comes from the "
        f"corrupt tile{'s' if len(CORRUPT_TILES) > 1 else ''} "
        f"{', '.join(f'`{t}`' for t in sorted(CORRUPT_TILES))} — {blocked:,} "
        f"patches, which a coverage fix cannot help.",
        "",
        "#### Truncated patches — does the fix reject them?",
        "",
        "This is the direct test of the bug, and the honest framing for these "
        "patches is rejection rather than recovery: one that stays accepted would "
        "be re-extracted and come back truncated again. A truncated crop is one "
        "the old bounding-box test wrongly accepted, so the exact footprint "
        "should now reject it.",
        "",
        "| split | truncated | now rejected | still accepted |",
        "|---|---|---|---|",
    ]
    for split, r in payload["truncation_check"].items():
        lines.append(
            f"| {split} | {r['n']:,} | {r['n_rejected']:,} "
            f"({r['frac_rejected']:.1%}) | {r['n_still_accepted']:,} |"
        )
    still = sum(r["n_still_accepted"] for r in payload["truncation_check"].values())
    z = payload.get("still_accepted_cause")
    if still and z:
        lines += [
            "",
            f"**{still:,} truncated patches are still accepted by the exact "
            f"footprint, and the coverage test is right about them** — "
            f"{z['multi_utm_zone']:,} of {z['n']:,} ({z['multi_utm_zone'] / z['n']:.1%}) "
            "straddle a **UTM zone boundary**, and every one spans more than one "
            "tile. The ground genuinely is covered; the truncation happens later, "
            "in the multi-tile mosaic, which is the known multi-UTM-zone "
            "`crop_patch` defect rather than anything the footprint fix touches. "
            "Their shapes bear that out — full extent on one axis and 5-8 px on "
            "the other, cut at the zone seam.",
            "",
            "The cities are exactly the ones a zone seam predicts: " +
            ", ".join(f"{c} ({n})" for c, n in list(z["by_city"].items())[:5]) +
            " — the prime meridian, 114°E and 120°E.",
            "",
            "**So a re-extraction on today's code would reproduce these 302, not "
            "fix them.** Removing them needs the mosaic defect fixed as well, "
            "which is a second change and a second re-run. That belongs in the "
            "GATE 3 decision alongside the recovery count.",
        ]

    lines += [
        "",
        "#### Where the recoverable patches are (all splits, absent patches)",
        "",
        "**Bold** = one of the 10 held-out cultural-split cities.",
        "",
        "| city | absent | would recover | share |",
        "|---|---|---|---|",
    ]
    for r in payload["by_city"][:20]:
        if not r["n"]:
            continue
        name = f"**{r['city']}**" if r["city"] in HELDOUT_CITIES else r["city"]
        lines.append(
            f"| {name} | {r['n']:,} | {int(r['recoverable']):,} | {r['frac']:.1%} |"
        )

    lines += [
        "",
        "#### By LCZ class",
        "",
        "| LCZ | absent | would recover | share |",
        "|---|---|---|---|",
    ]
    for r in payload["by_lcz"][:10]:
        if not r["n"]:
            continue
        lines.append(
            f"| {int(r['lcz'])} | {r['n']:,} | {int(r['recoverable']):,} | "
            f"{r['frac']:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 2.0d — re-extraction recovery scope. Measure only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--embedding-dir", type=Path, default=Path("/tessera/v1.1"))
    p.add_argument("--year", default="2017")
    p.add_argument("--bounds-csv", type=Path,
                   default=Path(__file__).resolve().parents[2] / "data"
                   / "so2sat_guppd_bounds.csv")
    p.add_argument("--invalid-frac-parquet", type=Path,
                   default=Path("diagnostics/invalid_fraction.parquet"))
    p.add_argument("--output-json", type=Path,
                   default=Path("diagnostics/reextraction_scope.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/reextraction_scope.md"))
    args = p.parse_args()

    import geopandas as gpd
    from datasets.tiles import build_tile_index

    # Which patches are absent, and which are truncated, from the Task 1.75.1 scan
    scan = pd.read_parquet(args.invalid_frac_parquet)
    scan = scan[scan["family"].astype(str) == FAMILY]
    scan["dataset"] = scan["dataset"].astype(str)

    truncated_keys: set[tuple[str, str]] = set()
    trunc_counts: dict[str, int] = {}
    for split in SPLITS:
        sub = scan[scan["dataset"] == split]
        mask, _, _ = truncation_mask(sub)
        keys = set(zip(sub.loc[mask, "dataset"], sub.loc[mask, "patch_id"]))
        truncated_keys |= keys
        trunc_counts[split] = len(keys)
    present_keys = set(zip(scan["dataset"], scan["patch_id"]))
    logger.info(f"Scan: {len(present_keys):,} on disk, "
                f"{len(truncated_keys):,} truncated")

    # The reference population, with geometry — the coverage test needs polygons
    gdf = gpd.read_file(args.so2sat_dir / "patches_reference_rxr.gpkg")
    gdf["patch_id"] = gdf["patch_id"].astype(str).str.zfill(6)
    gdf["dataset"] = gdf["dataset"].astype(str)
    keys = list(zip(gdf["dataset"], gdf["patch_id"]))
    gdf["absent"] = [k not in present_keys for k in keys]
    gdf["truncated"] = [k in truncated_keys for k in keys]

    target = gdf[gdf["absent"] | gdf["truncated"]].reset_index(drop=True)
    logger.info(
        f"Target: {len(target):,} patches "
        f"({int(gdf['absent'].sum()):,} absent, {int(gdf['truncated'].sum()):,} truncated)"
    )

    tile_paths, tree = build_tile_index(args.embedding_dir, FAMILY, year=args.year)
    logger.info(f"Tile index: {len(tile_paths):,} exact footprints")

    result = classify_patches(target, tile_paths, tree)
    target = target.merge(result, on=["patch_id", "dataset"])

    meta_df, meta_info = load_patch_metadata(args.so2sat_dir, args.bounds_csv)
    target = target.merge(
        meta_df[["patch_id", "dataset", "city", "LCZ_class"]].rename(
            columns={"LCZ_class": "lcz"}),
        on=["patch_id", "dataset"], how="left",
    )

    by_split: dict[str, dict] = {}
    truncation_check: dict[str, dict] = {}
    for split in SPLITS:
        sub = target[target["dataset"] == split]
        cats = {}
        absent = sub[sub["absent"]]
        trunc = sub[sub["truncated"]]
        if len(absent):
            cats["absent"] = summarize(absent, "absent")
        if len(trunc):
            cats["truncated"] = summarize(trunc, "truncated")
            truncation_check[split] = {
                "n": int(len(trunc)),
                "n_rejected": int((~trunc["fully_covered"]).sum()),
                "n_still_accepted": int(trunc["fully_covered"].sum()),
                "frac_rejected": float((~trunc["fully_covered"]).mean()),
            }
        if cats:
            by_split[split] = cats

    absent_all = target[target["absent"]]

    # Why the exact footprint still accepts some truncated patches. If it is
    # zone seams, the coverage test is right and the defect is downstream in the
    # mosaic — which means a re-extraction would reproduce them rather than fix
    # them, and that changes what the GATE 3 decision is choosing between.
    still = target[target["truncated"] & target["fully_covered"]]
    still_cause = None
    if len(still):
        still_cause = {
            "n": int(len(still)),
            "multi_utm_zone": int((still["n_utm_zones"] > 1).sum()),
            "multi_tile": int((still["n_tiles"] > 1).sum()),
            "by_city": {
                str(c): int(n) for c, n in
                still["city"].value_counts().head(8).items()
            },
        }

    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "family": FAMILY,
            "year": args.year,
            "embedding_dir": str(args.embedding_dir),
            "n_tiles": len(tile_paths),
            "n_target": int(len(target)),
            "corrupt_tiles": sorted(CORRUPT_TILES),
            "truncated_per_split": trunc_counts,
            **meta_info,
        },
        "by_split": by_split,
        "truncation_check": truncation_check,
        "still_accepted_cause": still_cause,
        "totals": {
            "absent": summarize(absent_all, "absent (all splits)"),
            "truncated": summarize(target[target["truncated"]], "truncated (all splits)"),
        },
        "by_city": by_group(absent_all.dropna(subset=["city"]), "city"),
        "by_lcz": by_group(absent_all.dropna(subset=["lcz"]), "lcz"),
    }

    tot = payload["totals"]["absent"]
    logger.info(
        f"Absent: {tot['n']:,} → {tot['n_recoverable']:,} recoverable "
        f"({tot['frac_recoverable']:.1%}), {tot['n_correctly_rejected']:,} "
        f"correctly rejected"
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str))
    args.output_md.write_text(markdown_report(payload))
    logger.info(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
