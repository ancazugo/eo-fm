"""Amendment B4 — how often does a zone-straddling patch come back truncated?

Task 2.0d found that 294 of the 302 truncated-but-correctly-accepted Tessera
patches straddle a UTM zone boundary. That is a numerator without a denominator:
it says the truncated patches are at zone seams, not that being at a zone seam
truncates a patch. If nearly every straddling patch fails, `crop_patch`'s
multi-CRS mosaic is broken outright and Phase 4 map production is blocked for
London, Shanghai, Guangzhou and Hong Kong. If only a minority fail, the defect is
narrower and something else selects which ones.

This supplies the denominator by scanning the **whole** population rather than
only the failures.

What counts as "spanning multiple tiles" is taken from what the extractor
actually does, not from a plausible definition:
`extract_so2sat_embeddings` calls `tree.query(patch_geom)` — bounding-box
candidates, no predicate — and hands *every* match to `crop_patch`, which then
drops the ones whose clip comes back empty. So the count that governs
`len(arrays)` is the number of tiles the patch truly intersects, and the count
the extractor passes is the bbox one. Both are reported; the intersecting count
is used for the ratio, because `crop_patch` routes on `len(arrays)` and on the
number of distinct CRSs among them (`merge_multi_crs` vs `numpy_mosaic`).

**Measure only — the mosaic is not fixed here.** Rev B is explicit, and Phase 2
benchmarking is unaffected: `native_frac >= 0.5` already excludes these patches
from the manifest. The exposure is Phase 4, where `infer_roi` shares the crop
path. Logged as a GATE 3 decision input beside the re-extraction recovery count.

Example (the committed run):

    python src/diagnostics/mosaic_scope.py \\
        --output-json diagnostics/mosaic_scope.json \\
        --output-md diagnostics/mosaic_scope.md
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
    SPLITS,
    load_patch_metadata,
    truncation_mask,
)
from diagnostics.patch_manifest import (                           # noqa: E402
    DENOMINATOR_SPLIT,
    MIN_NATIVE_FRAC,
)
from utils.constants import DATA_DIR                               # noqa: E402

FAMILY = "tesserav1.1_global"


def tile_zone_codes(tile_paths: list[Path]) -> np.ndarray:
    """The UTM CRS of every tile, as an integer code per tile.

    Derived through `tessera_grid_geometry`, the same function the reader and
    Task 2.0d use, rather than by re-deriving the zone rule here — Tessera's
    naming has polar and dateline exceptions that a hand-rolled
    `(lon + 180) // 6` gets wrong.
    """
    from datasets.tiles import tessera_grid_geometry, tile_index_name

    names = [tile_index_name(p) for p in tile_paths]
    crs_by_name: dict[str, str] = {}
    zones = []
    for name in tqdm(names, unit="tile", desc="tile CRS"):
        crs = crs_by_name.get(name)
        if crs is None:
            crs = str(tessera_grid_geometry(name)[0])
            crs_by_name[name] = crs
        zones.append(crs)
    codes, uniques = pd.factorize(pd.Series(zones))
    logger.info(f"Tile index spans {len(uniques)} UTM zones")
    return np.asarray(codes)


def scan_population(gdf, tree, zone_codes: np.ndarray) -> pd.DataFrame:
    """Per patch: how many tiles it touches and how many UTM zones they span.

    Bulk-queried rather than looped: `STRtree.query` over the whole geometry
    array returns `(2, M)` index pairs, which turns 390k tree descents into one.
    """
    geoms = gdf.geometry.values

    bbox_pairs = tree.query(geoms)
    hit_pairs = tree.query(geoms, predicate="intersects")

    n_bbox = np.bincount(bbox_pairs[0], minlength=len(gdf))

    hits = pd.DataFrame({"i": hit_pairs[0], "zone": zone_codes[hit_pairs[1]]})
    agg = hits.groupby("i").agg(n_tiles=("zone", "size"), n_zones=("zone", "nunique"))

    out = pd.DataFrame({
        "patch_id": gdf["patch_id"].to_numpy(),
        "dataset": gdf["dataset"].to_numpy(),
        "n_tiles_bbox": n_bbox,
        "n_tiles": agg["n_tiles"].reindex(range(len(gdf)), fill_value=0).to_numpy(),
        "n_zones": agg["n_zones"].reindex(range(len(gdf)), fill_value=0).to_numpy(),
    })
    return out


def add_truncation(df: pd.DataFrame, scan: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach both truncation rules, and the crop shape, to the on-disk patches.

    Task 2.0a showed the two rules disagree by design — `native_frac >= 0.5`
    keeps a 33x17 crop that the 0.85-per-side rule calls truncated — so the ratio
    is reported under both rather than under whichever is more flattering.
    """
    train = scan[scan["dataset"] == DENOMINATOR_SPLIT]
    if train.empty:
        raise SystemExit(f"No {DENOMINATOR_SPLIT} rows for {FAMILY} in the scan.")
    expected_area = float((train["h"].astype("int64") * train["w"].astype("int64")).median())

    scan = scan.copy()
    scan["native_frac"] = (
        scan["h"].astype("int64") * scan["w"].astype("int64")) / expected_area

    per_side = pd.Series(False, index=scan.index)
    thresholds = {}
    for split in SPLITS:
        sub = scan[scan["dataset"] == split]
        if sub.empty:
            continue
        mask, min_h, min_w = truncation_mask(sub)
        per_side.loc[sub.index] = mask
        thresholds[split] = {"min_h": min_h, "min_w": min_w}
    scan["truncated_per_side"] = per_side
    scan["truncated_native_frac"] = scan["native_frac"] < MIN_NATIVE_FRAC

    merged = df.merge(
        scan[["patch_id", "dataset", "h", "w", "native_frac",
              "truncated_per_side", "truncated_native_frac"]],
        on=["patch_id", "dataset"], how="left",
    )
    merged["on_disk"] = merged["h"].notna()
    meta = {
        "expected_native_area": expected_area,
        "denominator_split": DENOMINATOR_SPLIT,
        "min_native_frac": MIN_NATIVE_FRAC,
        "per_side_thresholds": thresholds,
    }
    return merged, meta


def _rate(sub: pd.DataFrame, col: str) -> dict:
    return {
        "n": int(len(sub)),
        "n_truncated": int(sub[col].sum()),
        "frac_truncated": float(sub[col].mean()) if len(sub) else 0.0,
    }


def ratios(disk: pd.DataFrame) -> dict:
    """Rev B's four questions, under both truncation rules."""
    out: dict = {}
    for rule, col in (("per_side", "truncated_per_side"),
                      ("native_frac", "truncated_native_frac")):
        multi_tile = disk[disk["n_tiles"] > 1]
        straddle = disk[disk["n_zones"] > 1]
        single = disk[disk["n_tiles"] <= 1]
        out[rule] = {
            "all": _rate(disk, col),
            "single_tile": _rate(single, col),
            "multi_tile": _rate(multi_tile, col),
            "multi_tile_same_zone": _rate(multi_tile[multi_tile["n_zones"] <= 1], col),
            "zone_straddling": _rate(straddle, col),
            # The converse direction, so the association reads both ways.
            "frac_of_truncated_that_straddle": (
                float((disk[disk[col]]["n_zones"] > 1).mean())
                if int(disk[col].sum()) else 0.0),
            "frac_of_truncated_that_are_multi_tile": (
                float((disk[disk[col]]["n_tiles"] > 1).mean())
                if int(disk[col].sum()) else 0.0),
        }
    return out


def multi_tile_only_cases(disk: pd.DataFrame, col: str) -> dict:
    """The truncated patches that span tiles WITHOUT crossing a zone.

    Rev B asks whether these share a property, which would point at a second
    mechanism distinct from the multi-CRS mosaic.
    """
    sub = disk[disk[col] & (disk["n_tiles"] > 1) & (disk["n_zones"] <= 1)]
    if sub.empty:
        return {"n": 0}
    return {
        "n": int(len(sub)),
        "by_city": {str(c): int(n) for c, n in sub["city"].value_counts().head(10).items()},
        "by_split": {str(s): int(n) for s, n in sub["dataset"].value_counts().items()},
        "shapes": sorted({f"{int(h)}x{int(w)}" for h, w in zip(sub["h"], sub["w"])})[:20],
        "n_tiles_values": {int(k): int(v) for k, v in sub["n_tiles"].value_counts().items()},
        "native_frac_median": float(sub["native_frac"].median()),
        "lat_range": [float(sub["lat"].min()), float(sub["lat"].max())]
        if "lat" in sub else None,
    }


def by_city(disk: pd.DataFrame, col: str, top: int = 12) -> list[dict]:
    sub = disk[disk["n_zones"] > 1].dropna(subset=["city"])
    if sub.empty:
        return []
    g = (sub.groupby("city", observed=True)
            .agg(straddling=("patch_id", "size"), truncated=(col, "sum"))
            .reset_index())
    g["frac"] = g["truncated"] / g["straddling"]
    return (g.sort_values("straddling", ascending=False)
             .head(top).to_dict("records"))


# ── Reporting ────────────────────────────────────────────────────────────────

def markdown_report(payload: dict) -> str:
    meta = payload["meta"]
    r = payload["ratios"]["native_frac"]
    rp = payload["ratios"]["per_side"]
    key = r["zone_straddling"]

    lines = [
        "### Amendment B4 — Multi-zone mosaic defect, scoped (measure only)",
        "",
        f"Every `{FAMILY}` patch on disk ({meta['n_on_disk']:,}) tested for how "
        f"many tiles it intersects and how many UTM zones those tiles span, "
        f"against {meta['n_tiles_indexed']:,} exact footprints. Generated "
        f"{meta['generated']}.",
        "",
        "Task 2.0d reported that 294 of 302 truncated-but-accepted patches sit at "
        "a zone seam. That is a numerator; this is the denominator.",
        "",
        "#### The key ratio",
        "",
        f"**Of the {key['n']:,} patches that straddle a UTM zone boundary, "
        f"{key['n_truncated']:,} come back truncated — {key['frac_truncated']:.1%}.**",
        "",
        "| population | n | truncated | rate |",
        "|---|---|---|---|",
    ]
    labels = [
        ("all on disk", "all"),
        ("single tile", "single_tile"),
        ("multiple tiles", "multi_tile"),
        ("multiple tiles, one zone", "multi_tile_same_zone"),
        ("straddling a zone boundary", "zone_straddling"),
    ]
    for label, k in labels:
        c = r[k]
        lines.append(
            f"| {label} | {c['n']:,} | {c['n_truncated']:,} | {c['frac_truncated']:.2%} |")

    lines += [
        "",
        f"Read the other way: {r['frac_of_truncated_that_straddle']:.1%} of all "
        f"truncated patches straddle a zone, and "
        f"{r['frac_of_truncated_that_are_multi_tile']:.1%} span multiple tiles.",
        "",
        f"Under Phase 1.75's stricter 0.85-per-side rule the straddling rate is "
        f"{rp['zone_straddling']['frac_truncated']:.1%} "
        f"({rp['zone_straddling']['n_truncated']:,} of "
        f"{rp['zone_straddling']['n']:,}) — reported alongside because Task 2.0a "
        f"showed the two rules disagree by design.",
        "",
        payload["verdict"],
        "",
        "#### Zone-straddling patches by city",
        "",
        "| city | straddling | truncated | rate |",
        "|---|---|---|---|",
    ]
    for c in payload["by_city"]:
        lines.append(
            f"| {c['city']} | {int(c['straddling']):,} | {int(c['truncated']):,} | "
            f"{c['frac']:.1%} |")

    m = payload["multi_tile_only"]
    lines += ["", "#### Truncated across tiles but *within* one zone", ""]
    if not m["n"]:
        lines.append(
            "None. Every truncated multi-tile patch crosses a zone boundary, so "
            "there is no evidence of a second mechanism: `numpy_mosaic` (the "
            "same-CRS path) is not implicated.")
    else:
        lines += [
            f"{m['n']:,} patches, which `crop_patch` sends through "
            "`numpy_mosaic` rather than `merge_multi_crs`. Rev B asks whether "
            "they share a property that would indicate a second mechanism.",
            "",
            f"- cities: {', '.join(f'{c} ({n})' for c, n in m['by_city'].items())}",
            f"- splits: {', '.join(f'{s} ({n})' for s, n in m['by_split'].items())}",
            f"- tiles per patch: {m['n_tiles_values']}",
            f"- crop shapes: {', '.join(m['shapes'][:12])}",
            f"- median native_frac: {m['native_frac_median']:.3f}",
        ]

    lines += [
        "",
        "#### Absent patches, for contrast",
        "",
        f"{payload['absent']['n']:,} reference patches have no npy at all; "
        f"{payload['absent']['n_straddling']:,} of those straddle a zone. Absence "
        "is a coverage failure, not a mosaic one (Task 2.0d), so they are counted "
        "here but excluded from the ratio above — a patch that never came back "
        "cannot have come back truncated.",
        "",
        "**Nothing was changed.** The mosaic is not fixed in Phase 2; this is a "
        "GATE 3 input beside the re-extraction recovery count.",
        "",
    ]
    return "\n".join(lines)


def verdict(payload_ratios: dict) -> str:
    """State what the number means for Phase 4 before anyone has to infer it."""
    f = payload_ratios["native_frac"]["zone_straddling"]["frac_truncated"]
    base = payload_ratios["native_frac"]["single_tile"]["frac_truncated"]
    if f >= 0.9:
        return ("**Straddling a zone boundary essentially always truncates the "
                "crop.** `merge_multi_crs` is broken for this family, and map "
                "production for the seam cities is blocked until it is fixed.")
    if f <= 0.05:
        return (f"**Straddling a zone is not on its own sufficient to truncate a "
                f"crop** — {f:.1%} against a {base:.2%} baseline for single-tile "
                f"patches. The defect needs a further trigger, which has to be "
                f"identified before Phase 4 can judge the exposure.")
    return (f"**Straddling a zone truncates a minority of crops** — {f:.1%} "
            f"against a {base:.2%} baseline for single-tile patches. Elevated by "
            f"orders of magnitude but not deterministic, so a further trigger "
            f"selects which straddling patches fail; identifying it is the "
            f"Phase 4 prerequisite.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Amendment B4 — scope the multi-UTM-zone mosaic defect. Measure only.",
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
                   default=Path("diagnostics/mosaic_scope.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/mosaic_scope.md"))
    args = p.parse_args()

    import geopandas as gpd
    from datasets.tiles import build_tile_index

    scan = pd.read_parquet(args.invalid_frac_parquet)
    scan = scan[scan["family"].astype(str) == FAMILY].copy()
    scan["dataset"] = scan["dataset"].astype(str)
    scan["patch_id"] = scan["patch_id"].astype(str)
    logger.info(f"Scan: {len(scan):,} {FAMILY} patches on disk")

    gdf = gpd.read_file(args.so2sat_dir / "patches_reference_rxr.gpkg")
    gdf["patch_id"] = gdf["patch_id"].astype(str).str.zfill(6)
    gdf["dataset"] = gdf["dataset"].astype(str)
    logger.info(f"Reference population: {len(gdf):,} patches")

    tile_paths, tree = build_tile_index(args.embedding_dir, FAMILY, year=args.year)
    logger.info(f"Tile index: {len(tile_paths):,} exact footprints")
    zone_codes = tile_zone_codes(tile_paths)

    geom = scan_population(gdf, tree, zone_codes)
    geom, geom_meta = add_truncation(geom, scan)

    meta_df, meta_info = load_patch_metadata(args.so2sat_dir, args.bounds_csv)
    geom = geom.merge(
        meta_df[["patch_id", "dataset", "city", "LCZ_class"]].rename(
            columns={"LCZ_class": "lcz"}),
        on=["patch_id", "dataset"], how="left",
    )

    disk = geom[geom["on_disk"]].copy()
    absent = geom[~geom["on_disk"]]
    logger.info(
        f"On disk {len(disk):,} — multi-tile {int((disk['n_tiles'] > 1).sum()):,}, "
        f"zone-straddling {int((disk['n_zones'] > 1).sum()):,}")

    rat = ratios(disk)
    key = rat["native_frac"]["zone_straddling"]
    logger.info(
        f"KEY RATIO: {key['n_truncated']:,} of {key['n']:,} zone-straddling "
        f"patches truncated ({key['frac_truncated']:.1%})")

    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "family": FAMILY,
            "year": args.year,
            "embedding_dir": str(args.embedding_dir),
            "n_reference": int(len(gdf)),
            "n_on_disk": int(len(disk)),
            "n_tiles_indexed": len(tile_paths),
            **geom_meta,
            **meta_info,
        },
        "ratios": rat,
        "by_split": {
            split: ratios(disk[disk["dataset"] == split])["native_frac"]["zone_straddling"]
            for split in SPLITS if (disk["dataset"] == split).any()
        },
        "by_city": by_city(disk, "truncated_native_frac"),
        "multi_tile_only": multi_tile_only_cases(disk, "truncated_native_frac"),
        "absent": {
            "n": int(len(absent)),
            "n_straddling": int((absent["n_zones"] > 1).sum()),
            "n_no_tile": int((absent["n_tiles"] == 0).sum()),
        },
    }
    payload["verdict"] = verdict(rat)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str))
    args.output_md.write_text(markdown_report(payload))
    logger.info(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
