"""Task 1.75.1 — the nodata population, measured on every patch rather than a sample.

GATE 1.5 established that the AlphaEarth sentinel was characterized on a
*filtered* population: Phase 0 sampled patch_ids paired across five families,
and the paired intersection carries 0.0184% sentinel pixels against 15.7292%
outside it — an 855x ratio. Phase 2 trains on all 352,366 cultural-split
training patches, so every number derived from the paired sample understates
what the model will actually see.

This script re-characterizes on the **full** population, no pairing, for the
three PLAN-v3 families. It answers three questions:

**How much invalid data is there?** Per-patch invalid fractions, their
distribution, and the counts above 5 / 10 / 25 / 50%.

**Where does it sit?** Broken down by city, by LCZ class and by split. If the
high-invalid patches concentrate in particular cities — especially any of the 10
held-out cultural-split cities — then `alpha_earth_coop` is evaluated on a
different city distribution from the other two families and the Task 2.2
cross-family comparison inherits a *city-level* confound, not a pixel-level one.

**Do the other two families have their own holes?** The pairing hid everyone's.
For Tessera the answer is not nodata at all: patches go missing from disk
entirely, or come back truncated because the extraction's coverage test used a
bounding box (fixed in `datasets.tiles.exact_footprint_4326`, but the extracted
data predates it). Both are counted here, per city, beside the nodata.

**One pass, not two.** The post-resize masked statistics are provably
independent of the fill value: ``PatchDataset._resize_valid`` bilinearly
interpolates the 0/1 mask and demands a full weight of 1, so any output pixel
whose interpolation touched an invalid input is itself invalid and never enters
a masked sum. Verified empirically before relying on it — filling with 0.0 and
with an absurd 3.7 gives bit-identical masked means and stds on the patches that
actually have nodata. So the fill value never has to be known in advance and the
whole audit is a single read of each file.

Measure-only. Writes JSON + markdown + a per-patch parquet; the parquet is what
``--max-invalid-frac`` reads at train time so a training run never has to
rescan 352k files.

Example (the Phase 1.75 run):

    python src/diagnostics/nodata_population.py --workers 12 \\
        --output-json diagnostics/nodata_population.json \\
        --output-md diagnostics/nodata_population.md \\
        --output-parquet diagnostics/invalid_fraction.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Before numpy and torch are imported, so the thread pools are never built.
# Workers are forked, so a later setting would not reach them. See
# _single_threaded() for what oversubscription cost when this was missing.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.registry import get_nodata_predicate                # noqa: E402
from diagnostics.embedding_stats import (                         # noqa: E402
    AUGMENT_NOISE_SIGMA,
    FAMILIES,
    build_id_index,
    load_raw,
)
from diagnostics.resize_audit import resize_image, resize_valid   # noqa: E402
from utils.constants import DATA_DIR                              # noqa: E402
from utils.runtime import resolve_dequantize                      # noqa: E402

# PLAN-v3 scope. Deliberately spelled out rather than taken from
# available_embeddings(): osm_evidence and aux_struct are available but are not
# per-patch npy families and have no place in this audit.
SCAN_FAMILIES = ["alpha_earth_coop", "tesserav1.1_global", "seamless"]

SPLITS = ["training", "validation", "testing"]

# The 10 cities held out of the cultural split. Validation and testing draw from
# the SAME 10 cities (roughly a 50/50 split within each), so a coverage hole in
# any of them lands on the evaluation set for both.
HELDOUT_CITIES = {
    "Guangzhou", "Jakarta", "Moscow", "Mumbai", "Munich", "Nairobi",
    "San Jose", "Santiago", "Sydney", "Tehran",
}

# A truncated crop is one materially smaller than its family's own native size,
# so the rule has to be relative: the 10 m families sit near 33 px and `seamless`
# at 30 m sits near 12, and a fixed pixel threshold would label every ESD patch
# truncated. 0.85 x the family median leaves the +/-1 px reprojection drift that
# Task 1.75.2 measures well inside the normal band while still catching a crop
# that lost a third of its extent.
TRUNCATION_RATIO = 0.85

# Fallback for callers that need a single number for the 10 m families (their
# median native size is 33 px, so this is 0.85 x 33 rounded).
TRUNCATION_PX = 28

INVALID_THRESHOLDS = (0.05, 0.10, 0.25, 0.50)


# ── Scanning ─────────────────────────────────────────────────────────────────

_WORKER_CACHE: dict[str, tuple] = {}


def _single_threaded() -> None:
    """Pin each worker to one thread.

    numpy's BLAS and torch both default to using every core, so a pool of N
    workers becomes N x ncores threads fighting over the same cache. Measured
    at 8 workers on this machine: ~3000% CPU *per worker* and a validation-split
    scan that had not finished in 13 minutes against a 3-minute serial estimate.
    The work here is one small array per file — there is nothing for a second
    thread to do inside a worker anyway.
    """
    import torch

    torch.set_num_threads(1)


def _worker_setup(family: str) -> tuple:
    """Per-process dequantizer + predicate, built once and cached.

    resolve_dequantize logs on every call, so without this the worker pool
    prints one line per chunk for no reason.
    """
    if family not in _WORKER_CACHE:
        _single_threaded()
        spec = FAMILIES[family]
        dequantize_fn, _ = resolve_dequantize(spec["embedding_name"])
        predicate = get_nodata_predicate(spec["embedding_name"])
        _WORKER_CACHE[family] = (spec, dequantize_fn, predicate)
    return _WORKER_CACHE[family]


def _scan_chunk(args: tuple) -> tuple:
    """Scan one block of patches. Returns per-patch rows plus partial moments.

    Moments are raw sums (s1, s2, n) in float64, which are additive, so the
    parent merges chunks by adding them — no approximation from chunking.
    """
    family, patch_size, items = args
    spec, dequantize_fn, predicate = _worker_setup(family)

    rows: list[tuple] = []
    acc: dict[str, list] = {}          # key -> [s1, s2, n]

    def _add(key: str, flat: np.ndarray, w: np.ndarray | None) -> None:
        if key not in acc:
            acc[key] = [np.zeros(flat.shape[0]), np.zeros(flat.shape[0]), 0.0]
        s = acc[key]
        if w is None:
            s[0] += flat.sum(axis=1)
            s[1] += (flat * flat).sum(axis=1)
            s[2] += flat.shape[1]
        else:
            s[0] += (flat * w).sum(axis=1)
            s[1] += ((flat * flat) * w).sum(axis=1)
            s[2] += float(w.sum())

    for pid, path in items:
        raw = load_raw(Path(path), spec["kind"])
        H, W = raw.shape[-2:]
        invalid = predicate(raw)                       # (H, W), stored units
        n_invalid = int(invalid.sum())
        rows.append((pid, H, W, n_invalid, int(invalid.size)))

        arr = np.nan_to_num(raw, nan=0.0)
        if dequantize_fn is not None:
            arr = dequantize_fn(arr)
        valid = ~invalid

        # No fill anywhere: the sentinel stays in place, so the *unmasked*
        # variants measure what the pre-Phase-1 pipeline actually saw and the
        # variance share is meaningful. The *masked* variants are unaffected by
        # that choice — leaving the sentinel in is just another fill value, and
        # a valid output pixel never drew on an invalid input one.
        flat_native = arr.reshape(arr.shape[0], -1).astype(np.float64)
        _add("native_unmasked", flat_native, None)
        _add("native_masked", flat_native, valid.reshape(-1).astype(np.float64))

        rs = resize_image(arr, patch_size)
        rv = resize_valid(valid, patch_size)
        flat_rs = rs.reshape(rs.shape[0], -1).astype(np.float64)
        _add("resized_unmasked", flat_rs, None)
        _add("resized_masked", flat_rs, rv.reshape(-1).astype(np.float64))

    return rows, {k: (v[0], v[1], v[2]) for k, v in acc.items()}


def scan_family_split(
    so2sat_dir: Path, family: str, split: str, year: str,
    patch_size: int, workers: int, chunk: int,
) -> tuple[pd.DataFrame, dict]:
    """Full scan of one family in one split. Every patch, no sampling."""
    index = build_id_index(so2sat_dir, family, split, year)
    items = sorted((pid, str(p)) for pid, p in index.items())
    blocks = [
        (family, patch_size, items[i:i + chunk])
        for i in range(0, len(items), chunk)
    ]

    rows: list[tuple] = []
    merged: dict[str, list] = {}

    def _merge(partial: dict) -> None:
        for key, (s1, s2, n) in partial.items():
            if key not in merged:
                merged[key] = [np.zeros_like(s1), np.zeros_like(s2), 0.0]
            merged[key][0] += s1
            merged[key][1] += s2
            merged[key][2] += n

    desc = f"{family}/{split}"
    if workers <= 1:
        for block in tqdm(blocks, desc=desc, unit="blk"):
            r, m = _scan_chunk(block)
            rows.extend(r)
            _merge(m)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_scan_chunk, b) for b in blocks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=desc, unit="blk"):
                r, m = fut.result()
                rows.extend(r)
                _merge(m)

    df = pd.DataFrame(rows, columns=["patch_id", "h", "w", "n_invalid", "n_pixels"])
    df["dataset"] = split
    df["family"] = family
    df["invalid_frac"] = df["n_invalid"] / df["n_pixels"]

    stats = {}
    for key, (s1, s2, n) in merged.items():
        mean = s1 / max(n, 1.0)
        std = np.sqrt(np.maximum(s2 / max(n, 1.0) - mean ** 2, 0.0))
        stats[key] = {
            "median_std": float(np.median(std)),
            "min_std": float(std.min()),
            "max_std": float(std.max()),
            "mean_of_means": float(mean.mean()),
            "n_pixel_samples": float(n),
        }
    return df.sort_values("patch_id").reset_index(drop=True), stats


# ── Metadata join ────────────────────────────────────────────────────────────

def load_patch_metadata(so2sat_dir: Path, bounds_csv: Path) -> tuple[pd.DataFrame, dict]:
    """``(dataset, patch_id) -> city, lcz, centre lat/lon`` for every So2Sat patch.

    City comes from a containment join against the 51 GUPPD city boxes. Two of
    them nest — the Guangzhou box fully contains the Hong Kong box — so a plain
    join double-matches those patches. ``datasets.so2sat.assign_cities`` keeps
    whichever row the join happens to emit first; here the **smallest containing
    box** wins instead, which resolves Hong Kong to Hong Kong deterministically.
    The count of rows that needed the tie-break is reported rather than hidden.
    """
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.read_file(so2sat_dir / "patches_reference_rxr.gpkg")
    with warnings.catch_warnings():
        # A So2Sat patch is 320 m across; the geographic-CRS centroid error is
        # sub-metre, far too small to move a patch into another city.
        warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")
        centres = gdf.geometry.centroid
    gdf["lon"] = centres.x
    gdf["lat"] = centres.y

    bounds = pd.read_csv(bounds_csv)
    boxes = gpd.GeoDataFrame(
        bounds[["JRC_NAME_MAIN"]].rename(columns={"JRC_NAME_MAIN": "city"}),
        geometry=[box(r.minx, r.miny, r.maxx, r.maxy) for r in bounds.itertuples()],
        crs="EPSG:4326",
    )
    boxes["box_area"] = boxes.geometry.area

    pts = gpd.GeoDataFrame(gdf[["patch_id", "dataset"]], geometry=centres,
                           crs=gdf.crs)
    joined = gpd.sjoin(pts, boxes, how="left", predicate="within")
    n_ambiguous = int(joined.index.duplicated().sum())
    joined = (joined.sort_values("box_area")
                    .loc[lambda d: ~d.index.duplicated(keep="first")]
                    .sort_index())

    meta = gdf[["patch_id", "dataset", "LCZ_class", "lat", "lon"]].copy()
    meta["city"] = joined["city"].values
    meta["patch_id"] = meta["patch_id"].astype(str).str.zfill(6)
    info = {
        "n_patches": int(len(meta)),
        "n_city_assigned": int(meta["city"].notna().sum()),
        "n_ambiguous_city_matches": n_ambiguous,
        "n_cities": int(meta["city"].nunique()),
    }
    return meta, info


# ── Reporting ────────────────────────────────────────────────────────────────

def _quantiles(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
    }


def truncation_mask(df: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """Which crops are truncated, judged against the family's own native size.

    Returns ``(mask, min_h, min_w)`` so the thresholds can be reported: a rule
    the reader cannot see is a rule they cannot check.
    """
    min_h = TRUNCATION_RATIO * float(df["h"].median())
    min_w = TRUNCATION_RATIO * float(df["w"].median())
    return (df["h"] < min_h) | (df["w"] < min_w), min_h, min_w


def summarize(df: pd.DataFrame, n_reference: int) -> dict:
    """Per-family, per-split headline numbers."""
    frac = df["invalid_frac"].to_numpy()
    truncated, min_h, min_w = truncation_mask(df)
    return {
        "n_on_disk": int(len(df)),
        "n_in_reference": int(n_reference),
        "n_missing": int(n_reference - len(df)),
        "frac_missing": float((n_reference - len(df)) / max(n_reference, 1)),
        "total_invalid_px_frac": float(df["n_invalid"].sum() / max(df["n_pixels"].sum(), 1)),
        "frac_patches_with_any_invalid": float((frac > 0).mean()),
        "per_patch_invalid_frac": _quantiles(frac),
        "n_above": {f"{t:.2f}": int((frac > t).sum()) for t in INVALID_THRESHOLDS},
        "n_truncated": int(truncated.sum()),
        "median_shape": [float(df["h"].median()), float(df["w"].median())],
        "truncation_threshold": [min_h, min_w],
    }


def histogram(frac: np.ndarray) -> list[tuple[str, int]]:
    """Per-patch invalid-fraction histogram, in the buckets a threshold is picked from."""
    edges = [0.0, 1e-12, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0 + 1e-9]
    labels = ["exactly 0", "0-0.1%", "0.1-1%", "1-5%", "5-10%",
              "10-25%", "25-50%", "50-75%", "75-100%"]
    counts = np.histogram(frac, bins=edges)[0]
    return list(zip(labels, counts.tolist()))


def markdown_report(payload: dict) -> str:
    meta = payload["meta"]
    per = payload["per_family_split"]
    families = meta["families"]
    splits = meta["splits"]
    lines = [
        "### Task 1.75.1 — Nodata population on the full cultural split",
        "",
        f"Every patch of every split, no pairing and no sampling — "
        f"{meta['n_patches_scanned']:,} patch reads across "
        f"{len(families)} families. Generated {meta['generated']}.",
        "",
        "#### Headline — coverage and invalid data per family and split",
        "",
        "| family | split | on disk | missing | invalid px | patches w/ any | "
        "p50 | p90 | p99 | max | truncated |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for family in families:
        for split in splits:
            s = per[family][split]
            q = s["per_patch_invalid_frac"]
            lines.append(
                f"| `{family}` | {split} | {s['n_on_disk']:,} | "
                f"{s['n_missing']:,} ({s['frac_missing']:.2%}) | "
                f"{s['total_invalid_px_frac']:.4%} | "
                f"{s['frac_patches_with_any_invalid']:.2%} | "
                f"{q['p50']:.4%} | {q['p90']:.4%} | {q['p99']:.4%} | "
                f"{q['max']:.2%} | {s['n_truncated']:,} |"
            )

    focus = "training" if "training" in splits else splits[0]
    lines += [
        "",
        "`missing` counts patches present in `patches_reference_rxr.gpkg` with no "
        "npy on disk. `truncated` counts native crops below "
        f"{TRUNCATION_RATIO:.2f} x the family's own median native size — "
        "extracted before `exact_footprint_4326`, and stretched to the model's "
        "patch size rather than dropped. The rule is relative because the 10 m "
        "families crop near 33 px and 30 m `seamless` near 12.",
        "",
        f"#### Counts above each candidate drop threshold ({focus} split)",
        "",
        "| family | >5% | >10% | >25% | >50% |",
        "|---|---|---|---|---|",
    ]
    for family in families:
        a = per[family][focus]["n_above"]
        lines.append(
            f"| `{family}` | {a['0.05']:,} | {a['0.10']:,} | "
            f"{a['0.25']:,} | {a['0.50']:,} |"
        )

    lines += ["", f"#### Per-patch invalid-fraction histogram ({focus} split)", "",
              "| bucket | " + " | ".join(f"`{f}`" for f in families) + " |",
              "|---|" + "---|" * len(families)]
    hists = {f: dict(payload["histograms"][f]) for f in families}
    for label, _ in payload["histograms"][families[0]]:
        cells = " | ".join(f"{hists[f][label]:,}" for f in families)
        lines.append(f"| {label} | {cells} |")

    lines += [
        "",
        "#### Coverage bias by city",
        "",
        "Cities sorted by AlphaEarth mean invalid fraction. **Bold** = one of the "
        "10 held-out cultural-split cities (validation and testing draw from the "
        "same 10, so a hole there lands on both).",
        "",
        "| city | held out | coop invalid | coop >25% | tessera missing | "
        "tessera truncated | seamless missing | n patches |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["by_city"]:
        name = f"**{row['city']}**" if row["held_out"] else row["city"]
        lines.append(
            f"| {name} | {'yes' if row['held_out'] else ''} | "
            f"{row['coop_invalid_frac']:.4%} | {row['coop_n_above_25']:,} | "
            f"{row['tessera_n_missing']:,} | {row['tessera_n_truncated']:,} | "
            f"{row['seamless_n_missing']:,} | {row['n_patches']:,} |"
        )

    lines += [
        "",
        f"#### Coverage bias by LCZ class ({focus} split, AlphaEarth)",
        "",
        "| LCZ | invalid frac | patches | | LCZ | invalid frac | patches |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = payload["by_lcz"]
    half = (len(rows) + 1) // 2
    for i in range(half):
        left = rows[i]
        cells = (f"| {left['lcz']} | {left['coop_invalid_frac']:.4%} | "
                 f"{left['n_patches']:,} |")
        if i + half < len(rows):
            r = rows[i + half]
            cells += (f" | {r['lcz']} | {r['coop_invalid_frac']:.4%} | "
                      f"{r['n_patches']:,} |")
        else:
            cells += " | | | |"
        lines.append(cells)

    lines += [
        "",
        f"#### Corrected channel statistics — full unfiltered {focus} population",
        "",
        "Supersedes the Phase 0 (paired, n=5000, unmasked) and GATE 1 (n=20000, "
        "masked) tables. `masked` excludes nodata pixels; post-resize masked "
        "statistics are independent of the fill value by construction, so no fill "
        "had to be assumed.",
        "",
        "| family | native unmasked | native masked | resized unmasked | "
        "resized masked | sentinel share of variance | `0.05 / median_std` |",
        "|---|---|---|---|---|---|---|",
    ]
    for family in payload["channel_stats"]:
        st = payload["channel_stats"][family]
        lines.append(
            f"| `{family}` | {st['native_unmasked']['median_std']:.4f} | "
            f"{st['native_masked']['median_std']:.4f} | "
            f"{st['resized_unmasked']['median_std']:.4f} | "
            f"**{st['resized_masked']['median_std']:.4f}** | "
            f"{st['sentinel_variance_share_native']:.2%} | "
            f"**{st['effective_noise_ratio']:.3f}** |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

FAMILY_PREFIX = {
    "alpha_earth_coop": "coop",
    "tesserav1.1_global": "tessera",
    "seamless": "seamless",
}


def build_city_table(
    full: pd.DataFrame, meta: pd.DataFrame, splits: list[str],
) -> list[dict]:
    """Per-city coverage across all three families, sorted by AlphaEarth nodata.

    ``missing`` is the reference count minus what each family has on disk, so a
    city can be short on patches without having a single invalid pixel — which
    is exactly Tessera's failure mode and AlphaEarth's non-failure mode.

    The reference is restricted to *splits* actually scanned; counting a city's
    training patches against a validation-only scan would report every training
    city as 100% missing.
    """
    reference = meta[meta["dataset"].isin(splits)].groupby("city", dropna=True).size()
    rows = {
        city: {"city": city, "held_out": city in HELDOUT_CITIES,
               "n_patches": int(n)}
        for city, n in reference.items()
    }
    for prefix in FAMILY_PREFIX.values():
        for row in rows.values():
            row.update({f"{prefix}_invalid_frac": 0.0, f"{prefix}_n_above_25": 0,
                        f"{prefix}_n_missing": row["n_patches"],
                        f"{prefix}_n_truncated": 0})

    have = full.dropna(subset=["city"]).copy()
    have["above_25"] = have["invalid_frac"] > 0.25
    have["truncated"] = False
    for family, group in have.groupby("family"):
        mask, _, _ = truncation_mask(group)
        have.loc[group.index, "truncated"] = mask
    agg = have.groupby(["family", "city"]).agg(
        n_invalid=("n_invalid", "sum"), n_pixels=("n_pixels", "sum"),
        n_present=("patch_id", "size"), n_above_25=("above_25", "sum"),
        n_truncated=("truncated", "sum"),
    ).reset_index()

    for r in agg.itertuples():
        row = rows.get(r.city)
        prefix = FAMILY_PREFIX.get(r.family)
        if row is None or prefix is None:
            continue
        row[f"{prefix}_invalid_frac"] = float(r.n_invalid / r.n_pixels) if r.n_pixels else 0.0
        row[f"{prefix}_n_above_25"] = int(r.n_above_25)
        row[f"{prefix}_n_missing"] = int(row["n_patches"] - r.n_present)
        row[f"{prefix}_n_truncated"] = int(r.n_truncated)

    return sorted(rows.values(), key=lambda r: -r["coop_invalid_frac"])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 1.75.1 — full-population nodata and coverage audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--bounds-csv", type=Path,
                   default=Path(__file__).resolve().parents[2] / "data"
                   / "so2sat_guppd_bounds.csv")
    p.add_argument("--year", default="2017")
    p.add_argument("--families", nargs="+", default=SCAN_FAMILIES,
                   choices=list(FAMILIES))
    p.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8,
                   help="Process pool size; 1 runs serially.")
    p.add_argument("--chunk", type=int, default=500,
                   help="Patches per work block.")
    p.add_argument("--output-json", type=Path,
                   default=Path("diagnostics/nodata_population.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/nodata_population.md"))
    p.add_argument("--output-parquet", type=Path,
                   default=Path("diagnostics/invalid_fraction.parquet"))
    args = p.parse_args()

    logger.info(f"Loading patch metadata from {args.so2sat_dir}")
    meta, meta_info = load_patch_metadata(args.so2sat_dir, args.bounds_csv)
    logger.info(
        f"{meta_info['n_patches']:,} patches, "
        f"{meta_info['n_city_assigned']:,} assigned to one of "
        f"{meta_info['n_cities']} cities "
        f"({meta_info['n_ambiguous_city_matches']} needed the smallest-box tie-break)"
    )
    reference_counts = meta.groupby("dataset").size().to_dict()

    frames: list[pd.DataFrame] = []
    per_family_split: dict[str, dict] = defaultdict(dict)
    raw_stats: dict[str, dict] = defaultdict(dict)

    for family in args.families:
        for split in args.splits:
            df, stats = scan_family_split(
                args.so2sat_dir, family, split, args.year,
                args.patch_size, args.workers, args.chunk,
            )
            frames.append(df)
            per_family_split[family][split] = summarize(
                df, reference_counts.get(split, len(df)))
            raw_stats[family][split] = stats
            s = per_family_split[family][split]
            logger.info(
                f"{family}/{split}: {s['n_on_disk']:,} on disk "
                f"({s['n_missing']:,} missing), invalid {s['total_invalid_px_frac']:.4%}, "
                f"{s['n_truncated']:,} truncated"
            )

    full = pd.concat(frames, ignore_index=True)
    full = full.merge(meta[["dataset", "patch_id", "city", "LCZ_class"]],
                      on=["dataset", "patch_id"], how="left")
    full = full.rename(columns={"LCZ_class": "lcz"})

    # The parquet is committed, so it is worth narrowing the dtypes: 1.2M rows
    # of str/int64 is 8.9 MB, and categorical + zstd is 3.8 MB for the same
    # content. `city` and `lcz` ride along so the audit can be re-analysed
    # without re-reading the 400k-row GPKG, which costs minutes.
    out = full[["family", "dataset", "patch_id", "invalid_frac", "n_invalid",
                "n_pixels", "h", "w", "city", "lcz"]].astype({
        "family": "category", "dataset": "category", "city": "category",
        "invalid_frac": "float32", "n_invalid": "int32", "n_pixels": "int32",
        "h": "int16", "w": "int16", "lcz": "float32",
    })
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_parquet, index=False, compression="zstd")
    logger.info(f"Wrote {args.output_parquet} ({len(full):,} rows)")

    # Channel statistics on the training split — that is the population the
    # normalizer is fitted on and the one the noise ratio is about.
    focus = "training" if "training" in args.splits else args.splits[0]
    channel_stats: dict[str, dict] = {}
    for family in args.families:
        st = dict(raw_stats[family].get(focus, {}))
        if not st:
            continue
        su = st["native_unmasked"]["median_std"]
        sm = st["native_masked"]["median_std"]
        st["sentinel_variance_share_native"] = (
            float(max(0.0, 1.0 - (sm / su) ** 2)) if su > 0 else 0.0)
        st["effective_noise_ratio"] = float(
            AUGMENT_NOISE_SIGMA / st["resized_masked"]["median_std"])
        channel_stats[family] = st

    train = full[full["dataset"] == focus]
    coop = train[train["family"] == "alpha_earth_coop"]
    by_lcz = [
        {"lcz": int(lcz),
         "coop_invalid_frac": float(g["n_invalid"].sum() / max(g["n_pixels"].sum(), 1)),
         "n_patches": int(len(g))}
        for lcz, g in coop.groupby("lcz")
    ]

    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "so2sat_dir": str(args.so2sat_dir),
            "year": args.year,
            "families": args.families,
            "splits": args.splits,
            "patch_size": args.patch_size,
            "truncation_ratio": TRUNCATION_RATIO,
            "focus_split": focus,
            "augment_noise_sigma": AUGMENT_NOISE_SIGMA,
            "n_patches_scanned": int(len(full)),
            "reference_counts": {k: int(v) for k, v in reference_counts.items()},
            **meta_info,
        },
        "per_family_split": {f: dict(v) for f, v in per_family_split.items()},
        "channel_stats": channel_stats,
        "histograms": {
            f: histogram(train.loc[train["family"] == f, "invalid_frac"].to_numpy())
            for f in args.families
        },
        "by_city": build_city_table(full, meta, args.splits),
        "by_lcz": sorted(by_lcz, key=lambda r: r["lcz"]),
    }

    # default=str: the per-city and per-LCZ rows come straight out of pandas and
    # carry numpy scalars, which json cannot serialise on its own.
    args.output_json.write_text(json.dumps(payload, indent=2, default=str))
    args.output_md.write_text(markdown_report(payload))
    logger.info(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
