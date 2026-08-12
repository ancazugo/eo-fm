"""Task 1.75.2 — where the varying native crop geometry comes from, and what it costs.

A So2Sat patch is nominally 320 m, but the 10 m families crop at 33-36 x 32-35
pixels natively, so the true footprint is ~330-360 m and it **varies between
patches**. Everything is then bilinearly resized to 32x32, which means the
effective ground resolution the model sees is not constant either.

The question this answers is whether that variation is *systematic*. If crop
extent tracks latitude, effective resolution differs systematically between
equatorial and high-latitude cities — a preprocessing-induced domain shift
sitting inside the cultural split, on exactly the axis Phase 4 attacks, and one
no amount of test-time adaptation could fix. A null result is worth just as
much: it lets Phase 4 drop a covariate on evidence rather than on assumption.

Three candidate mechanisms are separated here:

* **A degree grid.** If tiles were in EPSG:4326, longitude pixel size would vary
  as cos(phi) and width would collapse toward the poles. Checked directly by
  opening a tile and reading its CRS, not inferred.
* **Reprojection distortion.** The patch polygons are lon/lat rectangles;
  ``crop_patch`` reprojects each into the tile's UTM zone and clips to the
  *bounding box* of the result. A lat/lon rectangle maps to a trapezoid, whose
  bounding box grows with latitude and with distance from the zone's central
  meridian. This is the mechanism the data actually supports.
* **Truncation.** A patch straddling the edge of tile coverage comes back
  short — a different pathology from the +/-1 px drift, and one that would
  contaminate a correlation estimate if left in. Reported separately and
  excluded from the correlations.

Shapes come from the ``.npy`` headers only (``mmap_mode="r"``), so no pixel data
is read; with the Task 1.75.1 parquet available it uses the full population for
free.

**No resampling is changed here.** Bilinear at 32x32 is what every existing
number used; ``--resize-mode exact-crop`` is a Task 2.3 ablation.

Example (the Phase 1.75 run):

    python src/diagnostics/crop_geometry.py \\
        --invalid-frac-parquet diagnostics/invalid_fraction.parquet \\
        --output-json diagnostics/crop_geometry.json \\
        --output-md diagnostics/crop_geometry.md \\
        --output-png diagnostics/crop_geometry.png
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
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.embedding_stats import FAMILIES, build_id_index      # noqa: E402
from diagnostics.nodata_population import (                           # noqa: E402
    HELDOUT_CITIES,
    TRUNCATION_RATIO,
    load_patch_metadata,
    truncation_mask,
)
from utils.constants import DATA_DIR                                  # noqa: E402

GEOMETRY_FAMILIES = ["tesserav1.1_global", "alpha_earth_coop"]

# Both 10 m families rasterise at exactly 10 m in their tile's UTM zone, so a
# crop of H pixels spans H * 10 m on the ground.
PIXEL_METRES = 10.0


def utm_zone_and_meridian(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """UTM zone number and its central meridian, by the same rule as
    ``datasets.tiles.tessera_grid_geometry``.

    Distortion in a transverse Mercator projection grows with distance from the
    central meridian, so this is the second axis a bounding-box clip can widen
    along — and it is not the same axis as latitude.
    """
    zone = (np.floor((lon + 180) / 6) + 1).astype(int)
    return zone, -180.0 + 6.0 * zone - 3.0


def tile_crs_check() -> dict:
    """Report the CRS real tiles are stored in, read from their georeferencing.

    The degree-grid hypothesis is falsifiable in one read; inferring it from the
    shapes would be guessing at the mechanism from the symptom. Only the header
    is read — a Tessera tile decodes to ~630 MB, and opening one to learn its
    CRS would cost minutes for four bytes of answer.
    """
    import rasterio

    out: dict[str, str] = {}
    probes = {
        # v1.1's georeferencing lives in year-independent GeoTIFFs beside the
        # per-year arrays; coop tiles are GeoTIFFs in per-UTM-zone subdirs.
        "tesserav1.1_global": (Path("/tessera/v1.1/global_0.1_degree_tiff_all"),
                               "*.tif*"),
        "alpha_earth_coop": (DATA_DIR / "input" / "Google" / "AlphaEarth" / "coop"
                             / "2017", "*/*.tif*"),
    }
    for family, (directory, pattern) in probes.items():
        try:
            tile = next(iter(sorted(directory.glob(pattern))), None)
            if tile is None:
                out[family] = f"unavailable (no {pattern} under {directory})"
                continue
            with rasterio.open(tile) as ds:
                out[family] = f"{ds.crs} (res {abs(ds.transform.a):g} m, {tile.name})"
        except Exception as e:                                        # noqa: BLE001
            out[family] = f"unavailable ({type(e).__name__}: {e})"
    return out


# ── Shape collection ─────────────────────────────────────────────────────────

def shapes_from_parquet(parquet: Path, family: str, split: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet, columns=["family", "dataset", "patch_id", "h", "w"])
    return df[(df["family"] == family) & (df["dataset"] == split)].drop(
        columns=["family"]).reset_index(drop=True)


def shapes_from_headers(
    so2sat_dir: Path, family: str, split: str, year: str, n_sample: int, seed: int,
) -> pd.DataFrame:
    """Read only the npy headers — shape without touching a pixel."""
    index = build_id_index(so2sat_dir, family, split, year)
    ids = sorted(index)
    if len(ids) > n_sample:
        rng = np.random.default_rng(seed)
        ids = sorted(ids[i] for i in rng.choice(len(ids), n_sample, replace=False))
    rows = []
    for pid in tqdm(ids, desc=f"{family}/{split}", unit="hdr"):
        shape = np.load(index[pid], mmap_mode="r").shape
        rows.append((pid, shape[-2], shape[-1]))
    return pd.DataFrame(rows, columns=["patch_id", "h", "w"]).assign(dataset=split)


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyse(df: pd.DataFrame) -> dict:
    """Correlations and effective-resolution summary for one family.

    Truncated crops are excluded from the correlations: they are a coverage
    failure, not the reprojection drift being measured, and a handful of 12 px
    crops would dominate a rank correlation.
    """
    truncated, min_h, min_w = truncation_mask(df)
    clean = df[~truncated].assign(
        abs_lat=lambda d: d["lat"].abs(),
        dist_meridian=lambda d: (d["lon"] - d["central_meridian"]).abs(),
        res_h=lambda d: d["h"] * PIXEL_METRES / 32.0,
        res_w=lambda d: d["w"] * PIXEL_METRES / 32.0,
        # h and w move in OPPOSITE directions with latitude, so neither axis
        # alone says whether the ground sampling distance actually changes. The
        # geometric mean is the scale-invariant summary: it is what a patch's
        # total ground area per output pixel works out to.
        res_geo=lambda d: np.sqrt(d["h"] * d["w"]) * PIXEL_METRES / 32.0,
    )
    n_truncated = int(truncated.sum())

    def rho(a: str, b: str) -> dict:
        r = spearmanr(clean[a], clean[b])
        return {"rho": float(r.statistic), "p": float(r.pvalue)}

    by_city = (
        clean.groupby("city")
        .agg(n=("patch_id", "size"), lat=("lat", "mean"),
             h=("h", "mean"), w=("w", "mean"),
             h_std=("h", "std"), w_std=("w", "std"),
             res_h=("res_h", "mean"), res_w=("res_w", "mean"),
             res_geo=("res_geo", "mean"))
        .reset_index()
        .sort_values("lat")
    )
    by_city["held_out"] = by_city["city"].isin(HELDOUT_CITIES).astype(bool)

    # Latitude bands rather than raw scatter: the effect is a ~1 px drift on a
    # ~33 px crop, which only reads as ordered when binned.
    bands = clean.assign(band=pd.cut(clean["abs_lat"], bins=[0, 15, 30, 40, 50, 90]))
    by_band = (
        bands.groupby("band", observed=True)
        .agg(n=("patch_id", "size"), h=("h", "mean"), w=("w", "mean"),
             res_h=("res_h", "mean"), res_w=("res_w", "mean"),
             res_geo=("res_geo", "mean"))
        .reset_index()
    )
    by_band["band"] = by_band["band"].astype(str)

    return {
        "n_patches": int(len(df)),
        "n_analysed": int(len(clean)),
        "n_truncated": n_truncated,
        "frac_truncated": float(n_truncated / max(len(df), 1)),
        "truncation_threshold": [min_h, min_w],
        "shape_counts": {
            f"{int(h)}x{int(w)}": int(n) for (h, w), n in
            df.groupby(["h", "w"]).size().sort_values(ascending=False).head(10).items()
        },
        "h_range": [int(df["h"].min()), int(df["h"].max())],
        "w_range": [int(df["w"].min()), int(df["w"].max())],
        "spearman": {
            "h_vs_lat": rho("h", "lat"),
            "w_vs_lat": rho("w", "lat"),
            "h_vs_abs_lat": rho("h", "abs_lat"),
            "w_vs_abs_lat": rho("w", "abs_lat"),
            "h_vs_dist_meridian": rho("h", "dist_meridian"),
            "w_vs_dist_meridian": rho("w", "dist_meridian"),
            "res_geo_vs_abs_lat": rho("res_geo", "abs_lat"),
        },
        "effective_resolution_m": {
            "h": {"mean": float(clean["res_h"].mean()),
                  "min": float(clean["res_h"].min()),
                  "max": float(clean["res_h"].max())},
            "w": {"mean": float(clean["res_w"].mean()),
                  "min": float(clean["res_w"].min()),
                  "max": float(clean["res_w"].max())},
            "geo": {"mean": float(clean["res_geo"].mean()),
                    "min": float(clean["res_geo"].min()),
                    "max": float(clean["res_geo"].max())},
            "city_mean_range_h": [float(by_city["res_h"].min()),
                                  float(by_city["res_h"].max())],
            "city_mean_range_w": [float(by_city["res_w"].min()),
                                  float(by_city["res_w"].max())],
            "city_mean_range_geo": [float(by_city["res_geo"].min()),
                                    float(by_city["res_geo"].max())],
        },
        "by_city": by_city.to_dict("records"),
        "by_latitude_band": by_band.to_dict("records"),
    }


def scatter_png(frames: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(frames), 2, figsize=(11, 4 * len(frames)),
                             squeeze=False)
    for row, (family, df) in enumerate(frames.items()):
        clean = df[~truncation_mask(df)[0]]
        for col, axis in enumerate(("h", "w")):
            ax = axes[row][col]
            ax.scatter(clean["lat"], clean[axis], s=2, alpha=0.08,
                       edgecolors="none")
            binned = clean.groupby(pd.cut(clean["lat"], 30), observed=True)[axis].mean()
            centres = [iv.mid for iv in binned.index]
            ax.plot(centres, binned.values, color="crimson", lw=1.6)
            ax.set_xlabel("patch centre latitude (deg)")
            ax.set_ylabel(f"native crop {axis} (px)")
            ax.set_title(f"{family} — {axis}")
            ax.grid(alpha=0.25)
    fig.suptitle("Native crop size against latitude (red = binned mean)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── Reporting ────────────────────────────────────────────────────────────────

def markdown_report(payload: dict) -> str:
    meta = payload["meta"]
    lines = [
        "### Task 1.75.2 — Crop geometry and the latitude question",
        "",
        f"Native crop shapes from npy headers only (no pixel data read), "
        f"{meta['source']}. Generated {meta['generated']}.",
        "",
        "#### Tile CRS — the degree-grid hypothesis, checked rather than inferred",
        "",
        "| family | CRS of an actual tile |",
        "|---|---|",
    ]
    for family, crs in payload["tile_crs"].items():
        lines.append(f"| `{family}` | `{crs}` |")

    lines += [
        "",
        "#### Native crop shapes",
        "",
        "| family | n | h range | w range | most common (top 4) | truncated |",
        "|---|---|---|---|---|---|",
    ]
    for family, r in payload["families"].items():
        top = ", ".join(f"{k} ({v:,})" for k, v in list(r["shape_counts"].items())[:4])
        lines.append(
            f"| `{family}` | {r['n_patches']:,} | {r['h_range'][0]}-{r['h_range'][1]} | "
            f"{r['w_range'][0]}-{r['w_range'][1]} | {top} | "
            f"{r['n_truncated']:,} ({r['frac_truncated']:.3%}) |"
        )

    lines += [
        "",
        f"`truncated` = below {TRUNCATION_RATIO:.2f} x the family's own median "
        "native size; excluded from the correlations below, since a coverage "
        "failure is not the drift being measured and a handful of 12 px crops "
        "would dominate a rank correlation.",
        "",
        "#### Spearman correlations",
        "",
        "| family | h~lat | w~lat | h~\\|lat\\| | w~\\|lat\\| | h~merid. dist | "
        "w~merid. dist | **res_geo~\\|lat\\|** |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for family, r in payload["families"].items():
        s = r["spearman"]
        lines.append(
            f"| `{family}` | {s['h_vs_lat']['rho']:+.3f} | {s['w_vs_lat']['rho']:+.3f} | "
            f"{s['h_vs_abs_lat']['rho']:+.3f} | {s['w_vs_abs_lat']['rho']:+.3f} | "
            f"{s['h_vs_dist_meridian']['rho']:+.3f} | "
            f"{s['w_vs_dist_meridian']['rho']:+.3f} | "
            f"**{s['res_geo_vs_abs_lat']['rho']:+.3f}** |"
        )

    lines += [
        "",
        "`h` and `w` move in **opposite** directions with latitude, so neither "
        "axis alone answers whether ground sampling distance changes. "
        "`res_geo` = sqrt(h*w) * 10 / 32 is the scale-invariant summary and is "
        "the column to read for that question.",
    ]

    lines += ["", "#### Mean crop size by absolute latitude band", ""]
    for family, r in payload["families"].items():
        lines += [
            f"`{family}`:",
            "",
            "| \\|lat\\| band | n | mean h | mean w | eff. res h | eff. res w | "
            "**res_geo (m/px)** |",
            "|---|---|---|---|---|---|---|",
        ]
        for b in r["by_latitude_band"]:
            lines.append(
                f"| {b['band']} | {b['n']:,} | {b['h']:.2f} | {b['w']:.2f} | "
                f"{b['res_h']:.3f} | {b['res_w']:.3f} | **{b['res_geo']:.3f}** |"
            )
        lines.append("")

    lines += [
        "#### Effective resolution — native extent / 32, the ground scale the model sees",
        "",
        "| family | mean h | mean w | **mean res_geo** | city-mean range h | "
        "city-mean range w | **city-mean range res_geo** |",
        "|---|---|---|---|---|---|---|",
    ]
    for family, r in payload["families"].items():
        e = r["effective_resolution_m"]
        spread = (e["city_mean_range_geo"][1] / e["city_mean_range_geo"][0]) - 1
        lines.append(
            f"| `{family}` | {e['h']['mean']:.3f} | {e['w']['mean']:.3f} | "
            f"**{e['geo']['mean']:.3f}** | "
            f"{e['city_mean_range_h'][0]:.3f}-{e['city_mean_range_h'][1]:.3f} | "
            f"{e['city_mean_range_w'][0]:.3f}-{e['city_mean_range_w'][1]:.3f} | "
            f"**{e['city_mean_range_geo'][0]:.3f}-{e['city_mean_range_geo'][1]:.3f}** "
            f"({spread:+.1%}) |"
        )

    primary = GEOMETRY_FAMILIES[0]
    lines += [
        "",
        f"#### Per-city crop geometry (`{primary}`, sorted by latitude)",
        "",
        "**Bold** = one of the 10 held-out cultural-split cities.",
        "",
        "| city | lat | n | mean h | sd h | mean w | sd w | eff. res h | "
        "eff. res w | res_geo |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in payload["families"][primary]["by_city"]:
        name = f"**{c['city']}**" if c["held_out"] else c["city"]
        sd_h = c["h_std"] if c["h_std"] == c["h_std"] else 0.0
        sd_w = c["w_std"] if c["w_std"] == c["w_std"] else 0.0
        lines.append(
            f"| {name} | {c['lat']:+.2f} | {c['n']:,} | {c['h']:.2f} | {sd_h:.2f} | "
            f"{c['w']:.2f} | {sd_w:.2f} | {c['res_h']:.3f} | {c['res_w']:.3f} | "
            f"{c['res_geo']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 1.75.2 — crop geometry, latitude and effective resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--bounds-csv", type=Path,
                   default=Path(__file__).resolve().parents[2] / "data"
                   / "so2sat_guppd_bounds.csv")
    p.add_argument("--year", default="2017")
    p.add_argument("--split", default="training", choices=["training", "validation", "testing"])
    p.add_argument("--families", nargs="+", default=GEOMETRY_FAMILIES,
                   choices=list(FAMILIES))
    p.add_argument("--invalid-frac-parquet", type=Path,
                   default=Path("diagnostics/invalid_fraction.parquet"),
                   help="Task 1.75.1 output. Used when present (full population, "
                        "free); otherwise headers are scanned for --n-sample patches.")
    p.add_argument("--n-sample", type=int, default=5000,
                   help="Header-scan sample size when no parquet is available.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path,
                   default=Path("diagnostics/crop_geometry.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/crop_geometry.md"))
    p.add_argument("--output-png", type=Path,
                   default=Path("diagnostics/crop_geometry.png"))
    args = p.parse_args()

    meta, meta_info = load_patch_metadata(args.so2sat_dir, args.bounds_csv)
    meta = meta[meta["dataset"] == args.split]
    zone, meridian = utm_zone_and_meridian(meta["lon"].to_numpy())
    meta = meta.assign(utm_zone=zone, central_meridian=meridian)

    use_parquet = args.invalid_frac_parquet.exists()
    source = (f"full {args.split} population from "
              f"{args.invalid_frac_parquet.name}" if use_parquet
              else f"{args.n_sample} patches per family, seed {args.seed}")
    logger.info(f"Shape source: {source}")

    frames: dict[str, pd.DataFrame] = {}
    results: dict[str, dict] = {}
    for family in args.families:
        shapes = (shapes_from_parquet(args.invalid_frac_parquet, family, args.split)
                  if use_parquet else
                  shapes_from_headers(args.so2sat_dir, family, args.split,
                                      args.year, args.n_sample, args.seed))
        df = shapes.merge(meta, on=["dataset", "patch_id"], how="inner").dropna(
            subset=["city"])
        frames[family] = df
        results[family] = analyse(df)
        r = results[family]
        logger.info(
            f"{family}: n={r['n_analysed']:,} "
            f"rho(w,lat)={r['spearman']['w_vs_lat']['rho']:+.3f} "
            f"rho(h,|lat|)={r['spearman']['h_vs_abs_lat']['rho']:+.3f} "
            f"eff.res h {r['effective_resolution_m']['h']['mean']:.3f} m/px, "
            f"city range "
            f"{r['effective_resolution_m']['city_mean_range_h'][0]:.3f}-"
            f"{r['effective_resolution_m']['city_mean_range_h'][1]:.3f}"
        )

    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": args.split, "year": args.year, "source": source,
            "pixel_metres": PIXEL_METRES, "truncation_ratio": TRUNCATION_RATIO,
            **meta_info,
        },
        "tile_crs": tile_crs_check(),
        "families": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # default=str: the per-city rows come straight out of pandas and carry
    # numpy scalars, which json cannot serialise on its own.
    args.output_json.write_text(json.dumps(payload, indent=2, default=str))
    args.output_md.write_text(markdown_report(payload))
    scatter_png(frames, args.output_png)
    logger.info(f"Wrote {args.output_json}, {args.output_md} and {args.output_png}")


if __name__ == "__main__":
    main()
