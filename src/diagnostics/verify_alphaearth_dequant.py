"""Task 0.2 — verify the AlphaEarth coop int8 dequantization.

AlphaEarth embeddings are unit-norm 64-d vectors, so the per-pixel L2 norm
after a CORRECT decoding must be ~1.0. That test is decisive on its own and
needs no external reference data.

Candidate decodings of the stored int8 values:

  1. ``((v/127.5)**2) * sign(v)``            — what the codebase currently does
     (``dequantize_embeddings.dequantize_alphaearth_embeddings``)
  2. ``v/127.5``                             — plain linear
  3. ``sign(v)*(|v|/127.5)`` then per-pixel L2 renormalization

Optionally cross-checks against the GEE float32 AlphaEarth tiles on disk, by
cropping the same patch geometries with the same
``datasets.tiles.build_tile_index`` / ``crop_patch`` used by
``extract_so2sat_embeddings.py``.

Measure-only: nothing is written back into the pipeline.

Example:

    python src/diagnostics/verify_alphaearth_dequant.py \\
        --n-sample 5000 --seed 0 \\
        --gee-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017 --gee-n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.embedding_stats import (  # noqa: E402
    build_id_index,
    load_raw,
    sample_paired_ids,
)
from utils.constants import DATA_DIR  # noqa: E402

FAMILY = "alpha_earth_coop"


# ── Candidate decodings ──────────────────────────────────────────────────────

def cand_current(v: np.ndarray) -> np.ndarray:
    """((v/127.5)**2) * sign(v) — the decoding the codebase applies today."""
    return ((v / 127.5) ** 2) * np.sign(v)


def cand_linear(v: np.ndarray) -> np.ndarray:
    """v/127.5 — plain linear."""
    return v / 127.5


def cand_linear_renorm(v: np.ndarray) -> np.ndarray:
    """sign(v)*(|v|/127.5) then per-pixel L2 renormalization over channels."""
    x = np.sign(v) * (np.abs(v) / 127.5)
    n = np.sqrt((x.astype(np.float64) ** 2).sum(axis=0, keepdims=True))
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


CANDIDATES = {
    "current_sq_sign": cand_current,
    "linear": cand_linear,
    "linear_renorm": cand_linear_renorm,
}


class PairAccumulator:
    """Streaming sums for a Pearson r between two decodings over all values."""

    def __init__(self) -> None:
        self.n = 0
        self.sa = self.sb = self.saa = self.sbb = self.sab = 0.0

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        self.n += a.size
        self.sa += a.sum()
        self.sb += b.sum()
        self.saa += (a * a).sum()
        self.sbb += (b * b).sum()
        self.sab += (a * b).sum()

    def r(self) -> float:
        n = self.n
        cov = self.sab / n - (self.sa / n) * (self.sb / n)
        va = self.saa / n - (self.sa / n) ** 2
        vb = self.sbb / n - (self.sb / n) ** 2
        return float(cov / np.sqrt(max(va * vb, 1e-30)))


def norm_summary(norms: np.ndarray) -> dict:
    return {
        "n": int(norms.size),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "p1": float(np.percentile(norms, 1)),
        "p50": float(np.percentile(norms, 50)),
        "p99": float(np.percentile(norms, 99)),
        "frac_within_1pct_of_1": float(np.mean(np.abs(norms - 1.0) < 0.01)),
    }


# ── Norm test ────────────────────────────────────────────────────────────────

def run_norm_test(ids: list[str], index: dict[str, Path]) -> dict:
    norms: dict[str, list[np.ndarray]] = {k: [] for k in CANDIDATES}
    allzero: list[np.ndarray] = []
    pairs = {
        "current_sq_sign|linear": PairAccumulator(),
        "current_sq_sign|linear_renorm": PairAccumulator(),
        "linear|linear_renorm": PairAccumulator(),
    }
    n_values = 0
    n_int8_min = 0          # v == -128, a plausible nodata sentinel
    n_nan = 0
    vmin, vmax = np.inf, -np.inf
    all_integer = True

    for pid in tqdm(ids, desc="norm test", unit="patch"):
        raw = load_raw(index[pid], "npy")          # (64, H, W) int8 values as float32
        n_values += raw.size
        n_nan += int(np.isnan(raw).sum())
        n_int8_min += int((raw == -128).sum())
        vmin = min(vmin, float(raw.min()))
        vmax = max(vmax, float(raw.max()))
        if all_integer and not np.array_equal(raw, np.round(raw)):
            all_integer = False

        raw = np.nan_to_num(raw, nan=0.0)
        allzero.append((raw == 0).all(axis=0).ravel())

        decoded = {k: fn(raw) for k, fn in CANDIDATES.items()}
        for k, x in decoded.items():
            norms[k].append(
                np.sqrt((x.astype(np.float64) ** 2).sum(axis=0)).ravel()
            )
        for key, acc in pairs.items():
            a, b = key.split("|")
            acc.update(decoded[a], decoded[b])

    zero_mask = np.concatenate(allzero)
    out: dict = {
        "n_patches": len(ids),
        "raw": {
            "min": vmin, "max": vmax,
            "all_integer_valued": bool(all_integer),
            "frac_nan": float(n_nan / n_values),
            "frac_eq_int8_min(-128)": float(n_int8_min / n_values),
        },
        "frac_allzero_pixels": float(zero_mask.mean()),
        "candidates": {},
        "pearson_r_between_candidates": {k: a.r() for k, a in pairs.items()},
    }
    for k, chunks in norms.items():
        n = np.concatenate(chunks)
        out["candidates"][k] = {
            "all_pixels": norm_summary(n),
            "nonzero_pixels": norm_summary(n[~zero_mask]),
        }
    return out


# ── Optional GEE float32 cross-check ─────────────────────────────────────────

def open_gee_zarr(path: Path):
    """Open a GEE AlphaEarth ``.zarr`` tile as a ``(band, y, x)`` DataArray.

    Deliberately does NOT go through ``datasets.tiles.open_tile``: that function
    tests ``path.is_dir()`` before ``path.suffix == '.zarr'``, and a zarr store
    IS a directory, so every ``.zarr`` tile is misrouted to the Tessera-global
    NPY reader and raises "NPY files not found". Phase 0 is measure-only, so the
    bug is reported at GATE 0 rather than fixed here; this helper replicates the
    (unreachable) ``.zarr`` branch of ``open_tile``.
    """
    import xarray as xr

    ds = xr.open_zarr(str(path), chunks=False)
    da = ds["embedding"]
    if da.dims != ("band", "y", "x"):
        da = da.transpose("band", "y", "x")
    crs = None
    if "spatial_ref" in ds:
        crs = ds["spatial_ref"].attrs.get("crs_wkt")
    return da, crs


def crop_gee(da, bounds: tuple[float, float, float, float]) -> np.ndarray | None:
    """Clip a GEE tile to *bounds* (tile CRS) and return north-up (C, H, W)."""
    w, s, e, n = bounds
    x, y = da.x.values, da.y.values
    xs = slice(w, e) if x[0] <= x[-1] else slice(e, w)
    ys = slice(s, n) if y[0] <= y[-1] else slice(n, s)
    sub = da.sel(x=xs, y=ys)
    if sub.sizes["x"] == 0 or sub.sizes["y"] == 0:
        return None
    arr = sub.values.astype(np.float32)
    if y[0] <= y[-1]:                       # south-up tile → flip to north-up
        arr = arr[:, ::-1, :]
    return arr


def run_gee_check(
    so2sat_dir: Path,
    gee_dir: Path,
    year: str,
    coop_index: dict[str, Path],
    n_patches: int,
    seed: int,
) -> dict:
    """Compare the coop decodings against the GEE float32 tiles.

    Tile discovery reuses ``datasets.tiles.build_tile_index`` (filename-based,
    so unaffected by the ``open_tile`` bug); the clip is done here because the
    two products sit on different pixel grids, so per-channel patch means are
    the robust comparison and a per-pixel r is only reported when the crops
    happen to come out the same shape.
    """
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    from datasets.tiles import build_tile_index

    gpkg = so2sat_dir / "patches_reference_rxr.gpkg"
    gdf = gpd.read_file(gpkg)
    gdf = gdf[gdf["dataset"] == "training"]
    gdf = gdf[gdf["patch_id"].astype(str).isin(coop_index)]

    tile_paths, tree = build_tile_index(gee_dir, "alpha_earth", year=year)
    logger.info(f"GEE tile index: {len(tile_paths)} tiles from {gee_dir}")

    per_channel_pairs: dict[str, list[np.ndarray]] = {k: [] for k in CANDIDATES}
    gee_means: list[np.ndarray] = []
    pixel_pairs: dict[str, PairAccumulator] = {k: PairAccumulator() for k in CANDIDATES}
    gee_norms: list[np.ndarray] = []
    tile_cache: dict[Path, tuple] = {}
    transformers: dict[str, Transformer] = {}
    n_ok = 0
    n_no_cover = 0
    n_shape_mismatch = 0
    n_sentinel = 0

    rng = np.random.default_rng(seed)
    # Draw a pool at random, then order it by covering tile: each GEE tile is a
    # ~43 MB zarr store, so a random order re-opens one per patch (~20 s each)
    # while a tile-grouped order opens each tile once.
    pool = rng.permutation(len(gdf))[: n_patches * 6]
    keyed: list[tuple[int, int]] = []
    for i in pool:
        idxs = tree.query(gdf.iloc[int(i)].geometry)
        if len(idxs) == 0:
            n_no_cover += 1
            continue
        keyed.append((int(min(idxs)), int(i)))
    keyed.sort()
    order = [i for _, i in keyed]
    logger.info(
        f"GEE cross-check pool: {len(order)} patches over "
        f"{len({t for t, _ in keyed})} tiles"
    )

    pbar = tqdm(total=n_patches, desc="GEE cross-check", unit="patch")
    for i in order:
        if n_ok >= n_patches:
            break
        row = gdf.iloc[int(i)]
        pid = str(row["patch_id"])
        idxs = tree.query(row.geometry)
        if len(idxs) == 0:
            n_no_cover += 1
            continue

        gee = None
        for j in idxs:                          # first tile that actually covers it
            tp = tile_paths[int(j)]
            try:
                if tp not in tile_cache:
                    if len(tile_cache) > 4:
                        tile_cache.clear()
                    tile_cache[tp] = open_gee_zarr(tp)
                da, crs = tile_cache[tp]
            except Exception as e:              # noqa: BLE001
                logger.debug(f"{tp.name}: open failed ({e})")
                continue
            geom = row.geometry
            if crs is not None:
                if crs not in transformers:
                    transformers[crs] = Transformer.from_crs(
                        "EPSG:4326", crs, always_xy=True
                    )
                geom = shapely_transform(transformers[crs].transform, geom)
            gee = crop_gee(da, geom.bounds)
            if gee is not None:
                break
        if gee is None:
            n_no_cover += 1
            continue

        coop_raw = np.nan_to_num(load_raw(coop_index[pid], "npy"), nan=0.0)
        if gee.shape[0] != coop_raw.shape[0]:
            n_shape_mismatch += 1
            continue
        if (coop_raw == -128).any():
            # Patches carrying the all-channel -128 nodata sentinel are skipped:
            # they have no counterpart in the GEE floats and would dominate the
            # per-channel means. They are quantified separately in the norm test.
            n_sentinel += 1
            continue

        gee_means.append(np.nanmean(gee.reshape(gee.shape[0], -1), axis=1))
        gee_norms.append(
            np.sqrt(np.nansum(gee.astype(np.float64) ** 2, axis=0)).ravel()
        )
        if gee.shape != coop_raw.shape:
            n_shape_mismatch += 1
        for k, fn in CANDIDATES.items():
            dec = fn(coop_raw)
            per_channel_pairs[k].append(dec.reshape(dec.shape[0], -1).mean(axis=1))
            if dec.shape == gee.shape:                           # per-pixel comparison
                m = np.isfinite(gee)
                pixel_pairs[k].update(dec[m], gee[m])
        n_ok += 1
        pbar.update(1)
    pbar.close()

    if n_ok == 0:
        return {"status": "no overlapping GEE tiles found", "n_no_cover": n_no_cover}

    G = np.stack(gee_means)                                       # (N, 64)
    res: dict = {
        "status": "ok",
        "n_patches_compared": n_ok,
        "n_patches_no_gee_cover": n_no_cover,
        "n_grid_shape_mismatch": n_shape_mismatch,
        "n_skipped_nodata_sentinel": n_sentinel,
        "gee_l2_norm": norm_summary(np.concatenate(gee_norms)),
        "per_candidate": {},
    }
    for k in CANDIDATES:
        C = np.stack(per_channel_pairs[k])                        # (N, 64)
        # per-channel Pearson r over patch means, and RMSE of the patch means
        r = np.array([
            np.corrcoef(C[:, c], G[:, c])[0, 1] if np.std(C[:, c]) > 0 and np.std(G[:, c]) > 0
            else np.nan
            for c in range(C.shape[1])
        ])
        rmse = float(np.sqrt(np.nanmean((C - G) ** 2)))
        res["per_candidate"][k] = {
            "patchmean_r_median": float(np.nanmedian(r)),
            "patchmean_r_min": float(np.nanmin(r)),
            "patchmean_rmse": rmse,
            "pixelwise_r": (pixel_pairs[k].r() if pixel_pairs[k].n else None),
            "pixelwise_n": int(pixel_pairs[k].n),
        }
    return res


# ── Reporting ────────────────────────────────────────────────────────────────

def markdown_report(norm: dict, gee: dict | None) -> str:
    lines = [
        "### Task 0.2 — AlphaEarth coop dequantization verification",
        "",
        f"{norm['n_patches']} training patches. AlphaEarth embeddings are unit-norm "
        "64-d vectors, so the correct decoding is the one whose per-pixel L2 norm "
        "concentrates at 1.0.",
        "",
        "| candidate decoding | L2 norm mean | std | p1 | p50 | p99 | within 1% of 1.0 |",
        "|---|---|---|---|---|---|---|",
    ]
    label = {
        "current_sq_sign": "`((v/127.5)**2)*sign(v)` — **current**",
        "linear": "`v/127.5`",
        "linear_renorm": "`sign(v)*(|v|/127.5)` + L2 renorm",
    }
    for k, c in norm["candidates"].items():
        s = c["nonzero_pixels"]
        lines.append(
            f"| {label[k]} | {s['mean']:.4f} | {s['std']:.4f} | {s['p1']:.4f} | "
            f"{s['p50']:.4f} | {s['p99']:.4f} | {s['frac_within_1pct_of_1']:.2%} |"
        )
    lines += [
        "",
        "(Non-all-zero pixels; all-zero pixels are "
        f"{norm['frac_allzero_pixels']:.2%} of the sample and decode to norm 0 "
        "under every candidate.)",
        "",
        "Pearson r between decodings over all values: "
        + ", ".join(
            f"`{k.replace('|', '` vs `')}` = {v:.4f}"
            for k, v in norm["pearson_r_between_candidates"].items()
        ),
        "",
    ]
    if gee and gee.get("status") == "ok":
        lines += [
            f"GEE float32 cross-check ({gee['n_patches_compared']} patches):",
            "",
            "| candidate | per-channel r (median) | r (min) | RMSE | pixelwise r |",
            "|---|---|---|---|---|",
        ]
        for k, v in gee["per_candidate"].items():
            px = "n/a" if v["pixelwise_r"] is None else f"{v['pixelwise_r']:.4f}"
            lines.append(
                f"| {label[k]} | {v['patchmean_r_median']:.4f} | "
                f"{v['patchmean_r_min']:.4f} | {v['patchmean_rmse']:.4f} | {px} |"
            )
        lines.append("")
    elif gee:
        lines += [f"GEE float32 cross-check: **{gee.get('status')}**", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Task 0.2 — verify the AlphaEarth coop dequantization (measure only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--split", default="training",
                   choices=["training", "validation", "testing"])
    p.add_argument("--year", default="2017")
    p.add_argument("--n-sample", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gee-dir", type=Path, default=None,
                   help="GEE float32 AlphaEarth tile dir, e.g. .../Google/AlphaEarth/2017")
    p.add_argument("--gee-n", type=int, default=200,
                   help="Patches to compare against the GEE tiles.")
    p.add_argument("--output-json", type=Path,
                   default=Path("diagnostics/alphaearth_dequant.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/alphaearth_dequant.md"))
    args = p.parse_args()

    ids, indexes = sample_paired_ids(
        args.so2sat_dir, [FAMILY], args.split, args.year, args.n_sample, args.seed,
    )
    index = indexes[FAMILY]
    logger.info(f"Sampled {len(ids)} coop patches from {args.split}")

    norm = run_norm_test(ids, index)

    gee = None
    if args.gee_dir is not None:
        full_index = build_id_index(args.so2sat_dir, FAMILY, args.split, args.year)
        gee = run_gee_check(args.so2sat_dir, args.gee_dir, args.year, full_index,
                            args.gee_n, args.seed)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "so2sat_dir": str(args.so2sat_dir),
        "split": args.split, "year": args.year,
        "n_sample": len(ids), "seed": args.seed,
        "norm_test": norm,
        "gee_cross_check": gee,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {args.output_json}")

    md = markdown_report(norm, gee)
    args.output_md.write_text(md + "\n")
    logger.info(f"Wrote {args.output_md}")
    print()
    print(md)


if __name__ == "__main__":
    main()
