"""Typer CLI for the WUDAPT label source.

    python -m lcz_wudapt ingest                 # H1: clean + spatial AOI assignment
    python -m lcz_wudapt audit                  # G0: the label-only gate
    python -m lcz_wudapt inventory              # So2Sat sparsity / leakage report
    python -m lcz_wudapt build --aoi <key>      # H3/H5/H6: WRITE the harmonized labels
    python -m lcz_wudapt patches                # H7: combine 320 m patches + review table

Stage 0 needs no embeddings and no GPU. ``ingest`` caches on the ingest hash, so
re-running after a quality- or consensus-threshold change is free.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

import geopandas as gpd

from .audit import AUDIT_CITIES, run_audit, so2sat_aoi_map, so2sat_raster
from .config import WudaptConfig
from .consensus import consensus_for_aoi
from .export import consensus_regions, merge_so2sat, write_stage8
from .ingest import ingest as run_ingest
from .leakage import (
    SO2SAT_TEST_CITIES,
    assert_no_test_city_admitted,
    city_label_inventory,
    sparse_cities,
)
from .patch_bridge import region_centred_patches
from .quality import apply_gates, polygon_weights

app = typer.Typer(add_completion=False, help=__doc__)

__all__ = ["app", "load_config"]


def load_config(config_path: Path | None) -> WudaptConfig:
    return WudaptConfig.from_yaml(config_path) if config_path else WudaptConfig()


@app.command()
def ingest(
    config_path: Path = typer.Option(None, "--config", help="WudaptConfig YAML"),
    force: bool = typer.Option(False, "--force", help="Rebuild even if cached"),
) -> None:
    """H1 — read, clean and spatially assign the LCZ-Generator submissions."""
    cfg = load_config(config_path)
    clean_path, index_path = run_ingest(cfg, force=force)
    idx = pd.read_parquet(index_path)
    typer.echo(f"clean:  {clean_path}")
    typer.echo(f"index:  {index_path}  ({len(idx):,} AOIs)")
    typer.echo(f"AOIs >=100 polys: {int((idx.n_polys >= 100).sum()):,} | "
               f">=300: {int((idx.n_polys >= 300).sum()):,}")


@app.command()
def audit(
    config_path: Path = typer.Option(None, "--config"),
    city: list[str] = typer.Option(None, "--city", help="Repeatable; default = the 8 audit cities"),
    res_m: float = typer.Option(20.0, "--res-m", help="Audit grid resolution"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(None, "--out", help="Write the report parquet here"),
) -> None:
    """G0 — the label-only gate (consensus vs naive union, LOAO, confidence sweep)."""
    cfg = load_config(config_path)
    df = run_audit(cfg, cities=list(city) if city else AUDIT_CITIES, res_m=res_m, seed=seed)
    cols = ["city", "n_polys", "n_annotators", "n_submissions", "hard_frac", "mean_conf",
            "mean_n_eff", "oa_consensus", "oa_naive_union", "loao_consensus",
            "loao_random_peer", "conf_monotone"]
    typer.echo(df[cols].round(3).to_string(index=False))

    margin = (df.oa_consensus - df.oa_naive_union).mean()
    loao = (df.loao_consensus - df.loao_random_peer).mean()
    typer.echo(f"\nG0.3 consensus - naive union : {margin:+.3f} (gate >= +0.03)")
    typer.echo(f"G0.2 LOAO - random peer      : {loao:+.3f} (gate >= +0.08)")
    typer.echo(f"G0.5 confidence monotone     : {int(df.conf_monotone.sum())}/{len(df)} cities")

    out = out or (Path(cfg.cache_dir) / f"g0_audit_{cfg.config_hash}.parquet")
    df.to_parquet(out)
    typer.echo(f"\nwrote {out}")


@app.command()
def inventory(config_path: Path = typer.Option(None, "--config")) -> None:
    """So2Sat per-city label inventory + which cities may admit WUDAPT."""
    cfg = load_config(config_path)
    inv = city_label_inventory(cfg)
    admitted = sparse_cities(inv, cfg)
    assert_no_test_city_admitted(admitted)
    typer.echo(inv.to_string(index=False))
    typer.echo(f"\nsparse (WUDAPT admitted): {admitted}")
    typer.echo(f"held-out test cities (never admitted): {list(inv[inv.is_test_city].city)}")


@app.command()
def build(
    config_path: Path = typer.Option(None, "--config"),
    aoi: list[str] = typer.Option(None, "--aoi", help="AOI key(s); repeatable"),
    min_polys: int = typer.Option(300, "--min-polys", help="Build every AOI with >= this many"),
    res_m: float = typer.Option(None, "--res-m", help="Raster resolution (default: config)"),
    target_year: int = typer.Option(None, "--target-year", help="Embedding epoch to weight toward"),
    limit: int = typer.Option(None, "--limit", help="Cap the number of AOIs built"),
    force: bool = typer.Option(False, "--force", help="Rebuild AOIs that already have an export"),
    patches: bool = typer.Option(True, "--patches/--no-patches",
                                 help="Also emit region-centred 320 m patches per AOI"),
    so2sat_policy: str = typer.Option("gap-fill", "--so2sat-policy",
                                      help="sparse-only | gap-fill (never includes held-out test cities)"),
) -> None:
    """H3/H5/H6 — harmonize and WRITE the Stage 8 label rasters + block parquet."""
    cfg = load_config(config_path)
    clean_path = Path(cfg.cache_dir) / f"wudapt_clean_{cfg.ingest_hash}.parquet"
    if not clean_path.exists():
        raise typer.BadParameter(f"run `lcz_wudapt ingest` first ({clean_path} missing)")
    gdf = gpd.read_parquet(clean_path)

    index = pd.read_parquet(Path(cfg.cache_dir) / f"aoi_index_{cfg.ingest_hash}.parquet")
    targets = list(aoi) if aoi else index.loc[index.n_polys >= min_polys, "aoi"].tolist()
    if limit:
        targets = targets[:limit]

    # So2Sat is authoritative wherever it exists (merge_so2sat burns it last), so
    # the only question is which So2Sat cities WUDAPT may ADD to.
    #   sparse-only: just the 12 degenerate cities (Salvador has 1 patch).
    #   gap-fill:    every non-test So2Sat city. Measured, WUDAPT labels ground
    #                So2Sat never touches - Wuhan 908 km2 vs 286 km2, overlapping
    #                on only 105 km2 - so 77-95% of what it adds is on unlabelled
    #                ground rather than in conflict.
    # Held-out test cities are excluded under BOTH policies, unconditionally.
    amap = so2sat_aoi_map(cfg)
    admitted = set(sparse_cities(city_label_inventory(cfg), cfg))
    assert_no_test_city_admitted(sorted(admitted))
    if so2sat_policy not in {"sparse-only", "gap-fill"}:
        raise typer.BadParameter("--so2sat-policy must be sparse-only or gap-fill")
    test_cities = {c.replace(" ", "_") for c in SO2SAT_TEST_CITIES} | set(SO2SAT_TEST_CITIES)

    built, skipped = [], []
    for key in targets:
        sub = gdf[gdf["aoi"] == key]
        city = amap.get(key)
        if city:
            is_test = city in test_cities or city.replace("_", " ") in test_cities
            if is_test:
                skipped.append((key, "held-out So2Sat test city"))
                continue
            is_sparse = city in admitted or city.replace("_", " ") in admitted
            if so2sat_policy == "sparse-only" and not is_sparse:
                skipped.append((key, "So2Sat city with sufficient labels"))
                continue
        if sub.empty:
            skipped.append((key, "no polygons"))
            continue
        # Idempotent by default: a build of hundreds of AOIs takes hours, so a
        # re-run after an interruption must resume rather than start over.
        if not force and (Path(cfg.cache_dir) / key / f"blocks_labelled_{key}.parquet").exists():
            skipped.append((key, "already built (--force to rebuild)"))
            continue
        try:
            gated = apply_gates(sub, cfg)
            if gated.empty:
                skipped.append((key, "all polygons gated out"))
                continue
            w = polygon_weights(gated, cfg, target_year=target_year)["weight"].to_numpy()
            res = consensus_for_aoi(gated, w, cfg, res_m=res_m)
            if len(res) == 0:
                skipped.append((key, "no pixel reached the decision thresholds"))
                continue
            truth = so2sat_raster(city, res.grid, cfg) if city else None
            bitmask, conf, source = merge_so2sat(res, truth)
            from .export import _dense
            extra = {"n_eff": _dense(res, res.n_eff, "float32"),
                     "p_top": _dense(res, res.p_top, "float32")}
            regions = consensus_regions(bitmask, conf, source, res.grid, key, cfg, extra=extra)
            if regions.empty:
                skipped.append((key, "no region cleared the minimum area"))
                continue
            write_stage8(regions, bitmask, conf, res.grid, key, cfg)
            if patches:
                pg = region_centred_patches(key, cfg)
                if len(pg):
                    pg.to_parquet(Path(cfg.cache_dir) / key / f"patches_{key}.parquet")
            built.append(key)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append((key, str(exc)))

    typer.echo(f"\nbuilt {len(built)} AOIs -> {cfg.cache_dir}")
    for k, why in skipped[:10]:
        typer.echo(f"  skipped {k}: {why}")
    if len(skipped) > 10:
        typer.echo(f"  ... and {len(skipped) - 10} more skipped")


@app.command()
def patches(
    config_path: Path = typer.Option(None, "--config"),
    out: Path = typer.Option(None, "--out", help="Combined gpkg path"),
) -> None:
    """H7 — combine per-AOI 320 m patches into one pseudo-label gpkg + review table."""
    cfg = load_config(config_path)
    root = Path(cfg.cache_dir)
    frames = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / f"patches_{d.name}.parquet"
        if f.exists():
            frames.append(gpd.read_parquet(f))
    if not frames:
        raise typer.BadParameter("no per-AOI patch files; run `lcz_wudapt build` first")

    g = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    # patch_id restarts per AOI, so re-issue globally: build_patch_index keys on
    # (dataset, patch_id) and a collision would silently merge two patches.
    g["patch_id"] = [f"{i:07d}" for i in range(len(g))]
    g["dataset"] = "unlabeled"
    cols = ["patch_id", "dataset", "LCZ_class", "weight", "aoi", "dominant_frac",
            "labelled_frac", "block_kind", "geometry"]
    g = g[[c for c in cols if c in g.columns]]
    out = out or (root / "patches_wudapt_region_centred.gpkg")
    g.to_file(out, driver="GPKG")

    review = (g.groupby("aoi")
                .agg(patches=("patch_id", "size"), classes=("LCZ_class", "nunique"),
                     mean_purity=("dominant_frac", "mean"), mean_weight=("weight", "mean"))
                .reset_index().sort_values("patches", ascending=False))
    review.to_parquet(root / "patch_review.parquet", index=False)

    typer.echo(f"{len(g):,} patches from {g.aoi.nunique()} AOIs, {g.LCZ_class.nunique()} classes")
    typer.echo(f"mean purity {g.dominant_frac.mean():.3f} | mean weight {g.weight.mean():.3f}")
    typer.echo(f"AOIs with >=50 patches: {int((review.patches >= 50).sum())}")
    typer.echo(f"\nwrote {out}\nwrote {root / 'patch_review.parquet'}")


if __name__ == "__main__":  # pragma: no cover
    app()
