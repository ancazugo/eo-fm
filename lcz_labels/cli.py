"""Stage 8 — Typer CLI orchestrating the pipeline, with per-stage caching.

    python -m lcz_labels extract  --aoi Nairobi
    python -m lcz_labels ucp      --aoi Nairobi
    python -m lcz_labels label    --aoi Nairobi
    python -m lcz_labels mask     --aoi Nairobi
    python -m lcz_labels validate --aoi Nairobi --aoi London --aoi Milan
    python -m lcz_labels all      --all-aois

Every stage caches per AOI keyed on the config hash; ``--force`` rebuilds. Pass
``--config path.yaml`` to override defaults (thresholds, Overture release, paths).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import pandas as pd
import typer
from loguru import logger

from .change_mask import compute_change_mask
from .classify import classify_patches
from .config import LczLabelConfig
from .grid import load_grid
from .overture import extract_overture
from .ucp import compute_ucp
from .validate import validate_labels, write_report

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False,
                  help="Overture/OSM-fused LCZ pseudo-labelling.")

# Output column order (task Stage 8) followed by UCP/diagnostic columns.
_LEAD_COLS = ["patch_id", "aoi", "label_type", "lcz", "lcz_set", "lcz_name", "confidence",
              "stable_2017_to_label_year", "change_score", "label_year",
              "overture_release", "config_hash", "so2sat_lcz"]


@contextmanager
def _stage(name: str):
    t0 = time.perf_counter()
    logger.info(f"▶ {name}")
    yield
    logger.info(f"✔ {name} ({time.perf_counter() - t0:.1f}s)")


def load_config(config_path: str | None) -> LczLabelConfig:
    if config_path:
        logger.info(f"Loading config {config_path}")
        return LczLabelConfig.from_yaml(config_path)
    return LczLabelConfig()


def _resolve_aois(config: LczLabelConfig, aoi: list[str] | None, all_aois: bool) -> list[str]:
    if all_aois:
        if not config.cities_dir.is_dir():
            raise typer.BadParameter(f"cities dir not found: {config.cities_dir}")
        return sorted(d.name for d in config.cities_dir.iterdir() if d.is_dir())
    if not aoi:
        raise typer.BadParameter("provide --aoi <name> (repeatable) or --all-aois")
    return list(aoi)


def build_labels(
    aoi_name: str, config: LczLabelConfig, *, force: bool = False, run_mask: bool = True
) -> gpd.GeoDataFrame:
    """Run grid → overture → ucp → classify (→ mask) and assemble the label GDF."""
    with _stage(f"[{aoi_name}] grid"):
        grid = load_grid(aoi_name, config, force=force)
    with _stage(f"[{aoi_name}] overture extract"):
        extract = extract_overture(aoi_name, config, force=force)
    with _stage(f"[{aoi_name}] ucp"):
        ucp = compute_ucp(grid, extract, config, aoi_name, force=force)
    with _stage(f"[{aoi_name}] classify"):
        cls = classify_patches(ucp, config)

    # Assemble: grid geometry + ucp + classification
    gdf = grid.reset_index(drop=True).copy()
    gdf = gdf.rename(columns={"LCZ_class": "so2sat_lcz"})
    ucp_cols = [c for c in ucp.columns if c not in ("patch_id", "dataset")]
    for c in ucp_cols:
        gdf[c] = ucp[c].to_numpy()
    for c in cls.columns:
        gdf[c] = cls[c].to_numpy()

    if run_mask:
        with _stage(f"[{aoi_name}] change mask"):
            change = compute_change_mask(grid, config, aoi_name, force=force)
        gdf["stable_2017_to_label_year"] = change["stable_2017_to_label_year"].to_numpy()
        gdf["change_score"] = change["change_score"].to_numpy()
    else:
        gdf["stable_2017_to_label_year"] = False
        gdf["change_score"] = float("nan")

    gdf["label_year"] = config.label_year
    gdf["overture_release"] = config.overture_release
    gdf["config_hash"] = config.config_hash

    lead = [c for c in _LEAD_COLS if c in gdf.columns]
    rest = [c for c in gdf.columns if c not in lead and c != "geometry"]
    gdf = gdf[lead + rest + ["geometry"]]

    out = config.cache_dir / aoi_name / f"labels_{aoi_name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out)
    n_hard = int((gdf["label_type"] == "hard").sum())
    n_coarse = int((gdf["label_type"] == "coarse").sum())
    logger.info(f"[{aoi_name}] wrote {out} ({len(gdf)} rows, "
                f"{n_hard} hard incl. {int((gdf['lcz'] == 7).sum())} LCZ-7, {n_coarse} coarse)")
    return gdf


def to_training_pairs(labels: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Expand labels into (patch_id, dataset, label_type, lcz, lcz_set, year) pairs.

    Stable patches (``stable_2017_to_label_year``) are paired with every requested
    embedding year; unstable ones only with their own ``label_year`` (epoch-locked
    — labels must not travel across years where the built environment changed).

    CONSUMPTION CONTRACT (do not collapse a coarse label to one member!):
      * ``label_type == "hard"``  -> standard cross-entropy on the single ``lcz``.
      * ``label_type == "coarse"`` -> marginalised cross-entropy over ``lcz_set``,
        i.e. loss = ``-log Σ_{c ∈ lcz_set} p_c`` (the true class is known to be one
        of the set members, e.g. {3,7} or {8,10}, but not which).
    ``lcz_set`` is passed through untouched; ``lcz`` is null for coarse rows.
    """
    lab = labels[labels["label_type"].isin(["hard", "coarse"])].copy()
    rows = []
    for r in lab.to_dict("records"):
        yrs = years if r.get("stable_2017_to_label_year") else [int(r["label_year"])]
        for y in yrs:
            rows.append({
                "patch_id": r["patch_id"], "dataset": r.get("dataset"),
                "label_type": r["label_type"],
                "lcz": (int(r["lcz"]) if pd.notna(r["lcz"]) else None),
                "lcz_set": list(r["lcz_set"]),
                "confidence": r.get("confidence"), "year": y,
            })
    return pd.DataFrame(rows)


# ── Commands ──────────────────────────────────────────────────────────────────

_AOI_OPT = typer.Option(None, "--aoi", help="AOI name (repeatable)")
_ALL_OPT = typer.Option(False, "--all-aois", help="Process every So2Sat city")
_CFG_OPT = typer.Option(None, "--config", help="YAML config path")
_FORCE_OPT = typer.Option(False, "--force", help="Rebuild caches")


@app.command()
def extract(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
            config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 2: extract Overture features (and grid) per AOI."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        with _stage(f"[{name}] grid"):
            load_grid(name, cfg, force=force)
        with _stage(f"[{name}] overture extract"):
            extract_overture(name, cfg, force=force)


@app.command()
def ucp(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
        config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 4: compute UCP table per AOI."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        grid = load_grid(name, cfg, force=force)
        ex = extract_overture(name, cfg, force=force)
        compute_ucp(grid, ex, cfg, name, force=force)


@app.command()
def label(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
          config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stages 5-6: classify to LCZ + confidence (no temporal mask)."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        build_labels(name, cfg, force=force, run_mask=False)


@app.command()
def mask(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
         config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 7: temporal stability mask per AOI."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        grid = load_grid(name, cfg, force=force)
        compute_change_mask(grid, cfg, name, force=force)


@app.command()
def validate(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
             config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 9: agreement vs So2Sat; writes a markdown report."""
    cfg = load_config(config)
    names = _resolve_aois(cfg, aoi, all_aois)
    results = []
    for name in names:
        lab_path = cfg.cache_dir / name / f"labels_{name}.parquet"
        gdf = gpd.read_parquet(lab_path) if lab_path.exists() else build_labels(name, cfg, force=force)
        results.append(validate_labels(pd.DataFrame(gdf.drop(columns="geometry")), name))
    report = cfg.cache_dir / f"validation_{'_'.join(names[:3])}.md"
    write_report(results, report)


@app.command(name="all")
def run_all(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
            config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Run the full pipeline for each AOI, then merge + validate."""
    cfg = load_config(config)
    names = _resolve_aois(cfg, aoi, all_aois)
    all_gdfs, results, failed = [], [], []
    for i, name in enumerate(names, 1):
        logger.info(f"══ AOI {i}/{len(names)}: {name} ══")
        try:
            gdf = build_labels(name, cfg, force=force, run_mask=True)
        except Exception as e:  # noqa: BLE001 — one bad AOI must not abort the batch
            logger.error(f"[{name}] FAILED: {e}")
            failed.append(name)
            continue
        all_gdfs.append(gdf)
        results.append(validate_labels(pd.DataFrame(gdf.drop(columns="geometry")), name))
    if all_gdfs:
        merged = pd.concat(all_gdfs, ignore_index=True)
        merged_path = cfg.cache_dir / "labels_all.parquet"
        gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326").to_parquet(merged_path)
        logger.info(f"Merged {len(merged)} rows from {len(all_gdfs)} AOIs -> {merged_path}")
    if any(results):
        write_report(results, cfg.cache_dir / "validation_all.md")
    logger.info(f"Batch done: {len(all_gdfs)} ok, {len(failed)} failed"
                + (f" ({', '.join(failed)})" if failed else ""))


if __name__ == "__main__":
    app()
