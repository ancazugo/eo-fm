"""Typer CLI orchestrating the block-based pipeline, with per-stage caching.

    python -m lcz_labels extract  --aoi Nairobi
    python -m lcz_labels blocks   --aoi Nairobi
    python -m lcz_labels ucp      --aoi Nairobi
    python -m lcz_labels label    --aoi Nairobi
    python -m lcz_labels mask     --aoi Nairobi
    python -m lcz_labels export   --aoi Nairobi
    python -m lcz_labels validate --aoi Nairobi --aoi Paris
    python -m lcz_labels all      --all-aois

Stage graph per AOI: extract -> blocks (+adjacency) -> ucp -> classify ->
zones -> mask -> export (block parquet + 3 rasters + patch transfer) ->
validate. Every derivation stage caches per AOI keyed on the config hash
(``--force`` rebuilds); the Overture extraction keys on the extraction hash so
threshold changes never re-download. All joins are keyed on ``block_id`` with
``validate="1:1"`` — misalignment raises instead of silently scrambling rows.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import geopandas as gpd
import pandas as pd
import typer
from loguru import logger

from .blocks import build_adjacency, build_blocks
from .change_mask import compute_change_mask
from .classify import classify_blocks, form_zones
from .config import LczLabelConfig
from .export import patch_transfer, to_training_pairs, write_blocks_parquet, write_rasters
from .grid import load_grid
from .overture import extract_overture
from .ucp import compute_ucp
from .validate import block_homogeneity, validate_patch_transfer, write_report

__all__ = ["app", "build_block_labels", "load_config", "to_training_pairs"]

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False,
                  help="Overture/OSM-fused block-based LCZ pseudo-labelling.")


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


def build_block_labels(
    aoi_name: str, config: LczLabelConfig, *, force: bool = False, run_mask: bool = True
) -> gpd.GeoDataFrame:
    """extract -> blocks -> ucp -> classify -> zones (-> mask) -> labelled blocks."""
    with _stage(f"[{aoi_name}] overture extract"):
        extract = extract_overture(aoi_name, config, force=force)
    with _stage(f"[{aoi_name}] blocks"):
        blocks = build_blocks(aoi_name, extract, config, force=force)
        adjacency = build_adjacency(blocks, config, aoi_name, force=force)
    with _stage(f"[{aoi_name}] ucp"):
        ucp = compute_ucp(blocks, extract, config, aoi_name, force=force)
    with _stage(f"[{aoi_name}] classify"):
        cls = classify_blocks(ucp, config)
        cls.insert(0, "block_id", ucp["block_id"].to_numpy())
    with _stage(f"[{aoi_name}] zones"):
        zones = form_zones(cls, blocks, adjacency, config)

    gdf = (
        blocks.merge(ucp, on="block_id", validate="1:1")
        .merge(cls, on="block_id", validate="1:1")
        .merge(zones, on="block_id", validate="1:1")
    )
    # Stage 6 zone-grade penalty: labelled but not zone-grade -> x penalty.
    labelled = gdf["label_type"].isin(["hard", "coarse"])
    penalty = labelled & ~gdf["zone_grade"]
    gdf.loc[penalty, "confidence"] *= config.zones.zone_conf_penalty

    if run_mask:
        with _stage(f"[{aoi_name}] change mask"):
            change = compute_change_mask(blocks, config, aoi_name, force=force)
        gdf = gdf.merge(change, on="block_id", validate="1:1")
    else:
        gdf["stable_2017_to_label_year"] = False
        gdf["change_score"] = float("nan")

    gdf["label_year"] = config.label_year
    write_blocks_parquet(gdf, aoi_name, config)
    return gdf


def _load_labels(aoi_name: str, config: LczLabelConfig, *, force: bool = False,
                 run_mask: bool = True) -> gpd.GeoDataFrame:
    path = config.cache_dir / aoi_name / f"blocks_labelled_{aoi_name}.parquet"
    if path.exists() and not force:
        gdf = gpd.read_parquet(path)
        if (gdf.get("config_hash") == config.config_hash).all():
            logger.info(f"[{aoi_name}] labelled blocks cache hit: {path.name}")
            return gdf
        logger.info(f"[{aoi_name}] labelled blocks stale (config changed) — rebuilding")
    return build_block_labels(aoi_name, config, force=force, run_mask=run_mask)


def _export_aoi(aoi_name: str, config: LczLabelConfig, *, force: bool = False) -> pd.DataFrame:
    labels = _load_labels(aoi_name, config, force=force)
    with _stage(f"[{aoi_name}] rasters"):
        write_rasters(labels, aoi_name, config)
    with _stage(f"[{aoi_name}] patch transfer"):
        grid = load_grid(aoi_name, config, force=force)
        patches = patch_transfer(labels, grid, aoi_name, config)
    return patches


def _validate_aoi(aoi_name: str, config: LczLabelConfig, *, force: bool = False) -> dict:
    ppath = config.cache_dir / aoi_name / f"patch_labels_{aoi_name}.parquet"
    if ppath.exists() and not force:
        patches = pd.read_parquet(ppath)
    else:
        patches = _export_aoi(aoi_name, config, force=force)
    result = validate_patch_transfer(patches, aoi_name)
    if result:
        labels = _load_labels(aoi_name, config)
        grid = load_grid(aoi_name, config)
        result["homogeneity"] = block_homogeneity(labels, grid)
    return result


# ── Commands ──────────────────────────────────────────────────────────────────

_AOI_OPT = typer.Option(None, "--aoi", help="AOI name (repeatable)")
_ALL_OPT = typer.Option(False, "--all-aois", help="Process every So2Sat city")
_CFG_OPT = typer.Option(None, "--config", help="YAML config path")
_FORCE_OPT = typer.Option(False, "--force", help="Rebuild caches")


@app.command()
def extract(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
            config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 2: extract Overture features per AOI."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        with _stage(f"[{name}] overture extract"):
            extract_overture(name, cfg, force=force)


@app.command()
def blocks(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
           config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 4a: delineate blocks + adjacency per AOI."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        ex = extract_overture(name, cfg, force=force)
        with _stage(f"[{name}] blocks"):
            b = build_blocks(name, ex, cfg, force=force)
            build_adjacency(b, cfg, name, force=force)


@app.command()
def ucp(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
        config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 4b: compute the per-block UCP table per AOI."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        ex = extract_overture(name, cfg, force=force)
        b = build_blocks(name, ex, cfg, force=force)
        compute_ucp(b, ex, cfg, name, force=force)


@app.command()
def label(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
          config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stages 5-6: classify blocks + zones + confidence (no temporal mask)."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        build_block_labels(name, cfg, force=force, run_mask=False)


@app.command()
def mask(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
         config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 7: temporal stability mask per AOI."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        ex = extract_overture(name, cfg, force=force)
        b = build_blocks(name, ex, cfg, force=force)
        compute_change_mask(b, cfg, name, force=force)


@app.command()
def export(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
           config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 8: block parquet + bitmask/confidence/block_id rasters + patch transfer."""
    cfg = load_config(config)
    for name in _resolve_aois(cfg, aoi, all_aois):
        _export_aoi(name, cfg, force=force)


@app.command()
def validate(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
             config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Stage 9: agreement vs So2Sat via the patch transfer + homogeneity check."""
    cfg = load_config(config)
    names = _resolve_aois(cfg, aoi, all_aois)
    results = [_validate_aoi(name, cfg, force=force) for name in names]
    report = cfg.cache_dir / f"validation_{'_'.join(names[:3])}.md"
    write_report([r for r in results if r], report)


@app.command(name="all")
def run_all(aoi: list[str] = _AOI_OPT, all_aois: bool = _ALL_OPT,
            config: str = _CFG_OPT, force: bool = _FORCE_OPT):
    """Full pipeline per AOI, then merge labels + one validation report."""
    cfg = load_config(config)
    names = _resolve_aois(cfg, aoi, all_aois)
    all_gdfs, results, failed = [], [], []
    for i, name in enumerate(names, 1):
        logger.info(f"══ AOI {i}/{len(names)}: {name} ══")
        try:
            gdf = build_block_labels(name, cfg, force=force, run_mask=True)
            _export_aoi(name, cfg)
            results.append(_validate_aoi(name, cfg))
        except Exception as e:  # noqa: BLE001 — one bad AOI must not abort the batch
            logger.exception(f"[{name}] FAILED: {e}")
            failed.append(name)
            continue
        all_gdfs.append(gdf.to_crs("EPSG:4326"))
    if all_gdfs:
        merged = pd.concat(all_gdfs, ignore_index=True)
        merged_path = cfg.cache_dir / "labels_all.parquet"
        gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326").to_parquet(merged_path)
        logger.info(f"Merged {len(merged)} blocks from {len(all_gdfs)} AOIs -> {merged_path}")
    if any(results):
        write_report([r for r in results if r], cfg.cache_dir / "validation_all.md")
    logger.info(f"Batch done: {len(all_gdfs)} ok, {len(failed)} failed"
                + (f" ({', '.join(failed)})" if failed else ""))


if __name__ == "__main__":
    app()
