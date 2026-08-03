"""Typer CLI: splits, per-AOI mosaic build, and the experiment ladder.

    python -m lcz_train splits --config lcz_labels_config.yaml
    python -m lcz_train mosaic --aoi Nairobi --config lcz_labels_config.yaml
    python -m lcz_train run --exp A1 --config train_config.yaml
    python -m lcz_train run --exp ladder --config train_config.yaml
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import typer
from loguru import logger
from torch.utils.data import ConcatDataset

from .config import TrainConfig
from .datasets import BlockDataset, PixelWindowDataset, pool_blocks_mean
from .experiments import EXPERIMENTS, LADDER_ORDER, render_ladder_table, run_block_experiment, run_dense_experiment
from .mosaics import build_mosaic, load_mosaic
from .splits import load_splits

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False,
                  help="Block-based LCZ training harness: splits, mosaics, ladder.")


def _load_lcz_config(path: str):
    from lcz_labels.config import LczLabelConfig
    return LczLabelConfig.from_yaml(path)


def _city_rasters(city: str, lcz_config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bitmask, confidence, block_id) full-city rasters from the Stage 8 export."""
    d = lcz_config.cache_dir / city
    with rasterio.open(d / f"lcz_bitmask_{city}.tif") as src:
        bitmask = src.read(1)
    with rasterio.open(d / f"confidence_{city}.tif") as src:
        conf = src.read(1)
    with rasterio.open(d / f"block_id_{city}.tif") as src:
        block_idx = src.read(1)
    return bitmask, conf, block_idx


def _blocks_labelled(city: str, lcz_config) -> gpd.GeoDataFrame:
    return gpd.read_parquet(lcz_config.cache_dir / city / f"blocks_labelled_{city}.parquet")


def _dense_dataset_for_city(city: str, year, cfg: TrainConfig, lcz_config):
    mosaic_path = build_mosaic(city, year, cfg.embedding, Path(cfg.labels_dir).parent / "embeddings",
                               lcz_config, cfg.mosaic_dir)
    mosaic, _ = load_mosaic(mosaic_path)
    bitmask, conf, block_idx = _city_rasters(city, lcz_config)
    return PixelWindowDataset(
        np.asarray(mosaic), bitmask, conf, block_idx, window_px=cfg.window_px,
        min_conf=cfg.min_conf, erosion_px=cfg.erosion_px,
        samples_per_epoch=cfg.samples_per_epoch, seed=cfg.seed,
    ), mosaic, block_idx


def _block_dataset_for_city(city: str, year, cfg: TrainConfig, lcz_config):
    mosaic_path = build_mosaic(city, year, cfg.embedding, Path(cfg.labels_dir).parent / "embeddings",
                               lcz_config, cfg.mosaic_dir)
    mosaic, _ = load_mosaic(mosaic_path)
    _, _, block_idx = _city_rasters(city, lcz_config)
    blocks = _blocks_labelled(city, lcz_config)
    labelled = blocks[blocks["label_type"].isin(["hard", "coarse"])].reset_index(drop=True)
    n_blocks = int(labelled["block_idx"].max()) if len(labelled) else 0
    pooled_all = pool_blocks_mean(np.asarray(mosaic), block_idx, max(n_blocks, int(block_idx.max())))
    pooled = pooled_all[labelled["block_idx"].to_numpy() - 1]
    from lcz_labels.export import encode_lcz_set
    df = pd.DataFrame({
        "block_id": labelled["block_id"], "area_m2": labelled["area_m2"],
        "compactness": labelled.get("compactness", 0.0), "elongation": labelled.get("elongation", 0.0),
        "bitmask": encode_lcz_set(list(labelled["lcz_set"])), "confidence": labelled["confidence"],
    })
    return BlockDataset(pooled, df, use_ucp_features=cfg.use_ucp_features), labelled


def _ground_truth_frame(city: str, lcz_config) -> pd.DataFrame:
    blocks = _blocks_labelled(city, lcz_config)
    return blocks[["label_type", "lcz", "lcz_set", "block_kind"]]


@app.command()
def splits(
    aoi: list[str] = typer.Option(None, "--aoi", help="AOI name (repeatable)"),
    so2sat_aoi: list[str] = typer.Option(None, "--so2sat-aoi", help="So2Sat AOI forced into test"),
    version: str = typer.Option("v1", "--version"),
    seed: int = typer.Option(42, "--seed"),
    bounds_csv: str = typer.Option("data/guppd_bounds.csv", "--bounds-csv"),
    out: str = typer.Option(..., "--out"),
):
    """T1: build + save the region-stratified, city-held-out splits."""
    from .splits import make_splits, save_splits

    result = make_splits(list(aoi or []), set(so2sat_aoi or []), seed=seed, version=version,
                         bounds_csv=bounds_csv)
    path = save_splits(result, out)
    logger.info(f"splits -> {path} (train={len(result['train'])}, val={len(result['val'])}, "
                f"test={len(result['test'])})")


@app.command()
def mosaic(
    aoi: list[str] = typer.Option(..., "--aoi", help="AOI name (repeatable)"),
    year: int = typer.Option(..., "--year"),
    config: str = typer.Option(..., "--config", help="lcz_labels YAML config"),
    embedding: str = typer.Option("tesserav1.1_global", "--embedding"),
    embedding_dir: str = typer.Option(..., "--embedding-dir"),
    mosaic_dir: str = typer.Option(..., "--mosaic-dir"),
    force: bool = typer.Option(False, "--force"),
):
    """Build (or refresh) the cached per-AOI embedding mosaic."""
    lcz_config = _load_lcz_config(config)
    for name in aoi:
        build_mosaic(name, year, embedding, embedding_dir, lcz_config, mosaic_dir, force=force)


@app.command()
def run(
    exp: str = typer.Option(..., "--exp", help="Rung id (A1, A1_MS, ...) or 'ladder'"),
    config: str = typer.Option(..., "--config", help="TrainConfig YAML"),
    lcz_config_path: str = typer.Option(..., "--lcz-config", help="lcz_labels YAML config"),
    splits_path: str = typer.Option(..., "--splits"),
    year: int = typer.Option(2025, "--year"),
    out: str = typer.Option("ladder.md", "--out"),
):
    """T7: run one rung or the full ladder; writes the comparison table."""
    cfg = TrainConfig.from_yaml(config)
    lcz_config = _load_lcz_config(lcz_config_path)
    split = load_splits(splits_path)
    train_cities = split["train"]
    eval_city = (split["val"] or split["test"])[0]

    exp_ids = LADDER_ORDER if exp == "ladder" else [exp]
    results = []
    for exp_id in exp_ids:
        c = TrainConfig(**{**cfg.model_dump(), "exp_id": exp_id})
        family = EXPERIMENTS[exp_id]["family"]
        if family == "A":
            train_sets, in_channels = [], None
            for city in train_cities:
                ds, _, _ = _dense_dataset_for_city(city, year, c, lcz_config)
                train_sets.append(ds)
                in_channels = ds.mosaic.shape[0]
            eval_mosaic, _ = load_mosaic(
                build_mosaic(eval_city, year, c.embedding,
                            Path(c.labels_dir).parent / "embeddings", lcz_config, c.mosaic_dir)
            )
            _, _, eval_block_idx = _city_rasters(eval_city, lcz_config)
            gt = _ground_truth_frame(eval_city, lcz_config)
            result = run_dense_experiment(
                exp_id, ConcatDataset(train_sets), np.asarray(eval_mosaic), eval_block_idx,
                gt, c, in_channels=in_channels,
            )
        else:
            train_sets, in_channels = [], None
            for city in train_cities:
                ds, labelled = _block_dataset_for_city(city, year, c, lcz_config)
                train_sets.append(ds)
                in_channels = ds.pooled.shape[1]
            eval_ds, eval_labelled = _block_dataset_for_city(eval_city, year, c, lcz_config)
            gt = eval_labelled[["label_type", "lcz", "lcz_set", "block_kind"]]
            result = run_block_experiment(
                exp_id, ConcatDataset(train_sets), eval_ds, gt, c,
                in_channels=in_channels, extra_features=eval_ds.extra.shape[1],
            )
        results.append(result)

    table = render_ladder_table(results)
    Path(out).write_text(table)
    logger.info(f"ladder table -> {out}")


if __name__ == "__main__":
    app()
