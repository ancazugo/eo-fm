# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Earth Observation Foundation Model (eo-fm) — a geospatial ML project for Local Climate Zone (LCZ) classification using satellite embeddings (Tessera, AlphaEarth, Embedded Seamless Data). Two canonical pipelines — patch classification and semantic segmentation — share a model-family registry, a generic training loop, and a common evaluation/WandB stack. See README.md for full per-script usage. Benchmark results, honest-evaluation protocols (leave-one-city-out) and documented negative results for the global LCZ split live in `docs/global_lcz_campaign_2026-07.md` — read it before re-running or extending those experiments.

## Commands

```bash
source /maps/acz25/envs/eo_fm-env/bin/activate
uv sync --active           # Install/sync all dependencies into the activated env (not ./.venv)

# Patch classification (any classification family)
python src/patch_classification.py \
    --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \
    --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities --cities Nairobi \
    --output-name GeoTessera_v1.1 --year 2017 \
    --family resnet --preset small --patch-size 32 \
    --embedding-name tesserav1.1 \
    --embedding-dir ${DATA_DIR}/input/GeoTessera/v1.1/2017 \
    --output-dir ${DATA_DIR}/output/lcz-classification/dl

# Semantic segmentation (any segmentation family)
python src/semantic_segmentation.py \
    --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities --cities Nairobi \
    --output-name GeoTessera_v1.1 --year 2017 --label-source gpkg \
    --family unet --preset small \
    --embedding-name tesserav1.1 \
    --embedding-dir ${DATA_DIR}/input/GeoTessera/v1.1/2017 \
    --output-dir ${DATA_DIR}/output/lcz-classification/dl

# Linear probe (a classification family — trains like any other model)
python src/patch_classification.py ... --family linear_probe   # --arch mean_std for mean+std pooling

# kNN / GMM-density baselines on cached pooled features
python src/knn_baseline.py --so2sat-dir ... --cities Nairobi \
    --output-name GeoTessera_v1.1 --year 2017 --embedding-name tesserav1.1 \
    --pooling gap --output-dir ...   # --classifier gmm for per-class GMM density

# Standalone ROI inference from a checkpoint
python src/infer_roi.py --model-type resnet --preset small \
    --checkpoint <run_dir>/<model>-best.pt \
    --embedding-name tesserav1.1 --embedding-dir ... --year 2017 \
    --city Nairobi --output out.tif
```

`pytest` runs the offline `lcz_labels/tests` and `lcz_train/tests` suites (configured in pyproject; no network needed). No linter is configured. Smoke-test the DL pipelines with `--preset nano --max-epochs 1 --no-wandb` on Nairobi.

## Architecture

### Project Structure

```
src/
  patch_classification.py    # ENTRY: patch classification pipeline
  semantic_segmentation.py   # ENTRY: segmentation pipeline (+--family unet|resnet_unet)
  knn_baseline.py            # ENTRY: cosine-kNN / per-class GMM density baselines on cached pooled features
  infer_roi.py               # ENTRY + library: sliding-window ROI inference for ALL families;
                             #   legacy fc-only probe checkpoints via --stats-file
  ensemble_eval.py           # ENTRY: multi-checkpoint softmax ensemble on the global split + cached probs
  ensemble_stacking.py       # ENTRY: weighted/calibrated/stacked combining + leave-one-city-out audit
  tta_city_adapt.py          # ENTRY: per-city AdaBN/TENT test-time adaptation (negative result; kept as tool)
  sample_unlabeled_patches.py / generate_pseudo_labels.py
                             # ENTRY: noisy-student SSL prep (student trains via --pseudo-gpkg)
  models/                    # Architectures, one module per family + registry
    registry.py              #   ModelFamily dataclass, MODEL_REGISTRY, build_model(), families_for()
    timm_families.py         #   resnet/efficientnet/convnext/densenet/mobilenet/vit (classification)
    mlp.py                   #   GAP+MLP (classification)
    aspp.py                  #   LightASPPHead (classification)
    linear_probe.py          #   pooling + BatchNorm1d(affine=False) + Linear (classification);
                             #   arch payload = pooling ("gap"/"mean_std"); stats live in the checkpoint
    unet.py                  #   DoubleConv, UNet (segmentation)
    resnet_unet.py           #   ResNetUNet (segmentation)
  training/
    tasks.py                 #   LCZResNetModule (cls, monitor=val_f1), LCZUNetModule (seg, monitor=val_miou)
    loop.py                  #   run_training_loop() — generic Adam+cosine loop, early stopping, checkpointing
    evaluate.py              #   evaluate_classification/segmentation: OA, macro acc, F1, kappa,
    augment.py               #   per-class table, confusion matrix PNG; flips/rot90/noise augmentation
  datasets/
    registry.py              #   EMBEDDING_REGISTRY metadata (in_channels, tile filename patterns)
    tiles.py                 #   build_tile_index/open_tile/crop_patch for raw source tiles
    so2sat.py                #   patch items, PatchDataset/PatchDataModule, build_so2sat_items
    grid_tiles.py            #   grid-tile items, GridSegDataset/GridSegDataModule
    downloaders.py           #   GEE / geotessera / source.coop downloaders
  utils/
    runtime.py               #   resolve_device, resolve_dequantize, detect_in_channels,
                             #   init_run (wandb + run_dir), run_city_inference
    constants.py, paths.py, wandb.py, gee.py, grid_split.py, plot_lcz.py, ...
  extract_so2sat_embeddings.py / extract_grid_embeddings.py   # data prep (npy extraction)
  download_embeddings.py / download_missing_coop_tiles.py / create_city_grids.py
  dequantize_embeddings.py / mosaic_to_tessera11.py / plot_embeddings.py
lcz_labels/                  # Overture/OSM LCZ pseudo-labelling package (typer CLI, pydantic
                             #   config, offline pytest suite; see lcz_labels/README.md).
                             #   Block-based (momepy enclosures on the road/rail/water
                             #   network; blocks.py/export.py/classify.form_zones) — the
                             #   320 m So2Sat grid survives only as grid.py's compat export
                             #   (patch transfer + validation), not the primary label unit.
lcz_train/                   # Training harness consuming ONLY the lcz_labels Stage 8 export
                             #   (lcz_bitmask/confidence/block_id rasters + blocks_labelled
                             #   parquet + adjacency). Two formulations sharing one masked
                             #   marginalised-CE loss contract: A = dense per-pixel heads
                             #   (A1 linear/A2 MLP/A3 dilated-conv, + multi-scale pooling),
                             #   B = block-as-sample (B1 mean-pool/B2 attention-pool/B3 GNN,
                             #   torch_geometric optional via the "gnn" extra). City-held-out,
                             #   region-stratified splits (splits.py) — So2Sat cities always
                             #   test. `python -m lcz_train run --exp <id>|ladder`.
```

### Key Design Decisions

- **Model registry**: every architecture family registers a `ModelFamily(name, pipeline, presets, build)` in `models/`. `families_for("classification"/"segmentation")` feeds the `--family` CLI choices of the matching pipeline. **Adding a model = one module + one register() call + an import in models/__init__.py.**
- **One generic training loop** (`training/loop.py`) for both pipelines; the task modules (`training/tasks.py`) own loss, metrics and the monitored metric (`val_f1` cls / `val_miou` seg). Checkpoints store `{"model_state_dict", "epoch", <monitor>}` — only the inner model, so they are architecture-defined and backward compatible.
- **Dequantize is auto-applied** for `alpha_earth_coop` and `seamless` (`utils.runtime.resolve_dequantize`); `--dequantize` is just a force flag. Seamless expands 13→72 channels (in_channels override).
- **Label conventions**: classification labels LCZ 1-17 → 0-16; segmentation masks 1-17 → 0-16 with nodata 0 → -1 (`ignore_index=-1` everywhere).
- **infer_roi** routes by family pipeline: segmentation → Hanning-blended per-pixel logits; classification → patch-wise majority vote. Works directly from raw source tiles (no pre-extracted npy needed).
- **linear_probe is a classification family**: `LinearProbeModel` pools internally and normalises via BatchNorm running stats, so it fits the (B,C,H,W) registry contract and its checkpoints are self-contained (no stats npz). Legacy fc-only checkpoints from the retired standalone script load through `infer_roi.py --stats-file` via `models.linear_probe.load_legacy_linear_probe` (numerically identical conversion). The cosine-kNN and per-class GMM density baselines (no trainable model) stay in `knn_baseline.py` (`--classifier knn|gmm`).

## Environment

- **Python 3.12**, managed with **uv**; venv at `/maps/acz25/envs/eo_fm-env`
- Requires `.env` file with: `DATA_DIR`, `GEE_PROJECT_NAME`, `WANDB_KEY`
- CRS: EPSG:4326 (WGS84) as project default
- City boundaries are present in `data/so2sat_guppd_bounds.csv` in EPSG:4326 (WGS84)
- Per city labels are available in `${DATA_DIR}/input/So2Sat-LCZ42/v4/cities`, both as vector (.gpkg) and as raster (.tif)
- AlphaEarth embeddings: `${DATA_DIR}/input/Google/AlphaEarth/{year}` (coop: `.../AlphaEarth/coop`)
- Geotessera embeddings: `${DATA_DIR}/input/GeoTessera/{year}` (v1.1: `.../GeoTessera/v1.1/{year}`)
- Embedded Seamless Data embeddings: `${DATA_DIR}/input/EmbeddedSeamlessData/{year}`
- Demuzere LCZ files: `${DATA_DIR}/input/Demuzere_et_al_2022_LCZ`
- So2Sat LCZ42 info: `${DATA_DIR}/input/So2Sat-LCZ42`
- torch-based models log to the `lcz-classification-dl` project in WandB (entity `phd-thesis-team`)
- torch-based wandb runs are saved in `${DATA_DIR}/output/lcz-classification/dl`
