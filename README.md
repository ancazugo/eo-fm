# eo-fm — LCZ Classification Pipeline

Earth Observation Foundation Model pipeline for Local Climate Zone (LCZ) classification using satellite embedding tiles.

---

## Setup

```bash
source /maps/acz25/envs/eo_fm-env/bin/activate
uv sync --active   # --active targets the activated env instead of ./.venv
```

Requires a `.env` file at the repo root:

```
DATA_DIR="/maps/acz25/phd-thesis-data"
GEE_PROJECT_NAME="your-gee-project"
WANDB_KEY="your-wandb-key"
```

---

## Workflow Overview

```
1. Download embeddings        download_embeddings.py / download_missing_coop_tiles.py
        ↓
2. Extract patch/grid npy     extract_so2sat_embeddings.py / extract_grid_embeddings.py
        ↓
3a. Patch classification      patch_classification.py   (per-patch; any classification family)
3b. Semantic segmentation     semantic_segmentation.py  (per-grid-tile; any segmentation family)
3c. Seg distillation          generate_seg_pseudo_rasters.py → semantic_segmentation.py --label-tif-dir
                              → eval_seg_on_patches.py  (patch-benchmark kappa for seg checkpoints)
3d. Patch SSL (noisy student) sample_unlabeled_patches.py → generate_pseudo_labels.py
                              → patch_classification.py --pseudo-gpkg
        ↓
4. ROI inference              infer_roi.py              (auto-called at end of training)
        ↓
6. Ensembles & honest eval    ensemble_eval.py → ensemble_stacking.py / tta_city_adapt.py
```

All scripts are run from the repo root (`/home/acz25/repos/eo_fm`). Scripts are in `src/`.

### Code layout

```
src/
  patch_classification.py    # ENTRY: patch classification pipeline
  semantic_segmentation.py   # ENTRY: segmentation pipeline
  knn_baseline.py            # ENTRY: cosine-kNN / per-class GMM density baselines on cached pooled features
  infer_roi.py               # ENTRY + library: sliding-window ROI inference (all families;
                             #   --stats-file for legacy fc-only linear probe checkpoints)
  sample_unlabeled_patches.py / generate_pseudo_labels.py
                             # ENTRY: noisy-student SSL data prep (Step 3d)
  ensemble_eval.py           # ENTRY: multi-checkpoint softmax-ensemble eval + cached probs (Step 6)
  ensemble_stacking.py       # ENTRY: weighted/calibrated/stacked ensembling + leave-one-city-out audit
  tta_city_adapt.py          # ENTRY: per-city AdaBN/TENT test-time adaptation
  models/                    # Architectures + family registry
    registry.py              #   ModelFamily, build_model(), families_for()
    timm_families.py         #   resnet, efficientnet, convnext, densenet, mobilenet, vit
    mlp.py, aspp.py          #   classification families
    linear_probe.py          #   pooling + BatchNorm + Linear probe (classification family)
    unet.py, resnet_unet.py  #   segmentation families
  training/                  # Shared training library
    tasks.py                 #   LCZResNetModule (cls), LCZUNetModule (seg)
    loop.py                  #   run_training_loop() — one generic loop
    evaluate.py              #   test eval + confusion matrix for both pipelines
    augment.py               #   flip/rot90/noise augmentation
  datasets/                  # Data layer
    registry.py              #   EMBEDDING_REGISTRY metadata
    tiles.py                 #   raw tile index/open/crop (zarr, tif, tessera v1.1, coop)
    so2sat.py                #   patch items + PatchDataset/DataModule (classification)
    grid_tiles.py            #   grid-tile items + GridSegDataset/DataModule (segmentation)
  utils/
    runtime.py               #   device/dequantize/wandb-run/city-inference helpers
lcz_labels/                  # Overture/OSM LCZ pseudo-labelling package (own README + tests);
                             #   block-based (momepy enclosures on roads/rail/water, not the
                             #   320 m So2Sat grid, which survives only as a compat export)
                             #   python -m lcz_labels all --aoi Nairobi
lcz_train/                   # Training harness on the lcz_labels block export: dense (A) and
                             #   block-as-sample (B) heads under one masked marginalised-CE
                             #   loss, city-held-out region-stratified splits, block-level eval
                             #   python -m lcz_train run --exp A1|ladder
paper/                       # arXiv draft sources + verify_numbers.py (run_paper_*.sh reproduce)
R/                           # R figure scripts (reads data/wandb_export_*.csv)
```

### Model families

`--family` selects the architecture; `--preset` (nano/small/base/medium/large) the size.

| Pipeline | Families |
|---|---|
| `patch_classification.py` | `resnet` (default), `efficientnet`, `convnext`, `densenet`, `mobilenet`, `vit`, `aspp`, `mlp`, `linear_probe` |
| `semantic_segmentation.py` | `unet` (default), `resnet_unet` |

`linear_probe` is pooling + `BatchNorm1d(affine=False)` + Linear (the "BN + linear"
probe). Presets are all equivalent (GAP pooling); pass `--arch mean_std` for
mean+std pooling (doubled feature dim). Normalisation stats are stored in the
checkpoint, so `infer_roi.py --model-type linear_probe` works without a stats file.
The kNN and per-class GMM density counterparts live in `knn_baseline.py`
(cached pooled features; `--classifier knn|gmm`).

**Adding a new model** = one module in `src/models/` defining the architecture +
a `register(ModelFamily(...))` call + an import in `src/models/__init__.py`.
It automatically appears in the `--family` choices of the matching pipeline and
in `infer_roi.py --model-type`.

---

## Embedding Types

| Key | Source | Format | Channels | Resolution | Notes |
|---|---|---|---|---|---|
| `alpha_earth` | Google AlphaEarth | `.zarr` or `.tif` | 64 | 10 m | float32 (int8 range); use `--dequantize` |
| `alpha_earth_coop` | AlphaEarth COOP | `.tif` + `aef_index.gpkg` | 64 | 10 m | Same dequantize as `alpha_earth` |
| `tesserav1.1` | GeoTessera v1.1 | `.tiff` (geoinfo) + `.npy` pairs | 128 | 10 m | Auto-dequantized at extraction time |
| `tessera` | GeoTessera v1.0 | `.zarr` | 128 | 10 m | Already float32 |
| `seamless` | Embedded Seamless Data | `.tiff` | 13→72 | 30 m | 13 raw bands; use `--dequantize` to expand to 72 ch |

### Dequantization

Raw tiles for `alpha_earth_coop` and `seamless` store compressed values that must be expanded before training:

- **`alpha_earth_coop`**: `((v / 127.5)² × sign(v))` — int8 range → float32 (64 ch unchanged)
- **`seamless`**: ESD codebook factorization — 13 uint16 indices → 72 float32 channels in [-1, 1]
- **`tesserav1.1`**: `int8 × scales` — handled automatically at extraction time; no flag needed at training

Dequantization is **applied automatically** for `alpha_earth_coop` and `seamless`
in all training and inference scripts (`utils.runtime.resolve_dequantize`);
`--dequantize` remains as an explicit force flag for other embeddings.

---

## Step 1 — Download Embeddings

### `download_embeddings.py`

Downloads Tessera (v1.0) or AlphaEarth embeddings for all So2Sat GUPPD cities from the `data/so2sat_guppd_bounds.csv` bounding boxes.

```bash
# Download AlphaEarth for all cities (2 parallel GEE workers)
python src/download_embeddings.py \
    --year 2017 \
    --output-format tif \
    --alpha-earth-jobs 2

# Download Tessera for all cities (parallel, CPU count - 1 workers)
python src/download_embeddings.py \
    --year 2017 \
    --output-format zarr
```

Output paths are read from `utils/paths.py` (`TESSERA_DIR`, `ALPHA_EARTH_DIR`).

### `download_missing_coop_tiles.py`

After running `extract_so2sat_embeddings.py`, some COOP patches may be unextracted because their source tiles are listed in `aef_index.gpkg` but were not downloaded locally (typically patches at city-bbox edges). This script finds those tiles and downloads only the missing ones.

```bash
python src/download_missing_coop_tiles.py --workers 8 --year 2017
```

It compares the existing `.npy` files in `training/AlphaEarthCoop/{year}/` against `patches_reference_rxr.gpkg`, queries `aef_index.gpkg` for the covering tiles, and downloads any that are absent. Re-run `extract_so2sat_embeddings.py` with `--skip-existing` afterwards to extract the newly available patches.

### `download_coop_tiles_bbox.py`

Companion to `download_missing_coop_tiles.py` for arbitrary areas: downloads the COOP tiles covering one or more city bboxes (GUPPD `SMOD_ID`s looked up in `data/so2sat_guppd_bounds.csv`), rather than back-filling the So2Sat patch set.

```bash
python src/download_coop_tiles_bbox.py --smod-ids 30_4732 --year 2017 --workers 8
```

---

## Step 2a — Extract So2Sat Patch Embeddings

### `extract_so2sat_embeddings.py`

Clips source embedding tiles to each So2Sat patch bounding box and saves float32 `.npy` files. Used as input for **patch classification**.

```bash
# AlphaEarth COOP — all So2Sat cities
python src/extract_so2sat_embeddings.py \
    --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \
    --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \
    --embedding-name alpha_earth_coop \
    --output-name AlphaEarthCoop \
    --year 2017 \
    --workers 8

# GeoTessera v1.1
python src/extract_so2sat_embeddings.py \
    --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \
    --embedding-dir /maps/acz25/phd-thesis-data/input/GeoTessera/v1.1/2017 \
    --embedding-name tesserav1.1 \
    --output-name GeoTesserav1.1 \
    --year 2017 \
    --workers 8

# Embedded Seamless Data
python src/extract_so2sat_embeddings.py \
    --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \
    --embedding-dir /maps/acz25/phd-thesis-data/input/EmbeddedSeamlessData/2017 \
    --embedding-name seamless \
    --output-name EmbeddedSeamless \
    --year 2017 \
    --workers 4
```

**Output layout:**
```
{so2sat_dir}/{training,validation,testing}/{output_name}/{year}/patch_{id}.npy
```

Each `.npy` is float32 with shape `(C, H, W)`. For `tesserav1.1`, dequantization (`int8 × scales`) is applied automatically during extraction.

---

## Step 2b — Create City Grids (for segmentation)

### `create_city_grids.py`

Builds a regular grid over each city's ROI with a fixed 3×3 checkerboard train/val/test pattern. Required before extracting grid embeddings for semantic segmentation.

```bash
python src/create_city_grids.py \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --workers 4
```

**Output per city:**
- `patches_reference_{city}_split.gpkg` — original label polygons with `grid_id` and `split` columns added
- `{city}_grid.gpkg` — grid tiles with `split`, `coverage_pct`, `is_valid` columns

The `is_valid` flag marks tiles with label coverage in range `[2%, 95%]`. Only valid tiles are used for training.

---

## Step 2c — Extract Grid Tile Embeddings (for segmentation)

### `extract_grid_embeddings.py`

Clips source embedding tiles to each grid cell bounding box and saves float32 `.npy` files. Used as input for **semantic segmentation**.

```bash
# AlphaEarth COOP
python src/extract_grid_embeddings.py \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \
    --embedding-name alpha_earth_coop \
    --output-name AlphaEarthCoop \
    --year 2017 \
    --only-valid \
    --workers 8

# GeoTessera v1.1
python src/extract_grid_embeddings.py \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --embedding-dir /maps/acz25/phd-thesis-data/input/GeoTessera/v1.1/2017 \
    --embedding-name tesserav1.1 \
    --output-name GeoTesserav1.1 \
    --year 2017 \
    --only-valid \
    --workers 8

# Embedded Seamless Data
python src/extract_grid_embeddings.py \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --embedding-dir /maps/acz25/phd-thesis-data/input/EmbeddedSeamlessData/2017 \
    --embedding-name seamless \
    --output-name EmbeddedSeamless \
    --year 2017 \
    --only-valid \
    --workers 4
```

**Output layout:**
```
{cities_dir}/{city}/{output_name}/{year}/{train,val,test}/{city}_{grid_id:02d}.npy
```

Use `--only-valid` to skip tiles flagged as invalid (recommended). Use `--skip-existing` to resume interrupted runs.

---

## Step 3a — Patch Classification

### `patch_classification.py`

Trains a patch classifier on pre-extracted So2Sat patch `.npy` files. After training, automatically runs full-ROI inference via `infer_roi.py`.

**Families** (`--family`): `resnet` (default) · `efficientnet` · `convnext` · `densenet` · `mobilenet` · `vit` · `aspp` · `mlp`
**Presets** (`--preset`): `nano` · `small` · `base` · `medium` · `large` (resnet: resnet18 → resnet152)

#### Split modes

| Mode | Flag | Split source | When to use |
|---|---|---|---|
| Per-city | *(default)* | `patches_reference_{city}_split.gpkg` — grid-based `split` column | Training/evaluating on specific cities |
| Global | `--global-split` | `patches_reference_rxr.gpkg` — original So2Sat `dataset` column | Full 400 k-patch cross-city training |

**Per-city mode** requires `--cities-dir` and `--cities`. The train/val/test split comes from the grid-based assignment in each city's split GeoPackage (created by `create_city_grids.py`).

**Global mode** (`--global-split`) uses the original So2Sat split encoded in `patches_reference_rxr.gpkg` (`dataset` column: `training` / `validation` / `testing`), giving 352 k / 24 k / 24 k patches across all 51 cities. No city selection is needed; `--cities-dir` and `--cities` are ignored for data loading (they can still be passed to run post-training inference on specific cities).

```bash
# Per-city — AlphaEarth COOP, London
python src/patch_classification.py \
    --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --cities London \
    --output-name AlphaEarthCoop \
    --year 2017 \
    --preset small \
    --patch-size 32 \
    --batch-size 64 \
    --num-workers 4 \
    --max-epochs 50 \
    --embedding-name alpha_earth_coop \
    --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \
    --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl \
    --wandb-project lcz-classification-dl \
    --dequantize

# Global split — AlphaEarth COOP, all cities
python src/patch_classification.py \
    --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \
    --global-split \
    --output-name AlphaEarthCoop \
    --year 2017 \
    --preset large \
    --patch-size 32 \
    --batch-size 256 \
    --num-workers 8 \
    --max-epochs 50 \
    --embedding-name alpha_earth_coop \
    --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \
    --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl \
    --wandb-project lcz-classification-dl \
    --dequantize

# Per-city — GeoTessera v1.1, London (no --dequantize; already float32 after extraction)
python src/patch_classification.py \
    --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --cities London \
    --output-name GeoTesserav1.1 \
    --year 2017 \
    --preset small \
    --patch-size 32 \
    --batch-size 64 \
    --num-workers 4 \
    --max-epochs 50 \
    --embedding-name tesserav1.1 \
    --embedding-dir /maps/acz25/phd-thesis-data/input/GeoTessera/v1.1/2017 \
    --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl \
    --wandb-project lcz-classification-dl

# Per-city — Embedded Seamless Data, London
python src/patch_classification.py \
    --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --cities London \
    --output-name EmbeddedSeamless \
    --year 2017 \
    --preset small \
    --patch-size 32 \
    --batch-size 64 \
    --num-workers 4 \
    --max-epochs 50 \
    --embedding-name seamless \
    --embedding-dir /maps/acz25/phd-thesis-data/input/EmbeddedSeamlessData/2017 \
    --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl \
    --wandb-project lcz-classification-dl \
    --dequantize
```

**Key flags:**
- `--global-split` — use all So2Sat patches with the original dataset split (no city selection)
- `--global-gpkg` — override the default `{so2sat_dir}/patches_reference_rxr.gpkg` path (only with `--global-split`)
- `--family` — model family (see Model families table)
- `--preset` — model size (`nano`/`small`/`base`/`medium`/`large`)
- `--patch-size` — resize input patches to this square (pixels); use 32 for 10 m embeddings
- `--dequantize` — force dequantization (auto-applied for `alpha_earth_coop`/`seamless`)
- `--checkpoint` — skip training, load weights and run inference only

**Output per run** (under `--output-dir/{wandb-run-name}/`):
```
{family}_{preset}_{output_name}_{city}-best.pt   ← best checkpoint (per-city mode)
{family}_{preset}_{output_name}_global-best.pt   ← best checkpoint (global mode)
{run_name}_{family}-{preset}-classification-prediction_{city}.tif
{run_name}_{family}-{preset}-classification-prediction_{city}.png
test_confusion_matrix.png
```

---

## Step 3b — Semantic Segmentation

### `semantic_segmentation.py`

Trains a segmentation model on pre-extracted grid tile `.npy` files. Label masks are rasterized on the fly from `patches_reference_{city}_split.gpkg` polygons. After training, automatically runs full-ROI inference via `infer_roi.py`.

**Families** (`--family`): `unet` (default) · `resnet_unet` (timm ResNet encoder + U-Net decoder; presets nano/small/base → resnet18/34/50)

**U-Net presets:**

| Preset | Depth | Base features | Params |
|---|---|---|---|
| `nano` | 2 | 8 | ~0.1 M |
| `small` | 3 | 32 | ~2.0 M |
| `base` | 3 | 48 | ~4.3 M |
| `medium` | 4 | 32 | ~7.8 M |
| `large` | 4 | 48 | ~17 M |

```bash
# AlphaEarth COOP — London, small preset
python src/semantic_segmentation.py \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --cities London \
    --output-name AlphaEarthCoop \
    --year 2017 \
    --label-source gpkg \
    --preset small \
    --batch-size 4 \
    --num-workers 4 \
    --max-epochs 30 \
    --early-stopping-patience 8 \
    --embedding-name alpha_earth_coop \
    --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \
    --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl \
    --wandb-project lcz-classification-dl \
    --dequantize

# GeoTessera v1.1 — London (no --dequantize)
python src/semantic_segmentation.py \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --cities London \
    --output-name GeoTesserav1.1 \
    --year 2017 \
    --label-source gpkg \
    --preset small \
    --batch-size 4 \
    --num-workers 4 \
    --max-epochs 30 \
    --early-stopping-patience 8 \
    --embedding-name tesserav1.1 \
    --embedding-dir /maps/acz25/phd-thesis-data/input/GeoTessera/v1.1/2017 \
    --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl \
    --wandb-project lcz-classification-dl

# Embedded Seamless Data — London
python src/semantic_segmentation.py \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \
    --cities London \
    --output-name EmbeddedSeamless \
    --year 2017 \
    --label-source gpkg \
    --preset small \
    --batch-size 4 \
    --num-workers 4 \
    --max-epochs 30 \
    --early-stopping-patience 8 \
    --embedding-name seamless \
    --embedding-dir /maps/acz25/phd-thesis-data/input/EmbeddedSeamlessData/2017 \
    --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl \
    --wandb-project lcz-classification-dl \
    --dequantize
```

**Key flags:**
- `--label-source` — `gpkg` (rasterize polygons on the fly) or `tif` (clip label raster)
- `--label-tif-dir` — train on dense pseudo-label rasters (see Step 3c); val/test stay on gpkg GT
- `--family` — model family (`unet` or `resnet_unet`)
- `--preset` — size preset (see table above)
- `--dequantize` — force dequantization (auto-applied for `alpha_earth_coop`/`seamless`)
- `--checkpoint` — skip training, load weights and run inference only

Test metrics are reported at two scales: native 10 m per-pixel (`test_*`) and majority-pooled 10×10 blocks (`test_*_100m`) — the ~100 m scale LCZ is defined at.

**Output per run** (under `--output-dir/{wandb-run-name}/`):
```
{family}_{preset}_{output_name}_{city}-best.pt
{run_name}_{family}-{preset}-segmentation-prediction_{city}.tif
{run_name}_{family}-{preset}-segmentation-prediction_{city}.png
test_confusion_matrix.png
```

---

## Step 3c — Segmentation Distillation (dense pseudo-labels)

So2Sat supervision rasterizes to sparse single-class 32×32 blocks, so a UNet trained on it never
sees dense labels or class boundaries. The distillation route fixes this by teaching the UNet from
a strong patch classifier:

### `generate_seg_pseudo_rasters.py`

Runs a patch-classification teacher checkpoint over each city with infer_roi's soft-voting sliding
window (default: 320 m patches every 80 m, 10 m output) and writes a dense label raster:
teacher argmax kept where soft-voted confidence ≥ `--min-conf` (default 0.5), val/test-split patch
footprints zeroed (no leakage into eval pixels), train-split GT polygons burned in on top.

```bash
python src/generate_seg_pseudo_rasters.py \
    --checkpoint <run_dir>/resnet_small_GeoTessera_v1.1_global_global-best.pt \
    --family resnet --preset small \
    --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities --cities Nairobi \
    --embedding-name tesserav1.1 \
    --embedding-dir /maps/acz25/phd-thesis-data/input/GeoTessera/v1.1/2017 --year 2017 \
    --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/pseudo_seg_rasters
```

Outputs per city: `teacher_{city}.tif` + `teacher_{city}_conf.tif` (raw teacher argmax +
confidence) and `pseudo_seg_{city}.tif` (+ `.png`) — the training raster. Re-run with
`--skip-inference` to sweep `--min-conf` without redoing teacher inference.

Then train segmentation on the pseudo rasters (train split only; val/test evaluate against GT):

```bash
python src/semantic_segmentation.py ... \
    --label-tif-dir /maps/acz25/phd-thesis-data/output/lcz-classification/pseudo_seg_rasters
```

### `eval_seg_on_patches.py`

Evaluates a segmentation checkpoint on the So2Sat patch benchmark: runs the model over the
extracted 32×32 test patch npys (the exact inputs the classifiers see), mean-pools the per-pixel
logits into one prediction per patch, and reports OA / macro acc / F1 / kappa — directly comparable
to `patch_classification.py` numbers. Supports the same split modes (`--global-split`,
`--orig-test`, per-city `--cities`). Caveat: the model gets no context beyond the 32×32 patch, so
this is a conservative estimate.

```bash
python src/eval_seg_on_patches.py \
    --checkpoint <run_dir>/unet_large_...-best.pt --family unet --preset large \
    --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \
    --output-name GeoTessera_v1.1_global --year 2017 \
    --embedding-name tesserav1.1_global --global-split \
    --output-dir data/seg_patch_eval
```

---

## Step 3d — Noisy-Student SSL (patch classification)

Semi-supervised pipeline that imports unlabeled patches weakly labeled by the Demuzere et al.
2022 global 100 m LCZ map, filters them through a trained teacher, and trains a student with
down-weighted pseudo-labels. Delivered +3.1 kappa pts on the global split (0.619 → 0.6497 over
two iterations); full results and lessons in `docs/global_lcz_campaign_2026-07.md`.
End-to-end runners: `run_phase1_noisy_student.sh` (and `_iter2/_iter3/_coop_student` variants).

### `sample_unlabeled_patches.py`

Samples candidate 320 m boxes on a 480 m stride inside the Tessera-2017 tile footprints,
labels each by majority vote over its ~3×3 px Demuzere footprint, and writes
`data/patches_reference_unlabeled.gpkg` (`dataset="unlabeled"`). Filters: purity ≥ `--min-purity`
(0.65), map probability ≥ `--min-prob` (50), no So2Sat overlap; rare classes uncapped, common
classes capped per class and per tile. `--dry-run N` processes N tiles and prints yield stats.

```bash
python src/sample_unlabeled_patches.py \
    --tiles-gpkg data/tessera_v1.1_global_2017_tiles.gpkg \
    --demuzere-dir ${DATA_DIR}/input/Demuzere_2022_complete \
    --exclude-gpkg ${DATA_DIR}/input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg \
    --n-patches 300000 --output data/patches_reference_unlabeled.gpkg
```

Then extract embeddings for the pool (float16 halves the footprint; ~76 GB for 286k tessera
patches): `extract_so2sat_embeddings.py --patches-file data/patches_reference_unlabeled.gpkg
--splits unlabeled --dtype float16 --skip-existing`.

### `generate_pseudo_labels.py`

Teacher TTA inference over the unlabeled pool + label fusion. Keep rules (thresholds CLI-tunable):
**agree** — teacher top-1 = Demuzere label ∧ confidence ≥ `--min-conf` (0.7) → weight 0.5;
**rare-relax** — Demuzere label rare ∧ purity ≥ 0.75 ∧ teacher p ≥ `--rare-min-prob` (0.2) →
weight 0.3; else drop. Writes `pseudo_labels.parquet` (full audit trail) and
`patches_reference_pseudo.gpkg` (kept rows, for training).

**Calibration caveat**: a mixup + label-smoothed teacher has a softmax ceiling ≈ 0.9, so
`--min-conf` is stricter than it looks (0.8 collapsed the keep-rate to 7.9% and lost).

```bash
python src/generate_pseudo_labels.py \
    --checkpoint <run_dir>/resnet_small_GeoTessera_v1.1_global_global-best.pt \
    --family resnet --preset small \
    --unlabeled-gpkg data/patches_reference_unlabeled.gpkg \
    --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \
    --output-name GeoTessera_v1.1_global --year 2017 \
    --embedding-name tesserav1.1_global --tta --min-conf 0.7 \
    --output-dir data/pseudo_labels
```

### Student training

`patch_classification.py --pseudo-gpkg data/pseudo_labels/patches_reference_pseudo.gpkg`
appends the pseudo items to the train split with per-sample loss weights (weighted CE,
compatible with mixup and class weights); `--pseudo-weight-scale` is a global multiplier.
Without `--pseudo-gpkg` the labeled-only path is byte-identical to before.

---

## Step 4 — Standalone ROI Inference

### `infer_roi.py`

Runs inference over an arbitrary bounding box from raw source embedding tiles (no pre-extracted grid files needed). Called automatically at the end of `patch_classification.py` and `semantic_segmentation.py`, but can also be run standalone.

`--model-type` accepts any registered family: segmentation families run a sliding window with Hanning-weighted logit blending; classification families run patch-wise majority vote. Reprojects each tile's predictions into a single output GeoTIFF. For classification, `--save-confidence` additionally writes `<output>_conf.tif` with the soft-voted probability of the winning class per pixel (used by `generate_seg_pseudo_rasters.py`).

Legacy linear-probe checkpoints (fc-only state dict from the retired standalone script) are also handled: pass `--model-type linear_probe --stats-file <cache>/<key>_stats.npz` and the checkpoint is converted on load (numerically identical to the old normalisation). Probes trained via `patch_classification.py --family linear_probe` need no stats file.

```bash
# Segmentation inference (U-Net)
python src/infer_roi.py \
    --model-type unet \
    --preset small \
    --checkpoint /maps/acz25/phd-thesis-data/output/lcz-classification/dl/my-run/unet_small_AlphaEarthCoop_London-best.pt \
    --embedding-name alpha_earth_coop \
    --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \
    --year 2017 \
    --bbox "-0.51,51.28,0.33,51.69" \
    --output /maps/acz25/phd-thesis-data/output/lcz-classification/dl/my-run/London_seg.tif \
    --num-classes 17 \
    --patch-size 64 \
    --overlap 32

# Classification inference (ResNet)
python src/infer_roi.py \
    --model-type resnet \
    --preset small \
    --checkpoint /maps/acz25/phd-thesis-data/output/lcz-classification/dl/my-run/resnet_small_AlphaEarthCoop_London-best.pt \
    --embedding-name alpha_earth_coop \
    --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \
    --year 2017 \
    --bbox "-0.51,51.28,0.33,51.69" \
    --output /maps/acz25/phd-thesis-data/output/lcz-classification/dl/my-run/London_cls.tif \
    --num-classes 17 \
    --patch-size 32
```

---

## Step 5 — Embedding Analysis & Visualization

### `embedding_projection.py`

Pools every So2Sat patch to a vector, reduces with PCA → UMAP / t-SNE, and writes
a `projection_*.parquet` (2-D coords + per-patch metadata) plus static scatters
and separability / train-test-shift / entanglement diagnostics.

```bash
python src/embedding_projection.py \
    --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \
    --output-name GeoTessera_v1.1_global --year 2017 \
    --embedding-name tesserav1.1_global --global-split \
    --pooling gap --methods umap tsne \
    --output-dir ${DATA_DIR}/output/lcz-classification/embedding_viz
```

### `embedding_explorer.py`

Interactive Dash web app over the `projection_*.parquet`: pick the method
(UMAP/t-SNE/PCA), colour by one facet and set marker **shape** by a second, filter
by any facet values, and switch between a single plot, side-by-side **facet
panels**, or **linked dual plots** (box/lasso select in one highlights the same
patches in the other). A run dropdown lists every `projection_*.parquet` under
`--data-dir`.

```bash
python src/embedding_explorer.py \
    --data-dir ${DATA_DIR}/output/lcz-classification/embedding_viz \
    --port 8050
# then port-forward 8050 over SSH and open http://localhost:8050
```

---

## Step 6 — Ensembles, Stacking Audit & Test-Time Adaptation

Tools for combining trained checkpoints and evaluating the combination honestly.
Campaign results using them: `docs/global_lcz_campaign_2026-07.md`.

### `ensemble_eval.py`

Softmax-average ensemble of any number of checkpoints (typically one per embedding) on the
global So2Sat split. Aligns patches by `(dataset, patch_id)` across sources (only patches
present in ALL sources are scored), reports metrics for **every model subset**, and caches
per-model probs to `probs.npz` for the downstream tools. `--split val|test` — run both to
feed `ensemble_stacking.py`.

```bash
python src/ensemble_eval.py \
    --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 --year 2017 --split test \
    --model GeoTessera_v1.1_global,tesserav1.1_global,<ckpt>.pt \
    --model AlphaEarthCoop,alpha_earth_coop,<ckpt>.pt \
    --model EmbeddedSeamless,seamless,<ckpt>.pt \
    --tta --output-dir ${DATA_DIR}/output/lcz-classification/dl/my_ensemble
```

Model spec: `OUTPUT_NAME,EMBEDDING_NAME,CHECKPOINT[,FAMILY,PRESET]` (defaults `resnet,small`).

### `ensemble_stacking.py`

Fits combiners on the val probs and evaluates on the test probs: equal-weight baseline,
simplex weight grid search, LR stacker, per-model temperature calibration
(`--temperature-scale`), and — critically — a **leave-one-city-out audit** (`--city-holdout`):
So2Sat val and test contain the *same 10 cities*, so anything expressive fit on val reads
city-conditional structure back off test (a val-fit LR stacker inflates by ~16 kappa pts here;
its LOCO number *loses* to plain averaging). Report LOCO numbers, or at minimum the weighted
average with a DoF caveat.

```bash
python src/ensemble_stacking.py \
    --val-npz  .../my_ensemble/ensemble_3models_val/probs.npz \
    --test-npz .../my_ensemble/ensemble_3models_test/probs.npz \
    --temperature-scale \
    --city-holdout --global-gpkg ${DATA_DIR}/input/So2Sat-LCZ42/v4/patches_reference_rxr.gpkg
```

### `tta_city_adapt.py`

Per-(model, city) test-time adaptation: `--method adabn` (re-estimate BN running stats on the
city's unlabeled patches) or `tent` (entropy minimization on BN affine params);
`--adapt-split val` (default) adapts on val inputs and evaluates on test — fully honest.
Emits `ensemble_eval`-format npz pairs, so `ensemble_stacking.py` runs on the output
unchanged. `--method none` is a plain aligned-inference mode (regression checks).
Runner: `run_phase2_tta.sh`.

**Result on this benchmark: negative** — the city shift is label-shift-shaped, and per-city BN
statistics absorb the class mix (pooled kappa −5 pts; helps Tehran/Munich, craters
Santiago/Jakarta). Kept as a general tool and as the documented negative.

---

## Step 7 — Auxiliary Structural & OSM Features

Optional side-channel data used by the `ensemble_stacking.py --aux-parquet` offset corrector
(and, experimentally, as extra fusion channels via the `aux_struct` / `osm_evidence`
embedding registry entries).

### `extract_aux_features.py`

Per-patch zonal statistics from GHSL built height/surface and ETH canopy height
(GEE-downloaded 0.5° UTM tiles, cached) — writes one parquet per split for the
stacking corrector.

### `extract_osm_features.py`

Per-patch OSM landuse features via Overpass/osmnx (cached on `/maps`) — same parquet
format as `extract_aux_features.py`.

### `precompute_aux_tiles.py`

Merges GHSL (100 m) onto the ETH canopy 10 m grid into 4-band `merged_aux/aux_*.tif`
tiles so `extract_so2sat_embeddings.py --embedding-name aux_struct` can extract patch
npy files quickly.

### `build_osm_rasters.py`

Rasterizes OSM evidence layers into 15-band 10 m per-city GeoTIFFs (`osm_evidence`
registry entry) for use as fusion channels.

### `osm_lcz_relabel.py`

Turns `osm-rasterizer` output (see `docs/osm_lcz_tag_mapping.md`) into a properly
labelled LCZ raster. Its pixel values are 1-based indices into the *feature order*,
not LCZ codes — value 8 is `roads_minor`, not LCZ 8 — so this maps them onto the
So2Sat 1–17 convention and embeds the WUDAPT colour table, giving a GeoTIFF that
QGIS renders with no styling.

Both output modes are auto-detected: the multi-band raster is composed here in band
order (same "last feature wins" priority, without inheriting any `--fill-nodata`
artefact), the single-layer one is read directly. The three height-ambiguous
building features (3/6, 2/5, 1/4) are split by Building Surface Fraction over a
100 m window, binned at 0.20/0.40 per Stewart & Oke; `--density fixed` instead maps
them to the open classes 6/5/4.

A raster written with `--fill-nodata` is detected and reported, and `--density bsf`
refuses to run on one without `--force` — the fill inflates the building mask badly
enough to call ~90 % of Nairobi compact. Use `--fill-distance-m` to refill after
relabelling instead. That fill draws only from *areal* donors (`--fill-from`),
excluding the six features Command A buffers from lines: a buffered network is
pervasive, so under nearest-neighbour fill it wins nearly every contest. Filling
Nairobi from all donors takes LCZ 15 from 16.9 % to 54.3 %, and in Cairo — whose
Nile Delta irrigation canals are mapped as `waterway=drain`/`ditch` — water goes
10.1 % to 29.7 %. Buildings stay donors: they are small polygons rather than a
network, and the zone around a building genuinely is built.

```bash
python src/osm_lcz_relabel.py lcz_labels_multiband.tif \
    -o nairobi_lcz.tif --png --dpi 200 \
    --density bsf --bsf-window-m 100 --fill-distance-m 100 \
    --title "Nairobi — OSM-derived LCZ proxy (BSF 100 m)"
```

---

## Data Paths

| Dataset | Path |
|---|---|
| So2Sat LCZ42 v4 | `/maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/` |
| So2Sat cities dir | `/maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities/` |
| AlphaEarth 2017 | `/maps/acz25/phd-thesis-data/input/Google/AlphaEarth/2017/` |
| AlphaEarth COOP | `/maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop/` |
| GeoTessera v1.0 2017 | `/maps/acz25/phd-thesis-data/input/GeoTessera/2017/` |
| GeoTessera v1.1 2017 | `/maps/acz25/phd-thesis-data/input/GeoTessera/v1.1/2017/` |
| Embedded Seamless 2017 | `/maps/acz25/phd-thesis-data/input/EmbeddedSeamlessData/2017/` |
| DL model outputs | `/maps/acz25/phd-thesis-data/output/lcz-classification/dl/` |

### GeoTessera v1.1 tile layout

```
v1.1/{year}/
    geoinfo/
        grid_{lon}_{lat}.tiff          ← spatial metadata (UTM CRS, 10 m, north-up)
    infer_output/
        {prefix}_grid_{lon}_{lat}_all_data_emb128_int8.npy    ← (H, W, 128) int8
        {prefix}_grid_{lon}_{lat}_all_data_emb128_scales.npy  ← (H, W, 1) float32
```

Dequantized as: `arr_int8.astype(float32) * scales` → transposed to `(128, H, W)`.

---

## WandB

- **Project**: `lcz-classification-dl` (team: `phd-thesis-team`)
- Each run logs train/val loss and metrics per epoch, plus final test metrics
- Metrics (both pipelines): `test_acc`, `test_acc_macro`, `test_f1`, `test_f1_micro`, `test_kappa`, `test_loss`, per-class table, confusion matrix; segmentation adds `val_miou` / `test_miou`
- Run outputs (checkpoints, prediction rasters) are saved under `{output-dir}/{wandb-run-name}/`
