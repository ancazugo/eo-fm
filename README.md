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
    shallow_cnn.py           #   2-conv CNN + GAP + Linear (classification family)
    linear_probe.py          #   pooling + BatchNorm + Linear probe (classification family)
    unet.py, resnet_unet.py  #   segmentation families
    fcn8.py                  #   miniature FCN-8s with score fusion (segmentation family)
    attention_unet.py        #   U-Net + additive attention gates on the skips (segmentation family)
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
| `patch_classification.py` | `resnet` (default), `efficientnet`, `convnext`, `densenet`, `mobilenet`, `vit`, `aspp`, `mlp`, `shallow_cnn`, `linear_probe` |
| `semantic_segmentation.py` | `unet` (default), `resnet_unet`, `fcn8`, `attention_unet` |

`linear_probe` is pooling + `BatchNorm1d(affine=False)` + Linear (the "BN + linear"
probe). It has **one preset, `nano`** — the only parameters are `Linear(D, num_classes)`,
so there is no capacity ladder; `--preset small|base|medium|large` is an error rather
than an alias for the same model. Since `--preset` defaults to `large`, pass
`--preset nano` explicitly. Pass `--arch mean_std` for mean+std pooling (doubled
feature dim). Not the same as `mlp --preset nano`, which is the same size and also
linear but has no BatchNorm. Normalisation stats are stored in the
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
| `tesserav1.1_global` | GeoTessera v1.1 global | `grid_*/` npy dirs + `tiff_all` | 128 | 10 m | Covers 97.5% of So2Sat 2017 patches |
| `tesserav2` | GeoTessera v2 (`large_student`) | `grid_*/` npy dirs (no geoinfo) | 128 | 10 m | Georeferenced from the tile name; covers 99.2% of So2Sat 2017 patches |
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

**Families** (`--family`): `resnet` (default) · `efficientnet` · `convnext` · `densenet` · `mobilenet` · `vit` · `aspp` · `mlp` · `shallow_cnn`
**Presets** (`--preset`): `nano` · `small` · `base` · `medium` · `large` (resnet: resnet18 → resnet152)

#### Split modes

| Mode | Flag | Split source | When to use |
|---|---|---|---|
| Per-city | *(default)* | `patches_reference_{city}_split.gpkg` — grid-based `split` column | Training/evaluating on specific cities |
| Global | `--global-split` | `patches_reference_rxr.gpkg` — original So2Sat `dataset` column | Full 400 k-patch cross-city training |
| Custom | `--global-split --split-col <col>` | any other column of the global GeoPackage | Leave-one-city-out, region-stratified, k-fold |

**Per-city mode** requires `--cities-dir` and `--cities`. The train/val/test split comes from the grid-based assignment in each city's split GeoPackage (created by `create_city_grids.py`).

**Global mode** (`--global-split`) uses the original So2Sat split encoded in `patches_reference_rxr.gpkg` (`dataset` column: `training` / `validation` / `testing`), giving 352 k / 24 k / 24 k patches across all 51 cities. No city selection is needed; `--cities-dir` and `--cities` are ignored for data loading (they can still be passed to run post-training inference on specific cities).

**Custom splits** (`--split-col`) read the train/val/test assignment from a different column of the same GeoPackage, accepting either `training`/`validation`/`testing` or `train`/`val`/`test`. Author the column however you like — hold out cities, stratify by region, cut k folds — but **leave `dataset` untouched**: it also selects which of the `training/`, `validation/`, `testing/` directories holds each patch's `.npy`, and patch_ids restart at `000000` in each one. Rewriting `dataset` to express a split silently repoints a patch at a different file with the same id. An unrecognised value, or a column that leaves any of train/val/test empty, raises rather than training on a truncated set. The column name is part of the channel-stats cache key, so a custom split cannot inherit the culture-10 normaliser; if you edit the *values* inside a column between runs, pass `--recompute-stats`.

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

**Families** (`--family`): `unet` (default) · `resnet_unet` (timm ResNet encoder + U-Net decoder; presets nano/small/base → resnet18/34/50) · `fcn8` (miniature FCN-8s: 2-3 pooling stages, 1×1 score heads fused coarse-to-fine; presets nano (2, 8) → large (3, 64)) · `attention_unet` (U-Net with additive attention gates on every skip, Oktay et al. 2018; same preset ladder as `unet` so rows are matched-capacity, gates add ~1.1% params)

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
- `--family` — model family (`unet`, `resnet_unet`, `fcn8` or `attention_unet`)
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

`--model-type` accepts any registered family: segmentation families run a sliding window with Hanning-weighted logit blending, classification families accumulate softmax over overlapping patches. Either way the **per-pixel probability volume is kept**, summed onto the output grid with `reproject(Resampling.sum)`, and argmaxed there — so an output cell is a genuine pixel-count-weighted pool of the fine predictions under it, across tile boundaries as well as within a cell. `--save-confidence` writes `<output>_conf.tif` with the pooled probability of the winning class per cell (used by `generate_seg_pseudo_rasters.py`).

**Output resolution is a choice, not a by-product.** LCZ is a ~100 m urban-climate concept; a classification model predicts one class per 320 m patch and a segmentation model one per 10 m pixel, and neither is the resolution the map should be read at.

| flag | default | what it controls |
|---|---|---|
| `--target-res` | 320 m (cls) / **100 m (seg)** | resolution of the primary map |
| `--patch-physical-stride` | `--target-res` | how far the classification window slides — i.e. how many 320 m patches vote per pixel |
| `--aggregate` | `soft` | how the fine probabilities pool into a cell |
| `--no-native` | off | suppress the `<output>_<res>m.tif` sidecar at the embedding's own resolution |

`--aggregate soft` averages the probabilities (a confidence-weighted vote); `majority` counts fine argmax votes, matching the `test_*_100m` eval metrics; `gaussian` applies Demuzere et al. 2020's per-class kernel to the probabilities first. Every run also writes the native-resolution map alongside the primary one, so nothing that depended on the 10 m output loses it. `--coarsen-to` is now a *third*, post-hoc coarsening of the finished primary map (see `coarsen_lcz_map.py`).

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
# -> London_seg.tif @ 100 m (primary) + London_seg_10m.tif (native sidecar)

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
    --patch-size 32 \
    --target-res 100
# -> London_cls.tif @ 100 m, each cell voting among the overlapping 320 m
#    patches that cover it (omit --target-res for the 320 m one-cell-per-patch map)
```

### `coarsen_bakeoff.py`

Answers "which `--aggregate` and which `--target-res`" with numbers instead of a prior. Runs the checkpoint **once** over a city, then pools that single probability volume every way and scores each result against So2Sat patch labels rasterised onto the same grid — so rows differ only in the pooling.

```bash
python src/coarsen_bakeoff.py \
    --checkpoint <run_dir>/resnet_small_GeoTessera_v1.1_global_global-best.pt \
    --family resnet --preset small \
    --embedding-name tesserav1.1_global --embedding-dir /tessera/v1.1 \
    --year 2017 --city Nairobi \
    --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities \
    --target-res 320 100 --output-dir data/coarsen_bakeoff
```

A per-city checkpoint has seen the city it is scored on — pass `--splits val test` to restrict scoring to the patches its own grid split held out, and read the result as "which pooling is better", not "how good is this model".

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
| GeoTessera v1.1 global | `/tessera/v1.1/` (years 2015–2025) |
| GeoTessera v2 | `/tessera/v2/large_student/` (years 2017–2025) |
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

### GeoTessera v2 tile layout

```
v2/large_student/{year}/grid_{lon}_{lat}/
    grid_{lon}_{lat}.npy          ← (H, W, 128) int8
    grid_{lon}_{lat}_scales.npy   ← (H, W) float32
```

Same dequantization as v1.1, but **v2 ships no geoinfo tiff**. CRS and transform are derived
from the tile name (`datasets.tiles.tessera_grid_geometry`): the 0.1° cell centred on
`(lon, lat)`, reprojected into the UTM zone of that centre at 10 m. Verified identical to the
v1.1 geoinfo tiffs on 699 tiles, including the Norway/Svalbard/polar/dateline bands.

Coverage: after the 2026-09-06 tile top-up (40,075 -> 47,073 tiles for 2017) v2 fully covers
397,415 of the 400,673 So2Sat patches (99.2%), ahead of v1.1 global's 390,595 (97.5%).
Before that top-up it reached only 146,790 (36.6%). Full breakdown in
`data/tessera_v2_2017_so2sat_coverage.csv`, regenerable with:

```bash
python src/check_embedding_coverage.py \
    --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \
    --embedding-dir /tessera/v2/large_student --embedding-name tesserav2 --year 2017 \
    --out data/tessera_v2_2017_so2sat_coverage.csv
```

Because of those gaps, extract v2 with `--skip-partial-coverage` so patches straddling a
missing tile are dropped rather than written as truncated (and later stretched) crops.

---

## WandB

- **Project**: `lcz-classification-dl` (team: `phd-thesis-team`)
- Each run logs train/val loss and metrics per epoch, plus final test metrics
- Metrics (both pipelines): `test_acc`, `test_acc_macro`, `test_f1`, `test_f1_micro`, `test_kappa`, `test_loss`, per-class table, confusion matrix; segmentation adds `val_miou` / `test_miou`
- Run outputs (checkpoints, prediction rasters) are saved under `{output-dir}/{wandb-run-name}/`

---

## Step 8 — Paper & Poster Figures (R)

Summary figures are made in R (`R/`), run from the repo root so `.Renviron` supplies `DATA_DIR`.
The interpreter is the conda `r-environment`:
`/maps/acz25/miniconda3/envs/r-environment/bin/Rscript`.

R has no W&B client, so the results table caches its numbers first — the same pattern
`R/prepare_city_data.R` uses for the city summaries.

Output is filed by theme under `plots/` — `dataset/` (classes, splits, cities), `embeddings/`
(projection scatters), `models/` (results table, confusion matrices) and `maps/` (LCZ rasters).
Each script declares its own subfolder via `save_plot(..., subdir = PLOT_DIR_*)`, defined in
`R/constants.R`; the paths below are written relative to `plots/`.

```bash
# W&B -> data/model_metrics.csv (newest finished run per
# embedding x split x family x preset cell; supersedes are logged)
python src/export_run_metrics.py

# The segmentation campaign, same export: its metrics are the `*_patch_exact`
# ones -- the run scored on the So2Sat patches its prediction covers exactly,
# which is the only footing on which a pixel model and a patch model compare.
# `--include-unfinished` also emits the cells that are still training, with
# blank metrics, so the table shows them pending rather than not at all.
# (A segmentation run's split comes from its recorded `--split-mode`, not its
# config, and is mapped into the same vocabulary the classification rows use.)
python src/export_run_metrics.py --task segmentation --since 2026-09-10 \
    --include-unfinished --output data/seg_metrics.csv

# Saved test_confusion_matrix.npy run artefacts -> data/confusion_matrices.csv
# (R has no .npy reader in this env)
python src/export_confusion_matrix.py
# --run-root is repeatable and searched in order: while $DATA_DIR is full the
# campaign's run dirs are split across two filesystems, and which root holds a
# given run is an accident of when it was launched, not something to look up.
python src/export_confusion_matrix.py \
    --run-root ${DATA_DIR}/output/lcz-classification/dl --run-root /scratch/acz25/eo_fm/output

Rscript R/prepare_city_data.R   # -> data/so2sat_city_{summary,class_counts}.csv
Rscript R/plotting.R            # -> plots/dataset/class_*, dataset_*, so2sat_*
Rscript R/split_maps.R          # -> plots/dataset/split_map_{global,orig_test,orig_test_grid}.png
#   ... and each panel alone: split_map_{london,nairobi}_{cultural,gridded}.png
#   `gridded` always carries the 1,280 m split grid, drawn amber (GRID_COL) so it
#   reads as the cell lattice it is rather than as another patch outline
Rscript R/metrics_table.R       # -> plots/models/model_metrics_table.{png,pdf,html}
Rscript R/metrics_table.R --highlight dash|ring|halo|chip|bar|none   # best-value mark (default dash)
Rscript R/metrics_table_mirror.R   # -> plots/models/metrics_table_mirror.{png,pdf}
#   Both campaigns in one figure, mirrored about the Split column they share:
#   segmentation left, classification right, and outward from the axis on each
#   side embedding, model, # Params, the same four metrics (mIoU is the seg
#   table's own column: here it would face nothing across the axis). Alignment
#   is per (split, embedding) BLOCK -- as deep as its deeper half, shallower
#   half centred in it -- so the type size is the same on both sides, which is
#   the whole point of one table, and the leftover space is spread over four
#   bands instead of pooling into one gap. Each half is still normalised and
#   best-marked within itself: the two tasks are not one ranking. A star on the
#   left half's headers marks the metrics aggregated to the So2Sat patch -- they
#   share a header with the right half's, which is what makes them readable
#   across, and the star is what says they are not natively the same thing.

Rscript R/metrics_table.R --task segmentation   # -> plots/models/seg_metrics_table.{png,pdf,html}
#   The same table, same hues and same column heads, from data/seg_metrics.csv
#   plus an mIoU column -- one script with two TASK_PROFILES rather than a fork,
#   so the two figures cannot drift apart. Two kinds of blank, told apart by the
#   Model column: "(running)" marks a cell whose run has not finished, while the
#   mIoU column is empty for the whole gridded block because that split is
#   evaluated on So2Sat patches and has no per-pixel test set at all.

# -> plots/models/confusion_matrix_<run>_{proportions,counts}.png  (PNG only)
# Defaults to the best-kappa culture-10 run; --all loops every run in the CSV.
Rscript R/confusion_matrix.R [--run <run_name>] [--normalize true|none] [--all]

# A matrix with no row in the results table -- a segmentation run, an ad-hoc
# evaluation -- is named directly and cached beside the classification set
# rather than into it, so re-running the plain export cannot drop it:
python src/export_confusion_matrix.py --output data/confusion_matrices_seg.csv \
    --matrix seg-v2-global-unet-small=<run_dir>/test_confusion_matrix_patch_exact.npy
Rscript R/confusion_matrix.R --input data/confusion_matrices_seg.csv \
    --run seg-v2-global-unet-small --normalize true|none

# Embedding-projection scatter (PCA / UMAP / t-SNE), one point per patch
Rscript R/embedding_projection.R --list                        # runs available on disk
Rscript R/embedding_projection.R --run GeoTessera_v2 --all-colours [--legend [bottom|right]]
Rscript R/embedding_projection.R --run GeoTessera_v2 --method umap --full   # every patch
Rscript R/embedding_projection.R --run GeoTessera_v2 --colour lon|lat|lonlat --full --legend
# -> plots/embeddings/projection_<run>_<method>_<colour>.png  (PNG only; no key unless --legend)

# One row per patch: both embeddings' PCA/UMAP coords + location, LCZ, Koppen, M49
Rscript R/patch_table.R          # -> data/patch_master_table.parquet (--format csv also)

# An embedding as a picture: bare by default, no axes, ticks, legend or margin
Rscript R/embedding_raster.R --input <rgb.tif> --name <stem> [--bbox W,S,E,N | --window Nairobi]
Rscript R/embedding_raster.R --input <rgb.tif> --window Nairobi --patches --grid --name <stem>
Rscript R/embedding_raster.R --input <rgb.tif> --bbox W,S,E,N --city Nairobi --patches --grid \
    --name <stem>                       # overlays over an arbitrary bbox
Rscript R/embedding_raster.R --input <rgb.tif> --bbox W,S,E,N --axes --scalebar --name <stem>
# The ground itself, over exactly the frame an embedding image covers
Rscript R/embedding_raster.R --basemap Google.Satellite --bbox W,S,E,N --name <stem>
Rscript R/embedding_raster.R --basemap Google.Satellite --patch 006296 --city Nairobi \
    --name <stem>                       # ... or one So2Sat patch, by id
Rscript R/embedding_raster.R --basemap Google.Satellite --grid-id 911 --city Nairobi \
    --name <stem>                       # ... or one split-grid cell, by id
# What the label says is there, over the same frame: LCZ colours, nothing else
Rscript R/embedding_raster.R --labels --city Nairobi --patch 006296 --name <stem>
Rscript R/embedding_raster.R --labels --city Nairobi --grid-id 911 --name <stem>
Rscript R/embedding_raster.R --labels --city Nairobi --bbox W,S,E,N --name <stem>
Rscript R/embedding_raster.R --labels --rasterised --city Nairobi --bbox W,S,E,N \
    --name <stem>                       # the reference tif, not the polygons
Rscript R/embedding_raster.R --labels --rasterised --city Nairobi --grid-id 911 \
    --name <stem>                       # ... framed on a cell instead of a bbox
# ... or the labels over the imagery, the fill let down to 0.6 so both read
Rscript R/embedding_raster.R --labels --city Nairobi --grid-id 911 \
    --basemap Google.Satellite [--fill-alpha 0.6] --name <stem>
# Patch polygons instead of pixels, coloured from a projection run
Rscript R/embedding_raster.R --mosaic --run GeoTessera_v2 --city Nairobi \
    --colour rgb_pca|pca|umap|tsne --name <stem>
# --patches draws the 320 m label squares white and hairline, --grid the 1280 m
# split cells in amber at twice the weight: two overlays, two different questions
# Every bare panel is framed in grey15; --no-border makes the figure the panel exactly
# -> plots/rasters/<stem>.png   (plots/embeddings/ keeps the projection scatters)

# LCZ raster -> PNG for a lon/lat ROI
Rscript R/lcz_raster.R --input <file.tif> --bbox W,S,E,N --name <stem> [--legend]
Rscript R/lcz_raster.R --input <file.tif> --bbox W,S,E,N --name <stem> \
    --guppd --guppd-highlight Nairobi        # dim every other settlement
# ... on a web-map backdrop, which turns nodata and the panel transparent
Rscript R/lcz_raster.R --input <file.tif> --bbox W,S,E,N --name <stem> \
    --basemap [--basemap-zoom 13] [--basemap-provider Google.Satellite] [--basemap-alpha 1]
Rscript R/lcz_raster.R --list-basemaps       # providers, and which need a key
# Keyless: OpenStreetMap, every Esri.*, Google.{Satellite,Hybrid,Roads,Terrain}.
# Keyed:   the Stamen designs (Stadia hosts them since 2023) and CartoDB.* --
#          export STADIA_API_KEY=... (free tier) or pass --basemap-apikey.
# Several --input tiles are merged onto the first one's grid (the Demuzere map
# ships as 0.5-degree tiles that share a CRS but not a grid, and do not abut)
Rscript R/lcz_raster.R --input lcz_36.5_-1.5.tif lcz_37.0_-1.5.tif \
    --bbox W,S,E,N --name <stem> --no-resolution

# That raster's LCZ mix, as a pie plus a horizontal composition bar
Rscript R/lcz_composition.R --input <file.tif> --name <stem> [--bbox W,S,E,N] [--labels]
# -> plots/maps/<stem>_{pie,bar,composition}.png; --subdir composition puts them
#    in plots/composition/ instead, for marks that stand alone rather than
#    accompanying a map. The horizontal bar is a rule, not a chart: BAR_T_H in
#    R/composition.R keeps it a tenth as deep as it is long.
#
# The same mix for the reference map, over the same bounds, so the two are
# comparable. Demuzere et al. 2022 is one global GeoTIFF, so any city is a
# --bbox away and no tile merge is involved:
Rscript R/lcz_composition.R --subdir composition \
    --input ${DATA_DIR}/input/Demuzere_2022_complete/lcz_filter_v3.tif \
    --bbox=-0.62897183,51.27180859,0.40216756,51.78404384 --name london_demuzere
# lcz_filter_v3 (the Gaussian-filtered map the authors recommend) is the same
# file src/sample_unlabeled_patches.py draws its weak labels from. It is NOT the
# per-tile Demuzere_et_al_2022_LCZ/ set that plots/maps/nairobi_demuzere.png is
# drawn from: those tiles are reprojected to local UTM and differ by ~0.7% of
# cells over Nairobi, and they do not cover Bogota at all.

# ... or the same mix placed into the map itself
Rscript R/lcz_raster.R --input <file.tif> --bbox W,S,E,N --name <stem> \
    --distribution pie --pie-corner tl --pie-size 0.10
Rscript R/lcz_raster.R --input <file.tif> --bbox W,S,E,N --name <stem> \
    --distribution bar --dist-side bottom|top|left|right
```

`R/embedding_raster.R` is the figure end of `src/embedding_rgb.py`. The pixels have to be
coloured in Python — R cannot read the embedding sources at all (Tessera is int8 `.npy` plus a
separate scales array, AlphaEarth is zarr; only `src/datasets/tiles.py` opens them) — so
`embedding_rgb.py apply` paints an ROI into a 4-band uint8 GeoTIFF and this script draws it.
What it adds over that command's sidecar PNG is the R stack: `--window <city>` frames the panel
on **exactly the square `R/split_maps.R` draws** for that city, so an embedding image and the
black-and-white split panel show the same ground; `--patches` and `--grid` overlay the So2Sat
patch polygons and the 1280 m split-grid cells (over a `--window`, or over any `--bbox` once
`--city` names where to read them from); and the result goes through `save_plot()` like
every other figure. Uncovered ground keeps the GeoTIFF's alpha and stays transparent rather than
being painted — the shipped `Nairobi_v2_pca.tif` stops ~1.7 km short of the Nairobi window's
southern edge, and that shows as a gap, not as a colour.

For one patch or one grid tile there is no ROI to speak of, so that case lives on the Python
side: `python src/embedding_rgb.py image --input <patch.npy>` colours an extracted array
directly and writes the image at one file pixel per array pixel. It is worth knowing why the two
paths agree — the load/mask/dequantize order in `colour_array()` mirrors the fit-side
`sample_pixels()` exactly, because the extracted `.npy` are *not* uniformly dequantized (a
Tessera v2 patch is already real-valued; an AlphaEarth coop patch holds the stored integers as
float32 and must be dequantized here as it was during the fit). Checked on patch `006296` and
grid cell `911`: the npy path and the tile-mosaic path give **byte-identical colours**.

`--mosaic` draws patch polygons rather than pixels, and its two colourings are not the same
picture. `--colour rgb_pca` runs each patch's *pooled* vector through the same per-pixel colour
model the raster uses, so the two agree — it is the patch's mean colour, the flat version of the
image. `--colour pca|umap|tsne` uses the projection parquet's own patch-level basis, a different
fit of a different space, stretched per figure; those axes are comparable to nothing else,
including the raster beside them. Neither is wrong; reading one as the other is.

Both marks are drawn **without text**: they are graphical elements for a map, and the map's own
legend names the classes. `--labels` puts the class and share back on the standalone bar.

`--distribution pie` insets the pie in the panel on the **same translucent white backing as the
scale bar**, so the two read as one furniture set; `--pie-corner` and `--pie-size` (a fraction of
the panel width, default 0.10 — about the scale bar's own size, a mark on the map rather than a
second figure sitting on it) place and size it. `--distribution bar` butts a bare composition strip against
one edge — horizontal on `bottom`/`top`, vertical on `left`/`right` — spanning the map's full
width or height, with `--dist-thickness` as a fraction of the map's **width in either
orientation** (default 0.02, about 3.5 mm on a 7 in figure) — measuring it against the edge it
happens to span would make the strip down a landscape map's side thinner than the same strip
along its bottom, and the two should look alike. The strip is a separate
plot combined through patchwork, which aligns **panel regions**, so it lines up with the map
rather than with the figure and the tick labels do not push it out of register.

The inset pie must share the map's single `fill` scale (ggplot allows one per aesthetic, and
`ggnewscale` is not installed here), so it is keyed on the map's long class labels rather than
the short alt codes. The two always agree: both are counted from the same cropped raster.

`R/lcz_composition.R` counts a GeoTIFF's cells per class with `terra::freq()` (no decimation
needed — it never materialises the cells) and draws the **same two marks** the dataset figures
use: the pie from the city map and the stacked composition bar that acts as its legend, here
laid out horizontally. Both live in **`R/composition.R`**, shared with `R/plotting.R`, so the
standalone pie and the 52 pies on the city map are one mark and a change to either shows up in
both figures.

Shares are of **mapped area, not of patches** — a prediction raster and the patch table will not
agree, and should not. Cells are counted as stored, without reprojecting to an equal-area grid;
over a city ROI the latitude bias is under a tenth of a percent, which is why this is a city-ROI
tool. Labels on the horizontal bar are rotated a quarter turn, so what has to clear along the bar
is one *line height* rather than a line length — that is what lets all 17 LCZ labels sit on one
bar — and both that spacing and the figure's height are computed from the type at the chosen
`--width`. Shares under 0.1% get two extra decimals, since at one decimal nine of Nairobi's
seventeen classes read "0.0%", which says *absent* when the truth is *present and negligible*.

The `--guppd` overlay draws every GUPPD settlement footprint over the ROI, and a city ROI holds
many — 24 over Nairobi. `--guppd-highlight <name>` draws that one dark and thick and lets the
rest recede to a thin pale line; with no flag the **largest** settlement over the ROI is chosen
(the city, on the ROIs this figure is for) and reported, and `--guppd-highlight none` draws them
all alike. The name is matched against every GUPPD name field — `JRC_NAME_MAIN`, `CIESIN_NAME`,
`CIESIN_NAME_ADJ` and the comma-separated `JRC_NAME_LIST` — because they disagree: over Nairobi
the main agglomeration is `JRC_NAME_MAIN = Nairobi` while three smaller entities carry
`CIESIN_NAME = Nairobi`, and all four belong to the city.

The scale bar carries a **cell-size key** below the distance bar: a small filled square and the
raster's ground resolution, read off the tif (`--no-resolution` drops it). The square is a *key
drawn at a legible size*, not a to-scale cell — a 10 m pixel over a city ROI is a quarter of a
screen pixel. When the cells are anisotropic the key is drawn at their true aspect and labelled
with both sides: the So2Sat reference tifs are EPSG:4326 with a 320 m latitude step and an
independent longitude step, so at Nairobi they read **414x321 m**, not "320 m". If the raster had
to be aggregated to meet `--max-cells`, the key reports the aggregated cell actually drawn and
says so.

`model_metrics_table` is two stacked blocks sharing one header — culture-10 above, grid split
below — and each gets **its own hue and its own colour normalisation** (blue `#0072B2` and
terracotta `#946C51`). The grid split reuses So2Sat's own test cities and is leakage-inflated relative to
culture-10, so a shared ramp would read as a fair comparison and would flatten the contrast
inside each block.

The hues stay clear of the **LCZ palette**, since this figure shares a poster with the LCZ maps.
Verified by simulating protanopia/deuteranopia/tritanopia and measuring CIELAB dE to all 17 LCZ
colours: blue's worst case is dE 20.3 (LCZ 11) and terracotta's is 21.1 (LCZ 13). The Okabe-Ito
orange `#D55E00` used before scored **dE 2.2 against LCZ 3 under deuteranopia** — indistinguishable
— which is why it was replaced; muting the orange is what buys the distance, since a saturated
orange scores 2-7 at any lightness. Terracotta also separates from blue by dE 46 under the worst
simulation (a purple that cleared LCZ better managed only 19, because 255-315 degrees is the only
band LCZ leaves free and it sits close to blue). The tightest margin in the figure is terracotta
against the neutral grey of `# Params`, dE 18.8 — still clearly different, but do not mute the
hue further. `# Params` is the exception: one table-wide log10 grey ramp, since model size
is comparable everywhere and is not a performance metric. Pass `scale_by = "column"` to
`metrics_table_plot()` to see the shared-ramp alternative.

`R/confusion_matrix.R` is an R port of `src/training/evaluate.py::save_confusion_matrix` +
`style_lcz_ticklabels`: the tick labels are replaced by the short LCZ codes (`1`-`10`, `A`-`G`),
each set bold on a chip of its own class colour with the text colour chosen by luminance
(`contrast_text()`, the same rule the Python `_text_color()` applies). Two departures:

* **Not viridis.** The same CVD audit run against the ramp condemns it — viridis' green midtones
  are **dE 1.0 from LCZ A (Dense Trees) under tritanopia**, indistinguishable. The replacement is
  a white-to-magenta ramp (`CM_RAMP`), the only candidate scoring double digits against both the
  LCZ palette (dE 12.4, LCZ 10 under deuteranopia) and the metrics table's hues (dE 12.7, blue
  under protanopia). Measure a ramp on its *saturated half*: a white-ended sequential ramp must
  contain a pale sample near LCZ F, so scoring the whole ramp rejects every candidate. Its dark end
  is a deep plum, not black, because LCZ E *is* pure black.
* **The chips are geometry, not styled tick labels.** `ggtext`, `gridtext` and `marquee` are all
  absent from this env, so there is no `element_markdown()` to hang a background box on — the
  chips are `geom_tile` + `geom_text` outside the panel under `coord_cartesian(clip = "off")`,
  the same hand-built route the rest of the stack takes.

The matrix is always the full 17x17, unlike the Python plot's observed-labels axes, so two runs
compare cell for cell. Cells are sized in inches from the widest label they must hold, so the
counts variant does not overrun its tiles. Fills are normalised over the whole matrix, not per
row — the off-diagonal structure is only readable if a cell means the same thing wherever it sits.

`R/patch_table.R` merges everything onto one row per patch, keyed on `uid`, for downstream
analysis: identity (`uid, dataset, patch_id, split`), location (`lon, lat`), label
(`lcz_class, lcz_code, lcz_name`), settlement (`urban_rural, so2sat_city, guppd_city, guppd_id,
guppd_smod`), climate (`koppen_code, koppen_desc, water`), territory (`so2sat_country, un_country,
guppd_country, iso_a2, iso_a3, guppd_iso3, m49_code, un_subregion, un_region`) and one coordinate
block per projection run (`<run>_pca_1..3`, `<run>_umap_1..3`). 397,415 rows x 38 columns, ~45 MB
as parquet, gitignored and rebuildable.

**`urban_rural` is decided per patch against the full GUPPD**, all 123,034 settlement polygons —
not `big_cities_bbox.gpkg` (pop>1M) and not the 51 So2Sat cities, either of which would call a
patch in an unlisted town rural. A So2Sat city is a *square around* a city, so the countryside in
that square is real: **192,165 patches (48.4%) fall outside every settlement on Earth** and are
`Rural`, with `guppd_city`, `guppd_country`, `guppd_id` and `guppd_smod` all NA. Nothing is
filled from the nearest polygon — a field 20 km from Melbourne is not in Melbourne. The split
validates against the labels it never saw: **88–99.6% of patches in the built classes (LCZ 1–8,
10) are Urban, against 5.5–9.5% in the natural ones** (Dense Trees, Low Plants, Bush/Scrub), with
Sparsely Built (21.8%) and Water (15.7%) in between where they belong. 101 Urban patches sit in
polygons GUPPD gives no name at all, in either the JRC or the CIESIN field; they stay Urban with
a NA `guppd_city`.

**Three names for the place, and they disagree on purpose.** `so2sat_city` is the label So2Sat
gave a whole city bbox; `guppd_city` is the settlement the patch's own centroid falls in, and on
**14.8% of Urban patches the two differ** — Hong Kong patches in Guangzhou's polygon (the known
GUPPD merge), Vancouver's in Langley, Cologne's in Bonn, Düsseldorf and Wuppertal. Countries work
the same way: `so2sat_country` is the dataset's colloquial name, `un_country` the UN's formal long
name resolved by point-in-polygon (differing from So2Sat's on 25.8% of rows, naming only), and
`guppd_country` GUPPD's own name for the settlement — which differs from So2Sat's on only 0.1% of
Urban rows, all 170 of them Hong Kong, which GUPPD names as its own territory. Group by whichever
answers your question; do not assume any two agree row for row.

**The embedding blocks do not share a frame.** `geotessera_v2_umap_1` and
`alphaearthcoop_umap_1` are separate fits of separate feature spaces, on top of per-run
`StandardScaler` + PCA; no rotation relates them. Comparing a patch's position between blocks, or
computing a distance across them, is meaningless — what the join buys is per-patch comparison of
*structure* (neighbourhoods, cluster membership, how a class or region scatters). t-SNE is
excluded because it is fitted on a balanced per-class subsample, so only 34,000 of 397,415 rows
(8.6%) would carry a value; add `"tsne"` to `PROJ_DIMS` if you want it and expect the NAs.
`un_country` uses the **UN's formal long names** ("United States of America", "Türkiye"), which
differ from the colloquial names in the projection parquets' own `country` column on 25.8% of
rows — naming only, with no case where the two disagree about the actual country.

`R/embedding_projection.R` reads the parquets `src/embedding_projection.py` writes and draws the
first two dimensions of one projection, ~400k patches subsampled to a workable 60k
(`--max-points`). Three things it does that are not obvious:

* **The third coordinate is the depth channel.** Points are sorted by `<method>_3` before drawing
  (`--depth asc|desc|none`), so one range of depth values lands under the other. ggplot2 draws
  rows in data order, so sorting the frame *is* the encoding — there is no `order` aesthetic, and
  size or alpha would confound depth with a second channel. Runs written before 2026-09 carry only
  `<method>_x/_y`; those still plot, in file order, with one warning. Re-export to get `pca_1..3`
  (`--pca-keep`, default 3; UMAP and t-SNE additionally need `--fit-3d`, which is a *second*
  3-component fit, not a slice of the 2-D one).
* **The axes are framed on the central 99%** (`--clip`, `1` to disable). PCA on these embeddings
  grows a long thin arm — on AlphaEarth coop, 1,639 patches (0.41%) sit below PC1 = -10 while the
  rest of the cloud lives inside +-3 — and under `coord_fixed()` that arm squashes everything into
  a sliver. The arm is real signal, not corruption: 1,440 of those patches are LCZ 17 (Water), in
  coastal cities (Cape Town, New York, Mumbai). It is framed out, not removed, and the caption
  reports how many points fell outside. `--clip 0.999` is *not* enough to clear it.
* **High-cardinality colour variables collapse.** `--colour` takes any column; `lcz_name` and
  `split` use the canonical palettes, and anything else (city is 51 levels, country 30) keeps the
  12 largest and buckets the tail into a grey `Other`, so the key stays a fixed size whatever you
  colour by. The collapsed share is printed, never hidden.

Two more `--colour` modes classify each patch by **its own coordinates**, not by its city or
country name, and are built by `R/patch_geo_context.R` and cached to `data/patch_geo_context.csv`
alongside the GUPPD block above (delete the file to rebuild; a patch's coordinates never change —
the two lookups refill independently, so adding one does not redo the other):

* `subregion` — **UN M49 sub-region**, by point-in-polygon against `spData::world` then a join to
  the UN's own M49 table. Note the 22 familiar sub-regions are *not* the CSV's `Sub-region Name`
  column, which is the 17-member tier that lumps all of sub-Saharan Africa together — they come
  from taking `Intermediate Region Name` wherever the UN defines one (Africa and the Americas)
  and the sub-region otherwise. Colours are **one hue family per region** shaded within it, so
  the key reads as five continental blocks rather than 22 unrelated hues.
* `koppen` — **Köppen-Geiger class** (Beck et al. 2023) sampled from the 1 km 1991–2020 raster,
  in the official RGB colours parsed straight out of the shipped `legend.txt`.

**Water is a `koppen` category, and only a `koppen` one.** The Köppen raster is land-only, so a
patch whose centre falls on ocean or a large lake samples nothing — 37,093 patches, 9.3%. That NA
*is* the answer for a climate reading, and those patches get a slate `Water` entry rather than a
borrowed classification (an earlier version filled them from the nearest land cell within ~28 km,
which quietly gave a harbour patch its city's climate). `subregion` deliberately has no such
class: a sub-region is a *territorial* label, not a physical one, so a patch of harbour belongs
to the country whose harbour it is and simply takes the containing — or nearest — country's
sub-region. `spData::world` is 1:110m, so 31,366 patches (7.9%) fall outside every country
polygon; that is coastline coarseness rather than water, and nearest-country resolves it, largest
distance 37 km. Neither mode is collapsed to top-N: the full classification is the point.

**Both keys are laid out so like sits with like.** `guide_legend` fills a plain rectangle, so
consecutive groups run into each other mid-row; invisible spacer levels (unique whitespace
strings, transparent swatch, blank label — hence `drop = FALSE`) push each group onto a fresh row
or column. Köppen shows **codes only**, four rows, padded so each column is one main group
(A / B / C / D / E / Water) — the full names run to 45 characters and force a key wider and taller
than the panel. The M49 key pads to **one region per row** (five rows, Africa to Oceania), since
those labels cannot be shortened.

`--colour` also takes three **geographic** modes, derived from the `lon`/`lat` columns the
exporter writes: `lon` on a **RdYlBu** ramp, `lat` on **BrBG**, and `lonlat` on a **bivariate**
scale. Both univariate ramps are diverging on purpose — longitude and latitude each have a
meaningful middle (prime meridian, equator) and two opposed directions, which a sequential ramp
throws away. **All three are pinned so 0 is the exact centre**: `.unit_mid()` for the bivariate
axes, and for the two bars the coordinate's *full* range — `±180` longitude, `±90` latitude —
rather than the data's own. Full range is what puts the interior breaks on the round graticule
values (60/120 and 30/60) with the compass letters clear of them at the band ends, and it makes
two runs' colours mean the same thing; the cost is contrast, since So2Sat spans only −123..151
and −38..56 so neither ramp reaches its extremes. Band ends carry **W/E** and **S/N**, interior
breaks are bare degrees, and the bivariate key carries all four letters on its sides plus a
dashed cross on the origin. Seven labels share one bar, so the legend text is set at 0.78 of the
inherited size — ggplot silently *drops* colliding bar labels rather than shrinking them. The bivariate colours are a bilinear blend of four corners **in CIE Lab**, not sRGB:
blending two saturated hues in sRGB runs through a muddy dark middle, and mid-range is exactly the
quadrant most points land in. A bivariate scale has no ggplot guide, so its key is a small 2-D
square of tiles inset into the panel corner with `patchwork::inset_element` — the same technique
`R/plotting.R` uses for its map insets, and it renders **only if patchwork is attached**, since
`ggplot + inset_element(...)` dispatches on patchwork's own `+`. That key is drawn even under the
default `--legend none`: unlike a legend it costs no layout, and without it the figure cannot be
read at all.

**There is no colour key and no caption by default** — pass `--legend` and `--caption` for them.
`--legend` bare means **bottom**, which is what a square panel wants: a bottom key takes height
rather than stealing width from the cloud, and the figure grows by however many rows the key
wrapped to (the plot reports that as a `legend_rows` attribute, since only the caller sizes the
figure). Columns are chosen from label length — 3 for long labels, 6 for short — because ggplot
will otherwise run the last entries off the edge. `--legend right` gives the old side key.
These scatters are panelled beside figures that already carry the LCZ key and their own titling,
and a 17- or 13-entry legend takes more width than the cloud it explains; bare, the figure squares
up to 6.2 in and the panel gets the whole frame, leaving just the axis names. Nothing is lost:
run, method, `n`, colour variable, depth column and the count outside the frame all go to the
console on every call. Note that ggplot silently *clips* an overlong caption at that width rather
than wrapping it, which is why `--caption` wraps at 72 characters.

Subsampling is by `crc32(uid) %% 1000`, reproducing `stable_subsample_mask` in the Python
exporter, so the same patches are drawn on every run and two runs stay joinable point for point —
`patch_id` alone is **not** unique, it restarts at `000000` in each of training/validation/testing.
`scattermore` and `ggrastr` are absent here, which is what sets the 60k default; `--full` reads the
full parquet instead of the `_sample` sibling and lifts the cap, drawing every patch (397,415 for
GeoTessera v2, ~7 s), which is worth it — the subsample loses the fine structure of the dense
regions. `plot_density()` on the pre-binned density grid is still the cheaper whole-cloud view. t-SNE is fitted on a balanced per-class
subsample, so most rows have `NA` t-SNE coordinates by construction and the drawn count is much
smaller — the script reports it.

The table is expected to grow as runs land: columns, column widths, figure size, split hues and
the caption are all derived from the CSV and from `METRIC_COLS` at draw time, and every label
lookup is a preference rather than a filter, so a new embedding, split or architecture appears
with its raw name instead of vanishing. Adding a metric is one entry in `METRIC_COLS` plus one in
`METRIC_GLOSS`.
