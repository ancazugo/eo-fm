# eo-fm — LCZ Classification Pipeline

Earth Observation Foundation Model pipeline for Local Climate Zone (LCZ) classification using satellite embedding tiles.

---

## Setup

```bash
source /maps/acz25/envs/eo_fm-env/bin/activate
uv sync
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
1. Download embeddings        extract_so2sat_embeddings.py / extract_grid_embeddings.py
        ↓
2. Extract patch/grid npy     extract_so2sat_embeddings.py / extract_grid_embeddings.py
        ↓
3a. Patch classification      patch_classification.py   (ResNet per-patch)
3b. Semantic segmentation     semantic_segmentation.py  (U-Net per-grid-tile)
        ↓
4. ROI inference              infer_roi.py              (auto-called at end of training)
```

All scripts are run from the repo root (`/home/acz25/repos/eo_fm`). Scripts are in `src/`.

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

Always pass `--dequantize` for `alpha_earth_coop` and `seamless` in training and inference commands.

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

Trains a ResNet patch classifier on pre-extracted So2Sat patch `.npy` files. Uses the grid-based train/val/test split from `patches_reference_{city}_split.gpkg`. After training, automatically runs full-ROI inference via `infer_roi.py`.

**ResNet presets:** `nano` · `tiny` · `small` · `base` · `large` (resnet10t → resnet152)

```bash
# AlphaEarth COOP — London, small preset
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

# GeoTessera v1.1 — London (no --dequantize; already float32 after extraction)
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

# Embedded Seamless Data — London
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
- `--preset` — ResNet size (`nano`/`tiny`/`small`/`base`/`large`)
- `--patch-size` — resize input patches to this square (pixels); use 32 for 10 m embeddings
- `--dequantize` — apply embedding-specific dequantization at load time
- `--checkpoint` — skip training, load weights and run inference only

**Output per run** (under `--output-dir/{wandb-run-name}/`):
```
resnet_{preset}_{output_name}_{city}-best.pt   ← best checkpoint
{run_name}_resnet-{preset}-classification-prediction_{city}.tif
{run_name}_resnet-{preset}-classification-prediction_{city}.png
```

---

## Step 3b — Semantic Segmentation

### `semantic_segmentation.py`

Trains a U-Net segmentation model on pre-extracted grid tile `.npy` files. Label masks are rasterized on the fly from `patches_reference_{city}_split.gpkg` polygons. After training, automatically runs full-ROI inference via `infer_roi.py`.

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
- `--preset` — U-Net size preset (see table above)
- `--dequantize` — apply embedding-specific dequantization at load time
- `--checkpoint` — skip training, load weights and run inference only

**Output per run** (under `--output-dir/{wandb-run-name}/`):
```
unet_{preset}_{output_name}_{city}-best.pt
{run_name}_unet-{preset}-segmentation-prediction_{city}.tif
{run_name}_unet-{preset}-segmentation-prediction_{city}.png
```

---

## Step 4 — Standalone ROI Inference

### `infer_roi.py`

Runs inference over an arbitrary bounding box from raw source embedding tiles (no pre-extracted grid files needed). Called automatically at the end of `patch_classification.py` and `semantic_segmentation.py`, but can also be run standalone.

Uses a sliding window with Hanning-weighted logit blending (segmentation) or majority vote (classification). Reprojects each tile's predictions into a single output GeoTIFF.

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
- Metrics: `val_miou` / `test_miou` / `test_acc` (segmentation); `val_f1` / `test_f1` / `test_acc` (classification)
- Run outputs (checkpoints, prediction rasters) are saved under `{output-dir}/{wandb-run-name}/`
