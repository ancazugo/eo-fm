# So2Sat Segmentation — Experimental Plan (2017 embeddings)

**Task**: 17-class LCZ *semantic segmentation* at 10 m from AlphaEarth and Tessera v1.1 2017
embeddings, supervised by rasterised So2Sat-LCZ42 v4 patches under masked loss.
**Primary metric**: Cohen's kappa, reported at **patch level** so it is directly comparable to
the patch-classification campaign (`docs/global_lcz_campaign_2026-07.md`).

**Reference numbers to beat** (global split, 23,858 aligned test patches):

| | Test kappa | OA | Macro-F1 |
|---|---|---|---|
| Best single patch model (`student-noisy-v3`, tessera) | 0.6497 | 0.6794 | 0.5656 |
| Weighted ensemble, LOCO-honest | 0.6871 | 0.7154 | 0.6038 |
| + aux offset corrector, LOCO-honest | **0.7055–0.7081** | 0.7320–0.7343 | 0.6129–0.6227 |

---

## 0. What this plan changes relative to the current repo

Four gaps, in priority order. Everything else reuses the existing registry / loop / eval stack.

| # | Gap | Where |
|---|---|---|
| G1 | Segmentation has **no city-level split**. `utils/grid_split.py` only does within-city macro-block splits, which §7 of the campaign report quarantines as autocorrelation-inflated. | new `--split-mode {grid,global}` in `semantic_segmentation.py` |
| G2 | **No patch-level aggregation** of segmentation output, so seg results are not comparable to the 0.6497 / 0.6871 / 0.7055 ladder. | new `evaluate_segmentation_as_patches()` in `training/evaluate.py` |
| G3 | **No LCZ-specific metrics** (OAu / OAbu / OAw / weighted kappa). Needed for comparability with LCZ Generator factsheets and Demuzere et al. 2022. Similarity matrix now vendored at `docs/lcz_class_similarity.csv`. | new `lcz_metrics.py` + hook into `_make_metrics()` |
| G4 | Aux structural rasters (GHS-BUILT-H, ETH canopy height) enter only as **13 post-hoc zonal scalars**. In a segmentation formulation they are natively rasters and should be **input channels**. | `datasets/grid_tiles.py` channel stacking |

---

## 1. Data preparation

### 1.1 Label raster

Rasterise So2Sat v4 patches once per city onto the **embedding grid**, not a fresh grid —
nearest-neighbour the labels onto AlphaEarth's / Tessera's native 10 m pixels. Resampling
64/128-dim embeddings is lossy and buys nothing.

Existing convention holds: raw `1-17` → `0-16`, nodata `0` → `-1`, `ignore_index=-1`.
`rasterize_polys()` in `datasets/grid_tiles.py` already does this from the per-city `.gpkg`.

Emit **three aligned rasters per city**:

| Raster | dtype | Purpose |
|---|---|---|
| `lcz_label` | int8 | class index, −1 elsewhere |
| `patch_id` | int32 | So2Sat patch identity, −1 elsewhere |
| `split_id` | uint8 | 0 = train / 1 = val / 2 = test / 255 = excluded |

`patch_id` is the one that is easy to skip and expensive to retrofit — it is what makes G2
possible, and it is how you assert afterwards that no patch straddles a split boundary.

**Erode each patch footprint by 2 px** before burning. So2Sat patch edges inherit the
polygon-digitisation slop that WUDAPT explicitly tolerates (>100 m buffers between LCZs,
geometric accuracy of boundaries deemed non-critical), so edge pixels are unreliable by
construction.

### 1.2 Embeddings

| Source | Channels | 2017 global coverage | Role |
|---|---|---|---|
| AlphaEarth (`coop`) | 64 | yes, 2017 is the first annual layer | **primary** — only source covering all 52 cities |
| Tessera v1.1 | 128 | partial (Europe / US complete; rest rolling out) | second arm on the covered subset |

Pin `(version, variant)` in the run name and never mix: the geotessera docs are explicit that
1.0 and 1.1 feature spaces are independently learned and not interchangeable. Submit a
GeoTessera embedding request for the missing city bounding boxes now — it is a long-lead item
and it does not block anything else.

Caveat to record in the paper: 2017 is AlphaEarth's thinnest year (Sentinel-2B only reached
operations in March 2017), so the underlying time series is sparser than later layers.

### 1.3 Tiling

- **Tile size 128 × 128 px (1.28 km)**, not the current `--patch-size 64` default. A 64 px tile
  holds at most ~4 patch footprints; LCZ is defined at hundreds of metres to several km, and
  the encoder needs enough context to see the neighbourhood, not the block.
- Discard tiles with < 1 % labelled pixels so batches are not mostly `ignore`.
- **Tile split purity is mandatory**: a tile whose pixels carry more than one `split_id` is
  dropped, not majority-assigned. This is the segmentation-specific leak that has no analogue
  in the patch pipeline.

---

## 2. Splitting strategy

**Split by city. Inherit the So2Sat culture-10 assignment.** City membership survives
rasterisation untouched — the split is a property of the city, not of the tile. The tile grid
is a compute artefact and carries no evaluation meaning.

```
train   : the 42 So2Sat cities, minus the 6 validation cities below
val-inner: 6 of the 42, held out for early stopping / LR selection
val     : western halves of the 10 culture cities   (protocol-compat only)
test    : eastern halves of the 10 culture cities
```

Three notes:

1. **Add `val-inner`.** The campaign report's own §5 audit is the argument: val and test share
   the same 10 cities (median test→val patch distance 2.55 km), so anything selected on `val`
   can read city-conditional structure off `test`. Early stopping is a low-DoF fit, but it is
   not zero-DoF. Selecting on 6 held-out *training-pool* cities costs nothing and removes the
   objection pre-emptively.
2. **Buffer the val/test meridian.** With 128 px tiles the receptive field is ~1.3 km against a
   2.55 km median separation, so the exposure is small but non-zero. Drop tiles intersecting a
   ±1.3 km strip either side of each city's east/west dividing line.
3. **Retire the grid split for headline numbers.** Keep `--split-mode grid` for per-city
   demos; §7 already quarantines those results and this plan does not rehabilitate them.

Run **leave-one-city-out over the 10 test cities** for anything with fitted parameters
(ensemble weights, aux corrector). `ensemble_stacking.py --city-holdout` already implements
the protocol; the segmentation probs just need to be cached in the same layout.

---

## 3. Architecture

Start with `--family unet --preset small`. Deliberately modest, for three reasons drawn from
the existing results.

- **The input is already a learned representation.** 64–128 channels encoding a full year of
  time series means most representational work is done. Two or three downsampling levels is
  enough; what matters is that the receptive field reaches LCZ scale, not that the network is
  deep.
- **Capacity has already lost once here.** resnet101/152 did not beat resnet34 on the patch
  task. There is no reason to expect a deeper `resnet_unet` to behave differently, and the
  effective sample size is the number of *patches* (~400k), not pixels.
- **The recipe transfers.** Reuse opt3: batch 256, wd 1e-3, lr 5e-4 with 3-epoch warmup +
  cosine, label smoothing 0.1, `sqrt_inv_freq` class weights, dihedral TTA at eval, max 50
  epochs / patience 10. Monitor `val_kappa`, not `val_miou` — see §4.

Two deviations from the patch recipe:

- **Drop mixup.** §3 of the campaign report showed mixup + label smoothing caps teacher
  confidence at ≈0.9 and leaves models badly under-confident (fitted T as low as 0.46). Under
  masked loss with sparse labels this is worse, not better, and it will poison any downstream
  confidence gating. Keep label smoothing, drop mixup.
- **Set `--dice-weight 0.0` initially.** Dice on sparsely-labelled tiles optimises a quantity
  whose denominator is the labelled subset, which is not the segmentation objective you want.
  Add it back as an ablation, not a default.

### 3.1 Aux channels (do this in the first pass, not as Phase 3)

§8b established that 13 structural scalars alone beat the entire seamless FM model under LOCO
(0.579 vs 0.506), and that 50–58 % of residual ensemble error sits on structurally-defined
class pairs. In the segmentation formulation GHS-BUILT-H ANBH and ETH 10 m canopy height are
**already rasters** — concatenating them as input channels is strictly more informative than
recovering them as per-patch zonal means afterwards, and it costs one line in the channel
stacker.

Ablate: embeddings only → embeddings + built height → embeddings + built height + canopy.

---

## 4. Evaluation

It is still classification, just dense. The metric family barely changes; the **unit of
aggregation** is what changes, and that is where the reporting goes wrong.

### 4.1 Report at two levels

**Patch level (headline).** Mean softmax over each `patch_id` footprint → argmax → score
against the patch label. This is the number that goes in the comparison table against 0.6497 /
0.6871 / 0.7055. Restrict to the 23,858 aligned test patches so the comparison is exact.

**Pixel level (secondary).** Standard per-pixel metrics on labelled pixels only. State
explicitly that this is a *masked* evaluation: IoU computed over labelled pixels only is not
IoU against the true map, because the union is restricted. It is closer to a macro-F1 than to
the mIoU reported by papers scoring against dense reference maps — which is precisely why the
Swiss AlphaEarth/Tessera LCZ numbers (IoU 0.59–0.69 / 0.77–0.82) **are not comparable to
yours** and should not be tabled as if they were.

**Bootstrap over patches and cities, never over pixels.** Effective *n* is the patch count.

### 4.2 Metric suite

Existing (`_make_metrics`): OA, macro accuracy, macro/micro F1, kappa, mIoU, per-class table,
confusion matrix PNG. **Add the WUDAPT set** so results are comparable to LCZ Generator
factsheets and to Demuzere et al. 2022.

All four new metrics are pure functions of the 17×17 confusion matrix `cm`, which the
evaluators already build. Put them in a new `src/training/lcz_metrics.py` and call it once
from `_evaluate()` after `cm` is materialised — do **not** implement them as streaming
torchmetrics.

| Metric | Definition |
|---|---|
| `OAu` | OA restricted to the urban classes, LCZ 1–10 (rows *and* columns) |
| `OAbu` | OA after collapsing to built (1–10) vs natural (A–G) |
| `OAw` | similarity-weighted OA (Bechtel et al. 2020) |
| `kappa_w` | chance-corrected analogue of `OAw` — **new primary metric** |

#### The similarity matrix

Source: `docs/lcz_class_similarity.csv`, transcribed from the supplementary XLSX of
Bechtel, Demuzere & Stewart (2020), *Remote Sensing* 12(11):1769,
<https://doi.org/10.3390/rs12111769>.

Underlying construction (for the methods section, not for implementation): a 12-point
diagnostic scheme scoring dissimilarity over four surface characteristics — openness
(compact/open/sparse/nil), height (high/medium/low/nil), land cover (pervious/impervious) and
surface objects (buildings/plants). One point per degree of separation on openness and height;
three points per degree on land cover and surface objects. Summed 0–12, normalised to [0,1],
inverted, symmetrised. Worked example from the paper: LCZ 1 vs LCZ 4 scores 1 (compact vs
open) + 0 (both high-rise) + 3 (impervious vs pervious) + 0 (both building classes) = 4.

**Loader contract — assert all of these on load, do not silently coerce:**

- shape `(17, 17)`, float
- `np.allclose(W, W.T)` — the metric is defined as symmetric
- `np.allclose(np.diag(W), 1.0)`
- `0.0 <= W.min()` and `W.max() <= 1.0`
- **row/column order is `1..10, A, B, C, D, E, F, G` mapped to indices `0..16`**, matching
  the existing `1-17 → 0-16` label convention. This is the single most likely silent bug:
  a transposed or alphabetically-sorted CSV will still produce plausible-looking numbers.
  Add a unit test asserting `W[0, 3]` (LCZ 1 vs 4) is materially higher than `W[0, 11]`
  (LCZ 1 vs B) — that ordering check catches every permutation error worth catching.

**Version-pin it.** The 2020 paper states the scheme is under development and that newer
versions carry slightly updated scores, so LCZ Generator factsheets may use a later matrix.
Record the source and revision in the CSV header comment; without that, `OAw` is not
comparable to published factsheet values.

#### Formulas

```python
def oa_weighted(cm, W):
    """Bechtel et al. 2020: WA = (1/N) · Σᵢⱼ wᵢⱼ cᵢⱼ. OA is the W = I special case."""
    return float((W * cm).sum() / cm.sum())

def kappa_weighted(cm, W):
    """Chance-corrected OAw. Reduces to Cohen's kappa when W = I."""
    N = cm.sum()
    E = np.outer(cm.sum(1), cm.sum(0)) / N        # expected under independence
    po = (W * cm).sum() / N
    pe = (W * E).sum() / N
    return float((po - pe) / (1.0 - pe))
```

Note `torchmetrics.MulticlassCohenKappa` cannot do this — its `weights` argument only accepts
`linear`/`quadratic`, both of which assume ordinal classes. LCZ's 17 types are not ordinal.

#### Reporting rules

- **`kappa` stays primary** for the comparison against the 0.6497 / 0.6871 / 0.7055 ladder.
  `kappa_w` is reported alongside it, not instead of it.
- The **gap between `kappa` and `kappa_w`** is itself a result. Given that §8b attributed
  50–58 % of residual ensemble error to structurally-defined class pairs, expect `kappa_w`
  to be substantially kinder. Report the delta per city: a large gap means errors are
  within-family and forgivable; a small gap means built↔natural confusion, which is a real
  failure.
- **Do not put `W` in the loss and then headline `OAw`.** Similarity-smoothed soft targets are
  implementable via the existing marginalised-CE contract, but the model is currently already
  running `sqrt_inv_freq` class weights *and* label smoothing 0.1 — both reshape the target
  distribution. Phase 0 established that stacking a second imbalance mechanism double-corrects
  (sqrt-freq sampler and logit adjustment both lost). Treat `W`-in-loss as a late ablation with
  unweighted kappa as the reported metric, not as a default.
- Always emit the full 17×17 confusion matrix and **per-city metrics with spread**, never
  pooled alone. The per-city LOCO spread on the patch task ran Munich 0.90 down to Nairobi and
  Santiago at 0.47; a pooled number hides exactly the signal you care about.
- WandB: log `test_oaw`, `test_kappa_w`, `test_oau`, `test_oabu` at both aggregation levels,
  suffixed `_patch` and `_pixel` per §4.1.

### 4.3 Comparability ledger

| Comparison | Comparable? |
|---|---|
| This repo's patch-classification ladder | **yes**, via §4.1 patch aggregation on the aligned test set |
| LCZ Generator factsheets / WUDAPT maps | **yes**, with their protocol: 25 bootstraps, 70/30 stratified polygon splits preserving class frequency |
| Demuzere et al. 2022 global 100 m map | **yes**, aggregate the 10 m prediction to 100 m over held-out cities |
| Swiss AlphaEarth/Tessera LCZ paper | **no** — dense reference maps vs sparse expert patches |
| GEO-Bench `m-so2sat` | **no** — ~22 k subset, different splits and protocol |

---

## 5. Ablation ladder

Rows 3→4 differ only in features; that contrast is what isolates the foundation model's
contribution and is what makes the result defensible.

| # | Run | Purpose |
|---|---|---|
| 1 | LCZ Generator RF @ 100 m | community baseline |
| 2 | RF on raw S1/S2 2017 annual composites, same tiles | features baseline |
| 3 | U-Net on raw S1/S2 composites | **architecture matched, features not** |
| 4 | U-Net on AlphaEarth 2017 | primary |
| 5 | U-Net on Tessera v1.1 2017 (covered subset) | second embedding |
| 6 | U-Net on AlphaEarth + aux rasters | §8b carried into the seg formulation |
| 7 | Softmax-average ensemble over 4 + 5 (+ seamless) | §8 lesson 1 |

Expected ordering from the existing evidence: 7 > 6 > 4 ≈ 5 > 3 > 2 ≈ 1. **Cross-embedding
ensembling was the single biggest lever at every stage of the patch campaign (+2–4 pts) while
every within-model lever lost** — budget for row 7 from the start rather than treating it as a
stretch goal.

---

## 6. Commands

```bash
# Primary: AlphaEarth 2017, city-level global split
python src/semantic_segmentation.py \
    --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities --cities all \
    --output-name AlphaEarth_seg_global --year 2017 --label-source gpkg \
    --family unet --preset small --patch-size 128 \
    --split-mode global --val-inner-cities 6 --buffer-km 1.3 \
    --embedding-name alpha_earth_coop \
    --embedding-dir ${DATA_DIR}/input/Google/AlphaEarth/coop \
    --dice-weight 0.0 --no-mixup \
    --output-dir ${DATA_DIR}/output/lcz-segmentation/dl

# Second arm: Tessera v1.1 on the covered subset
python src/semantic_segmentation.py ... \
    --embedding-name tesserav1.1 \
    --embedding-dir ${DATA_DIR}/input/GeoTessera/v1.1/2017

# Smoke test first
python src/semantic_segmentation.py ... --preset nano --max-epochs 1 --no-wandb --cities Nairobi
```

---

## 7. Order of work

1. `patch_id` raster emission (§1.1) — blocks G2, cheap, do it first.
2. `--split-mode global` + tile purity + buffer (§2) — blocks every headline number.
3. Patch-level aggregation evaluator (§4.1) — blocks all comparability.
4. Rows 3, 4 of the ladder. Stop and check row 4 beats row 3 before building anything else.
5. `lcz_metrics.py` — loader asserts, `OAu` / `OAbu` / `OAw` / `kappa_w` (§4.2). Independent
   of everything above, so it can be done in parallel; it is pure confusion-matrix maths and
   should land with unit tests (identity-matrix reduction to OA and kappa, plus the class
   ordering check).
6. Aux channels (§3.1), then the ensemble (row 7).

Step 4 is the real gate. If a U-Net on AlphaEarth does not beat the same U-Net on raw S1/S2
composites, the problem is in the pipeline, not in the idea — and every later row inherits it.
---

## 8. Execution log — foundation pass (2026-08-20)

This section records what was built, what the repo disagreed with, and what was
deliberately deferred. Sections 0–7 above are the original plan and are left
unedited; where they conflict with this section, this section is current.

### 8.1 Decisions taken before building

1. **Scope: code foundation only.** The single T4 was at 100 % utilisation with
   7 processes — the same contention that paused PLAN-v3 Phase 2 revC. No
   headline training runs and no launchers were created; the GPU stays with
   Phase 2.
2. **Ablation rows 1–3 deferred entirely** (§5). There are no raw S1/S2 2017
   composites on disk and no `sentinel*` entry in `EMBEDDING_REGISTRY`, so
   "U-Net on raw S1/S2" — the plan's own declared gate at §7 step 4 — cannot be
   run without a multi-week GEE export for 51 cities. **The gate is therefore
   still open: nothing below establishes that the foundation model earns its
   keep against a features baseline.**
3. **Guangzhou is test-only.** It is the one city carrying both `training`
   (5,540 patches) and `testing`/`validation` (2,402 / 2,407), so §2's "train :
   the 42 So2Sat cities" overlaps its own 10-city test set. Its training
   patches are dropped, keeping all 10 test cities so the comparison against
   the published ladder stays exact.
4. **The opt3 recipe is expressible but not baked in.** Phase 2 Task 2.1c
   exists because opt3 looks mis-specified on the cultural split (2 of 3 seeds
   peak at epoch 2, during warmup). The flags were added; the schedule choice
   is deferred to launch time.

### 8.2 Where the plan and the repo disagreed

| §  | Plan said | Actual |
|---|---|---|
| 1.3 | tile size is `--patch-size 64`, raise to 128 | `--patch-size` is the **inference** sliding-window knob. Training tiles are already 128 px (`create_city_grids.py --sub-tile-size 1280`). No change needed. |
| 2 | val = western halves, test = eastern halves; buffer a ±1.3 km meridian | **There is no meridian.** val and test x-ranges overlap heavily in every culture city — in Santiago the validation range sits entirely inside the testing range. Implemented instead as a *proximity* buffer: drop a tile within `--buffer-km` of a patch in a different split. Same protection, no assumed geometry. |
| 6 | `--embedding-name tesserav1.1`, dir `GeoTessera/v1.1/2017` | `tesserav1.1` is `status="deprecated"` and rejected by `available_embeddings()`; that directory is an archive, not a tile store. Use **`tesserav1.1_global`** with `--output-name GeoTessera_v1.1_global` (already extracted for 51 cities, 17,530 npy). |
| 6 | `--cities all`, "all 52 cities" | `--cities` filters by exact directory name; there are **51** cities. `--cities all` is now accepted as "every city". |
| 3 | drop mixup (`--no-mixup`) | `LCZUNetModule` never had mixup. No flag needed; the reasoning is recorded in its docstring so it is not added later by accident. |
| 3 | "the recipe transfers" | It did not. The seg task had no class weights, no label smoothing, no configurable monitor and no `val_kappa`; `--warmup-epochs`, `--tta` and channel normalisation were all unwired. Added (except normalisation, still absent). |
| 4.2 | call the LCZ metrics "after `cm` is materialised" | `_evaluate` never materialised a dense matrix — `save_confusion_matrix` used `labels=present` and returned only a path, and subsampled at 2 M pixels. Added `dense_confusion_matrix`, computed on the full stream. |
| 4.2 | suffix every metric `_patch` / `_pixel` | Unsuffixed now means **pixel** (preserving the historical `test_acc` / `test_kappa` keys that existing WandB runs use); `_patch` is the new aggregation and `_100m` the coarse suite. Suffixing only the four new LCZ metrics while `test_acc` stayed bare would have been worse than either convention. |
| G2 | "no patch-level aggregation" exists | `src/eval_seg_on_patches.py` already scored seg checkpoints on isolated 32×32 patches. It is context-free, so the in-tile `patch_id` pooling was still built — but the gap was narrower than stated. |
| G4 | aux rasters need channel-stacking work | The **code was already there** (multi `--output-name` fusion, `aux_struct` registry entry, `precompute_aux_tiles.py`). What is missing is the *extraction*: 0 of 51 cities have `aux_struct` grid npys. |

### 8.3 What was built

- **`src/training/lcz_metrics.py`** — `load_similarity_matrix` (asserts shape,
  symmetry, unit diagonal, range and class ordering), `oa_urban`,
  `oa_built_natural`, `oa_weighted`, `kappa_weighted`, `lcz_metrics_from_cm`.
  `docs/lcz_class_similarity.csv` gained the provenance header §4.2 requires.
- **`src/utils/city_split.py`** — `assign_city_roles` (continent-stratified,
  size-aware) and `tile_splits_for_city` (tile purity, proximity buffer,
  minimum labelled fraction, the Guangzhou rule).
- **`src/datasets/grid_tiles.py`** — 2 px patch erosion, an aligned `patch_uid`
  raster, `--split-mode global` plumbing, and a `culture_val` split with its own
  dataloader so `ensemble_stacking.py --city-holdout` can be fed later.
- **`src/training/evaluate.py`** — `dense_confusion_matrix`, `save_metrics_json`,
  the LCZ suite wired into `_evaluate`, TTA forwarded to segmentation,
  `evaluate_segmentation_as_patches` (§4.1 headline aggregation, caching
  `probs_patch.npz` in the `ensemble_eval` layout), and `per_city_metrics`.
- **`src/training/tasks.py`** — class weights, label smoothing, configurable
  `monitor` with `val_kappa` / `val_f1`, and the missing all-nodata guard in
  `val_step` (an all-`-1` val tile used to NaN-poison `val_loss` for the epoch).
- **`src/semantic_segmentation.py`** — `--split-mode`, `--val-inner-cities`,
  `--buffer-km`, `--min-labelled-frac`, `--erode-px`, `--class-weights`,
  `--label-smoothing`, `--monitor`, `--tta`, `--warmup-epochs`; dual-level
  evaluation; `test_metrics.json` (the evaluator's return value was previously
  discarded, so runs left no machine-readable record of their own numbers).
- **Tests** — `tests/test_lcz_metrics.py`, `tests/test_city_split.py`,
  `tests/test_grid_tile_labels.py`.

### 8.4 The coverage caveat (read before tabling any patch-level number)

§4.1 asks for the headline to be restricted to "the 23,858 aligned test patches
so the comparison is exact". Under a city-level split that is **not achievable**,
and the reason is structural rather than fixable: tiles failing split purity or
the proximity buffer are dropped whole, and their patches go with them. A tile
is the unit of prediction, so a patch inside a discarded tile has no prediction
at all.

Measured on Nairobi in the verification run: **992 of 2,417 testing patches
scored (41 %)**. Nairobi lost 27 tiles to purity and 61 to the buffer out of
237; the survivors split 69 test / 80 culture-val.

So a segmentation kappa is computed over a strict subset of the patch
campaign's test set. Two consequences:

- The evaluator now logs the coverage fraction, stores it as
  `test_coverage`, and **warns below 95 %**.
- For an exact comparison against 0.6497 / 0.6871 / 0.7055, intersect on
  `patch_id` and re-score *both sides* on the intersection. `probs_patch.npz`
  carries `patch_ids`, `cities` and `datasets` precisely so this is possible.

Reporting a seg kappa against the published ladder without that intersection
compares different test sets and will flatter whichever side lost fewer hard
patches.

### 8.4b A silent failure the new `--monitor val_kappa` exposed

`run_training_loop` maximises the monitored metric from `-inf`. **NaN compares
greater than nothing**, so a monitor that goes NaN checkpoints nothing,
`best_ckpt_path` stays `None`, and the pipeline goes on to evaluate a model
that was never selected — producing a full set of plausible test numbers from
effectively untrained weights. Caught in verification: a 6-tile val split made
`MulticlassCohenKappa` return NaN (a single class present means expected
agreement is 1, so the chance correction divides by zero) and the run reported
`Best checkpoint: None` without further comment.

Fixed in `training/loop.py`: a NaN monitor is warned about per epoch, and a run
that never checkpointed says so explicitly, naming the consequence. Pinned by
`tests/test_training_loop_monitor.py`. This is latent for `val_miou` too — it
is not specific to the metric this work added, only exposed by it.

### 8.4c Comparing a seg number to the ladder

`src/compare_seg_to_patch.py` intersects a segmentation `probs_patch.npz` with
a patch-pipeline `probs.npz` on `patch_id` and re-scores **both** sides on the
intersection, per-city as well as pooled. Nothing has to be re-run; both
pipelines already emit the inputs.

Why it is not optional: re-scoring the published patch models on the 992
Nairobi patches from the verification run gives kappa **0.17–0.31**, against
their 0.6497 headline — because Nairobi is the hardest city in the set. Any
comparison that skips the intersection is measuring city mix, not model
quality.

The LOCO combiner is also fed now: a global-split run writes
`probs_culture_val.npz` (the culture cities' validation patches) alongside
`probs_patch.npz`, which is the pair `ensemble_stacking.py --city-holdout`
needs.

### 8.4d Input normalisation (added on request, 2026-08-21)

`--normalize {none,channel}` on `semantic_segmentation.py`, **defaulting to
`channel`** to match the patch pipeline. Every segmentation run before this
date was unnormalised; `--normalize none` reproduces that path exactly and logs
a warning saying so.

Statistics come from `datasets.channel_stats.compute_grid_channel_stats`, which
estimates them by iterating `GridSegDataset` itself. That is the whole design:
because the statistics are produced by the same class that produces training
batches, the per-family treatment is inherited rather than re-implemented —

- **coop / seamless**: dequantization is applied by the dataset, so the
  statistics describe dequantized values, which is what the model sees.
- **seamless**: its 13 stored bands become 72 channels during dequantization,
  so the arrays come out 72 long automatically.
- **fusion**: dequantization applies to source 0 only, exactly as in training,
  and the statistics span the full concatenated stack.
- **nodata**: the family predicate runs on the **raw** array *before*
  dequantization, because that is the only place the sentinel is visible —
  coop's all-64-channels `-128` becomes a plausible vector of L2 norm 8.06
  afterwards. A fused pixel is invalid if any source marks it invalid.

Ordering is the subtle part and it pulls both ways: detect nodata **before**
dequantize, measure statistics **after** it. `tests/test_grid_normalize.py`
pins both directions.

Statistics are written into the checkpoint as `normalize` / `channel_mean` /
`channel_std`, so `infer_roi.py --normalize auto` reproduces them — previously
segmentation checkpoints carried no normalisation metadata at all and
`infer_roi` refused them under `auto`. `run_city_inference` now receives the
stats too.

### 8.4e Tessera coverage, and making rows 4 and 5 comparable (2026-08-21)

Grid extraction is **complete** for both primary arms — AlphaEarthCoop 18,523
npy and GeoTessera_v1.1_global 17,530, all 51 cities. `aux_struct` remains the
only unextracted embedding (0 of 51).

The 993-tile difference is **not** an unfinished extraction. Checked against
`data/tessera_v1.1_global_2017_tiles.gpkg`: **zero** of the missing grid cells
intersect any Tessera tile, so the tiles are absent from the archive and
re-running extraction yields nothing. All 14 affected cities are coastal.

AlphaEarth coop covers every valid tile in all 51 cities, so it is the
reference; the percentage below is what Tessera cannot supply.

| City | tiles | missing | | City | tiles | missing |
|---|---|---|---|---|---|---|
| Qingdao | 541 | **37.3 %** | | Shanghai | 385 | 6.8 % |
| Istanbul | 1192 | **32.3 %** | | Dongying | 306 | 6.5 % |
| Lisbon | 225 | 20.9 % | | Melbourne | 1261 | 4.6 % |
| Amsterdam | 372 | 12.9 % | | Sydney | 522 | 3.8 % |
| Mumbai | 612 | 10.6 % | | Vancouver | 673 | 3.0 % |
| Cape Town | 565 | 10.3 % | | Guangzhou | 562 | 0.2 % |
| New York | 611 | 6.9 % | | London | 679 | 0.1 % |

The remaining **37 cities match exactly**.

**Why this matters for the ladder.** Rows 4 and 5 exist to isolate the
embedding. Unrestricted, Qingdao would contribute 541 tiles to the AlphaEarth
arm and 339 to the Tessera arm, so a gap between the rows would partly be a gap
in which cities each model saw.

`--require-embeddings OUTPUT_NAME [...]` keeps only cells present under every
named output-name as well as the training one. Membership only — channels are
untouched, which is what separates it from `--output-name` fusion. Applied
before fusion, statistics and class weights, so every downstream quantity is
computed on the same tile set. Verified: both arms end up on the *same cells*,
not merely the same count (`tests/test_common_tiles.py`).

Run both arms with it for any row 4 vs row 5 claim:

```bash
# AlphaEarth arm, held to Tessera's ground
--output-name AlphaEarthCoop --embedding-name alpha_earth_coop \
    --require-embeddings GeoTessera_v1.1_global

# Tessera arm, same ground (a no-op here, but keep it symmetric and explicit)
--output-name GeoTessera_v1.1_global --embedding-name tesserav1.1_global \
    --require-embeddings AlphaEarthCoop
```

Add `EmbeddedSeamless` to both lists if the row 7 ensemble is meant to span all
three.

### 8.5 Still outstanding

1. **The §7 step-4 gate is open** — no features baseline exists (decision 2).
2. **`aux_struct` grid extraction**, the one prerequisite for ladder row 6:
   ```bash
   python src/extract_grid_embeddings.py \
       --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities --cities all \
       --embedding-name aux_struct --output-name aux_struct --year 2017 \
       --embedding-dir ${DATA_DIR}/input/aux_struct/merged_aux
   ```
   Then train with `--output-name AlphaEarthCoop aux_struct` (fusion).
3. **The row 7 ensemble** — deferred with the GPU.
4. ~~Per-city metrics~~ — done, §9.2. ~~Training runs, launchers~~ — done for
   rows 4/5 at small and medium, §9.

### 8.6 Corrected commands

```bash
# Primary: AlphaEarth 2017, city-level global split
python src/semantic_segmentation.py \
    --cities-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4/cities --cities all \
    --output-name AlphaEarthCoop --year 2017 --label-source gpkg \
    --family unet --preset small \
    --split-mode global --val-inner-cities 6 --buffer-km 1.3 \
    --monitor val_kappa --dice-weight 0.0 --tta \
    --embedding-name alpha_earth_coop \
    --embedding-dir ${DATA_DIR}/input/Google/AlphaEarth/coop \
    --output-dir ${DATA_DIR}/output/lcz-segmentation/dl

# Second arm: Tessera v1.1 global (NOT tesserav1.1, which is deprecated)
python src/semantic_segmentation.py ... \
    --output-name GeoTessera_v1.1_global \
    --embedding-name tesserav1.1_global \
    --embedding-dir /tessera/v1.1

# Smoke test
python src/semantic_segmentation.py ... --preset nano --max-epochs 1 \
    --no-wandb --cities Nairobi
```

Note `--embedding-dir` for coop is the **root** (containing `aef_index.gpkg`),
not the year subdirectory. `--embedding-dir` for Tessera is the **mount root**
`/tessera/v1.1`, not `${DATA_DIR}/input/GeoTessera/2017` — the latter is a
smaller local mirror (3,116 tiles) and every actual row 4/5 run below used the
mount (8,108 tiles, 97.6% coverage). Confirmed by both `run_seg_ladder.sh` and
`run_seg_ladder_medium.sh`.

---

## 9. Results — ladder rows 4 and 5 (2026-08-31 / 2026-09-01)

First real numbers from the pipeline. **n=1 seed per cell** — nothing here has
the replication the patch campaign's numbers do, and campaign-scale seed
spread (2.1b-iii found ±0.02–0.03 kappa from seed alone on the patch task) is
plausibly larger than some of the gaps reported below. Read every comparison
as a lean, not a settled result.

All four runs: `--split-mode global`, all 51 cities, `--require-embeddings`
holding rows 4 and 5 to their intersection (14,411 common tiles — see §8.4e),
`lr 5e-4` / `warmup 3` (Task 2.1c's winner on the patch task, carried over as
an assumption — no seg-specific schedule sweep has been run), `--dice-weight
0.0`, `--tta`, `--normalize channel`. Config: `run_seg_ladder.sh` (small),
`run_seg_ladder_medium.sh` (medium).

### 9.1 Headline table

| | row 4 small (coop) | row 5 small (Tessera) | row 4 medium (coop) | row 5 medium (Tessera) |
|---|---|---|---|---|
| params | 1.95M | 1.96M | 7.78M | 7.80M |
| peak epoch / val_kappa | 3 / 0.680 | 19 / 0.673 | 9 / 0.679 | 13 / 0.663 |
| stopped at epoch | 13 | 29 | 19 | 23 |
| pixel kappa | 0.4533 | 0.5425 | 0.4083 | 0.5531 |
| **patch kappa** | **0.4294** | **0.5314** | **0.3820** | **0.5316** |
| patch OA | 0.4796 | 0.5751 | 0.4298 | 0.5766 |
| patch F1 (macro) | 0.3030 | 0.4196 | 0.3167 | 0.4160 |
| patch OAw | 0.8472 | 0.8776 | 0.8176 | 0.8672 |
| patch kappa_w | 0.6064 | 0.6784 | 0.5209 | 0.6605 |
| coverage | 40.6% (9,801/24,170) | 40.6% (9,804/24,170) | 40.6% | 40.6% |
| per-city kappa (mean ± std) | 0.400 ± 0.207 | 0.484 ± 0.201 | 0.357 ± 0.217 | 0.489 ± 0.194 |

The **kappa vs kappa_w gap** (§4.2's own reporting rule) is large everywhere —
0.18–0.21 pixel-level, 0.14–0.18 at patch level — meaning most of the residual
error is within-family (LCZ 1↔2, A↔B) rather than built↔natural confusion.
That is the kinder read of an otherwise weak set of numbers.

### 9.2 Per-city patch kappa

| City | r4 small | r4 medium | r5 small | r5 medium |
|---|---|---|---|---|
| Munich | 0.691 | **0.737** | 0.758 | 0.713 |
| San Jose | 0.627 | 0.600 | **0.726** | 0.686 |
| Jakarta | 0.556 | 0.290 | **0.807** | 0.774 |
| Moscow | 0.509 | 0.536 | 0.561 | **0.624** |
| Tehran | 0.466 | 0.306 | 0.457 | **0.508** |
| Sydney | 0.464 | 0.453 | 0.319 | **0.484** |
| Guangzhou | 0.255 | 0.152 | 0.307 | **0.329** |
| Santiago | 0.231 | 0.198 | **0.339** | 0.287 |
| Mumbai | 0.191 | **0.336** | 0.267 | 0.263 |
| Nairobi | 0.009 | **−0.033** | **0.300** | 0.224 |

Two things worth flagging rather than averaging away:

- **Nairobi is the pipeline's hard case, consistently** — the campaign's own
  patch-classification LOCO results put Nairobi at 0.47, already the worst or
  near-worst city there. Here it is catastrophic for AlphaEarth (0.009 small,
  **−0.033 medium — worse than chance**) and merely weak for Tessera
  (0.22–0.30). One city, one seed, but a −0.033 kappa on a model that trained
  successfully everywhere else is worth a look before it is dismissed as noise.
- **Tessera wins 8 of 10 cities at both presets.** The two exceptions (Munich,
  Mumbai) are close. This is the more replicable-looking part of the result —
  consistent across both the small and medium capacity, which a seed artefact
  would not obviously produce.

### 9.3 Capacity: medium does not help (small vs medium)

| | row 4 (coop) | row 5 (Tessera) |
|---|---|---|
| small → medium | 0.4294 → 0.3820 (**−0.047**) | 0.5314 → 0.5316 (+0.0002, noise) |

Medium **hurt** on coop and was a dead heat on Tessera. This is the same
pattern §3 of this plan predicted from the patch campaign (resnet101/152 did
not beat resnet34) without testing it directly for segmentation — now tested,
and it holds. No reason from this to run `base` or `large`.

### 9.4 The comparison that actually matters: seg vs. the patch ladder

§4.1 requires this comparison to run through the `patch_id` intersection, not
side by side in isolation (§8.4c) — a segmentation kappa computed on its own
40.6%-coverage subset means nothing next to a patch kappa computed on the full
23,858. `src/compare_seg_to_patch.py` re-scores all three cached patch models
(`resnet-small-{GeoTessera_v1.1_global,AlphaEarthCoop,EmbeddedSeamless}` from
`ensemble_coopv1/ensemble_3models_test/probs.npz`) on the same ~9,800-patch
intersection each seg run covers:

| | seg (this run) | patch: Tessera | patch: coop | patch: seamless |
|---|---|---|---|---|
| row 4 small | 0.4294 | 0.6674 | 0.5454 | 0.4981 |
| row 5 small | 0.5314 | 0.6675 | 0.5455 | 0.4986 |
| row 4 medium | 0.3820 | 0.6674 | 0.5454 | 0.4981 |
| row 5 medium | 0.5316 | 0.6675 | 0.5455 | 0.4986 |

**Every patch model beats every segmentation model on the identical patches.**
The single best seg result (row 5, either preset, kappa ≈ 0.531) edges out only
the *weakest* patch arm (seamless, 0.498) and loses clearly to both coop
(0.545) and Tessera (0.667) run through the patch pipeline. This is the honest
headline of this ladder pass: segmentation is not yet competitive with patch
classification on the cities where both were scored, on one seed at `small`
and `medium` U-Net capacity.

This does not settle whether segmentation *can* win — the seg runs are single-
seed, `small`/`medium` only, no aux channels, and the schedule is borrowed
rather than tuned for this task. It does mean nothing claiming segmentation
superiority should ship from this pass, and that row 6 (aux channels) and a
seg-specific schedule check are more urgent than they looked before this
comparison was run.

### 9.5 What's still not done

- ROI inference maps for Nairobi/Cairo/London, all four checkpoints — running
  (small preset complete; medium in progress in tmux, `eofm-infer-medium`).
- Row 6 (aux channels) — blocked on `aux_struct` extraction, §8.5 item 2.
- Row 7 (ensemble) — not started.
- No replication (n=1 everywhere in §9) and no seg-specific LR/warmup sweep —
  the schedule is inherited from the patch task's Task 2.1c, not verified here.
