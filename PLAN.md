# eo-fm — Correctness Fixes and Experiment Plan

## How to use this document

Work through phases **in order**. Each phase ends with a **GATE**: stop, write results
to `RESULTS.md`, and report back before starting the next phase. Do not skip ahead —
Phase 2 results are meaningless if Phase 1 is incomplete, and Phase 3 onwards assumes
the Phase 2 numbers are trustworthy.

**Working rules for the whole plan:**

- One logical change per commit. One branch per phase (`fix/p1-normalization`, etc.).
- Never change two things at once in a training run. Every experiment varies exactly
  one factor from the run it is being compared against.
- Every run gets a W&B tag naming its phase (`p2-revalidation`, `p3-ablation`, ...) and
  a `config` entry recording every new flag introduced by this plan, even when set to
  its default. Runs that cannot be reconstructed from their W&B config are worthless.
- Append every result to `RESULTS.md` as a markdown table row: phase, run name, W&B id,
  embedding, split, model, seed, OA, macro-F1, kappa, and the one factor that changed.
- Do not refactor beyond what a task asks for. This codebase currently produces numbers
  that go into a thesis; silent behaviour changes are the main risk.
- Default seeds for all multi-seed experiments: `0, 1, 2`.

---

## Context

Repo: `eo-fm`, LCZ classification from EO foundation-model embeddings.
Data root is `DATA_DIR` in `.env` (`/maps/acz25/phd-thesis-data`).

Three embedding families, all extracted to per-patch `.npy` of shape `(C, 32, 32)`:

| Key | C | Dequantization |
|---|---|---|
| `alpha_earth_coop` | 64 | `((v/127.5)**2) * sign(v)`, auto-applied |
| `tesserav1.1` | 128 | `int8 * per-pixel scale`, applied at extraction |
| `seamless` (ESD) | 13→72 | codebook factorization to [-1, 1], auto-applied |

Raw **Sentinel-1 and Sentinel-2** patches are also already extracted as per-patch GeoTIFFs,
sitting in the same `training/` / `validation/` / `testing/` subfolders as the labels and
the embedding `.npy` files. These are the baseline modality for Task 3.4 and are treated as
a fourth "embedding family" throughout this plan.

Two splits: `--global-split` is the So2Sat **cultural** split (42 train cities, 10
held-out cities for val+test) and is the one comparable to the literature. The per-city
**grid** split is the default and is not comparable to published numbers.

Current headline results on the cultural split (Tessera v1.1): ResNet34 OA 0.65 /
kappa 0.62; GAP+linear probe OA 0.61 / kappa 0.57. Grid split ResNet34: OA 0.85.
Published SOTA on the cultural split is ~0.73–0.74.

**Suspected problem:** the CNN training path applies no input normalization, and the
augmentation adds Gaussian noise at a fixed absolute sigma. The three families have
very different native scales, so the cross-family comparison is likely confounded.

---

## Phase 0 — Diagnostics (measure only, change nothing)

Do not touch anything under `src/training/`, `src/models/`, or `src/datasets/` in this
phase. Create new files only.

### Task 0.1 — Embedding scale audit

Create `src/diagnostics/embedding_stats.py`.

For each of the three embedding families **and for raw Sentinel-1 and Sentinel-2**, sample
5000 random patches from the cultural-split **training** set, apply the same dequantization
the training path applies (via `utils.runtime.resolve_dequantize`; a no-op for S1/S2), and
report:

- per-channel mean and std (all C channels)
- pooled per-channel std: median, min, max across channels
- distribution of the **per-pixel L2 norm** across the channel axis (mean, std, p1, p50, p99)
- fraction of exactly-zero values and of NaN values before `nan_to_num`
- per-channel skewness and the p0.1 / p99.9 percentiles (needed for S1/S2, where heavy tails
  make naive z-scoring a poor choice — see Task 3.4)

Write `diagnostics/embedding_stats.json` and a markdown summary table in `RESULTS.md`.

Then compute, for each family, the **effective noise ratio**:

```
effective_noise_ratio = 0.05 / median_per_channel_std
```

This is what `augment_images` is currently doing to each family. Report it explicitly.

**Expected finding, to be confirmed or refuted:** AlphaEarth's per-channel std is around
0.125 (unit-norm vectors over 64 dims), ESD's is around 0.5, Tessera's is something else
again — so the ratio differs several-fold across families and the cross-family comparison
is confounded.

### Task 0.2 — AlphaEarth dequantization verification

Create `src/diagnostics/verify_alphaearth_dequant.py`.

AlphaEarth embeddings are unit-norm 64-d vectors, so **the per-pixel L2 norm after a
correct dequantization must be ≈ 1.0**. This is decisive on its own and needs no
external reference data.

Test three candidate decodings of the coop int8 values on the same sample of patches:

1. `((v/127.5)**2) * sign(v)` — what the codebase currently does
2. `v/127.5` — plain linear
3. `sign(v) * (|v|/127.5)` followed by L2 renormalization per pixel

For each, report the per-pixel L2 norm distribution. Whichever gives norms tightly
concentrated near 1.0 is the correct one.

If GEE float32 tiles for any city are still on disk, add a second check: per-channel
Pearson r and RMSE between the GEE floats and each candidate decoding over the
overlapping extent. Do not spend more than an hour trying to obtain GEE tiles — the
norm test is sufficient to make the call.

### GATE 0

Report:
- the scale table and effective noise ratios
- which AlphaEarth decoding is correct
- whether the current decoding is wrong, and if so your estimate of how badly it
  distorts the embeddings (e.g. correlation between correct and current decoding)

Do not proceed until I have seen these numbers.

---

## Phase 1 — Correctness fixes  — **DONE** (branch `fix/p1-correctness`)

> **Amendments following GATE 0 (A1–A9).** Phase 0 confirmed the noise-ratio
> confound (14× across families) but also refuted one assumption and turned up
> three things this plan did not anticipate, so Phase 1 was revised before
> execution:
>
> - **A1** New Task 1.0, done first: fix the `open_tile` ordering bug.
> - **A2** Task 1.1 is a **no-op** — the AlphaEarth decoding is verified correct.
> - **A3** Task 1.5 (nodata) runs **before** Task 1.2 (normalization).
> - **A4** Nodata predicates are **per family**, not global.
> - **A5** `compute_channel_stats` excludes invalid pixels; n_sample 5000 → 20000;
>   corrected stats reported as a diff against the Phase 0 table.
> - **A6** `--noise-sigma` is added but treated as untuned; the sweep is Task 2.3.
> - **A7** Task 1.7 demoted to optional, with corrected semantics.
> - **A8** Task 3.4a's layout is already known — spec updated below.
> - **A9** New pre-close check on the Sentinel-1 extremes — see Task 3.4b.
>
> Measured outcomes are in `RESULTS.md`; each task below is one commit.

### Task 1.0 (A1) — Fix the `open_tile` format dispatch — DONE

`datasets/tiles.py` tested `path.is_dir()` before `path.suffix == ".zarr"`, but a
zarr store **is** a directory, so every `.zarr` tile was misrouted to the Tessera
NPY reader and the `.zarr` branch was unreachable — breaking the `alpha_earth`
(GEE) and `tessera` families end to end. Regression test in `tests/` (registered
in `pyproject` `testpaths`) covers both branches. No published run is affected:
all current work uses `alpha_earth_coop` / `tesserav1.1_global`.

### Task 1.1 — SKIPPED (A2)

Phase 0 verified `((v/127.5)**2)*sign(v)` is correct, by two independent routes.
No fix, no legacy alias, no re-extraction. `verify_alphaearth_dequant.py` is kept
as a methods-appendix artefact.

### Task 1.5 (A4) — Handle nodata properly — DONE, runs before 1.2

None of the families stores nodata as NaN, so `np.nan_to_num` never caught any of
it. Per-family predicates now live in `EMBEDDING_REGISTRY`
(`get_nodata_predicate`), tested on the array **as stored**:

- `alpha_earth_coop`: all 64 int8 channels == −128 (0.41% of pixels). Decoded, each
  such pixel had L2 norm 8.06 instead of 1.0.
- `tesserav1.1` / `_global` / `v2`: all 128 channels == 0 (0.05–0.14%).
- `seamless`, `sentinel1`, `sentinel2`: none measurable.

The test must be **all-channel**: Tessera has ~0.9% per-channel quantization zeros
touching 61% of pixels, and a per-channel rule would discard most of the dataset.

`--nodata-mode {zero,mask}`, default `mask`, emits a `(1,H,W)` `valid` channel;
invalid pixels are filled **after** dequantization so the sentinel cannot bleed
into neighbours through the resize, and the mask rides through the same resize
requiring full weight. `LinearProbeModel`/`MLPModel` pool with the mask;
conv families only log `train/val_invalid_frac`. `zero` reproduces the old path
exactly, so it stays usable as the Task 2.3 ablation.

### Task 1.2 (A5) — Input normalization — DONE

`datasets/channel_stats.py`: streaming per-channel mean/std over the **train split
only**, over **valid pixels only**, on the **post-resize** tensor (the array the
model consumes; native-grid stds run ~15% higher and are not interchangeable).
Cached under `{DATA_DIR}/cache/channel_stats`, `--recompute-stats` to invalidate.
`--normalize {none,channel}` defaults to `channel`; `--stats-sample` defaults to
20000. Invalid pixels are filled with the channel mean so they normalise to 0.

Statistics travel **inside the checkpoint** (as tensors, so `weights_only=True`
still loads it) and `infer_roi.py` reads them back; a checkpoint without them is
refused unless `--normalize none` is passed explicitly.

### Task 1.3 (A6) — Scale-relative augmentation noise — DONE

`--noise-sigma` / `--noise-prob` on both pipelines, documented as units of the
**normalized** per-channel std. 0.05 is the historical value and is **untuned** —
Task 2.3 sweeps σ ∈ {0, 0.025, 0.05, 0.1, 0.2} on Tessera and AlphaEarth, 3 seeds.
`--normalize none` with non-zero sigma warns, since it reintroduces the confound.

### Task 1.4 — `PatchItem` dataclass — DONE

Replaces the positional tuples, unpacked in six places across five files. The
`city` field needed a spatial join: `patches_reference_rxr.gpkg` has **no** city
column, so `assign_cities()` joins patch centroids against the `JRC_NAME_MAIN`
polygons in `so2sat_guppd_bounds.gpkg` — 98.6% of patches resolve to one of 51
cities. Per-city normalization itself is Phase 4.

### Task 1.6 — Logit adjustment — ALREADY IMPLEMENTED

`logit_adjustment_tau`, the `log_prior` buffer, `_adjust_logits` and the
`--logit-adjustment` flag already existed. Only the mutual exclusion with
`--class-weights` was missing; it now errors.

### Task 1.8 — Data pipeline throughput — DONE

Augmentation vectorized over the batch and moved from the collate into
`train_step` so it runs on GPU (**1,010 → 3,051 patches/s** end to end; the
augmentation itself 631 → 30,715 patches/s). It now also transforms the validity
mask — the collate augmented only `image`, which after Task 1.5 would have left
the mask misaligned. `src/pack_patches.py` + `--packed-dir` add memory-mapped
shards, format-agnostic at pack time so Task 3.4 reuses them; worth only +7.6% on
a warm page cache, where dequantize and resize dominate.

Benchmarks must be run **alternating in one process**
(`src/diagnostics/bench_dataloader.py`): page-cache drift between separate
invocations is larger than the effect being measured.

### Task 1.7 (A7) — Mixup on the unit sphere — DONE (optional flag)

Per-channel z-scoring is affine and mixup weights sum to 1, so mixup in normalized
space is identical to mixup in raw space — but z-scored AlphaEarth vectors no
longer lie on the unit sphere, so renormalizing there would be meaningless.
`--mixup-renorm` therefore maps back through the stored statistics, projects onto
the sphere, and maps forward; it requires `--normalize channel` and warns for
non-AlphaEarth embeddings.

### GATE 1 — REACHED

Reported in `RESULTS.md`: throughput before/after, invalid-pixel fractions per
family, corrected statistics as a diff against Phase 0, and confirmation that
`infer_roi.py` with `--normalize none` reproduces a pre-change map bit for bit
(md5 `7c2222b0`). Behaviour changes to carry into Phase 2: the defaults are now
`--normalize channel --nodata-mode mask`, and the augmentation RNG stream changed.


## Phase 2 — Re-validation

Branch `exp/p2-revalidation`. **No code changes to model or training logic in this phase.**

The question: do the Phase 1 fixes change the ranking Tessera > AlphaEarth > ESD?

### Task 2.1 — Reproduce the pre-fix baseline

With `--normalize none --noise-sigma 0.05` and the legacy AlphaEarth decoding, rerun the
three best-known cultural-split configs and confirm they land within noise of 0.65 / 0.56 /
0.55 OA. If they do not, stop — something in Phase 1 changed behaviour unintentionally.

### Task 2.2 — The fixed comparison

Cultural split (`--global-split`), ResNet34 (`--family resnet --preset small`), 3 seeds
each, for all three families, with the Phase 1 defaults on. Same LR schedule, epochs, and
early-stopping patience as the current best Tessera config
(`opt3-lr5e-4-warmup3`: `--lr 5e-4 --warmup-epochs 3`).

Report mean ± std over seeds for OA, macro-F1, kappa. Include a confusion matrix for the
best run of each family (`training/evaluate.py` already plots these).

### Task 2.3 — Isolate the contribution of each fix

Ablate on Tessera only, 3 seeds, cultural split, one factor at a time from the Task 2.2
config: normalization off; noise sigma 0; logit adjustment at tau ∈ {0.0, 0.5, 1.0};
nodata-mode zero. This tells us which fix mattered and by how much — needed for the paper's
methods section, not just for our own confidence.

### Task 2.4 — Model selection metric

Rerun the best Tessera config with `--monitor val_f1` and `--monitor val_kappa`, 3 seeds
each. Add `val_acc` as a third `--monitor` option if it is not already there. Report the
full metric triplet for each, so we can see the cost of selecting on the wrong one.

### GATE 2

Report the fixed three-family table with error bars, the per-fix ablation, and the monitor
comparison. **This is the decision point for the whole chapter** — if the ranking or the
absolute numbers moved substantially, several conclusions in the existing draft need
rewriting before any new modelling happens.

---

## Phase 3 — Ablations that define the contribution

Branch `exp/p3-ablations`.

### Task 3.1 — Permutation test

Add `--permute-pixels` to `patch_classification.py`. When set, randomly permute pixel
positions within each patch (same permutation for all channels of a sample, fresh
permutation per sample per epoch), applied in the augmentation step for train and
deterministically for val/test.

Run best-Tessera cultural, 3 seeds, with and without. If accuracy is unchanged, spatial
arrangement within a 320 m patch carries no usable signal and the CNN is the wrong
inductive bias. Either outcome is a publishable figure.

### Task 3.2 — The 2×2 pooling × split table

Fill in all four cells, 3 seeds each, Tessera only:

| | cultural | grid |
|---|---|---|
| GAP + linear probe | ~0.61 (rerun) | **missing** |
| ResNet34 | ~0.65 (rerun) | ~0.85 (rerun) |

If the CNN's margin over GAP is much larger on the grid split than on the cultural split,
the spatial features it learns are city-specific and do not transfer. That is the cleanest
possible statement of the problem this chapter addresses.

### Task 3.3 — Set-encoder family

Register a new `netvlad` family in `src/models/` following the pattern in
`src/models/linear_probe.py` and `src/models/mlp.py` (registry-based, `build()` signature
matching `ModelFamily`).

Architecture: input `(B, C, H, W)` → flatten to `(B, HW, C)` → soft assignment to K
learned prototypes → per-cluster residual sums → intra-normalize, flatten, L2-normalize →
linear classifier. Presets: `nano` K=16, `small` K=32, `base` K=64, `medium` K=128,
`large` K=256.

Initialize prototypes by k-means on a 200k-pixel sample from the training split (cache the
result). Respect the validity mask from Task 1.5 in the assignment.

Motivation to record in the docstring: LCZ classes differ in the *mixture* of surface
types within the patch, which mean pooling destroys and a permutation-invariant set
encoder preserves.

Run all presets, cultural split, Tessera, 3 seeds. Compare against the Task 3.2 numbers.

### Task 3.4 — Matched Sentinel-1/2 baseline

This is the most important missing experiment in the plan. Without it there is no controlled
claim that foundation-model embeddings help: the comparison against Zhong et al. 2024 and
Lin et al. 2024 is not controlled, since those papers add prior-knowledge coupling and
semi-supervised learning respectively.

The patches are **already extracted** as per-patch GeoTIFFs in the same
`training/` / `validation/` / `testing/` layout as the embedding `.npy` files, so no
extraction work is needed. The job is to make the existing data layer read them.

#### 3.4a — Generalize the loader to GeoTIFF

**(A8) The layout is already known** — Phase 0 inspected it, so no reconnaissance is
needed:

```
{split}/sentinel1/sen1_patch_{id}.tif    8 bands, float64, 32x32 @ 10 m, local UTM
{split}/sentinel2/sen2_patch_{id}.tif   10 bands, float64, 32x32 @ 10 m, local UTM
```

S1 and S2 are **separate files**, with **no `{output_name}/{year}` nesting** (unlike the
embedding `.npy` files) and a `sen{1,2}_patch_` prefix rather than `patch_`. Counts match
the embeddings exactly (352,366 / 24,119 / 24,188) and `{id}` is the same `patch_id`, so
pairing by id is safe. **`nodata` is None and there are no NaNs** in any of the three
splits — so the plan below to "read the nodata value from the raster profile" has nothing
to read, and these families are registered with no nodata predicate.

Then make two changes in `src/datasets/so2sat.py`:

- `build_patch_index` currently hardcodes `p.glob("patch_*.npy")` and
  `p.stem[len("patch_"):]`. It needs three things generalized, not one: the **extension**,
  the **filename prefix** (`sen1_patch_` vs `patch_`), and an optional **nesting level**
  (the raw patches have no `{output_name}/{year}` directories), so the same `patch_id`
  keys come out regardless of format.
- `PatchDataset.__getitem__` currently calls `np.load` unconditionally. Dispatch on suffix:
  `.npy` → `np.load`; `.tif`/`.tiff` → `rasterio.open(...).read()` returning `(C, H, W)`
  float32 — cast on read, since the files are float64 and that doubles I/O for nothing at
  32×32. (`src/pack_patches.py` already dispatches this way; reuse its `read_patch`.)
  There is no nodata value in the profile to feed the Task 1.5 mask, so these families
  register no predicate and are treated as fully valid.

Register `sentinel1`, `sentinel2`, and `sentinel12` in the embedding registry with no
dequantization function, so they flow through `resolve_dequantize` as a no-op. `sentinel12`
loads both and concatenates on the channel axis (verify both resolve to the same
`patch_id` and the same 32×32 grid before concatenating; assert on shape mismatch).

Everything downstream — cultural/grid splits, `PatchItem`, normalization, augmentation,
logit adjustment, the model registry, `infer_roi.py` — then works unchanged. **That shared
code path is the point:** it is what makes this a matched baseline rather than a
reimplementation with different defaults.

#### 3.4b — Give the baseline a fair preprocessing

This deserves real care. If the raw modality gets naive preprocessing while the embeddings
got a tuned pipeline, the comparison is rigged and a reviewer will say so. Tune the
baseline at least as hard as the embeddings were tuned.

Sentinel-1 in So2Sat is heavy-tailed, and some of its 8 bands are real/imaginary components
that take negative values, so a blanket log transform is not applicable. Add
`--clip-percentile P` (default `0.0` = off) that clips each channel to its
[P, 100−P] training-set percentiles before z-scoring, and ablate P ∈ {0, 0.1, 1.0} on S1.

**(A9) The extremes are a genuine heavy tail, not corruption** — checked before Phase 1
closed, over 5000 patches. No infinities, no NaNs; values beyond 10× a band's p99.9 are
0.0000–0.0103% of that band, spread over 199/5000 patches (4.0%), and the top-10 patch
maxima decay smoothly (band 6: 7012, 3130, 2349, 2259, 2180, …) rather than showing an
isolated bad value. So **keep percentile clipping; do not switch to dropping bad pixels.**

One refinement this surfaced: bands 5–6 are **strictly positive** (0% negative, min ~1e−4),
i.e. intensity-like, while 1–4 and 7–8 are ~49% negative. A per-band log transform IS
applicable to the two positive bands even though it is not to the signed ones — worth a
fourth preprocessing variant. Confirm the band identities against the So2Sat documentation
as part of variant 3.

Run three preprocessing variants for the raw modality and take the best forward:

1. Channel z-scoring only (identical treatment to the embeddings).
2. Percentile clipping, then channel z-scoring.
3. The scaling used in the original So2Sat LCZ42 release — check the dataset documentation
   for the published per-band scaling and replicate it.

Record all three in `RESULTS.md`. Reporting the variant sweep, not just the winner, is what
makes the fairness of the comparison auditable.

#### 3.4c — The comparison runs

Cultural split, ResNet34 (`--family resnet --preset small`), 3 seeds, identical LR schedule,
epochs, early-stopping patience, monitor metric, and augmentation as the Phase 2 Tessera
config. Modalities: `sentinel1`, `sentinel2`, `sentinel12`.

Report against the Task 2.2 embedding results in a single table. The headline comparison is
`sentinel12` + ResNet34 versus `tesserav1.1` + ResNet34, everything else held constant.

Also run one fusion configuration — `tesserav1.1` concatenated with `sentinel12` — as an
upper-bound reference. If fusion substantially beats either alone, the embeddings are
discarding information the raw bands retain, which is worth knowing and worth reporting
even though it is not the chapter's main claim.

#### 3.4d — Note on throughput

400k small GeoTIFFs will read considerably slower than the `.npy` files, since each carries
its own header and compression. Extend the Task 1.8 packing script to cover the raster
formats before launching the seed sweeps, and re-benchmark patches/sec.

### Task 3.5 — Label-efficiency curve

Subsample the cultural-split training set to {1%, 2%, 5%, 10%, 25%, 50%, 100%}, stratified
by class, 3 seeds per point. Run the best embedding config and the winning `sentinel12`
config from Task 3.4 at every point, using the identical subsample indices for both so the
two curves are paired. Plot OA and macro-F1 against training-set size.

This is the claim foundation models actually make, and it is likely to be the strongest
figure in the chapter regardless of where the 100% numbers land.

### GATE 3

Report all four experiments with error bars and the label-efficiency plot.

---

## Phase 4 — Closing the cross-city gap

Branch `exp/p4-domain`. Start only after Gate 3.

### Task 4.1 — Per-city standardization

Uses the `city` field from Task 1.4. Add `--normalize percity` as a third mode: subtract
the per-city channel mean and divide by the per-city channel std, computed over all
patches of that city regardless of split (label-free, so legitimate at test time; state
this explicitly in the docstring).

Also add `--normalize percity_mean_only`, subtracting the city mean but keeping the global
std, to separate the offset effect from the scale effect.

Cultural split, Tessera, 3 seeds each against the Task 2.2 baseline.

### Task 4.2 — AdaBN

Add `--adabn` to `infer_roi.py` and to the test-time evaluation path: before predicting on
a held-out city, run a forward pass in `train()` mode over that city's unlabelled patches
to refresh BatchNorm running statistics, then predict in `eval()` mode. No gradients, no
labels.

Evaluate per held-out city, and report the gain city by city rather than pooled — the
variance across cities is itself a result.

### Task 4.3 — Domain-adversarial training

Add a `--dann-lambda` option: city-ID classifier head on the pooled features with a
gradient reversal layer, lambda ramped up over training on the standard schedule. Also
implement `--coral-weight` as the simpler alternative (align feature covariances across
city minibatches). Run both, cultural split, 3 seeds.

### Task 4.4 — Factorized head

Add a `--factorized-head` option that predicts three factors — surface type (built /
natural), density, and height — and composes the 17-class logits from them, instead of a
flat 17-way softmax. Define the class-to-factor mapping in `src/utils/constants.py`
alongside `lcz_dict`, following Stewart & Oke (2012).

Rationale: LCZ 1/2/3 differ from 4/5/6 only in density and within each triple only in
height, so a flat softmax discards the label structure. The factors should also transfer
across cities better than class appearance does.

Report both flat 17-class metrics and per-factor accuracy.

### Task 4.5 — Stratified reporting

Extend `training/evaluate.py` to break test metrics down by held-out city and by Köppen
zone (`koppen_dict` is already in `src/utils/constants.py`; the per-patch Köppen class
needs joining from the reference GPKG or sampling the Köppen raster).

Every headline result from Phase 2 onwards should be re-reported this way. "Which cities
and climates does this fail in" is a more useful result than a single pooled number, and
it directly supports the Global South motivation.

### GATE 4

Report the domain-adaptation table, per-city and per-Köppen breakdowns, and a
recommendation on which combination to take forward.

---

## Out of scope for now

Do not start these without checking in first: WUDAPT label ingestion, the noise-transition
matrix for Demuzere pseudo-labels, semantic segmentation with polygon supervision,
GeoClimate, Overture labels. They are the next chapter of work and they depend on the
Phase 2 numbers being trustworthy.