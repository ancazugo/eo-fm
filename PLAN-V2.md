# eo-fm — PLAN v2 (supersedes PLAN.md from Phase 2 onward)

Phases 0 and 1 are complete. `PLAN.md` stays in the repo as the record of what was
done; this document is the live one from here.

## Status carried forward

**Phase 0 findings that stand:** the effective-noise ratio spans 14× across families
(Tessera 0.044, AlphaEarth 0.477, ESD 0.091, Sentinel-2 0.613); the cross-family
comparison was confounded in the direction that flattered Tessera. The AlphaEarth coop
decoding is correct and Task 1.1 was a no-op.

**Phase 1 corrections to my amendments, accepted:** the AlphaEarth sentinel is 28% of
variance / 18% of std on the native grid and <1% post-resize, not 37%/26%. Tessera's
all-channel-zero nodata is genuine but numerically inert (sentinel value sits at the
channel mean). S1 bands 5–8 are a genuine heavy tail, not corruption, so percentile
clipping stays; bands 5–6 are strictly positive and additionally admit a per-band log.

**Working rules from PLAN.md remain in force:** one factor per run, three seeds
(`0, 1, 2`), W&B tags per phase, every result appended to `RESULTS.md`, no refactoring
beyond scope, stop at each GATE.

---

## Phase 1.5 — Provenance, versioning, and one open diagnostic

Branch `fix/p15-provenance`. This phase blocks Phase 2: every cross-family number
depends on knowing which product produced it.

### Task 1.5.0 — Resolve the resize question (blocking, do first)

Still unanswered from GATE 1. The sentinel's variance contribution falling from 28% to
<1% implies bilinear resampling cut its amplitude by roughly 5.5×, which needs spreading
over ~30 output pixels. A 33×33 → 32×32 correction spreads one pixel over ~4 and would
cut the variance contribution to about a quarter, not a thirtieth. Something else is
going on, and AlphaEarth at 10 m over a 320 m patch should be 32×32 natively with no
resize at all.

Report, per family, from a 2000-patch sample:

- the distribution of native crop shapes before `F.interpolate`
- the fraction of patches where the resize is an exact no-op
- where it is not, the effective resampling factor per dimension
- the same 28%-vs-<1% variance decomposition recomputed with the native shapes attached,
  so the two numbers can be reconciled

If meaningful resampling is happening to 10 m embeddings, that is an information loss
sitting underneath every result in the chapter and belongs in the methods section
regardless of what Phase 2 shows. It is also a candidate explanation for part of the gap
to 0.73 that no amount of Phase 4 domain adaptation would fix.

Two follow-ons if the resize is real:

- **Mask ordering.** Dilution means contamination spreads into neighbours rather than
  staying in one pixel, so a mask computed *after* the resize marks the contaminated ring
  valid. Confirm whether the Phase 1 implementation computes the mask on the native grid
  and propagates conservatively (any output pixel receiving nonzero weight from an invalid
  input pixel is invalid), or inpaints on the native grid before resizing. If it does
  neither, fix it — this is the one place where the "numerically inert" verdict on
  Tessera's nodata could stop holding.
- **Norm shrinkage.** Bilinear interpolation of unit-norm vectors does not preserve the
  norm; AlphaEarth's L2 p1 of 0.986 is consistent with this. Small, but note it in the
  methods.

### Task 1.5.1 — Are `tesserav1.1` and `tesserav1.1_global` the same product?

`datasets/tiles.py` shows `_open_tile_tessera11` and `_open_tile_tessera11_global` both
calling `load_and_dequantize_tessera_representation` on an int8 + scales pair. Same model
version, same decode. The 0.80 vs 1.14 median-std gap therefore should not exist. The two
Phase 0 measurements also came from different sample pools (51-city subset vs global),
so the comparison was never controlled.

Decisive test: intersect the `patch_id` sets present in **both** extractions, sample
N = 2000 of those *same* ids, and compare the two arrays pixel by pixel.

Report per-channel Pearson r, RMSE, the per-channel std ratio, and whether the ratio is
approximately constant across channels and pixels. Three outcomes, three actions:

| Outcome | Interpretation | Action |
|---|---|---|
| Values match, aggregate std still differs | Sampling artifact from different coverage | Record it, treat as one product, move on |
| Values differ by a near-constant factor | One path has a scale bug | Find and fix before Phase 2 |
| Values genuinely uncorrelated | Different inference passes despite the shared version label | Treat as distinct products, keep both registry keys, pick one canonical |

Do not proceed to Phase 2 until this has an answer. If it is a scale bug, every existing
Tessera number is affected.

### Task 1.5.2 — Provenance schema in the registry

`EMBEDDING_REGISTRY` in `src/datasets/registry.py` currently has flat keys with no field
distinguishing model version from tile source. All Tessera entries declare
`in_channels: 128`, so a model built for one will silently accept another's data, and
nothing prevents a normalizer fitted on one being applied to another.

Add to every entry, without renaming any key (renaming would break existing extraction
paths and W&B history):

- `product` — `tessera` | `alphaearth` | `esd` | `sentinel`
- `version` — `v1` | `v1.1` | `v2` | `coop` | `null`
- `source` — `percity_geotessera` | `global_0.1deg` | `source_coop` | `gee_zarr` | `local_tif`
- `status` — `canonical` | `supported` | `deprecated` | `untested`

Then add:

```python
def is_comparable(a: str, b: str) -> bool:
    """True only if two registry entries are the same product, version and source."""
```

Mark `tessera` (v1 GEE zarr) and `alpha_earth` (GEE zarr) as `untested` — the `open_tile`
ordering bug fixed in Task 1.0 means neither has been exercised end-to-end, so they should
not look available until someone runs them.

### Task 1.5.3 — Enforce it

Three guards, each with a test:

1. **Normalizer cache key.** The `compute_channel_stats` cache path must include the full
   provenance triple (`product`, `version`, `source`) plus year and split source — not just
   `output_name`. Two extractions of the same product from different tile sources must
   never share a stats file.
2. **Checkpoint binding.** Write `embedding_name`, `product`, `version`, `source`, and
   `year` into the checkpoint alongside the channel stats from Task 1.2. Loading a
   checkpoint whose provenance differs from the requested embedding must **raise**, naming
   both sides. Not a warning — this is the failure mode that silently produces a plausible
   wrong number.
3. **W&B config.** Log the same five fields on every run, so the `RESULTS.md` table can be
   reconstructed from W&B alone and no row is ambiguous about which Tessera it used.

### GATE 1.5

Report: the resize reconciliation, the `tesserav1.1` vs `tesserav1.1_global` verdict with
its three-column table, and confirmation that the three guards have tests. State which
Tessera entry is being nominated as `canonical` and why.

---

## Phase 2 — Re-validation (revised)

Branch `exp/p2-revalidation`. **No changes to model or training logic in this phase.**

The Tessera version question makes this a two-part phase: first fix a canonical family and
reproduce against it, then run the cross-family comparison, then — separately and clearly
labelled — the within-Tessera version comparison.

### Task 2.0 — Establish the canonical Tessera

Confirm which entry produced the existing 0.65 / 0.62 headline. Task 2.1's reproduction is
meaningless against the wrong one. Check the W&B config of that run rather than inferring
from the results file.

Nominate that entry `canonical` in the registry. Every cross-family claim in the chapter
uses it and only it. Note the choice explicitly in `RESULTS.md` with the W&B run id it was
derived from.

### Task 2.1 — Reproduce the pre-fix baseline

`--normalize none --nodata-mode zero`, canonical Tessera, cultural split, 3 seeds.

The Phase 1 RNG change means seed-level, not bit-level, agreement is the standard. Make the
test explicit: report mean ± std across seeds and check 0.65 falls inside. That spread is
itself a deliverable — every later "improvement" in the chapter has to clear it, and if the
original run's seed variance was never recorded, this is where it gets established.

Add, if not already present, a test that the pre- and post-Phase-1 augmentation produce
statistically indistinguishable output distributions (KS test on a large sample). That
separates "the RNG stream changed" from "the augmentation semantics changed" and turns an
open doubt into a footnote.

### Task 2.2 — The fixed cross-family comparison

Cultural split, ResNet34 (`--family resnet --preset small`), Phase 1 defaults
(`--normalize channel --nodata-mode mask`), 3 seeds, same schedule as the current best
config (`--lr 5e-4 --warmup-epochs 3`).

Families: `alpha_earth_coop`, canonical Tessera, `seamless`. Report mean ± std for OA,
macro-F1, kappa, plus a confusion matrix for the best run of each.

### Task 2.3 — Factorial ablation of the Phase 1 fixes

Do **not** run these as single-factor rows. Once σ is expressed relative to the normalized
channel std, disabling normalization silently reverts σ to absolute units, so
"normalization off" and "noise miscalibration" are entangled and the result would not answer
which fix mattered.

| | σ = 0 | σ = 0.05 |
|---|---|---|
| `--normalize none` | **A** | **C** (old behaviour) |
| `--normalize channel` | **B** | **D** (new default) |

B − A isolates normalization. C − A is the damage the old absolute-σ noise was doing.
D − B is what calibrated noise buys. Run on canonical Tessera and `alpha_earth_coop` — they
sit at opposite ends of the 14× spread. 4 configs × 2 families × 3 seeds = 24 runs.

Expect **B − A to be near zero**: ResNet has BatchNorm immediately after `conv1`, so the
network already largely self-normalizes. That would not mean the normalization work was
wasted — its value is making σ mean the same thing across families, which is what C − A and
D − B measure. A flat B − A with a large D − B is the clean version of the story, and it
should be written up that way rather than as a disappointment.

Separately, on the `--normalize channel` arm only, sweep σ ∈ {0, 0.025, 0.05, 0.1, 0.2},
3 seeds, both families. 0.05 was inherited, never tuned.

Also run `--nodata-mode zero` versus `mask` on `alpha_earth_coop` only — the one family
where the sentinel is concentrated and severe. Phase 1 predicts near-neutral post-resize;
confirm it.

### Task 2.4 — Model selection metric

Best canonical-Tessera config with `--monitor val_f1`, `val_kappa`, and `val_acc`, 3 seeds
each. Report the full metric triplet for each so the cost of selecting on the wrong one is
visible.

### Task 2.5 — Tessera version comparison (separate table, separate claim)

This is **not** part of the cross-family comparison and must not be presented as if it were.

Run `tesserav1.1`, `tesserav1.1_global`, and `tesserav2` under identical settings, 3 seeds.
Restrict to the **intersection of `patch_id`s present in all three extractions** so coverage
is not a confound — if the intersection is materially smaller than any individual set,
report both the intersection size and each family's own coverage.

If Task 1.5.1 concluded that two of these are the same product, say so here and run only the
distinct ones. Frame the result as "does the newer Tessera generation help on LCZ," which is
a cheap and worthwhile result, and state plainly that it compares model generations under
fixed extraction and split, not architectures.

### GATE 2

Report the reproduction with seed variance, the cross-family table with error bars, the 2×2
factorial with its three contrasts, the σ sweep, the monitor comparison, and the version
sub-table. **This is the decision point for the chapter** — if the ranking or absolute
numbers moved, existing draft conclusions need rewriting before any new modelling.

**Pre-register before launching.** Write predictions into `RESULTS.md` and commit them
first. Mine, for the record: canonical Tessera moves less than 1 point in either direction;
AlphaEarth improves most and entirely via the noise-ratio change; ESD improves modestly;
nodata masking is near-neutral for every family; B − A ≈ 0 and D − B carries the effect. I
put the AlphaEarth–Tessera gap closing by 1–6 of its 9 points, with the ranking flip
genuinely uncertain rather than unlikely. If AlphaEarth does not improve at all, the noise
was never the binding constraint and the gap needs a different explanation — most likely
that 64-d annual composites carry less LCZ-relevant structure than 128-d time-series
representations, which is a real result and a cleaner one to write up than a confound.

---

## Phase 3 — Ablations that define the contribution

Branch `exp/p3-ablations`. All runs use the canonical Tessera unless stated.

### Task 3.1 — Permutation test

Add `--permute-pixels`: randomly permute pixel positions within each patch (same permutation
across channels of a sample, fresh per sample per epoch), applied in the augmentation step
for train and deterministically for val/test. Canonical Tessera, cultural, 3 seeds, with and
without.

If accuracy is unchanged, spatial arrangement within a 320 m patch carries no usable signal
and the CNN is the wrong inductive bias. Either outcome is a publishable figure.

### Task 3.2 — The 2×2 pooling × split table

Fill all four cells, 3 seeds each:

| | cultural | grid |
|---|---|---|
| GAP + linear probe | ~0.61 (rerun) | **missing** |
| ResNet34 | ~0.65 (rerun) | ~0.85 (rerun) |

If the CNN's margin over GAP is much larger on the grid split than on the cultural split,
its spatial features are city-specific and do not transfer. That is the cleanest statement
of the problem this chapter addresses.

### Task 3.3 — Set-encoder family

Register a `netvlad` family in `src/models/`, following the registry pattern in
`linear_probe.py` and `mlp.py`.

Input `(B, C, H, W)` → flatten to `(B, HW, C)` → soft assignment to K learned prototypes →
per-cluster residual sums → intra-normalize, flatten, L2-normalize → linear classifier.
Presets: `nano` K=16, `small` K=32, `base` K=64, `medium` K=128, `large` K=256. Initialize
prototypes by k-means on a 200k-pixel training sample (cache it). Respect the Task 1.5
validity mask in the assignment.

Rationale for the docstring: LCZ classes differ in the *mixture* of surface types within the
patch, which mean pooling destroys and a permutation-invariant set encoder preserves.

All presets, cultural, 3 seeds, compared against Task 3.2.

### Task 3.4 — Matched Sentinel-1/2 baseline

The most important missing experiment. Without it there is no controlled claim that
foundation-model embeddings help — the comparison against Zhong et al. 2024 and Lin et al.
2024 is not controlled, since those add prior-knowledge coupling and semi-supervised
learning respectively.

**Layout is known from Phase 0 reconnaissance:** `{split}/sentinel{1,2}/sen{1,2}_patch_{id}.tif`,
directly under each split dir with **no** `{output_name}/{year}` nesting. S1 (8 bands) and
S2 (10 bands) are separate files, both 32×32 at 10 m in local UTM, `float64`,
`nodata=None`, no NaNs. Counts match the embeddings exactly (352,366 / 24,119 / 24,188) and
`{id}` is the same `patch_id`, so pairing is safe.

**3.4a — Loader.** Generalize `build_patch_index` with a configurable **extension**,
**filename prefix**, and **optional nesting level** (the embeddings nest
`{output_name}/{year}`, these do not). Dispatch `PatchDataset.__getitem__` on suffix:
`.npy` → `np.load`, `.tif`/`.tiff` → `rasterio` read as `(C, H, W)`, cast to `float32` on
read (the `float64` files double I/O for no benefit at 32×32).

Register `sentinel1`, `sentinel2`, `sentinel12` with `product: sentinel`, `version: null`,
`source: local_tif`, no dequantization function. `sentinel12` loads both and concatenates on
the channel axis — assert matching `patch_id` and 32×32 shape before concatenating. Per
Task 1.5.2's schema, mark them fully valid and say so in the docstring rather than searching
for a nodata mask at read time.

Everything downstream — splits, `PatchItem`, normalization, augmentation, logit adjustment,
model registry, `infer_roi.py` — then works unchanged. **That shared code path is the
point:** it is what makes this a matched baseline rather than a reimplementation with
different defaults.

**3.4b — Fair preprocessing.** If the raw modality gets naive preprocessing while the
embeddings got a tuned pipeline, the comparison is rigged and a reviewer will say so. Tune
it at least as hard.

Phase 0 established that S1 bands 5–8 have skew 560–1138 (band 6: std 7.02 against p99.9 of
18.55) and that these are a genuine heavy tail, not corruption. Bands 1–4 are near-symmetric
signed components; bands 5–6 are strictly positive.

Add `--clip-percentile P` (default `0.0` = off), clipping each channel to its [P, 100−P]
training-set percentiles before z-scoring. Run four variants and take the best forward:

1. Channel z-scoring only (identical treatment to the embeddings)
2. Percentile clipping at P ∈ {0.1, 1.0}, then z-scoring
3. Per-band log on bands 5–6 only, then z-scoring
4. The published So2Sat LCZ42 per-band scaling — check the dataset documentation

Record all four in `RESULTS.md`. Expect the ablation to matter enormously on S1 and barely
at all on S2 (skew 0.3–2.9). Reporting the sweep rather than only the winner is what makes
the fairness auditable.

**3.4c — Comparison runs.** Cultural split, ResNet34, 3 seeds, identical schedule, epochs,
patience, monitor and augmentation as the Phase 2 canonical-Tessera config. Modalities:
`sentinel1`, `sentinel2`, `sentinel12`. Headline comparison is `sentinel12` versus canonical
Tessera, everything else held constant.

Also run one fusion configuration — canonical Tessera concatenated with `sentinel12` — as an
upper-bound reference. If fusion substantially beats either alone, the embeddings discard
information the raw bands retain. Not the chapter's main claim, but a reviewer will ask.

**3.4d — Throughput.** 400k GeoTIFFs will read slower than `.npy`. Phase 1 found packing
worth only +7.6% on a warm cache because the per-patch cost was dequantize + resize, but
these files have per-file headers and compression and no dequantization, so packing should
matter considerably more here. Re-benchmark before launching the seed sweeps.

### Task 3.5 — Label-efficiency curve

Subsample the cultural-split training set to {1%, 2%, 5%, 10%, 25%, 50%, 100%}, stratified by
class, 3 seeds per point. Run the best embedding config and the winning `sentinel12` config
from 3.4 at every point, using **identical subsample indices** for both so the curves are
paired. Plot OA and macro-F1 against training-set size.

This is the claim foundation models actually make, and it is likely the strongest figure in
the chapter regardless of where the 100% numbers land.

### GATE 3

All five experiments with error bars, plus the label-efficiency plot.

---

## Phase 4 — Closing the cross-city gap

Branch `exp/p4-domain`. Start only after GATE 3.

### Task 4.1 — Per-city standardization

Uses the `city` field from Task 1.4. Add `--normalize percity`: subtract the per-city channel
mean and divide by the per-city channel std, computed over all patches of that city
regardless of split (label-free, so legitimate at test time — state this in the docstring).
Add `--normalize percity_mean_only` to separate the offset effect from the scale effect.
Cultural split, canonical Tessera, 3 seeds each against the Task 2.2 baseline.

Per Task 1.5.3, per-city stats caches carry the same provenance key as the global ones.

### Task 4.2 — AdaBN

Add `--adabn` to `infer_roi.py` and the test-time evaluation path: before predicting on a
held-out city, forward-pass in `train()` mode over that city's unlabelled patches to refresh
BatchNorm running statistics, then predict in `eval()` mode. No gradients, no labels.

Evaluate per held-out city and report city by city rather than pooled — the variance across
cities is itself a result.

### Task 4.3 — Domain-adversarial training

`--dann-lambda`: city-ID head on the pooled features with a gradient reversal layer, lambda
ramped over training. Also `--coral-weight` as the simpler alternative (align feature
covariances across city minibatches). Both, cultural, 3 seeds.

### Task 4.4 — Factorized head

`--factorized-head`: predict surface type (built/natural), density, and height as separate
factors and compose the 17-class logits from them. Define the class-to-factor mapping in
`src/utils/constants.py` alongside `lcz_dict`, following Stewart & Oke (2012).

LCZ 1/2/3 differ from 4/5/6 only in density, and within each triple only in height, so a flat
softmax discards the label structure. The factors should also transfer across cities better
than class appearance. Report both flat 17-class metrics and per-factor accuracy.

### Task 4.5 — Stratified reporting

Extend `training/evaluate.py` to break test metrics down by held-out city and by Köppen zone
(`koppen_dict` is already in `src/utils/constants.py`; the per-patch Köppen class needs
joining from the reference GPKG or sampling the Köppen raster).

Re-report every headline result from Phase 2 onward this way. "Which cities and climates does
this fail in" is more useful than a pooled number and directly supports the Global South
motivation.

### GATE 4

Domain-adaptation table, per-city and per-Köppen breakdowns, and a recommendation on which
combination to take forward.

---

## Out of scope

Do not start without checking in: WUDAPT label ingestion, the noise-transition matrix for
Demuzere pseudo-labels, semantic segmentation with polygon supervision, GeoClimate, Overture
labels. They are the next block of work and they depend on the Phase 2 numbers being
trustworthy.