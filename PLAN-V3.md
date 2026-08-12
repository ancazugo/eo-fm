# eo-fm — PLAN v3 (supersedes PLAN-v2.md from Phase 2 onward)

Phases 0, 1 and 1.5 are complete. `PLAN.md` and `PLAN-v2.md` stay in the repo as the
record; this is the live document.

---

## Scope restriction (new, applies to everything below)

**In scope — the only three families used from here on:**

| key | role |
|---|---|
| `tesserav1.1_global` | **canonical Tessera**, confirmed by the Task 1.5.4 audit |
| `alpha_earth_coop` | canonical AlphaEarth |
| `seamless` | ESD |

**Out of scope — do not run, do not include in any table:**

- `tessera` (v1 GEE zarr) and `alpha_earth` (GEE zarr) — outdated products, and unverified
  against current code since the `open_tile` bug postdates their 42 and 60 runs.
- `tesserav1.1` (per-city archive) — superseded. Task 1.5.1 established it is a separate
  inference run with a rotated basis, not a scale variant, and it has never produced a
  cultural-split number.
- `tesserav2` — extraction still in progress. Deferred, not cancelled; see the deferred
  section at the end.

Mark all four `deprecated` or `pending` in the registry so they do not appear available.
Task 2.5 from PLAN-v2 (the Tessera version comparison) is **removed** from Phase 2 and
moved to the deferred section.

## Carried forward

**Standing rules:** one factor per run, three seeds (`0, 1, 2`), W&B tags per phase, every
result appended to `RESULTS.md`, no refactoring beyond scope, stop at each GATE.

**GATE 1.5 findings that change downstream work:**

- The AlphaEarth sentinel is **not** diluted by the resize (31.04% → 30.86% of per-channel
  variance). The GATE 0 vs GATE 1 discrepancy was a **sampling** artifact: the paired
  intersection carries 0.0184% sentinel pixels against 15.7292% outside it, an 855× ratio.
  Masking therefore matters **more** than GATE 1 implied, since Phase 2 trains on all
  352,366 coop patches.
- Corrected effective noise ratio for AlphaEarth: with ~31% of variance from the sentinel,
  true signal std ≈ √0.69 × 0.1054 ≈ **0.088**, so the ratio is **~0.57**, not 0.477. The
  14× cross-family spread stands.
- Geometry: 10 m families crop natively at 33–36 × 32–35 and are downsampled ~1.031× to
  32×32 (0% no-op); `seamless` is 11–13 px natively and **upsampled 2.67× per dimension**;
  raw Sentinel is 32×32 with no resampling at all.
- Mask ordering needed no fix — Phase 1 already builds masks natively, mean-fills before the
  resize, and thresholds conservatively.

---

## Phase 1.75 — Two blocking data-characterization tasks

Branch `fix/p175-population`. Small — target two days. Both change what the model sees in
Phase 2, so both are blocking.

### Task 1.75.1 — Characterize the AlphaEarth nodata population

The 855× density ratio means the sentinel was characterized on a filtered population.
Re-characterize on the **full** `alpha_earth_coop` cultural-split training set, no pairing:

- per-patch invalid-fraction distribution (histogram, and p50/p90/p99/max)
- the count of patches above 5%, 10%, 25%, 50% invalid
- total invalid-pixel fraction across the whole training set
- the same for the val and test splits, reported separately

**Then the coverage-bias question, which is the important one.** Break the high-invalid
patches down by **city**, by **LCZ class**, and by **split**. Report the per-city invalid
fraction as a sorted table.

If high-invalid patches concentrate in particular cities, then `alpha_earth_coop` is
effectively being evaluated on a different city distribution from the other two families,
and the cross-family comparison in Task 2.2 inherits a city-level confound rather than a
pixel-level one. Check specifically whether any of the 10 held-out cultural-split cities are
affected, and whether any city exceeds 25% mean invalid.

Also report whether the *other* two families have their own coverage holes on the same
patch ids — the pairing hid everyone's, not just AlphaEarth's.

**Then set a drop policy.** Add `--max-invalid-frac F` (default `1.0` = keep everything).
Patches above the threshold are excluded from training *and* evaluation. Recommend a value
from the histogram and justify it. Report Task 2.2 both with and without the drop so the
choice is visible rather than baked in.

Finally, re-report the corrected channel stats and effective noise ratios for all three
families computed on the **full unfiltered training population** with masking applied, as a
diff against the Phase 0 and GATE 1 tables. This supersedes both.

### Task 1.75.2 — Crop geometry and the latitude question

The 10 m families crop at 33–36 × 32–35 native for a nominal 320 m patch, so the true
footprint is ~330–360 m at ~10.3–11.3 m effective resolution, and the extent **varies
between patches**.

Determine the source of that variation. Report, for a 5000-patch sample of
`tesserav1.1_global` and `alpha_earth_coop`:

- native crop width and height against **patch centre latitude** (scatter, plus Spearman ρ)
- native crop shape against the tile CRS — confirm whether tiles are in EPSG:4326 or a
  projected CRS, since a degree-based grid gives longitude pixel size varying as cos φ
- native crop shape against city, to see whether it is city-constant or within-city variable

**If crop extent correlates with latitude, effective spatial resolution varies systematically
between equatorial and high-latitude cities.** That is a preprocessing-induced domain shift
sitting inside the cultural split, on exactly the axis Phase 4 is trying to attack, and no
amount of test-time adaptation would fix it. Report ρ and the effective-resolution range
across the 52 cities either way — a null result here is worth having in the methods.

Do **not** change the resampling in this task. Bilinear at 32×32 is what every existing
number used, and changing it would invalidate Task 2.1's reproduction. The alternative is
tested as an ablation in Task 2.3.

### GATE 1.75

Report the invalid-fraction histogram and per-city table, the recommended
`--max-invalid-frac`, the corrected stats table superseding Phase 0 and GATE 1, and the
latitude correlation with its effective-resolution range.

---

## Phase 2 — Re-validation (revised)

Branch `exp/p2-revalidation`. **No changes to model or training logic in this phase.**

### Task 2.1 — Reproduce the pre-fix baseline

`--normalize none --nodata-mode zero --max-invalid-frac 1.0`, `tesserav1.1_global`, cultural
split, 3 seeds.

The Phase 1 RNG change means seed-level, not bit-level, agreement is the standard. Report
mean ± std and check the existing 0.6190 (opt3) falls inside. That spread is itself a
deliverable — every later improvement in the chapter has to clear it.

If not already present, add a test that pre- and post-Phase-1 augmentation produce
statistically indistinguishable output distributions (KS test, large sample). That separates
"the RNG stream changed" from "the augmentation semantics changed."

### Task 2.2 — The fixed cross-family comparison

Cultural split, ResNet34 (`--family resnet --preset small`), Phase 1 defaults
(`--normalize channel --nodata-mode mask`), the Task 1.75.1 drop policy, 3 seeds,
`--lr 5e-4 --warmup-epochs 3`.

Families: `tesserav1.1_global`, `alpha_earth_coop`, `seamless`. Report mean ± std for OA,
macro-F1, kappa, plus a confusion matrix for the best run of each. Run the AlphaEarth arm
twice — with and without the drop policy — per Task 1.75.1.

### Task 2.3 — Factorial ablation of the Phase 1 fixes

Do **not** run these as single-factor rows. Once σ is expressed relative to the normalized
channel std, disabling normalization silently reverts σ to absolute units, so "normalization
off" and "noise miscalibration" are entangled and the result would not answer which fix
mattered.

| | σ = 0 | σ = 0.05 |
|---|---|---|
| `--normalize none` | **A** | **C** (old behaviour) |
| `--normalize channel` | **B** | **D** (new default) |

B − A isolates normalization. C − A is the damage the old absolute-σ noise was doing.
D − B is what calibrated noise buys. Run on `tesserav1.1_global` and `alpha_earth_coop` —
opposite ends of the 14× spread. 4 × 2 × 3 = 24 runs.

Expect **B − A to be near zero**: ResNet has BatchNorm immediately after `conv1`, so the
network already largely self-normalizes. That would not mean the normalization work was
wasted — its value is making σ mean the same thing across families, which is what C − A and
D − B measure. A flat B − A with a large D − B is the clean version of the story and should
be written up that way.

Then, on the `--normalize channel` arm only:

- **σ sweep** {0, 0.025, 0.05, 0.1, 0.2}, 3 seeds, both families. 0.05 was inherited, never
  tuned.
- **nodata mode** `zero` vs `mask` on `alpha_earth_coop`, now on the unfiltered population
  where the sentinel is 855× denser than when this was last predicted neutral.
- **resize mode**: add `--resize-mode {bilinear, exact-crop}`, where `exact-crop` takes the
  centre 32×32 of the native crop with no interpolation. Tests whether the ~1.031×
  downsample costs anything, and it makes the validity mask exact rather than thresholded.
  `tesserav1.1_global` and `alpha_earth_coop`, 3 seeds.

### Task 2.4 — Model selection metric

Best `tesserav1.1_global` config with `--monitor val_f1`, `val_kappa`, `val_acc`, 3 seeds
each. Report the full metric triplet for each so the cost of selecting on the wrong one is
visible.

### Task 2.5 — Resolution versus representation (new, replaces the version comparison)

`seamless` is upsampled 2.67× per dimension from ~12 native pixels, so resolution and
representation quality are currently entangled and every ESD number in the chapter is
computed on interpolated data.

Disentangle them. Add `--native-resolution` (skip the resize, use the native crop) and
confirm the ResNet stem handles 12×12 — with the 3×3 stride-1 stem and no maxpool, the
spatial trace is 12 → 12 → 6 → 3 → 2, which is fine.

Three runs, 3 seeds each:

| arm | input | question |
|---|---|---|
| `seamless` @ ~12×12 native | no interpolation | does upsampling help, hurt, or neither? |
| `tesserav1.1_global` @ 12×12 | downsampled from native | matched-resolution control |
| `alpha_earth_coop` @ 12×12 | downsampled from native | matched-resolution control |

If Tessera at 12×12 ≈ ESD at 12×12, ESD's deficit is purely resolution. If Tessera still wins
at matched resolution, the ESD representation is worse independently of resolution. Either
answer is stronger than the 32×32 comparison alone.

Frame this correctly in the write-up: 30 m giving ~11 pixels per 320 m patch is a **genuine
physical limitation** for a class system defined by sub-100 m morphology, not an unfairness
in the pipeline. The experiment separates the limitation from the artifact.

### GATE 2

Report the reproduction with seed variance, the three-family table with error bars, the 2×2
factorial with its three contrasts, the σ / nodata / resize ablations, the monitor
comparison, and the resolution-versus-representation table. **Decision point for the
chapter** — if the ranking or absolute numbers moved, existing draft conclusions need
rewriting before any new modelling.

**Pre-register before launching.** Commit predictions to `RESULTS.md` first. Mine, revised
for the corrected AlphaEarth noise ratio of ~0.57: `tesserav1.1_global` moves less than 1
point either way; `alpha_earth_coop` improves most and largely via the noise-ratio change;
`seamless` improves modestly; B − A ≈ 0 with D − B carrying the effect; nodata masking now
**non-trivial** for AlphaEarth (I revise GATE 1's "near-neutral" — that rested on the
filtered population); the resize-mode ablation is near-neutral for the 10 m families. I put
the AlphaEarth–Tessera gap closing by 1–6 of its 9 points, ranking flip genuinely uncertain.
If AlphaEarth does not improve at all, the noise was never the binding constraint and the gap
needs a different explanation — most likely that 64-d annual composites carry less
LCZ-relevant structure than 128-d time-series representations, which is a cleaner result to
write up than a confound.

---

## Phase 3 — Ablations that define the contribution

Branch `exp/p3-ablations`. All runs use `tesserav1.1_global` unless stated.

### Task 3.1 — Permutation test

Add `--permute-pixels`: randomly permute pixel positions within each patch (same permutation
across channels of a sample, fresh per sample per epoch), applied in augmentation for train
and deterministically for val/test. Cultural, 3 seeds, with and without.

If accuracy is unchanged, spatial arrangement within a 320 m patch carries no usable signal
and the CNN is the wrong inductive bias. Either outcome is a publishable figure.

### Task 3.2 — The 2×2 pooling × split table

| | cultural | grid |
|---|---|---|
| GAP + linear probe | ~0.61 (rerun) | **missing** |
| ResNet34 | ~0.65 (rerun) | ~0.85 (rerun) |

3 seeds each. If the CNN's margin over GAP is much larger on the grid split, its spatial
features are city-specific and do not transfer — the cleanest statement of the problem this
chapter addresses.

### Task 3.3 — Set-encoder family

Register `netvlad` in `src/models/`, following the registry pattern in `linear_probe.py` and
`mlp.py`. Input `(B, C, H, W)` → flatten to `(B, HW, C)` → soft assignment to K learned
prototypes → per-cluster residual sums → intra-normalize, flatten, L2-normalize → linear
classifier. Presets `nano` K=16 through `large` K=256. Initialize prototypes by k-means on a
200k-pixel training sample (cache it). Respect the validity mask in the assignment.

Rationale for the docstring: LCZ classes differ in the *mixture* of surface types within the
patch, which mean pooling destroys and a permutation-invariant set encoder preserves.

All presets, cultural, 3 seeds, against Task 3.2.

### Task 3.4 — Matched Sentinel-1/2 baseline

The most important missing experiment. Without it there is no controlled claim that
foundation-model embeddings help — the comparison against Zhong et al. 2024 and Lin et al.
2024 is not controlled, since those add prior-knowledge coupling and semi-supervised
learning respectively.

**Layout, confirmed in Phase 0:** `{split}/sentinel{1,2}/sen{1,2}_patch_{id}.tif`, directly
under each split dir with **no** `{output_name}/{year}` nesting. S1 (8 bands) and S2 (10
bands) separate, 32×32 at 10 m in local UTM, `float64`, `nodata=None`, no NaNs. Counts match
the embeddings exactly (352,366 / 24,119 / 24,188); `{id}` is the same `patch_id`.

**Note from Task 1.5.0:** raw Sentinel is the **only** modality reaching the model without
resampling — 100% no-op. Every embedding family is interpolated to some degree. State this
in the write-up; it is a point in the baseline's favour and pre-empts the obvious reviewer
objection that the baseline was handicapped.

**3.4a — Loader.** Generalize `build_patch_index` with configurable **extension**, **filename
prefix**, and **optional nesting level**. Dispatch `PatchDataset.__getitem__` on suffix:
`.npy` → `np.load`; `.tif`/`.tiff` → `rasterio` read as `(C, H, W)`, cast to `float32` on
read. Register `sentinel1`, `sentinel2`, `sentinel12` with `product: sentinel`,
`version: null`, `source: local_tif`, no dequantization. `sentinel12` loads both and
concatenates on the channel axis — assert matching `patch_id` and shape first. Mark them
fully valid in the docstring rather than searching for a nodata mask at read time.

Everything downstream then works unchanged. **That shared code path is the point** — it is
what makes this a matched baseline rather than a reimplementation with different defaults.

**3.4b — Fair preprocessing.** If the raw modality gets naive preprocessing while the
embeddings got a tuned pipeline, the comparison is rigged and a reviewer will say so.

Phase 0 established S1 bands 5–8 have skew 560–1138 (band 6: std 7.02 against p99.9 of
18.55), confirmed in Phase 1 as a genuine heavy tail, not corruption. Bands 1–4 are
near-symmetric signed components; bands 5–6 are strictly positive.

Add `--clip-percentile P` (default `0.0`). Four variants, best taken forward:

1. Channel z-scoring only (identical treatment to the embeddings)
2. Percentile clipping at P ∈ {0.1, 1.0}, then z-scoring
3. Per-band log on bands 5–6 only, then z-scoring
4. The published So2Sat LCZ42 per-band scaling — check the dataset documentation

Record all four. Expect large effects on S1, negligible on S2 (skew 0.3–2.9). Reporting the
sweep rather than only the winner is what makes the fairness auditable.

**3.4c — Comparison runs.** Cultural, ResNet34, 3 seeds, identical schedule, epochs, patience,
monitor and augmentation as the Phase 2 `tesserav1.1_global` config. Modalities `sentinel1`,
`sentinel2`, `sentinel12`. Headline: `sentinel12` versus `tesserav1.1_global`.

Also one fusion arm — `tesserav1.1_global` + `sentinel12` concatenated — as an upper bound.
If fusion substantially beats either alone, the embeddings discard information the raw bands
retain. Not the main claim, but a reviewer will ask.

**3.4d — Throughput.** Phase 1 found packing worth only +7.6% warm because the per-patch cost
was dequantize + resize. These files have per-file headers and compression, no dequantization,
and no resize — packing should matter considerably more. Re-benchmark before the seed sweeps.

### Task 3.5 — Label-efficiency curve

Subsample the cultural-split training set to {1%, 2%, 5%, 10%, 25%, 50%, 100%}, stratified by
class, 3 seeds per point. Best embedding config and winning `sentinel12` config at every
point, **identical subsample indices** for both so the curves are paired. Plot OA and
macro-F1 against training-set size.

This is the claim foundation models actually make, and likely the strongest figure in the
chapter regardless of where the 100% numbers land.

### GATE 3

All five experiments with error bars, plus the label-efficiency plot.

---

## Phase 4 — Closing the cross-city gap

Branch `exp/p4-domain`. Start only after GATE 3.

If Task 1.75.2 found a latitude–resolution correlation, add it as a covariate to every
stratified result below — it would be a preprocessing-induced component of the domain gap and
must be separated from the genuine one.

### Task 4.1 — Per-city standardization

`--normalize percity`: subtract the per-city channel mean, divide by the per-city channel std,
over all patches of that city regardless of split (label-free, legitimate at test time — state
this in the docstring). Plus `--normalize percity_mean_only` to separate offset from scale.
Cultural, `tesserav1.1_global`, 3 seeds each against Task 2.2. Per-city stats caches carry the
same provenance key as the global ones.

### Task 4.2 — AdaBN

`--adabn` on `infer_roi.py` and the test-time path: before predicting on a held-out city,
forward-pass in `train()` mode over that city's unlabelled patches to refresh BatchNorm
statistics, then predict in `eval()` mode. No gradients, no labels. Report city by city, not
pooled — the variance across cities is itself a result.

### Task 4.3 — Domain-adversarial training

`--dann-lambda`: city-ID head on pooled features with a gradient reversal layer, lambda ramped
over training. Also `--coral-weight` as the simpler alternative. Both, cultural, 3 seeds.

### Task 4.4 — Factorized head

`--factorized-head`: predict surface type (built/natural), density and height as separate
factors, compose the 17-class logits from them. Mapping in `src/utils/constants.py` alongside
`lcz_dict`, per Stewart & Oke (2012). LCZ 1/2/3 differ from 4/5/6 only in density and within
each triple only in height, so a flat softmax discards the label structure. Report both flat
17-class metrics and per-factor accuracy.

### Task 4.5 — Stratified reporting

Extend `training/evaluate.py` to break test metrics down by held-out city and by Köppen zone
(`koppen_dict` is in `src/utils/constants.py`; per-patch Köppen needs joining from the
reference GPKG or sampling the raster). Re-report every headline result from Phase 2 onward
this way.

### GATE 4

Domain-adaptation table, per-city and per-Köppen breakdowns, recommendation on what to carry
forward.

---

## Deferred

**Tessera v2**, once extraction completes. Run the Task 2.2 configuration under identical
settings, restricted to the intersection of `patch_id`s present in both extractions, reported
as a **separate table** from the cross-family comparison — it compares model generations under
fixed extraction and split, not architectures. Note from Task 1.5.1 that v2's per-pixel L2
norm is effectively constant (11.295 / 11.314 / 11.333 at p1/p50/p99) where v1.1's spans
7.9–36.1, so per-pixel magnitude carries information in v1.1 and none in v2; if v2
underperforms, that is the first hypothesis to test.

**Out of scope entirely for this chapter** — do not start without checking in: WUDAPT label
ingestion, the noise-transition matrix for Demuzere pseudo-labels, semantic segmentation with
polygon supervision, GeoClimate, Overture labels.

---

## Write-up note

Task 1.5.1's basis-rotation result belongs in the paper even though it is now out of
operational scope: two Tessera archives sharing a version label have matched-channel r ≈ 0,
yet a linear 128→128 map recovers 89% held-out R² against a −0.02 null. Same information,
arbitrary basis. The implications — per-channel statistics and normalizers do not transfer
between archives, channel-wise interpretability claims are meaningless, and linear probes are
basis-invariant where channel-wise operations are not — make this a reproducibility finding
the EO foundation-model community should hear. One paragraph, with the table.

---

## Housekeeping (do these first, they take minutes)

- `git add PLAN-v2.md` and this file — `RESULTS.md` cites them as plan of record while both
  are untracked.
- Resolve the uncommitted exact-footprint change in `src/datasets/tiles.py`: commit it with a
  message explaining the change, or revert it. It must not sit uncommitted through Phase 2,
  where it would silently affect extraction geometry and be invisible in the run record.