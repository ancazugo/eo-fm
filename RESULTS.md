# RESULTS

Running log of every result produced by `PLAN.md` (Phases 0–1) and `PLAN-V2.md`
(Phase 1.5 onward). Diagnostics get a section per
task; training runs get one table row each (phase, run name, W&B id, embedding,
split, model, seed, OA, macro-F1, kappa, and the one factor that changed).

---

## Phase 0 — Diagnostics (measure only)

Both scripts are measure-only: nothing under `src/training/`, `src/models/` or
`src/datasets/` was touched, and `git status` after the runs shows only new files.

Reproduce:

```bash
python src/diagnostics/embedding_stats.py --n-sample 5000 --seed 0
python src/diagnostics/embedding_stats.py --families tesserav1.1 --n-sample 5000 --seed 0 \
    --output-md diagnostics/embedding_stats_tesserav11_51city.md
python src/diagnostics/verify_alphaearth_dequant.py --n-sample 5000 --seed 0 \
    --gee-dir ${DATA_DIR}/input/Google/AlphaEarth/2017 --gee-n 200
```

Artefacts: `diagnostics/embedding_stats.json`, `diagnostics/alphaearth_dequant.json`
(+ the `.md` fragments reproduced below). Both runs were executed twice and produced
identical numbers.

### Task 0.1 — Embedding scale audit

Cultural-split **training** set (the original So2Sat `training/` dir *is* the culture-10
train set: 352,366 patches), 5000 patches sampled **paired** across families (intersection
342,297; seed 0). Measured after the exact training-path transform — `np.nan_to_num` →
the `utils.runtime.resolve_dequantize` function → bilinear resize to 32×32 — verified
bit-identical to `datasets.so2sat.PatchDataset.__getitem__` on sample patches.

| family | C | per-channel std (median) | std min | std max | L2 norm p1 / p50 / p99 | frac zero (raw) | frac NaN (raw) | all-zero px | **`0.05 / median_std`** |
|---|---|---|---|---|---|---|---|---|---|
| `alpha_earth_coop` | 64 | **0.1054** | 0.0735 | 0.1493 | 0.986 / 0.998 / 1.003 | 0.01% | 0.00% | 0.00% | **0.474** |
| `tesserav1.1_global` | 128 | **1.1408** | 0.7217 | 1.7900 | 7.979 / 15.126 / 36.132 | 0.89% | 0.00% | 0.00% | **0.044** |
| `seamless` | 72 | **0.5260** | 0.4192 | 0.7486 | 3.767 / 4.987 / 6.621 | 0.02% | 0.00% | 0.00% | **0.095** |
| `sentinel1` | 8 | **0.5909** | 0.1767 | 7.0227 | 0.052 / 0.341 / 3.366 | 1.03% | 0.00% | 0.00% | **0.085** |
| `sentinel2` | 10 | **0.0816** | 0.0408 | 0.0988 | 0.102 / 0.470 / 1.165 | 0.00% | 0.00% | 0.00% | **0.613** |
| `tesserav1.1` (51-city subset) | 128 | **0.7975** | 0.4203 | 1.5295 | 6.490 / 12.553 / 22.586 | 0.92% | 0.00% | 0.00% | **0.063** |

Median vs RMS per-channel std (the RMS is the scale-free summary; the median is what
the ratio above uses):

| family | median std | RMS std | std p10 | std p90 | ratio (median) | ratio (RMS) |
|---|---|---|---|---|---|---|
| `alpha_earth_coop` | 0.1054 | 0.1104 | 0.0904 | 0.1329 | 0.474 | 0.453 |
| `tesserav1.1_global` | 1.1408 | 1.1726 | 0.8903 | 1.4090 | 0.044 | 0.043 |
| `seamless` | 0.5260 | 0.5484 | 0.4690 | 0.6340 | 0.095 | 0.091 |
| `sentinel1` | 0.5909 | 2.6468 | 0.1774 | 3.6113 | 0.085 | 0.019 |
| `sentinel2` | 0.0816 | 0.0782 | 0.0480 | 0.0982 | 0.613 | 0.640 |

Tail / asymmetry (per-channel, pooled across channels):

| family | mean of per-channel means | median \|skew\| | max \|skew\| | min p0.1 | max p99.9 |
|---|---|---|---|---|---|
| `alpha_earth_coop` | −0.0061 | 0.28 | 0.82 | −0.4746 | 0.4453 |
| `tesserav1.1_global` | −0.1542 | 0.46 | 1.42 | −9.206 | 10.54 |
| `seamless` | +0.0300 | 0.39 | 1.69 | −1 | 1 |
| `sentinel1` | +0.0403 | **282.65** | **1138.44** | −3.313 | 18.55 |
| `sentinel2` | +0.1438 | 1.53 | 2.92 | 0.0006 | 0.6292 |
| `tesserav1.1` (51-city) | −0.0242 | 0.47 | 1.73 | −6.91 | 6.93 |

**Effective noise ratio — PLAN.md's expected finding is CONFIRMED, and larger than
expected.** `augment_images` adds `N(0, 0.05)` in absolute units to unnormalized inputs,
so the relative perturbation spans **14×** across families (0.044 for Tessera up to 0.613
for Sentinel-2) and **10.8×** between the two headline embeddings:

- Tessera v1.1 global sees noise at **4.4%** of a channel std — barely an augmentation.
- AlphaEarth coop sees **47.4%** — a very strong corruption.
- ESD/seamless sees **9.5%**; Sentinel-2 sees **61.3%**.

The predicted magnitudes hold: AlphaEarth ≈ 0.125 (measured 0.105 median / 0.110 RMS —
slightly under 1/√64 because the per-channel means are not zero), ESD ≈ 0.5 (measured
0.526). Tessera is the outlier at 1.14, an order of magnitude above AlphaEarth.
**The cross-family comparison is confounded as suspected**, and in the direction that
flatters Tessera: the family that currently wins is the one receiving the least
effective augmentation noise.

`tesserav1.1` (51-city) and `tesserav1.1_global` differ in scale (0.80 vs 1.14 median
std), so they are not interchangeable for any normalization-sensitive result.

#### Sentinel-1 per-band detail (input to Task 3.4b)

| band | mean | std | skew | p0.1 | p99.9 |
|---|---|---|---|---|---|
| 1 | −0.0000 | 0.1777 | −5.83 | −1.0480 | 1.0417 |
| 2 | −0.0001 | 0.1767 | −1.05 | −1.0341 | 1.0543 |
| 3 | +0.0006 | 0.4712 | −1.34 | −3.0799 | 3.2219 |
| 4 | +0.0000 | 0.4746 | −0.97 | −3.3125 | 3.1484 |
| 5 | +0.0451 | 1.0475 | **1138.44** | 0.0008 | 1.5732 |
| 6 | +0.2749 | **7.0227** | **559.46** | 0.0020 | **18.5514** |
| 7 | +0.0012 | 2.1493 | **777.82** | −1.2458 | 1.2232 |
| 8 | +0.0009 | 0.7071 | **797.20** | −0.7490 | 0.7207 |

Bands 1–4 (the real/imaginary components) are near-symmetric and well behaved; bands 5–8
are extraordinarily heavy-tailed (skew 560–1138, band 6 std 7.02 against a p99.9 of 18.6).
Naive z-scoring will be dominated by a handful of extreme pixels — this is the concrete
justification for `--clip-percentile` in Task 3.4b, and the ablation should be expected
to matter a great deal on S1 and very little on S2 (skew 0.3–2.9).

### Task 0.2 — AlphaEarth coop dequantization verification

5000 training patches. AlphaEarth embeddings are unit-norm 64-d vectors, so the correct
decoding is the one whose per-pixel L2 norm concentrates at 1.0.

| candidate decoding | L2 norm mean | std | p1 | p50 | p99 | within 1% of 1.0 |
|---|---|---|---|---|---|---|
| `((v/127.5)**2)*sign(v)` — **current** | 1.0292 | 0.4524 | 0.9955 | **1.0001** | 1.0051 | **99.59%** |
| `v/127.5` | 2.5556 | 0.3543 | 2.4419 | 2.5339 | 2.6230 | 0.00% |
| `sign(v)*(\|v\|/127.5)` + L2 renorm | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% † |

† Candidate 3 forces norm 1.0 by construction, so it carries no evidence; the
discriminating comparison is candidate 1 against candidate 2.

Pearson r between decodings over all values: current vs linear **0.9305**,
current vs linear+renorm 0.8836, linear vs linear+renorm 0.9905.

#### GEE float32 cross-check (200 patches, 615-tile pool)

The GEE reference tiles are themselves unit-norm — per-pixel L2 norm mean 1.00009,
std 0.0020, **100%** within 1% of 1.0 — which independently confirms the premise.

| candidate | per-channel r (median) | r (min) | RMSE | pixelwise r |
|---|---|---|---|---|
| `((v/127.5)**2)*sign(v)` — **current** | **0.9990** | 0.9959 | **0.0042** | n/a |
| `v/127.5` | 0.9716 | 0.9319 | 0.1829 | n/a |
| `sign(v)*(\|v\|/127.5)` + L2 renorm | 0.9719 | 0.9341 | 0.0266 | n/a |

Compared on per-channel patch means over 200 patches (13 more had no GEE coverage).
The coop and GEE products sit on different pixel grids — the crops came out a different
shape in 200/200 cases — so a per-pixel r is not available; the per-channel comparison is
alignment-tolerant and is the appropriate statistic.

**Verdict: the current decoding is CORRECT.** Both the internal norm test and the external
GEE cross-check agree, so **Task 1.1 is a no-op** — no fix, no re-extraction, no legacy
alias needed. The distortion estimate GATE 0 asks for is moot, but for the record the
plain-linear alternative correlates with the current decoding at r = 0.93, i.e. had the
current decoding been wrong the embeddings would have been substantially distorted.

### Incidental findings

**1. AlphaEarth coop carries an undocumented nodata sentinel that `nan_to_num` does not
catch.** The value −128 occurs *only* as an all-64-channel pixel — never in a single
channel alone (verified over 1000 patches: 7,676 pixels with any −128, all 7,676 with all
64 channels at −128, and exactly that set fails the norm test). There are no NaNs at all.
Under the current decoding each such pixel becomes 64 channels of −1.00786, i.e. an L2
norm of **8.0629** rather than 1.0 — an 8× magnitude spike fed straight into training.

- 0.41% of pixels in the 5000-patch sample (0.68% in the 1000-patch sample), concentrated
  in ~0.8% of patches.
- This is the entire explanation for the current decoding's L2 mean of 1.0292 and std of
  0.4524 despite p1…p99 sitting at 1.0.
- **Bearing on Task 1.5:** `--nodata-mode mask` must treat all-channel −128 as invalid for
  `alpha_earth_coop`, not just NaN. PLAN.md assumed the `.npy` files carry no explicit
  nodata value; for this family they do, it is simply not NaN. Post-Task-1.2 these pixels
  would otherwise normalize to roughly −10 σ.

**2. `datasets.tiles.open_tile` cannot open any `.zarr` tile.** It tests `path.is_dir()`
and routes to the Tessera-global NPY reader *before* testing `path.suffix == ".zarr"` —
but a zarr store **is** a directory, so every `.zarr` tile raises "NPY files not found"
and the `.zarr` branch below is unreachable. This breaks the `alpha_earth` (GEE) and
`tessera` families end-to-end (extraction and `infer_roi`); it goes unnoticed because all
current work uses `alpha_earth_coop` / `tesserav1.1_global`. Phase 0 forbids editing
`src/datasets/`, so `verify_alphaearth_dequant.py` works around it with a local
`open_gee_zarr()` and the bug is reported here — the one-line reordering belongs in
Phase 1.

**3. Sentinel-1/2 layout (early answer to Task 3.4a).** The raw patches sit **directly**
under each split dir — `{split}/sentinel1/sen1_patch_{id}.tif` and
`{split}/sentinel2/sen2_patch_{id}.tif` — with no `{output_name}/{year}` nesting, unlike
the embedding `.npy` files. S1 (8 bands) and S2 (10 bands) are **separate files**, both
32×32 at 10 m in the patch's local UTM zone, `float64`, and **no nodata value is set** in
the profile (`nodata=None`), nor are there any NaNs — checked on 15 sampled patches per
modality in each of the three splits, consistent throughout.
Counts match the embeddings exactly (352,366 / 24,119 / 24,188), and the
`{id}` field is the same `patch_id`, so pairing by id is safe. Task 3.4a's
`build_patch_index` generalization therefore needs both a configurable extension **and** a
configurable filename prefix, and Task 3.4a's plan to "read the nodata value from the
raster profile" will find nothing to read — the validity mask for S1/S2 must come from
somewhere else.

### GATE 0 status

Reached; Phase 1 followed (see below).

---

## Phase 1 — Correctness fixes

Branch `fix/p1-correctness`, one commit per task, amendments A1–A9 applied.
`pytest` = 22 passed (`tests/test_tiles_open.py`, `test_nodata_masking.py`,
`test_channel_stats.py`, plus the two existing offline suites).

### Follow-up measurements requested at GATE 0

**A4 — is Tessera's zero an all-channel sentinel?** Yes, but a small one, and a
per-channel test would be catastrophic (2000 patches per family):

| family | values == 0 | pixels with **all** C channels 0 | pixels with ≥1 zero channel | patches affected |
|---|---|---|---|---|
| `tesserav1.1_global` | 0.878% | **0.116%** | 61.0% | 74 / 2000 |
| `tesserav1.1` | 0.926% | **0.146%** | 58.0% | 89 / 2000 |
| `tesserav2` | 2.301% | **0.053%** | 94.0% | 33 / 2000 |

The zeros-per-pixel histogram decays smoothly (0→1→2→3… channels zero), i.e.
int8 quantization; but at a ~0.9% independent per-channel rate P(all 128 zero)
≈ 0, so the all-zero pixels are a separate nodata population. `seamless` has
0.0000% all-zero pixels. Predicate adopted: **all channels equal the sentinel**,
tested on the array as stored.

**A3/A5 premise — the sentinel's contribution, and where it applies.** Masked vs
unmasked per-channel std, 3000 patches, **native grid**:

| family | invalid | median std unmasked | masked | std inflation | variance share |
|---|---|---|---|---|---|
| `alpha_earth_coop` | 0.433% | 0.1244 | 0.1057 | **+17.8%** | **27.9%** |
| `tesserav1.1_global` | 0.117% | 1.1642 | 1.1646 | −0.0% | ~0% |
| `seamless` | 0.000% | 0.5583 | 0.5583 | 0.0% | 0% |

So A3's reasoning holds in direction but the magnitude is **28% of variance /
18% of std**, not 37%/26% — and it applies to AlphaEarth *only*. Tessera's
sentinel value is 0, which sits at the channel mean, so masking it moves σ not
at all. Reordering 1.5 before 1.2 remains the right dependency.

**A9 — are the Sentinel-1 extremes corruption or a real tail?** A real tail
(5000 patches):

| band | min | max | p99.9 | > 10×p99.9 | patches hit | inf | NaN | % negative |
|---|---|---|---|---|---|---|---|---|
| 1 | −34.15 | 30.17 | 1.297 | 0.0006% | 5 | 0 | 0 | 48.5% |
| 2 | −29.95 | 21.23 | 1.304 | 0.0000% | 6 | 0 | 0 | 48.6% |
| 3 | −68.09 | 103.1 | 4.062 | 0.0000% | 10 | 0 | 0 | 49.2% |
| 4 | −89.46 | 83.92 | 3.971 | 0.0003% | 8 | 0 | 0 | 49.3% |
| 5 | 3.73e−05 | 857.1 | 1.422 | 0.0031% | 74 | 0 | 0 | **0.0%** |
| 6 | 1.60e−04 | **7012** | 15.04 | 0.0103% | 114 | 0 | 0 | **0.0%** |
| 7 | −2168 | 224 | 1.816 | 0.0041% | 101 | 0 | 0 | 49.5% |
| 8 | −179 | 1128 | 1.114 | 0.0053% | 105 | 0 | 0 | 49.0% |

No infinities, no NaNs. The extremes spread over **199/5000 patches (4.0%)** and
the top-10 patch maxima decay smoothly (band 6: 7012, 3130, 2349, 2259, 2180,
2180, 1927, …) rather than showing an isolated corrupt value. **Keep percentile
clipping for Task 3.4b; do not switch to dropping bad pixels.** Bands 5–6 are
strictly positive (intensity-like), so a per-band log transform is applicable to
those two even though it is not to the signed bands — worth a fourth variant.

### Invalid-pixel fraction per family (Task 1.5, n=5000)

| family | invalid pixels | patches containing ≥1 |
|---|---|---|
| `alpha_earth_coop` | 0.4120% | 0.46% |
| `tesserav1.1_global` | 0.1261% | 4.04% |
| `tesserav1.1` | 0.1376% | 4.10% |
| `tesserav2` | 0.0722% | 1.84% |
| `seamless` | 0.0000% | 0.00% |
| `sentinel1` | 0.0000% | 0.00% |
| `sentinel2` | 0.0000% | 0.00% |

The two nodata populations have opposite shapes: AlphaEarth's is concentrated
(few patches, heavily affected), Tessera's is diffuse (4% of patches, lightly).

### Corrected channel statistics (Task 1.2 / A5, n=20000, masked, post-resize)

| family | Phase 0 median std | corrected | Δ | Phase 0 `0.05/std` | corrected |
|---|---|---|---|---|---|
| `alpha_earth_coop` | 0.1054 | 0.1049 | −0.5% | 0.474 | **0.477** |
| `tesserav1.1_global` | 1.1408 | 1.1324 | −0.7% | 0.044 | **0.044** |
| `seamless` | 0.5260 | 0.5468 | +4.0% | 0.095 | **0.091** |
| `tesserav1.1` | 0.7975 | 0.7946 | −0.4% | 0.063 | **0.063** |
| `tesserav2` | — | 0.3724 | n/a | — | **0.134** |

**The corrections are small and the Phase 0 conclusion is unchanged**: the
effective noise ratio still spans ~14× across families.

> **Correction (Task 1.5.0).** This section originally attributed the small
> corrections to bilinear resize diluting an isolated sentinel pixel. That is
> **wrong** — the resize changes the sentinel's variance share by 0.18 pp, not
> 27 pp. The real reason is reason 2 below, which turns out to explain the
> AlphaEarth row as well: Phase 0's paired sample is essentially sentinel-free.
> See "Task 1.5.0 — Resize audit" for the measurement.

1. ~~The Phase 0 table was already **post-resize**, and bilinear resize dilutes
   an isolated sentinel pixel.~~ Superseded — see the correction above.
2. The corrected run samples each family's own index at n=20000, whereas Phase 0
   used the 5-family paired intersection at n=5000 — which is most of the
   +4.0% on `seamless`, a family with no nodata at all, and **all** of the
   AlphaEarth agreement: the paired intersection carries 0.0184% sentinel pixels
   against 15.73% outside it.

`tesserav2` is new here: median std 0.372, i.e. a third of `tesserav1.1_global`'s
scale, so the two Tessera generations are **not** interchangeable under a shared
normalizer either.

### Throughput (Task 1.8)

Nairobi, AlphaEarthCoop, batch 64, 4 workers, 3 alternating A/B rounds in one
process (separate invocations are dominated by page-cache drift — a naive
before/after made the change look like a 1.8× regression):

| | patches/s |
|---|---|
| loader, augmentation in collate (pre-1.8) | 1,010 |
| loader, augmentation in train_step (post-1.8) | **3,051** (3.0×) |
| augmentation alone, CPU | 631 |
| augmentation alone, GPU | **30,715** (49×) |
| per-file reads | 726 |
| packed shard reads | 781 (+7.6%, warm cache) |

Packing is worth much less than expected: with a warm page cache the per-patch
cost is dominated by dequantize and resize, not by opening files. It should
matter more on cold cache and for the GeoTIFF sources in Task 3.4. Packed output
is byte-identical to per-file reads (verified over 50 patches).

### Regression check

`infer_roi.py` from a pre-Phase-1 checkpoint (`student-coop-v1`, resnet/small,
AlphaEarthCoop) over Nairobi with `--normalize none` reproduces the pre-change
GeoTIFF **bit for bit**: md5 `7c2222b04ad830fe0c08eb4ee686df51`, both before any
Phase 1 commit and at the branch tip. Without `--normalize none` the same
checkpoint is refused with an error naming the flag, rather than silently
running unnormalised.

### Behaviour changes to be aware of

- **Defaults changed**: `--normalize channel` and `--nodata-mode mask` are now
  the defaults, so a re-run of an old command does not reproduce an old run.
  `--normalize none --nodata-mode zero` is the pre-Phase-1 path.
- **The augmentation RNG stream changed** (one batched draw instead of N scalar
  draws), so a fixed seed no longer reproduces the old augmentation sequence.
  The distribution is unchanged.
- **`--class-weights` with `--logit-adjustment` is now an error** (both correct
  the same imbalance).
- A latent bug was fixed in passing: the collate augmented `image` but not the
  new `valid` mask, which would have left the mask misaligned from its data.

### GATE 1 status

Reached. Phase 2 not started.

---

## Phase 1.5 — Provenance, versioning, and the resize diagnostic

Branch `fix/p15-provenance`, cut from `fix/p1-correctness`. Plan of record:
`PLAN-V2.md` §Phase 1.5.

```bash
python src/diagnostics/resize_audit.py --n-sample 2000 --seed 0
```

Artefacts: `diagnostics/resize_audit.json`, `diagnostics/resize_audit.md`.

### Task 1.5.0 — Resize audit

2000 patches per family, cultural-split **training** set, each family sampled
over its own patch_ids (seed 0), target 32×32.

#### Geometry — nothing is 32×32 natively except the raw Sentinel patches

| family | native shapes (top 3) | exact no-op | H factor | W factor |
|---|---|---|---|---|
| `alpha_earth_coop` | 33×34 (674), 34×33 (547), 33×33 (319) | 0.0% | 1.031 | 1.031 |
| `tesserav1.1_global` | 33×34 (571), 34×33 (517), 33×33 (359) | 0.0% | 1.031 | 1.031 |
| `tesserav1.1` | 35×33 (1104), 36×33 (627), 36×32 (129) | 0.0% | **1.094** | 1.031 |
| `tesserav2` | 33×34 (496), 34×33 (483), 33×33 (386) | 0.0% | 1.031 | 1.031 |
| `seamless` | 12×12 (1027), 11×12 (327), 12×11 (324) | 0.0% | **0.375** | **0.375** |
| `sentinel1` | 32×32 (2000) | **100.0%** | 1.000 | 1.000 |
| `sentinel2` | 32×32 (2000) | **100.0%** | 1.000 | 1.000 |

Factor > 1 is a downsample, < 1 an upsample. Three things follow:

1. **The 10 m embeddings are mildly downsampled, ~1.03× per dimension.** A 320 m
   patch reprojected into the tile's local UTM circumscribes a 33×33-ish window
   rather than landing on exactly 32×32, so `F.interpolate` fires on every
   patch. It is small, but it is not nothing and it is universal.
2. **`seamless` is upsampled 2.67× per dimension** — 12×12 real 30 m pixels
   inflated to 32×32, ~7× more output pixels than input. Every ESD result in
   the chapter is computed on interpolated data, and a `--patch-size 12` arm is
   the honest comparison. This belongs in the methods section.
3. **`sentinel1` / `sentinel2` are 32×32 natively and never resized**, so the
   Task 3.4 raw baseline is the one modality that reaches the model untouched.

`tesserav1.1`'s 1.094 H-factor differs from every other family's 1.031 — the
per-city extraction crops a taller window than the global one. First hint that
the two are not the same product (Task 1.5.1).

#### Sentinel variance decomposition — five variants

| family | invalid px | native unmasked | native masked | resized unmasked | resized masked | resized masked+filled |
|---|---|---|---|---|---|---|
| `alpha_earth_coop` | 0.4817% | 0.1266 | 0.1051 | 0.1259 | 0.1047 | 0.1047 |
| `tesserav1.1_global` | 0.1543% | 1.1674 | 1.1675 | 1.1349 | 1.1356 | 1.1356 |
| `tesserav1.1` | 0.1341% | 0.8132 | 0.8120 | 0.7965 | 0.7956 | 0.7956 |
| `tesserav2` | 0.1050% | 0.3890 | 0.3891 | 0.3806 | 0.3808 | 0.3808 |
| `seamless` | 0.0000% | 0.5682 | 0.5682 | 0.5524 | 0.5524 | — |
| `sentinel1` | 0.0000% | 0.3645 | 0.3645 | 0.3645 | 0.3645 | — |
| `sentinel2` | 0.0000% | 0.0817 | 0.0817 | 0.0817 | 0.0817 | — |

Sentinel share of per-channel variance:

| family | native | post-resize | change |
|---|---|---|---|
| `alpha_earth_coop` | **31.04%** | **30.86%** | **−0.18 pp** |
| `tesserav1.1` | 0.30% | 0.24% | −0.06 pp |
| all others | 0.00% | 0.00% | 0.00 pp |

**The resize does not dilute the sentinel.** 31.04% → 30.86% across a 1.031×
resample. The GATE 1 explanation ("bilinear resize dilutes an isolated sentinel
pixel") required spreading one input pixel over ~30 output pixels and is wrong;
the entry in "Corrected channel statistics" above has been struck.

The `resized masked` and `resized masked+filled` columns agree to four decimals
for every family, i.e. the Phase 1 mean-fill is numerically inert for the
statistics — which is what it should be, since it exists to stop the sentinel
bleeding into neighbours through the interpolation, not to move the moments.

#### Paired vs unpaired — the actual explanation

`paired` = the patch_id exists in every one of `alpha_earth_coop`,
`tesserav1.1_global`, `seamless`, `sentinel1`, `sentinel2` (intersection
342,297), i.e. it was eligible for the GATE 0 sample.

| family | paired n | paired invalid px | unpaired n | unpaired invalid px |
|---|---|---|---|---|
| `alpha_earth_coop` | 1941 | **0.0184%** | 59 | **15.7292%** |
| `tesserav1.1_global` | 1998 | 0.1545% | 2 | 0.0000% |
| `tesserav1.1` | 2000 | 0.1341% | 0 | — |
| `tesserav2` | 1976 | 0.1062% | 24 | 0.0000% |
| `seamless` / `sentinel1` / `sentinel2` | ~1940 | 0.0000% | ~58 | 0.0000% |

**AlphaEarth's nodata lives almost entirely in the ~3% of patches no other
family covers** — 855× denser outside the paired intersection than inside it.
That is the reconciliation: GATE 0 sampled paired and therefore measured an
essentially sentinel-free AlphaEarth (median std 0.1054), while the A3/A5
reconnaissance sampled coop's own index and hit the sentinel (0.1244–0.1266).
Both numbers are correct; they describe different populations. Nothing about the
GATE 0 cross-family comparison is invalidated — if anything it was cleaner than
we knew, because the pairing acted as an accidental nodata filter.

The practical consequence is the opposite of the one recorded at GATE 1: masking
matters **more**, not less, than the corrected-stats table suggested, because
training on the global split uses all 352,366 coop patches including those 3%.

#### L2 norm, native vs post-resize (valid pixels)

| family | native p1 / p50 / p99 | resized p1 / p50 / p99 |
|---|---|---|
| `alpha_earth_coop` | 0.995 / 1.000 / 1.005 | 0.987 / 0.998 / 1.003 |
| `tesserav1.1_global` | 7.943 / 15.594 / 36.132 | 7.929 / 15.420 / 35.606 |
| `tesserav1.1` | 6.588 / 12.989 / 22.722 | 6.554 / 12.746 / 22.558 |
| `tesserav2` | 11.295 / 11.314 / 11.333 | 11.154 / 11.279 / 11.319 |
| `seamless` | 4.077 / 5.214 / 6.828 | 3.772 / 5.000 / 6.800 |
| `sentinel1` | 0.048 / 0.333 / 3.283 | 0.049 / 0.331 / 3.113 |
| `sentinel2` | 0.099 / 0.466 / 1.150 | 0.099 / 0.467 / 1.143 |

Norm shrinkage is real but small: AlphaEarth's unit-norm vectors lose ~0.2% at
p50 and ~0.8% at p1, exactly as expected when bilinear interpolation mixes two
unit vectors that are not parallel. Worth one sentence in the methods; not worth
correcting. `tesserav2` is the outlier in a different way — its L2 norm is
**effectively constant** (11.295 / 11.314 / 11.333, a 0.3% spread), so v2
embeddings are norm-normalised in a way v1.1's are not (7.9 → 36.1).

#### Mask ordering — already correct, now pinned by tests

Phase 1's implementation does both things the plan asked about:
`PatchDataset._load_source` builds the mask on the **native** grid and mean-fills
invalid pixels *before* the resize, and `_resize_valid` propagates conservatively
(bilinear, then threshold at `>= 1 - 1e-6`, so any output pixel whose
interpolation touched an invalid input is invalid). Two tests added to
`tests/test_nodata_masking.py` pin this at the real 33×33 → 32×32 factor and
assert the fill equals the channel mean rather than zero. No fix required, and
the "numerically inert" verdict on Tessera's nodata continues to hold.

### Task 1.5.1 — Tessera product identity

```bash
python src/diagnostics/tessera_product_check.py --n-sample 2000 --seed 0
```

2000 shared patch_ids per pair, compared on the **native** grid with no resize
(a resize would blur exactly the disagreement being measured). `tesserav1.1`'s
27,229 ids are a strict subset of both `tesserav1.1_global` and `tesserav2`.

| pair | ids in common | compared | shape mismatches |
|---|---|---|---|
| `tesserav1.1` vs `tesserav1.1_global` | 27,229 | 2,000 | 0 |
| `tesserav1.1_global` vs `tesserav2` | 131,086 | 1,956 | 44 |
| `tesserav1.1` vs `tesserav2` | 27,229 | 1,993 | 7 |

| pair | matched-channel r | best cross-channel r | std ratio (med / CV) | linear R² (held out / null) | L2-norm r | verdict |
|---|---|---|---|---|---|---|
| `tesserav1.1` vs `tesserav1.1_global` | **−0.0049** | 0.6503 | 1.259 / **0.397** | **0.8924** / −0.0163 | 0.8315 | distinct products |
| `tesserav1.1_global` vs `tesserav2` | −0.0064 | 0.4824 | 0.343 / 0.648 | 0.8872 / −0.0271 | 0.0759 | distinct products |
| `tesserav1.1` vs `tesserav2` | −0.0164 | 0.6615 | 0.469 / 0.749 | 0.9081 / −0.0148 | 0.1224 | distinct products |

The linear map is fitted on half the patches and scored on the other half —
**held out by patch, not by pixel**, since neighbouring pixels are strongly
autocorrelated and a pixel-level split would score the map on near-copies of its
own training data. The null rolls one side by half the sample so every pixel is
paired with a pixel from a different patch; it lands at −0.02, so the 0.89
held-out R² is real and not 128 free parameters fitting marginals.

**Verdict: outcome 3 — distinct products carrying the same information in
different feature bases.** Not the scale bug PLAN-V2 flagged for an immediate
stop, and not a sampling artifact either:

- Native shapes match **exactly** on all 2000 pairs, so the crop geometry is
  identical and the two are spatially aligned (per-pixel L2-norm r = 0.83).
- Channel *k* of one is not channel *k* of the other (matched r ≈ 0.00) but each
  channel has a strong best match somewhere in the other's 128 (median 0.65).
- The std ratio has **CV 0.40** — a scale bug would be one number, not a
  distribution spanning 0.54–4.54.
- A linear map recovers 89% of one from the other on held-out patches.

Two separately produced Tessera archives, then: the per-city `geotessera`
download and the internal `/tessera/v1.1/global_0.1_degree_representation`
re-run. Same architecture and version label, different inference pass, and an
embedding basis is arbitrary up to rotation across runs.

**Consequences.**

1. A model trained on one and applied to the other produces garbage, and nothing
   currently prevents that — both declare `in_channels: 128`, so
   `build_model` accepts either without complaint. Task 1.5.3 guard 2 closes it.
2. A normalizer fitted on one is invalid on the other. Task 1.5.3 guard 1
   closes it.
3. The GATE 0 "0.80 vs 1.14 median std" line was never a like-for-like scale
   comparison and should not be read as one. Corrected: on the *same* 2000
   patches the medians are 0.813 and 1.024, and the difference is basis, not
   scale.
4. No existing number is wrong — a run takes one `--output-name` /
   `--embedding-name` pair, so no run could have mixed the two. They are
   *unlabelled*, which Task 1.5.4 fixes.
5. Downstream accuracy should be close between them, since R² = 0.89 means
   little information is lost either way. That is a Phase 2 prediction (Task
   2.5), not a claim.

`tesserav2` is a third distinct basis, and differs in kind as well: its per-pixel
L2 norm is **effectively constant** (11.295 / 11.314 / 11.333 at p1/p50/p99)
where v1.1's spans 7.9–36.1. v2 embeddings are norm-normalised; v1.1's are not.
That is why the L2-norm correlation against either v1.1 product is ≈ 0.1 — there
is no norm variation left to correlate.

**Canonical nomination: `tesserav1.1_global`.** It covers 342,944 of the 352,366
cultural-split training patches (97.3%) against `tesserav1.1`'s 27,229 (7.7%,
51 cities only), so it is the only Tessera entry that can carry a global-split
claim at all. Confirmation that it is the entry behind the existing 0.65/0.62
headline is Task 1.5.4's job below; Task 2.0 owns the formal sign-off.
