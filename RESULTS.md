# RESULTS

Running log of every result produced by `PLAN.md` (Phases 0–1), `PLAN-V2.md`
(Phase 1.5), `PLAN-V3.md` (Phase 1.75 onward) and `PLAN-v3-phase2-revA.md`,
which replaces `PLAN-V3.md`'s Phase 2 section in full. Diagnostics get a section
per task; training runs get one table row each (phase, run name, W&B id,
embedding, split, model, seed, OA, macro-F1, kappa, and the one factor that
changed).

From `PLAN-V3.md` onward the scope is three families only —
`tesserav1.1_global`, `alpha_earth_coop`, `seamless`. Rows above that point may
name others; see Task 1.75.0.

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

### Task 1.5.2 / 1.5.3 — Provenance schema and the three guards

Every registry entry now carries `product` / `version` / `source` / `status`.
No key was renamed: keys appear in extraction paths on disk and in the W&B
config of every historical run, so a rename orphans both.

| key | product | version | source | status |
|---|---|---|---|---|
| `alpha_earth_coop` | alphaearth | coop | source_coop | canonical |
| `alpha_earth` | alphaearth | v1 | gee_zarr | untested |
| `tesserav1.1_global` | tessera | v1.1 | global_0.1deg | **canonical** |
| `tesserav1.1` | tessera | v1.1 | percity_geotessera | supported |
| `tesserav2` | tessera | v2 | global_0.1deg | supported |
| `tessera` | tessera | v1 | gee_zarr | untested |
| `seamless` | esd | — | local_tif | canonical |
| `osm_evidence` | osm | — | local_tif | untested |
| `aux_struct` | aux | — | local_tif | supported |

`untested` means *not verified against the current code*, not never used: the
`open_tile` ordering bug fixed in Task 1.0 was introduced after those runs, and
nothing has been re-run through the fixed path. `is_comparable(a, b)` compares
product, version and source but deliberately **not** status — status records how
much an entry is trusted, not what the data is, so promoting one to `canonical`
must not invalidate existing checkpoints.

The three guards, each with tests in `tests/test_provenance.py` (37 tests):

1. **Normalizer cache key.** `stats_cache_path` now takes `embedding_name` and
   puts the provenance triple in both the filename stem and the digest. Two
   extractions of the same product from different tile sources can no longer
   share a stats file — which matters because `--output-name` is free text and
   nothing stopped both Tessera archives being given the same label. Existing
   caches are invalidated by the new filenames, so the first run after this
   recomputes.
2. **Checkpoint binding.** `embedding_name`, `product`, `version`, `source` and
   `year` are written into every checkpoint by both pipelines, alongside the
   Task 1.2 channel statistics. `check_checkpoint_provenance` **raises** on a
   mismatch and names both sides; **absence only warns**, since every
   pre-1.5.3 checkpoint carries nothing — including the one the GATE 1
   bit-identity regression replays.
3. **W&B config.** The same five fields go into `run_cfg` for both pipelines, so
   this table can be rebuilt from W&B alone.

Verified end to end:

| check | result |
|---|---|
| `pytest` | 65 passed in `tests/`, 213 + 37 across the full suite |
| classification checkpoint | carries all five fields, still loads under `weights_only=True` |
| segmentation checkpoint | carries all five fields |
| wrong `--embedding-name` on a `tesserav1.1_global` checkpoint | **refused**, exit 1, message naming both products |
| GATE 1 regression (pre-Phase-1 ckpt + `--normalize none`) | md5 `7c2222b04ad830fe0c08eb4ee686df51` — **unchanged**, guard warns only |

### Task 1.5.4 — W&B provenance audit

```bash
python src/diagnostics/wandb_provenance_audit.py
```

All **225** runs in `phd-thesis-team/lcz-classification-dl` resolved: 35
`explicit` (config carries `embedding_name`) and 190 `inferred` from the
`--output-name` label via a mapping of the labels this project has actually
used. Nothing was left unresolved and nothing was guessed — six 2026-05-05/06
runs used the spelling `GeoTesserav1.1`, for which no directory survives on
disk; they resolve to the per-city archive on dating rather than on the label,
since they are per-city grid-split runs and the global extraction did not exist
yet.

| embedding | product | version | source | runs |
|---|---|---|---|---|
| `alpha_earth_coop` | alphaearth | coop | source_coop | 60 |
| `seamless` | esd | none | local_tif | 54 |
| `tessera` | tessera | v1 | gee_zarr | 42 |
| `tesserav1.1` | tessera | v1.1 | percity_geotessera | 37 |
| `tesserav1.1_global` | tessera | v1.1 | global_0.1deg | 30 |
| `tesserav1.1_global+alpha_earth_coop` | fused | — | — | 1 |
| `tesserav1.1_global+aux_struct` | fused | — | — | 1 |

**Every cultural-split Tessera run used `global_0.1deg`.** The headline numbers
resolve unambiguously:

| run | id | source | test_kappa | test_f1 |
|---|---|---|---|---|
| `student-noisy-v3` | ey10pcob | `global_0.1deg` | **0.6497** | 0.5656 |
| `student-noisy-v1` | k4ka3wjs | `global_0.1deg` | 0.6419 | 0.5780 |
| `student-noisy-v2` | 2pz0icbd | `global_0.1deg` | 0.6320 | 0.5766 |
| `opt3-tessera-seed1` | h6tv58sp | `global_0.1deg` | 0.6268 | 0.5586 |
| `opt3-lr5e-4-warmup3` | lhmxayfl | `global_0.1deg` | **0.6190** | 0.5652 |

So the 0.6497 single-model and 0.619 opt3 headlines both belong to
`tesserav1.1_global`, which confirms the canonical nomination on evidence rather
than on coverage alone. Task 2.0 still owns the formal sign-off.

The 37 `percity_geotessera` runs are **all** per-city grid-split (best 0.9742,
autocorrelation-inflated and not comparable to anything above). The per-city
archive has therefore never produced a cultural-split number, which is worth
knowing before Task 2.5 tries to compare the two products.

**No run could have mixed the two products.** A run takes one `--output-name` /
`--embedding-name` pair and `infer_roi` reads its embedding from the CLI, so
there is no path by which one run trained on one archive and evaluated on the
other. The audit's job was labelling, not damage assessment, and nothing needs
re-running.

---

## Phase 1.75 — Population characterization

Branch `fix/p175-population`, cut from `fix/p15-provenance`. Plan of record:
`PLAN-V3.md` §Phase 1.75.

Housekeeping first: `PLAN-V2.md` and `PLAN-V3.md` are now tracked (`27b8411`),
and the uncommitted exact-footprint change in `datasets/tiles.py` is committed
(`a958a3f`) rather than reverted — see Task 1.75.1's truncation numbers for why.

### Task 1.75.0 — Scope restriction in the registry

`tessera`, `alpha_earth` and `tesserav1.1` are now `deprecated`; `tesserav2` is
`pending` (extraction in progress, version comparison deferred). A new
`available_embeddings()` excludes both statuses and feeds `choices=` on
`patch_classification.py`, `semantic_segmentation.py`, `infer_roi.py` and
`knn_baseline.py`, so an out-of-scope training run is an argparse error rather
than a number:

```
$ python src/patch_classification.py --embedding-name tesserav1.1 ...
error: argument --embedding-name: invalid choice: 'tesserav1.1'
(choose from 'alpha_earth_coop', 'aux_struct', 'osm_evidence', 'seamless',
 'tesserav1.1_global')
```

Extraction, coverage-check and diagnostics scripts keep the full registry — the
`tesserav2` extraction has to finish and the deferred comparison has to read
every archive. No key was renamed and no entry removed, so every historical W&B
run and every existing checkpoint still resolves; `is_comparable` and
`check_checkpoint_provenance` continue to ignore `status` by design.

### Task 1.75.1 — Nodata population on the full cultural split

```bash
python src/diagnostics/nodata_population.py --workers 8
```

**1,191,379 patch reads** — every patch of every split for all three families,
no pairing and no sampling. Artefacts: `diagnostics/nodata_population.json`,
`diagnostics/nodata_population.md`, `diagnostics/invalid_fraction.parquet`.

One pass, not two: the post-resize masked statistics are provably independent of
the fill value, because `_resize_valid` marks any output pixel whose
interpolation touched an invalid input as invalid, so no filled value ever
enters a masked sum. Verified before relying on it — filling with `0.0` and with
an absurd `3.7` gives **bit-identical** masked means and stds.

#### Coverage and invalid data per family and split

| family | split | on disk | missing | invalid px | patches w/ any | p99 | max | truncated |
|---|---|---|---|---|---|---|---|---|
| `alpha_earth_coop` | training | 352,366 | 0 (0.00%) | 0.4006% | 0.46% | 0.0000% | 100.00% | 22 |
| `alpha_earth_coop` | validation | 24,119 | 0 (0.00%) | 0.4433% | 0.46% | 0.0000% | 100.00% | 0 |
| `alpha_earth_coop` | testing | 24,188 | 0 (0.00%) | **0.0000%** | 0.00% | 0.0000% | 0.00% | 0 |
| `tesserav1.1_global` | training | 342,944 | **9,422 (2.67%)** | 0.1399% | 4.38% | 3.0303% | 77.82% | 560 |
| `tesserav1.1_global` | validation | 23,878 | 241 (1.00%) | 0.1437% | 4.16% | 3.1142% | 8.22% | 0 |
| `tesserav1.1_global` | testing | 23,858 | 330 (1.36%) | 0.1362% | 3.86% | 3.6332% | 5.80% | 17 |
| `seamless` | training | 351,719 | 647 (0.18%) | 0.0000% | 0.00% | 0.0000% | 0.00% | 77 |
| `seamless` | validation | 24,119 | 0 (0.00%) | 0.0000% | 0.00% | 0.0000% | 0.00% | 0 |
| `seamless` | testing | 24,188 | 0 (0.00%) | 0.0000% | 0.00% | 0.0000% | 0.00% | 0 |

**The three families have three different, non-overlapping failure modes**, and
the pairing hid all of them:

* `alpha_earth_coop` — complete on disk, but **bimodal** nodata. 99.54% of
  training patches are pixel-perfect; the invalid data lives in 1,617 patches,
  **1,277 of which are 75–100% invalid**. Spot-checked against the pipeline:
  patches `000002`, `000012`, `000222` are 100% sentinel — labelled samples
  carrying no data at all.
* `tesserav1.1_global` — the least nodata but the **most missing data**: 9,422
  training patches (2.67%) have no npy at all, and its nodata is diffuse
  (4.38% of patches, mostly 1–5%) rather than catastrophic.
* `seamless` — zero invalid pixels anywhere, 647 missing training patches.

The parquet's per-patch fractions match `PatchDataset`'s own predicate exactly
(`|Δ| < 1e-12` on a clean/mid/empty spread).

#### Per-patch invalid-fraction histogram (training)

| bucket | `alpha_earth_coop` | `tesserav1.1_global` | `seamless` |
|---|---|---|---|
| exactly 0 | 350,749 | 327,906 | 351,719 |
| 0–0.1% | 3 | 24 | 0 |
| 0.1–1% | 12 | 591 | 0 |
| 1–5% | 27 | 12,881 | 0 |
| 5–10% | 28 | 1,513 | 0 |
| 10–25% | 63 | 13 | 0 |
| 25–50% | 106 | 11 | 0 |
| 50–75% | 101 | 4 | 0 |
| 75–100% | **1,277** | 1 | 0 |

#### The coverage-bias answer: yes, and it is city-level

AlphaEarth nodata is not spread across the split — it is four coastal cities and
one class. Every city not listed is exactly 0.0000%:

| city | held out | coop invalid | coop >25% | tessera missing | tessera truncated |
|---|---|---|---|---|---|
| Cape Town | | **8.1781%** | 896 | 983 | 32 |
| Lisbon | | 3.5652% | 45 | 71 | 0 |
| **Mumbai** | **yes** | 2.2049% | 108 | 242 | 1 |
| New York | | 1.5748% | 332 | 2,026 | 0 |
| Amsterdam | | 0.2684% | 10 | 107 | 3 |
| Hong Kong | | 0.2037% | 15 | 7 | 22 |
| **Guangzhou** | **yes** | 0.1860% | 39 | 26 | 99 |
| Qingdao | | 0.1429% | 14 | 542 | 17 |
| London | | 0.1158% | 60 | 111 | 120 |
| Istanbul | | 0.0000% | 0 | **2,617** | 29 |
| Vancouver | | 0.0000% | 0 | 1,094 | 36 |
| Melbourne | | 0.0084% | 4 | 914 | 53 |
| **Sydney** | **yes** | 0.0000% | 0 | 329 | 16 |

By class it is almost entirely **LCZ 17 (water) at 2.6772%** — 14× the next
highest (LCZ 12 at 0.1866%) and 0.0000% for eight of the seventeen classes. The
mechanism is coherent: coastal cities, water patches, no AlphaEarth coverage
over open sea.

**No city exceeds 25% mean invalid** (the worst, Cape Town, is 8.18%), and two
held-out cities are affected — Mumbai (2.20%) and Guangzhou (0.19%).

**But the split asymmetry matters more than the city concentration.** AlphaEarth
invalid pixels are 0.4433% of validation and **0.0000% of testing** — not one
invalid pixel in 24,188 test patches. Mumbai's coastal water fell entirely on
the validation side of the within-city val/test halving. So the drop policy
changes what the model trains on and what it is selected on, but cannot change
the headline test metric by removing test patches.

#### Recommended `--max-invalid-frac`: 0.25

The distribution is bimodal, so the choice is insensitive — anything from 0.10
to 0.50 drops within 200 patches of the same set (training split):

| threshold | coop dropped | tessera dropped | seamless dropped |
|---|---|---|---|
| >5% | 1,575 | 1,542 | 0 |
| >10% | 1,547 | 29 | 0 |
| >25% | **1,484** | 16 | 0 |
| >50% | 1,378 | 5 | 0 |

End to end on the real 400,673-item global list, AlphaEarth:

| threshold | total dropped | train | val | test |
|---|---|---|---|---|
| 0.50 | 1,486 | 1,378 | 108 | **0** |
| **0.25** | **1,592** | 1,484 | 108 | **0** |
| 0.10 | 1,656 | 1,547 | 109 | **0** |

The test split loses nothing at any threshold, which follows from its 0.0000%
invalid rate — the drop policy cannot flatter a test metric by removing hard
patches from it.

`0.25` is the recommendation: it drops 1,484 training patches (0.42%) that are
more fill than data, keeps every patch with a usable majority, and costs Tessera
only 16 and `seamless` nothing. `0.05` is rejected — it would drop 1,542 Tessera
patches whose nodata is a diffuse 1–5%, which is a different phenomenon.

Implemented as `--max-invalid-frac` (default `1.0`) on
`patch_classification.py`, applied to training **and** evaluation in
`build_so2sat_items`, reading the parquet keyed on `(dataset, patch_id)` — a
missing parquet is an error naming the command that writes it, never a silent
pass. Recorded in `run_cfg` and in the checkpoint beside the provenance fields.
Task 2.2 runs the AlphaEarth arm both ways.

#### Corrected channel statistics — full unfiltered training population

Supersedes both the Phase 0 (paired, n=5000) and GATE 1 (n=20000) tables.

| family | native unmasked | native masked | resized unmasked | resized masked | sentinel share of variance | `0.05 / median_std` |
|---|---|---|---|---|---|---|
| `alpha_earth_coop` | 0.1221 | 0.1050 | 0.1214 | **0.1048** | **26.07%** | **0.477** |
| `tesserav1.1_global` | 1.1592 | 1.1597 | 1.1339 | **1.1344** | 0.00% | **0.044** |
| `seamless` | 0.5610 | 0.5610 | 0.5446 | **0.5446** | 0.00% | **0.092** |

The sentinel contributes **26.07%** of AlphaEarth's per-channel variance on the
real population — close to GATE 1.5's paired-corrected 31.04%, and confirming
that the paired sample understated it by roughly 1400×.

**This supersedes `PLAN-V3.md`'s pre-registered revision of the effective noise
ratio to ~0.57.** That figure applied a √0.69 signal-only correction to `0.1054`,
but `0.1054` was already effectively a masked number — the paired sample it came
from was almost sentinel-free — so the discount was applied twice. Measured
directly against the masked std the model actually sees, the ratio is **0.477**,
which is where Phase 0 had it. The cross-family spread is therefore
**10.8×** (0.477 / 0.044), not 14×.

### Task 1.75.2 — Crop geometry and the latitude question

```bash
python src/diagnostics/crop_geometry.py
```

Full training population (shapes from npy headers only, no pixel data read).
Artefacts: `diagnostics/crop_geometry.{json,md,png}`.

#### The degree-grid hypothesis is falsified

| family | CRS of an actual tile |
|---|---|
| `tesserav1.1_global` | `EPSG:32630`, 10 m (`grid_-0.05_10.05.tiff`) |
| `alpha_earth_coop` | `EPSG:32610`, 10 m (`x0dyfir8mjpv8ty4m-…tiff`) |

Both are per-zone **projected UTM at exactly 10 m**, not EPSG:4326, so longitude
pixel size does not vary as cos φ. The variation is reprojection distortion at
clip time: `crop_patch` reprojects each lon/lat patch rectangle into the tile's
UTM zone and clips to the **bounding box** of the resulting trapezoid.

#### Correlations — strong per axis, weak in aggregate

| family | h~lat | w~lat | h~\|lat\| | w~\|lat\| | h~merid. dist | w~merid. dist | **res_geo~\|lat\|** |
|---|---|---|---|---|---|---|---|
| `tesserav1.1_global` | +0.455 | −0.475 | +0.325 | −0.266 | +0.344 | +0.262 | **+0.199** |
| `alpha_earth_coop` | +0.505 | −0.519 | +0.397 | −0.303 | +0.303 | +0.193 | **+0.228** |

Height and width move in **opposite directions** with latitude, so neither axis
alone answers the question. The scale-invariant summary is
`res_geo = √(h·w)·10/32`, the ground sampling distance per output pixel:

| \|lat\| band | tessera h | tessera w | **tessera res_geo** | coop h | coop w | **coop res_geo** |
|---|---|---|---|---|---|---|
| 0–15° | 33.33 | 33.15 | **10.387** | 33.00 | 33.00 | **10.312** |
| 15–30° | 33.45 | 33.43 | **10.449** | 33.23 | 33.53 | **10.430** |
| 30–40° | 33.39 | 33.72 | **10.485** | 33.36 | 33.75 | **10.484** |
| 40–50° | 33.25 | 33.43 | **10.419** | 33.29 | 33.45 | **10.427** |
| >50° | 34.83 | 33.05 | **10.603** | 34.69 | 33.01 | **10.573** |

| family | mean res_geo | city-mean range res_geo |
|---|---|---|
| `tesserav1.1_global` | 10.482 m/px | 10.312–10.663 (**+3.4%**) |
| `alpha_earth_coop` | 10.474 m/px | 10.312–10.633 (**+3.1%**) |

**Verdict: a real but bounded effect.** Ground sampling distance rises
monotonically-ish with latitude but only by **~2%** between the equatorial and
>50° bands, and the full spread across all 51 cities is **3.1–3.4%**. Per-axis
extent varies more (up to 7.9% on h), but the two axes compensate.

It is a **city-level constant**, not within-city noise: per-city standard
deviations are 0.00–0.70 px against city means separated by up to 1.7 px. So it
is exactly the shape of covariate that could confound a cross-city domain-gap
analysis — which is why Phase 4 should carry it — but at 3% it is far too small
to be a meaningful component of the cross-city accuracy gap. A null worth
having, not a finding that changes Phase 2.

#### Truncation is the geometry problem that does matter

| family | n | h range | w range | truncated (<0.85 × family median) |
|---|---|---|---|---|
| `tesserav1.1_global` | 339,285 | **2–36** | **2–36** | 484 (0.143%) |
| `alpha_earth_coop` | 348,217 | 32–36 | 32–35 | 0 (0.000%) |

Tessera crops go down to **2 px on a side**, stretched to 32×32 by the resize.
These are the patches the bounding-box coverage test wrongly accepted; `a958a3f`
fixes the test, but the extracted data predates it. 560 in training, 17 in
testing, concentrated in London (120), Guangzhou (99), Melbourne (53) and
Shanghai (50). Re-extraction is a separate decision, not taken here.

### Verification

| check | result |
|---|---|
| `pytest` | **285 passed, 1 skipped** (was 256/1 at GATE 1.5) |
| deprecated key on a training CLI | argparse error listing only the 5 available keys |
| deprecated key on the extraction CLI | still accepted — all 9 keys offered |
| classification smoke (nano, 1 epoch, Nairobi) | exit 0, OA 0.8802 / κ 0.8411 |
| segmentation smoke (nano, 1 epoch, Nairobi) | exit 0, through ROI inference and raster output |
| checkpoint contents | `embedding_name`/`product`/`version`/`source`/`status`/`year`/**`max_invalid_frac`**, loads under `weights_only=True` |
| provenance guard, wrong embedding | **refused**, exit 1, message naming both products |
| `--max-invalid-frac 1.0` | returns the item list **object itself** — provable no-op |
| filter order preservation | exact, on the real 400,673-item list |
| parquet vs `PatchDataset` predicate | identical, \|Δ\| < 1e-12 |
| full scan reproducibility | second run reproduces every figure exactly |
| **GATE 1 regression** | md5 `7c2222b04ad830fe0c08eb4ee686df51` — **unchanged** |

### Behaviour changes to be aware of

- **Four embeddings are no longer selectable** on `patch_classification.py`,
  `semantic_segmentation.py`, `infer_roi.py` and `knn_baseline.py`. Extraction
  and diagnostics scripts are unaffected.
- **`datasets.tiles` indexes exact footprints**, so `_fully_covered` is now
  strict. A re-extraction would drop patches the previous one accepted; nothing
  already on disk changed.
- `--max-invalid-frac` defaults to `1.0` and is inert at that value, but it is
  now recorded in every checkpoint and W&B config, so filtered and unfiltered
  runs are distinguishable after the fact.

### Open items for Phase 2 to decide

1. ~~**Whether Task 2.2 restricts to the common patch-id intersection.**~~
   **Settled by Rev A: both.** Arm A runs the manifest intersection (headline,
   the only arm cross-family claims may be drawn from), Arm B each family's
   native coverage, and the A−B difference is itself a deliverable. See Task 2.0.
2. ~~**Whether to re-extract Tessera.**~~ **Deferred to GATE 3 by Rev A, and
   Task 2.0d now measures what it would buy: 223 of 9,422 absent training
   patches, and zero of the 571 absent validation and test patches.** See Task
   2.0d.
3. **`datasets.so2sat.assign_cities` resolves the Guangzhou/Hong Kong box
   overlap by first match**, where this audit used smallest-containing-box. The
   527 affected patches are Hong Kong patches inside the Guangzhou box. Matters
   for Phase 4's per-city normalization, not for anything before it.

### GATE 1.75 status

Reached. Phase 2 not started.

---

## Phase 2 — Revalidation (`PLAN-v3-phase2-revA.md`, amended by `PLAN-v3-phase2-revB.md` and `-revC.md`)

Branch `exp/p2-revalidation`, cut from `master` after Phases 1, 1.5 and 1.75 were
merged (`30092a6`). No changes to model or training logic in this phase.

Rev B amends Rev A following GATE 2.0; Rev A stands except where superseded. Its
amendments and where they land:

| | amendment | lands in |
|---|---|---|
| B1 | Tessera's water gap is a chapter-level finding (archive-documentation check, coverage figure, deployment consequence) | write-up |
| B2 | Task 2.2 becomes a 2×2 (train arm × eval set), replacing Rev A's A−B difference; **plus one pre-registered prediction, below** | Task 2.2 |
| B3 | Metrics separate the easy classes: macro-F1 first, built-only LCZ 1–10 alongside the 17-class figures, per-class recall with LCZ 17 flagged | Task 2.2 onward |
| B4 | Scope the multi-UTM-zone mosaic defect — measure only, no fix in Phase 2 | Task 2.1 turn |
| B5 | `native_frac` threshold sensitivity: one Arm A config under the tighter per-side rule | end of Phase 2 |
| B6 | Tasks 2.1, 2.3, 2.4, 2.5 confirmed unchanged | — |

Two supersessions worth stating explicitly, because earlier text in this file
says otherwise:

- **B2 replaces the "A−B difference" with the full 2×2.** §2.0c below records a
  dual-eval of Arm A checkpoints only, which is the A row (A/A and A/B). Rev B
  adds the B row: Arm B checkpoints are also evaluated on both test sets. A/A vs
  A/B is the test class-mix effect; A/A vs B/A the training-data effect.
  Cross-family claims use the **common-test column only** — native test differs
  per family, so B/B may never appear in a cross-family table.
- **B3 supersedes OA as the headline.** LCZ 17 alone is 12–14% of the dataset,
  near-ceiling for every family, and the manifest moves that share, so OA is
  dominated by a class nobody competes on. Metric plumbing lands with Task 2.2,
  where B3 first applies; Task 2.1 below already reports macro-F1 first.

### Pre-registration

Committed before any Phase 2 run launched, verbatim from Rev A:

- `tesserav1.1_global` moves less than 1 point either way from the Phase 1 fixes.
- `alpha_earth_coop` improves most, and largely via the noise-ratio change
  (0.477 → 0.05).
- `seamless` improves modestly.
- B − A ≈ 0; D − B carries the effect.
- **Nodata masking: near-null on the test metric by construction.** GATE 1.5's
  "matters more than we thought" is withdrawn — it rested on a population that
  includes no test patches. Any effect arrives through training and selection,
  and should be under a point.
- Resize mode near-neutral for the 10 m families.
- Arm A versus Arm B: within noise for AlphaEarth and seamless; genuinely
  uncertain for Tessera, since its extra native patches are disproportionately
  the hard tile-edge cases.
- The AlphaEarth–Tessera gap closes by 1–6 of its 9 points; ranking flip
  genuinely uncertain.
- If AlphaEarth does not improve at all, the noise was never the binding
  constraint and the gap needs a different explanation — most likely that 64-d
  annual composites carry less LCZ-relevant structure than 128-d time-series
  representations.

#### Addition from Rev B (Amendment B2)

Committed before the Task 2.1 anchor runs launched and before any Task 2.2 run
exists, verbatim from Rev B:

> LCZ 17 recall will be **lower** on the common test set than the native one,
> because the excluded patches are pure ocean while the surviving water patches
> are coastal and mixed. If water recall rises instead, the filter is not doing
> what the audit says and Task 2.2 stops pending an explanation.

This is a prediction about the A/A-versus-A/B contrast, so Task 2.2 tests it;
Task 2.1 runs no manifest and cannot. The stop condition is the point of
recording it: a rise would mean the 2.0c audit mischaracterises the filter, and
every Arm A number rests on that characterisation.

#### Addition from Rev C (Tasks 2.1b-iii and 2.1c)

Committed before any run of either task exists.

**Task 2.1c**, verbatim from Rev C:

> best val_kappa will occur at a later epoch under a lower LR, and the
> seed-to-seed variance in peak epoch will narrow. If a lower LR does not move
> the peak epoch later, the early peak is overfitting driven by capacity rather
> than step size, and the answer is regularization or a smaller model rather
> than a schedule change.

**Task 2.1b-iii.** The four extra seeds test whether peak epoch at `opt3` is
**bimodal**, not merely whether its mean shifted. The anchor's three seeds peak
at epochs **2, 2, 15** — two during warmup at a low LR, one well after it. The
prediction is that seeds 3-6 land in the same two clusters rather than spreading
evenly, and that runs peaking during warmup score lower. If both hold, the
anchor's ~0.8-point shortfall is a sampling artefact of an unstable schedule —
three draws from a bimodal distribution against a historical three — and needs
no code mechanism. Nothing further is then chased.

The falsifier is a unimodal peak-epoch distribution across n=7: that would leave
the shortfall unexplained with all three named mechanisms excluded, and Rev C's
instruction applies — use the new-code distribution as the Phase 2 reference and
carry a ~0.8-point uncertainty on comparisons to published pre-fix numbers.

**One arm of Task 2.1c is dropped, recorded here before results exist.** Rev C
asks for "cosine decay versus the current schedule" at the best LR. The current
schedule **is** cosine: `training/loop.py:62-73` builds `CosineAnnealingLR`
unconditionally and prepends `LinearLR` warmup through `SequentialLR` when
`--warmup-epochs > 0`. There is no flag selecting anything else, so the arm
compares cosine with itself. The LR sweep and the warmup arm already test the
hypothesis behind it — that the peak lands early because the step size is wrong
— so the arm is skipped rather than made real with a new CLI flag, which would
be a production code change in a revalidation phase. 15 + 6 runs, not 24.

### Task 2.0 — Common manifest and coverage control

Full artefacts: `diagnostics/patch_manifest.md` /
`diagnostics/patch_manifest_v1.json`, `diagnostics/reextraction_scope.md`.

**The manifest.** `diagnostics/patch_manifest_v1.parquet`, sha256
`ad7fdae39ae75b77d0b9c477fdf051aaf12ce9894615b05c74b39399f37ca728`. One row per
reference patch (400,673), membership requiring — for **all three** families —
present on disk, `invalid_frac <= 0.25`, `native_frac >= 0.5`. Non-members are
kept in the table with `in_manifest = False`, so the loader can tell "excluded"
from "outside the manifest's universe".

| split | reference | manifest | retained |
|---|---|---|---|
| training | 352,366 | 341,754 | 96.99% |
| validation | 24,119 | 23,878 | 99.00% |
| testing | 24,188 | 23,852 | 98.61% |

#### 2.0a — The native-fraction filter is relative, and it found a third tail

`native_frac = (h·w) / expected_native_area[family]`, the denominator being each
family's own median native crop area on the training split. An absolute pixel
threshold would erase `seamless`, which is 30 m data.

| family | expected native area | median shape | area min | below `native_frac` 0.5 (train) | below 0.85-per-side (train) |
|---|---|---|---|---|---|
| `alpha_earth_coop` | 1122 | 33×33 | 99 | 20 | 22 |
| `tesserav1.1_global` | 1122 | 33×33 | 12 | 314 | 560 |
| `seamless` | 144 | 12×12 | 33 | 30 | 77 |

Two things Phase 1.75 did not have:

- **`seamless` has its own truncated tail** — 77 training patches under the
  per-side rule, 30 under `native_frac`, smallest 3×11. It was never checked
  because `crop_geometry.py` covered only the two 10 m families.
- **`alpha_earth_coop` has 22 truncated training patches**, which
  `crop_geometry.md` reports as 0. That table drops patches with no city
  assignment and all 22 sit outside every city box. `nodata_population.md`'s
  headline table had it right.

The two rules disagree by design: `native_frac >= 0.5` keeps a 33×17 crop that
the per-side rule calls truncated, so the area rule is the more permissive of the
two. Both counts are reported so the gap is visible rather than implicit.

#### 2.0b — What the manifest costs each family

| split | family | on disk | own filtered | manifest | loss vs own |
|---|---|---|---|---|---|
| training | `alpha_earth_coop` | 352,366 | 350,862 | 341,754 | 2.60% |
| training | `tesserav1.1_global` | 342,944 | 342,614 | 341,754 | 0.25% |
| training | `seamless` | 351,719 | 351,689 | 341,754 | 2.82% |
| testing | `alpha_earth_coop` | 24,188 | 24,188 | 23,852 | 1.39% |
| testing | `tesserav1.1_global` | 23,858 | 23,852 | 23,852 | 0.00% |
| testing | `seamless` | 24,188 | 24,188 | 23,852 | 1.39% |

Tessera gives up almost nothing because it is the binding constraint; the other
two pay for matching it.

#### 2.0c — The intersection and the nodata policy are the same filter

**All 1,277** of AlphaEarth's >75%-invalid training patches are *also* absent
from Tessera, against a 2.67% base rate of Tessera absence — and **none** are
absent from `seamless`. The same holds on validation (105 of 105). After a
coverage-only intersection, just **178** high-invalid training patches remain for
`--max-invalid-frac` to remove, and **zero** on validation and testing.

So Arm A's coverage restriction has already applied the drop policy. The two
levers cannot be reasoned about as independent, and Rev A's instruction to report
this rather than discover it later is the reason it is stated here.

**Both failure modes are one phenomenon: water.** AlphaEarth's sentinel sits over
ocean; Tessera has no tile over ocean.

| split | class losing >5% | loss | cities losing >5% |
|---|---|---|---|
| training | LCZ 17 | 18.88% | Istanbul 12.89, Qingdao 12.62, New York 9.95, Cape Town 8.97, Lisbon 5.64, Shanghai 5.19 |
| validation | LCZ 17 | 9.24% | **Mumbai** 9.99 |
| testing | LCZ 17 | 13.23% | **Sydney** 13.85 |

No other class loses more than 2.1%. LCZ 17 is 87.8% of all excluded training
patches and its training share falls 14.01% → 11.72% (−2.29 pp); every other
class gains at most +0.32 pp.

**The entire Tessera evaluation-coverage hole is LCZ 17** — 330 of 330 absent
test patches and 241 of 241 absent validation patches, concentrated in two
held-out cities. This is the finding that most affects Task 2.2: Arm A's headline
test set is common by construction, but it is *not* the same test set the 0.6190
anchor was computed on. Per the decision taken at planning, every Arm A
checkpoint will therefore be evaluated **twice** — on the common manifest test
set (headline, cross-family comparable) and on each family's full native test set
(anchor-comparable) — so the size of that shift is measured rather than assumed.

Attribution of the 10,612 training exclusions: `tesserav1.1_global` absent 9,422
/ failed 330; `alpha_earth_coop` absent 0 / failed 1,504; `seamless` absent 647 /
failed 30.

#### 2.0d — Re-extraction scoping (measure only, nothing re-extracted)

The fixed `_fully_covered` (a958a3f) replayed over the 9,993 absent and 577
truncated patches against 8,108 exact tile footprints.

| split | absent | would recover | correctly rejected | no tile at all |
|---|---|---|---|---|
| training | 9,422 | 191 (2.0%) | 9,199 | 9,144 |
| validation | 241 | 0 (0.0%) | 241 | 241 |
| testing | 330 | 0 (0.0%) | 330 | 319 |

> **Corrected 2026-08-12 while doing B4** (`e750f8e`). This table first reported
> **223 (2.4%)** recoverable training patches and **zero** blocked by the corrupt
> tile. The corrupt-tile test could never fire: `build_tile_index` returns the NPY
> *directory*, Tessera names tiles by fractional coordinates, and `Path.stem`
> reads the trailing `.25` as a suffix — so every name was compared as
> `grid_121.35_31` and never matched `grid_121.35_31.25`. All 8,108 v1.1 global
> tiles were affected. **32 of the 223 depend solely on the corrupt Shanghai tile
> and are not recoverable.** Fixed by `datasets.tiles.tile_index_name`; the
> conclusion below is unchanged, and so are every truncation count and the
> zero-recoverable finding on validation and test.

**The evaluation-coverage hole is permanent.** Sydney's 329 missing test patches
and Mumbai's 241 missing validation patches are ground Tessera does not cover, so
re-extraction cannot restore test-split comparability. The 191 recoverable
training patches are ordinary urban classes in London (111), Qingdao (26),
Guangzhou (22), Hong Kong (7) and Wuhan (6) — Shanghai now contributes none — and
of the 9,690 absent LCZ 17 patches, 9 are recoverable.

Truncation splits in two, and the second half is not what it looks like. 258 of
560 truncated training patches are now correctly rejected (and 17 of 17 on test).
The remaining **302 are still accepted, and the coverage test is right about
them**: 294 (97.4%) straddle a UTM zone boundary and all 302 span multiple tiles.
The ground is covered; the truncation happens downstream in `crop_patch`'s
multi-tile mosaic — the known multi-UTM-zone defect, not anything the footprint
fix touches. Their shapes show it (full extent on one axis, 5–8 px on the other)
and their cities are the seams: London, Guangzhou, Hong Kong (114°E), Qingdao and
Shanghai (120°E). **A re-extraction on today's code would reproduce those 302
rather than remove them.**

### Verification

| check | result |
|---|---|
| `pytest` | **303 passed, 1 skipped** (285 + 18 new manifest tests) |
| `--patch-manifest` unset | returns the item-list **object itself** on the real 400,673-item list |
| manifest filtering | 341,754 / 23,878 / 23,852 kept, matching the parquet exactly; order preserved |
| manifest reproducibility | second run byte-identical, sha256 `ad7fdae3…` unchanged |
| independent recomputation | 389,484 members, **set-identical** to the manifest |
| flag attribution | every non-member has a failing family; every member passes all three |
| smoke (`--preset nano`, Nairobi, manifest) | exit 0; drops 0 patches, as the audit predicts for Nairobi |
| checkpoint | carries `patch_manifest_sha256` beside the five provenance fields and `max_invalid_frac`; loads under `weights_only=True` |
| 2.0d wrote nothing | Tessera extraction dir 342,944 files before and after |
| **GATE 1 regression** | md5 `7c2222b04ad830fe0c08eb4ee686df51` — **unchanged** |

### Behaviour changes to be aware of

- **`--patch-manifest` is new and unset by default.** Unset reproduces current
  behaviour exactly (native per-family coverage); `filter_by_manifest(items,
  None)` returns the input object, which is what makes Task 2.1's anchor and
  every Arm B run provably unfiltered.
- The manifest applies to **train, val and test alike**. Restricting only
  training would report accuracy on patches the model was never allowed to learn
  from.
- Patches outside the manifest's universe (unlabeled and pseudo-label pools) are
  **kept with a warning**, matching `--max-invalid-frac`'s rule, so the filter
  never becomes a silent second coverage restriction.
- `run_cfg` and the checkpoint gain `patch_manifest` and
  `patch_manifest_sha256`, so Arm A and Arm B are distinguishable in W&B and a
  manifest rebuilt with different thresholds under the same filename is caught.

### Open items

1. ~~**The multi-UTM-zone `crop_patch` defect**~~ **Scoped by Amendment B4
   below**: 22.1% of zone-straddling patches come back truncated, all of them in
   the training split. Still out of Phase 2 scope to *fix*; still a GATE 3 input.
   What remains unmeasured is what it costs the other two families.
2. **Whether Arm B should use `--max-invalid-frac 0.25 --min-native-frac 0.5`
   per family as Rev A specifies** is already settled; noted only because 2.0c
   shows the nodata criterion does independent work on just 178 training patches
   once coverage is held common, so Arm A's two criteria are nearly the same
   filter while Arm B's are not.

### GATE 2.0 status

Reached. Tasks 2.1 and 2.1b complete. Task 2.1b-iii is unblocked by Rev C's own
gate but not launched; Tasks 2.1c and 2.2-2.5 not started.

---

### Amendment B4 — The multi-zone mosaic defect, scoped (measure only)

Full artefacts: `diagnostics/mosaic_scope.md` / `diagnostics/mosaic_scope.json`.

Task 2.0d found 294 of the 302 truncated-but-accepted patches at a UTM zone seam.
That is a numerator without a denominator — it says truncated patches sit at
seams, not that sitting at a seam truncates a patch. B4 supplies the denominator
by scanning all 390,680 `tesserav1.1_global` patches on disk instead of only the
failures.

| population | n | truncated | rate |
|---|---|---|---|
| all on disk | 390,680 | 320 | 0.08% |
| single tile | 353,386 | 149 | 0.04% |
| multiple tiles | 37,294 | 171 | 0.46% |
| multiple tiles, **one** zone | 36,534 | 3 | 0.01% |
| **straddling a zone boundary** | **760** | **168** | **22.11%** |

**Of the 760 patches that straddle a UTM zone boundary, 168 come back truncated —
22.1%, against 0.01% for patches that span multiple tiles within a single zone.**

That contrast is the result. `merge_multi_crs` is implicated and `numpy_mosaic`
is essentially clean, but straddling is **not on its own sufficient**: a further
trigger selects which 22% fail, and identifying it is the prerequisite before
Phase 4 can judge its map-production exposure. Read the other way, 52.5% of all
truncated patches straddle a zone and 53.4% span multiple tiles — the rest are
ordinary tile-edge coverage failures, a separate phenomenon.

Two things bound the risk:

- **All 760 straddling patches are in the training split.** Validation and
  testing contain none, so the defect cannot reach any Phase 2 evaluation metric
  by any route.
- The manifest already excludes them via `native_frac >= 0.5`, so Arm A is
  unaffected regardless.

By city, all at the seams a UTM zone predicts: London 271 straddling / 75
truncated (27.7%), Guangzhou 139 / 24 (17.3%), Hong Kong 68 / 16 (23.5%),
Qingdao 49 / 11 (22.4%), Wuhan 20 / 1 (5.0%).

Rev B also asks whether the truncated patches that span tiles *without* crossing
a zone share a property, which would indicate a second mechanism. There are
three, and they are strikingly uniform: all Shanghai, all training, all exactly
two tiles, all cropped to `7x33`. None touch the corrupt tile. Too few to
generalise from, and `numpy_mosaic`'s 0.01% rate argues against a second
systematic mechanism.

Definition note: "spans multiple tiles" is taken from what the extractor does —
`extract_so2sat_embeddings` queries the STRtree with no predicate and passes
every bounding-box candidate to `crop_patch`. The table uses truly-intersecting
tiles, since `crop_patch` routes on how many clips come back non-empty; under the
extractor's looser bbox definition the ratio is 19.7% (168 of 852). Under Phase
1.75's stricter 0.85-per-side truncation rule it is 38.9% (296 of 760).

**Nothing was fixed.** Rev B is explicit that the mosaic is not repaired in
Phase 2; this is a GATE 3 input beside the re-extraction recovery count.

### Task 2.1 — Reproduce the pre-fix baseline

Launcher `run_phase2_anchor.sh`. Three seeds of the opt3 recipe on
`tesserav1.1_global` with the pre-fix flags — `--normalize none --nodata-mode
zero --max-invalid-frac 1.0`, **native per-family coverage, no manifest** —
against which every later Phase 2 number is measured.

Metrics ordered macro-F1 first per Amendment B3.

| seed | epochs | best val_kappa (epoch) | macro-F1 | OA | kappa | macro acc |
|---|---|---|---|---|---|---|
| 0 | 12 | 0.5667 (2) | 0.5641 | 0.6409 | 0.6097 | 0.5822 |
| 1 | 12 | 0.5857 (2) | 0.5460 | 0.6497 | 0.6175 | 0.5690 |
| 2 | 25 | 0.5892 (15) | 0.5651 | 0.6438 | 0.6111 | 0.5904 |
| **mean ± sd** | | | **0.5584 ± 0.0108** | **0.6448 ± 0.0045** | **0.6128 ± 0.0042** | **0.5805 ± 0.0108** |

W&B `p2-anchor-tessera-seed{0,1,2}` (`p3x6xm0v`, `vh8rzsee`, `49tr6tzw`).

#### The comparison is against three historical runs, not one number

The 0.6190 anchor is not alone in W&B: two further pre-fix Tessera runs with the
same flags already existed, so the pre-fix baseline is itself a distribution.

| run (pre-Phase-1) | epochs | best val_kappa (epoch) | macro-F1 | OA | kappa |
|---|---|---|---|---|---|
| `opt3-lr5e-4-warmup3` | 17 | 0.5747 (7) | 0.5652 | 0.6514 | 0.6190 |
| `opt3-tessera-seed1` | 12 | 0.5892 (2) | 0.5586 | 0.6586 | 0.6268 |
| `opt3-tessera-seed2` | 14 | 0.5869 (4) | 0.5634 | 0.6484 | 0.6171 |
| **mean ± sd** | | | **0.5624 ± 0.0034** | **0.6528 ± 0.0052** | **0.6210 ± 0.0051** |

Early stopping between epochs 12 and 25, with val_kappa peaking in the first
handful and training loss still falling, is the **normal shape of this recipe**,
not a failure — the historical runs stopped at 12, 14 and 17. Seed 0's stop at
epoch 12 with its peak at epoch 2 matches `opt3-tessera-seed1` almost exactly.

#### Verdict: reproduced within seed noise, but not by Rev A's literal test

Rev A asks whether the existing 0.6190 "falls inside" mean ± std. **It does
not** — it sits **+1.50 sd** above the Task 2.1 mean, inside a 2 sd interval
([0.6044, 0.6211]) but outside 1 sd ([0.6086, 0.6169]). Stated plainly rather
than widened to 2 sd and declared a pass.

Against the full historical distribution, no metric differs significantly at
n = 3 per side (Welch):

| metric | Task 2.1 | historical | difference | p |
|---|---|---|---|---|
| macro-F1 | 0.5584 ± 0.0108 | 0.5624 ± 0.0034 | −0.0040 | 0.593 |
| OA | 0.6448 ± 0.0045 | 0.6528 ± 0.0052 | −0.0080 | 0.116 |
| kappa | 0.6128 ± 0.0042 | 0.6210 ± 0.0051 | −0.0082 | 0.101 |

So the honest reading is: **a consistent ~0.8-point shortfall on kappa and OA
that n = 3 cannot distinguish from seed noise**, and a macro-F1 gap of 0.4
points that is nowhere near significant. Under B3, macro-F1 is the headline
metric, and it is the one that agrees best.

No mechanism is available to explain a real shift. The flags reproduce the old
behaviour by construction (`--normalize none` skips the stats pass entirely,
`--nodata-mode zero` restores zero-filling, σ 0.05 is absolute under
`--normalize none`); `tests/test_augment_distribution.py` shows the augmentation
is distributionally unchanged; the extracted npys are byte-identical to those the
historical runs read (342,944 files, untouched); and the `patch_id` collision fix
predates the original run (2026-06-10 against 2026-06-16), so both sides are on
the corrected split.

Two caveats belong with these numbers:

- **The historical runs' seeds are not recorded** — `seed` is absent from all
  three configs, so their spread mixes seed variation with ordinary run-to-run
  nondeterminism, and it cannot be assumed they used three distinct seeds. Task
  2.1's three are explicitly 0, 1 and 2.
- **n = 3 per side is weak.** A −0.008 shift at p ≈ 0.1 is exactly the regime
  where more seeds would resolve the question and three cannot.

**The band every later improvement must clear.** Pooling all six runs carrying
pre-fix flags gives **kappa 0.6169 ± 0.0061**, range [0.6097, 0.6268]. That is
the more defensible anchor than either triple alone, and it is the number Phase 2
improvements should be measured against.

#### Verification

| check | result |
|---|---|
| `pytest` | **319 passed, 1 skipped** (303 + 12 augmentation + 4 tile-name) |
| split sizes, all three runs | **342,944 / 23,878 / 23,858** — the native counts, not the manifest's 341,754 / 23,878 / 23,852 |
| W&B config, all three | `patch_manifest: None`, `patch_manifest_sha256: None`, `max_invalid_frac: 1`, `normalize: none`, `nodata_mode: zero`, `noise_sigma: 0.05`, `seed: 0/1/2` |
| checkpoints, all three | carry `normalize`, `max_invalid_frac`, `patch_manifest_sha256` and the five provenance fields; load under `weights_only=True` |
| the `--normalize none` warning | emitted verbatim in every run, as intended for a deliberate ablation |
| chain | three runs, all exit 0 |
| B4 wrote nothing | Tessera extraction dir 342,944 files before and after |
| **GATE 1 regression** | md5 `7c2222b04ad830fe0c08eb4ee686df51` — **unchanged** |
| pre-registration ordering | B2 committed 22:58:22, first run started 22:59:10 |

#### The augmentation is distributionally unchanged

`f45fd80` replaced a per-sample loop with batched tensor ops, so a fixed seed no
longer reproduces the old draw sequence and the anchor can only be held to
**seed-level** agreement with kappa 0.6190. `tests/test_augment_distribution.py`
pins that this is an RNG-stream change and not a semantics change, comparing the
current implementation against the pre-Phase-1 one vendored from `e12c7c2`:

| property | instrument | result |
|---|---|---|
| values, pooled | KS, 768,000 per side | D = 4.60e-04, **p = 1.0000** |
| values, per channel | KS across a 250× scale range | p ≥ 0.982 |
| geometry | frequency of the 8 dihedral transforms | uniform in both; χ² 5.68 and 2.87 on df 7 |
| noise gate | application rate | 0.4997 vs 0.5047 |
| noise magnitude | KS on the perturbations | p = 0.985; σ 0.05002 vs 0.05004 |
| **RNG stream** | equality from an identical seed | **differs — the negative control** |

Geometry needs its own instrument because a pooled KS is blind to it: flips and
rotations permute pixel positions, and the pooled distribution is invariant to
permutation. Every test seeds `torch` explicitly so the p-values are fixed rather
than resampled per run.

---

### Task 2.1b — Two named mechanisms for the shortfall, both excluded

Rev C names two things the GATE 2.1 verification could not have detected and blocks
Task 2.2 on both. Neither needed a training run. **Both come back clean, and the
~0.8-point shortfall therefore stands unexplained.**

#### 2.1b-i — Augmentation independence within a batch

GATE 1 described the Phase 1 change as "one batched draw instead of N scalar draws",
which reads two ways. `torch.rand(n)` keeps per-sample randomness; `torch.rand(1)`
broadcast hands every sample in a batch the **same** transform, collapsing
augmentation diversity per epoch from 8^B states to 8. Every test committed with
Task 2.1 passes under both, because the marginal distribution of augmented values is
identical either way — that is precisely why the KS results above could not settle it.

`src/training/augment.py:69-71` draws length-`n` tensors, so the source says
per-sample. Measured rather than read off the source, by recovering the transform
each sample **actually received** (distinct-pixel inputs, exact match against the 8
dihedral elements) so a correct draw that is broadcast during application would also
be caught:

| measurement | old (pre-Phase-1) | new (current) |
|---|---|---|
| distinct `(hflip, vflip, rot_k)` per batch of 64, mean of 16 | 15.73 | 15.72 |
| distinct **dihedral elements** per batch of 64, mean of 8 | 8.000 | **8.000** |
| joint over 1000 batches | all 16 states, 0.0601–0.0646 | all 16, 0.0608–0.0643 |
| within-batch pairwise agreement, 200 batches (chance = 0.1250) | 0.1254 | **0.1242** |
| noise gate, per-batch fraction noised | — | mean 0.5016, range 0.312–0.703 |
| noise gate, batches at exactly 0.0 or 1.0 | — | **0 of 1000** |

All four draws — hflip, vflip, rot_k and the noise gate — are per-sample.
**The mechanism is excluded.** Pinned by three new tests in
`tests/test_augment_distribution.py` (`2de495c`), each verified to fail against a
deliberately broadcast implementation and to pass at six seeds.

Two honest notes on this result:

- `test_every_dihedral_transform_is_reachable_and_equally_likely`, committed with
  Task 2.1, **would already have failed** under broadcast: it asserts all 8 transform
  counts are non-zero from a single 4000-sample call. The mechanism was excluded
  before Rev C proposed it. It was not framed that way, and the new tests measure it
  at the real batch size and add the within-batch dimension, so they still earn their
  place — but the exclusion is not new evidence.
- Rev C specified "a chi-square test of independence between sample index and
  augmentation state". That test was written, measured and **dropped**. It does not
  detect broadcast — verified against the broken implementation, where it *passes*,
  because broadcast makes samples dependent on each other and not on their position,
  so every position keeps the same marginal distribution. It is also flaky: a 64x8
  table over 200 batches leaves ~25 counts per cell and the old path returned
  p = 0.0035 at one of four seeds tried. Pairwise agreement is the statistic that
  separates the two readings; the reasoning is recorded in the test file.

#### 2.1b-ii — `Path.stem` tile-name collisions

The collision is real and larger than "truncation" suggests. On a 0.1° grid,
`grid_0.55_52.05` through `grid_0.55_52.95` all stem to `grid_0.55_52` — ten-way:

| family | tile dirs | distinct names | distinct stems | collisions |
|---|---|---|---|---|
| `tesserav1.1_global` | 8,108 | 8,108 | **1,546** | 6,562 |
| `tesserav2` | 40,075 | 40,075 | **23,938** | 16,137 |

Every `.stem` in the repository, classified:

| site | subject | class |
|---|---|---|
| `diagnostics/reextraction_scope.py:76-80` | tile | diagnostic — uses `tile_index_name` since `e750f8e` |
| `diagnostics/mosaic_scope.py:73-81` | tile | diagnostic — uses `tile_index_name` |
| `datasets/tiles.py:47` | tile | inside `tile_index_name`, geoinfo `.tiff` branch (no fractional suffix) |
| `infer_roi.py:595` | **output** GeoTIFF filename | not a tile |
| `osm_lcz_relabel.py:362,426` | output filename, plot title | not a tile |
| `embedding_explorer.py:80` | parquet UI label | not a tile |
| `datasets/so2sat.py:80,414,615`, `pack_patches.py:98`, `knn_baseline.py:401`, `embedding_projection.py:90`, `diagnostics/embedding_stats.py:113`, `download_missing_coop_tiles.py:50` | `patch_NNNNNN.npy` | patch id — 342,944 files, **0 stem collisions**, 0 filenames with an extra dot |

**No consumer is in the data path**, and four independent lines of evidence say the
truncated name never reached it:

- extraction resolves tiles geometrically: `tree.query(patch_geom)` → positions →
  `tile_paths[i]` → `crop_patch` (`extract_so2sat_embeddings.py:219-238`). No name is
  formed at any point. `infer_roi.py:439-445` uses the same path; its `.name` appears
  only in log lines.
- `open_tile` routes on `path.is_dir()`, and `_open_tile_tessera_npy_dir` reads
  `npy_dir.name` — the full directory name (`datasets/tiles.py:457`).
- git history: `.stem` has **never** appeared in `datasets/tiles.py` before `e750f8e`
  (the commit that added `tile_index_name` and documented stem as wrong), and has
  never appeared in `extract_so2sat_embeddings.py` at all.
- `data/tessera_v1.1_global_2017_tiles.gpkg` carries all 8,108 **full** names, 0
  truncated.

**Verdict: no patch could have resolved to a neighbouring tile; 0 patches affected.**
The bug's entire blast radius was the two diagnostics, both already fixed at
`e750f8e`. Rev C's stop condition — counts differ **and** a consumer is in the data
path — is not met: the counts differ, no consumer is in the data path. Task 2.2 is
not blocked by this. Pinned by a new test in `tests/test_tiles_open.py` (`f5c852c`)
requiring two stem-colliding tiles to stay two index entries 0.9° apart.

#### Where this leaves the shortfall

Both mechanisms Rev C named are excluded, so the ~0.8-point kappa/OA gap between the
new reproduction (0.6128 ± 0.0042) and the historical runs (0.6210 ± 0.0051) has no
identified cause. It remains not separable from noise at n=3 (Welch p = 0.10 for
kappa, 0.12 for OA, 0.59 for macro-F1).

Rev C's 2.1b-iii is therefore **unblocked**: four additional seeds (3–6) of the Task
2.1 configuration, taking the new side to n=7 for roughly 70% power against the fixed
historical n=3. Not launched — held for a decision on whether the GPU goes to this or
to Task 2.1c's learning-rate audit first.

Until it is resolved either way, the pooled six-run band **kappa 0.6169 ± 0.0061,
range [0.6097, 0.6268]** stands as the Phase 2 reference, with Rev C's caveat that
pooling assumes both sides are the same population — which is the thing under test.
It is conservative for setting a bar and must not be cited as evidence of equivalence.

#### Verification

| check | result |
|---|---|
| `pytest` | **323 passed, 1 skipped** (319 + 3 augmentation + 1 tile-index) |
| new augmentation tests vs a broadcast implementation | **3 of 3 fail**, as required |
| new augmentation tests at six seeds | pass at all six — not seed-luck |
| tile-index guard: two stem-colliding tiles | 2 distinct paths, 2 distinct names, footprints 0.9° apart |
| scope | only `tests/` and `RESULTS.md` touched; **`src/` byte-identical to `1534c08`** |
| Tessera extraction dir | 342,944 files, unchanged |
| GATE 1 regression md5 `7c2222b04ad830fe0c08eb4ee686df51` | unchanged **by construction** — see below |

The GATE 1 md5 is a function of `src/`, the `student-coop-v1` checkpoint and the
coop tiles, none of which this task touched, and `git diff 1534c08 HEAD -- src/` is
empty. It was not re-run: with a provably zero source diff a re-run cannot produce
information, only the appearance of it. Task 2.1's md5 came from an actual
invocation, and any task that does touch `src/` must re-run it rather than inherit
this reasoning.
