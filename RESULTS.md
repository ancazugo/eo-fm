# RESULTS

Running log of every result produced by `PLAN.md`. Diagnostics get a section per
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
effective noise ratio still spans ~14× across families. Two reasons the masking
barely moves these numbers, both worth recording:

1. The Phase 0 table was already **post-resize**, and bilinear resize dilutes an
   isolated sentinel pixel; the +17.8% inflation measured above is a
   native-grid effect.
2. The corrected run samples each family's own index at n=20000, whereas Phase 0
   used the 5-family paired intersection at n=5000 — which is most of the
   +4.0% on `seamless`, a family with no nodata at all.

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
