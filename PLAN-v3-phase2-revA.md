# eo-fm — PLAN v3, Phase 2 revision (Rev A)

**Replaces the Phase 2 section of `PLAN-v3.md` in its entirety.**
Phases 3, 4, Deferred, and the Write-up note in `PLAN-v3.md` are **unchanged** — read them
there, do not duplicate them here. The scope restriction in `PLAN-v3.md` still stands:
`tesserav1.1_global`, `alpha_earth_coop`, `seamless` only.

Branch `exp/p2-revalidation`. **No changes to model or training logic in this phase** —
Task 2.0 is data-layer plumbing, everything else is runs.

---

## What GATE 1.75 changed

**Corrections accepted, superseding earlier text:**

- The effective noise ratio for `alpha_earth_coop` is **0.477**, cross-family spread
  **10.8×**. `PLAN-v3.md`'s pre-registered revision to ~0.57 is **withdrawn** — it applied a
  √0.69 discount to a figure measured on the paired sample, which was already effectively
  masked at 0.0184% sentinel, so the discount landed twice.
- The AlphaEarth sentinel is 26.07% of variance on the real population.
- The latitude hypothesis is a **bounded null**: tiles are projected UTM at exactly 10 m, so
  the degree-grid mechanism is falsified. Against √(h·w), ρ = +0.199/+0.228, GSD rises ~2%
  from the equatorial band to >50°, spread 3.1–3.4% across 51 cities, city-constant. Too
  small to be part of the 20-point cross-city gap. Bank it as a methods sentence and, per
  `PLAN-v3.md` Phase 4, keep it as a covariate.

**Findings that restructure this phase:**

- **The test split has zero invalid pixels** — 0.0000% across all 24,188 test patches for
  every family. Nodata handling therefore **cannot** move the headline test metric. It affects
  training and model selection only. Validation, by contrast, is 0.4433% invalid, so selection
  and evaluation sit on mildly different distributions within the same 10 cities.
- **Patch coverage differs across families and is the live confound.** On the training split:
  `alpha_earth_coop` 352,366 (100%), `seamless` 351,719 (99.82%), `tesserav1.1_global` 342,944
  (97.33%). Tessera also evaluates on **330 fewer test patches**. No nodata threshold touches
  this.
- **Truncation:** 484 Tessera training patches crop to as little as 2 px per side and are
  stretched to 32×32 — 4 real pixels presented as 1024. Concentrated in London, Guangzhou,
  Melbourne, Shanghai.

---

## Task 2.0 — Common manifest and coverage control (new, blocking)

Nothing else in Phase 2 runs until this is frozen.

### 2.0a — Native-fraction filter

Do **not** threshold on raw native pixel count: `seamless` is legitimately 11–13 px and would
be wiped out. Define, per family, using the per-family median native crop area measured in
Task 1.75.2 as the denominator:

```
native_frac = (native_h * native_w) / expected_native_area[family]
```

Add `--min-native-frac` (default `0.5`). Report the count dropped per family per split, and
the native-area histogram per family.

`seamless` was not checked for truncation in Phase 1.75 — check it here. If ESD has its own
truncated tail, the same filter catches it.

### 2.0b — Build and freeze the manifest

One versioned parquet, one row per `patch_id`, columns: `patch_id`, `split`, `city`, plus a
per-family availability and validity flag. Membership requires, for **all three** families:

1. present and readable on disk
2. `invalid_frac <= 0.25` (the GATE 1.75 recommendation — the distribution is bimodal, so the
   choice is insensitive; 0.05 is rejected because it would drop 1,542 Tessera patches whose
   nodata is a diffuse 1–5%, a different phenomenon)
3. `native_frac >= 0.5`

Report the manifest size per split and, per family, the loss relative to its own native
coverage. Freeze the file, version it, and record its hash in the W&B config of every run
that uses it. Add `--patch-manifest PATH`; absent, behaviour is native per-family coverage as
now.

### 2.0c — Report what the intersection is silently doing

GATE 1.5 established that pairing acted as an accidental nodata filter (855× density ratio).
Applying an intersection deliberately is fine; applying it without reporting what it removes
would repeat the same mistake with better intentions.

Report:

- how many of AlphaEarth's 1,277 high-invalid patches are **also** absent from Tessera — if
  most are, the intersection is already doing the drop policy's work and the two filters are
  not independent
- the manifest's per-city and per-LCZ-class composition against the full training set, flagging
  any city or class losing more than 5% of its patches
- specifically whether Cape Town, Lisbon, Mumbai, New York, London, Guangzhou, Melbourne or
  Shanghai are disproportionately affected, since those are where the two failure modes
  concentrate

If any single class loses a large share — LCZ 17 (water) is the obvious candidate, at 2.68%
invalid against 0.19% for the next highest — say so, because it changes what the per-class
metrics mean.

### 2.0d — Re-extraction scoping (measure only)

Dry-run the fixed `_fully_covered` from a958a3f over Tessera's 9,422 absent and 560 truncated
patches. Report how many would be recovered, how many correctly rejected, and the per-city
breakdown.

**Do not re-extract.** Task 2.1's reproduction anchors to the current extraction, and
re-extraction would break comparability with every existing Tessera number. The decision is
deferred to GATE 3, where it costs one anchor re-run rather than the phase. Record the
recovery count so the decision can be made on evidence.

---

## Task 2.1 — Reproduce the pre-fix baseline

`--normalize none --nodata-mode zero --max-invalid-frac 1.0 --min-native-frac 0.0`, **native
per-family coverage, no manifest** — that is what the original run used, and the anchor must
match it.

`tesserav1.1_global`, cultural split, 3 seeds. The Phase 1 RNG change means seed-level, not
bit-level, agreement is the standard: report mean ± std and check the existing 0.6190 (opt3)
falls inside. That spread is a deliverable in its own right — every later improvement has to
clear it.

If not already present, add a KS test on a large sample confirming pre- and post-Phase-1
augmentation produce statistically indistinguishable output distributions, separating "the RNG
stream changed" from "the augmentation semantics changed."

---

## Task 2.2 — The fixed cross-family comparison, two arms

ResNet34 (`--family resnet --preset small`), cultural split, Phase 1 defaults
(`--normalize channel --nodata-mode mask`), 3 seeds, `--lr 5e-4 --warmup-epochs 3`.

**Arm A — manifest-restricted (headline).** `--patch-manifest` set. Identical patches for all
three families in train, val and test. This is the controlled representation comparison and the
only arm from which cross-family claims may be drawn.

**Arm B — native coverage (secondary).** Each family uses everything it has, with
`--max-invalid-frac 0.25 --min-native-frac 0.5` applied per family. This is the
deployment-realistic comparison: Tessera's 2.67% lower coverage is a genuine product property,
not an artifact to be normalized away.

Report both, plus the A−B difference per family. That difference answers whether Tessera's
extra native patches help (more data) or hurt (they are the hard coastal and tile-edge cases) —
a question worth asking directly rather than leaving implicit in a coverage footnote.

For each arm: mean ± std for OA, macro-F1, kappa, and a confusion matrix for the best run of
each family.

**State plainly in the write-up that Arm A's test set is common by construction.** With the
test split already at 0.0000% invalid for every family, the headline metric depends on neither
the nodata policy nor the coverage policy — that is a robustness claim worth making explicitly
rather than leaving a reader to wonder.

---

## Task 2.3 — Factorial ablation of the Phase 1 fixes

Arm A manifest throughout, so coverage is held fixed while these vary.

Do **not** run these as single-factor rows. Once σ is expressed relative to the normalized
channel std, disabling normalization silently reverts σ to absolute units, so "normalization
off" and "noise miscalibration" are entangled and a single-factor ablation cannot say which fix
mattered.

| | σ = 0 | σ = 0.05 |
|---|---|---|
| `--normalize none` | **A** | **C** (old behaviour) |
| `--normalize channel` | **B** | **D** (new default) |

B − A isolates normalization. C − A is the damage the old absolute-σ noise was doing. D − B is
what calibrated noise buys. Run on `tesserav1.1_global` and `alpha_earth_coop` — opposite ends
of the 10.8× spread. 4 × 2 × 3 = 24 runs.

Expect **B − A near zero**: ResNet has BatchNorm immediately after `conv1`, so the network
already largely self-normalizes. That would not mean the normalization work was wasted — its
value is making σ mean the same thing across families, which is what C − A and D − B measure. A
flat B − A with a large D − B is the clean version of the story and should be written up that
way rather than as a disappointment.

Then, on the `--normalize channel` arm only:

- **σ sweep** {0, 0.025, 0.05, 0.1, 0.2}, 3 seeds, both families. 0.05 was inherited, never
  tuned. With the corrected ratios (0.044 Tessera, 0.477 AlphaEarth) the prior expectation is
  that the optimum sits well below 0.05 for AlphaEarth and near or above it for Tessera.
- **nodata mode** `zero` vs `mask` on `alpha_earth_coop`. This is now a **confirmation, not a
  discovery** — the test split has zero invalid pixels, so any effect must come through
  training dynamics and model selection, and the drop policy already removes the 1,277 severe
  patches. Predict near-null, run it anyway, and report it as the check that pins the claim.
- **resize mode**: add `--resize-mode {bilinear, exact-crop}`, where `exact-crop` takes the
  centre 32×32 of the native crop with no interpolation. Tests whether the ~1.031× downsample
  costs anything and makes the validity mask exact rather than thresholded.
  `tesserav1.1_global` and `alpha_earth_coop`, 3 seeds.

---

## Task 2.4 — Model selection metric

Best `tesserav1.1_global` config, Arm A manifest, with `--monitor val_f1`, `val_kappa`,
`val_acc`, 3 seeds each. Report the full metric triplet for each so the cost of selecting on
the wrong one is visible.

Add one line to the write-up on the val/test asymmetry: validation carries 0.4433% invalid
pixels where test carries none, so selection and evaluation sit on mildly different
distributions within the same 10 held-out cities. Small, but it is the kind of thing a reviewer
notices before the author does.

---

## Task 2.5 — Resolution versus representation

`seamless` is upsampled 2.67× per dimension from ~11–13 native pixels, so resolution and
representation quality are entangled and every ESD number in the chapter is computed on
interpolated data.

Add `--native-resolution` (skip the resize, use the native crop). Confirm the ResNet stem
handles 12×12 — with the 3×3 stride-1 stem and no maxpool the spatial trace is 12 → 12 → 6 → 3
→ 2, which is fine. Arm A manifest, 3 seeds each:

| arm | input | question |
|---|---|---|
| `seamless` @ native | no interpolation | does upsampling help, hurt, or neither? |
| `tesserav1.1_global` @ 12×12 | downsampled from native | matched-resolution control |
| `alpha_earth_coop` @ 12×12 | downsampled from native | matched-resolution control |

If Tessera at 12×12 ≈ ESD at 12×12, ESD's deficit is purely resolution. If Tessera still wins
at matched resolution, the ESD representation is worse independently of resolution. Either
answer is stronger than the 32×32 comparison alone.

Frame it correctly: 30 m giving ~11 pixels per 320 m patch is a **genuine physical limitation**
for a class system defined by sub-100 m morphology, not an unfairness in the pipeline. The
experiment separates the limitation from the artifact.

---

## GATE 2

Report: the manifest with its composition audit and the re-extraction recovery count; the
reproduction with seed variance; both arms of the cross-family table with error bars and the
A−B difference; the 2×2 factorial with its three contrasts; the σ, nodata and resize ablations;
the monitor comparison; and the resolution-versus-representation table.

**Decision point for the chapter.** If the ranking or absolute numbers moved, existing draft
conclusions need rewriting before any new modelling.

### Pre-registration (commit to `RESULTS.md` before launching)

Revised for the corrected ratios and the zero-invalid test split:

- `tesserav1.1_global` moves less than 1 point either way from the Phase 1 fixes.
- `alpha_earth_coop` improves most, and largely via the noise-ratio change (0.477 → 0.05).
- `seamless` improves modestly.
- B − A ≈ 0; D − B carries the effect.
- **Nodata masking: near-null on the test metric by construction.** I withdraw GATE 1.5's
  "matters more than we thought" — that rested on a population that includes no test patches.
  Any effect arrives through training and selection, and should be under a point.
- Resize mode near-neutral for the 10 m families.
- Arm A versus Arm B: within noise for AlphaEarth and seamless; genuinely uncertain for
  Tessera, since its extra native patches are disproportionately the hard tile-edge cases.
- The AlphaEarth–Tessera gap closes by 1–6 of its 9 points; ranking flip genuinely uncertain.
- If AlphaEarth does not improve at all, the noise was never the binding constraint and the gap
  needs a different explanation — most likely that 64-d annual composites carry less
  LCZ-relevant structure than 128-d time-series representations. That is a cleaner result to
  write up than a confound.

---

## Housekeeping

`data/guppd_bounds_big_cities.csv` is still untracked. Either commit it or add it to
`.gitignore` — an untracked data file in a repo whose results depend on reproducible inputs is a
small liability that costs nothing to close.

---

## Note for the write-up

Two GATE 1.75 findings belong in the methods regardless of what Phase 2 shows:

**The `_fully_covered` bug.** A UTM rectangle's WGS84 bounding box over-claims 0.15% of its area
at the equator and 10.7% at 78°N, so the coverage test accepted patches that came back
truncated. This is a general trap for anyone cropping projected rasters via geographic bounds,
and the measured residual after correction (0.09 m equatorial, 3.1 m at 78°N against a 10 m
pixel — sub-pixel, so densifying the boundary buys nothing) is the useful part. Worth a short
methods paragraph.

**The latitude null.** GSD varies 3.1–3.4% across the 51 cities and is city-constant. Reporting a
bounded null closes a hole a reviewer would otherwise open, and it strengthens the Phase 4
argument by ruling out a preprocessing explanation for the cross-city gap.