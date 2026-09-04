# eo-fm — PLAN v3, Phase 2 Rev C

**Amends `PLAN-v3-phase2-revA.md` and `-revB.md`. Both remain in force except where
superseded below.** Phases 3, 4 and Deferred in `PLAN-v3.md` are unchanged.

Task 2.1 is complete. New reproduction (3 seeds): macro-F1 0.5584 ± 0.0108, OA 0.6448 ±
0.0045, kappa 0.6128 ± 0.0042. Historical pre-fix runs (n=3): kappa 0.6210 ± 0.0051.
Welch p = 0.59 / 0.12 / 0.10 for macro-F1 / OA / kappa. Consistent ~0.8-point shortfall on
kappa and OA in the same direction on all three metrics, not separable from noise at n=3.

**Reference band for Phase 2, pooling all six pre-fix runs: kappa 0.6169 ± 0.0061, range
[0.6097, 0.6268].** Use this as what improvements must clear. Record the caveat that pooling
assumes both sides are the same population, which is the thing under test — it is conservative
for setting a bar, and must not be cited as evidence of equivalence.

---

## Task 2.1b — Rule out two mechanisms before Task 2.2 (blocking)

The GATE 2.1 verification could not have detected either of these. Both are cheap. Neither
requires a training run.

### 2.1b-i — Augmentation independence within a batch (highest priority)

GATE 1 described the change as "one batched draw vs N scalar draws." Two readings:

- `torch.rand(B)` → B independent values, per-sample randomness preserved. Harmless.
- `torch.rand(1)` broadcast → every sample in a batch receives the **same** flip and rotation.

**A KS test on output values passes under both**, because the marginal distribution of
augmented images is identical. What differs is independence *across samples within a batch*:
under the second reading, effective augmentation diversity per epoch collapses from ~8^B states
to ~8. That is a uniform handicap on every run in Phases 2–4, invisible to any marginal test,
and of roughly the magnitude observed.

Measure, for a batch of 64 on both the pre- and post-Phase-1 paths:

- the number of distinct `(hflip, vflip, rot_k)` states realised within a single batch
- the empirical joint distribution of those states across ~1000 batches
- a permutation or chi-square test of independence between sample index and augmentation state

Old path should realise essentially all 8 combinations per batch. If the new path realises 1,
that is the mechanism. **Fix it and re-run the Task 2.1 anchor before proceeding.**

If the new path realises 8 with an independent joint distribution, the augmentation is exonerated
and the shortfall stands unexplained — proceed to 2.1b-iii.

### 2.1b-ii — `Path.stem` tile-name collisions in the data path

`grid_121.35_31.25` stems to `grid_121.35_31` because `.25` parses as a file suffix. On a 0.1°
grid, latitudes 31.05 through 31.95 all collapse to the same string — a ten-way collision, not a
truncation. GATE 2.1 established this disabled the 2.0d corrupt-tile test; it did not establish
that the truncated name was confined to that test.

Report:

- the count of distinct full tile directory names against the count of distinct stems returned by
  `build_tile_index`, per family
- every consumer of the stem, each classified as **diagnostic-only** or **data-path**
- if any consumer is in the data path: whether a patch could have resolved to a neighbouring
  tile, and how many patches are affected

**If the counts differ and any consumer is in the data path, stop and report before Task 2.2.**
That would be a data-integrity issue in the extracted embeddings, not a diagnostic gap, and it
would change what the anchor is anchoring to.

### 2.1b-iii — Resolve or accept the shortfall

Only if 2.1b-i and 2.1b-ii both come back clean.

At sd ≈ 0.005 and a 0.008 shift, n=3 gives roughly 30% power; n=7 on the new side gives about
70% against the fixed historical n=3. Run **four additional seeds** (3–6) of the Task 2.1
configuration and re-test.

- If the shortfall resolves into noise: document it, keep the pooled band, proceed.
- If it hardens into significance with no identified mechanism: report it as an unexplained
  systematic offset, use the **new-code** distribution rather than the pooled band as the Phase 2
  reference, and note in the write-up that absolute comparison to published pre-fix numbers
  carries a ~0.8-point uncertainty. Do not chase it further — Phase 2's internal comparisons are
  all on new code and remain valid either way.

---

## Task 2.1c — Learning-rate and schedule audit (new, high value)

Seeds 0 and 1 selected their best checkpoint at **epoch 2**, with `--warmup-epochs 3`. The best
model is being chosen before warmup completes, at a low learning rate, after which val_kappa
degrades while training loss continues to fall. Seed 2 peaked at epoch 15. Historical runs
stopped at 12, 14 and 17 with the same shape.

The `opt3` recipe (`--lr 5e-4 --warmup-epochs 3`) was inherited from grid-split work, where
memorizing city-specific features is rewarded. On the cultural split it may simply be too
aggressive: the model finds its best cross-city solution almost immediately and then overfits the
42 training cities. Two of three seeds peaking during warmup is not a healthy schedule, and the
bimodality across seeds says the current recipe is sitting on an unstable point.

This is likely the cheapest unclaimed performance in the project, and it reframes the epoch-2
peak as **evidence for the domain-shift thesis** — the generalizing solution is found early and
then trained away.

Sweep on `tesserav1.1_global`, Arm A manifest, 3 seeds each:

- `--lr` ∈ {5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3}, warmup fixed at 3
- at the best LR: `--warmup-epochs` ∈ {0, 1, 3}
- at the best LR: cosine decay versus the current schedule
- report the **epoch of best val_kappa** for every run, not just the metric

Run this **before** Task 2.3. The factorial, σ sweep, resize and nodata ablations should all sit
on a schedule that is not mis-specified, or they will be measuring interactions with a bad LR.
If the best configuration differs materially from `opt3`, re-run the Task 2.2 baseline on it and
say so explicitly in `RESULTS.md`.

**Pre-registration (commit before running):** best val_kappa will occur at a later epoch under a
lower LR, and the seed-to-seed variance in peak epoch will narrow. If a lower LR does not move
the peak epoch later, the early peak is overfitting driven by capacity rather than step size, and
the answer is regularization or a smaller model rather than a schedule change.

---

## Task 2.2 — unchanged in design, one prerequisite

The B2 2×2 and B3 metrics stand as written. The deferred prerequisite is real: add
`--min-native-frac` to the training CLI so Arm B can apply the per-family filter at native
coverage. Add `--max-invalid-frac` at the same time if it is not already on the training path,
and confirm both are logged to the W&B config and written into the checkpoint alongside
`patch_manifest_sha256`.

---

## B4 — closed for Phase 2

22.1% of zone-straddling patches truncated (168 of 760), against 0.01% for multi-tile patches
within a single zone. `merge_multi_crs` implicated, `numpy_mosaic` clean. All 760 sit in the
training split, none in validation or testing, and the manifest excludes them — so no Phase 2
metric is reachable by this route.

Carry to GATE 3 as a Phase 4 prerequisite: identify the trigger that selects the 22%. The three
same-zone cases (all Shanghai, two tiles, all cropped to exactly 7×33) suggest a deterministic
geometric cause and are the cheapest place to start.

---

## Correction accepted

Recoverable training patches are 191 (2.0%), not 223 (2.4%), with 32 depending solely on the
corrupt Shanghai tile. The GATE 2.0 conclusion is unaffected: still zero of the 571 absent
validation and test patches. Correctly marked as a correction rather than swapped silently.