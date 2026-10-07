# WUDAPT training areas under LCZ-Generator QC, reshaped to So2Sat geometry

**Status:** built 2026-09-08. Supersedes the consensus route documented in
[`wudapt_harmonization.md`](wudapt_harmonization.md), which remains importable
and tested but is off the default path.

## Why this exists

So2Sat-LCZ42 covers 51 cities. The LCZ-Generator submission database
(`$DATA_DIR/input/WUDAPT/LCZ-Generator_training_areas_2024-10-01.gpkg`) covers
**1,251 GUPPD urban areas with 630,311 polygons across 8,827 submissions** — 25×
the city count, and concentrated in exactly the parts of Asia, Africa and Latin
America the So2Sat benchmark under-samples.

The earlier attempt treated the variance between annotators as a *consensus*
problem (Dirichlet posteriors over per-pixel class sets). That was the wrong
frame: the LCZ Generator and WUDAPT already publish an explicit quality-control
regime for training areas, and the ESSD global map resolves annotator conflict
with a priority rule, not a posterior. This module follows the published rules
and reshapes the survivors into artefacts the existing training stack already
understands.

**Goal:** take a model trained on So2Sat and give it more geographic context
from WUDAPT — globally, and per city with a small label budget — across both the
classification and segmentation pathways, with the So2Sat culture-10 split
frozen throughout.

## The published rules

Sources: [Zenodo 13869766](https://zenodo.org/records/13869766) ·
[Demuzere et al. 2021, *The LCZ Generator*](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2021.637455/full) ·
[Demuzere et al. 2022, *A global map of LCZs* (ESSD)](https://essd.copernicus.org/articles/14/3835/2022/) ·
[WUDAPT digitizing guide](https://www.wudapt.org/digitize-training-areas/).

| Rule | Source | Value | Where implemented |
|---|---|---|---|
| Min area | Generator QC step 1 | 0.04 km² | `qc.qc_step1` |
| Max shape complexity | Generator QC step 1 | `P²/(4πA) < 3` | `qc.shape_index` |
| Narrowest width | WUDAPT guide | > 200 m | retargeted to 320 m, `so2sat_shape.place_patches` |
| Optimal size | WUDAPT guide | > 1 km² | `w_area` in `qc.polygon_weights` |
| Buffer between different LCZs | WUDAPT guide | > 100 m | `qc.neighbour_relations` |
| Examples per class | WUDAPT guide | 5–15 | `min_polys_per_class` (flag, not cap) |
| Oversize handling | Generator | > 1.5 km² → ~350 m radius | `qc.reduce_oversize` |
| Spectral outliers | Generator QC steps 2/3 | DBSCAN ε=0.3, MinPts=Cᵢ/10 | optional, embedding-space |
| Accuracy floor | Bechtel 2019a / ESSD | OA > 0.50 | `qc.submission_gates` |
| Duplicate resolution | ESSD | keep highest-OA submission | `qc.resolve_duplicates` |

## Two findings from the data

### 1. `qc_step1` is exactly reconstructible — and latitude-biased as shipped

`qc_step1 == True` means **passed**, not "flagged", and is precisely
`area ≥ 0.04 km² AND shape < 3` with **zero exceptions in 630,311 rows**.

But it was computed on the shipped `area` column, which is **Web Mercator** and
therefore inflated by 1/cos²(lat). Measured against true UTM area on a
30,000-polygon sample:

| \|latitude\| | median area ratio | 1/cos²(lat) | agreement | shipped pass | corrected pass |
|---|---|---|---|---|---|
| 0–10° | 1.0122 | 1.0124 | 0.998 | 0.561 | 0.559 |
| 10–20° | 1.0811 | 1.0814 | 0.989 | 0.529 | 0.519 |
| 20–30° | 1.2198 | 1.2208 | 0.964 | 0.495 | 0.459 |
| 30–40° | 1.3837 | 1.3857 | 0.914 | 0.615 | 0.529 |
| 40–50° | 2.0206 | 2.0262 | 0.884 | 0.700 | 0.584 |
| 50–60° | 2.6646 | 2.6745 | 0.832 | 0.859 | 0.690 |

Correlation between the measured ratio and 1/cos²(lat) across bands is **1.0**,
and **all 2,013 disagreements run one way**: shipped passes, recomputed fails.

So the released QC is progressively *more lenient* away from the equator — at
55°N a nominal 0.04 km² floor admits polygons that are really 0.015 km². The
apparent rise in pass rate with latitude (0.56 → 0.86) is almost entirely a
projection artefact; corrected, it is far flatter (0.56 → 0.69).

**This matters directly for a globally representative label set.** Trusting the
flag admits small, unreliable polygons preferentially in Europe, Russia and
Canada while holding tropical cities to a stricter standard — biasing against
precisely the coverage this label set exists to add. Area, perimeter and shape
are therefore recomputed in each AOI's local UTM.

### 2. The 100 m inter-LCZ buffer rule is the best quality signal in the dataset

Per city, the fraction of candidate patches *not* within 100 m of a
differently-labelled polygon tracks So2Sat agreement better than `oa`, `oau` or
any other shipped column:

| City | polys | qc1 pass | contains 320 m | not within 100 m of another class | So2Sat agreement |
|---|---|---|---|---|---|
| Berlin | 549 | 0.964 | 0.599 | 0.795 | 0.94 |
| Lagos | 763 | 0.961 | 0.909 | 0.568 | — |
| São Paulo | 2,323 | 0.829 | 0.609 | 0.499 | 0.86 |
| Beijing | 11,819 | 0.827 | 0.236 | 0.372 | 0.74 |
| Bogotá | 80 | 0.812 | 0.446 | 0.966 | — |
| Nairobi | 449 | 0.659 | 0.372 | 0.736 | 0.77 |
| Tehran | 6,547 | 0.550 | 0.422 | **0.188** | 0.45 |
| Delhi | 9,158 | **0.323** | 0.475 | **0.228** | 0.38 |

It also explains Tehran's previously-unresolved confidence inversion: 81% of its
candidate patches sit on contested ground. It is emitted as `nbr_dist_m` and a
soft weight, **never as a hard filter** — at a 100 m cut it deletes most of
Tehran and Delhi, and those cities are the point.

## Geometry: what "So2Sat-shaped" costs

Fraction of polygons that can contain an inscribed square (40,000-polygon
sample):

| Side | All | QC-pass |
|---|---|---|
| 200 m (WUDAPT rule) | 43.6% | 72.2% |
| **320 m (So2Sat patch)** | 24.0% | **39.8%** |
| 1280 m (segmentation tile) | 1.9% | **3.1%** |

Containment is strongly class-biased — LCZ 1 22.4%, LCZ 2 26.2%, LCZ 4 25.4%,
LCZ 5 26.6% versus water 55.6% and dense trees 57.1%. That bias is the mechanism
behind water reaching 42% of an earlier blind-grid patch pool.

Two corrections worth recording, because both are easy to get wrong:

* **`buffer(-160)` does not guarantee a 320 m square fits.** It guarantees a
  320 m *disc*; a square of side `s` needs clearance `s/√2` ≈ 226 m. Since a
  square of side `s` contains that disc, the erosion is a valid **necessary**
  prefilter — so the 39.8% above is an upper bound — and every candidate is then
  checked with an exact `square.within(polygon)`.
* **The 100 m buffer rule cannot be applied by eroding polygons.** The median
  polygon is 0.048 km², about 219 m across; eroding 100 m per side leaves 19 m
  and destroys the dataset. The raster honours it by writing nodata where two
  classes claim a pixel, with a 20 m boundary erosion (matching the segmentation
  CLI's existing `--erode-px 2`).

## Temporal alignment: soft, and probably weaker than year-matching

So2Sat's imagery is 2016–18. WUDAPT's `representative_date` parses cleanly
(0 failures; 403 values outside 1990–2025 — 2029, 2117, 2323, 202 — treated as
missing).

| Label year | Share | | Lag from 2017 | Coverage |
|---|---|---|---|---|
| ≤ 2014 | 8.6% | | = 0 | **2.0%** |
| 2015–2018 | 9.6% | | ≤ 1 | 6.9% |
| 2019–2021 | 41.8% | | ≤ 2 | 19.9% |
| 2022–2024 | 40.1% | | ≤ 3 | 35.4% |

Median lag from 2017 is **4 years**.

* **`time_decay_years` defaults to 8.0, not 3.0.** At τ=3, 80.1% of the corpus
  falls below weight 0.5 — a hard filter wearing a soft filter's clothes. At τ=8
  only 29.6% does. LCZs change slowly; age should tilt the weighting, not gut it.
* **Year-matching the embedding is the primary correction.** With coop's
  {2017, 2025} the residual lag never exceeds 4 years, against a median of 4 and
  a maximum of 27 for a fixed 2017. `w_time` only penalises the residual.
* **Caution:** `oa` *declines* with recency (2019: 0.768, 2021: 0.690, 2022:
  0.638, 2023: 0.615, 2024: 0.631), so `w_time` and `w_acc` partly cancel. Fit
  and report them together, and include **τ = ∞** as an explicit arm — the null
  result is live.

## What gets written

`lcz_wudapt shape` writes `$DATA_DIR/input/WUDAPT/cities/{aoi}/`, mirroring
`$DATA_DIR/input/So2Sat-LCZ42/v4/cities/`:

* **`patches_reference_{aoi}.gpkg`** — 320 m squares, EPSG:4326, columns
  `patch_id, dataset, LCZ_class` (byte-for-byte the So2Sat schema) plus
  provenance: `weight, nbr_dist_m, nbr_conflict, oa, label_year,
  embedding_year, w_time, src_area_km2, src_shape, overlap_frac_diff_class`.
  **`oa` is the class-appropriate accuracy, not the submission OA:** it holds
  `oau` (urban-only) for built classes 1–10 and `oa` for natural 11–17, the same
  `acc` that `w_acc` ramps on. So `oa < 0.50` on a patch does *not* mean the
  0.50 submission floor failed — every patch passed that floor. In the
  2026-09-08 pool, 1,420 patches (all built-class, 177 AOIs) sit below 0.50 on
  `oau`, which no published rule gates (`min_oau` is 0 by default).
* **`patches_reference_{aoi}.tif`** — 10 m label raster from the full QC-passing
  polygons, EPSG:4326, uint8 1–17, **0 = nodata**.

Because that is exactly the pair `src/create_city_grids.py` requires, the whole
downstream stack runs unchanged: `create_city_grids`, `extract_grid_embeddings`,
`extract_so2sat_embeddings --patches-file`, both trainers' per-city mode, and
`--global-split` / `--split-col`. **No dataset, split or loss code knows WUDAPT
exists.**

Splits (`lcz_wudapt splits`) are city-disjoint and region-stratified over 10
regions built from GUPPD ISO3 codes (all 173 present codes mapped):
**867 train / 187 val / 197 test**. All ten So2Sat culture cities are **forced
into WUDAPT-test** and `assert_split_integrity` fails the run otherwise — WUDAPT
holds 29,376 Guangzhou and 6,547 Tehran polygons, so this is the difference
between a valid benchmark and a contaminated one.

## Training-side additions

| File | Change |
|---|---|
| `src/utils/adapt.py` *(new)* | `--init-checkpoint` (warm start **then** train, distinct from `--checkpoint` which loads and skips training), `--freeze {none,backbone,backbone_keep_bn}`, `--shots-per-class`. Head discovery is structural (any Linear/Conv2d of output width `num_classes`), verified across all 14 registered families. |
| `src/training/loop.py` | Optimiser now sees only `requires_grad` parameters. `requires_grad=False` alone does not freeze a weight Adam still holds — weight decay and momentum keep moving it. |
| `src/patch_classification.py`, `src/semantic_segmentation.py` | Wire the above; warm-start report and freeze summary go into the W&B run config. |

Shape mismatches are handled explicitly: `strict=False` forgives missing keys
but **not** differing shapes, so a 17-class So2Sat checkpoint seeding a
different class count would raise. Mismatched tensors are dropped and reported.

## The four arms

| Arm | Command shape |
|---|---|
| Global patch | `patch_classification.py --global-split --global-gpkg patches_wudapt_rxr.gpkg --split-col wudapt_split --init-checkpoint <so2sat-best>.pt` |
| Global segmentation | `semantic_segmentation.py --cities-dir $DATA_DIR/input/WUDAPT/cities --split-mode global --init-checkpoint <so2sat-seg-best>.pt --family unet` |
| Per-city patch | per held-out city: **A** `--freeze backbone` (head-only), **B** full fine-tune, at `--shots-per-class 1,5,10,25,all` |
| | *2026-10-06:* arm **A** is now truly head-only — `--freeze backbone` also keeps the backbone's BatchNorm running statistics fixed. Before that fix BN stats re-estimated on the adaptation data, so any arm-A run before 2026-10-06 was "AdaBN + head refit". `backbone_keep_bn` is unchanged. |
| Per-city segmentation | same two-stage protocol on the segmentation family |

Both must report **two** numbers: kappa on the frozen So2Sat culture-10 (does
adaptation *cost* anything? a drop is an honest finding, not a tuning failure)
and kappa on WUDAPT-test (does it generalise to the 800+ new cities?).
Reference points from [`global_lcz_campaign_2026-07.md`](global_lcz_campaign_2026-07.md):
single-model best **0.6497**, LOCO-weighted ensemble **0.6871**.

## Verification (measured 2026-09-08 over all 460,802 urban polygons)

### V1 — the recomputation corrects a real latitude bias

| \|lat\| | n | agreement | shipped pass | recomputed pass |
|---|---|---|---|---|
| 0–10° | 36,941 | 0.9952 | 0.516 | 0.512 |
| 10–20° | 33,101 | 0.9840 | 0.478 | 0.462 |
| 20–30° | 132,356 | 0.9585 | 0.467 | 0.426 |
| 30–40° | 192,145 | 0.9079 | 0.603 | 0.511 |
| 40–50° | 48,972 | 0.8621 | 0.723 | 0.588 |
| 50–60° | 16,566 | 0.8439 | 0.846 | 0.692 |
| 60–90° | 721 | 0.9098 | 0.913 | 0.823 |

Overall agreement 0.9277; Spearman(agreement, latitude band) = **−0.786**. Of
33,303 disagreements, **33,148 (99.53%)** are shipped-pass / recomputed-fail.
The shipped pass rate climbs 0.52 → 0.85 with latitude; corrected, it is nearly
flat at 0.51 → 0.69.

The remaining **155 reverse flips are fully explained**: 154 of them are
*shape*-index flips, not area — shipped shape median 6.66 (max 1374) against a
recomputed 1.88 (max 2.99). `ingest.clean` explodes multipolygons, and a
multi-part polygon's combined perimeter makes `P²/(4πA)` meaningless as a
compactness measure. The recomputation is more correct there too.

### V2 — the QC gates predict So2Sat agreement

| City | n | So2Sat agreement | `qc1_pass` | `qc_pass` | not conflicted | mean overlap |
|---|---|---|---|---|---|---|
| Paris | 666 | 0.98 | 0.722 | 0.640 | 0.608 | 0.010 |
| Berlin | 536 | 0.94 | 0.929 | 0.841 | 0.733 | 0.103 |
| São Paulo | 2,335 | 0.86 | 0.797 | 0.612 | 0.347 | 0.454 |
| Nairobi | 439 | 0.77 | 0.665 | 0.579 | 0.756 | 0.108 |
| Beijing | 11,325 | 0.74 | 0.666 | 0.415 | 0.222 | 0.396 |
| Mumbai | 1,018 | 0.66 | 0.574 | 0.516 | 0.695 | 0.043 |
| Guangzhou | 29,583 | 0.59 | 0.544 | 0.086 | 0.048 | 0.912 |
| Tehran | 6,692 | 0.45 | 0.456 | 0.171 | 0.040 | 0.797 |

| Signal | Spearman ρ | p |
|---|---|---|
| **`qc_pass`** | **+0.929** | **0.001** |
| **`qc1_pass`** | **+0.905** | **0.002** |
| `mean_overlap` | −0.667 | 0.071 |
| `not_conflicted` | +0.619 | 0.102 |
| `mean_weight` | +0.500 | 0.207 |

**Gate passed.** The full `qc_pass` (which includes the ESSD duplicate-priority
rule) predicts agreement better than geometry alone, and the priority rule does
the heavy lifting in the worst cities — Guangzhou falls to 0.086 and Tehran to
0.171. So2Sat agreement figures are those measured in the earlier harmonization
work; the QC statistics are measured here.

### Corpus totals

`qc1` 0.498 (vs shipped 0.570) · contains a 320 m square 0.204 · within 100 m of
another class 0.656 · **`qc_pass` 0.314 (144,660 polygons across 1,162 AOIs)**.

Year-matching cuts mean lag from **4.65 to 3.13 years** (271,461 polygons →
2017, 189,341 → 2025); mean `w_time` 0.698 with only **7.1%** below 0.5.

### V3 — class balance (45,760 patches, 915 AOIs, all 17 classes)

| LCZ | | n | WUDAPT % | So2Sat % | earlier blind grid % |
|---|---|---|---|---|---|
| 1 | compact high-rise | 491 | 1.07 | 1.40 | 0.5 |
| 2 | compact mid-rise | 2,018 | 4.41 | 6.73 | |
| 3 | compact low-rise | 5,768 | 12.60 | 9.11 | |
| 4 | open high-rise | 1,126 | 2.46 | 2.58 | |
| 5 | open mid-rise | 2,064 | 4.51 | 4.50 | |
| 6 | open low-rise | 6,307 | 13.78 | 9.77 | 4.1 |
| **7** | **lightweight low-rise** | **1,134** | **2.48** | 1.06 | 0.4 |
| 8 | large low-rise | 4,333 | 9.47 | 11.53 | |
| 9 | sparsely built | 2,197 | 4.80 | 4.37 | |
| 10 | heavy industry | 2,117 | 4.63 | 3.42 | |
| 11 | dense trees | 4,108 | 8.98 | 11.87 | |
| 12 | scattered trees | 1,424 | 3.11 | 2.58 | |
| 13 | bush/scrub | 818 | 1.79 | 2.88 | |
| 14 | low plants | 4,680 | 10.23 | 11.62 | |
| 15 | bare rock/paved | 539 | 1.18 | 0.70 | |
| 16 | bare soil/sand | 1,141 | 2.49 | 2.28 | |
| **17** | **water** | 5,495 | **12.01** | 13.60 | **42.2** |

* **Max class share 13.8%** (LCZ 6) against the blind grid's 42.2% and So2Sat's
  own 13.6%. Gate passed. Water is back from 42.2% to 12.0%.
* **LCZ 7 = 1,134 patches**, more of the pool than So2Sat carries (2.48% vs
  1.06%). The Demuzere pseudo-label pool kept **0**, so this closes a stated open
  direction in the campaign.
* **LCZ 1 = 1.07%, which fails the >1.5% gate I set.** This is a real cost of
  strict containment, not noise: only 22.4% of LCZ-1 polygons admit a 320 m
  square, the lowest of any class. It still lands close to So2Sat's own 1.40%,
  so the pool is not more compact-high-rise-starved than the benchmark it
  extends. The fallback if it matters is the relaxed-containment arm
  (`dominant_frac ≥ 0.9`), reported separately rather than mixed in.

Coverage — **915 AOIs against So2Sat's 51**, weighted toward the target regions:

| Region | patches | AOIs | | Region | patches | AOIs |
|---|---|---|---|---|---|---|
| Asia-East | 12,984 | 287 | | Asia-South | 3,912 | 128 |
| Africa | 8,733 | 106 | | Asia-Southeast | 1,493 | 20 |
| Europe | 6,653 | 165 | | Asia-West | 1,359 | 40 |
| America-South | 5,524 | 89 | | Oceania | 823 | 9 |
| America-North | 3,955 | 64 | | America-Central | 324 | 7 |

Splits: 29,225 train / 7,966 val / 8,569 test. All **2,983 patches in So2Sat
culture cities are in `test`**, verified.

### V4 — the artefacts really are So2Sat-shaped

`src/create_city_grids.py` runs **unmodified** on the generated directories: on
Lagos it produced 2,340 grid tiles (749 valid) and a `_split.gpkg` with
597 train / 72 val / 98 test. The So2Sat field list
(`patch_id, dataset, LCZ_class`) is an exact prefix of ours, CRS and geometry
type match, and both `.tif`s are EPSG:4326 uint8 with nodata 0. Patches measure
320.0 × 320.0 m on the ground.

Note `create_city_grids.py` logs "no CSV bbox found, falling back to label
extent" for WUDAPT AOIs — its `--bounds-csv` default only knows the 51 So2Sat
cities. The fallback is correct and arguably better here: it fits the grid to the
labelled footprint rather than a GUPPD bounding box that is mostly empty.
(Superseded 2026-10-06 for the 51 AOIs that are So2Sat cities: those now reuse
So2Sat's bbox by SMOD_ID, because a label-extent grid there is a different
checkerboard and leaks So2Sat evaluation patches into WUDAPT training tiles —
see `wudapt_label_suitability.md`. Non-So2Sat AOIs keep the fallback.)

## Is any of it usable?

Capacity, trust and leakage per city and per region — which cities support
classification, which only support segmentation, and which regions are worth
requesting embeddings for — are assessed separately in
[`wudapt_label_suitability.md`](wudapt_label_suitability.md)
(`python -m lcz_wudapt suitability`).

## Tests

`lcz_wudapt/tests/test_qc.py` (28), `test_so2sat_shape.py` (21),
`test_splits.py` (12), `tests/test_adapt.py` (35),
`tests/test_patch_item_contract.py` (6). Full suite: **379 passed, 1 skipped**
(excluding the numba-blocked collection below).

Also fixed while here, from the approved plan: `tta_city_adapt.py`,
`ensemble_eval.py` and `generate_pseudo_labels.py` still built bare
`(path, label, split)` tuples and would have raised `AttributeError` inside
`PatchDataset`; they now construct `PatchItem`, pinned by
`test_patch_item_contract.py`.

## Operational notes

**`/maps` is at 100% with ~129 GB free** (not the ~2.1 TB recorded earlier).
The label artefacts are negligible — uint8 + LZW on mostly-nodata rasters gives
0.09–0.52 MB per city, ~350 MB for the whole roster — but the *embedding*
extraction that follows is not, at roughly 72 KB/patch for coop and 143 KB for
Tessera. Check free space before extracting a large pool.

## Known environment issue

`momepy → libpysal → numba` requires NumPy ≤ 2.4 and the environment has 2.5, so
`lcz_labels/tests/test_blocks.py` and `lcz_wudapt/tests/test_export.py` fail at
**collection** — which breaks the bare `pytest` command documented in
`CLAUDE.md`. This predates and is unrelated to this work. The `lcz_wudapt` CLI
imports the superseded consensus modules lazily so a dead dependency in a
retired code path cannot take the whole CLI down; the fix proper is to pin
`numpy<2.5` or upgrade `numba`.
