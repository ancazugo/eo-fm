# Global split design: which cities to train on, which to leave out, and when

**Question.** With So2Sat (51 cities, dense, 2016–18) and WUDAPT (~1,250 urban areas,
variable quality, 1990s–2024 labels), how do we choose training and held-out cities for a
global, multi-year segmentation model? Choose on Köppen class, geography, or something else?
And how does time enter?

**Short answer.**

1. **Unit.** Group neighbouring urban areas into *blocks* of bounded diameter (≤ 50 km).
   Assign whole blocks to splits, and drop training polygons within 20 km of anything held out.
2. **Strata.** Stratify on **region × Köppen main group**. Neither alone is enough: climate
   drives what Tessera's annual time series sees, and region drives built form and labelling
   convention.
3. **Choose test blocks to look like the map, not like the labels.** Hold out blocks whose mix
   of strata and whose distance to the nearest training city match the 5,558 GUPPD urban areas
   the map will cover. Require every LCZ class and every label-year bin to be present.
4. **Fixed and designed tests.** Keep the So2Sat culture-10 as a frozen test (A). Add a
   designed WUDAPT test (B) that fills the strata A misses. Tune with grouped CV on training
   blocks only.
5. **Settle climate vs geography empirically.** Run leave-one-region-out and
   leave-one-climate-group-out once, and see which hurts more.
6. **Time.** Pair labels with embeddings from their own year. Never split by year inside the
   same ground. Use relabelled-ground pairs inside the held-out blocks as the temporal
   reference.

`src/design_global_splits.py` implements all of this. The numbers below on the target
population are real. The selection itself has only been run on a synthetic WUDAPT, because
the WUDAPT file is not reachable from where this was written. Run the script on
`wudapt_clean_<hash>.parquet` to get the real assignment.

---

## 1. What the map has to serve

Target: all **5,558 GUPPD urban areas**. Köppen class is the modal land class over a 5×5 grid
inside each bounding box (Rubel et al. 2016 via `kgcpy`; use `--koppen-tif` with Beck et al.
2023 1 km for the real run). Regions are `lcz_wudapt.splits.ISO_TO_REGION`.

| Köppen | share of urban areas | So2Sat train cities | culture-10 |
|---|---|---|---|
| **Aw** tropical savanna | **18.1%** | 3 (Caracas 12 patches, Dhaka 30, Rio) | 0 |
| Cfa humid subtropical | 13.5% | 12 | 1 (Sydney) |
| Cwa monsoon subtropical | 12.8% | 5 | 1 (Guangzhou) |
| **BSh** hot semi-arid | **8.4%** | 0 | 0 |
| BWh hot desert | 7.5% | 3 (Cairo, Karachi, Lima 48) | 0 |
| Cfb oceanic | 6.3% | 9 | 2 |
| Am tropical monsoon | 6.0% | 1 (Manila 384) | 1 (Mumbai) |
| BSk cold semi-arid | 5.6% | 2 | 2 |
| Csa Mediterranean | 5.3% | 5 | 0 |
| Af tropical rainforest | 3.9% | 1 (Salvador, 1 patch) | 1 (Jakarta) |
| Dfb humid continental | 3.0% | **0** | 1 (Moscow) |
| Dwa monsoon continental | 2.4% | 0 | 0 |

The three most common urban climates on Earth are Aw, Cfa and Cwa. The culture-10 tests none
of Aw, BSh or BWh. Together those are a third of the world's urban areas.

By region × Köppen group, weighted by bounding-box area (`--target-weight area`):

| stratum | area share | count share | in culture-10? |
|---|---|---|---|
| Asia-East \| C | 17.8% | 15.1% | yes |
| America-North \| C | 10.8% | 2.7% | yes |
| Asia-Southeast \| A | 9.4% | 6.7% | yes |
| Asia-South \| A | 8.8% | 8.7% | yes |
| Europe \| C | 7.9% | 7.3% | yes |
| Asia-South \| B | 4.0% | 7.6% | yes |
| **Africa \| A** | 3.9% | **7.9%** | no |
| America-North \| D | 3.6% | 1.2% | no |
| America-North \| B | 3.6% | 1.4% | no |
| **Asia-South \| C** | 3.3% | **7.0%** | no |
| **Africa \| B** | 3.0% | **6.1%** | no |
| Europe \| D | 2.9% | 2.9% | yes |
| Asia-East \| B | 2.7% | 3.5% | no |
| America-South \| A | 2.6% | 3.7% | no |
| Asia-West \| B | 2.6% | 2.8% | no |

The strata the culture-10 touches hold **65% of urban area, but only 55% of urban areas by
count**. The gap is the many smaller cities of tropical and semi-arid Africa and South Asia.
Each culture-10 city also weighs 10% of the test whatever its stratum's real share: Sydney
stands in for a stratum that is 1.1% of the world.

**Distance.** Against So2Sat training cities (≥ 500 patches), the culture-10 is a fair
geographic test. Test-to-train distance (median 1,255 km) matches map-to-train distance
(median 1,071 km). But 52% of urban areas are more than 1,000 km from any So2Sat training
city, and the regional medians reach 2,850 km in Africa and 2,621 km in Central America.
WUDAPT closes most of that, which is why the test design has to be redone once WUDAPT is
in training.

## 2. The unit: blocks, not cities

Spatial autocorrelation does not stop at a GUPPD boundary. Shenzhen (So2Sat training) is
94 km from Guangzhou (culture-10), Hong Kong a little further, and WUDAPT AOIs sit closer
still (Dongguan, Foshan).

- **Don't group by "bounding boxes touch after a buffer."** On GUPPD that chains megaregions:
  at 25 km one component swallows 661 urban areas, and at 10 km the largest still holds 138.
- **Do use complete-linkage clustering of AOI centroids** at `--block-km 50`. Every pair
  inside a block is within 50 km, so a block is a metro region, never a subcontinent.
- **Handle neighbours with the buffer instead.** Training polygons within `--buffer-km 20` of
  a held-out AOI's hull are dropped (`polygon_split.parquet: buffer_drop`).
- **Rural polygons.** The 27% of WUDAPT polygons outside every GUPPD box are 62% natural
  classes. They are snapped to the nearest AOI within 30 km, else to their own 1° cell, so they
  can't leak into training from beside a test city.

**Calibrating 50 / 20 km.** Both are defaults, not measurements. Replace them with the range
at which leakage stops mattering:

1. Score a So2Sat-only model on WUDAPT polygons in the culture-10 cities.
2. Bin kappa by each polygon's distance to the nearest training polygon.
3. The buffer is where that curve flattens; the block size is about twice that.

## 3. Strata: why region × Köppen, and how to check

- **Köppen** sets phenology and vegetation. Tessera summarises a year of Sentinel-1/2, so an
  open low-rise district in Aw looks unlike one in Dfb, and natural classes A–D are defined by
  vegetation. It is the axis the embedding itself varies most along.
- **Region** sets building materials, density, block structure and informality. It is also
  where labelling conventions differ: lczkit measured WUDAPT–So2Sat agreement at 91% in Europe
  against about 72% elsewhere.
- **Neither alone separates the cases that matter.** Cfa spans Shanghai, São Paulo, New York,
  Milan and Sydney. Aw spans Lagos, Kolkata and Rio.

**Weighting.** `area` emphasises megacities; `count` emphasises the many small cities where
most of Aw/BSh lives. Design on `area`, and report test B under both weightings.

**Data-driven alternative.** Cluster each block's mean Tessera embedding (k ≈ 12) and use the
cluster as a third stratum. It describes the domain the model actually sees. Do this only if
the stress test below shows neither Köppen nor region explains the error.

**The stress test that decides the question.** With fcn8-small (130 k parameters, cheap):

| Protocol | Hold out | Measures |
|---|---|---|
| LORO | each region in turn (Africa, Asia-South, Asia-Southeast, Americas) | building-culture shift |
| LOKO | each Köppen group in turn (A, B, D) | climate shift |

Compare each protocol's drop against grouped CV. If LOKO drops more, climate is the primary
axis: make Köppen the outer stratum and raise its weight. If LORO drops more, the reverse.
Report this table in the paper either way: it shows which shift the model cannot cross.

## 4. Eligibility: who may be tested on

A block may enter test B or val only if:

- it holds no culture-10 city (those are test A) and no So2Sat training city with ≥ 500
  patches (those are training);
- it has ≥ `--min-test-km2` 5 km² of labelled area across ≥ `--min-test-classes` 8 classes;
- `nbr_conflict` ≤ `--max-conflict` 0.40. Pass `--aoi-quality` from `lcz_wudapt suitability`.
  The suitability doc measured Spearman +0.929 between `nbr_conflict` and So2Sat agreement;
- it is not on `--quarantine`. Tehran by default: 0.46 agreement over 90 patches. India's
  0.46–0.72 conflict will mostly fail the gate on its own.

`--force-test` and `--force-train` take AOI keys for manual overrides. Use the full
`{slug}__{SMOD_ID}` key: "lagos" alone also matches Lagos de Moreno, Mexico.

**The cap.** No more than `--max-stratum-holdout` 40% of any stratum's labelled area may be
held out across test A, B and val. Without it the optimiser can test on most of West Africa,
which supplies the LCZ 7 the rest of the corpus lacks (8.3% of its patches, against 1.1% in
So2Sat). Test A is fixed and may exceed the cap on its own: in the synthetic run Santiago held
74% of its stratum and Moscow 59%.

## 5. Selection

Test B is chosen to minimise:

| Term | Meaning |
|---|---|
| strata L1 | half the L1 distance between test B's stratum shares and the map target's |
| distance KS | KS statistic between test→nearest-train and map→nearest-train distances (nearest-neighbour distance matching, Milà et al. 2022). A test set that is too close to training reads optimistic; one that is too far reads pessimistic |
| class deficit | mean shortfall below `--min-class-km2` per LCZ class |
| year deficit | mean shortfall below `--min-year-km2` per label-year bin |
| size | deviation from `--test-frac` 15% of WUDAPT labelled area |

The search is a greedy build-up over 64 sampled candidates per step, then 400 swap steps,
best of 8 restarts, with a fixed seed. Validation is then chosen the same way from what is
left (10%). The remaining training blocks are dealt into 5 CV folds, balanced on stratum and
class area.

At realistic scale (1,302 AOIs, 1,068 blocks, 930 eligible) this takes about 4.5 minutes.
In the synthetic run test B's stratum shares landed within 2 points of the target in most
strata, and test-B-to-train distance matched map-to-train (median 73 km vs 66 km).

**How many test blocks.** Per-city kappa has an sd of about 0.2 (0.18–0.23 measured across
the segmentation ladder, before and after the TTA fix). The 95% half-width of the mean city kappa is ±0.12 at 10 cities, ±0.07
at 30, ±0.055 at 50 and ±0.04 at 80. **Aim for ≥ 50 blocks in test B.** If WUDAPT's
heavy-tailed sizes (Guangzhou alone is 1,113 km²) make 15% of area fewer than 50 blocks,
raise `--test-frac` rather than accept a wide interval.

## 6. The splits and what each one is for

| Split | Contents | Used for | Reported as |
|---|---|---|---|
| **test A** (frozen) | So2Sat culture-10 blocks | comparison with every earlier number | patch kappa, per city, unchanged protocol |
| **test B** (designed) | WUDAPT blocks chosen above | global performance where the map will be used | polygon-level kappa / OA_w per city; mean over cities with a city-bootstrap CI; area- and count-weighted; per stratum; per label-year bin |
| **val** | designed the same way | early stopping only | — |
| **CV folds** | training blocks, 5 balanced folds | recipe choices (crop size, sampling α/β, source weights) with a small model | mean ± sd over folds |
| **LORO / LOKO** | one region or Köppen group out | which shift the model can't cross | drop vs grouped CV |

Two rules:

- **Score polygons, not pixels.** Score a WUDAPT test polygon by its majority prediction, so
  one 50 km² polygon doesn't outweigh fifty small ones.
- **One final training run.** Train the final models once, on all training blocks. CV is for
  choosing, not for reporting.

`lcz_wudapt.splits` currently assigns a hash split per AOI, stratified by region. Moving to
this design changes which WUDAPT cities are in test. The E1–E3 runs were scored on test A, so
their numbers stay comparable. Only "WUDAPT-test" numbers have to be recomputed.

## 7. Time

The labels are not spread evenly in time, and the imbalance runs with source and region.
WUDAPT polygons date ≤2014 8.6%, 2015–18 9.6%, 2019–21 41.8%, 2022–24 40.1%. So2Sat is
entirely 2016–18 and Europe/China-heavy. Left alone, a model learns "2017 = European
morphology". The design handles time in six places:

1. **Pair each label with its own year's embedding**, `clip(label_year, 2017, 2025)`.
   Pre-2017 labels pair with 2017 only where the change mask says the ground is stable.
   Otherwise down-weight them with `w_time` (τ = 8) or drop them.
2. **Decorrelate year from source.** Stable So2Sat patches draw 2017 or 2018 at random. Stable
   WUDAPT polygons occasionally draw a neighbouring year.
3. **Every test set covers every year bin**, enforced by the year-deficit term. Report test B
   accuracy per bin. A drop in one bin is drift or year-specific learning, not a city effect.
4. **Never split by time within the same ground.** Same block, different years, one in train
   and one in test is leakage wherever the ground didn't change, which is most of it.
5. **Relabelled ground is the temporal reference.** `revisit_pairs.parquet` lists polygons
   inside val/test blocks drawn by different submissions in different years with ≥ 50%
   overlap.
   - **Same class:** the map should agree across those years. Score how often it does.
   - **Class differs:** a candidate change. It is noisy, because annotators also disagree
     without any change, so weight it by the pair's agreement and use it as a recall check
     for detected change, not as ground truth.
6. **Optional forward test.** Train on labels ≤ 2020 (plus So2Sat), and test on ≥ 2022 labels
   in test B. This measures how well the model extrapolates in time, separately from space.
   Only worth running if step 3 shows a year effect.

## 8. What the real run will probably pick, and why

This is provisional, a sanity check on the script's output. The statistics come from
`docs/wudapt_label_suitability.md`, the Köppen classes from the lookup above. Some of those
classes are suspect at 3 km (Bhopal reads Csa; Nairobi reads Cfb rather than Cwb), which is
why the real run should use the 1 km map.

| Candidate | Stratum | WUDAPT | Role it should play |
|---|---|---|---|
| Abidjan, Lomé (Aw) | Africa \| A | 398 / 269 patches; region conflict 0.19 | **test B / val**: the largest stratum A misses |
| Ouagadougou (BSh) | Africa \| B | 334 patches | **test B**: Sahelian semi-arid, absent everywhere else |
| Lagos, Accra (Aw) | Africa \| A | 767 / 564 patches | **train**: keep most West African LCZ 7 in training (the cap enforces this) |
| Chennai, Kolkata (Aw) | Asia-South \| A | 149 / 351 patches | test B if `nbr_conflict` passes; Kolkata is already `wudapt_split=test` |
| Delhi (BSh/Cwa) | Asia-South \| B/C | 483 patches, conflict 0.72 | **train or quarantine**: too contested to test on |
| Singapore, Kuala Lumpur (Af) | Asia-Southeast \| A | 256 / 174 patches | one in test B, one in training (Jakarta is already test A) |
| Santo Domingo, Havana (Am) | America-Central \| A | 99 / 90 patches | test B: a whole region with no So2Sat at all |
| Chicago (Dfa) | America-North \| D | 497 km² segmentation, So2Sat 48 patches | test B: the only continental-climate North American option |
| Salvador (Af) | America-South \| A | 107 km², 17 classes | train: rescues a city So2Sat has one patch of |

## 9. Run, inspect, freeze

```bash
python src/design_global_splits.py \
    --wudapt-clean ${DATA_DIR}/output/lcz_wudapt/wudapt_clean_<hash>.parquet \
    --aoi-quality  <suitability per-AOI csv: aoi, nbr_conflict[, agree_on_overlap]> \
    --koppen-tif   <Beck et al. 2023 1 km Köppen GeoTIFF> \
    --out-dir data/global_split_design
```

Read `design_report.md` before training on it:

- stratum shares for test B against the map target;
- per-class and per-year-bin km² in each split;
- distance percentiles, test vs map;
- how much the buffer costs;
- the held-out AOI list.

Override with `--force-test` / `--force-train` / `--quarantine` and re-run. When the design is
accepted:

- commit `design_config.json` and `aoi_table.parquet` (or their sha256), as
  `patch_manifest_v1.parquet` is committed, so every run can record which split it used;
- have `lcz_wudapt` read `aoi_table.parquet`'s `split` column in place of its hash split.

## Caveats

- **Köppen source.** `kgcpy` is the Rubel et al. 2016 raster at ~3 km and misclassifies some
  cities. The script takes a 1 km GeoTIFF for exactly this reason.
- **So2Sat area proxy.** So2Sat label area is `patches × 0.1024 km²`, which overcounts where
  patches overlap. It only affects test A's and the training set's stratum shares.
- **WUDAPT is not independent of the Demuzere 2022 map.** The LCZ-Generator training areas up
  to December 2021 trained that map. Any experiment using Demuzere pseudo-labels must
  de-duplicate against WUDAPT test B.
- **Licence.** Polygons submitted since LCZ-Generator v2 are CC BY-NC-SA, which carries over
  to any released split table.
