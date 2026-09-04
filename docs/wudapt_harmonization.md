# Harmonizing WUDAPT/LCZ-Generator with So2Sat-LCZ42

Status: **Stage 0 + Stage 8 export complete, gate G0 passed** (2026-08-19). Package: `lcz_wudapt/`.

## Why

So2Sat-LCZ42 v4 covers 51 cities and ~400k patches, but 11 of those cities are
degenerate — measured from `patches_reference_rxr.gpkg`, Salvador has **1**
patch, Philadelphia 2, Buenos Aires 5, Bogota 8, Caracas 12, Dhaka 30, Chicago
48, Lima 48, Quezon City 384, Karachi 1140, **all single-class**. The
LCZ-Generator submission database covers **1,251 GUPPD urban areas**, 25x the
city count, and is the obvious route to a globally representative label set.

It is also far noisier, which is the entire problem.

## What the data actually is

`$DATA_DIR/input/WUDAPT/LCZ-Generator_training_areas_2024-10-01.gpkg`:
630,311 `Polygon Z`, EPSG:4326, 8,827 submissions, 2021-04 -> 2024-10.

Defects found and handled in `lcz_wudapt/ingest.py`:

| Defect | Scale | Handling |
|---|---|---|
| Invalid geometries | 4,883 | `make_valid` + explode (630,311 -> 634,448 rows) |
| `class` 18 and 19 | 633 | dropped; a nonzero count in a future release is a schema alarm |
| `qc_step*` mixed `True`/`T`, `False`/`F` | ~34k affected | normalised to nullable bool; unknown tokens -> NA, never silent False |
| `area` is **Web Mercator km²** | all rows, 1.35x median inflation | recomputed on EPSG:6933 (exact to +/-0.002% vs geodesic) |
| `city` free-text, multilingual (`北京`, `wuhan`, `..`) | all rows | never read; AOI assignment is spatial |
| `representative_date` typos (2029, 2117, 2323) | 404 | out-of-range -> NA, falls back to submission year |
| `JRC_NAME_MAIN` not unique (León x3) | 149 rows in GUPPD | AOI key is `{slug(name)}__{SMOD_ID}` |

Two structural findings that changed the design:

**The annotator is the author, not the submission.** 8,827 submissions collapse
to **1,491 named authors** (Wuhan 388 -> 41, Guangzhou 486 -> 79), averaging
3.02 versions per author per city. 106,793 polygons (16.8%) carry no name;
keying those on their submission invents 1,764 pseudo-annotators — *more than
the real author count*. Since `n_eff` drives confidence, that would inflate
apparent consensus precisely where it is least trustworthy.
`blank_name_policy` defaults to `collapse_per_aoi`: it may merge distinct
people, but understating consensus costs coverage while overstating it corrupts
labels.

**Filtering to GUPPD urban areas is class-biased.** The 173,646 polygons (27.4%)
outside every GUPPD bbox are **61.8% natural classes (LCZ 11-17) against 35.1%
inside** (LCZ 14: 14.2% vs 5.7%; LCZ 11: 12.4% vs 6.9%). Restricting to "big
urban areas" would systematically strip the classes So2Sat already
under-samples. They are retained in a `_rural` partition.

**Label epochs span 1990-2024**, not 2023-24 as the first sample suggested:
91.4% fall inside Tessera's 2015-2025 range, but **8.6% (54,638 polygons)
predate 2015** and can be matched to no available embedding year.

## The harmonization problem, measured

Agreement between *overlapping* WUDAPT polygons, per city (UTM intersections
>100 m²):

| City | Polys | Submissions | Overlap area / total | Class agreement |
|---|---|---|---|---|
| Wuhan | 44,341 | 388 | **6.03** | 0.870 |
| Guangzhou | 29,376 | 486 | 0.86 | **0.437** |
| Delhi | 8,947 | 72 | 0.86 | **0.383** |
| Tehran | 6,551 | 23 | 0.74 | **0.487** |
| Beijing | 11,307 | 205 | 1.05 | 0.656 |

Pooled: **0.71 by area**. A naive union is a badly noisy label set.

## Approach

`lcz_wudapt` is a *label producer* emitting the existing `lcz_labels` Stage 8
contract (uint32 LCZ bitmask + uint8 confidence + uint32 block-id), so
`lcz_train` consumes community labels with no change to its loss layer.

Votes are cast **per annotator** on the canonical 10 m UTM grid — author
collapse is a deterministic burn order (oldest submission first), so a revision
overwrites its own earlier version while disjoint earlier work survives. A
Dirichlet-smoothed posterior then decides hard (`|S|=1`), coarse (`2<=|S|<=3`),
or unlabelled. The coarse case is the point: `marginalized_ce` scores
`-log sum_{c in S} p_c`, so ambiguity costs nothing when the truth is in the set
and the model never has to guess which annotator was right.

## Gate G0 — label-only, no embeddings, no GPU

Consensus scored against So2Sat, versus a naive-union (largest-polygon-wins)
baseline, on the 8 cities spanning the measured agreement range.

**Methodological note, applied here.** Set-scored OA (a coarse label counts as
correct if the truth is anywhere in its set) is **not comparable across operating
points** — a larger set mechanically scores higher. Comparing a coarse-heavy
consensus against an all-hard naive union would flatter consensus for free. The
gate below therefore compares **hard consensus labels against the naive union**,
single-class against single-class.

| City | Raw agreement | Annotators | mean n_eff | Naive union | **Hard consensus** | Margin | Hard coverage |
|---|---|---|---|---|---|---|---|
| Tehran | 0.454 | 11 | 1.45 | 0.582 | **0.790** | **+0.209** | 0.421 |
| Guangzhou | 0.585 | 79 | 1.71 | 0.671 | **0.913** | **+0.242** | 0.597 |
| Mumbai | 0.655 | 5 | 1.06 | 0.579 | 0.475 | **-0.104** | 0.318 |
| Beijing | 0.740 | 25 | 1.71 | 0.625 | **0.851** | **+0.226** | 0.431 |
| Nairobi | 0.771 | 3 | 1.12 | 0.791 | 1.000 | +0.209 | 0.173 |
| Sao Paulo | 0.861 | 26 | 1.71 | 0.849 | 0.961 | +0.112 | 0.640 |
| Berlin | 0.938 | 5 | 1.77 | 0.905 | 0.893 | -0.013 | 0.555 |
| Paris | 0.980 | 7 | 1.08 | 0.958 | 0.992 | +0.035 | 0.391 |

- **G0.3 PASS** — mean hard-only margin **+0.114** against a +0.03 gate, winning
  in 6 of 8 cities at 44% hard coverage. Correlation between raw agreement and
  margin is **-0.69**: the gain is largest exactly where WUDAPT was worst, and
  the already-good cities are not damaged. (At the higher-precision operating
  point below, the margin is **+0.147** winning 7/8 at 26% coverage.)
- **G0.2 PASS** — leave-one-author-out beats a random-peer baseline by **+0.092**
  mean against a +0.08 gate. Negative in Mumbai and Paris; Paris's peer baseline
  scores 1.000, i.e. its annotators are near-duplicates of each other, so LOAO is
  degenerate there rather than failing.
- **G0.4 PASS** — the weight model is informative where it matters. Correctness
  by weight tercile rises monotonically in Tehran (0.43/0.52/0.86), Guangzhou
  (0.52/0.66/0.79), Beijing (0.57/0.63/0.71) and Sao Paulo (0.72/0.92/0.92) —
  the multi-annotator cities. It is flat or inverted only in the 3-7 annotator
  cities, where it is noise.
- **G0.5 PARTIAL — 7/8.** Confidence ranks correctness cleanly almost everywhere
  (Beijing 0.857 -> 0.884 -> 0.952 -> 0.985; Guangzhou 0.861 -> 0.870 -> 0.926 ->
  0.966). **Tehran inverts**: 0.794 -> 0.823 -> 0.655 -> 0.550, with n=18,113 at
  conf>=0.6 — far too many samples to be noise.

### `prior_alpha` is the calibrated precision/coverage knob

The Dirichlet prior strength, in units of one average annotator, trades hard-label
coverage against hard-label precision. Measured:

| `prior_alpha` | Beijing hard frac / OA | Tehran hard frac / OA |
|---|---|---|
| 0.50 | 0.596 / 0.765 | 0.630 / 0.740 |
| **1.00** (default) | 0.431 / 0.851 | 0.421 / 0.790 |
| 1.75 | 0.194 / 0.944 | 0.147 / 0.895 |
| 3.00 | 0.139 / 0.979 | 0.006 / 0.784 |

Tehran collapses at 3.00 (0.6% coverage, and precision falls again) — that is
over-smoothing, not a better operating point. Pick alpha in S1 against downstream
training, not by staring at OA.

### Two findings worth acting on

**Tehran's confidence inversion is diagnostic, not fatal.** High confidence means
many annotators agreeing *with each other* and disagreeing with So2Sat. Given
So2Sat's imagery is 2016-18 and Tehran's WUDAPT labels are ~2021-24, in one of
the fastest-changing cities in the set, the leading hypothesis is **real urban
change rather than annotation error** — which is exactly what H4's
change-vs-error triage exists to separate. The campaign doc independently records
Tehran as a label-shift city (§6). Do not "fix" this by tightening thresholds
before testing the temporal explanation.

**Mumbai inverts hard vs coarse**: hard-only OA 0.475 is *below* its set-scored
0.623 and below its own naive union 0.579 — its confident labels are worse than
its ambiguous ones, and it is the only city where consensus loses. Mumbai has
mean n_eff 1.06, essentially no annotator overlap, so a "hard" label there is one
person's opinion wearing a confidence badge. Candidate rule for S1: require
corroboration (`n_eff > 1`) before emitting a hard label.

### Leakage

The audit deliberately measures on So2Sat cities including held-out test cities
(Tehran, Nairobi, Mumbai, Guangzhou) **as a measuring instrument only**. No
WUDAPT label from them enters training: WUDAPT is admitted inside a So2Sat city
only where So2Sat is sparse or degenerate (`n_patches < 2000 or n_classes < 8`),
which selects 12 single-class training cities. Every held-out test city has
>= 4798 patches and >= 13 classes, so the rule *cannot* select one, and
`leakage.assert_no_test_city_admitted` enforces it rather than trusting the
coincidence.

## Bugs found and fixed during S0 (all pinned by tests)

1. **LOAO compared the wrong pixels.** Each fold rebuilt its own footprint grid
   while truth was rasterized on the full-AOI grid, so `ConsensusResult.index`
   indexed a different raster. Gate G0.2 would have been meaningless.
   `consensus_for_aoi(..., grid=)` now pins one grid across folds.
2. **`depth` was discontinuous at `n_eff == 1`**, so confidence depended on
   whether the arithmetic ran in float32 or float64. Now continuous.
3. **The Dirichlet prior was an absolute `alpha = 1.0`** while real annotator
   weights average 0.3-0.6, so a virtual annotator outvoted every real one — the
   hard-label fraction in Nairobi was 0.6% in a city that agrees with So2Sat at
   0.77. The prior is now scaled to one average annotator of that AOI.

## Where the labels are

`$DATA_DIR/output/lcz_wudapt/{aoi}/` — one directory per AOI, keyed
`{slug(JRC_NAME_MAIN)}__{SMOD_ID}`:

| File | Type | Contents |
|---|---|---|
| `lcz_bitmask_{aoi}.tif` | uint32 | bit `c-1` set per class in the label set; 0 = unlabelled |
| `confidence_{aoi}.tif` | uint8 | confidence x 100 |
| `block_id_{aoi}.tif` | uint32 | dense consensus-region index (0 = none) |
| `blocks_labelled_{aoi}.parquet` | GeoParquet | one row per region: `label_type, lcz, lcz_set, confidence, block_kind, n_eff, p_top, area_m2, label_source` |
| `adjacency_{hash}.parquet` | Parquet | empty, schema-correct (see below) |

Plus the shared inputs at the cache root: `wudapt_clean_{ingest_hash}.parquet`
(634,448 cleaned, attributed polygons) and `aoi_index_{ingest_hash}.parquet`
(1,251 AOIs).

Rasters are written on the AOI's **labelled-footprint** grid, not the full GUPPD
bbox — WUDAPT covers a small fraction of most city boxes. All three are
self-describing (CRS + transform), so alignment is by geo-reference rather than
convention. Local UTM, 10 m by default.

**Consensus regions play the role of blocks.** WUDAPT has no block geometry, only
overlapping opinions; a region is a connected component of pixels sharing an
identical `(lcz_set, source)`, which is the unit `erosion_valid_mask` needs for
its block-interior logic to mean anything.

**Adjacency is empty on purpose.** Consensus regions are islands in unlabelled
space, so a rook-contiguity graph over them would be near-empty and misleading.
An empty file with the right schema makes `block_graph_edges` return an empty
`edge_index`, so experiment B3 reports as *not run* rather than crashing.

### Verified against the `lcz_train` contract

On `Guiyang__30_1806` (720 regions, 556,904 labelled px):

- three rasters share CRS + transform; `(bitmask > 0) == (block_id > 0)` exactly
- every decoded label set is non-empty and within LCZ 1-17; confidence <= 100
- `erosion_valid_mask(..., min_conf=0.5, erosion_px=1)` -> 463,770 valid px
- `marginalized_ce` on a labelled window: **3.28** with random logits vs
  **0.0001** with oracle logits (chance is `log(17) = 2.83`; coarse sets score
  below it, as they should)

### So2Sat merge, observed

So2Sat is burned over WUDAPT, so it wins every contested pixel. The counts check
out against each city's actual So2Sat inventory: Salvador has exactly **1**
So2Sat patch and yields **1** So2Sat-sourced region; Buenos Aires 5 patches -> 1
region (contiguous patches merge); Dhaka 30 patches -> 5 regions.

A well-labelled So2Sat city is **refused outright** — building Tehran returns
"So2Sat city with sufficient labels". Only the 12 sparse/degenerate cities admit
WUDAPT, and the leakage guard runs before any build.

## The 320 m patch pool

LCZ is a neighbourhood-scale class by definition (Stewart & Oke: hundreds of
metres to kilometres), so patch classification — not per-pixel segmentation — is
the task. The Stage 8 rasters are an **intermediate representation**: they
resolve annotator overlap and define the regions. Nothing trains on them
directly.

### Placement is the whole game

The first attempt gridded each city blindly at 320 m and kept whatever happened
to be pure. That produced 23,406 patches with a badly distorted class mix,
because a lake trivially dominates a 320 m square while a fragmented downtown
parcel never does.

**So2Sat was not built that way** — its patches were placed *inside* labelled
polygons, which is why its coverage is clustered rather than wall-to-wall.
Placing patches on the consensus regions instead reproduces that construction:

| LCZ | Blind grid % | **Region-centred %** |
|---|---|---|
| 1 Compact High-Rise | 0.5 | **2.2** |
| 2 Compact Mid-Rise | 2.5 | 6.0 |
| 4 Open High-Rise | 1.9 | 4.9 |
| 5 Open Mid-Rise | 2.0 | 6.6 |
| 6 Open Low-Rise | 4.1 | **12.6** |
| 7 Lightweight Low-Rise | 0.4 | **1.6** |
| **17 Water** | **42.2** | **11.8** |
| *max class share* | *42.2* | *13.4* |

`max_per_region` is the control that matters: large regions are
disproportionately water and natural classes, so an uncapped tiling
re-introduces exactly the bias the placement removes.

### The pool

**41,371 patches, 1,099 AOIs, all 17 classes, mean purity 0.994**
-> `$DATA_DIR/output/lcz_wudapt/patches_wudapt_region_centred.gpkg`

* **27,195 patches (66%) in 1,059 cities that So2Sat does not cover at all** —
  against So2Sat's 51 cities.
* 14,176 patches in 40 So2Sat cities via gap-fill (So2Sat still wins every
  contested pixel; these sit on ground So2Sat never labelled).
* **660 LCZ 7 patches**, where the Demuzere pseudo-label pool kept **0** — the
  campaign's section 9 lists supplying LCZ 7 as an open direction it could not
  close.
* Schema is `patch_id, dataset, LCZ_class, weight, aoi, dominant_frac,
  labelled_frac, block_kind, geometry` (EPSG:4326) — exactly what
  `src/datasets/so2sat.py::build_pseudo_items` reads, so
  `patch_classification.py --pseudo-gpkg` needs no new training code.

For reference the Demuzere pool that produced the best single model (kappa
0.6497) held 89,320 patches; this is roughly half that, hand-drawn, and far
better balanced.

**Superseded:** `patches_reference_wudapt.gpkg` (the blind-grid pool) is kept
only for comparison. Do not train on it.

## Reproduce

```bash
python -m lcz_wudapt ingest      # ~90 s, caches on the ingest hash
python -m lcz_wudapt inventory   # So2Sat sparsity + leakage guard
python -m lcz_wudapt audit       # G0 on the 8 audit cities, ~12 min
python -m lcz_wudapt build --min-polys 1000 --res-m 10 --target-year 2023
pytest lcz_wudapt/tests          # 42 offline tests
# patch bridge: lcz_wudapt.patch_bridge.write_pseudo_gpkg(aois, cfg, out)
```

## Next (S1)

Consensus-region formation and the Stage 8 raster export, then the So2Sat patch
bridge. Reference points for G1 are the campaign's honest bests: single-model
kappa **0.6497**, LOCO-weighted ensemble **0.6871**
(`docs/global_lcz_campaign_2026-07.md`).
