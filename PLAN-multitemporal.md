# eo-fm — PLAN: global LCZ map on Tessera v1.1, then 2017–2025

Written against `55ea103` on `design/global-splits` (2026-10-23). This is a separate track from
`PLAN-V3.md`. The benchmark chapter (Phases 2–4 there) keeps its own gates, and nothing here
blocks them.

**Deliverable:** a global LCZ map on **`tesserav1.1_global`**, produced by a **segmentation**
model trained on three label sources:

- So2Sat LCZ42 (51 cities, imagery 2016–18);
- WUDAPT LCZ-Generator training areas (~1,250 urban areas, mostly labelled 2019–2024, variable quality);
- OSM, for a few classes and only as partial label sets.

The work runs in **two stages**, with a gate between them:

| Stage | Question | Labelling needed |
|---|---|---|
| **1 — single-epoch model** | How good is one model trained on every label we have, each paired with the year it describes? | none |
| **2 — multitemporal** | Does that model give a stable, believable 2017–2025 time series, and where does it detect change? | a stratified sample of a few hundred sites, once, at the end |

Stage 1 makes no claims about change. It can be finished, and written up, without stage 2.

Standing rules carry over from PLAN-V3:

- one factor per run;
- seeds per arm as stated;
- pre-register before launching;
- every result goes to `RESULTS.md`;
- stop at each GATE.

Contents:

- §0 decisions this plan fixes;
- §1 what is already in the repo;
- **Stage 1:** S1.0 embedding sign-off, S1.1 splits, S1.2 labels, S1.3 training, S1.4 evaluation, S1.5 single-year maps, GATE S1;
- **Stage 2:** S2.1 OSM-history stability rule, S2.2 embedding drift, S2.3 temporal training, S2.4 yearly maps, S2.5 time-series evaluation and the change sample;
- carried-over work, open decisions, budget.

---

## 0. Decisions this plan fixes

| # | Decision | Why |
|---|---|---|
| D0 | **The embedding product is `tesserav1.1_global`** (`/tessera/v1.1`, the "cambridge" v1.1 run). No other v1.1 product is mixed in, at any stage: not the per-city geotessera download, and not the streamed `tesserav1.1_dclimate` run. The same applies to OlmoEarth. | Same version label, different inference runs. Task 1.5.1 found matched-channel r ≈ 0 between v1.1 products (a linear map recovers only R² 0.89). Mixing them would make product differences look like LCZ differences. If S1.0 finds a year missing from `/tessera/v1.1`, the fallback is to move **everything** to one other complete run and retrain. It is never filling the gaps from another product. |
| D1 | **Segmentation is the main pipeline.** The patch classifier stays as baseline, as the test-A reference, and as the E3 teacher. | About 60% of QC-passing WUDAPT polygons cannot contain a 320 m square, so only segmentation can learn from them. The gap after the TTA fix is ~0.06 kappa (U-Net 0.602–0.607 against classifier 0.6675 on the same patches). |
| D2 | **One pooled model, applied to every year.** There are no per-year models and nothing to merge. Per-year predictions are combined only by seed averaging and, in stage 2, the temporal HMM. | Per-year models would each see a fraction of the labels, each with its own biases, which show up as spurious change. One model with one decision rule makes year-to-year differences attributable to the embeddings. |
| D3 | **The sample is (place, year), and each label is paired with the year it describes.** So2Sat uses 2017–18, WUDAPT uses `clip(label_year, 2017, 2025)`, and OSM uses its 2024–25 snapshot. Stage 1 uses these source years only. Stage 2 may extend a label to other years, but only where the stability rule (S2.1) allows. | This needs no change labels: it only means extracting the 1–3 label years each AOI has. Forcing everything onto one late year would put So2Sat (the cleanest source and test A) 7–9 years out of date. Year matching is still tested against that alternative (ablation A1, S1.3). |
| D4 | **The model gets no year, Köppen, region or lat/lon input.** These are used only for sampling, stratification and reporting. | A year input lets the model learn year-specific priors, which become spurious change. Köppen and region inputs become shortcuts tied to the training cities, fail on unseen ones, and cause seams at zone boundaries. |
| D5 | **No AI-generated product is used as a filter, label, or stability mask.** That covers GHSL, WSF, Dynamic World, ESA WorldCover, Google Open Buildings (incl. Temporal), Microsoft footprints, the ML-derived buildings inside Overture, the Demuzere 2022 LCZ map, and ETH canopy height. They may appear only as clearly labelled **evaluation comparators**. | Using them would make us inherit their errors and biases, and they would correlate with whatever they are compared against. See S1.2.5. |
| D6 | **OSM is used in three ways only.** (1) As low-weight **partial label sets** for a few classes. (2) As 2D **auxiliary training targets**, never inputs. (3) In stage 2, through its **history**, as evidence that something *existed* by a date. It is never a label for LCZ 1–7 or 9, and absence in OSM is never read as evidence. | OSM is reliable for water and large land uses. It cannot tell compact from open, or low-rise from high-rise, without height tags. Before 2017 it is incomplete in most of the Global South. |
| D7 | **The split design is frozen before any training run.** This means `docs/global_split_design.md` and `src/design_global_splits.py`. | Any city used for model choice cannot later serve as a test city. |

---

## 1. Already in the repo (do not rebuild)

| Need | Where it lives |
|---|---|
| `tesserav1.1_global` tile index and reader | `src/datasets/tiles.py::build_tile_index` (needs the sidecar fix, S1.0) |
| Product identity / basis checker | `src/diagnostics/tessera_product_check.py`, last report in `diagnostics/tessera_product_check.md` |
| WUDAPT harmonisation: author collapse, QC weights, `w_time`, consensus bitmask | `lcz_wudapt` (`quality.py`, `consensus.py`, `qc.py`) |
| Culture-10 protection: forced to test, 1.3 km buffer, So2Sat wins | `lcz_wudapt/splits.py`, `src/build_wudapt_train_gpkg.py` |
| Teacher–student WUDAPT arms E1/E2/E3, veto rate by label year | `src/build_wudapt_train_gpkg.py` |
| Global block split, strata, distance matching, coverage floors, revisit pairs | `src/design_global_splits.py`, `docs/global_split_design.md`, `tests/test_design_global_splits.py` |
| Marginalised CE over label sets (class weights, confidence, adjacency smoothing) | `lcz_train/losses.py::marginalized_ce` |
| Bechtel et al. (2020) class similarity matrix | `docs/lcz_class_similarity.csv`, `training/lcz_metrics.py` |
| Anchor-pixel window sampling with erosion and class balancing | `lcz_train/datasets.py::PixelWindowDataset` (the window range is 96–256, so **relax to 64**) |
| Segmentation families (U-Net, FCN8, Attention U-Net, ResNet-U-Net) and the generic training loop | `src/models/`, `src/training/loop.py`, `src/semantic_segmentation.py` |
| Sliding-window ROI inference with Hanning blending, pooled to a target resolution | `infer_roi.py --target-res 100 --aggregate soft`, `coarsen_bakeoff.py` |
| Demuzere et al. 2020 per-class Gaussian smoothing | `utils/lcz_smoothing.py` |
| Seed ensembles | `src/seed_ensemble.py` |
| OA_w / kappa_w / OA_u / OA_bu (Bechtel 2020), per-city kappa | `training/lcz_metrics.py`, `training/evaluate.py` |
| Stability-mask stage (to be **rewritten** in stage 2, S2.1) | `lcz_labels/change_mask.py` |
| OSM tag → LCZ evidence | `docs/osm_lcz_tag_mapping.md` |
| Streamed dClimate v1.1 and OlmoEarth v1.3 readers | `src/datasets/zone_zarr.py`. Benchmark comparisons only (D0) |

Current standing (W&B, culture-10 test, MobileNet-small, Tessera v2 2017):

- baseline 0.639 ± 0.022 (n=7);
- E3 0.679 ± 0.019 (n=10);
- E3 + `--min-epochs 30` 0.682 ± 0.018 (n=7).

The E-arms exclude culture-10 plus a buffer, so this gain holds for unseen cities. The Demuzere noisy-student gain does not yet: 7.4% of its pool sits in culture-10 cities. It is also an AI product, so under D5 it is out of the map recipe.

---

# Stage 1 — Single-epoch model

Each label is used at its own year (D3). No label is moved between years, and no change is claimed.

## S1.0 — Embedding sign-off for `tesserav1.1_global` (blocking, CPU only, ~1 day)

### Task S1.0.1 — Per-year inventory and the sidecar filter

`build_tile_index("tesserav1.1_global")` skips every NPY tile without a `global_0.1_degree_tiff_all/{tile}.tiff` sidecar. This is despite `open_tile` georeferencing from the tile name when the sidecar is missing, and `tessera_grid_geometry` matching the sidecar geometry on 699/699 tiles.

1. Report, for each year 2017–2025 under `/tessera/v1.1/global_0.1_degree_representation/{year}`:
   - NPY tile count;
   - the count that has a sidecar;
   - the count the index admits.
2. Let the index fall back to the tile-name geometry, as the v2 branch does.
3. Regenerate the coverage table for the **label years actually needed**: the footprint coverage of So2Sat cities at 2017–18, of every WUDAPT AOI at its label years, and of OSM-label AOIs at 2024–25. Today's figure is 3% mean footprint coverage of non-So2Sat cities at 2017, which most likely reflects the sidecar filter, not missing embeddings.

**Pre-registration:** tile counts are similar across years, and most of the gap between v1.1's 3% and v2's 34% (same cities, 2017) closes once the sidecar filter is removed.

**If a needed year is genuinely missing** from `/tessera/v1.1`, stop and decide per D0. Do not fill it from dClimate.

### Task S1.0.2 — Basis stability across years

Year matching mixes years in one training set, so stage 1 needs this.

Run `tessera_product_check.py` year against year on identical So2Sat patch ids: 2017 vs 2018, 2017 vs 2021, 2017 vs 2024.

- Matched-channel r should be high: same weights, different inputs.
- A year with r ≈ 0 is a different inference run, and **stops the plan** for that year.

### Task S1.0.3 — Coarse drift readout (no OSM history needed)

On So2Sat LCZ G patches in all 51 cities, report the per-year distribution of cosine and L2 distance to 2017, together with the L2-norm distribution. Call out the 2021→2022 step, when Sentinel-1B was lost.

This is a sanity check that year matching is not mixing embeddings on different scales. The full drift protocol is S2.2.

### GATE S1.0

Report:

- the per-year inventory;
- the coverage table for the needed years;
- the year-pair basis table;
- the G-patch drift curve.

**Decision:** go, or a D0 product decision.

---

## S1.1 — Freeze the splits (CPU, ~1 day on the cluster)

Full design and rationale: `docs/global_split_design.md`. Tool: `src/design_global_splits.py`.

| Split | What | Used for |
|---|---|---|
| **Test A** | So2Sat culture-10 cities, frozen | comparability with every past number |
| **Test B** | WUDAPT blocks chosen by the designer: 50 km complete-linkage blocks; strata region × Köppen main group; distance-matched to the 5,558 GUPPD urban areas (NNDM/kNNDM); coverage floors per class and year bin; ≤ 40% of any stratum held out | the headline global number; **aim ≥ 50 blocks** (kappa CI ≈ ±0.055) |
| **Val** | disjoint blocks, same design | early stopping and model choice |
| **CV** | 5 grouped folds over the training blocks | decisions the test sets cannot afford to see (FCN8 sweeps) |
| **Buffers** | 20 km around every test/val polygon; 1.3 km around culture-10 patches | no training crop may touch a buffer |
| **Quarantine** | `nbr_conflict` > 0.40; Tehran | excluded from all splits |
| **Revisit pairs** | the same ground labelled in two different years | kept aside for stage 2 (S2.1, S2.5) |

Stress tests are reported as transferability, not as the headline:

- LORO: leave one region out;
- LOKO: leave one Köppen main group out.

### Task S1.1.1 — Real run

Run `python src/design_global_splits.py` on the real WUDAPT gpkg with the Beck GeoTIFF.

Inspect `design_report.md` for:

- stratum coverage;
- distance KS against the target;
- class and year-bin floors;
- the forced and quarantined lists. Use full `{slug}__{SMOD_ID}` keys: a bare `lagos` also matches Lagos de Moreno.

### GATE S1.1

Commit `polygon_split.parquet`, `design_config.json` and `design_report.md`. **From here on, test B is read once per ladder rung (S1.3.6), never during tuning.**

---

## S1.2 — Labels

### S1.2.1 Sources and how each enters the loss

Every label is a **set** of LCZ classes per pixel (a 17-bit bitmask), with a per-pixel weight and a label year. One loss handles all sources: `marginalized_ce`, which scores `-log Σ_{c∈S} p_c`.

| Source | Label form | Weight | Embedding year(s) in stage 1 |
|---|---|---|---|
| So2Sat (train cities) | hard class per 320 m patch, rasterised to 10 m; per-city gpkg/tif | 1.0 | 2017 or 2018, drawn per sample (imagery 2016–18) |
| WUDAPT (train blocks) | `lcz_wudapt` consensus bitmask: hard where `\|S\|=1`, coarse set for 2–3 | polygon `w = w_qc · w_acc · w_size` × consensus confidence; `w_time` = 1 (year-matched) | `clip(label_year, 2017, 2025)` |
| OSM | partial sets, S1.2.3 | 0.2–0.3 (sweep once) | 2024 or 2025 (the snapshot used) |

- So2Sat stays authoritative inside its cities. WUDAPT is admitted there only where So2Sat is sparse (`lcz_wudapt` leakage rule).
- Labels from ≤ 2016 (8.6% of WUDAPT) are clipped to 2017. That leaves a lag of up to several years for those polygons, so report them separately in the label inventory.
- Keep the E3 teacher veto, rescored on year-matched embeddings. The teacher is our own So2Sat model, whose errors we measure, so D5 does not apply.
- Caveat: `oa` falls with recency (0.768 in 2019 → 0.615 in 2023), so `w_time` and `w_acc` partly cancel. Report them jointly.

#### WUDAPT polygon filtering

Do **not** use a So2Sat patch (10.24 ha) as the minimum polygon size. That constraint came from the patch classifier, and it would discard most of the small, carefully drawn polygons that motivated segmentation (D1).

| Step | Rule | Source of the threshold |
|---|---|---|
| Hard minimum | area ≥ 4 ha **and** shape < 3 (`perimeter² / 4π·area`), recomputed on EPSG:6933 | LCZ-Generator QC step 1. It reproduces `qc_step1` on all 630,311 rows, so it is applied from geometry rather than from the flag |
| Hard maximum | area ≤ 100 km² (unchanged `lcz_wudapt` default) | removes city-scale boxes only |
| Large polygons | no further hard cut. Polygons > 1.5 km² are reduced to a ~350 m-radius core around their pole of inaccessibility; `w_size` down-weights > 1 km² as now | the Generator does the same reduction before classifying; large polygons are typically coarse boxes over mixed ground |
| Submission and class quality | `oa` ≥ 0.50; `qc_step2/3` as soft penalties; `w_acc` from `oa`/`oau` × per-class F1 | Bechtel et al. 2019 floor; existing `lcz_wudapt` weight model |
| Crop cap | ≤ ~5 crops per polygon per epoch (S1.3.2) | stops large or numerous polygons dominating |

Report the surviving polygon count and area per class and region after each step, so it is clear which classes the filters thin out.

#### Conflicting overlaps

Where overlapping labels survive the filters and disagree, they are **not** resolved by picking a winner. The existing `lcz_wudapt` consensus already handles most of this:

- author collapse (a later revision by the same author overwrites the earlier one);
- per-annotator, quality-weighted votes per pixel;
- a Dirichlet-smoothed posterior that yields a hard label, a 2–3-class set, or no label;
- city quarantine at `nbr_conflict` > 0.40.

Two changes:

1. **Vote within a label year, not across years.** At present, all years are pooled with a `w_time` decay. Under year matching, a 2018 polygon and a 2023 polygon that disagree may both be right. Run the consensus separately per `label_year`. Stage 2 pools across years only where the stability rule covers both years (S2.3).
2. **Only allow sets of similar classes.** A coarse set is admitted only if every pair in it has Bechtel et al. (2020) similarity ≥ 0.67 (`docs/lcz_class_similarity.csv`; e.g. {2, 3}, {5, 6}, {8, 10}, {A, B}). A conflict between dissimilar classes (e.g. 2 vs G) indicates an annotation error, not ambiguity, so the pixel is left unlabelled. Report how much area each change moves between hard, set and unlabelled.

**So2Sat vs WUDAPT.** The two never both supervise the same pixel. Where they overlap, report their agreement per city and per WUDAPT author as a diagnostic of WUDAPT quality. It is not fed back into the weights. In stage 1 this agreement includes real 2017→label-year change, so treat it as an upper bound on disagreement. Stage 2 re-reports it on stable sites.

### S1.2.2 Label years in stage 1

Every labelled polygon carries its label year and nothing else. The sampler (S1.3.4) draws only that year, or 2017/2018 for So2Sat and 2024/2025 for OSM. Stage 2 adds `valid_years` (S2.1).

### S1.2.3 OSM partial label sets (current snapshot)

The class lists below are bitmask *sets*. A pixel labelled `{A, B}` costs nothing whether the model says A or B.

| OSM evidence | Label set | Conditions |
|---|---|---|
| `natural=water`, riverbank/`water=*`, coastline polygons | {G} | erode 2 px; drop `intermittent=yes`, `tidal=yes`, wetlands |
| large `landuse=forest` / `natural=wood` | {A, B} | ≥ 4 ha, no buildings inside, erode 2 px |
| `landuse=farmland/meadow/grass`, `natural=grassland/heath` | {D} | ≥ 4 ha, no buildings inside |
| `landuse=industrial` with large footprints (median ≥ 1,000 m²) | {8, 10} | ≥ 4 ha; buildings must cover ≥ 10% of the polygon; erode 2 px (these polygons are zoning-like and include parking, grass and vacant plots) |
| `natural=sand/beach`, dunes | {F} | ≥ 4 ha |
| `natural=bare_rock/scree`, `landuse=quarry` | {E} | ≥ 4 ha |
| `natural=scrub` | {C} | ≥ 4 ha, no buildings |

- **Never for LCZ 1–7 or 9:** those need density and height, which tags do not give (`docs/osm_lcz_tag_mapping.md`).
- Minimum area matches the WUDAPT QC minimum (4 ha ≈ 400 px).
- **Industrial is kept as the set {8, 10}.** Do not split it into 8 or 10 from tags such as `man_made=works` or `building=warehouse`. Those tags are sparse and unevenly mapped, so a split would be decided by mapping effort more than by form. The model resolves 8 vs 10 from the embeddings, supervised by So2Sat and WUDAPT.
- **Overture is used only for OSM-sourced features** (`sources[].dataset == "OpenStreetMap"`). Its buildings layer also merges Microsoft and Google ML footprints, which D5 excludes. That applies to the building filters above, to the morphometric targets (S1.3.3), and to the stage-2 history rules.

### S1.2.4 Label inventory

Report label pixels per source × class × region × label year, after filtering and consensus. The aim is to see which classes and regions depend on OSM alone, and which on a single year bin.

### S1.2.5 No-AI-filter policy (D5) in practice

| Product | Status |
|---|---|
| GHSL (BUILT-S/H/C), WSF, WSF-Evolution | comparator only; never a mask |
| Google Open Buildings (incl. 2.5D Temporal), Microsoft footprints, Overture's ML-derived buildings | comparator only; not a building source for OSM rules or targets |
| Dynamic World, ESA WorldCover | comparator only |
| Demuzere 2022 global LCZ map, noisy-student pseudo-labels from it | out of the map recipe; kept as a benchmark-chapter result. It was trained on LCZ-Generator TAs up to Dec 2021, so it is not independent of WUDAPT anyway |
| ETH canopy height, GHS-BUILT-H aux channels | out of the production path (also single-epoch, so they would freeze one year's structure onto all years) |
| Own So2Sat-trained teacher (E3 veto) | allowed: our model, our measured errors |
| Sentinel-1/2 band statistics (backscatter, NDVI, NDBI) | allowed, as stage-2 vetoes only |

### GATE S1.2

Report the label inventory, the filtering attrition table, and the consensus hard/set/unlabelled shares. **Decision:** the label build for training.

---

## S1.3 — Training (segmentation)

### S1.3.1 What goes in

| | Content | Shape / dtype |
|---|---|---|
| **x** | `tesserav1.1_global` embedding of the sample's year, dequantised (int8 × scale), per-channel normalised with train-set statistics stored in the checkpoint (repo convention) | 128 × H × W, float32, H = W ∈ {64, 96} |
| **target** | per-pixel 17-bit label set valid at that year | H × W, int32 bitmask |
| **valid** | labelled ∧ eroded 1 px ∧ embedding not nodata ∧ outside every held-out buffer | H × W, bool |
| **weight** | source weight × polygon/consensus confidence | H × W, float |

- **Not inputs:** year, Köppen, region, lat/lon (D4), or any OSM-derived parameter (D6).
- **Not loaded:** all nine years at once. A sample is one place at one year (D3).

### S1.3.2 Crops

- **Size:** 64 px (640 m) for WUDAPT and OSM anchors; 96 px (960 m) where So2Sat is dense. Both divide by 16, so U-Net depth 4 works.
- **No pre-cut tiles.** Crops are cut on the fly around anchor pixels from per-year mosaics.
- **Keep a crop only if:**
  - it has **≥ 200 labelled pixels after 1-px erosion** (≈ 2 ha; ≈ 5% of 64², ≈ 2% of 96²). The threshold is absolute, not a percentage, because WUDAPT's minimum polygon is 4 ha ≈ 400 px and a fraction would penalise the larger crop;
  - it has **≥ 75% valid embedding pixels**;
  - it touches **no** held-out buffer.
- **No maximum label fraction.** Fully labelled crops are fine: the unlabelled context is what the model uses, not what it is scored on.
- **At most ~5 crops per polygon per epoch**, so the big So2Sat cities and Wuhan's 44k polygons cannot dominate.

### S1.3.3 Loss

- `marginalized_ce` over label sets.
- Class weights: sqrt inverse frequency of train-set pixel counts, where a set-labelled pixel counts 1/|S| toward each member.
- Per-pixel weight from S1.3.1.

#### Similar-class confusions (rung R5)

LCZ classes are not equally different: confusing 2 with 3 matters far less than confusing 2 with G. Two options are tested, and nothing else:

| Option | Form | What it does |
|---|---|---|
| **R5a adjacency smoothing** | `marginalized_ce(smoothing_eps=ε)`, ε ∈ {0.05, 0.1}; already implemented | moves ε of the target mass onto the hand-coded morphological neighbours (1–2, 2–3, 4–5, 5–6, 3–7, 8–10, B–C, C–D) |
| **R5b expected-cost term** | `L = marginalized_ce + λ · Σ_c p_c · (1 − S[y, c])`, with S the Bechtel et al. (2020) similarity matrix and λ ∈ {0.1, 0.3}; for a set label, take the max similarity over the set | penalises a prediction in proportion to its physical dissimilarity; this is the quantity OA_w / kappa_w measure |

- Choose between R5a and R5b, and set ε or λ, with FCN8 on the CV folds. Then run the winner with U-Net small, 3 seeds.
- Do not combine either with heavy smoothing on pixels that already carry a coarse set (S1.2.1): the set already removes the penalty for those confusions.
- **Pre-registered trade-off:** cheaper similar-class errors make them more frequent. R5 is kept only if test-B kappa_w rises **and** unweighted kappa falls by no more than 0.01. Unweighted kappa stays the headline (S1.4.2).

Not adopted, deliberately:
- similarity-derived soft targets: redundant with R5a/R5b;
- factorised openness/height heads: more engineering for a gain R5 should already capture. Revisit only if confusions within LCZ 1–6 dominate the R5 error matrix.

#### Morphometric auxiliary targets (rung R6, optional)

OSM-derived urban form parameters (the lczkit family) are used as **auxiliary regression targets, never as inputs**.

Why not as inputs:
- They are a 2024–25 snapshot. As inputs they would stamp that year's structure onto every year, and OSM history would turn mapping growth into apparent change.
- OSM completeness varies strongly by region, so the model would behave differently exactly where labels are scarcest.

As targets, the model learns urban form from OSM during training and needs only embeddings at inference. Every year is then mapped the same way (D2).

| Target (per 10 m pixel, computed over a 100 m window) | From |
|---|---|
| building surface fraction | OSM-sourced footprints only (S1.2.3) |
| impervious surface fraction from roads/rail | OSM ways buffered with GeoClimate default widths, overridden by `width`/`lanes` tags (`docs/osm_lcz_tag_mapping.md`) |
| log median footprint area | OSM-sourced footprints |

Rules:
- **2D only.** Heights are excluded: tags are too sparse (≈ 1% of building area in Cairo vs ≈ 80% in Berlin), and the raster height products are AI products (D5).
- **Masked to where OSM is complete.** Use a history-saturation test per 1 km cell: the cell's building-count curve has flattened, < 5% growth over the last two years. This is computed from OSM history alone, with no external reference.
- **Stage 1:** targets only at 2024–25 embeddings. Stage 2 may extend them to earlier years in stable blocks (S2.3).
- **Model and loss:** a 1×1-conv regression head on the decoder, masked Huber loss, weight 0.1. The head is dropped at inference.
- **Keep rule:** as the ladder, plus a check that test-B kappa does not fall in regions where the mask is mostly empty. Otherwise the targets would be helping only the already well-mapped regions.

### S1.3.4 Sampling (hierarchical, per crop)

Draw each crop in five stages:

1. **Source**, with a fixed mix. Start at 40 / 50 / 10 So2Sat / WUDAPT / OSM and sweep once with FCN8.
2. **City / AOI** ∝ area^α, with α ≈ 0.3–0.5, so small and Global-South cities get seen.
3. **Class** ∝ inverse frequency^β, with β ≈ 0.5.
4. **Year**: the label year (WUDAPT); 2017 or 2018 with p = 0.5 (So2Sat); 2024 or 2025 (OSM).
5. **Anchor pixel**, uniform inside eroded pixels of that class, then the crop is centred on it.

Extend `PixelWindowDataset` or add `src/datasets/multisource_seg.py`. Either way it must:

- read per-year mosaics lazily;
- keep the seeded-stream semantics (one RNG per worker);
- allow 64 px windows;
- take the year from a per-label field, so stage 2 can swap in `valid_years` without a rewrite;
- log realised source, class, region and year frequencies per epoch.

### S1.3.5 Models, epochs, seeds

| Model | Role |
|---|---|
| **U-Net small** (depth 3, 32 features) | main model, 3 seeds per rung |
| **FCN8 small** | cheap: LR sweep, source-mix sweep, R5 choice, 5-fold CV. **First re-score the existing FCN8 rows with the fixed TTA**, because `data/seg_metrics.csv` used the broken one |
| **U-Net medium** (depth 4) | once, at the best rung after WUDAPT is added, to check whether more data now rewards capacity (it did not for So2Sat alone) |
| MobileNet-small patch classifier on v1.1 (E3 recipe) | baseline, test-A reference, E3 teacher, and the GATE S1 comparison |
| Shallow CNN | classification-only in the registry; not planned for segmentation |

- **Epoch:** 20,000 crops.
- **Early stopping:** on **val polygon-level kappa_w** (S1.4.2 scoring applied to val), patience 10, min 10, max ~50 epochs.
- **LR:** Adam + cosine (`training/loop.py`). Sweep {1e-4, 3e-4, 1e-3} with FCN8 small on rung R1, one seed each, then fix it.
- Report the epoch of best val for every run. Classification peaked during warm-up (PLAN-v3 revC, Task 2.1c), and segmentation should be checked for the same.

### S1.3.6 Run ladder (one factor per rung)

| Rung | Adds | Runs |
|---|---|---|
| R0 | re-score existing FCN8 / U-Net with fixed TTA; v1.1 So2Sat seg baseline at 2017 | eval only + 3 |
| R1 | So2Sat only, 2017 + 2018 embeddings | 3 + LR sweep 3 |
| R2 | + WUDAPT, year-matched, consensus sets, QC weights | 3 + U-Net medium 1 |
| **A1** | ablation on R2: every WUDAPT and So2Sat sample at **one fixed year (2024)** instead of its label year | 3 |
| R3 | + OSM partial sets (weight 0.2–0.3) | 3 |
| R5 | + similar-class loss: adjacency smoothing **or** expected cost, chosen on CV (S1.3.3) | FCN8 CV 4 settings + U-Net 3 |
| R6 | + OSM morphometric auxiliary targets (optional, S1.3.3) | 3 |

**Keep rule (pre-register):** a rung is kept if test-B mean city kappa rises by more than its bootstrap CI half-width, or by ≥ 0.01, **and** test A does not drop by more than 0.01. R5 has its own trade-off rule (S1.3.3).

Source-mix and α/β sweeps run on CV folds with FCN8, never on test.

**Ablation A1: is year matching worth it?** This tests the assumption that most labels did not change between 2017 and 2025.

- Both arms are scored the same way: test A at 2017, test B at each polygon's label year.
- **A1 ties R2** (within the CI): the assumption holds for the labelled ground. Year matching is kept anyway, because it costs nothing at training time, but stage 2 can rely more on stability extension.
- **R2 beats A1:** the gap measures how much label-year mismatch costs. Report it per region and label-year bin: the expected losers are fast-growing peri-urban areas, and So2Sat cities at 2024.
- **A1 beats R2:** a single year has better embeddings, e.g. later years are cleaner or 2017 is weaker. Check S1.0.3 before accepting it, and record it for S2.2.

### GATE S1.3

Report the ladder table (test A, test B mean ± CI, val, by seed) and the A1 readout. **Decision:** the stage-1 recipe.

---

## S1.4 — Evaluation

### S1.4.1 Test A — So2Sat culture-10, 2017

Pool the segmentation probabilities into each 320 m patch (soft mean, purity and buffer filters off, 83.8% patch coverage), take the argmax, and report kappa / OA / OA_w.

This is directly comparable to the patch classifier (0.6675 on the same patches) and to every E-arm number. Report both models side by side.

### S1.4.2 Test B — held-out WUDAPT blocks, at each polygon's label year

- **Per polygon:** majority of predicted pixels after 1-px erosion. Score it against the hard label, and separately as set-correct (prediction ∈ consensus set).
- **Per city:** kappa and OA_w. Headline = **mean across cities with a city-bootstrap 95% CI**. Also report the pooled Bechtel-2020 kappa_w / OA_u / OA_bu.
- Report per region, per Köppen group and per label-year bin too, from the same predictions.

### S1.4.3 Validation and CV

- Val, scored as S1.4.2, drives early stopping.
- The 5 grouped folds score the FCN8 sweeps.
- LORO and LOKO are run once on the final stage-1 recipe.

### S1.4.4 Comparators (D5, reported, never used)

Run the same test-B polygons through the Demuzere 2022 LCZ map. Report it in a separate table labelled "external products". Note its training overlap with WUDAPT up to Dec 2021: on test-B polygons from before 2022 it may have seen the same ground.

---

## S1.5 — Single-year maps

### Task S1.5.1 — Pooling and resolution, chosen once

Run `coarsen_bakeoff.py` on the stage-1 winner, `--target-res 320 100`, on culture-10. The expected answer is 100 m `--aggregate soft`; confirm it.

### Task S1.5.2 — Inference and calibration

- `infer_roi.py` with 128 × 128 sliding windows, Hanning blending, `--target-res 100 --aggregate soft`.
- Add `--save-probs` to write the `(17, H, W)` volume, uint8-scaled, beside the label GeoTIFF.
- Average the softmax over the 3 seeds; run them in one pass per tile so each tile is read once.
- Fit temperature scaling on val. If calibrated numbers are reported, fit it leave-one-city-out (`ensemble_stacking.py --city-holdout`). Earlier models were under-confident (T 0.46–0.98).

### Task S1.5.3 — One map year for inspection

Map the ~1,250 WUDAPT/So2Sat AOIs for **2024**: the latest WUDAPT label year and the OSM snapshot. Also map culture-10 at 2017 so test A can be inspected.

This is for visual QA and for spotting regional failure modes. It is not a change product.

## GATE S1 — Go to stage 2?

Report:

- test A and test B (mean ± CI, by region and Köppen group) for the stage-1 recipe and the v1.1 patch classifier on the same polygons;
- LORO / LOKO;
- the A1 readout;
- the 2024 maps for a stratified set of test-B cities.

**Pre-registered go criteria:**

1. test-B mean city kappa of the segmentation recipe ≥ that of the v1.1 patch classifier on the same polygons;
2. test A within 0.03 of the patch classifier;
3. no region in test B with kappa < 0.3 unless explained (e.g. by label scarcity in the S1.2.4 inventory).

If a criterion fails, stay in stage 1. A time series built on a model that fails these would mostly show its errors.

---

# Stage 2 — Multitemporal

Starts only after GATE S1. It adds three things to the stage-1 recipe: labels reused across years where nothing changed, drift control, and temporal decoding.

## S2.1 — OSM-history stability rule (no AI products)

**Purpose.** Decide, without manual checks and without AI change products, where a label from one year may be reused in other years. This replaces the Google-Temporal design in `lcz_labels/change_mask.py`.

**Principle.** OSM proves that an object **existed** by a date, *if it was mapped by then*. It never proves absence. A change in OSM after 2017 can be a mapping edit or real construction, and the two cannot be told apart, so such cells are **ambiguous** and do not propagate.

`change_mask.py`'s warning ("mapping growth is not urban growth") stays true. This rule never reads OSM growth as change. It reads only stable, early-mapped objects as no-change, and abstains everywhere else.

**Data.** OSM full-history planet (or the ohsome API), snapshotted at 1 Jan of each year 2017–2025.

**Rules**, evaluated per WUDAPT/So2Sat polygon and per `lcz_labels` block (momepy enclosure):

| Rule | Condition | Effect |
|---|---|---|
| **R1 water** | water polygon present in the 2017 snapshot and in 2025, with geometry IoU ≥ 0.9; not `water=reservoir` with `start_date` ≥ 2015; not intermittent/tidal | {G} valid for 2017–2025 (erode 2 px) |
| **R2 old fabric** | 2017 building footprint area ≥ 0.90–0.95 × 2025 area (sweep), **and** no building deletions 2017→2025, **and** 2017 building count above a completeness floor. Boosted (the threshold relaxes to 0.90) by `historic=*`, `heritage=*`, `building` `start_date` < 2000, or a conservation/heritage area. | an observed label propagates across 2017–2025, subject to the height guard |
| **R3 stable large land use** | `leisure=park`, `landuse=cemetery`, `aeroway=aerodrome`, `landuse=industrial` ≥ 10 ha; same geometry 2017→2025 (IoU ≥ 0.9); no building additions or deletions inside | the label propagates across 2017–2025 |
| **R4 abstain** | in the surrounding 1 km cell, 2017 building count < 0.8 × 2025 count (OSM was not complete in 2017, so mapping and building cannot be separated) | no propagation; source year only |
| **Height guard** | for classes 1–6, without `building:levels`/`height` history or a heritage tag | propagate at most ±3 years from the label year |

The rules use OSM-sourced features only (S1.2.3).

**Direction.**

- **Forward propagation** (2017 label → later years) needs no change through 2025.
- **Backward propagation** (e.g. a 2023 WUDAPT label → 2017) is riskier. The 2017 snapshot must show the objects already existed, which is exactly what R1–R3 test. Low-rise → high-rise redevelopment on the same footprint is invisible to footprints, which is why the height guard exists.

**Optional physical vetoes.** These come from measurements, not ML products, and can only **remove** stability, never add it:

- Sentinel-1 VV/VH annual-median backscatter change;
- NDVI/NDBI annual-median difference from Sentinel-2;
- Tessera embedding distance above the p95 of R1-water drift for that region and year.

**Output.** A `stability.parquet` beside the label gpkg: one row per polygon or block, with `valid_years` (9 bits over 2017–2025), the rule that fired, the inputs to that rule, and veto flags. It starts at the stage-1 source years, and every extension carries a reason code.

**Implementation.** Rewrite `lcz_labels/change_mask.py`:

- keep its graceful-degrade default (no evidence → source year only);
- drop the Google-Temporal primary source (D5);
- keep the `export_google_temporal_ee` path only for the comparator in S2.5.

Unit-test each rule on synthetic histories:

- a mapped-late building must not propagate;
- a reservoir must not propagate;
- R4 must abstain.

**Expected coverage.** Propagation will be high in Europe, North America and East Asian old cores, and low in Africa and South Asia, where 2017 OSM was sparse. That is correct behaviour: those places keep their stage-1 label years only. Report the propagated area per region × year so the imbalance is visible.

### Validating the rule (not every place)

We validate the *rule*, then trust it where it fires.

1. **Revisit pairs** (from S1.1). For pairs the rule calls stable, two-year label agreement should approach same-year inter-annotator agreement (0.71 pooled by area; per-city where n allows). For pairs it abstains on, agreement should be lower. Report both, by rule and region.
2. **Manual check of ~200 sites.** Use Google Earth historical imagery, 2017 against 2024, stratified by rule (R1/R2/R3) × region × class group. One person labels "unchanged / changed / can't tell".
   - Targets: precision ≥ 0.95 for R1, ≥ 0.90 for R2 and R3.
   - Report Wilson 95% CIs. With ~200 sites the overall CI is ±0.04 at p = 0.90; per stratum it is wider, so stratify to make every rule × region cell ≥ 10.
   - Run it in the **same labelling session and interface as the S2.5 change sample**, so both are one labelling task.
3. **Propagated area per region × year**, as above.

If R2 misses its precision target, tighten the coverage ratio and the height guard before giving up on it. R1 and R3 carry most of the value for the natural and large-land-use classes either way.

### GATE S2.1

Report the `valid_years` histogram, rule precision with CIs, revisit-pair agreement by rule, and propagated area per region. **Decision:** which rules are enabled, and their thresholds.

---

## S2.2 — Embedding drift

v1.1 embeddings move from year to year even where the ground has not changed. Different S1/S2 acquisitions, clouds, and the 2022 loss of Sentinel-1B all contribute. The model would read that drift as LCZ change.

### Measure

- **Stable pixels:** S2.1 rules R1/R2/R3, never an AI product.
- **Per region × year, report:**
  - mean per-channel shift against 2017;
  - the distribution of cosine/L2 distance to 2017;
  - the L2 norm distribution (v1.1's norm varies 7.9–36.1, so magnitude drift matters);
  - matched-channel r.
- **Watch** the 2021→2022 step and any region-specific steps (high-latitude winter, monsoon cloud).

### Mitigate, cheapest first, each kept only if it helps

1. **Per-year channel normalisation.** Compute the statistics on stable pixels only, so real change does not leak into them.
2. **Per-year linear alignment** onto 2017: a 128×128 affine map fitted on stable pixels, validated on **held-out** stable pixels (by block). Adopt it only if it lowers the held-out flip rate without lowering test B.
3. **Year jitter in training** (R4 below). The model sees the same place and label through several years of drift.
4. **Temporal-consistency loss** (R4b below).
5. **HMM** at inference (S2.4).

**Metric:** the flip rate on stable sites, per year pair, before and after each step, with the 2021→2022 pair reported separately.

---

## S2.3 — Temporal training rungs

These are added to the stage-1 recipe, one factor per rung:

| Rung | Adds | Runs |
|---|---|---|
| R4 | stability-extended windows: the sampler draws the year uniformly from `valid_years` (year jitter across years). WUDAPT consensus may pool votes across years where `valid_years` covers both; R6 targets may extend to earlier years in R2/R3 blocks | 3 |
| R4b | + temporal-consistency loss: for pixels the rule marks stable but which carry no label in the drawn year, a symmetric KL between predictions at two years of the window. It is label-free and pushes the model to ignore drift | 3 |

**Keep rule:** as stage 1 (test A, test B), **and** judged on the stable-site flip rate (S2.2). A rung may be kept for temporal stability at equal accuracy.

---

## S2.4 — Yearly maps

- **Inference:** as S1.5.2, run with the same checkpoint for every year 2017–2025 (D2), with `--save-probs`.
- **Ensemble and calibration:** as S1.5.2.
- **Temporal smoothing:** a new `src/utils/lcz_temporal.py`, a per-cell HMM decoded with Viterbi.
  - Emissions: the calibrated yearly probabilities.
  - Transitions: 17×17 with a heavy diagonal (stay probability 0.95–0.99, swept on stable-site flip rate against revisit-pair detection).
  - Off-diagonal mass only on plausible moves: natural → built, open → compact, low → mid/high rise. Reversals and built → natural get near zero.
  - After Viterbi, run the spatial Gaussian already in `lcz_smoothing.py`.
- **Rule:** single-epoch auxiliary inputs (GHS-BUILT-H, ETH canopy, aux-fusion channels, the offset corrector) stay out of the yearly path (D5).
- **Scope:** first the ~1,250 WUDAPT/So2Sat AOIs × 9 years. Go global only after S2.5 passes.

### GATE S2.4

Report culture-10 + test-B cities × 9 years (raw and smoothed), the HMM stay-probability sweep, and runtime per AOI-year.

---

## S2.5 — Evaluating the time series

Most of this needs **no new labels**:

| Measure | Data | Target |
|---|---|---|
| Per-year accuracy | test-B polygons at their label year; test A at 2017 and, on S2.1-stable patches, 2018–25 | no year more than ~0.03 kappa below 2017 |
| Per-city spread | as above | report alongside the pooled figure (per-city sd currently ≈ 0.18–0.23) |
| Spurious change | S2.1 stable sites | flip rate per cell-year, before and after HMM; 2021→22 reported separately |
| Change on revisit pairs | pairs with differing labels that S2.1 does not call stable | recall of labelled change (a small, free set) |
| So2Sat vs WUDAPT agreement | S1.2.1 diagnostic, re-reported on stable sites only | per city and author |
| Comparator change (D5) | GHSL/WSF built-up growth 2017→2025, Open Buildings Temporal | reported only; disagreement is not error by definition |

### The change sample (the one labelling task)

Change accuracy and change area cannot be checked without labels. Use the standard stratified design, Olofsson et al. (2014):

- **Strata:** from the map itself, e.g. "stable built", "stable natural", "natural → built", "built → built (other class)", by region group. Change strata are rare, so they are over-sampled.
- **Size:** a few hundred sites in total (≈ 50 per change stratum, fewer in stable strata), sized for the CI on user's accuracy of the change classes.
- **Reference:** Google Earth historical imagery at the start and end years, labelled "same LCZ / changed to …/ can't tell".
- **Outputs:**
  - user's and producer's accuracy per change stratum;
  - area-adjusted change estimates with CIs.
- Combine it with the ~200 stability-rule sites (S2.1) into one session.

---

## Carried over, lower priority

- **v1.1 patch-classifier track.** Rebuild E1–E3 on `tesserav1.1_global` (`build_wudapt_train_gpkg.py --output-name GeoTessera_v1.1_global --embedding-name tesserav1.1_global`) with a v1.1 teacher. It is needed anyway as the E3 veto teacher, the test-A reference, and the GATE S1 comparison.
  - Compare against v2-E3 ensemble-to-ensemble (`seed_ensemble.py --k`).
  - v2 cannot be the map product: it lacks global coverage for all years, and its basis differs (D0).
- **Other embeddings** (dClimate v1.1, OlmoEarth v1.3, v2). Benchmark-chapter comparisons on test A/B only; never part of the map (D0).
- **Label-shift correction per city** (BBSE/EM on predicted priors). Campaign doc open direction #1. Cheap and post-hoc.
- **lczkit / `lcz_labels` morphometric *labels*** (rule-based LCZ from form parameters). Their 2D parameters enter the plan only as auxiliary targets (R6). Rule-based labels need heights, which here would come from tags or AI rasters, so they stay dormant.
- **Demuzere noisy student** (`--pseudo-holdout-km 30` re-run). Benchmark chapter only (D5).

## Open decisions (for the author)

1. The source mix (start 40/50/10) and the OSM weight (0.2–0.3): fix by FCN8 sweep on CV, or by hand?
2. The fixed year for ablation A1 (2024 proposed; 2021 is the median WUDAPT label year).
3. The GATE S1 go criteria (proposed above): confirm or change before R0 runs.
4. Demuzere-based labels: confirm they stay out of the map recipe (D5).
5. The similarity threshold for admitting a coarse WUDAPT set (0.67 proposed): check how much area it moves to "unlabelled" before fixing it.
6. Stage 2: the R2 coverage threshold (0.90 or 0.95), after the manual check; and whether to accept per-year linear alignment if it helps flip rate but costs ≤ 0.01 test-B kappa.

## Budget (rough)

| Stage | Phase | Cost |
|---|---|---|
| 1 | S1.0–S1.2 | CPU; embedding extraction for the label years only (1–3 per AOI) |
| 1 | S1.3 | ~28 U-Net/FCN8 runs (incl. A1 and U-Net medium) + ~19 FCN8 CV/sweep runs |
| 1 | S1.5 | inference for ~1,250 AOIs × 1 year × 3 seeds, plus culture-10 at 2017 |
| 2 | S2.1 | CPU; OSM history extraction for ~1,250 AOIs |
| 2 | S2.2–S2.3 | CPU on cached stable-pixel embeddings + 6 runs |
| 2 | S2.4 | inference for ~1,250 AOIs × 9 years × 3 seeds, one tile read per pass |
| 2 | S2.5 | one labelling session: ~200 stability sites + a few hundred change sites |
