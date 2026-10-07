# eo-fm — PLAN: multitemporal LCZ map, 2017–2025

Written against `c4d4b95` on `design/global-splits` (2026-10-07). This is a separate track from
`PLAN-V3.md`. The benchmark chapter (Phases 2–4 there) keeps its own gates, and nothing here
blocks them.

**Deliverable:** one global LCZ map per year, 2017–2025, on Tessera v1.1. It is produced by a
**segmentation** model trained on three label sources:

- So2Sat LCZ42 (51 cities, imagery 2016–18);
- WUDAPT LCZ-Generator training areas (hundreds of cities, 2017–2024, variable quality);
- OSM/Overture, for a few classes and only as partial label sets.

Standing rules carry over from PLAN-V3:

- one factor per run;
- seeds per arm as stated;
- pre-register before launching;
- every result goes to `RESULTS.md`;
- stop at each GATE.

Contents:

- §0 decisions this plan fixes;
- §1 what is already in the repo;
- M0 embedding sign-off;
- M1 splits;
- M2 labels: sources, validity windows, and the OSM-history stability rule;
- M3 training: inputs, crops, sampling, models, epochs, run ladder;
- M4 embedding drift;
- M5 evaluation;
- M6 producing the yearly maps;
- M7 evaluating the time series;
- carried-over work, open decisions, budget.

---

## 0. Decisions this plan fixes

| # | Decision | Why |
|---|---|---|
| D1 | **Segmentation is the main pipeline.** The patch classifier stays as baseline, as the test-A reference, and as the E3 teacher. | About 60% of QC-passing WUDAPT polygons cannot contain a 320 m square, so only segmentation can learn from them. The gap after the TTA fix is ~0.06 kappa (U-Net 0.602–0.607 against classifier 0.6675 on the same patches), not ~0.10. |
| D2 | **One pooled model, applied to every year.** There are no per-year models and nothing to merge. Per-year predictions are combined only by seed averaging and the temporal HMM (M6). | Per-year models would each see a fraction of the labels, and each would have its own biases, which show up as spurious change. One model with one decision rule makes year-to-year differences attributable to the embeddings. |
| D3 | **The sample is (place, year).** Each sample is a 128 × H × W crop of *that year's* embedding, paired with a label valid *at that year*. | Labels are tied to the epoch they describe. A label is reused in another year only where M2's stability rule allows it. |
| D4 | **The model gets no year, Köppen, region or lat/lon input.** These are used only for sampling, stratification and reporting. | A year input lets the model learn year-specific priors, which become spurious change. Köppen and region inputs become shortcuts tied to the training cities, fail on unseen ones, and cause seams at zone boundaries. |
| D5 | **No AI-generated product is used as a filter, label, or stability mask.** That covers GHSL, WSF, Dynamic World, ESA WorldCover, Google Open Buildings (incl. Temporal), Microsoft footprints, the Demuzere 2022 LCZ map, and ETH canopy height. They may appear only as clearly labelled **evaluation comparators**. | Using them would make us inherit their errors and biases, and they would correlate with whatever they are compared against. See §M2.5. |
| D6 | **OSM is used in two ways only.** First, as low-weight **partial label sets** for a few classes. Second, through its **history**, as evidence that something *existed* by a date, which can extend a label's validity window. It is never used for the built classes LCZ 1–6, and absence in OSM is never read as evidence. | OSM is reliable for water and large land uses. It cannot tell compact from open, or low-rise from high-rise, without height tags. Before 2017 it is incomplete in most of the Global South. |
| D7 | **The split design is frozen before any M3 run.** This means `docs/global_split_design.md` and `src/design_global_splits.py`. | Any city used for model choice cannot later serve as a test city. |

---

## 1. Already in the repo (do not rebuild)

| Need | Where it lives |
|---|---|
| WUDAPT harmonisation: author collapse, QC weights, `w_time`, consensus bitmask | `lcz_wudapt` (`quality.py`, `consensus.py`, `qc.py`) |
| Culture-10 protection: forced to test, 1.3 km buffer, So2Sat wins | `lcz_wudapt/splits.py`, `src/build_wudapt_train_gpkg.py` |
| Teacher–student WUDAPT arms E1/E2/E3, veto rate by label year | `src/build_wudapt_train_gpkg.py` |
| Global block split, strata, distance matching, coverage floors, revisit pairs | `src/design_global_splits.py`, `docs/global_split_design.md`, `tests/test_design_global_splits.py` |
| Marginalised CE over label sets (class weights, confidence, adjacency smoothing) | `lcz_train/losses.py::marginalized_ce` |
| Anchor-pixel window sampling with erosion and class balancing | `lcz_train/datasets.py::PixelWindowDataset` (the window range is 96–256, so **relax to 64**) |
| Segmentation families (U-Net, FCN8, Attention U-Net, ResNet-U-Net) and the generic training loop | `src/models/`, `src/training/loop.py`, `src/semantic_segmentation.py` |
| Sliding-window ROI inference with Hanning blending, pooled to a target resolution | `infer_roi.py --target-res 100 --aggregate soft`, `coarsen_bakeoff.py` |
| Demuzere et al. 2020 per-class Gaussian smoothing | `utils/lcz_smoothing.py` |
| Seed ensembles | `src/seed_ensemble.py` |
| OA_w / kappa_w / OA_u / OA_bu (Bechtel 2020), per-city kappa | `training/lcz_metrics.py`, `training/evaluate.py` |
| Stability-mask stage (to be **rewritten**, see M2.4) | `lcz_labels/change_mask.py` |
| OSM tag → LCZ evidence | `docs/osm_lcz_tag_mapping.md` |

Current standing (W&B, culture-10 test, MobileNet-small, Tessera v2 2017):

- baseline 0.639 ± 0.022 (n=7);
- E3 0.679 ± 0.019 (n=10);
- E3 + `--min-epochs 30` 0.682 ± 0.018 (n=7).

The E-arms exclude culture-10 plus a buffer, so this gain holds for unseen cities.

The Demuzere noisy-student gain does not yet: 7.4% of its pool sits in culture-10 cities. It is also an AI product, so under D5 it is out of the map recipe.

---

## Phase M0 — Embedding product sign-off (blocking, CPU only, ~1 day)

The complete global v1.1 archive for 2017–2025 lives **outside** the `/tessera/v1.1` mount the
current checkpoints were trained on. The map needs one basis across nine years.

### Task M0.1 — Identity of the complete archive

Run `diagnostics/tessera_product_check.py` between the complete archive and `tesserav1.1_global`, for 2017, on So2Sat patch ids.

- **Identical** (matched-channel r ≈ 1, max abs diff at int8 quantisation): existing v1.1 checkpoints and caches stay valid. Register the archive in `datasets/registry.py` as the canonical `tesserav1.1_global` source.
- **Different basis** (as Task 1.5.1 found for the per-city download: r ≈ 0, linear-map R² 0.89): retrain every v1.1 model on the complete archive. The 0.89-R² map is **not** acceptable for production.

Also report, per year, the tile count and footprint coverage of So2Sat + WUDAPT AOIs.

Then fix the sidecar filter. `build_tile_index` drops NPY tiles that lack a `global_0.1_degree_tiff_all` sidecar, even though `tessera_grid_geometry` reproduces the sidecar geometry exactly (699/699 tiles). Make it fall back to the tile-name geometry, as the v2 branch does.

### Task M0.2 — Basis stability across years

Run the same checker year against year, on identical patch ids: 2017 vs 2018, 2017 vs 2021, 2017 vs 2025.

- Matched-channel r should be high: same weights, different inputs.
- A year with r ≈ 0 is a different inference run, and **stops the plan** for that year.

### Task M0.3 — First drift readout

This is a coarse check before M2's stability rule exists: on LCZ G pixels inside OSM water present in 2017 (rule R1, M2.4), report the per-year distribution of embedding distance to 2017, per region.

The full drift protocol is M4. Do **not** use GHS-BUILT-S as the "unchanged" mask here (D5).

### GATE M0

Report:

- the identity table;
- the per-year coverage table;
- the year-pair basis table;
- the water drift curve, with the 2021→2022 step called out (loss of Sentinel-1B).

**Decision:** the canonical archive, or a stop.

---

## Phase M1 — Freeze the splits (CPU, ~1 day on the cluster)

Full design and rationale: `docs/global_split_design.md`. Tool: `src/design_global_splits.py`.

The parts that constrain training:

| Split | What | Used for |
|---|---|---|
| **Test A** | So2Sat culture-10 cities, frozen | comparability with every past number |
| **Test B** | WUDAPT blocks chosen by the designer: 50 km complete-linkage blocks; strata region × Köppen main group; distance-matched to the 5,558 GUPPD urban areas (NNDM/kNNDM); coverage floors per class and year bin; ≤ 40% of any stratum held out | the headline global number; **aim ≥ 50 blocks** (kappa CI ≈ ±0.055) |
| **Val** | disjoint blocks, same design | early stopping and model choice |
| **CV** | 5 grouped folds over the training blocks | decisions the test sets cannot afford to see (FCN8 sweeps) |
| **Buffers** | 20 km around every test/val polygon; 1.3 km around culture-10 patches | no training crop may touch a buffer |
| **Quarantine** | `nbr_conflict` > 0.40; Tehran | excluded from all splits |
| **Revisit pairs** | the same ground labelled in two different years | the temporal reference (M5.4, M2.6) |

Stress tests are reported as transferability, not as the headline:

- LORO: leave one region out;
- LOKO: leave one Köppen main group out.

### Task M1.1 — Real run

Run `python src/design_global_splits.py` on the real WUDAPT gpkg with the Beck GeoTIFF.

Inspect `design_report.md` for:

- stratum coverage;
- distance KS against the target;
- class and year-bin floors;
- the forced and quarantined lists. Use full `{slug}__{SMOD_ID}` keys: a bare `lagos` also matches Lagos de Moreno.

### GATE M1

Commit `polygon_split.parquet`, `design_config.json` and `design_report.md`. **From here on, test B is read once per ladder rung (M3.6), never during tuning.**

---

## Phase M2 — Labels

### M2.1 Sources and how each enters the loss

Every label is a **set** of LCZ classes per pixel (a 17-bit bitmask), with a per-pixel weight and a validity window of years. One loss handles all sources: `marginalized_ce`, which scores `-log Σ_{c∈S} p_c`.

| Source | Label form | Weight | Validity window (before M2.4 extends it) |
|---|---|---|---|
| So2Sat (train cities) | hard class per 320 m patch, rasterised to 10 m; per-city gpkg/tif | 1.0 | 2017–2018 (imagery 2016–18) |
| WUDAPT (train blocks) | `lcz_wudapt` consensus bitmask: hard where `\|S\|=1`, coarse set for 2–3 | polygon `w = w_qc · w_acc · w_size` × consensus confidence; `w_time` = 1 once year-matched | `clip(label_year, 2017, 2025)` only |
| OSM / Overture | partial sets, M2.3 | 0.2–0.3 (sweep once) | 2024–2025 (the snapshot used) |

So2Sat stays authoritative inside its cities, and WUDAPT is admitted there only where So2Sat is sparse (`lcz_wudapt` leakage rule).

**WUDAPT is year-matched:** each polygon trains against the embedding of its own `label_year`, not 2017. The current E-arms pair 2019–24 labels with 2017 embeddings and let the teacher veto absorb real change. Year matching removes that confound. Keep the E3 teacher veto, rescored on year-matched embeddings. The teacher is our own So2Sat model, whose errors we measure, so D5 does not apply.

Caveat: `oa` falls with recency (0.768 in 2019 → 0.615 in 2023), so `w_time` and `w_acc` partly cancel. Report them jointly.

### M2.2 Validity windows

Every labelled polygon or block carries `valid_years`, a 9-bit mask over 2017–2025. It starts at the source window above. It is **extended only by M2.4**, and every extension is recorded with a reason code. The sampler (M3.4) draws a year only from `valid_years`.

### M2.3 OSM partial label sets (current snapshot)

The class lists below are bitmask *sets*. A pixel labelled `{A, B}` costs nothing whether the model says A or B.

| OSM evidence | Label set | Conditions |
|---|---|---|
| `natural=water`, riverbank/`water=*`, coastline polygons | {G} | erode 2 px; drop `intermittent=yes`, `tidal=yes`, wetlands |
| large `landuse=forest` / `natural=wood` | {A, B} | ≥ 4 ha, no buildings inside, erode 2 px |
| `landuse=farmland/meadow/grass`, `natural=grassland/heath` | {D} | ≥ 4 ha, no buildings inside |
| `landuse=industrial` with large footprints (median ≥ 1,000 m²) | {8, 10} | ≥ 4 ha |
| `natural=sand/beach`, dunes | {F} | ≥ 4 ha |
| `natural=bare_rock/scree`, `landuse=quarry` | {E} | ≥ 4 ha |
| `natural=scrub` | {C} | ≥ 4 ha, no buildings |

- **Never for LCZ 1–6, 7 or 9:** those need density and height, which tags do not give (`docs/osm_lcz_tag_mapping.md`).
- Minimum area matches the WUDAPT QC minimum (4 ha ≈ 400 px).
- Valid for 2024–25 only, unless M2.4 extends them.

### M2.4 OSM-history stability rule (no AI products)

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
| **R4 abstain** | in the surrounding 1 km cell, 2017 building count < 0.8 × 2025 count (OSM was not complete in 2017, so mapping and building cannot be separated) | no propagation; source window only |
| **Height guard** | for classes 1–6, without `building:levels`/`height` history or a heritage tag | propagate at most ±3 years from the label year |

**Direction.**

- **Forward propagation** (2017 label → later years) needs no change through 2025.
- **Backward propagation** (e.g. a 2023 WUDAPT label → 2017) is riskier. The 2017 snapshot must show the objects already existed, which is exactly what R1–R3 test. Low-rise → high-rise redevelopment on the same footprint is invisible to footprints, which is why the height guard exists.

**Optional physical vetoes.** These come from measurements, not ML products, and can only **remove** stability, never add it:

- Sentinel-1 VV/VH annual-median backscatter change;
- NDVI/NDBI annual-median difference from Sentinel-2;
- Tessera embedding distance above the p95 of R1-water drift for that region and year.

**Output.** Write a `stability.parquet` beside the label gpkg: one row per polygon or block, with `valid_years` (9 bits), the rule that fired, the inputs to that rule, and veto flags. The sampler consumes it.

**Implementation.** Rewrite `lcz_labels/change_mask.py`:

- keep its graceful-degrade default (no evidence → source window only);
- drop the Google-Temporal primary source (D5);
- keep the `export_google_temporal_ee` path only for the comparator in M7.

Unit-test each rule on synthetic histories:

- a mapped-late building must not propagate;
- a reservoir must not propagate;
- R4 must abstain.

**Expected coverage.** Propagation will be high in Europe, North America and East Asian old cores, and low in Africa and South Asia, where 2017 OSM was sparse. That is correct behaviour: those places fall back to year-matched labels only. Report the propagated area per region × year so the imbalance is visible.

### M2.5 No-AI-filter policy (D5) in practice

| Product | Status |
|---|---|
| GHSL (BUILT-S/H/C), WSF, WSF-Evolution | comparator only (M7); never a mask |
| Google Open Buildings (incl. 2.5D Temporal), Microsoft footprints | comparator only; not a building source for OSM rules |
| Dynamic World, ESA WorldCover | comparator only |
| Demuzere 2022 global LCZ map, noisy-student pseudo-labels from it | out of the map recipe; kept as a benchmark-chapter result. Note it was trained on LCZ-Generator TAs up to Dec 2021, so it is not independent of WUDAPT anyway |
| ETH canopy height, GHS-BUILT-H aux channels | out of the production path (also single-epoch, so they would freeze one year's structure onto all years) |
| Own So2Sat-trained teacher (E3 veto) | allowed: our model, our measured errors |
| Sentinel-1/2 band statistics (backscatter, NDVI, NDBI) | allowed as vetoes only |

### M2.6 Validating the stability rule (not every place)

We validate the *rule*, then trust it where it fires.

1. **Revisit pairs** (`revisit_pairs.parquet` from M1). For pairs the rule calls stable, two-year label agreement should approach same-year inter-annotator agreement (0.71 pooled by area; per-city where n allows). For pairs it abstains on, agreement should be lower. Report both, by rule and region.
2. **Manual check of ~200 sites.** Use Google Earth historical imagery, 2017 against 2024, stratified by rule (R1/R2/R3) × region × class group. One person labels "unchanged / changed / can't tell".
   - Targets: precision ≥ 0.95 for R1, ≥ 0.90 for R2 and R3.
   - Report Wilson 95% CIs. With ~200 sites the overall CI is ±0.04 at p = 0.90; per stratum it is wider, so stratify to make every rule × region cell ≥ 10.
3. **Propagated area per region × year**, as above.

If R2 misses its precision target, tighten the coverage ratio and the height guard before giving up on it. R1 and R3 carry most of the value for the natural and large-land-use classes either way.

### GATE M2

Report:

- label inventory per source × class × year, before and after propagation;
- the `valid_years` histogram;
- rule precision with CIs;
- revisit-pair agreement by rule;
- propagated area per region.

**Decision:** which rules are enabled, and their thresholds.

---

## Phase M3 — Training (segmentation)

### M3.1 What goes in

| | Content | Shape / dtype |
|---|---|---|
| **x** | v1.1 embedding of year *y*, dequantised (int8 × scale), per-channel normalised with train-set statistics stored in the checkpoint (repo convention); per-year statistics only if M4 calls for them | 128 × H × W, float32, H = W ∈ {64, 96} |
| **target** | per-pixel 17-bit label set valid at *y* | H × W, int32 bitmask |
| **valid** | labelled ∧ eroded 1 px ∧ embedding not nodata ∧ outside every held-out buffer | H × W, bool |
| **weight** | source weight × polygon/consensus confidence | H × W, float |

- **Not inputs:** year, Köppen, region, lat/lon (D4).
- **Not loaded:** all nine years at once. A sample is one place at one year (D3). The year is drawn from the label's `valid_years`, which is what "year jitter" means here.

### M3.2 Crops

- **Size:** 64 px (640 m) for WUDAPT and OSM anchors; 96 px (960 m) where So2Sat is dense. Both divide by 16, so U-Net depth 4 works.
- **No pre-cut tiles.** Crops are cut on the fly around anchor pixels from per-year mosaics.
- **Keep a crop only if:**
  - it has **≥ 200 labelled pixels after 1-px erosion** (≈ 2 ha; ≈ 5% of 64², ≈ 2% of 96²). The threshold is absolute, not a percentage, because WUDAPT's minimum polygon is 4 ha ≈ 400 px and a fraction would penalise the larger crop;
  - it has **≥ 75% valid embedding pixels**;
  - it touches **no** held-out buffer.
- **No maximum label fraction.** Fully labelled crops are fine: the unlabelled context is what the model uses, not what it is scored on.
- **At most ~5 crops per polygon per epoch**, so the big So2Sat cities and Wuhan's 44k polygons cannot dominate.

### M3.3 Loss

- `marginalized_ce` over label sets.
- Class weights: sqrt inverse frequency of train-set pixel counts, where a set-labelled pixel counts 1/|S| toward each member.
- Per-pixel weight from M3.1.
- Optional adjacency smoothing (`smoothing_eps`), Stewart & Oke morphology; off by default.
- **Optional temporal-consistency term** (rung R4b): for pixels the rule marks stable but which carry no label in the drawn year, a symmetric KL between predictions at two years of the window. It is label-free and pushes the model to ignore drift.

### M3.4 Sampling (hierarchical, per crop)

Draw each crop in five stages:

1. **Source**, with a fixed mix. Start at 40 / 50 / 10 So2Sat / WUDAPT / OSM and sweep once with FCN8.
2. **City / AOI** ∝ area^α, with α ≈ 0.3–0.5, so small and Global-South cities get seen.
3. **Class** ∝ inverse frequency^β, with β ≈ 0.5.
4. **Year**, uniform over that label's `valid_years`.
5. **Anchor pixel**, uniform inside eroded pixels of that class, then the crop is centred on it.

Extend `PixelWindowDataset` or add `src/datasets/multisource_seg.py`. Either way it must:

- read per-year mosaics lazily;
- keep the seeded-stream semantics (one RNG per worker);
- allow 64 px windows;
- log realised source, class, region and year frequencies per epoch.

### M3.5 Models, epochs, seeds

| Model | Role |
|---|---|
| **U-Net small** (depth 3, 32 features) | main model, 3 seeds per rung |
| **FCN8 small** | cheap: LR sweep, source-mix sweep, 5-fold CV. **First re-score the existing FCN8 rows with the fixed TTA**, because `data/seg_metrics.csv` used the broken one |
| **U-Net medium** (depth 4) | once, at the best rung after WUDAPT is added, to check whether more data now rewards capacity (it did not for So2Sat alone) |
| MobileNet-small patch classifier | baseline, test-A reference, E3 teacher on v1.1 |
| Shallow CNN | classification-only in the registry; not planned for segmentation |

- **Epoch:** 20,000 crops.
- **Early stopping:** on **val polygon-level kappa_w** (M5.2 scoring applied to val), patience 10, min 10, max ~50 epochs.
- **LR:** Adam + cosine (`training/loop.py`). Sweep {1e-4, 3e-4, 1e-3} with FCN8 small on rung R1, one seed each, then fix it.
- Report the epoch of best val for every run. Classification peaked during warm-up (PLAN-v3 revC, Task 2.1c), and segmentation should be checked for the same.

### M3.6 Run ladder (one factor per rung)

| Rung | Adds | Runs |
|---|---|---|
| R0 | re-score existing FCN8 / U-Net with fixed TTA; v1.1 So2Sat seg baseline at 2017 | eval only + 3 |
| R1 | So2Sat only, 2017 + 2018 embeddings (year jitter within the So2Sat window) | 3 + LR sweep 3 |
| R2 | + WUDAPT, year-matched, consensus sets, QC weights | 3 + U-Net medium 1 |
| R3 | + OSM partial sets (weight 0.2–0.3) | 3 |
| R4 | + stability-extended windows (M2.4), i.e. year jitter across years | 3 |
| R4b | + temporal-consistency loss | 3 |

**Keep rule (pre-register):** a rung is kept if test-B mean city kappa rises by more than its bootstrap CI half-width, or by ≥ 0.01, **and** test A does not drop by more than 0.01.

R4 and R4b are also judged on the stable-site flip rate (M4). They may be kept for temporal stability at equal accuracy.

Source-mix and α/β sweeps run on CV folds with FCN8, never on test.

### GATE M3

Report the ladder table: test A, test B mean ± CI, val, and stable-site flip rate, by seed. **Decision:** the production recipe.

---

## Phase M4 — Embedding drift

v1.1 embeddings move from year to year even where the ground has not changed. Different S1/S2 acquisitions, clouds, and the 2022 loss of Sentinel-1B all contribute. The model would read that drift as LCZ change.

### M4.1 Measure

- **Stable pixels:** M2.4 rules R1/R2/R3, never an AI product.
- **Per region × year, report:**
  - mean per-channel shift against 2017;
  - the distribution of cosine/L2 distance to 2017;
  - the L2 norm distribution (v1.1's norm varies 7.9–36.1, so magnitude drift matters);
  - matched-channel r.
- **Watch** the 2021→2022 step and any region-specific steps (high-latitude winter, monsoon cloud).

### M4.2 Mitigate, cheapest first, each kept only if it helps

1. **Per-year channel normalisation.** Compute the statistics on stable pixels only, so real change does not leak into them.
2. **Per-year linear alignment** onto 2017: a 128×128 affine map fitted on stable pixels, validated on **held-out** stable pixels (by block). Adopt it only if it lowers the held-out flip rate without lowering test B.
3. **Year jitter in training** (M3.6 R1/R4). The model sees the same place and label through several years of drift.
4. **Temporal-consistency loss** (R4b).
5. **HMM** at inference (M6.4).

**Metric:** the flip rate on stable sites, per year pair, before and after each step, with the 2021→2022 pair reported separately.

**GATE M4** sits inside the M3 ladder: which normalisation the production checkpoint uses.

---

## Phase M5 — Evaluation

### M5.1 Test A — So2Sat culture-10, 2017

Pool the segmentation probabilities into each 320 m patch (soft mean, purity and buffer filters off, 83.8% patch coverage), take the argmax, and report kappa / OA / OA_w.

This is directly comparable to the patch classifier (0.6675 on the same patches) and to every E-arm number. Report both models side by side.

### M5.2 Test B — held-out WUDAPT blocks, at each polygon's `label_year`

- **Per polygon:** majority of predicted pixels after 1-px erosion. Score it against the hard label, and separately as set-correct (prediction ∈ consensus set).
- **Per city:** kappa and OA_w. Headline = **mean across cities with a city-bootstrap 95% CI**. Also report the pooled Bechtel-2020 kappa_w / OA_u / OA_bu.
- Report per region and per Köppen group too, from the same predictions.

### M5.3 Validation and CV

- Val, scored as M5.2, drives early stopping.
- The 5 grouped folds score the FCN8 sweeps.
- LORO and LOKO are run once on the final recipe.

### M5.4 Time

1. **Revisit pairs** (the M1 output), at their two years:
   - agreement where the two labels agree;
   - detection where they differ and M2.4 does not call the site stable.
2. **Test A re-scored at 2018–2025**, on patches M2.4 marks stable. Accuracy should hold; any decline is drift.
3. **Stable-site flip rate** per year pair (M4).

### M5.5 Comparators (D5, reported, never used)

Run the same test-B polygons through the Demuzere 2022 LCZ map, and the same stable sites through GHSL change. Report them in a separate table labelled "external products".

---

## Phase M6 — Producing the yearly maps

### Task M6.1 — Pooling and resolution, chosen once

Run `coarsen_bakeoff.py` on the M3 winner, `--target-res 320 100`, on culture-10. The expected answer is 100 m `--aggregate soft`; confirm it.

### Task M6.2 — Inference

- `infer_roi.py` with 128 × 128 sliding windows, Hanning blending, `--target-res 100 --aggregate soft`.
- The same checkpoint is run for every year (D2).
- Add `--save-probs` to write the `(17, H, W)` volume, uint8-scaled, beside the label GeoTIFF.
- Run the three seeds in one pass per tile, so each tile is read once.

### Task M6.3 — Ensemble and calibration

- Average the softmax over the seeds.
- Fit temperature scaling on val. If calibrated numbers are reported, fit it leave-one-city-out (`ensemble_stacking.py --city-holdout`). Earlier models were under-confident (T 0.46–0.98), which would distort the HMM.

### Task M6.4 — Temporal smoothing

New `src/utils/lcz_temporal.py`: a per-cell HMM decoded with Viterbi.

- Emissions: the calibrated yearly probabilities.
- Transitions: 17×17 with a heavy diagonal (stay probability 0.95–0.99, swept on stable-site flip rate against revisit-pair detection).
- Off-diagonal mass only on plausible moves: natural → built, open → compact, low → mid/high rise. Reversals and built → natural get near zero.
- After Viterbi, run the spatial Gaussian already in `lcz_smoothing.py`.

**Rule:** single-epoch auxiliary inputs (GHS-BUILT-H, ETH canopy, aux-fusion channels, the offset corrector) stay out of the yearly path (D5).

### Scope

First pass: the ~1,250 WUDAPT/So2Sat AOIs × 9 years. Go global only after M7 passes.

### GATE M6

Report:

- culture-10 + test-B cities × 9 years, raw and smoothed;
- the HMM stay-probability sweep;
- runtime per AOI-year.

---

## Phase M7 — Evaluating the time series

| Measure | Data | Target |
|---|---|---|
| Per-year accuracy | test-B polygons at `label_year`; test A at 2017 and, on stable patches, 2018–25 | no year more than ~0.03 kappa below 2017 |
| Per-city spread | as above | report alongside the pooled figure (per-city sd currently ≈ 0.18–0.23) |
| Spurious change | M2.4 stable sites | flip rate per cell-year, before and after HMM; 2021→22 reported separately |
| Detected change | revisit pairs with differing labels; the M2.6 manual sites marked "changed" | recall of labelled change; precision on the manual sample |
| Comparator change (D5) | GHSL/WSF built-up growth 2017→2025, Open Buildings Temporal | reported only; disagreement is not error by definition |
| Area estimates | stratified probability sample on historical VHR (Olofsson et al. 2014) | later; needs manual labelling, plan once the first three rows pass |

---

## Carried over, lower priority

- **v1.1 patch-classifier track.** Rebuild E1–E3 on v1.1 (`build_wudapt_train_gpkg.py --output-name GeoTessera_v1.1_global --embedding-name tesserav1.1_global`) with a v1.1 teacher. It is needed anyway as the E3 veto teacher and the test-A reference.
  - Compare against v2-E3 ensemble-to-ensemble (`seed_ensemble.py --k`).
  - v2 cannot be the map product: it lacks global coverage for all years, and its basis differs.
- **Label-shift correction per city** (BBSE/EM on predicted priors). Campaign doc open direction #1. Cheap and post-hoc.
- **lczkit / `lcz_labels` morphometric labels** (OSM/Overture height-to-width, building surface fraction). These are a current-snapshot source. Through M2.4 they could span years in R2 blocks. Leave dormant until M7 passes.
- **Demuzere noisy student** (`--pseudo-holdout-km 30` re-run). Benchmark chapter only (D5).

## Open decisions (for the author)

1. The source mix (start 40/50/10) and the OSM weight (0.2–0.3): fix by FCN8 sweep on CV, or by hand?
2. R2 coverage threshold, 0.90 or 0.95. Decide after the M2.6 manual check.
3. Whether to accept per-year linear alignment (M4.2 step 2) if it helps flip rate but costs ≤ 0.01 test-B kappa.
4. Demuzere-based labels: confirm they stay out of the map recipe (D5).

## Budget (rough)

| Phase | Cost |
|---|---|
| M0, M1, M2 | CPU; OSM history extraction for ~1,250 AOIs; ~1 day of manual checks (M2.6) |
| M3 | ~25 U-Net/FCN8 runs + ~15 FCN8 CV/sweep runs |
| M4 | CPU on cached stable-pixel embeddings |
| M6 | inference for ~1,250 AOIs × 9 years × 3 seeds, one tile read per pass |
| M7 | CPU, except the VHR sample |
