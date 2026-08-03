# Global LCZ Classification — Experimental Campaign Report (June–July 2026)

*(A plain-language companion version is `global_lcz_campaign_2026-07_plain.md`.)*

**Task**: 17-class Local Climate Zone classification of So2Sat-LCZ42 v4 patches (320 m) from
satellite foundation-model embeddings, under the **global split** (train cities disjoint from
the 10 evaluation cities). Primary metric: Cohen's kappa on the test split.

## Headline results

| | Test kappa | OA | Macro-F1 | Protocol |
|---|---|---|---|---|
| Best single model (`student-noisy-v3`) | **0.6497** | 0.6794 | 0.5656 | clean |
| Best ensemble, fully honest (LOCO-weighted) | **0.6871** | 0.7154 | 0.6038 | leave-one-city-out weight fit |
| Best ensemble, mild caveat (weighted [.20/.70/.10]) | 0.6923 | 0.7202 | 0.6065 | 2 weight DoF fit on val |

Campaign ladder: single model 0.619 → 0.6497 (+3.1 pts, semi-supervised noisy-student);
ensemble 0.6504 → 0.6871/0.6923 (+3.7/+4.2 pts, cross-embedding ensembling + weight fitting).

Three levers were tested and **definitively lost** — capacity, per-city test-time adaptation,
and learned stacking — each with a diagnosis worth keeping (§6, §8, §9).

## 1. Setup

- **Data**: So2Sat-LCZ42 v4, ~400k patches. Global split: training cities / validation and
  testing drawn from the **same 10 held-out cities** (Guangzhou, Jakarta, Moscow, Mumbai,
  Munich, Nairobi, San Jose, Santiago, Sydney, Tehran) — median test-patch → nearest-val-patch
  distance is **2.55 km**, which matters for evaluation honesty (§7).
- **Embeddings** (one model per embedding, same recipe):
  Tessera v1.1 global (128 ch, 10 m) · AlphaEarth coop (64 ch, 10 m) · Embedded Seamless
  (13→72 ch, 30 m). Ensemble tables use the 23,878 val / 23,858 test patches covered by all
  three sources.
- **Architecture**: timm `resnet34` (`--family resnet --preset small`, stem surgery for 32×32
  inputs). Capacity was tested and rejected: resnet101/152 do **not** beat resnet34 here.
- **Recipe ("opt3")**: batch 256, AdamW-style wd 1e-3, lr 5e-4 + 3-epoch warmup + cosine,
  mixup 0.4, label smoothing 0.1, `sqrt_inv_freq` class weights, monitor `val_kappa`,
  dihedral TTA at eval, max 50 epochs / patience 10.
- **Teachers (pre-campaign bests)**: tessera `opt3-lr5e-4-warmup3` **0.619**, coop
  `wandering-firefly-267` **0.513**, seamless `good-flower-268` **0.506**.

## 2. Phase 0 — squeezing the labeled data (2026-07-02)

All single-model levers on labeled data **lost** to opt3 (0.619):
multi-embedding feature fusion 0.5974 · sqrt-freq batch sampler 0.6099 · logit adjustment
0.5541. Conclusion: loss-level class weighting already saturates imbalance handling; adding a
second imbalance mechanism double-corrects.

The only Phase 0 winner: **softmax-average ensemble across embeddings = 0.6504** (tessera +
coop + seamless teachers). Errors across embedding families are substantially decorrelated.

## 3. Phase 1 — noisy-student SSL with Demuzere weak labels (2026-07-03 → 06)

Pipeline (`sample_unlabeled_patches.py` → `extract_so2sat_embeddings.py --splits unlabeled` →
`generate_pseudo_labels.py` → `patch_classification.py --pseudo-gpkg`):
**286,207 unlabeled 320 m patches** sampled inside the Tessera-2017 tile footprints, weakly
labeled by the Demuzere et al. 2022 global 100 m LCZ map (majority vote over the ~3×3 px
footprint, purity ≥ 0.65, map probability ≥ 50, no So2Sat overlap). Pseudo-label acceptance:
**agree** (teacher top-1 = Demuzere ∧ conf ≥ 0.7 → loss weight 0.5) or **rare-relax**
(rare class ∧ purity ≥ 0.75 ∧ teacher p ≥ 0.2 → weight 0.3); pseudo items train with
weighted CE alongside the labeled set.

| Iteration | Teacher | min-conf | Kept (rate) | Test kappa |
|---|---|---|---|---|
| teacher (opt3) | — | — | — | 0.619 |
| student-noisy-v1 | opt3 | 0.7 | 67,392 (23.6%) | 0.6419 |
| student-noisy-v2 | v1 | 0.8 | 22,499 (7.9%) | 0.6320 ✗ |
| **student-noisy-v3** | v1 | 0.7 | 89,320 (31.2%) | **0.6497** |
| student-coop-v1 (coop self-teacher) | coop 0.513 | 0.7 | 16,234 (11.4%) | 0.5218 |

Findings:
- **v2's failure is a calibration lesson**: a mixup + label-smoothed teacher has a softmax
  ceiling ≈ 0.9, so an absolute threshold of 0.8 is far stricter than intended — keep-rate
  collapsed and the class mix flipped rare-heavy. Thresholds must be set relative to the
  teacher's calibration (all three final models are *under*-confident: fitted temperatures
  0.92 / 0.46 / 0.98).
- **Iteration gains diminish and students converge toward the ensemble consensus**: the
  ensemble gain over the solo model shrank v1 → v3 (+2.7 → +2.0 pts), and v3 absorbed
  seamless's complementary signal (v3+coop pair 0.6701 ≈ v3 3-model 0.6696). Iteration 4 was
  vetoed on this basis.
- **A weak teacher gains little from self-training**: the coop round (+0.9 solo) did not carry
  into the ensemble (equal-weight unchanged) — self-teaching on agreement-filtered labels
  moves a model toward its own consensus.
- **SSL cannot rescue classes the weak-label pool lacks**: LCZ 7 had 30 candidates globally
  and 0 kept pseudo-labels.

## 4. Final ensemble (v3 + coop-v1 + seamless)

On 23,858 aligned test patches, TTA, probs cached in `dl/ensemble_coopv1/`:

| Method | Kappa | OA | Macro-F1 |
|---|---|---|---|
| solo tessera-v3 / coop-v1 / seamless | 0.6497 / 0.5153 / 0.5059 | | |
| equal-weight average | 0.6691 | 0.6983 | 0.5978 |
| equal-weight, temperature-calibrated | 0.6824 | 0.7109 | 0.6056 |
| weighted [.20/.70/.10] (2 DoF on val) | **0.6923** | 0.7202 | 0.6065 |
| weighted, leave-one-city-out (fully honest) | **0.6871** | 0.7154 | 0.6038 |
| stacked LR (fit on val) | ~~0.7756~~ | | **inflated — see §7** |
| stacked LR, leave-one-city-out | 0.6121 | 0.6461 | 0.5167 |

The lopsided [.20/.70/.10] weights are mostly **implicit calibration**: after per-model
temperature scaling (T fit on val by NLL: 0.92/0.46/0.98) the optimum becomes balanced
[.35/.40/.25] at 0.6903 — same optimum, two routes. Coop carries heavy weight because it is
severely under-confident *and* the most complementary member.

## 5. Honest-evaluation audit (leave-one-city-out)

Because val and test contain the **same 10 cities**, anything fit on val with enough capacity
can read city-conditional confusion structure back off test. The audit protocol: fit
weights/stacker on val patches of 9 cities, evaluate on test patches of the held-out city,
pool (`ensemble_stacking.py --city-holdout`; base models never saw any of the 10 cities, so
only the combiner needs the LOCO treatment).

- **Weighted averaging survives**: LOCO 0.6871 vs 0.6923 plain — the ~0.5 pt gap is the price
  of the 2 DoF, and the fitted weights are stable across folds ([.2/.7/.1] in 7/10).
- **Stacking dies**: LOCO stacked-LR 0.6121 — *worse than equal-weight averaging*, losing in
  8/10 cities. The 0.7756 headline was essentially all city-identity leakage. Honest stacking
  on this benchmark would require out-of-fold base-model predictions over training cities.

Per-city LOCO-weighted kappa: Munich 0.90 · Jakarta 0.80 · San Jose 0.80 · Moscow 0.70 ·
Sydney 0.68 · Guangzhou 0.63 · Tehran 0.62 · Mumbai 0.62 · **Nairobi 0.47 · Santiago 0.47**.
The spread is the domain-shift signature that motivated Phase 2.

## 6. Phase 2 — per-city test-time adaptation (NEGATIVE, 2026-07-06)

Hypothesis: models carry train-city BatchNorm statistics that are wrong for unseen cities.
Test: per-(model, city) **AdaBN** (BN running-stat re-estimation) and **TENT** (entropy
minimization on BN affine params), adapting only on the city's *val-split inputs* (no labels,
no test patches) and evaluating on its test patches (`tta_city_adapt.py`, `run_phase2_tta.sh`).

| Test kappa | Unadapted | AdaBN | TENT |
|---|---|---|---|
| solo tessera-v3 | **0.6497** | 0.5982 | 0.6084 |
| solo coop-v1 | **0.5153** | 0.4866 | 0.4874 |
| solo seamless | **0.5059** | 0.4709 | 0.4813 |
| LOCO-weighted ensemble | **0.6871** | 0.6090 | 0.6164 |

Per-city it *does* help where shift is covariate-shaped — Tehran 0.561 → 0.677 (+11.6),
Munich +2.6 — but craters Santiago (−16) and Jakarta (−10). **Diagnosis**: the BN statistics
moved enormously (first-layer |Δmean| ≈ 1.2), but per-city statistics absorb the city's LCZ
class mix; So2Sat's city shift is heavily **label shift**, which AdaBN/TENT mis-correct.
Train-pooled BN stats, co-adapted with the weights, win on net. The DANN follow-up gate was
therefore not triggered.

## 7. Quarantined numbers — do not report

- **Stacked-LR 0.7756**: city-identity leakage (§5).
- **`--orig-test` runs at 0.75–0.84**: likely autocorrelation-inflated (grid train/val near
  original test patches).
- **Per-city grid-split results** (e.g. London tables): spatial-autocorrelation-inflated;
  not comparable to global-split numbers.

## 8. Lessons

1. **Cross-embedding ensembling is the single biggest lever** (+2–4 pts at every stage);
   every within-model lever (capacity, fusion, sampler, logit adjustment) lost.
2. **Weak-label SSL delivers** (+3.1 pts) but self-distillation converges toward consensus —
   both across iterations and into the ensemble — so budget at most 2–3 rounds.
3. **Confidence thresholds are meaningless without calibration**: mixup + label smoothing
   caps teacher confidence ≈ 0.9 and leaves models under-confident (T as low as 0.46);
   calibrate or threshold relative to the teacher's ceiling.
4. **Audit any val-fit combiner for split leakage**: with val/test sharing cities, a
   leave-one-city-out re-fit costs minutes and here it flipped the conclusion entirely.
5. **Diagnose the shift before adapting to it**: the per-city spread is label-shift-shaped;
   statistics-matching TTA makes it worse. Nairobi/Santiago are a data problem, not an
   optimization problem.

## 8b. Phase 3 — auxiliary structural data (GHSL / canopy height / OSM), 2026-07-09

Motivation: ~50–58% of the residual ensemble errors sit on class pairs whose *defining*
discriminant is structural (built fraction 3↔6/6↔9, function 8↔10, building height 2↔3/4↔5,
canopy height C↔D) — information a 10 m 2-D optical embedding only infers indirectly.

**Features** (13 per-patch zonal scalars, `extract_aux_features.py` + `extract_osm_features.py`,
parquets in `data/`): GHS-BUILT-H ANBH 2018 (mean/p90/frac>10 m), GHS-BUILT-S 2020
(built fraction, non-residential share), ETH 10 m canopy height 2020 (mean/p90/std/cover>3 m),
OSM landuse fractions (industrial/commercial/residential/any). Aux tiles in
`${DATA_DIR}/input/aux_struct/`; GEE registry entries in `datasets/downloaders.py`.

**Combiner** (`ensemble_stacking.py --aux-parquet`, all under LOCO): a *frozen-pathway offset
corrector* — logits = log p_weighted + A·aux + b, correction applied only where ensemble
max-prob < τ; A (17×13), λ, τ fit strictly on the 9 fit-cities per fold.

| Method (all fully honest LOCO) | Kappa | OA | Macro-F1 |
|---|---|---|---|
| weighted ensemble (§5 baseline) | 0.6871 | 0.7154 | 0.6038 |
| aux-only LR (13 scalars, no FM model!) | 0.5785 | 0.6166 | 0.4521 |
| free LR on [log-probs ⊕ aux] | 0.6584 | — | — | 
| **offset corrector, confidence-gated** | **0.7055–0.7081** | 0.7320–0.7343 | 0.6129–**0.6227** |

**= +1.8–2.1 kappa over the honest baseline for zero training runs** (range = τ/λ grid width:
the finer grid picks stricter gates τ 0.15–0.2, trading ~0.3 pooled kappa for strictly
non-negative per-city deltas and the best macro-F1). New honest state of the art.

Findings:
1. **Aux features carry large, city-transferable signal**: 13 scalars alone beat the entire
   seamless FM model LOCO (0.579 vs 0.506); per-class means reproduce the LCZ height ladders
   (LCZ1>2>3: 30/16/9 m; LCZ4>5>6: 20/14/7.5 m) and canopy structure (A dense 24 m >
   B scattered 12 m; C bush 1.4 m vs D low plants 3.2 m).
2. **The carrier matters**: any full-rank LR on probs collapses under LOCO (control without aux
   0.6167 ≈ dead stacker §5); freezing the probs pathway at coefficient 1 is what rescues it.
3. **Aux helps exactly where the FM embeddings fail**: Nairobi +8.0, Tehran +5–9, Santiago +2–4,
   Moscow, Jakarta — including both §5 "data problem" cities; confidence gating removes the
   damage to strong cities (Munich).
4. Canopy resolves C↔D nearly purely (180 fixed / 47 new); 6↔9 and 8↔10 positive;
   **4↔5 and 3↔6 do NOT yield** to 100 m zonal stats post-hoc — they need spatially resolved
   aux channels at training time.

Results JSONs: `dl/ensemble_coopv1/ensemble_3models_test/stacking_results_auxgate_*.json`
(`_full_confgated{,_v2,_v3}` = coarse/mid/fine gate grids).

**Implication**: this un-gates training-time integration — (a) aux as extra 10 m input
channels (few, physically orthogonal — the §2 fusion negative concerned redundant learned
embeddings), (b) aux-consistency-gated pseudo-labels to finally supply LCZ 7/10/1 (§9
rare-class direction). OSM stays combiner/supervision-side only (completeness ≈ city identity).

### 8c. Stage 2a — aux as input channels (`aux-fusion-v1`, 2026-07-12)

Trained tessera(128) ⊕ aux(4: ANBH/50, built-frac, nres-frac, canopy/50) = 132 ch through the
existing multi-source fusion path, plain opt3 recipe (no SSL), global split. New `aux_struct`
EMBEDDING_REGISTRY source; GHSL→canopy merge precomputed once per tile into tiled+DEFLATE
GeoTIFFs (`precompute_aux_tiles.py`, `aux_struct/merged_aux/`) after a per-patch-reproject
handler OOM-crashed the extractor and a striped/uncompressed first cut ran at ~10 patch/s.

| Model | Recipe | Test kappa | OA | Macro-F1 |
|---|---|---|---|---|
| tessera-only opt3 (baseline) | plain | 0.619 | | |
| **aux-fusion-v1** (tessera+aux) | plain | **0.6310** | 0.6620 | 0.5682 |
| student-noisy-v3 (tessera-only) | +SSL | 0.6497 | | |

**Single-model: +1.2 kappa** — input fusion of *orthogonal physical* bands helps, unlike the
§2 negative (redundant learned embeddings 0.5974). ckpt `dl/aux-fusion-v1/`, wandb yo26w6fu.

**But it is redundant with the corrector at the honest-ensemble level.** 4-model LOCO on the
aligned 23,717-patch universe (`ensemble_eval.py` extended for fused specs; `dl/ensemble_aux4/`):

| Method (honest LOCO, same universe) | 3-model | 4-model (+aux-fusion) |
|---|---|---|
| weighted ensemble (no aux features) | 0.6857 | **0.6918** (+0.6) |
| **+ gated corrector** | **0.7018** | **0.7023** (+0.05) |
| macro-F1 (with corrector) | 0.6184 | 0.6278 (+0.9) |

As a raw member aux-fusion adds decorrelated signal (+0.6 at the weighting level), but the
gated corrector already injects the same structural features, so on top of it the gain vanishes
(+0.05 kappa; only macro-F1 improves +0.9). **Lesson: aux-as-input and aux-as-corrector are two
routes to the same ~0.702 honest kappa — the zero-training corrector banks essentially all of
it.** Training aux into the model is worth it only for a stronger single model or the small
rare-class F1 bump, not for headline kappa.

## 9. Open directions (not pursued, in rough order of promise)

- **Label-shift-aware adaptation**: estimate each city's class prior from model predictions
  (EM / BBSE-style) and correct the logits — attacks exactly the failure mode §6 exposed.
- **Training-time domain adaptation (DANN)** across training cities — gated off by the TTA
  negative for statistics-based variants, still plausible for class-conditional ones.
- **Honest stacking** via out-of-fold base predictions over training-city folds (expensive:
  retraining base models per fold).
- **Rare-class supervision**: LCZ 7/1/10/15 need labeled data or a richer weak-label source;
  no amount of SSL on the current pool helps (§3).

## Appendix — artifacts

| Item | Path |
|---|---|
| Best single checkpoint | `dl/student-noisy-v3/resnet_small_GeoTessera_v1.1_global_global-best.pt` |
| Other members | `dl/student-coop-v1/resnet_small_AlphaEarthCoop_global-best.pt`, `dl/good-flower-268/resnet_small_EmbeddedSeamless_global-best.pt` |
| Ensemble probs + all combiner results | `dl/ensemble_coopv1/ensemble_3models_{val,test}/` (+ `stacking_results*.json`) |
| TTA outputs | `dl/tta_{adabn,tent}_val/` |
| Pseudo-label pools | `data/pseudo_labels{,_v2,_v3,_coop}/`, unlabeled gpkg `data/patches_reference_unlabeled.gpkg` |
| Run scripts | `run_phase0_experiments.sh`, `run_phase1_*.sh`, `run_phase2_tta.sh` |
| Logs | `dl/_experiment_logs/` |
| WandB | project `lcz-classification-dl`, entity `phd-thesis-team` (e.g. v1 = `k4ka3wjs`) |

`dl` = `${DATA_DIR}/output/lcz-classification/dl`.
