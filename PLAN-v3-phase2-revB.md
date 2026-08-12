# eo-fm — PLAN v3, Phase 2 Rev B

**Amends `PLAN-v3-phase2-revA.md`. Rev A remains in force except where superseded below.**
Phases 3, 4, Deferred and the Write-up note in `PLAN-v3.md` are unchanged.

Task 2.0 is complete. Manifest `diagnostics/patch_manifest_v1.parquet`, sha256
`ad7fdae39ae75b77…`, 341,754 / 23,878 / 23,852. Rev A's pre-registration is committed
verbatim and is **not** revised by this document — additions below are marked as such and
must be committed before any run they apply to.

---

## What GATE 2.0 established

- **The two filters are one filter, and it is a water filter.** All 1,277 of AlphaEarth's
  >75%-invalid training patches are also absent from Tessera (105 of 105 on validation),
  against a 2.67% base rate, and none are absent from `seamless`. One mechanism: the
  AlphaEarth sentinel sits over ocean, and the Tessera v1.1 global archive has no ocean tile.
- **Tessera's entire val/test coverage hole is LCZ 17** — 330 of 330 test and 241 of 241
  validation patches, landing on Sydney (test −13.85%) and Mumbai (validation −9.99%), both
  held-out cities.
- **Re-extraction cannot fix it.** 223 of 9,422 absent training patches are recoverable
  (2.4%); **zero** of the 571 absent val/test patches are. 9,144 training and 560 of 571
  evaluation patches have no Tessera tile at any distance.
- **A second, separate defect exists.** 302 truncated patches remain correctly accepted by the
  coverage test; 294 (97.4%) straddle a UTM zone boundary and all 302 span multiple tiles. The
  truncation happens downstream in `crop_patch`'s multi-tile mosaic, at the 114°E and 120°E
  seams — London, Guangzhou/Hong Kong, Qingdao/Shanghai.
- `seamless` has its own truncated tail (77 training patches per-side, 30 under `native_frac`,
  smallest 3×11), invisible to Phase 1.75 because `crop_geometry.py` covered only the 10 m
  families.
- LCZ 17 loses 18.88% of its training patches and is 87.8% of all exclusions; its share falls
  14.01% → 11.72%. Six cities lose >5%, with Istanbul (12.9%) and Qingdao (12.6%) driven by
  Tessera absence rather than nodata.

---

## Amendment B1 — Tessera's water gap is a chapter-level finding

Promote from data-cleaning footnote to a reported result. Three things to produce:

**Establish what kind of limitation it is.** Tessera is trained on Sentinel-1/2, which do image
water, so the gap appears to be a coverage decision in the v1.1 global 0.1° archive rather than
a limit on the representation. Check whether the archive documentation states a land mask. If it
does, cite it; if it does not, report the finding empirically and precisely — 9,144 training and
560 evaluation patches with no tile at any distance, concentrated over ocean. Do not claim more
than the evidence supports: this is a statement about the distributed archive, not about what
Tessera could represent.

**Produce a coverage figure.** All three families' tile coverage over one affected held-out city
— Sydney or Mumbai — showing where each has data. One panel per family. This is a
straightforward figure that no one in the LCZ foundation-model literature has published, and it
makes the limitation legible in a way a table cannot.

**State the deployment consequence and the mitigation.** A global LCZ map built on Tessera v1.1
has holes at every coastline and large lake. The standard mitigation is that operational LCZ
maps assign water from a global water mask rather than classifying it, in which case the gap
costs nothing at deployment — and the manifest's removal of pure-ocean patches becomes more
defensible, not less. Decide which approach the chapter's map takes and say so explicitly.

---

## Amendment B2 — Task 2.2 becomes a 2×2, replacing Rev A's "A−B difference"

The manifest changes the test set's class composition. Because LCZ 17 is near-trivially
separable, removing it **lowers OA for every family** for reasons unrelated to representation
quality. A single A−B number would conflate that with the training-data effect.

Report the full grid. Each cell is 3 seeds, per family:

| | eval: common test | eval: native test |
|---|---|---|
| trained on manifest (Arm A) | **A/A** | A/B |
| trained on native coverage (Arm B) | **B/A** | B/B |

- **A/A vs A/B** — test class-mix effect, same checkpoint, within family.
- **A/A vs B/A** — training-data effect, same test set, within family.
- **Cross-family claims use the common-test column only.** Native test differs per family, so
  B/B is not comparable across families and must never appear in a cross-family table.

**Additional pre-registration (commit before running):** LCZ 17 recall will be **lower** on the
common test set than the native one, because the excluded patches are pure ocean while the
surviving water patches are coastal and mixed. If water recall rises instead, the filter is not
doing what the audit says and Task 2.2 stops pending an explanation.

---

## Amendment B3 — Metrics must separate the easy classes

LCZ 17 alone is 12–14% of the dataset and is near-ceiling for every family, so a large share of
every OA figure is water, and the manifest moves that share. Report, for every headline result
from Task 2.2 onward:

1. **macro-F1 first, OA second.** OA on this class distribution is dominated by classes nobody
   is competing on.
2. **Built-only metrics (LCZ 1–10) alongside the full 17-class figures.** The open problem in
   LCZ classification lives inside the built classes; natural-class performance is near-saturated
   and dilutes every aggregate.
3. **Per-class recall and the confusion matrix**, with LCZ 17 flagged as manifest-affected in
   the caption.

This supersedes any Rev A text implying OA is the headline number.

---

## Amendment B4 — Scope the multi-zone mosaic defect (measure only)

The 302 truncated patches are excluded from the manifest by `native_frac ≥ 0.5`, so **Phase 2
benchmarking is unaffected**. But `infer_roi` uses the same crop path, and London, Shanghai,
Guangzhou and Hong Kong sit on the affected seams, so this is a **potential blocker for map
production in Phase 4** and needs scoping before that decision.

Measure only — do not fix in Phase 2. Report:

- total patches in the dataset spanning multiple tiles, and how many of those are truncated
- total patches straddling a UTM zone boundary, and how many of those are truncated
- **the key ratio: of all zone-straddling patches, what fraction come back truncated?** If it
  approaches 100%, zone-straddling always fails and map production for seam cities is blocked
  until fixed. If it is a minority, the defect is narrower and the trigger condition needs
  identifying.
- whether the 8 truncated patches that span multiple tiles *without* straddling a zone share any
  other property, which would indicate a second mechanism

Log the result as the GATE 3 decision input alongside the re-extraction recovery count.

---

## Amendment B5 — `native_frac` threshold sensitivity (low priority)

`native_frac ≥ 0.5` is looser than Phase 1.75's 0.85-per-side rule (314 vs 560 Tessera patches),
and a 33×17 crop — a 2× vertical stretch after resize — survives it. The population at stake is
~246 patches out of 341,754, so this is almost certainly immaterial.

Do not re-cut the manifest. Instead, at the end of Phase 2, run **one** Arm A configuration
(`tesserav1.1_global`, 3 seeds) with the tighter per-side rule and confirm the difference falls
within seed noise. That closes the question permanently at a cost of three runs. If it somehow
does not fall within noise, report it and stop.

---

## Amendment B6 — Rev A tasks, confirmed unchanged

- **Task 2.1** — reproduction, native coverage, no manifest, old flags. Unaffected by everything
  above; run it first.
- **Task 2.3** — factorial, σ sweep, nodata, resize ablations, all on the Arm A manifest.
  Unchanged. If compute is tight, the σ sweep can run 1 seed across all five values for both
  families to locate the optimum, then 3 seeds at the best value and at 0.05 only — roughly 20
  runs instead of 30, with no loss to the reported claim.
- **Task 2.4** — monitor comparison. Unchanged, plus the val/test asymmetry note from Rev A.
- **Task 2.5** — resolution versus representation. Unchanged. Note that `seamless`'s newly
  found truncated tail is excluded by the manifest, so the native-resolution arm is not
  contaminated by 3×11 crops.

---

## GATE 2

As Rev A, plus: the coverage figure and the archive-documentation finding from B1; the full 2×2
grid from B2 with the class-mix and training-data effects separated; built-only and macro-F1
metrics per B3; the zone-straddling ratio from B4; and the threshold sensitivity result from B5.