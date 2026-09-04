# lcz_wudapt — community LCZ training areas as a Stage 8 label source

Turns the WUDAPT / LCZ-Generator submission database into the **same export
contract** `lcz_labels` produces, so `lcz_train` consumes community labels with
no change to its loss or dataset layer.

Source: `$DATA_DIR/input/WUDAPT/LCZ-Generator_training_areas_2024-10-01.gpkg`
— 630,311 polygons, 8,827 submissions, 2021-04 → 2024-10, 1,251 GUPPD urban
areas (25× So2Sat's 51 cities).

## Why this is not just a file read

The same ground is labelled by many annotators who disagree. Measured on the
2024-10-01 release:

| City | Polys | Submissions | Overlap area / total | Class agreement on overlaps |
|---|---|---|---|---|
| Wuhan | 44,341 | 388 | **6.03** | 0.870 |
| Guangzhou | 29,376 | 486 | 0.86 | **0.437** |
| Delhi | 8,947 | 72 | 0.86 | **0.383** |
| Tehran | 6,551 | 23 | 0.74 | **0.487** |

Pooled agreement on overlapping pairs is **0.71 by area**. A naive union of
polygons is therefore a badly noisy label set — harmonisation is the point of
this package, not a nicety.

## Stage 0 — no embeddings, no GPU

```bash
python -m lcz_wudapt ingest      # H1 clean + spatial AOI assignment (~90 s)
python -m lcz_wudapt inventory   # So2Sat sparsity + leakage guard
python -m lcz_wudapt audit       # G0 gate on the 8 audit cities
```

### H1 `ingest.py` — defects in the shipped file, all handled

| Defect | Handling |
|---|---|
| 4,883 invalid geometries; all `Polygon Z` | `make_valid` + explode; `force_2d` |
| 633 rows of class 18/19 | dropped and logged (a schema-change alarm) |
| `qc_step*` mixed `True`/`T` and `False`/`F` | normalised to nullable bool; unknown tokens → NA, never a silent False |
| `area` is **Web Mercator km²** (1.35× geodesic) | recomputed on EPSG:6933 (exact to ±0.002%); raw kept as `area_km2_mercator_raw` |
| `city` free-text and multilingual (`北京`, `wuhan`, `..`) | never read; AOI assignment is spatial |
| `representative_date` includes 2029, 2117, 2323 | out-of-range → NA, falls back to submission year |
| `JRC_NAME_MAIN` not unique (León ×3) | every AOI key is `{slug(name)}__{SMOD_ID}` |

**The annotator is the author, not the submission.** 8,827 submissions collapse
to ~1,500 named authors (Wuhan 388 → 41), averaging 3.02 versions per city.
106,793 polygons (16.8%) have no name at all; keying each on its submission
invents 1,764 pseudo-annotators — *more than the real author count* — which
inflates `n_eff` and therefore confidence. `blank_name_policy` defaults to
`collapse_per_aoi` because understating consensus costs coverage, while
overstating it corrupts labels.

**The 27.4% of polygons outside every GUPPD bbox are retained** in a `_rural`
partition. They are **61.8% natural classes (LCZ 11–17) against 35.1% inside** —
filtering to urban areas would systematically strip the classes So2Sat already
under-samples.

### H2 `quality.py` — author collapse, gates, weights

Author collapse is expressed as a deterministic **burn order** (oldest
submission first) rather than a geometric merge, so a later revision overwrites
its own earlier version on overlapping ground while disjoint earlier work
survives — exact, and free.

`w = w_qc · w_acc · w_time · w_size`, one weight per polygon. `w_acc` reads
`oau` for built classes 1–10 and `oa` for natural 11–17, scaled by the
submission's F1 *for that specific class* relative to its peers.

### H3 `consensus.py` — per-pixel posterior

Votes are accumulated per **annotator** on the canonical 10 m UTM grid, then a
Dirichlet-smoothed posterior decides:

- `|S| = 1` → **hard** label
- `2 ≤ |S| ≤ 3` → **coarse** set (a genuine multi-class bitmask)
- too diffuse → unlabelled

This is why the bitmask contract was chosen: `lcz_train.losses.marginalized_ce`
scores `-log Σ_{c∈S} p_c`, so ambiguity costs nothing when the truth is in the
set, and the model is never forced to guess which annotator was right.

Two calibration facts learned the hard way, both pinned by tests:

- **The prior is in units of one average annotator of that AOI**, not an
  absolute. A fixed `alpha = 1.0` outvotes every real annotator (real weights
  average 0.3–0.6) and drove the hard-label fraction to 0.6% in Nairobi — a city
  that agrees with So2Sat at 0.77.
- **`depth` is continuous in `n_eff`.** A branch at `n_eff == 1` made confidence
  depend on whether the arithmetic ran in float32 or float64.

### Leakage

So2Sat is authoritative. WUDAPT is admitted inside a So2Sat city only where
So2Sat is sparse or degenerate (`n_patches < 2000 or n_classes < 8`), which
selects 12 single-class training cities (Salvador has **1** patch). Every
held-out test city has ≥ 4798 patches and ≥ 13 classes, so the rule *cannot*
select one — and `leakage.assert_no_test_city_admitted` enforces that in code
rather than trusting the coincidence.
