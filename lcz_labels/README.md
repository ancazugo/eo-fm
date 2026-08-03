# `lcz_labels` — Overture/OSM-fused LCZ pseudo-labels

Generates global **Local Climate Zone** (LCZ) pseudo-labels from **Overture Maps**
(OSM-fused) vector data, aligned onto the existing So2Sat **320 m embedding patch
grid**, with per-patch **confidence** and **temporal-stability** flags. Outputs
join directly onto the embedding pipeline (same `patch_id` / geometry).

The core idea (design principle 1): Overture/OSM is a *semantic* database, LCZ is
*morphology*. We never map tags → LCZ. Instead we extract geometric + height
evidence, compute **Urban Canopy Parameters** (UCPs) per patch, and threshold
them into LCZ classes per **Stewart & Oke (2012)**.

## Install

Dependencies are declared in the repo `pyproject.toml` (`duckdb`, `exactextract`,
`pydantic`, `rasterio`, `pyproj`, `shapely`, `geopandas`, `typer`, …):

```bash
source /maps/acz25/envs/eo_fm-env/bin/activate
uv sync --active
```

## Usage

```bash
python -m lcz_labels extract  --aoi Nairobi        # Overture extraction + grid
python -m lcz_labels ucp       --aoi Nairobi        # zonal UCP table
python -m lcz_labels label     --aoi Nairobi        # LCZ + confidence
python -m lcz_labels mask       --aoi Nairobi        # temporal stability
python -m lcz_labels validate  --aoi Nairobi --aoi London --aoi Milan
python -m lcz_labels all        --aoi Nairobi        # full pipeline + validation
python -m lcz_labels all        --all-aois           # every So2Sat city
```

All thresholds, the pinned Overture release, and paths live in `config.py`
(pydantic). Override with `--config my.yaml`; a `config_hash` is stamped into
every output row and used as the per-stage cache key (`--force` rebuilds).

Output: `labels_{aoi}.parquet` (one row per patch: `patch_id, geometry, aoi, lcz,
lcz_name, confidence, stable_2017_to_label_year, change_score, label_year,
overture_release, config_hash` + all UCP/diagnostic columns) and a merged
`labels_all.parquet`. Use `cli.to_training_pairs(labels, years)` to expand stable
patches across embedding years.

## LCZ 7 (informal / lightweight low-rise) and coarse labels

OSM/Overture *tags* cannot separate formal compact low-rise (LCZ 3) from
lightweight/informal (LCZ 7), but **footprint morphology + road topology can,
partially**: informal fabric has tiny footprints, extreme building-count density,
high size irregularity, and many buildings per mapped road. Every compact + low
patch therefore goes through a **Stage 5b router** (`classify._route_compact_low`)
that emits one of three outcomes:

- **hard 7** — informal morphology + road deficit (or Million Neighborhoods
  corroboration). Confidence is capped (0.6, or 0.75 with MN) and the height-
  evidence requirement is waived (informal areas legitimately lack height tags).
- **hard 3** — clear formal morphology (larger median footprint, height tags,
  roads present).
- **coarse {3,7}** — morphology unreadable (ML footprints merged into blobs, or no
  clear signal). This is a **first-class label**, not a failure.

**Coarse labels** carry `label_type="coarse"`, `lcz=null`, and `lcz_set=[3,7]`
(the same mechanism also emits `{8,10}` for the large-lowrise/heavy-industry
ambiguity). Downstream training MUST consume them with a **marginalised loss**
`−log Σ_{c∈lcz_set} p_c` — never collapse a coarse label to one member. See
`cli.to_training_pairs` for the contract.

The validation report includes a mandatory **LCZ-7 audit** (hard-7
precision/recall vs So2Sat-7, and the *contamination* rate = So2Sat-7 patches
wrongly emitted as hard 3, which must stay ≤ 10 %).

## What it deliberately does NOT do

- It never assigns a sparse/natural label on absence of data alone: a positive
  land-cover verdict requires positive evidence and is cross-checked against
  GHS-BUILT-S (the "informal-settlement completeness trap").
- It never uses OSM/Overture edit history for temporal stability (mapping growth
  ≠ urban growth). Stability comes only from observed change products.
- When morphology is ambiguous it emits **coarse {3,7}** rather than guess a hard
  class — the safe direction is always coarse.

## Data sources & attribution

This module and its outputs are derived from:

- **Overture Maps** data — © Overture Maps Foundation, released under
  **CDLA-Permissive-2.0** (schema) with feature-level source licences. See
  <https://docs.overturemaps.org/attribution/>.
- **OpenStreetMap** — © OpenStreetMap contributors, **ODbL 1.0**
  (<https://www.openstreetmap.org/copyright>). Overture fuses OSM with Microsoft,
  Google Open Buildings, Esri and others; each feature's `sources` array records
  provenance.
- **GHS-BUILT-S / GHS-BUILT-H** (GHSL, JRC, European Commission) — free reuse
  with attribution: Pesaresi & Politis, GHSL R2023A.
- **Google Open Buildings 2.5D Temporal** (optional, Stage 7) — © Google,
  **CC BY-4.0**.

### ⚠️ Share-alike (ODbL) caveat — unresolved

Because Overture includes **OSM-derived (ODbL)** features, LCZ pseudo-labels
computed from them may constitute a *derivative database* and could carry
**ODbL share-alike obligations**. Whether the aggregate, thresholded UCP → LCZ
product is a "Produced Work" or a "Derivative Database" under ODbL is a
legal question this module does **not** resolve. Consult the ODbL and, if the
labels or models trained on them are redistributed, seek guidance before
choosing a licence. This README surfaces the obligation; it does not discharge
it.

## Layout

```
config.py      pydantic config, thresholds, config_hash          (Stage 1)
grid.py        reuse So2Sat 320 m grid / generate for new AOIs
overture.py    DuckDB-on-S3 extraction, source-aware              (Stage 2)
heights.py     tiered height model + GHS-BUILT-H backfill         (Stage 3)
ucp.py         zonal Urban Canopy Parameters per patch            (Stage 4)
classify.py    UCP → LCZ decision rules + confidence              (Stages 5-6)
change_mask.py temporal stability from observed change products   (Stage 7)
validate.py    agreement vs So2Sat                                (Stage 9)
cli.py         Typer CLI orchestration + output                   (Stage 8)
tests/         offline unit + end-to-end tests (+ ~1.5 km² fixture)
```
