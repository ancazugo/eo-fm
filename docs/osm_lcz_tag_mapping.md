# OSM Tag Mapping to Local Climate Zone (LCZ) Classes

## LCZ Label Generation from OpenStreetMap Data with `osm-rasterizer`

This document selects a **curated, globally applicable set of OSM features** for differentiating the 17 Local Climate Zone classes of Stewart and Oke (2012), and shows the exact `osm-rasterizer` commands that turn them into rasters for any bounding box. The selection is informed by:

- **Fonte et al. (2019)** — *Using OpenStreetMap (OSM) to enhance the classification of Local Climate Zones in the framework of WUDAPT* (`docs/1-s2.0-S221209551930094X-main.pdf`), which established OSM key/value→LCZ correspondences and a Building/Impervious Surface Fraction (BSF/ISF) methodology.
- **GeoClimate** ([orbisgis/geoclimate](https://github.com/orbisgis/geoclimate); Bocher et al. 2021), an operational toolbox that computes LCZ from OSM worldwide, whose layer definitions (building, road, rail, vegetation, water, impervious) encode a battle-tested global tag selection.

An [appendix](#appendix-exhaustive-tag-reference) preserves the exhaustive per-class tag reference, including historic and deprecated tags.

---

## What the sources teach

### Fonte et al. (2019): OSM→LCZ conversion in the WUDAPT framework

The paper converts OSM into LCZ evidence in two tracks:

1. **Direct correspondence for land-cover classes** (LCZ A–D, G): polygons matching a key/value list (their Table 2) are merged and intersected with a ~120 m grid; the fraction of each cell covered gives a (fuzzy) membership per class.
2. **BSF/ISF for built classes** (LCZ 1–10): building footprints give the Building Surface Fraction; roads and railways — *buffered from lines to areas* using per-type widths or the distance to adjacent buildings — give the Impervious Surface Fraction. Trapezoidal membership functions over the Stewart & Oke ranges (table below) then yield candidate built classes per cell.

Key findings that shape the selection here:

- **OSM water is highly reliable** — in their weighted combination tests the OSM water layer was weighted 60% against 40% for satellite classification, and produced a far more detailed water network. → put water at the **highest priority** in a label map.
- **OSM cannot separate dense from scattered trees** (LCZ A vs B) — tags describe the *presence* of woodland, not canopy density. Function-based tags (`orchard`, `plant_nursery`, `tree_row`) are the only tag-level proxies for scattered trees.
- **Building height tags are decisive but sparse.** Without `building:levels`/`height`, LCZ 1/4, 2/5, 3/6 cannot be separated; BSF/ISF only narrow the candidates. LCZ 4 (open high-rise) remained poorly classified even after combination.
- **Compact vs open** (1–3 vs 4–6) and **sparsely built** (LCZ 9) are *density* properties — they require fraction aggregation over a neighbourhood-scale grid, never a per-pixel tag.

### GeoClimate: an operational, worldwide OSM→LCZ chain

GeoClimate extracts thematic input layers from OSM with fixed tag lists, computes Stewart & Oke's indicators per spatial unit, and classifies LCZ by minimum distance to the class prototypes. What transfers to rasterization:

- **Layer-based extraction.** Buildings, roads, rail, vegetation, water and impervious surfaces are separate layers — mirrored below as separate raster features rather than one tag soup.
- **Indicator weights reveal what matters.** In GeoClimate's LCZ distance the building surface fraction has weight 8 and the height of roughness elements 6 — by far the largest (sky view factor 4, aspect ratio 3, terrain roughness 0.5, ISF/PSF 0). → invest tag effort in **building footprints and height**, exactly what the curated set does.
- **Rule-based LCZ 8 and 10.** GeoClimate assigns *large low-rise* and *heavy industry* not by distance but by the fraction of large low-rise / industrial buildings — justifying dedicated building-type features for these classes.
- **Vegetation split high/low**: high = `wood`, `forest`, mangrove, `orchard`, banana plants; low = `grass`, `grassland`, `heath`, `meadow`, `farmland`, `scrub`, `vineyard`, `park`, `garden`, wetlands. This is the LCZ A/B vs C/D axis.
- **Default building levels per type** when height is untagged: most types 1 (house, education, healthcare, office), commercial/hotel 2, religious/agricultural/sport 0. → an **untagged building is best assumed low-rise**, which is how the label-map command below treats it.
- **Default road widths per type** (metres): motorway 24, trunk 16, primary/secondary 10, tertiary/residential 8, unclassified/service/pedestrian 3, track 2, cycleway/footway/path 1, aeroway 18. → used as `line_width` fallbacks with `width_from_tags` so per-feature `width`/`lanes` tags win when present.

Indicators that do **not** transfer to a tag-based raster: sky view factor, aspect ratio, terrain roughness — these need 3-D morphology, not tags.

**Stewart & Oke (2012) reference ranges for built types:**

| LCZ | BSF (%) | ISF (%) | Height (m) | Levels |
|-----|---------|---------|------------|--------|
| 1 – Compact high-rise | 40–60 | 40–60 | >25 | >10 |
| 2 – Compact mid-rise | 40–70 | 30–50 | 10–25 | 3–9 |
| 3 – Compact low-rise | 40–70 | 20–50 | 3–10 | 1–3 |
| 4 – Open high-rise | 20–40 | 30–40 | >25 | >10 |
| 5 – Open mid-rise | 20–40 | 30–50 | 10–25 | 3–9 |
| 6 – Open low-rise | 20–40 | 20–50 | 3–10 | 1–3 |
| 7 – Lightweight low-rise | 60–90 | <20 | 2–4 | 1 |
| 8 – Large low-rise | 30–50 | 40–50 | 3–10 | 1–3 |
| 9 – Sparsely built | 10–20 | <20 | 3–10 | 1–3 |
| 10 – Heavy industry | 20–30 | 20–40 | 5–15 | 1–5 |

---

## The curated global feature set

Seventeen raster features, ordered for `--single-layer` mode where **later features overwrite earlier ones** — so the list runs from broad/uncertain context to specific/reliable evidence, ending with water (the most reliable OSM layer per Fonte et al.).

| # | Feature | LCZ | Source tags (osmnx dict) | Options |
|---|---------|-----|--------------------------|---------|
| 1 | `low_plants` | D | `landuse`: farmland, meadow, grass, allotments, recreation_ground, village_green, greenfield, flowerbed · `natural`: grassland, fell, wetland · `leisure`: park, garden, golf_course, pitch, common · `landcover`: grass | — |
| 2 | `scrub` | C | `natural`: scrub, heath, shrubbery · `landuse`: vineyard | — |
| 3 | `scattered_trees` | B | `landuse`: orchard, plant_nursery · `natural`: tree_row | `line_width: 5` |
| 4 | `dense_trees` | A | `natural`: wood · `landuse`: forest · `wetland`: mangrove, swamp | — |
| 5 | `bare_soil` | F | `natural`: sand, beach, dune, mud, shingle · `landuse`: quarry, brownfield, construction, landfill | — |
| 6 | `bare_rock` | E | `natural`: bare_rock, scree, rock, stone | — |
| 7 | `rail` | E/ISF | `railway`: rail, light_rail, tram, narrow_gauge, funicular, monorail · `landuse`: railway | `line_width: 6` |
| 8 | `roads_minor` | E/ISF | `highway`: tertiary(+link), residential, unclassified, service, living_street, track | `line_width: 6`, `width_from_tags` |
| 9 | `roads_major` | E/ISF | `highway`: motorway(+link), trunk(+link), primary(+link), secondary(+link) | `line_width: 12`, `width_from_tags` |
| 10 | `paved` | E | `amenity`: parking · `aeroway`: runway, taxiway, apron · `highway`: pedestrian | `line_width: 8`, `width_from_tags` |
| 11 | `heavy_industry` | 10 | `landuse`: industrial, port · `man_made`: works, wastewater_plant, water_works, chimney, storage_tank, silo, gasometer, kiln, petroleum_well · `power`: plant, substation · `industrial`: * | — |
| 12 | `buildings_lowrise` | 3/6 | `building`: * (all footprints — untagged height ⇒ low-rise, per GeoClimate defaults) | — |
| 13 | `large_lowrise` | 8 | `building`: warehouse, industrial, retail, supermarket, hangar, train_station, transportation, stadium, sports_hall, parking, service, manufacture, depot | — |
| 14 | `lightweight` | 7 | `building`: hut, shed, cabin, ger, yurt, static_caravan, tent, slum, shanty | — |
| 15 | `buildings_midrise` | 2/5 | `building`: * | `filter`: `building:levels` ∈ 4–9 |
| 16 | `buildings_highrise` | 1/4 | `building`: * | `filter`: `building:levels` ≥ 10 |
| 17 | `water` | G | `natural`: water, bay, strait · `waterway`: river, canal, stream, drain, ditch · `landuse`: reservoir, basin, salt_pond · `leisure`: swimming_pool | `line_width: 8`, `width_from_tags` |

Design notes:

- **Priority order is the classifier.** In single-layer mode the last feature wins each pixel, so a warehouse is first painted as a generic low-rise building (12) and then overwritten by `large_lowrise` (13); a 12-storey block ends up `buildings_highrise` (16). If your interest is industrial zones as a whole rather than the buildings inside them, move `heavy_industry` after the building features — the whole `landuse=industrial` polygon will then win. In multi-band fraction workflows, deciding 8 vs 10 by the industrial-building fraction (GeoClimate's rule) is more faithful.
- **Height classes partition cleanly**: untagged or 1–3 levels ⇒ low-rise, 4–9 ⇒ mid-rise, ≥10 ⇒ high-rise (Stewart & Oke put 3 levels on the low/mid boundary; it is assigned low here so that the sparse-tag default stays conservative).
- **Compact vs open (1–3 vs 4–6) and LCZ 9 are not in the table** — they are density classes, only decidable from BSF aggregated over ~100 m cells (Command B below). A per-pixel label map cannot express them; treat labels 12/15/16 as "low/mid/high-rise built" proxies.
- **Lightweight (LCZ 7) is typed by building value, not material.** `building:material=wood` marks LCZ 7 in informal-settlement contexts but ordinary LCZ 6 housing in North America and Scandinavia, so material is *not* used globally. Where you know the regional context, add a material-based variant with the attribute filter (AND semantics), e.g.:
  `'lightweight_material:{"tags": {"building": true}, "filter": {"building:material": ["corrugated_iron", "metal", "thatch", "bamboo", "mud", "plastic"]}}'`
- **Buffered lines.** Roads, rail, waterways and tree rows are line geometries; each carries a `line_width` fallback (from GeoClimate's width table) and `width_from_tags: true` so an explicit `width` tag (or `lanes` × 3.5 m) wins per geometry. Footways, paths and cycleways (~1 m) are omitted: sub-pixel at 10 m resolution.
- **Ambiguous tags are assigned once.** `quarry`/`landfill` sit in `bare_soil` (their surface signature) rather than LCZ 10 (their function); `natural=wetland` sits in `low_plants` (LCZ D-w) with mangrove/swamp pulled out to `dense_trees`. The appendix lists all alternatives.

---

## Command A (primary): single-layer LCZ-proxy label map

Set the bounding box (WGS84 `minx,miny,maxx,maxy`) and run — no other edits needed:

```bash
BBOX="minx,miny,maxx,maxy"    # e.g. "-0.15,51.48,-0.08,51.52"

osm-rasterizer \
  --bbox "$BBOX" \
  --feature 'low_plants:{"landuse": ["farmland", "meadow", "grass", "allotments", "recreation_ground", "village_green", "greenfield", "flowerbed"], "natural": ["grassland", "fell", "wetland"], "leisure": ["park", "garden", "golf_course", "pitch", "common"], "landcover": ["grass"]}' \
  --feature 'scrub:{"natural": ["scrub", "heath", "shrubbery"], "landuse": ["vineyard"]}' \
  --feature 'scattered_trees:{"tags": {"landuse": ["orchard", "plant_nursery"], "natural": ["tree_row"]}, "line_width": 5}' \
  --feature 'dense_trees:{"natural": ["wood"], "landuse": ["forest"], "wetland": ["mangrove", "swamp"]}' \
  --feature 'bare_soil:{"natural": ["sand", "beach", "dune", "mud", "shingle"], "landuse": ["quarry", "brownfield", "construction", "landfill"]}' \
  --feature 'bare_rock:{"natural": ["bare_rock", "scree", "rock", "stone"]}' \
  --feature 'rail:{"tags": {"railway": ["rail", "light_rail", "tram", "narrow_gauge", "funicular", "monorail"], "landuse": ["railway"]}, "line_width": 6}' \
  --feature 'roads_minor:{"tags": {"highway": ["tertiary", "tertiary_link", "residential", "unclassified", "service", "living_street", "track"]}, "line_width": 6, "width_from_tags": true}' \
  --feature 'roads_major:{"tags": {"highway": ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link"]}, "line_width": 12, "width_from_tags": true}' \
  --feature 'paved:{"tags": {"amenity": ["parking"], "aeroway": ["runway", "taxiway", "apron"], "highway": ["pedestrian"]}, "line_width": 8, "width_from_tags": true}' \
  --feature 'heavy_industry:{"landuse": ["industrial", "port"], "man_made": ["works", "wastewater_plant", "water_works", "chimney", "storage_tank", "silo", "gasometer", "kiln", "petroleum_well"], "power": ["plant", "substation"], "industrial": true}' \
  --feature 'buildings_lowrise:{"building": true}' \
  --feature 'large_lowrise:{"building": ["warehouse", "industrial", "retail", "supermarket", "hangar", "train_station", "transportation", "stadium", "sports_hall", "parking", "service", "manufacture", "depot"]}' \
  --feature 'lightweight:{"building": ["hut", "shed", "cabin", "ger", "yurt", "static_caravan", "tent", "slum", "shanty"]}' \
  --feature 'buildings_midrise:{"tags": {"building": true}, "filter": {"building:levels": ["4", "5", "6", "7", "8", "9"]}}' \
  --feature 'buildings_highrise:{"tags": {"building": true}, "filter": {"building:levels": ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"]}}' \
  --feature 'water:{"tags": {"natural": ["water", "bay", "strait"], "waterway": ["river", "canal", "stream", "drain", "ditch"], "landuse": ["reservoir", "basin", "salt_pond"], "leisure": ["swimming_pool"]}, "line_width": 8, "width_from_tags": true}' \
  --output lcz_labels.tif \
  --resolution 10 \
  --single-layer \
  --fill-nodata \
  --fill-nodata-distance 50
```

The output is one categorical band; pixel values are 1-based indices into the feature order (0 = no data):

| Value | Feature | LCZ proxy | Value | Feature | LCZ proxy |
|-------|---------|-----------|-------|---------|-----------|
| 1 | `low_plants` | D | 10 | `paved` | E |
| 2 | `scrub` | C | 11 | `heavy_industry` | 10 |
| 3 | `scattered_trees` | B | 12 | `buildings_lowrise` | 3/6 |
| 4 | `dense_trees` | A | 13 | `large_lowrise` | 8 |
| 5 | `bare_soil` | F | 14 | `lightweight` | 7 |
| 6 | `bare_rock` | E | 15 | `buildings_midrise` | 2/5 |
| 7 | `rail` | E | 16 | `buildings_highrise` | 1/4 |
| 8 | `roads_minor` | E | 17 | `water` | G |
| 9 | `roads_major` | E | | | |

Practical notes:

- `--fill-nodata --fill-nodata-distance 50` fills unmapped pixels from their nearest labelled neighbour up to 500 m (50 px × 10 m), leaving genuinely unmapped border areas as 0. Drop both flags to keep gaps explicit — often preferable for training labels.
- **The fill is destructive where OSM is sparse, and it is not reversible.** Measured on the Nairobi run (bbox `36.554,-1.443,37.063,-1.065`, 2026-08): only **28.4 %** of the AOI carries any OSM feature at all, so nearest-neighbour fill invents the other ~70 %. Because roads form a pervasive network they win almost every contest: `roads_minor` goes from **4.15 % → 53.4 %** of the raster, and the building union from **3.36 % → 13.8 %**, which in turn drags the BSF "compact" bin (Command B) from **28.5 % → 89.7 %** — i.e. it would call nearly all of Nairobi Compact Low-Rise. Prefer Command B (multi-band, unfilled) as the archival product and fill only at the *end* of a workflow, from areal donors, as `src/osm_lcz_relabel.py --fill-distance-m` does. A single-layer raster can be tested for fill after the fact: buffered lines are 1–2 px wide at 10 m, so `binary_erosion(mask, 3×3).sum() / mask.sum()` on `roads_minor` is 0.000 unfilled and 0.877 filled.
- `--resolution 10` preserves individual buildings and road corridors. LCZ is defined at neighbourhood scale — aggregate to 100–120 m (majority vote, or the fraction workflow below) before comparing to WUDAPT products.
- The `building:levels` filter matches strings, so the high-rise spec enumerates "10"–"60"; buildings taller than 60 tagged levels, or tagged only with `height`, are missed. The Python workflow below handles both robustly.
- Reproducibility: add `--date 2024-01-01` to query OSM as it existed at that date (or `--provider ohm` for OpenHistoricalMap).
- `{"building": true}` is fetched three times (features 12, 15, 16). The Python workflow fetches it once.

To turn either output into an actual LCZ raster, run `src/osm_lcz_relabel.py` (see the README): it maps the proxy indices onto the So2Sat 1–17 convention, resolves 12/15/16 into compact/open/sparse via the Command B building surface fraction, and embeds the WUDAPT colour table so the file renders in QGIS unstyled.

## Command B (secondary): multi-band masks for BSF/ISF fraction workflows

Dropping `--single-layer` (and the nodata filling) writes one 0/1 band per feature — overlaps preserved — which is what a GeoClimate/WUDAPT-style classification needs:

```bash
osm-rasterizer \
  --bbox "$BBOX" \
  ... same seventeen --feature arguments ... \
  --output lcz_layers.tif \
  --resolution 10
```

Aggregate to 100 m cells (10 × 10 px blocks) and compute the Stewart & Oke fractions:

```python
import numpy as np
import rasterio

with rasterio.open("lcz_layers.tif") as src:
    bands = {name: src.read(i + 1) for i, name in enumerate(src.descriptions)}

BLOCK = 10  # 10 px × 10 m = 100 m cells

def fraction(mask: np.ndarray) -> np.ndarray:
    h, w = (s // BLOCK * BLOCK for s in mask.shape)
    blocks = mask[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    return blocks.mean(axis=(1, 3))

building_bands = ["buildings_lowrise", "large_lowrise", "lightweight",
                  "buildings_midrise", "buildings_highrise"]
impervious_bands = ["rail", "roads_minor", "roads_major", "paved"]

bsf = fraction(np.maximum.reduce([bands[b] for b in building_bands]) > 0)
isf = fraction(np.maximum.reduce([bands[b] for b in impervious_bands]) > 0)
```

`bsf` and `isf` are per-cell fractions in [0, 1]; apply the Stewart & Oke ranges (table above) to score built-class candidates — crisply, or with trapezoidal memberships as in Fonte et al. (2019). This is where compact vs open (BSF 40–70 % vs 20–40 %) and sparsely built (BSF 10–20 %) become decidable, and where LCZ 8 vs 10 can follow GeoClimate's rule (fraction of large low-rise vs industrial buildings per cell).

## Python API: one fetch, robust height splits

The CLI's dict filter compares strings. The Python API accepts **pre-fetched GeoDataFrames** and **callable filters**, so buildings are fetched once and split numerically — parsing `building:levels` *and* `height` (≈3 m per storey), with untagged buildings defaulting to low-rise:

```python
import numpy as np
import pandas as pd

from osm_rasterizer import fetch_features, rasterize

bbox = (-0.15, 51.48, -0.08, 51.52)  # minx, miny, maxx, maxy
buildings = fetch_features(bbox, {"building": True})

def _numeric(gdf, col):
    if col not in gdf.columns:
        return pd.Series(np.nan, index=gdf.index)
    return pd.to_numeric(
        gdf[col].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
    )

def height_class(lo, hi):
    """Keep buildings whose storey count (levels, else height/3, else 1) is in [lo, hi]."""
    def _filter(gdf):
        levels = _numeric(gdf, "building:levels")
        storeys = levels.fillna(_numeric(gdf, "height") / 3.0).fillna(1.0)
        return ((storeys >= lo) & (storeys <= hi)).to_numpy()
    return _filter

LARGE_LOWRISE = ["warehouse", "industrial", "retail", "supermarket", "hangar",
                 "train_station", "transportation", "stadium", "sports_hall",
                 "parking", "service", "manufacture", "depot"]
LIGHTWEIGHT = ["hut", "shed", "cabin", "ger", "yurt", "static_caravan",
               "tent", "slum", "shanty"]

rasterize(
    bbox,
    features=[
        # ... land-cover, road, rail and industry features as in Command A ...
        ("buildings_lowrise", buildings, {"filter": height_class(0, 3.99)}),
        ("large_lowrise", buildings,
         {"filter": lambda g: g["building"].isin(LARGE_LOWRISE).to_numpy()}),
        ("lightweight", buildings,
         {"filter": lambda g: g["building"].isin(LIGHTWEIGHT).to_numpy()}),
        ("buildings_midrise", buildings, {"filter": height_class(4, 9.99)}),
        ("buildings_highrise", buildings, {"filter": height_class(10, np.inf)}),
        ("water", {"natural": ["water", "bay", "strait"],
                   "waterway": ["river", "canal", "stream", "drain", "ditch"],
                   "landuse": ["reservoir", "basin", "salt_pond"],
                   "leisure": ["swimming_pool"]},
         {"line_width": 8, "width_from_tags": True}),
    ],
    resolution=10.0,
    single_layer=True,
    fill_nodata=True,
    fill_nodata_distance=50,
    output_path="lcz_labels.tif",
)
```

Dict filters remain available in Python too, with the same semantics as the CLI: AND across columns, OR within a value list, and `;`-separated OSM multi-values (`"soccer;basketball"`) matched element-wise. See [examples](examples.md) and the [filtering notebook](notebooks/filtering-by-attribute.ipynb).

---

## Appendix: exhaustive tag reference

The remainder of this document is the full per-class OSM tag reference — current, historic, and deprecated tags sourced from the OSM Wiki, TagInfo, and Fonte et al. (2019) — useful when extending the curated set for a specific region or historical snapshot.

---

## BUILT TYPES (LCZ 1–10)

### LCZ 1 – Compact High-Rise

Dense mix of tall buildings (>10 stories). Few or no trees. Land cover mostly paved. Concrete, steel, stone, and glass construction materials.

**Key: `building`** (with `building:levels` >= 10 OR `height` >= 25m)

| Tag | Notes |
|-----|-------|
| `building=apartments` | When levels >= 10 |
| `building=residential` | When levels >= 10 |
| `building=commercial` | When levels >= 10 |
| `building=office` | When levels >= 10 |
| `building=hotel` | When levels >= 10 |
| `building=yes` | Generic; requires height/levels filter |
| `building=public` | When levels >= 10 |
| `building=civic` | When levels >= 10 |
| `building=hospital` | When levels >= 10 |
| `building=university` | When levels >= 10 |
| `building=skyscraper` | Informal/deprecated, but found in historic data |
| `building=tower` | Used for tower-form buildings |

**Key: `landuse`** (areas where compact high-rise is dominant)

| Tag | Notes |
|-----|-------|
| `landuse=commercial` | Dense CBD areas (combine with building density analysis) |
| `landuse=retail` | High-density commercial cores |
| `landuse=residential` | High-rise residential estates (combine with building data) |

**Key: `building:levels`** — Values >= 10 strongly indicate LCZ 1 or 4. Combined with BSF to distinguish compact (1) vs open (4).

**Key: `height`** — Values >= 25m.

**Key: `building:material` / `building:facade:material`**

| Tag | Notes |
|-----|-------|
| `building:material=concrete` | Typical for high-rise |
| `building:material=glass` | Typical for modern high-rise |
| `building:material=steel` | Typical for high-rise |
| `building:facade:material=glass` | |

---

### LCZ 2 – Compact Mid-Rise

Dense mix of mid-rise buildings (3–9 stories). Few or no trees. Land cover mostly paved. Stone, brick, tile, and concrete construction materials.

**Key: `building`** (with `building:levels` 3–9 OR `height` 10–25m)

| Tag | Notes |
|-----|-------|
| `building=apartments` | When levels 3–9 |
| `building=residential` | When levels 3–9 |
| `building=commercial` | When levels 3–9 |
| `building=hotel` | When levels 3–9 |
| `building=dormitory` | Student/worker housing blocks |
| `building=office` | When levels 3–9 |
| `building=public` | When levels 3–9 |
| `building=civic` | When levels 3–9 |
| `building=hospital` | When levels 3–9 |
| `building=school` | When levels 3–9 |
| `building=university` | When levels 3–9 |
| `building=yes` | Generic; requires height/levels filter |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=residential` | Dense mid-rise neighbourhoods |
| `landuse=commercial` | Mid-rise commercial districts |
| `landuse=education` | Campus areas with mid-rise buildings |

**Key: `building:material` / `building:facade:material`**

| Tag | Notes |
|-----|-------|
| `building:material=brick` | Typical for European mid-rise |
| `building:material=stone` | Historic mid-rise |
| `building:material=concrete` | Modern mid-rise |
| `building:material=plaster` | Rendered masonry |

---

### LCZ 3 – Compact Low-Rise

Dense mix of low-rise buildings (1–3 stories). Few or no trees. Land cover mostly paved. Stone, brick, tile, and concrete construction materials.

**Key: `building`** (with `building:levels` 1–3 OR `height` 3–10m, AND high density)

| Tag | Notes |
|-----|-------|
| `building=terrace` | Row houses — strong indicator |
| `building=house` | When in dense arrangement |
| `building=residential` | When levels 1–3, dense |
| `building=apartments` | Low-rise apartments (levels 1–3) |
| `building=semidetached_house` | When in dense layout |
| `building=yes` | Requires height/density filter |
| `building=commercial` | Low-rise shops |
| `building=retail` | Low-rise retail |
| `building=church` | Historic compact centres |
| `building=chapel` | |
| `building=cathedral` | Typically compact historic core |
| `building=mosque` | |
| `building=temple` | |
| `building=synagogue` | |
| `building=shrine` | |
| `building=civic` | When low-rise |
| `building=public` | When low-rise |
| `building=school` | When low-rise |
| `building=kindergarten` | |
| `building=bakehouse` | Deprecated/rare but found in historic data |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=residential` | Dense traditional neighbourhoods (old towns, medinas) |
| `landuse=commercial` | Low-rise commercial streets |
| `landuse=retail` | Low-rise shopping areas |

---

### LCZ 4 – Open High-Rise

Open arrangement of tall buildings (>10 stories). Abundance of pervious land cover (low plants, scattered trees). Concrete, steel, stone, and glass construction materials.

Same building tags as LCZ 1, but with BSF 20–40% (lower density). Typical of modernist tower-block housing estates, campus-style office parks with tall buildings.

**Key: `building`** (with `building:levels` >= 10 OR `height` >= 25m, in lower-density setting)

All tags from LCZ 1 apply, differentiated by lower BSF.

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=residential` | Tower-block estates with green space |
| `landuse=education` | University campuses with high-rise buildings |

---

### LCZ 5 – Open Mid-Rise

Open arrangement of mid-rise buildings (3–9 stories). Abundance of pervious land cover. Concrete, steel, stone, and glass construction materials.

Same building tags as LCZ 2, but with BSF 20–40%.

**Key: `building`** (with `building:levels` 3–9 OR `height` 10–25m, lower-density)

All tags from LCZ 2 apply, differentiated by lower BSF.

---

### LCZ 6 – Open Low-Rise

Open arrangement of low-rise buildings (1–3 stories). Abundance of pervious land cover. Wood, brick, stone, tile, and concrete construction materials.

**Key: `building`** (with `building:levels` 1–3 OR no levels tag, in suburban/low-density setting)

| Tag | Notes |
|-----|-------|
| `building=house` | Detached houses — strong indicator |
| `building=detached` | Explicitly detached |
| `building=bungalow` | Single-story house |
| `building=semidetached_house` | Semi-detached (suburban) |
| `building=terrace` | If in open/suburban arrangement |
| `building=residential` | Low-rise, open layout |
| `building=farm` | Farmhouse buildings |
| `building=farm_auxiliary` | Associated farm structures |
| `building=cabin` | Small rural building |
| `building=yes` | Generic low-rise in suburban areas |
| `building=garage` | Residential garages |
| `building=garages` | Garage blocks |
| `building=carport` | |
| `building=shed` | Garden/utility sheds |
| `building=stable` | |
| `building=barn` | When near residential areas |
| `building=conservatory` | Residential conservatory |
| `building=static_caravan` | Permanent mobile homes |
| `building=houseboat` | Residential watercraft |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=residential` | Suburban neighbourhoods |
| `landuse=allotments` | Allotment gardens with sheds |
| `landuse=farmyard` | Farm complexes |
| `landuse=cemetery` | Often open low-rise character |
| `landuse=garages` | Garage areas (deprecated in some regions but common) |

**Key: `building:material`**

| Tag | Notes |
|-----|-------|
| `building:material=wood` | Common in suburban/rural |
| `building:material=brick` | |
| `building:material=stone` | |
| `building:material=concrete` | |

---

### LCZ 7 – Lightweight Low-Rise

Dense mix of single-story buildings. Few or no trees. Land cover mostly hard-packed. Lightweight construction materials (wood, thatch, corrugated metal).

**Key: `building`**

| Tag | Notes |
|-----|-------|
| `building=hut` | Informal/lightweight structure |
| `building=shed` | When in dense informal settlement |
| `building=kiosk` | Small single-story structure |
| `building=cabin` | When lightweight material |
| `building=static_caravan` | Mobile homes |
| `building=ger` | Mongolian yurt/ger |
| `building=tent` | Informal tag found in data |
| `building=roof` | Open-sided roofed structure |
| `building=yes` | In informal settlement context |
| `building=slum` | Informal/deprecated but found in historic data for favelas etc. |
| `building=shanty` | Informal tag |

**Key: `building:material`**

| Tag | Notes |
|-----|-------|
| `building:material=metal` | Corrugated iron/zinc |
| `building:material=wood` | Lightweight timber |
| `building:material=thatch` | Thatched roof/walls |
| `building:material=bamboo` | |
| `building:material=mud` | Adobe/cob |
| `building:material=plastic` | Informal construction |
| `roof:material=metal` | Tin/corrugated roof |
| `roof:material=thatch` | |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=residential` | In informal settlement context |

**Key: `place`** (contextual)

| Tag | Notes |
|-----|-------|
| `place=village` | In developing regions |
| `place=hamlet` | |

---

### LCZ 8 – Large Low-Rise

Open arrangement of large low-rise buildings (1–3 stories). Few or no trees. Land cover mostly paved. Steel, concrete, metal, and stone construction materials.

**Key: `building`**

| Tag | Notes |
|-----|-------|
| `building=warehouse` | Strong indicator |
| `building=industrial` | Factories, workshops |
| `building=commercial` | Large commercial buildings (big-box retail) |
| `building=retail` | Large retail stores |
| `building=supermarket` | Supermarket buildings |
| `building=hangar` | Aircraft hangars |
| `building=stadium` | Sports venues |
| `building=train_station` | Rail station buildings |
| `building=transportation` | Transport-related buildings |
| `building=service` | Service buildings |
| `building=storage_tank` | Deprecated on some wikis but found |
| `building=parking` | Multi-story car parks (low-rise) |
| `building=hospital` | When large, low-rise campus style |
| `building=manufacture` | Informal tag for factories |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=retail` | Big-box / out-of-town retail parks |
| `landuse=commercial` | Large commercial complexes |
| `landuse=education` | Large low-rise school/campus buildings |

**Key: `amenity`**

| Tag | Notes |
|-----|-------|
| `amenity=parking` | Large surface car parks |
| `amenity=school` | Large school campuses |
| `amenity=hospital` | Hospital complexes |
| `amenity=marketplace` | |

**Key: `shop`** (large format)

| Tag | Notes |
|-----|-------|
| `shop=supermarket` | Large retail |
| `shop=mall` | Shopping centres |
| `shop=department_store` | |
| `shop=wholesale` | |

---

### LCZ 9 – Sparsely Built

Sparse arrangement of small or medium-sized buildings in a natural setting. Abundance of pervious land cover (low plants, scattered trees).

**Key: `building`** (very low density, BSF 10–20%)

| Tag | Notes |
|-----|-------|
| `building=farm` | Isolated farmsteads |
| `building=farm_auxiliary` | |
| `building=barn` | |
| `building=cowshed` | |
| `building=stable` | |
| `building=sty` | |
| `building=cabin` | Isolated cabins |
| `building=house` | When very isolated |
| `building=detached` | When very isolated |
| `building=yes` | Sparse rural buildings |
| `building=ruins` | |
| `building=bunker` | |
| `building=greenhouse` | |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=farmyard` | Working farms |
| `landuse=farmland` | With sparse buildings |
| `landuse=allotments` | With sparse structures |
| `landuse=village_green` | Rural village setting |

**Key: `place`**

| Tag | Notes |
|-----|-------|
| `place=isolated_dwelling` | Strong indicator |
| `place=farm` | |
| `place=hamlet` | |

---

### LCZ 10 – Heavy Industry

Low-rise and mid-rise industrial structures (towers, tanks, stacks). Few or no trees. Land cover mostly paved or hard-packed. Metal, steel, and concrete construction materials.

**Key: `building`**

| Tag | Notes |
|-----|-------|
| `building=industrial` | Strong indicator |
| `building=warehouse` | In industrial context |
| `building=manufacture` | Informal tag |
| `building=yes` | In `landuse=industrial` areas |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=industrial` | Primary indicator |
| `landuse=port` | Deprecated but found in historic data |
| `landuse=quarry` | Extractive industry |
| `landuse=landfill` | Waste management |
| `landuse=brownfield` | Former industrial land |
| `landuse=construction` | Major construction sites |
| `landuse=railway` | Rail yards |

**Key: `man_made`**

| Tag | Notes |
|-----|-------|
| `man_made=works` | Factory/processing plant |
| `man_made=wastewater_plant` | |
| `man_made=water_works` | |
| `man_made=chimney` | Industrial chimney |
| `man_made=storage_tank` | |
| `man_made=silo` | |
| `man_made=gasometer` | |
| `man_made=kiln` | |
| `man_made=petroleum_well` | |
| `man_made=mineshaft` | |
| `man_made=adit` | Mine entrance |
| `man_made=tower` | When type=cooling |
| `man_made=crane` | Port/industrial crane |
| `man_made=pipeline` | |

**Key: `industrial`** (sub-classification)

| Tag | Notes |
|-----|-------|
| `industrial=port` | |
| `industrial=warehouse` | |
| `industrial=factory` | |
| `industrial=manufacturing` | |
| `industrial=oil` | |
| `industrial=gas` | |
| `industrial=refinery` | |
| `industrial=mine` | |
| `industrial=quarry` | |
| `industrial=slaughterhouse` | |
| `industrial=sawmill` | |
| `industrial=scrap_yard` | |
| `industrial=distributor` | |
| `industrial=well_cluster` | |

**Key: `power`**

| Tag | Notes |
|-----|-------|
| `power=plant` | Power stations |
| `power=generator` | |
| `power=substation` | |

**Key: `aeroway`** (airport infrastructure)

| Tag | Notes |
|-----|-------|
| `aeroway=aerodrome` | Airport areas |
| `aeroway=terminal` | |
| `aeroway=hangar` | |
| `aeroway=apron` | |
| `aeroway=taxiway` | |
| `aeroway=runway` | |

**Key: `railway`** (rail infrastructure contributing to ISF)

| Tag | Notes |
|-----|-------|
| `railway=rail` | Mainline railways |
| `railway=light_rail` | |
| `railway=tram` | |
| `railway=narrow_gauge` | |
| `railway=funicular` | |
| `railway=monorail` | |
| `railway=miniature` | |
| `railway=station` | |
| `railway=yard` | Rail yards — strong LCZ 10 indicator |
| `railway=turntable` | |
| `railway=transfer_table` | Found in Fonte et al. (2019) |

---

## LAND COVER TYPES (LCZ A–G)

### LCZ A – Dense Trees

Heavily wooded landscape of deciduous and/or evergreen trees. Land cover mostly pervious (low plants). Zone function is natural forest, tree cultivation, or urban park.

**Key: `natural`**

| Tag | Notes |
|-----|-------|
| `natural=wood` | Natural/semi-natural woodland — strong indicator |
| `natural=tree_row` | Lines of trees (when dense) |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=forest` | Managed forest — strong indicator |
| `landuse=nature_reserve` | When forested (deprecated for `boundary=protected_area` in some areas, but widely used) |
| `landuse=orchard` | Dense tree cultivation (can also be LCZ B or C) |

**Key: `leaf_type`** (sub-attributes useful for seasonal variation)

| Tag | Notes |
|-----|-------|
| `leaf_type=broadleaved` | Deciduous/broadleaf forest |
| `leaf_type=needleleaved` | Coniferous forest |
| `leaf_type=mixed` | Mixed forest |

**Key: `leaf_cycle`**

| Tag | Notes |
|-----|-------|
| `leaf_cycle=deciduous` | Relevant for LCZ seasonal variant (bare trees - b) |
| `leaf_cycle=evergreen` | |
| `leaf_cycle=mixed` | |

**Key: `leisure`**

| Tag | Notes |
|-----|-------|
| `leisure=park` | When heavily wooded |
| `leisure=nature_reserve` | When forested |
| `leisure=garden` | When heavily wooded (e.g., botanical gardens) |

**Key: `boundary`**

| Tag | Notes |
|-----|-------|
| `boundary=national_park` | When forested |
| `boundary=protected_area` | When forested |
| `boundary=forest` | Deprecated but found in historic data |
| `boundary=forest_compartment` | |

> **Note:** OSM does not typically distinguish dense vs scattered trees. Distinguishing LCZ A from LCZ B requires canopy density estimation from remote sensing or additional spatial analysis. Default assignment for `natural=wood` and `landuse=forest` polygons is LCZ A (dense trees). Smaller, fragmented, or narrow tree features may be assigned LCZ B.

---

### LCZ B – Scattered Trees

Lightly wooded landscape of deciduous and/or evergreen trees. Land cover mostly pervious (low plants). Zone function is natural forest, tree cultivation, or urban park.

**Key: `natural`**

| Tag | Notes |
|-----|-------|
| `natural=wood` | When fragmented/small patches |
| `natural=tree_row` | Rows of trees along roads/boundaries |
| `natural=tree` | Individual trees (when aggregated in area) |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=forest` | Small or fragmented forest patches |
| `landuse=orchard` | Spaced fruit/nut trees |
| `landuse=vineyard` | Can have scattered tree cover |
| `landuse=plant_nursery` | Tree nurseries |
| `landuse=recreation_ground` | When tree-lined |
| `landuse=village_green` | When tree-lined |

**Key: `leisure`**

| Tag | Notes |
|-----|-------|
| `leisure=park` | Parks with scattered trees |
| `leisure=garden` | When with scattered trees |
| `leisure=golf_course` | Often has scattered trees |

**Key: `natural` (deprecated/historic)**

| Tag | Notes |
|-----|-------|
| `natural=trees` | Deprecated plural form; found in older data |

---

### LCZ C – Bush, Scrub

Open arrangement of bushes, shrubs, and short, woody trees. Land cover mostly pervious (bare soil or sand). Zone function is natural scrubland or agriculture.

**Key: `natural`**

| Tag | Notes |
|-----|-------|
| `natural=scrub` | Strong indicator |
| `natural=heath` | Heathland — strong indicator |
| `natural=shrub` | Deprecated; use `natural=scrub` |
| `natural=moor` | Deprecated; was used for moorland/heathland |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=heath` | Deprecated in some areas but widely used |
| `landuse=orchard` | Short/bush orchards |
| `landuse=vineyard` | Vine cultivation |
| `landuse=scrub` | Deprecated variant — found in historic data |
| `landuse=scrubs` | Typo variant found in data (Fonte et al. 2019) |
| `landuse=plant_nursery` | Shrub nurseries |

---

### LCZ D – Low Plants

Featureless landscape of grass or herbaceous plants/crops. Few or no trees. Zone function is natural grassland, agriculture, or urban park.

**Key: `natural`**

| Tag | Notes |
|-----|-------|
| `natural=grassland` | Natural grassland — strong indicator |
| `natural=grass` | Deprecated; prefer `landuse=grass` or `landcover=grass` |
| `natural=fell` | Mountain grassland/tundra |
| `natural=meadow` | Deprecated; prefer `landuse=meadow` |
| `natural=wetland` | When herbaceous (with `wetland=marsh` or `wetland=fen`) |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=farmland` | Arable crops — strong indicator |
| `landuse=farm` | Deprecated; replaced by `landuse=farmland` |
| `landuse=meadow` | Meadow/pasture — strong indicator |
| `landuse=grass` | Maintained grass areas |
| `landuse=farmyard` | Without significant buildings |
| `landuse=greenfield` | Undeveloped green land |
| `landuse=recreation_ground` | Sports fields, open green |
| `landuse=village_green` | Open village green |
| `landuse=allotments` | When mostly open |
| `landuse=pasture` | Deprecated; use `landuse=meadow` + `meadow=pasture` |
| `landuse=field` | Deprecated; use `landuse=farmland` |
| `landuse=crop` | Deprecated variant |

**Key: `leisure`**

| Tag | Notes |
|-----|-------|
| `leisure=park` | When mostly grass (no trees) |
| `leisure=pitch` | Sports pitches |
| `leisure=golf_course` | When open/grassy |
| `leisure=garden` | When open |
| `leisure=playground` | When grassy |
| `leisure=common` | Common land |
| `leisure=sports_centre` | When outdoor |

**Key: `landcover`** (less common but growing)

| Tag | Notes |
|-----|-------|
| `landcover=grass` | Explicit ground cover tag |
| `landcover=cropland` | Proposed |

**Key: `crop`** (sub-attribute on farmland)

| Tag | Notes |
|-----|-------|
| `crop=*` | Any value indicates agricultural LCZ D |

---

### LCZ E – Bare Rock or Paved

Featureless landscape of rock or paved cover. Few or no trees or plants. Zone function is natural desert (rock) or urban transportation.

**Key: `natural`**

| Tag | Notes |
|-----|-------|
| `natural=bare_rock` | Rock surface — strong indicator |
| `natural=scree` | Loose rock/talus |
| `natural=rock` | Exposed rock formation |
| `natural=stone` | Individual large stones |
| `natural=cliff` | Rock cliff faces |
| `natural=ridge` | Rocky ridges |

**Key: `surface`** (on large paved areas)

| Tag | Notes |
|-----|-------|
| `surface=asphalt` | Paved surface |
| `surface=concrete` | |
| `surface=paved` | Generic paved |
| `surface=paving_stones` | |
| `surface=sett` | Cobblestone |
| `surface=cobblestone` | Deprecated; prefer `surface=sett` or `surface=unhewn_cobblestone` |
| `surface=unhewn_cobblestone` | |
| `surface=metal` | |

**Key: `highway`** (roads contributing to impervious surface)

| Tag | Notes |
|-----|-------|
| `highway=motorway` | Major roads |
| `highway=trunk` | |
| `highway=primary` | |
| `highway=secondary` | |
| `highway=tertiary` | |
| `highway=residential` | |
| `highway=service` | |
| `highway=unclassified` | |
| `highway=primary_link` | |
| `highway=secondary_link` | |
| `highway=tertiary_link` | |
| `highway=trunk_link` | |
| `highway=living_street` | |
| `highway=pedestrian` | Pedestrian plazas |
| `highway=road` | Generic/unspecified road |
| `highway=raceway` | Race tracks |
| `highway=bus_guideway` | Found in Fonte et al. (2019) |

**Key: `amenity`** (large paved surfaces)

| Tag | Notes |
|-----|-------|
| `amenity=parking` | Parking lots |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=highway` | Proposed but not widely adopted |

**Key: `aeroway`** (paved airport surfaces)

| Tag | Notes |
|-----|-------|
| `aeroway=runway` | |
| `aeroway=taxiway` | |
| `aeroway=apron` | |

---

### LCZ F – Bare Soil or Sand

Featureless landscape of soil or sand cover. Few or no trees or plants. Zone function is natural desert or agriculture.

**Key: `natural`**

| Tag | Notes |
|-----|-------|
| `natural=sand` | Sand areas — strong indicator |
| `natural=beach` | Sandy/pebbly beaches |
| `natural=dune` | Sand dunes |
| `natural=desert` | Deprecated; use `natural=sand` or `natural=bare_rock` |
| `natural=mud` | Mudflats |
| `natural=shingle` | Pebble/gravel beaches |

**Key: `surface`** (on large unpaved areas)

| Tag | Notes |
|-----|-------|
| `surface=sand` | |
| `surface=dirt` | |
| `surface=earth` | |
| `surface=ground` | |
| `surface=mud` | |
| `surface=gravel` | |
| `surface=fine_gravel` | |
| `surface=compacted` | |
| `surface=unpaved` | Generic unpaved |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=brownfield` | Cleared/derelict land (when bare) |
| `landuse=quarry` | When exposing bare soil/rock |
| `landuse=landfill` | When bare/exposed |
| `landuse=construction` | When bare/cleared ground |

**Key: `geological`**

| Tag | Notes |
|-----|-------|
| `geological=moraine` | Glacial deposits |

---

### LCZ G – Water

Large, open water bodies such as seas and lakes, or small bodies such as rivers, reservoirs, and lagoons.

**Key: `natural`**

| Tag | Notes |
|-----|-------|
| `natural=water` | Any water body — strong indicator |
| `natural=bay` | Bays |
| `natural=strait` | Straits |
| `natural=coastline` | Coastline delineation |
| `natural=spring` | Springs (point features) |
| `natural=hot_spring` | |

**Key: `water`** (sub-classification of `natural=water`)

| Tag | Notes |
|-----|-------|
| `water=lake` | |
| `water=reservoir` | |
| `water=pond` | |
| `water=river` | River areas |
| `water=canal` | Canal areas |
| `water=stream` | Deprecated as area; use `waterway=stream` |
| `water=lagoon` | |
| `water=oxbow` | |
| `water=moat` | |
| `water=basin` | |
| `water=wastewater` | |

**Key: `waterway`** (linear features — must be buffered to create area)

| Tag | Notes |
|-----|-------|
| `waterway=river` | Rivers |
| `waterway=stream` | Streams |
| `waterway=canal` | Canals |
| `waterway=drain` | Drainage channels |
| `waterway=ditch` | Ditches |
| `waterway=brook` | Small streams (deprecated; use `waterway=stream`) |
| `waterway=riverbank` | Deprecated; river area polygons. Use `natural=water` + `water=river` |
| `waterway=dock` | Docks |
| `waterway=boatyard` | |
| `waterway=dam` | |
| `waterway=lock_gate` | |
| `waterway=waterfall` | |
| `waterway=rapids` | Deprecated |

**Key: `landuse`**

| Tag | Notes |
|-----|-------|
| `landuse=reservoir` | Deprecated; use `natural=water` + `water=reservoir` |
| `landuse=basin` | Water basins (deprecated) |
| `landuse=harbour` | Port/harbour areas (deprecated) |
| `landuse=port` | Deprecated variant |
| `landuse=salt_pond` | Salt evaporation ponds |

**Key: `leisure`**

| Tag | Notes |
|-----|-------|
| `leisure=swimming_pool` | Pools |
| `leisure=marina` | Marinas |
| `leisure=fishing` | Fishing areas (when water-body) |

---

## VARIABLE LAND COVER PROPERTIES

These modifiers apply seasonally to the base LCZ classes.

### LCZ variant (b) – Bare Trees
Leafless deciduous trees (e.g., winter). Relevant tags: `leaf_cycle=deciduous` on `natural=wood` or `landuse=forest`.

### LCZ variant (s) – Snow Cover
Snow cover >10 cm depth. Potentially identifiable via remote sensing, not directly from OSM tags. `surface=snow` exists for paths but is rarely used on land areas.

### LCZ variant (d) – Dry Ground
Parched soil. Not directly tagged in OSM.

### LCZ variant (w) – Wet Ground
Waterlogged soil. Relevant tags:

| Tag | Notes |
|-----|-------|
| `natural=wetland` | General wetland |
| `wetland=marsh` | |
| `wetland=swamp` | |
| `wetland=bog` | |
| `wetland=fen` | |
| `wetland=reedbed` | |
| `wetland=wet_meadow` | |
| `wetland=mangrove` | |
| `wetland=saltmarsh` | |
| `wetland=tidalflat` | |

---

## ADDITIONAL TAGS FOR HEIGHT DISCRIMINATION

These tags are critical for distinguishing between compact/open and high/mid/low-rise classes.

| Tag | Description |
|-----|-------------|
| `building:levels=*` | Number of above-ground floors |
| `building:min_level=*` | Lowest level (for elevated buildings) |
| `height=*` | Building height in metres |
| `building:height=*` | Alternative height tag |
| `roof:levels=*` | Number of roof levels |
| `roof:height=*` | Height of roof structure |
| `building:levels:underground=*` | Underground floors |
| `building:flats=*` | Number of residential units |
| `stories=*` | Deprecated; use `building:levels` |
| `floors=*` | Deprecated; use `building:levels` |

---

## DEPRECATED AND HISTORIC TAGS

These tags are no longer recommended but appear in older OSM data extracts and historical snapshots.

| Deprecated Tag | Replacement | Relevance |
|----------------|-------------|-----------|
| `natural=grass` | `landuse=grass` | LCZ D |
| `natural=meadow` | `landuse=meadow` | LCZ D |
| `natural=desert` | `natural=sand` / `natural=bare_rock` | LCZ E/F |
| `natural=trees` | `natural=wood` | LCZ A/B |
| `natural=moor` | `natural=heath` | LCZ C |
| `natural=shrub` | `natural=scrub` | LCZ C |
| `landuse=farm` | `landuse=farmland` | LCZ D |
| `landuse=pasture` | `landuse=meadow` + `meadow=pasture` | LCZ D |
| `landuse=field` | `landuse=farmland` | LCZ D |
| `landuse=crop` | `landuse=farmland` | LCZ D |
| `landuse=reservoir` | `natural=water` + `water=reservoir` | LCZ G |
| `landuse=basin` | `natural=water` + `water=basin` | LCZ G |
| `landuse=harbour` | Use other tags | LCZ 10/G |
| `landuse=port` | Use other tags | LCZ 10 |
| `landuse=scrubs` | `natural=scrub` | LCZ C |
| `landuse=scrub` | `natural=scrub` | LCZ C |
| `waterway=riverbank` | `natural=water` + `water=river` | LCZ G |
| `waterway=brook` | `waterway=stream` | LCZ G |
| `waterway=rapids` | Removed | LCZ G |
| `building=public_building` | `building=public` | LCZ 2–6 |
| `building=slum` | No replacement | LCZ 7 |
| `building=skyscraper` | `building=*` + `building:levels` | LCZ 1/4 |
| `stories=*` | `building:levels=*` | Height |
| `floors=*` | `building:levels=*` | Height |
| `cobblestone` (surface) | `surface=sett` or `surface=unhewn_cobblestone` | LCZ E |
| `boundary=forest` | `landuse=forest` / `natural=wood` | LCZ A |

---

## REFERENCES

- Stewart, I.D. and Oke, T.R. (2012). Local Climate Zones for Urban Temperature Studies. *Bulletin of the American Meteorological Society*, 93, 1879–1900.
- Fonte, C., Lopes, P., See, L. and Bechtel, B. (2019). Using OpenStreetMap (OSM) to enhance the classification of Local Climate Zones in the framework of WUDAPT. *Urban Climate*, 28, 100456.
- Bocher, E., Bernard, J., Wiederhold, E., Leconte, F., Petit, G., Palominos, S. and Noûs, C. (2021). GeoClimate: a Geospatial processing toolbox for environmental and climate studies. *Journal of Open Source Software*, 6(65), 3541. <https://doi.org/10.21105/joss.03541>
- GeoClimate documentation — input data layers (building, road, rail, vegetation, water, impervious) and the LCZ classification chain: <https://geoclimate.readthedocs.io/>
- OpenStreetMap Wiki: Key:landuse, Key:natural, Key:building, Key:surface, Key:waterway, Key:man_made, Buildings, Map features, Deprecated features.
- Lopes, P., Fonte, C., See, L. and Bechtel, B. (2017). Using OpenStreetMap data to assist in the creation of LCZ maps. *2017 Joint Urban Remote Sensing Event (JURSE)*.
