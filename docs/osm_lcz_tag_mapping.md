# OSM Tag Mapping to Local Climate Zone (LCZ) Classes

## Comprehensive Reference for Label Generation from OpenStreetMap Data

This document provides an exhaustive mapping of OpenStreetMap (OSM) tags to the 17 Local Climate Zone classes defined by Stewart and Oke (2012). It includes current, historic, and deprecated tags sourced from the OSM Wiki, TagInfo, and the literature (notably Fonte et al. 2019). The mappings are organized by LCZ class, with tags grouped by OSM key.

---

## Important Notes on OSM-to-LCZ Conversion

**Building height is critical for differentiating built types.** OSM tags alone (without `building:levels` or `height`) cannot distinguish between compact/open high-rise (LCZ 1/4), mid-rise (LCZ 2/5), and low-rise (LCZ 3/6). The building footprint density (Building Surface Fraction, BSF) and impervious surface coverage (Impervious Surface Fraction, ISF) help narrow candidates, but height data is required for final assignment.

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
- OpenStreetMap Wiki: Key:landuse, Key:natural, Key:building, Key:surface, Key:waterway, Key:man_made, Buildings, Map features, Deprecated features.
- Lopes, P., Fonte, C., See, L. and Bechtel, B. (2017). Using OpenStreetMap data to assist in the creation of LCZ maps. *2017 Joint Urban Remote Sensing Event (JURSE)*.
