# Are the WUDAPT labels usable? A per-city and per-region assessment

**Measured 2026-09-08** against the QC label set built in
[`wudapt_qc_labels.md`](wudapt_qc_labels.md) — 45,760 patches and 915 AOIs,
plus the 10 m segmentation rasters covering 1,162 AOIs.

Reproduce with `python -m lcz_wudapt suitability`.

## The question

Producing labels is not the same as producing *useful* labels, and "useful"
turns out to mean something quite different for the two training pathways. Three
things decide it:

1. **Capacity.** Patch classification needs 320 m patches and enough of them per
   class; segmentation needs labelled km² and grid cells carrying enough label.
   These diverge much more than they sound like they should — see below.
2. **Trust.** Where So2Sat exists, class agreement on overlapping ground is a
   direct per-city trust score. Where it does not — the ~1,100 AOIs that are the
   entire point of this exercise — `nbr_conflict` stands in, licensed by its
   Spearman **+0.929** correlation with So2Sat agreement on the audit cities.
3. **Leakage.** In the 51 So2Sat cities WUDAPT patches physically overlap So2Sat
   patches, and the culture-10 carry the benchmark every headline number is
   measured on.

## The finding that drives everything else

**Roughly 60% of QC-passing polygons cannot contain a 320 m square.** The patch
arm discards them outright; segmentation keeps every interior pixel. So the two
pathways see wildly different amounts of the same data:

| | patch arm | segmentation arm |
|---|---|---|
| Nairobi | 73 patches, 9 classes, 7.5 km² | 15 classes, **38.7 km²** |
| Buenos Aires | **4 patches**, 3 classes | 15 classes, **22.9 km²** |
| Bogotá | 11 patches, 4 classes | **16 classes**, 18.8 km² |

Buenos Aires is the clearest case: four patches is nothing, 23 km² across 15
classes is a usable city. Any judgement of "is WUDAPT good enough here?" that
looks only at patch counts will be wrong by an order of magnitude for the
sparsest cities — which are exactly the cities the label set exists to rescue.

## Tier 1 — the 10 held-out culture cities

| City | patches | classes | clean¹ | agree² | pairs | mean w | conflict | **seg km²** | seg cls | train cells ≥5% |
|---|---|---|---|---|---|---|---|---|---|---|
| Guangzhou | 808 | 16 | 711 | 0.931 | — | 0.26 | 0.63 | **1113** | 17 | 1214 |
| Jakarta | 491 | 14 | 386 | 0.861 | — | 0.35 | 0.34 | **543** | 17 | 839 |
| Santiago | 398 | 14 | 253 | 0.892 | — | 0.32 | 0.48 | 151 | 16 | 298 |
| Tehran | 373 | 14 | 283 | **0.464** | 90 | 0.27 | **0.82** | 401 | 17 | 520 |
| Sydney | 213 | 12 | 122 | 0.950 | — | 0.27 | 0.30 | **541** | 17 | 527 |
| San Jose | 192 | 15 | 174 | **0.419** | 18 | 0.42 | 0.26 | 361 | 16 | 463 |
| Munich | 173 | 12 | 121 | 0.981 | — | 0.30 | 0.54 | 98 | 14 | 158 |
| Mumbai | 170 | 12 | 92 | 0.623 | 78 | 0.47 | 0.24 | 128 | 16 | 280 |
| Moscow | 92 | 12 | 56 | 1.000 | — | 0.36 | 0.00 | 86 | 15 | 177 |
| Nairobi | 73 | 9 | 38 | 0.910 | — | 0.39 | 0.22 | 39 | 15 | 95 |

¹ patches not overlapping any So2Sat patch ² class agreement where they do overlap

**Classification: viable in 4, marginal in 4, not viable in 2.** Only Guangzhou,
Jakarta, Santiago and Tehran clear ~250 leakage-safe patches. Nairobi (38) and
Moscow (56) support a demonstration, not a result.

**The trust score has a cliff in it.** Moscow 1.00, Munich 0.98, Sydney 0.95,
Guangzhou 0.93, Nairobi 0.91, Santiago 0.89, Jakarta 0.86 — then **Mumbai 0.62,
Tehran 0.46, San Jose 0.42**. Two different situations:

* **Tehran is systematically contested** — 0.46 agreement over 90 overlapping
  patches with `nbr_conflict` 0.82. This is a real disagreement, not sampling
  noise, and it reproduces the confidence inversion diagnosed during the earlier
  consensus work. Exclude Tehran from any arm that treats WUDAPT as ground truth.
* **San Jose is unmeasured, not bad** — 0.42 rests on only 18 overlapping
  patches. Report it as unknown rather than as a low score.

**Segmentation is viable in all 10.** Sydney makes the point: 213 patches (weak)
but 541 km² across 17 classes (strong).

## Tier 2 — So2Sat-sparse cities

Every one of these is single-class in So2Sat. This is where WUDAPT adds the most
per unit of effort, and where the two pathways diverge hardest.

| City | So2Sat | WUDAPT patches | classes | **seg km²** | seg cls | verdict |
|---|---|---|---|---|---|---|
| Salvador | **1 patch, 1 class** | 137 | 12 | 107 | **17** | both |
| Chicago | 48, 1 class | 61 | 11 | 497 | 14 | both |
| Lima | 48, 1 class | 53 | 6 | 25 | 8 | segmentation only |
| Bogotá | 8, 1 class | 11 | 4 | 19 | **16** | segmentation only |
| Caracas | 12, 1 class | 10 | 4 | 15 | 10 | segmentation only |
| Buenos Aires | 5, 1 class | **4** | 3 | 23 | **15** | segmentation only |

**Salvador is the headline** — So2Sat contributes literally one patch of one
class; WUDAPT turns it into a usable city on both pathways. **Classification
rescues 2 of the 6; segmentation rescues all 6.**

Tier-2 trust scores are mostly unavailable: with So2Sat contributing single
digits, there is almost no overlapping ground to compare against (Chicago's 1.00
rests on 4 patches). Fall back to `nbr_conflict`, which is reassuringly low here
— 0.00 for Lima, Bogotá, Caracas and Buenos Aires.

## Tier 3 — regions with no So2Sat coverage at all

| Region | AOIs w/ patches | AOIs w/ raster | countries | patches | AOIs ≥50p | classes | conflict | **seg km²** | AOIs ≥5 km² | 0.1° tiles |
|---|---|---|---|---|---|---|---|---|---|---|
| **West Africa** | 36 | 39 | 10 | **4,180** | 18 | 16 | **0.19** | **2,033** | 22 | 207 |
| India | 92 | **132** | 1 | 2,381 | 9 | 17 | **0.46** | 1,587 | 40 | 326 |
| Southeast Asia | 20 | 23 | 7 | 1,493 | 17 | 17 | 0.31 | 1,324 | 16 | 186 |
| Central America | 7 | 10 | 6 | 324 | 2 | 15 | 0.04 | 167 | 6 | 44 |

The two AOI columns differ for the same reason the two pathways differ: an AOI
whose polygons all fail the 320 m containment test produces **zero patches and a
perfectly good label raster**. India has 92 AOIs with patches against **132 with
a raster** — 40 Indian cities are segmentation-only. Counting rasters only over
AOIs that have patches undercounts segmentation capacity, which is a mistake this
assessment made on its first pass and `region_suitability` now avoids.

Leading AOIs: **West Africa** — Lagos 767, Accra 564, Abidjan 398,
Ouagadougou 334, Bamako 285, Lomé 269. **India** — Delhi 483, Kolkata 351,
Bhopal 219, Chennai 149, Pune 104. **Southeast Asia** — Jakarta 491,
Singapore 256, Surabaya 212, Kuala Lumpur 174. **Central America** — Santo
Domingo 99, Havana 90, Managua 49.

* **West Africa is the strongest case in the entire dataset.** 4,180 patches,
  2,033 km², 10 countries, and the *lowest* conflict rate of any region measured
  — 0.19, below Europe's.
* **India is the largest AOI count but the least trustworthy**: conflict 0.46
  overall, Delhi 0.72, Pune 0.69, Bhopal 0.55. Sweep `nbr_dist_m` before
  trusting it.
* **Central America is too thin to plan around** — 7 AOIs, two of them
  meaningful.

### What tier 3 supplies that So2Sat structurally lacks

Class share, % of each region's patches:

| LCZ | Central Am. | W Africa | India | SE Asia | So2Sat |
|---|---|---|---|---|---|
| 1 compact high-rise | 0.0 | 0.1 | 0.1 | 0.8 | **1.4** |
| 2 compact mid-rise | 4.0 | 1.7 | 2.8 | 2.0 | 6.7 |
| **3 compact low-rise** | 20.1 | **30.4** | 16.7 | 16.5 | 9.1 |
| **6 open low-rise** | 21.6 | **25.1** | 11.1 | 13.5 | 9.8 |
| **7 lightweight low-rise** | 0.3 | **8.3** | 1.1 | 0.1 | 1.1 |
| 8 large low-rise | 3.1 | 3.3 | 4.3 | 9.1 | 11.5 |
| 10 heavy industry | 0.9 | 0.6 | **8.6** | 5.2 | 3.4 |
| 17 water | 17.6 | 4.5 | 14.9 | 6.8 | 13.6 |

**West Africa alone carries 8.3% LCZ 7** against So2Sat's 1.1% — the
lightweight-low-rise class of which the Demuzere pseudo-label pool kept
*zero*. Compact and open low-rise run 2–3× So2Sat's share, which is precisely
the Global South morphology the benchmark under-samples.

The mirror image is a genuine limit: **LCZ 1 is essentially absent outside the
So2Sat cities** (0.0–0.8%). Tier 3 will not fix compact high-rise; that has to
keep coming from So2Sat.

## Verdict

1. **Segmentation dominates in all three tiers, and the gap widens as So2Sat
   coverage falls.** In tier 2 it is the difference between unusable and usable.
   If only one arm is run, run that one.
2. **Classification is worth it in ~4 tier-1 cities, 2 tier-2 cities, and the top
   ~40 tier-3 AOIs** (West Africa and Southeast Asia mainly). Elsewhere the
   patch counts sit below what few-shot can use.
3. **Trust per city, never globally.** Use `agree_on_overlap` where So2Sat
   exists, `nbr_conflict` where it does not. Quarantine Tehran; treat Delhi and
   Pune as provisional; record San Jose as unmeasured rather than poor.
4. **~763 tiles at 0.1°** covers all four tier-3 regions for a Tessera v2 or coop
   request (West Africa 207, India 326, SE Asia 186, Central America 44).

Suggested order of work, by return on effort: **West Africa segmentation** →
tier-2 segmentation (six cities rescued for almost no cost) → Southeast Asia →
tier-1 classification on the four viable cities → India, after the conflict
sweep.

## Two caveats before acting on this

**The split assignment already placed several strong tier-3 AOIs in `test`** —
Bamako, Kolkata, Bhopal, Havana, Managua, Jakarta, Mumbai among them. Check
`wudapt_split` before planning to train on a named city; the assignment is
city-disjoint and region-stratified by design, and moving a city between splits
invalidates comparisons.

**Tier-1 segmentation numbers assume the grid-alignment fix.** `create_city_grids.py`
resolves a city bbox by `city.replace("_", " ")` against
`data/so2sat_guppd_bounds.csv`, which does not contain the `__SMOD_ID`-suffixed
WUDAPT AOI names, so it falls back to the label extent and builds a **different**
3×3 checkerboard from So2Sat's. Measured on Nairobi, that puts 1,327 So2Sat
*testing* and 1,124 *validation* patches under WUDAPT training tiles. Aligning
the two grids (reusing the So2Sat bbox for the 51 shared AOIs) removes it by
construction. **Implemented 2026-10-06:** `create_city_grids.py` now matches a
`{City}__{SMOD_ID}` AOI to the So2Sat bbox by its SMOD_ID (not its name, which
repeats across countries), so the 51 shared AOIs get So2Sat's grid. Existing
WUDAPT grids predate it and must be rebuilt with `--overwrite`. Residual caveat:
the UTM zone is still estimated from each label set, so an AOI whose polygons
straddle a zone edge differently from So2Sat's could still land in another CRS.

For the patch arm the equivalent control is simply to drop the overlapping
patches — the `w_clean` column above is that count.
