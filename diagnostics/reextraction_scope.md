### Task 2.0d — Re-extraction scoping (measure only)

The fixed `_fully_covered` from a958a3f replayed over `tesserav1.1_global`'s absent and truncated patches, against a tile index of 8,108 exact footprints. Generated 2026-08-12T22:25:16+00:00.

**Nothing was re-extracted.** Task 2.1's anchor is tied to the current extraction; per PLAN-v3 Rev A the decision is deferred to GATE 3 with these counts on the record.

Patches **absent** from disk — how many a re-extraction would bring back. `no tile at all` means the STRtree returns no candidate whatsoever: there is no Tessera tile over that ground, so no coverage fix can help.

| split | absent | would recover | correctly rejected | no tile at all |
|---|---|---|---|---|
| training | 9,422 | 191 (2.0%) | 9,199 | 9,144 |
| validation | 241 | 0 (0.0%) | 241 | 241 |
| testing | 330 | 0 (0.0%) | 330 | 319 |

`would recover` excludes patches whose only coverage comes from the corrupt tile `grid_121.35_31.25` — 40 patches, which a coverage fix cannot help.

#### Truncated patches — does the fix reject them?

This is the direct test of the bug, and the honest framing for these patches is rejection rather than recovery: one that stays accepted would be re-extracted and come back truncated again. A truncated crop is one the old bounding-box test wrongly accepted, so the exact footprint should now reject it.

| split | truncated | now rejected | still accepted |
|---|---|---|---|
| training | 560 | 258 (46.1%) | 302 |
| testing | 17 | 17 (100.0%) | 0 |

**302 truncated patches are still accepted by the exact footprint, and the coverage test is right about them** — 294 of 302 (97.4%) straddle a **UTM zone boundary**, and every one spans more than one tile. The ground genuinely is covered; the truncation happens later, in the multi-tile mosaic, which is the known multi-UTM-zone `crop_patch` defect rather than anything the footprint fix touches. Their shapes bear that out — full extent on one axis and 5-8 px on the other, cut at the zone seam.

The cities are exactly the ones a zone seam predicts: London (118), Guangzhou (65), Hong Kong (22), Qingdao (17), Shanghai (8) — the prime meridian, 114°E and 120°E.

**So a re-extraction on today's code would reproduce these 302, not fix them.** Removing them needs the mosaic defect fixed as well, which is a second change and a second re-run. That belongs in the GATE 3 decision alongside the recovery count.

#### Where the recoverable patches are (all splits, absent patches)

**Bold** = one of the 10 held-out cultural-split cities.

| city | absent | would recover | share |
|---|---|---|---|
| London | 111 | 111 | 100.0% |
| Qingdao | 542 | 26 | 4.8% |
| **Guangzhou** | 26 | 22 | 84.6% |
| Hong Kong | 7 | 7 | 100.0% |
| Wuhan | 6 | 6 | 100.0% |
| Istanbul | 2,617 | 0 | 0.0% |
| Amsterdam | 107 | 0 | 0.0% |
| Cape Town | 983 | 0 | 0.0% |
| Melbourne | 914 | 0 | 0.0% |
| Lisbon | 71 | 0 | 0.0% |
| New York | 2,026 | 0 | 0.0% |
| **Mumbai** | 242 | 0 | 0.0% |
| Shanghai | 358 | 0 | 0.0% |
| **Sydney** | 329 | 0 | 0.0% |
| Vancouver | 1,094 | 0 | 0.0% |
| 东营区 | 70 | 0 | 0.0% |

#### By LCZ class

| LCZ | absent | would recover | share |
|---|---|---|---|
| 14 | 51 | 51 | 100.0% |
| 6 | 43 | 43 | 100.0% |
| 4 | 33 | 33 | 100.0% |
| 8 | 26 | 26 | 100.0% |
| 11 | 14 | 14 | 100.0% |
| 17 | 9,690 | 9 | 0.1% |
| 12 | 9 | 9 | 100.0% |
| 2 | 38 | 6 | 15.8% |
| 16 | 89 | 0 | 0.0% |
