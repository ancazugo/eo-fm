### Amendment B4 — Multi-zone mosaic defect, scoped (measure only)

Every `tesserav1.1_global` patch on disk (390,680) tested for how many tiles it intersects and how many UTM zones those tiles span, against 8,108 exact footprints. Generated 2026-08-12T22:21:56+00:00.

Task 2.0d reported that 294 of 302 truncated-but-accepted patches sit at a zone seam. That is a numerator; this is the denominator.

#### The key ratio

**Of the 760 patches that straddle a UTM zone boundary, 168 come back truncated — 22.1%.**

| population | n | truncated | rate |
|---|---|---|---|
| all on disk | 390,680 | 320 | 0.08% |
| single tile | 353,386 | 149 | 0.04% |
| multiple tiles | 37,294 | 171 | 0.46% |
| multiple tiles, one zone | 36,534 | 3 | 0.01% |
| straddling a zone boundary | 760 | 168 | 22.11% |

Read the other way: 52.5% of all truncated patches straddle a zone, and 53.4% span multiple tiles.

Under Phase 1.75's stricter 0.85-per-side rule the straddling rate is 38.9% (296 of 760) — reported alongside because Task 2.0a showed the two rules disagree by design.

**Straddling a zone truncates a minority of crops** — 22.1% against a 0.04% baseline for single-tile patches. Elevated by orders of magnitude but not deterministic, so a further trigger selects which straddling patches fail; identifying it is the Phase 4 prerequisite.

#### Zone-straddling patches by city

| city | straddling | truncated | rate |
|---|---|---|---|
| London | 271 | 75 | 27.7% |
| Guangzhou | 139 | 24 | 17.3% |
| Hong Kong | 68 | 16 | 23.5% |
| Qingdao | 49 | 11 | 22.4% |
| Wuhan | 20 | 1 | 5.0% |

#### Truncated across tiles but *within* one zone

3 patches, which `crop_patch` sends through `numpy_mosaic` rather than `merge_multi_crs`. Rev B asks whether they share a property that would indicate a second mechanism.

- cities: Shanghai (3)
- splits: training (3)
- tiles per patch: {2: 3}
- crop shapes: 7x33
- median native_frac: 0.206

#### Absent patches, for contrast

9,993 reference patches have no npy at all; 3 of those straddle a zone. Absence is a coverage failure, not a mosaic one (Task 2.0d), so they are counted here but excluded from the ratio above — a patch that never came back cannot have come back truncated.

**Nothing was changed.** The mosaic is not fixed in Phase 2; this is a GATE 3 input beside the re-extraction recovery count.
