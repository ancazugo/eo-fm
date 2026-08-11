### Task 1.5.0 — Resize audit

2000 patches per family from the cultural-split **training** set (each family sampled over its own ids, seed 0), resized to 32x32.

#### Geometry — the resize is essentially never a no-op

| family | native shapes (top 3) | exact no-op | H factor (med) | W factor (med) |
|---|---|---|---|---|
| `alpha_earth_coop` | 33x34 (674), 34x33 (547), 33x33 (319) | 0.0% | 1.031 | 1.031 |
| `tesserav1.1_global` | 33x34 (571), 34x33 (517), 33x33 (359) | 0.0% | 1.031 | 1.031 |
| `tesserav1.1` | 35x33 (1104), 36x33 (627), 36x32 (129) | 0.0% | 1.094 | 1.031 |
| `tesserav2` | 33x34 (496), 34x33 (483), 33x33 (386) | 0.0% | 1.031 | 1.031 |
| `seamless` | 12x12 (1027), 11x12 (327), 12x11 (324) | 0.0% | 0.375 | 0.375 |
| `sentinel1` | 32x32 (2000) | 100.0% | 1.000 | 1.000 |
| `sentinel2` | 32x32 (2000) | 100.0% | 1.000 | 1.000 |

A factor > 1 is a downsample, < 1 an upsample.

#### Sentinel variance decomposition — five variants

| family | invalid px | native unmasked | native masked | resized unmasked | resized masked | resized masked+filled |
|---|---|---|---|---|---|---|
| `alpha_earth_coop` | 0.4817% | 0.1266 | 0.1051 | 0.1259 | 0.1047 | 0.1047 |
| `tesserav1.1_global` | 0.1543% | 1.1674 | 1.1675 | 1.1349 | 1.1356 | 1.1356 |
| `tesserav1.1` | 0.1341% | 0.8132 | 0.8120 | 0.7965 | 0.7956 | 0.7956 |
| `tesserav2` | 0.1050% | 0.3890 | 0.3891 | 0.3806 | 0.3808 | 0.3808 |
| `seamless` | 0.0000% | 0.5682 | 0.5682 | 0.5524 | 0.5524 | — |
| `sentinel1` | 0.0000% | 0.3645 | 0.3645 | 0.3645 | 0.3645 | — |
| `sentinel2` | 0.0000% | 0.0817 | 0.0817 | 0.0817 | 0.0817 | — |

Sentinel share of per-channel variance, native vs post-resize (unfilled):

| family | native | post-resize | change |
|---|---|---|---|
| `alpha_earth_coop` | 31.04% | 30.86% | -0.18% |
| `tesserav1.1_global` | 0.00% | 0.00% | +0.00% |
| `tesserav1.1` | 0.30% | 0.24% | -0.06% |
| `tesserav2` | 0.00% | 0.00% | +0.00% |
| `seamless` | 0.00% | 0.00% | +0.00% |
| `sentinel1` | 0.00% | 0.00% | +0.00% |
| `sentinel2` | 0.00% | 0.00% | +0.00% |

#### Paired vs unpaired — where the nodata actually lives

`paired` = the patch_id is present in every one of `alpha_earth_coop`, `tesserav1.1_global`, `seamless`, `sentinel1`, `sentinel2`, i.e. it would have been eligible for the GATE 0 sample.

| family | paired n | paired invalid px | unpaired n | unpaired invalid px |
|---|---|---|---|---|
| `alpha_earth_coop` | 1941 | 0.0184% | 59 | 15.7292% |
| `tesserav1.1_global` | 1998 | 0.1545% | 2 | 0.0000% |
| `tesserav1.1` | 2000 | 0.1341% | 0 | 0.0000% |
| `tesserav2` | 1976 | 0.1062% | 24 | 0.0000% |
| `seamless` | 1945 | 0.0000% | 55 | 0.0000% |
| `sentinel1` | 1942 | 0.0000% | 58 | 0.0000% |
| `sentinel2` | 1937 | 0.0000% | 63 | 0.0000% |

#### L2 norm, native vs post-resize (valid pixels)

| family | native p1 / p50 / p99 | resized p1 / p50 / p99 |
|---|---|---|
| `alpha_earth_coop` | 0.995 / 1.000 / 1.005 | 0.987 / 0.998 / 1.003 |
| `tesserav1.1_global` | 7.943 / 15.594 / 36.132 | 7.929 / 15.420 / 35.606 |
| `tesserav1.1` | 6.588 / 12.989 / 22.722 | 6.554 / 12.746 / 22.558 |
| `tesserav2` | 11.295 / 11.314 / 11.333 | 11.154 / 11.279 / 11.319 |
| `seamless` | 4.077 / 5.214 / 6.828 | 3.772 / 5.000 / 6.800 |
| `sentinel1` | 0.048 / 0.333 / 3.283 | 0.049 / 0.331 / 3.113 |
| `sentinel2` | 0.099 / 0.466 / 1.150 | 0.099 / 0.467 / 1.143 |
