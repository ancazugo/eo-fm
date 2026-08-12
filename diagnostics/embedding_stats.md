### Task 0.1 — Embedding scale audit

Cultural-split **training** set, 5000 patches (paired across families, seed 0), measured after the exact training-path transform (nan_to_num -> dequantize -> bilinear resize to 32x32).

| family | C | per-channel std (median) | std min | std max | L2 norm p1 / p50 / p99 | frac zero (raw) | frac NaN (raw) | all-zero px | `0.05 / median_std` |
|---|---|---|---|---|---|---|---|---|---|
| `alpha_earth_coop` | 64 | **0.1054** | 0.0735 | 0.1493 | 0.986 / 0.998 / 1.003 | 0.01% | 0.00% | 0.00% | **0.474** |
| `tesserav1.1_global` | 128 | **1.1408** | 0.7217 | 1.7900 | 7.979 / 15.126 / 36.132 | 0.89% | 0.00% | 0.00% | **0.044** |
| `seamless` | 72 | **0.5260** | 0.4192 | 0.7486 | 3.767 / 4.987 / 6.621 | 0.02% | 0.00% | 0.00% | **0.095** |
| `sentinel1` | 8 | **0.5909** | 0.1767 | 7.0227 | 0.052 / 0.341 / 3.366 | 1.03% | 0.00% | 0.00% | **0.085** |
| `sentinel2` | 10 | **0.0816** | 0.0408 | 0.0988 | 0.102 / 0.470 / 1.165 | 0.00% | 0.00% | 0.00% | **0.613** |

Tail / asymmetry summary (per-channel, pooled across channels):

| family | mean of per-channel means | median \|skew\| | max \|skew\| | min p0.1 | max p99.9 |
|---|---|---|---|---|---|
| `alpha_earth_coop` | -0.0061 | 0.28 | 0.82 | -0.4746 | 0.4453 |
| `tesserav1.1_global` | -0.1542 | 0.46 | 1.42 | -9.206 | 10.54 |
| `seamless` | +0.0300 | 0.39 | 1.69 | -1 | 1 |
| `sentinel1` | +0.0403 | 282.65 | 1138.44 | -3.313 | 18.55 |
| `sentinel2` | +0.1438 | 1.53 | 2.92 | 0.0006 | 0.6292 |

