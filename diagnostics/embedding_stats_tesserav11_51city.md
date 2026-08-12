### Task 0.1 — Embedding scale audit

Cultural-split **training** set, 5000 patches (paired across families, seed 0), measured after the exact training-path transform (nan_to_num -> dequantize -> bilinear resize to 32x32).

| family | C | per-channel std (median) | std min | std max | L2 norm p1 / p50 / p99 | frac zero (raw) | frac NaN (raw) | all-zero px | `0.05 / median_std` |
|---|---|---|---|---|---|---|---|---|---|
| `tesserav1.1` | 128 | **0.7975** | 0.4203 | 1.5295 | 6.490 / 12.553 / 22.586 | 0.92% | 0.00% | 0.00% | **0.063** |

Tail / asymmetry summary (per-channel, pooled across channels):

| family | mean of per-channel means | median \|skew\| | max \|skew\| | min p0.1 | max p99.9 |
|---|---|---|---|---|---|
| `tesserav1.1` | -0.0242 | 0.47 | 1.73 | -6.91 | 6.93 |

