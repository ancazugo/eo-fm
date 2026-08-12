### Task 1.5.1 — Tessera product identity

2000 patch_ids sampled from the intersection of each pair (seed 0), cultural-split **training** set, compared on the **native** grid with no resize.

| pair | ids in common | compared | shape mismatches |
|---|---|---|---|
| `tesserav1.1` vs `tesserav1.1_global` | 27,229 | 2,000 | 0 |
| `tesserav1.1_global` vs `tesserav2` | 131,086 | 1,956 | 44 |
| `tesserav1.1` vs `tesserav2` | 27,229 | 1,993 | 7 |

| pair | matched-channel r (med) | best cross-channel r (med) | std ratio (med / CV) | linear-map R² (held out / null) | L2-norm r | verdict |
|---|---|---|---|---|---|---|
| `tesserav1.1` vs `tesserav1.1_global` | -0.0049 | 0.6503 | 1.259 / 0.397 | 0.8924 / -0.0163 | 0.8315 | **distinct_products_shared_information** |
| `tesserav1.1_global` vs `tesserav2` | -0.0064 | 0.4824 | 0.343 / 0.648 | 0.8872 / -0.0271 | 0.0759 | **distinct_products_shared_information** |
| `tesserav1.1` vs `tesserav2` | -0.0164 | 0.6615 | 0.469 / 0.749 | 0.9081 / -0.0148 | 0.1224 | **distinct_products_shared_information** |

- **tesserav1.1 vs tesserav1.1_global** — Matched channels are uncorrelated but a linear map recovers most of one from the other: same scene, different feature basis, i.e. different inference passes sharing a version label. Treat as distinct products, keep both registry keys, pick one canonical.
- **tesserav1.1_global vs tesserav2** — Matched channels are uncorrelated but a linear map recovers most of one from the other: same scene, different feature basis, i.e. different inference passes sharing a version label. Treat as distinct products, keep both registry keys, pick one canonical.
- **tesserav1.1 vs tesserav2** — Matched channels are uncorrelated but a linear map recovers most of one from the other: same scene, different feature basis, i.e. different inference passes sharing a version label. Treat as distinct products, keep both registry keys, pick one canonical.
