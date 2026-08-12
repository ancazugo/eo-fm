### Task 0.2 — AlphaEarth coop dequantization verification

5000 training patches. AlphaEarth embeddings are unit-norm 64-d vectors, so the correct decoding is the one whose per-pixel L2 norm concentrates at 1.0.

| candidate decoding | L2 norm mean | std | p1 | p50 | p99 | within 1% of 1.0 |
|---|---|---|---|---|---|---|
| `((v/127.5)**2)*sign(v)` — **current** | 1.0292 | 0.4524 | 0.9955 | 1.0001 | 1.0051 | 99.59% |
| `v/127.5` | 2.5556 | 0.3543 | 2.4419 | 2.5339 | 2.6230 | 0.00% |
| `sign(v)*(|v|/127.5)` + L2 renorm | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% |

(Non-all-zero pixels; all-zero pixels are 0.00% of the sample and decode to norm 0 under every candidate.)

Pearson r between decodings over all values: `current_sq_sign` vs `linear` = 0.9305, `current_sq_sign` vs `linear_renorm` = 0.8836, `linear` vs `linear_renorm` = 0.9905

GEE float32 cross-check (200 patches):

| candidate | per-channel r (median) | r (min) | RMSE | pixelwise r |
|---|---|---|---|---|
| `((v/127.5)**2)*sign(v)` — **current** | 0.9990 | 0.9959 | 0.0042 | n/a |
| `v/127.5` | 0.9716 | 0.9319 | 0.1829 | n/a |
| `sign(v)*(|v|/127.5)` + L2 renorm | 0.9719 | 0.9341 | 0.0266 | n/a |

