### Schedule audit — `p2-anchor-tessera*`

7 finished runs. Generated 2026-09-04T08:10:53+00:00.

Peak epoch is the epoch of best `val_kappa`, which is the checkpoint that
gets evaluated. A peak inside the warmup window means the best model was
selected before the learning rate finished ramping.

#### Per run

| run | lr | warmup | seed | peak epoch | stop epoch | best val_kappa | macro-F1 | OA | kappa |
|---|---|---|---|---|---|---|---|---|---|
| `p2-anchor-tessera-seed0` | 0.0005 | 3 | 0 | **2** | 12 | 0.5667 | 0.5641 | 0.6409 | 0.6097 |
| `p2-anchor-tessera-seed1` | 0.0005 | 3 | 1 | **2** | 12 | 0.5857 | 0.5460 | 0.6497 | 0.6175 |
| `p2-anchor-tessera-seed2` | 0.0005 | 3 | 2 | 15 | 25 | 0.5892 | 0.5651 | 0.6438 | 0.6111 |
| `p2-anchor-tessera-seed3` | 0.0005 | 3 | 3 | 5 | 15 | 0.5947 | 0.5672 | 0.6602 | 0.6286 |
| `p2-anchor-tessera-seed4` | 0.0005 | 3 | 4 | 16 | 26 | 0.5754 | 0.5527 | 0.6529 | 0.6218 |
| `p2-anchor-tessera-seed5` | 0.0005 | 3 | 5 | 22 | 32 | 0.6027 | 0.5741 | 0.6597 | 0.6283 |
| `p2-anchor-tessera-seed6` | 0.0005 | 3 | 6 | 19 | 29 | 0.5914 | 0.5483 | 0.6452 | 0.6130 |

Bold peak epoch = the best checkpoint was chosen during warmup.

#### Grouped

| lr | warmup_epochs | n | peak epochs | in warmup | macro-F1 | OA | kappa |
|---|---|---|---|---|---|---|---|
| 0.0005 | 3 | 7 | 2, 2, 5, 15, 16, 19, 22 | 2/7 | 0.5597 ± 0.0106 | 0.6503 ± 0.0077 | 0.6186 ± 0.0079 |

#### Peak-epoch distribution

n = 7, peak epochs [2, 2, 5, 15, 16, 19, 22]

Widest gap between consecutive peaks: **10 epochs** over a span of 20 (50% of the span), splitting [2, 2, 5] from [15, 16, 19, 22].

This is a description, not a test — n is far too small for one.
