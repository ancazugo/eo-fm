### Task 2.0 — The common patch manifest

Built from `diagnostics/invalid_fraction.parquet` (sha256 `5cd86a9c10c6de0b…`), no patch data read. Generated 2026-08-12T14:16:52+00:00.

Membership requires, for **all** of `alpha_earth_coop`, `tesserav1.1_global`, `seamless`: present on disk, `invalid_frac <= 0.25`, and `native_frac >= 0.5`.

**Manifest** `diagnostics/patch_manifest_v1.parquet` — sha256 `ad7fdae39ae75b77d0b9c477fdf051aaf12ce9894615b05c74b39399f37ca728`

| split | reference | manifest | retained |
|---|---|---|---|
| training | 352,366 | 341,754 | 96.99% |
| validation | 24,119 | 23,878 | 99.00% |
| testing | 24,188 | 23,852 | 98.61% |

#### 2.0a — Native crop area, and why the filter is relative

| family | expected native area | median shape | modal shape | area min | area p1 | area max |
|---|---|---|---|---|---|---|
| `alpha_earth_coop` | 1122 | 33x33 | 33x34 | 99 | 1089 | 1188 |
| `seamless` | 144 | 12x12 | 12x12 | 33 | 121 | 156 |
| `tesserav1.1_global` | 1122 | 33x33 | 33x34 | 12 | 1089 | 1224 |

`seamless` is 30 m data and crops natively to ~12 px per side against ~33 for the 10 m families, so an absolute pixel threshold would erase it entirely. The denominator is each family's own median native crop area on the `training` split.

| family | native_frac <0.25 | 0.25-0.5 | 0.5-0.75 | 0.75-0.95 | >=0.95 |
|---|---|---|---|---|---|
| `alpha_earth_coop` | 14 | 6 | 2 | 7 | 400,644 |
| `seamless` | 2 | 28 | 29 | 143,890 | 256,077 |
| `tesserav1.1_global` | 156 | 164 | 166 | 396 | 389,798 |

The `seamless` mass in the 0.75-0.95 bucket is not damage: an 11x12 crop is 0.92 of a 12x12 median, so ordinary +/-1 px reprojection drift moves a large share of ESD patches a full bucket that the same drift barely registers for a 33 px crop. Only the `<0.5` buckets are truncation.

**Two truncation rules, side by side.** Task 1.75.1 counted a crop truncated below 0.85 x the family median *per side*; the manifest thresholds 0.5 of the median *area*. They disagree — a 33x17 crop is 0.52 of the area but 0.5 of one side — so the area rule is the more permissive of the two and the gap is reported rather than left implicit.

| family | split | below native_frac | below per-side rule |
|---|---|---|---|
| `alpha_earth_coop` | training | 20 | 22 |
| `seamless` | training | 30 | 77 |
| `tesserav1.1_global` | training | 314 | 560 |
| `tesserav1.1_global` | testing | 6 | 17 |

#### 2.0b — What the manifest costs each family

`own_filtered` = the patches that family has on disk and that pass both criteria on its own — what Arm B would use. `loss` is what Arm A gives up relative to that.

| split | family | on disk | own filtered | manifest | loss vs own |
|---|---|---|---|---|---|
| training | `alpha_earth_coop` | 352,366 | 350,862 | 341,754 | 2.60% |
| training | `tesserav1.1_global` | 342,944 | 342,614 | 341,754 | 0.25% |
| training | `seamless` | 351,719 | 351,689 | 341,754 | 2.82% |
| validation | `alpha_earth_coop` | 24,119 | 24,011 | 23,878 | 0.55% |
| validation | `tesserav1.1_global` | 23,878 | 23,878 | 23,878 | 0.00% |
| validation | `seamless` | 24,119 | 24,119 | 23,878 | 1.00% |
| testing | `alpha_earth_coop` | 24,188 | 24,188 | 23,852 | 1.39% |
| testing | `tesserav1.1_global` | 23,858 | 23,852 | 23,852 | 0.00% |
| testing | `seamless` | 24,188 | 24,188 | 23,852 | 1.39% |

#### 2.0c — What the intersection is silently doing

**The intersection and the nodata policy are not independent filters.** Of `alpha_earth_coop`'s severely invalid patches (>75% of pixels), this many are *also* absent from each other family — against that family's base rate of absence over the whole split:

| split | severe patches | other family | also absent | base absence rate |
|---|---|---|---|---|
| training | 1,277 | `tesserav1.1_global` | 1,277 (100.0%) | 2.67% |
| training | 1,277 | `seamless` | 0 (0.0%) | 0.18% |
| validation | 105 | `tesserav1.1_global` | 105 (100.0%) | 1.00% |
| validation | 105 | `seamless` | 0 (0.0%) | 0.00% |
| testing | 0 | `tesserav1.1_global` | 0 (0.0%) | 1.36% |
| testing | 0 | `seamless` | 0 (0.0%) | 0.00% |

So the coverage intersection already performs most of the drop policy. How much independent work the nodata criterion is left with — the count of high-invalid patches that survive a **coverage-only** intersection:

| split | coop patches >max_invalid | survive coverage-only intersection |
|---|---|---|
| training | 1,484 | 178 |
| validation | 108 | 0 |
| testing | 0 | 0 |

##### training — composition against the full reference population

341,754 of 352,366 retained. Attribution of the exclusions:

| family | absent from disk | present but failed a criterion |
|---|---|---|
| `alpha_earth_coop` | 0 | 1,504 |
| `tesserav1.1_global` | 9,422 | 330 |
| `seamless` | 647 | 30 |

**Cities losing more than 5%** (6). **Bold** = one of the 10 held-out cultural-split cities.

| city | held out | n | retained | loss |
|---|---|---|---|---|
| Istanbul |  | 20,431 | 17,798 | 12.89% |
| Qingdao |  | 5,482 | 4,790 | 12.62% |
| New York |  | 20,371 | 18,345 | 9.95% |
| Cape Town |  | 11,252 | 10,243 | 8.97% |
| Lisbon |  | 1,258 | 1,187 | 5.64% |
| Shanghai |  | 7,304 | 6,925 | 5.19% |

**LCZ classes losing more than 5%** (1).

| LCZ | n | retained | loss |
|---|---|---|---|
| 17 | 49,359 | 40,041 | 18.88% |

Excluded patches by class: LCZ 17 9,318 (87.8%), LCZ 11 291 (2.7%), LCZ 14 208 (2.0%).

Watch cities (Rev A):

| city | n | retained | loss |
|---|---|---|---|
| New York | 20,371 | 18,345 | 9.95% |
| Cape Town | 11,252 | 10,243 | 8.97% |
| Lisbon | 1,258 | 1,187 | 5.64% |
| Shanghai | 7,304 | 6,925 | 5.19% |
| Melbourne | 35,093 | 34,153 | 2.68% |
| Guangzhou | 5,013 | 4,912 | 2.01% |
| London | 27,340 | 27,096 | 0.89% |

##### validation — composition against the full reference population

23,878 of 24,119 retained. Attribution of the exclusions:

| family | absent from disk | present but failed a criterion |
|---|---|---|
| `alpha_earth_coop` | 0 | 108 |
| `tesserav1.1_global` | 241 | 0 |
| `seamless` | 0 | 0 |

**Cities losing more than 5%** (1). **Bold** = one of the 10 held-out cultural-split cities.

| city | held out | n | retained | loss |
|---|---|---|---|---|
| **Mumbai** | yes | 2,413 | 2,172 | 9.99% |

**LCZ classes losing more than 5%** (1).

| LCZ | n | retained | loss |
|---|---|---|---|
| 17 | 2,609 | 2,368 | 9.24% |

Excluded patches by class: LCZ 17 241 (100.0%).

Watch cities (Rev A):

| city | n | retained | loss |
|---|---|---|---|
| Mumbai | 2,413 | 2,172 | 9.99% |
| Guangzhou | 2,407 | 2,407 | 0.00% |

##### testing — composition against the full reference population

23,852 of 24,188 retained. Attribution of the exclusions:

| family | absent from disk | present but failed a criterion |
|---|---|---|
| `alpha_earth_coop` | 0 | 0 |
| `tesserav1.1_global` | 330 | 6 |
| `seamless` | 0 | 0 |

**Cities losing more than 5%** (1). **Bold** = one of the 10 held-out cultural-split cities.

| city | held out | n | retained | loss |
|---|---|---|---|---|
| **Sydney** | yes | 2,419 | 2,084 | 13.85% |

**LCZ classes losing more than 5%** (1).

| LCZ | n | retained | loss |
|---|---|---|---|
| 17 | 2,540 | 2,204 | 13.23% |

Excluded patches by class: LCZ 17 336 (100.0%).

Watch cities (Rev A):

| city | n | retained | loss |
|---|---|---|---|
| Mumbai | 2,421 | 2,420 | 0.04% |
| Guangzhou | 2,402 | 2,402 | 0.00% |

##### Class-share shift (training)

| LCZ | before | after | delta |
|---|---|---|---|
| 17 | 14.01% | 11.72% | -2.29 pp |
| 15 | 0.68% | 0.70% | +0.02 pp |
| 12 | 2.70% | 2.73% | +0.03 pp |
| 7 | 0.93% | 0.96% | +0.03 pp |
| 4 | 2.46% | 2.50% | +0.04 pp |
| 16 | 2.24% | 2.28% | +0.04 pp |
| 1 | 1.44% | 1.48% | +0.04 pp |
| 13 | 2.60% | 2.68% | +0.08 pp |
| 10 | 3.39% | 3.49% | +0.10 pp |
| 9 | 3.86% | 3.97% | +0.12 pp |
| 5 | 4.68% | 4.82% | +0.14 pp |
| 2 | 6.93% | 7.13% | +0.20 pp |
| 6 | 10.02% | 10.27% | +0.26 pp |
| 3 | 8.99% | 9.27% | +0.27 pp |
| 11 | 12.18% | 12.47% | +0.29 pp |
| 14 | 11.74% | 12.05% | +0.30 pp |
| 8 | 11.16% | 11.48% | +0.32 pp |
