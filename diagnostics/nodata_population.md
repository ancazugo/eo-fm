### Task 1.75.1 — Nodata population on the full cultural split

Every patch of every split, no pairing and no sampling — 1,191,379 patch reads across 3 families. Generated 2026-08-12T02:06:06+00:00.

#### Headline — coverage and invalid data per family and split

| family | split | on disk | missing | invalid px | patches w/ any | p50 | p90 | p99 | max | truncated |
|---|---|---|---|---|---|---|---|---|---|---|
| `alpha_earth_coop` | training | 352,366 | 0 (0.00%) | 0.4006% | 0.46% | 0.0000% | 0.0000% | 0.0000% | 100.00% | 22 |
| `alpha_earth_coop` | validation | 24,119 | 0 (0.00%) | 0.4433% | 0.46% | 0.0000% | 0.0000% | 0.0000% | 100.00% | 0 |
| `alpha_earth_coop` | testing | 24,188 | 0 (0.00%) | 0.0000% | 0.00% | 0.0000% | 0.0000% | 0.0000% | 0.00% | 0 |
| `tesserav1.1_global` | training | 342,944 | 9,422 (2.67%) | 0.1399% | 4.38% | 0.0000% | 0.0000% | 3.0303% | 77.82% | 560 |
| `tesserav1.1_global` | validation | 23,878 | 241 (1.00%) | 0.1437% | 4.16% | 0.0000% | 0.0000% | 3.1142% | 8.22% | 0 |
| `tesserav1.1_global` | testing | 23,858 | 330 (1.36%) | 0.1362% | 3.86% | 0.0000% | 0.0000% | 3.6332% | 5.80% | 17 |
| `seamless` | training | 351,719 | 647 (0.18%) | 0.0000% | 0.00% | 0.0000% | 0.0000% | 0.0000% | 0.00% | 77 |
| `seamless` | validation | 24,119 | 0 (0.00%) | 0.0000% | 0.00% | 0.0000% | 0.0000% | 0.0000% | 0.00% | 0 |
| `seamless` | testing | 24,188 | 0 (0.00%) | 0.0000% | 0.00% | 0.0000% | 0.0000% | 0.0000% | 0.00% | 0 |

`missing` counts patches present in `patches_reference_rxr.gpkg` with no npy on disk. `truncated` counts native crops below 0.85 x the family's own median native size — extracted before `exact_footprint_4326`, and stretched to the model's patch size rather than dropped. The rule is relative because the 10 m families crop near 33 px and 30 m `seamless` near 12.

#### Counts above each candidate drop threshold (training split)

| family | >5% | >10% | >25% | >50% |
|---|---|---|---|---|
| `alpha_earth_coop` | 1,575 | 1,547 | 1,484 | 1,378 |
| `tesserav1.1_global` | 1,542 | 29 | 16 | 5 |
| `seamless` | 0 | 0 | 0 | 0 |

#### Per-patch invalid-fraction histogram (training split)

| bucket | `alpha_earth_coop` | `tesserav1.1_global` | `seamless` |
|---|---|---|---|
| exactly 0 | 350,749 | 327,906 | 351,719 |
| 0-0.1% | 3 | 24 | 0 |
| 0.1-1% | 12 | 591 | 0 |
| 1-5% | 27 | 12,881 | 0 |
| 5-10% | 28 | 1,513 | 0 |
| 10-25% | 63 | 13 | 0 |
| 25-50% | 106 | 11 | 0 |
| 50-75% | 101 | 4 | 0 |
| 75-100% | 1,277 | 1 | 0 |

#### Coverage bias by city

Cities sorted by AlphaEarth mean invalid fraction. **Bold** = one of the 10 held-out cultural-split cities (validation and testing draw from the same 10, so a hole there lands on both).

| city | held out | coop invalid | coop >25% | tessera missing | tessera truncated | seamless missing | n patches |
|---|---|---|---|---|---|---|---|
| Cape Town |  | 8.1781% | 896 | 983 | 32 | 0 | 11,252 |
| Lisbon |  | 3.5652% | 45 | 71 | 0 | 0 | 1,258 |
| **Mumbai** | yes | 2.2049% | 108 | 242 | 1 | 0 | 4,834 |
| New York |  | 1.5748% | 332 | 2,026 | 0 | 0 | 20,371 |
| Amsterdam |  | 0.2684% | 10 | 107 | 3 | 0 | 3,451 |
| Hong Kong |  | 0.2037% | 15 | 7 | 22 | 0 | 4,298 |
| **Guangzhou** | yes | 0.1860% | 39 | 26 | 99 | 0 | 9,822 |
| Qingdao |  | 0.1429% | 14 | 542 | 17 | 111 | 5,482 |
| London |  | 0.1158% | 60 | 111 | 120 | 0 | 27,340 |
| Wuhan |  | 0.0362% | 7 | 6 | 3 | 106 | 8,261 |
| Melbourne |  | 0.0084% | 4 | 914 | 53 | 0 | 35,093 |
| Beijing |  | 0.0000% | 0 | 0 | 0 | 0 | 5,392 |
| Berlin |  | 0.0000% | 0 | 0 | 0 | 0 | 19,130 |
| Bogota |  | 0.0000% | 0 | 0 | 0 | 0 | 8 |
| Buenos Aires |  | 0.0000% | 0 | 0 | 0 | 0 | 5 |
| Cairo |  | 0.0000% | 0 | 0 | 0 | 0 | 17,445 |
| Caracas |  | 0.0000% | 0 | 0 | 0 | 0 | 12 |
| Changsha |  | 0.0000% | 0 | 0 | 0 | 0 | 4,667 |
| Chicago |  | 0.0000% | 0 | 0 | 0 | 0 | 48 |
| Cologne |  | 0.0000% | 0 | 0 | 0 | 0 | 15,265 |
| Dhaka |  | 0.0000% | 0 | 0 | 0 | 0 | 30 |
| Istanbul |  | 0.0000% | 0 | 2,617 | 29 | 0 | 20,431 |
| **Jakarta** | yes | 0.0000% | 0 | 0 | 0 | 0 | 4,831 |
| Karachi |  | 0.0000% | 0 | 0 | 0 | 0 | 1,140 |
| Lima |  | 0.0000% | 0 | 0 | 0 | 0 | 48 |
| Los Angeles |  | 0.0000% | 0 | 0 | 0 | 0 | 11,905 |
| Madrid |  | 0.0000% | 0 | 0 | 0 | 162 | 6,785 |
| Milan |  | 0.0000% | 0 | 0 | 0 | 0 | 2,963 |
| **Moscow** | yes | 0.0000% | 0 | 0 | 0 | 0 | 4,831 |
| **Munich** | yes | 0.0000% | 0 | 0 | 0 | 0 | 4,832 |
| **Nairobi** | yes | 0.0000% | 0 | 0 | 0 | 0 | 4,828 |
| Nanjing |  | 0.0000% | 0 | 0 | 0 | 0 | 6,356 |
| Osaka [Kyoto] |  | 0.0000% | 0 | 0 | 0 | 0 | 6,443 |
| Paris |  | 0.0000% | 0 | 0 | 0 | 92 | 12,511 |
| Philadelphia |  | 0.0000% | 0 | 0 | 0 | 0 | 2 |
| Quezon City [Manila] |  | 0.0000% | 0 | 0 | 0 | 0 | 384 |
| Rawalpindi [Islamabad] |  | 0.0000% | 0 | 0 | 0 | 0 | 8,108 |
| Rio De Janeiro |  | 0.0000% | 0 | 0 | 0 | 0 | 13,596 |
| Rome |  | 0.0000% | 0 | 0 | 0 | 0 | 5,347 |
| Salvador |  | 0.0000% | 0 | 0 | 0 | 0 | 1 |
| **San Jose** | yes | 0.0000% | 0 | 0 | 0 | 0 | 4,812 |
| **Santiago** | yes | 0.0000% | 0 | 0 | 0 | 0 | 4,798 |
| Shanghai |  | 0.0000% | 0 | 358 | 50 | 0 | 7,304 |
| **Sydney** | yes | 0.0000% | 0 | 329 | 16 | 0 | 4,830 |
| São Paulo |  | 0.0000% | 0 | 0 | 0 | 0 | 19,570 |
| **Tehran** | yes | 0.0000% | 0 | 0 | 0 | 0 | 4,827 |
| Tokyo |  | 0.0000% | 0 | 0 | 7 | 0 | 884 |
| Vancouver |  | 0.0000% | 0 | 1,094 | 36 | 0 | 30,657 |
| Washington D.C. |  | 0.0000% | 0 | 0 | 0 | 0 | 4,206 |
| Zurich |  | 0.0000% | 0 | 0 | 0 | 0 | 2,515 |
| 东营区 |  | 0.0000% | 0 | 70 | 13 | 0 | 3,240 |

#### Coverage bias by LCZ class (training split, AlphaEarth)

| LCZ | invalid frac | patches | | LCZ | invalid frac | patches |
|---|---|---|---|---|---|---|
| 1 | 0.0000% | 5,068 | | 10 | 0.0000% | 11,954 |
| 2 | 0.0053% | 24,431 | | 11 | 0.0308% | 42,902 |
| 3 | 0.0000% | 31,693 | | 12 | 0.1866% | 9,514 |
| 4 | 0.0315% | 8,651 | | 13 | 0.0000% | 9,165 |
| 5 | 0.0000% | 16,493 | | 14 | 0.0794% | 41,377 |
| 6 | 0.0445% | 35,290 | | 15 | 0.0000% | 2,392 |
| 7 | 0.0000% | 3,269 | | 16 | 0.0000% | 7,898 |
| 8 | 0.0159% | 39,326 | | 17 | 2.6772% | 49,359 |
| 9 | 0.0000% | 13,584 | | | | |

#### Corrected channel statistics — full unfiltered training population

Supersedes the Phase 0 (paired, n=5000, unmasked) and GATE 1 (n=20000, masked) tables. `masked` excludes nodata pixels; post-resize masked statistics are independent of the fill value by construction, so no fill had to be assumed.

| family | native unmasked | native masked | resized unmasked | resized masked | sentinel share of variance | `0.05 / median_std` |
|---|---|---|---|---|---|---|
| `alpha_earth_coop` | 0.1221 | 0.1050 | 0.1214 | **0.1048** | 26.07% | **0.477** |
| `tesserav1.1_global` | 1.1592 | 1.1597 | 1.1339 | **1.1344** | 0.00% | **0.044** |
| `seamless` | 0.5610 | 0.5610 | 0.5446 | **0.5446** | 0.00% | **0.092** |
