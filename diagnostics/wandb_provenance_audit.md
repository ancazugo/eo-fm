### Task 1.5.4 — W&B provenance audit

225 runs in `phd-thesis-team/lcz-classification-dl`, read 2026-08-11T23:07:09+00:00. Resolution: explicit 35, inferred 190.

| embedding | product | version | source | runs |
|---|---|---|---|---|
| `alpha_earth_coop` | alphaearth | coop | source_coop | 60 |
| `seamless` | esd | none | local_tif | 54 |
| `tessera` | tessera | v1 | gee_zarr | 42 |
| `tesserav1.1` | tessera | v1.1 | percity_geotessera | 37 |
| `tesserav1.1_global` | tessera | v1.1 | global_0.1deg | 30 |
| `tesserav1.1_global+alpha_earth_coop` | tessera+alphaearth | v1.1+coop | global_0.1deg+source_coop | 1 |
| `tesserav1.1_global+aux_struct` | tessera+aux | v1.1+none | global_0.1deg+local_tif | 1 |

#### Tessera runs by product (111 runs)

| source | runs | best test_kappa | best run |
|---|---|---|---|
| `gee_zarr` | 42 | 0.9316 | `zany-star-207` (x8o8bsn3) |
| `global_0.1deg` | 30 | 0.9682 | `percity-london-tv11global-medium` (hhbge57z) |
| `global_0.1deg+local_tif` | 1 | 0.6310 | `aux-fusion-v1` (yo26w6fu) |
| `global_0.1deg+source_coop` | 1 | 0.5974 | `fusion-tessera-coop` (n70tc6du) |
| `percity_geotessera` | 37 | 0.9742 | `percity-london-tv11-small` (7jpl4jpc) |

Top Tessera runs on the cultural split (`global_so2sat`) — the headline numbers:

| run | id | source | test_kappa | test_f1 | resolved by |
|---|---|---|---|---|---|
| `student-noisy-v3` | ey10pcob | `global_0.1deg` | 0.6497 | 0.5656 | explicit |
| `student-noisy-v1` | k4ka3wjs | `global_0.1deg` | 0.6419 | 0.5780 | explicit |
| `student-noisy-v2` | 2pz0icbd | `global_0.1deg` | 0.6320 | 0.5766 | explicit |
| `aux-fusion-v1` | yo26w6fu | `global_0.1deg+local_tif` | 0.6310 | 0.5682 | explicit |
| `opt3-tessera-seed1` | h6tv58sp | `global_0.1deg` | 0.6268 | 0.5586 | explicit |
| `opt3-lr5e-4-warmup3` | lhmxayfl | `global_0.1deg` | 0.6190 | 0.5652 | inferred |
| `opt3-tessera-seed2` | 9z9lj6ch | `global_0.1deg` | 0.6171 | 0.5634 | explicit |
| `opt2-reg-mixup0.2-wd3e-4` | cwqtk910 | `global_0.1deg` | 0.6091 | 0.5516 | inferred |
| `clean-universe-260` | ltm7j0zq | `global_0.1deg` | 0.6047 | 0.5357 | inferred |
| `opt1-resnet101-medium` | 0c3xiy4w | `global_0.1deg` | 0.6033 | 0.5384 | inferred |

Top Tessera runs on the per-city grid split — autocorrelation-inflated, not comparable:

| run | id | source | test_kappa | test_f1 | resolved by |
|---|---|---|---|---|---|
| `percity-london-tv11-small` | 7jpl4jpc | `percity_geotessera` | 0.9742 | 0.9266 | explicit |
| `percity-london-tv11global-medium` | hhbge57z | `global_0.1deg` | 0.9682 | 0.9163 | explicit |
| `percity-london-tv11global-small` | ykwmrz2k | `global_0.1deg` | 0.9639 | 0.9070 | explicit |
| `percity-london-tv11global-large` | dvvs2t21 | `global_0.1deg` | 0.9589 | 0.8848 | explicit |
| `zany-star-207` | x8o8bsn3 | `gee_zarr` | 0.9316 | 0.8290 | inferred |
| `treasured-water-162` | if1wbhby | `percity_geotessera` | 0.9316 | 0.8492 | inferred |
| `expert-microwave-208` | cpmel34m | `gee_zarr` | 0.9312 | 0.8107 | inferred |
| `copper-water-159` | 7xbccv1x | `gee_zarr` | 0.9311 | 0.8131 | inferred |
| `stilted-grass-184` | 6vjdhy5c | `gee_zarr` | 0.9296 | 0.7476 | inferred |
| `fast-firefly-161` | s5rvkvu5 | `percity_geotessera` | 0.9280 | 0.8408 | inferred |
