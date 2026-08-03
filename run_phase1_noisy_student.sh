#!/usr/bin/env bash
# Phase 1: noisy-student semi-supervised training with Demuzere weak labels.
#
# Pipeline: (1) sample ~300k unlabeled patches from the 2017 Tessera tiles,
# weakly labeled by the Demuzere 2022 100m LCZ map; (2) extract float16
# Tessera npys (~80 GB under {so2sat-dir}/unlabeled/); (3) pseudo-label with
# the teacher (agree + rare-relax rules); (4) train the student with weighted
# pseudo samples appended to the labeled train split.
#
# Rough costs: sampling ~30-60 min (window reads over 8,108 tiles);
# extraction is the long step (~300k patches; hours, IO-bound — geographic
# sort keeps the tile LRU warm); pseudo-labeling ~30 min on the T4;
# student training same order as a Phase 0 run (several hours).
#
# NOT run automatically — launch manually when GPU and ~80 GB disk are free.
set -euo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
SO2SAT=$D/input/So2Sat-LCZ42/v4
DL=$D/output/lcz-classification/dl
LOGS=$DL/_experiment_logs
TEACHER=$DL/opt3-lr5e-4-warmup3/resnet_small_GeoTessera_v1.1_global_global-best.pt
UNLABELED_GPKG=data/patches_reference_unlabeled.gpkg
PSEUDO_DIR=data/pseudo_labels
mkdir -p "$LOGS"

echo "=== $(date -Is) [1/4] sampling unlabeled patches ==="
python src/sample_unlabeled_patches.py \
    --output "$UNLABELED_GPKG" \
    --exclude-gpkg "$SO2SAT/patches_reference_rxr.gpkg" \
    --n-patches 300000 \
    > "$LOGS/phase1-sample.log" 2>&1

echo "=== $(date -Is) [2/4] extracting float16 Tessera patches ==="
python src/extract_so2sat_embeddings.py \
    --so2sat-dir "$SO2SAT" \
    --patches-file "$UNLABELED_GPKG" \
    --embedding-dir /tessera/v1.1 \
    --embedding-name tesserav1.1_global \
    --output-name GeoTessera_v1.1_global \
    --year 2017 --splits unlabeled --dtype float16 --skip-existing \
    > "$LOGS/phase1-extract.log" 2>&1

echo "=== $(date -Is) [3/4] generating pseudo-labels (teacher: opt3) ==="
python src/generate_pseudo_labels.py \
    --checkpoint "$TEACHER" \
    --family resnet --preset small \
    --unlabeled-gpkg "$UNLABELED_GPKG" \
    --so2sat-dir "$SO2SAT" \
    --output-name GeoTessera_v1.1_global --year 2017 \
    --embedding-name tesserav1.1_global --tta \
    --output-dir "$PSEUDO_DIR" \
    > "$LOGS/phase1-pseudo.log" 2>&1

echo "=== $(date -Is) [4/4] training student (opt3 recipe + pseudo) ==="
python src/patch_classification.py \
    --so2sat-dir "$SO2SAT" --global-split --year 2017 \
    --output-name GeoTessera_v1.1_global \
    --embedding-name tesserav1.1_global \
    --embedding-dir /tessera/v1.1 \
    --family resnet --preset small --patch-size 32 \
    --batch-size 256 --num-workers 8 \
    --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4 \
    --class-weights sqrt_inv_freq --label-smoothing 0.1 \
    --monitor val_kappa --tta \
    --max-epochs 50 --early-stopping-patience 10 \
    --pseudo-gpkg "$PSEUDO_DIR/patches_reference_pseudo.gpkg" \
    --run-name student-noisy-v1 \
    --output-dir "$DL" \
    > "$LOGS/student-noisy-v1.log" 2>&1

echo "=== $(date -Is) chain complete ==="

# ── Iteration 2 (uncomment after v1: student becomes teacher) ────────────────
# Re-generate pseudo-labels with the v1 student (usually with a slightly higher
# --min-conf) and retrain, optionally with more capacity (--preset base).
#
# python src/generate_pseudo_labels.py \
#     --checkpoint "$DL/student-noisy-v1/resnet_small_GeoTessera_v1.1_global_global-best.pt" \
#     --family resnet --preset small \
#     --unlabeled-gpkg "$UNLABELED_GPKG" \
#     --so2sat-dir "$SO2SAT" \
#     --output-name GeoTessera_v1.1_global --year 2017 \
#     --embedding-name tesserav1.1_global --tta --min-conf 0.8 \
#     --output-dir data/pseudo_labels_v2
# python src/patch_classification.py ... --preset base \
#     --pseudo-gpkg data/pseudo_labels_v2/patches_reference_pseudo.gpkg \
#     --run-name student-noisy-v2 ...
