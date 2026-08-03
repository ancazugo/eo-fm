#!/usr/bin/env bash
# Phase 1 iteration 3: student-noisy-v1 becomes the teacher.
# Re-pseudo-labels the extracted 286k unlabeled pool at a stricter
# --min-conf 0.7, then trains student-noisy-v3 with the same opt3 recipe.
# Extraction is already on disk — this is pseudo-labeling (~20 min on the T4)
# + one training run (~5-7 h).
set -euo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
SO2SAT=$D/input/So2Sat-LCZ42/v4
DL=$D/output/lcz-classification/dl
LOGS=$DL/_experiment_logs
TEACHER=$DL/student-noisy-v1/resnet_small_GeoTessera_v1.1_global_global-best.pt
UNLABELED_GPKG=data/patches_reference_unlabeled.gpkg
PSEUDO_DIR=data/pseudo_labels_v3

echo "=== $(date -Is) [1/2] pseudo-labels v3 (teacher: student-noisy-v1, min-conf 0.7) ==="
python src/generate_pseudo_labels.py \
    --checkpoint "$TEACHER" \
    --family resnet --preset small \
    --unlabeled-gpkg "$UNLABELED_GPKG" \
    --so2sat-dir "$SO2SAT" \
    --output-name GeoTessera_v1.1_global --year 2017 \
    --embedding-name tesserav1.1_global --tta --min-conf 0.7 \
    --output-dir "$PSEUDO_DIR" \
    > "$LOGS/phase1-pseudo-v3.log" 2>&1

echo "=== $(date -Is) [2/2] training student-noisy-v3 (opt3 recipe + pseudo v2) ==="
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
    --run-name student-noisy-v3 \
    --output-dir "$DL" \
    > "$LOGS/student-noisy-v3.log" 2>&1

echo "=== $(date -Is) iteration 3 complete ==="
