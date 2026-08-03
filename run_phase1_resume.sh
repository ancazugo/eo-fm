#!/usr/bin/env bash
# Resume of run_phase1_noisy_student.sh after the original extractor was
# killed at 21% (2026-07-03 ~21:00) and took the driver chain down with it.
# Extraction was parallelized into three south->north bands of the
# (cx + cy*1000) sort order:
#   slice0 rows 0-130k   -> re-run here (--skip-existing skips the ~59k done)
#   sliceA rows 130k-208k -> helper PID 1384448 (already running)
#   sliceB rows 208k-286k -> helper PID 1384447 (already running)
# Then waits for the helpers and continues with steps 3-4 of the original
# runner (pseudo-labeling + student training), unchanged.
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
SLICE_A_PID=${SLICE_A_PID:-1384448}
SLICE_B_PID=${SLICE_B_PID:-1384447}

echo "=== $(date -Is) [2/4 resume] extracting slice0 (rows 0-130k, skip-existing) ==="
# The extractor has been OOM-killed twice mid-run; --skip-existing makes each
# retry pure forward progress, so loop until it exits cleanly.
for attempt in $(seq 1 20); do
    if python src/extract_so2sat_embeddings.py \
        --so2sat-dir "$SO2SAT" \
        --patches-file data/patches_reference_unlabeled_slice0.gpkg \
        --embedding-dir /tessera/v1.1 \
        --embedding-name tesserav1.1_global \
        --output-name GeoTessera_v1.1_global \
        --year 2017 --splits unlabeled --dtype float16 --skip-existing \
        > "$LOGS/phase1-extract-slice0.log" 2>&1; then
        break
    fi
    echo "=== $(date -Is) slice0 extractor died (attempt $attempt), retrying in 60s ==="
    sleep 60
done

echo "=== $(date -Is) [2/4 resume] slice0 done; waiting for sliceA/B helpers ==="
while kill -0 "$SLICE_A_PID" 2>/dev/null || kill -0 "$SLICE_B_PID" 2>/dev/null; do
    sleep 60
done
echo "=== $(date -Is) helpers finished ==="

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
