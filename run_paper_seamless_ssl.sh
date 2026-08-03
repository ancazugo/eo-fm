#!/usr/bin/env bash
# Seamless noisy-student round — closes the SSL asymmetry in the paper's
# coverage matrix (tessera 3 iters / coop 1 / seamless 0). Mirrors
# run_phase1_coop_student.sh: seamless is its own teacher (good-flower-268)
# to keep ensemble members decorrelated.
# (1) extract float16 seamless npys for the 286k unlabeled pool (CPU/NFS;
#     retried on OOM-kill; coverage will be whatever the MGRS tiles span);
# (2+3) WAIT until the GPU is free of the probe/seed and per-city chains,
#     then pseudo-label (min-conf 0.7; seamless T=0.98, well calibrated)
#     and train student-seamless-v1 (opt3 recipe).
set -uo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
SO2SAT=$D/input/So2Sat-LCZ42/v4
DL=$D/output/lcz-classification/dl
LOGS=$DL/_experiment_logs
SEAMLESS_DIR=$D/input/EmbeddedSeamlessData/2017
TEACHER=$DL/good-flower-268/resnet_small_EmbeddedSeamless_global-best.pt
UNLABELED_GPKG=data/patches_reference_unlabeled.gpkg
PSEUDO_DIR=data/pseudo_labels_seamless
mkdir -p "$LOGS"

echo "=== $(date -Is) [1/3] extracting float16 seamless patches ==="
for attempt in $(seq 1 20); do
    if python src/extract_so2sat_embeddings.py \
        --so2sat-dir "$SO2SAT" \
        --patches-file "$UNLABELED_GPKG" \
        --embedding-dir "$SEAMLESS_DIR" \
        --embedding-name seamless \
        --output-name EmbeddedSeamless \
        --year 2017 --splits unlabeled --dtype float16 --skip-existing \
        > "$LOGS/paper-extract-seamless-unlabeled.log" 2>&1; then
        break
    fi
    echo "=== $(date -Is) seamless extractor died (attempt $attempt), retrying in 60s ==="
    sleep 60
done

echo "=== $(date -Is) [gate] waiting for GPU chains to finish ==="
while pgrep -f "run_paper_probes_seeds.sh|run_paper_percity.sh" > /dev/null; do
    sleep 600
done

echo "=== $(date -Is) [2/3] pseudo-labels (teacher: good-flower-268 seamless) ==="
python src/generate_pseudo_labels.py \
    --checkpoint "$TEACHER" \
    --family resnet --preset small \
    --unlabeled-gpkg "$UNLABELED_GPKG" \
    --so2sat-dir "$SO2SAT" \
    --output-name EmbeddedSeamless --year 2017 \
    --embedding-name seamless --tta --min-conf 0.7 \
    --output-dir "$PSEUDO_DIR" \
    > "$LOGS/paper-pseudo-seamless.log" 2>&1

echo "=== $(date -Is) [3/3] training student-seamless-v1 (opt3 recipe + pseudo) ==="
python src/patch_classification.py \
    --so2sat-dir "$SO2SAT" --global-split --year 2017 \
    --output-name EmbeddedSeamless \
    --embedding-name seamless \
    --embedding-dir "$SEAMLESS_DIR" \
    --family resnet --preset small --patch-size 32 \
    --batch-size 256 --num-workers 8 \
    --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4 \
    --class-weights sqrt_inv_freq --label-smoothing 0.1 \
    --monitor val_kappa --tta \
    --max-epochs 50 --early-stopping-patience 10 \
    --pseudo-gpkg "$PSEUDO_DIR/patches_reference_pseudo.gpkg" \
    --run-name student-seamless-v1 \
    --output-dir "$DL" \
    > "$LOGS/student-seamless-v1.log" 2>&1

echo "=== $(date -Is) seamless student chain complete ==="
