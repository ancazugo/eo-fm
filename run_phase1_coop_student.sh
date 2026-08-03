#!/usr/bin/env bash
# Noisy-student round for the AlphaEarth-coop model, to re-diversify the
# ensemble (v3 absorbed seamless's signal; coop is the complementary member).
# Coop is its own teacher — using the tessera teacher's labels would correlate
# the two models and defeat the purpose. ~143k of the 286k unlabeled patches
# have coop coverage; the extractor and pseudo-labeler skip the rest.
# (1) extract float16 coop npys (~21 GB, retried on OOM-kill);
# (2) pseudo-label with wandering-firefly-267 (agree+rare-relax, min-conf 0.7);
# (3) train student-coop-v1 (opt3 recipe).
set -euo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
SO2SAT=$D/input/So2Sat-LCZ42/v4
DL=$D/output/lcz-classification/dl
LOGS=$DL/_experiment_logs
COOP_DIR=$D/input/Google/AlphaEarth/coop
TEACHER=$DL/wandering-firefly-267/resnet_small_AlphaEarthCoop_global-best.pt
UNLABELED_GPKG=data/patches_reference_unlabeled.gpkg
PSEUDO_DIR=data/pseudo_labels_coop

echo "=== $(date -Is) [1/3] extracting float16 coop patches ==="
for attempt in $(seq 1 20); do
    if python src/extract_so2sat_embeddings.py \
        --so2sat-dir "$SO2SAT" \
        --patches-file "$UNLABELED_GPKG" \
        --embedding-dir "$COOP_DIR" \
        --embedding-name alpha_earth_coop \
        --output-name AlphaEarthCoop \
        --year 2017 --splits unlabeled --dtype float16 --skip-existing \
        > "$LOGS/phase1-extract-coop.log" 2>&1; then
        break
    fi
    echo "=== $(date -Is) coop extractor died (attempt $attempt), retrying in 60s ==="
    sleep 60
done

echo "=== $(date -Is) [2/3] pseudo-labels (teacher: wandering-firefly-267 coop) ==="
python src/generate_pseudo_labels.py \
    --checkpoint "$TEACHER" \
    --family resnet --preset small \
    --unlabeled-gpkg "$UNLABELED_GPKG" \
    --so2sat-dir "$SO2SAT" \
    --output-name AlphaEarthCoop --year 2017 \
    --embedding-name alpha_earth_coop --tta --min-conf 0.7 \
    --output-dir "$PSEUDO_DIR" \
    > "$LOGS/phase1-pseudo-coop.log" 2>&1

echo "=== $(date -Is) [3/3] training student-coop-v1 (opt3 recipe + pseudo) ==="
python src/patch_classification.py \
    --so2sat-dir "$SO2SAT" --global-split --year 2017 \
    --output-name AlphaEarthCoop \
    --embedding-name alpha_earth_coop \
    --embedding-dir "$COOP_DIR" \
    --family resnet --preset small --patch-size 32 \
    --batch-size 256 --num-workers 8 \
    --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4 \
    --class-weights sqrt_inv_freq --label-smoothing 0.1 \
    --monitor val_kappa --tta \
    --max-epochs 50 --early-stopping-patience 10 \
    --pseudo-gpkg "$PSEUDO_DIR/patches_reference_pseudo.gpkg" \
    --run-name student-coop-v1 \
    --output-dir "$DL" \
    > "$LOGS/student-coop-v1.log" 2>&1

echo "=== $(date -Is) coop student chain complete ==="
