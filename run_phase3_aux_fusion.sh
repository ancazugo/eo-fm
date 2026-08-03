#!/usr/bin/env bash
# Phase 3 Stage 2a: tessera + auxiliary structural bands as input fusion.
# Chain: [1] download GHSL/canopy GEE tiles covering ALL So2Sat splits (~227
# tiles, idempotent) → [2] extract 4-band AuxStruct npys for all 400k patches
# (float16, resumable) → [3] sanity count → [4] train tessera(128)+aux(4)
# fusion with the exact opt3 recipe (baseline: tessera-only opt3 = 0.619).
# No pseudo-labels in this run — isolates the aux effect (see campaign doc §8b).
set -euo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
SO2SAT=$D/input/So2Sat-LCZ42/v4
AUX=$D/input/aux_struct
DL=$D/output/lcz-classification/dl
LOGS=$DL/_experiment_logs
mkdir -p "$LOGS"

echo "=== $(date -Is) [1/4] download aux tiles (all splits, cached tiles skipped) ==="
python src/extract_aux_features.py \
    --global-gpkg "$SO2SAT/patches_reference_rxr.gpkg" \
    --splits training validation testing \
    --aux-dir "$AUX" \
    --download-only \
    > "$LOGS/phase3-aux-download.log" 2>&1

echo "=== $(date -Is) [1b/4] precompute merged 4-band aux tiles (once) ==="
python src/precompute_aux_tiles.py \
    --aux-dir "$AUX" --workers 8 --skip-existing \
    > "$LOGS/phase3-aux-precompute.log" 2>&1

echo "=== $(date -Is) [2/4] extract AuxStruct npys (all splits, resumable) ==="
python src/extract_so2sat_embeddings.py \
    --so2sat-dir "$SO2SAT" \
    --embedding-dir "$AUX/merged_aux" \
    --embedding-name aux_struct \
    --output-name AuxStruct --year 2017 \
    --dtype float16 --workers 4 --skip-existing \
    > "$LOGS/phase3-aux-extract.log" 2>&1

echo "=== $(date -Is) [3/4] npy counts ==="
for s in training validation testing; do
    echo "  $s: $(ls "$SO2SAT/$s/AuxStruct/2017" | wc -l) npys"
done

echo "=== $(date -Is) [4/4] training aux-fusion-v1 (opt3 recipe, tessera+aux) ==="
python src/patch_classification.py \
    --so2sat-dir "$SO2SAT" --global-split --year 2017 \
    --output-name GeoTessera_v1.1_global AuxStruct \
    --embedding-name tesserav1.1_global aux_struct \
    --embedding-dir /tessera/v1.1 \
    --family resnet --preset small --patch-size 32 \
    --batch-size 256 --num-workers 8 \
    --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4 \
    --class-weights sqrt_inv_freq --label-smoothing 0.1 \
    --monitor val_kappa --tta \
    --max-epochs 50 --early-stopping-patience 10 \
    --run-name aux-fusion-v1 \
    --output-dir "$DL" \
    > "$LOGS/aux-fusion-v1.log" 2>&1

echo "=== $(date -Is) phase 3 stage 2a complete ==="
