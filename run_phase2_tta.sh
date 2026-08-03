#!/usr/bin/env bash
# Phase 2: per-city test-time adaptation (AdaBN, then TENT) of the 3 final
# ensemble members, honest protocol (adapt on val inputs, eval on test).
# Each run writes ensemble_eval-format npz pairs, then ensemble_stacking.py
# computes calibrated + LOCO-weighted ensemble numbers on them.
# Baselines (dl/ensemble_coopv1): solo tessera 0.6497; LOCO-weighted 0.6871.
set -euo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
SO2SAT=$D/input/So2Sat-LCZ42/v4
DL=$D/output/lcz-classification/dl
LOGS=$DL/_experiment_logs

MODELS=(
  --model "GeoTessera_v1.1_global,tesserav1.1_global,$DL/student-noisy-v3/resnet_small_GeoTessera_v1.1_global_global-best.pt"
  --model "AlphaEarthCoop,alpha_earth_coop,$DL/student-coop-v1/resnet_small_AlphaEarthCoop_global-best.pt"
  --model "EmbeddedSeamless,seamless,$DL/good-flower-268/resnet_small_EmbeddedSeamless_global-best.pt"
)

for METHOD in adabn tent; do
    OUT=$DL/tta_${METHOD}_val
    echo "=== $(date -Is) [$METHOD] per-city adaptation ==="
    python src/tta_city_adapt.py \
        --so2sat-dir "$SO2SAT" --year 2017 \
        "${MODELS[@]}" \
        --method $METHOD --adapt-split val --tta \
        --output-dir "$OUT" \
        > "$LOGS/phase2-tta-$METHOD.log" 2>&1
    echo "=== $(date -Is) [$METHOD] stacking analysis ==="
    python src/ensemble_stacking.py \
        --val-npz  "$OUT/ensemble_3models_val/probs.npz" \
        --test-npz "$OUT/ensemble_3models_test/probs.npz" \
        --temperature-scale \
        --city-holdout --global-gpkg "$SO2SAT/patches_reference_rxr.gpkg" \
        --output-json "$OUT/stacking_results.json" \
        > "$LOGS/phase2-tta-$METHOD-stacking.log" 2>&1
done
echo "=== $(date -Is) phase 2 TTA chain complete ==="
