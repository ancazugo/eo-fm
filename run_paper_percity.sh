#!/usr/bin/env bash
# Fill the London per-city grid patch-classification table (paper coverage matrix):
#   - tesserav1.1_global: small + medium + large (no per-city run existed at all)
#   - alpha_earth_coop:   small (had medium/large only)
#   - tesserav1.1:        small (had medium/large only)
# Recipe mirrors the existing London rows (hokey-tie-fighter-66 / tusken-rancor-61):
# batch 64, lr 1e-3, wd 1e-4, 50 epochs, patience 10, patch 32 — NO mixup/TTA/
# warmup/class-weights (pre-opt3 defaults), so rows stay comparable.
# Tiny dataset (~679 patches) → runs share the T4 with the seed chain fine.
set -uo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

COMMON=(
  --so2sat-dir "$D/input/So2Sat-LCZ42/v4"
  --cities-dir "$D/input/So2Sat-LCZ42/v4/cities" --cities London
  --year 2017 --family resnet --patch-size 32
  --batch-size 64 --num-workers 8
  --lr 1e-3 --weight-decay 1e-4
  --max-epochs 50 --early-stopping-patience 10
  --output-dir "$D/output/lcz-classification/dl"
)

run () {
  local name=$1; shift
  echo "=== $(date -Is) starting $name ==="
  python src/patch_classification.py "${COMMON[@]}" --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

TV11G=(--output-name GeoTessera_v1.1_global --embedding-name tesserav1.1_global --embedding-dir /tessera/v1.1)
TV11=(--output-name GeoTessera_v1.1 --embedding-name tesserav1.1 --embedding-dir "$D/input/GeoTessera/v1.1/2017")
COOP=(--output-name AlphaEarthCoop --embedding-name alpha_earth_coop --embedding-dir "$D/input/Google/AlphaEarth/coop")

run percity-london-tv11global-small "${TV11G[@]}" --preset small
run percity-london-tv11global-medium "${TV11G[@]}" --preset medium
run percity-london-tv11global-large "${TV11G[@]}" --preset large
run percity-london-coop-small "${COOP[@]}" --preset small
run percity-london-tv11-small "${TV11[@]}" --preset small

echo "=== $(date -Is) per-city chain complete ==="
