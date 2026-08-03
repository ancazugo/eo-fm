#!/usr/bin/env bash
# Phase 0 experiment chain vs opt3-lr5e-4-warmup3 (global kappa 0.619).
# Base recipe = opt3: resnet/small, patch 32, lr 5e-4 + 3-epoch warmup,
# mixup 0.4, wd 1e-3, label smoothing 0.1, monitor val_kappa, TTA, batch 256.
# Each run changes exactly one lever:
#   1. fusion-tessera-coop : Tessera v1.1 + AlphaEarth coop channel fusion (192ch)
#   2. logit-adj-tau1      : logit-adjusted CE (tau=1) instead of sqrt_inv_freq weights
#   3. sampler-sqrt        : sqrt-balanced sampling instead of sqrt_inv_freq weights
set -uo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

COMMON=(
  --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017
  --family resnet --preset small --patch-size 32
  --batch-size 256 --num-workers 8
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4
  --label-smoothing 0.1 --monitor val_kappa --tta
  --max-epochs 50 --early-stopping-patience 10
  --embedding-dir /tessera/v1.1
  --output-dir "$D/output/lcz-classification/dl"
)

run () {
  local name=$1; shift
  echo "=== $(date -Is) starting $name ==="
  python src/patch_classification.py "${COMMON[@]}" --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

run fusion-tessera-coop \
  --output-name GeoTessera_v1.1_global AlphaEarthCoop \
  --embedding-name tesserav1.1_global alpha_earth_coop \
  --class-weights sqrt_inv_freq

run logit-adj-tau1 \
  --output-name GeoTessera_v1.1_global \
  --embedding-name tesserav1.1_global \
  --class-weights none --logit-adjustment 1.0

run sampler-sqrt \
  --output-name GeoTessera_v1.1_global \
  --embedding-name tesserav1.1_global \
  --class-weights none --sampler sqrt_balanced

echo "=== $(date -Is) chain complete ==="
