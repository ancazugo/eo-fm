#!/usr/bin/env bash
# Paper (embedding comparison) — probe-family grid + seed variance, 2026-07-14.
# 1. kNN k=20 GAP for coop & seamless (tessera row = wise-lake-259 0.499) — cached feats, fast.
# 2. Linear probe (GAP) per embedding, plain recipe (probe convention: no mixup/ls),
#    sqrt_inv_freq weights, val_kappa monitor, dihedral TTA at eval.
# 3. opt3 recipe x 2 extra seeds per embedding (seed 1, 2; originals used default seed)
#    for mean±sd on the solo comparison table.
set -uo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

COMMON=(
  --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017
  --patch-size 32 --batch-size 256 --num-workers 8
  --monitor val_kappa --tta
  --max-epochs 50 --early-stopping-patience 10
  --output-dir "$D/output/lcz-classification/dl"
)

OPT3=(
  --family resnet --preset small
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4
  --label-smoothing 0.1 --class-weights sqrt_inv_freq
)

PROBE=(
  --family linear_probe
  --class-weights sqrt_inv_freq
)

# output-name / embedding-name / embedding-dir triplets
TESSERA=(--output-name GeoTessera_v1.1_global --embedding-name tesserav1.1_global --embedding-dir /tessera/v1.1)
COOP=(--output-name AlphaEarthCoop --embedding-name alpha_earth_coop --embedding-dir "$D/input/Google/AlphaEarth/coop")
SEAMLESS=(--output-name EmbeddedSeamless --embedding-name seamless --embedding-dir "$D/input/EmbeddedSeamlessData/2017")

run_cls () {
  local name=$1; shift
  echo "=== $(date -Is) starting $name ==="
  python src/patch_classification.py "${COMMON[@]}" --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

run_knn () {
  local name=$1; shift
  echo "=== $(date -Is) starting $name ==="
  python src/knn_baseline.py \
    --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017 \
    --pooling gap --classifier knn --knn-k 20 \
    --output-dir "$D/output/lcz-classification/dl" \
    --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

# --- 1. kNN grid completion (CPU-heavy, cached features) ---
run_knn paper-knn-coop --output-name AlphaEarthCoop --embedding-name alpha_earth_coop
run_knn paper-knn-seamless --output-name EmbeddedSeamless --embedding-name seamless

# --- 2. Linear probes (all three; tessera's only prior probe was pre-bugfix) ---
run_cls paper-lp-tessera "${PROBE[@]}" "${TESSERA[@]}"
run_cls paper-lp-coop "${PROBE[@]}" "${COOP[@]}"
run_cls paper-lp-seamless "${PROBE[@]}" "${SEAMLESS[@]}"

# --- 3. opt3 seed variance (2 extra seeds each) ---
run_cls opt3-tessera-seed1 "${OPT3[@]}" "${TESSERA[@]}" --seed 1
run_cls opt3-tessera-seed2 "${OPT3[@]}" "${TESSERA[@]}" --seed 2
run_cls opt3-coop-seed1 "${OPT3[@]}" "${COOP[@]}" --seed 1
run_cls opt3-coop-seed2 "${OPT3[@]}" "${COOP[@]}" --seed 2
run_cls opt3-seamless-seed1 "${OPT3[@]}" "${SEAMLESS[@]}" --seed 1
run_cls opt3-seamless-seed2 "${OPT3[@]}" "${SEAMLESS[@]}" --seed 2

echo "=== $(date -Is) paper probe/seed chain complete ==="
