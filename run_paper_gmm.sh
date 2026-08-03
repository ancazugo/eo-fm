#!/usr/bin/env bash
# Paper probe grid — GMM density baseline (global split) for all three embeddings.
# CPU-only (cached GAP features); runs in parallel with the GPU probe/seed chain.
# Identical defaults across embeddings (diag covariance, default components/prior).
set -uo pipefail

cd "$(dirname "$0")"
# OpenBLAS on this box segfaults when thread count exceeds its compiled
# NUM_THREADS ("precompiled NUM_THREADS exceeded" warning) — cap BLAS threads.
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

run_gmm () {
  local name=$1; shift
  echo "=== $(date -Is) starting $name ==="
  python src/knn_baseline.py \
    --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017 \
    --pooling gap --classifier gmm \
    --output-dir "$D/output/lcz-classification/dl" \
    --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

run_gmm paper-gmm-tessera --output-name GeoTessera_v1.1_global --embedding-name tesserav1.1_global
run_gmm paper-gmm-coop --output-name AlphaEarthCoop --embedding-name alpha_earth_coop
run_gmm paper-gmm-seamless --output-name EmbeddedSeamless --embedding-name seamless

echo "=== $(date -Is) GMM chain complete ==="
