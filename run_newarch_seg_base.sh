#!/usr/bin/env bash
# fcn8/base on both embeddings — locating the capacity ceiling.
#
# The nano->large jump was worth +9.4 kappa on Tessera (0.3955 -> 0.4892) but
# +0.2 on coop (0.3750 -> 0.3768), and coop's large run early-stopped at epoch 15
# against Tessera's 36. The reading is that coop runs out of extractable signal
# long before the model runs out of capacity. `base` (477,373 params at 128
# channels, 462k at 64) sits between the two rungs and says where each curve
# flattens: on Tessera, whether the gain is already banked by 477k or needs the
# full 1.8M; on coop, whether the ceiling is below 477k or the whole nano->large
# range is flat.
#
# Rows differ from their large counterparts ONLY in --preset, and from each other
# only in the embedding triple and the matching --require-embeddings, mirroring
# run_seg_ladder.sh rows 5 and 4 respectively. Default --seed 42 throughout keeps
# the six val-inner cities identical across all of it.
#
# Gate: base measured 2,211 MiB allocator peak at the real training shape
# (batch 16, 129x129, 128ch) against large's 4,171. nvidia-smi will read roughly
# a GB above that, so 4000 is the honest admission threshold here.
set -uo pipefail

cd "$(dirname "$0")"
export GPU_NEED_MIB=4000
source ./run_phase2_gpu_gate.sh
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

SEG_COMMON=(
  --cities-dir "$D/input/So2Sat-LCZ42/v4/cities" --cities all
  --year 2017 --label-source gpkg
  --split-mode global --val-inner-cities 6 --buffer-km 1.3
  --family fcn8 --preset base
  --normalize channel --dice-weight 0.0 --monitor val_kappa --tta
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3
  --label-smoothing 0.1 --class-weights sqrt_inv_freq
  --max-epochs 50 --early-stopping-patience 10
  --batch-size 16 --num-workers 4 --no-inference
  --output-dir "$D/output/lcz-segmentation/dl"
)

run () {
  local name=$1; shift
  if already_complete "$LOGS/$name.log"; then
    echo "=== $(date -Is) skipping $name (already finished) ==="; return 0
  fi
  wait_for_gpu "$name"
  echo "=== $(date -Is) starting $name ==="
  python src/semantic_segmentation.py "${SEG_COMMON[@]}" --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

run "newarch-seg-fcn8-base-tessera" \
  --output-name GeoTessera_v1.1_global \
  --embedding-name tesserav1.1_global \
  --embedding-dir /tessera/v1.1 \
  --require-embeddings AlphaEarthCoop

run "newarch-seg-fcn8-base-coop" \
  --output-name AlphaEarthCoop \
  --embedding-name alpha_earth_coop \
  --embedding-dir "$D/input/Google/AlphaEarth/coop" \
  --require-embeddings GeoTessera_v1.1_global

echo "=== $(date -Is) fcn8/base chain complete ==="
