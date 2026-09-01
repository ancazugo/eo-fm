#!/usr/bin/env bash
# Segmentation ablation ladder, rows 4 and 5, at --preset medium.
#
# Same two rows as run_seg_ladder.sh (AlphaEarth coop vs Tessera v1.1 global,
# both held to their common tiles via --require-embeddings), same schedule
# (lr 5e-4, warmup 3 — the winner of Task 2.1c stage 1 and 2), same
# --dice-weight 0 / no mixup reasoning from the plan. The only change is
# capacity: medium = (depth 4, base_features 32) vs small's (depth 3, 32) —
# one more downsampling level at the same width.
#
# This is a capacity check, not a rerun of a suspect config: the campaign's own
# prior on the patch task was that capacity does not help (resnet101/152 did
# not beat resnet34), and the plan's §3 predicted the same for a deeper
# resnet_unet without testing it. This is that test, for the plain unet family.
#
# Distinct run-names (…-medium) so the small-preset checkpoints and metrics
# from run_seg_ladder.sh are not overwritten.
#
# --no-inference as before: ROI rasters cost far more than training and are not
# needed for the ladder numbers.
set -uo pipefail

cd "$(dirname "$0")"
source ./run_phase2_gpu_gate.sh
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

COMMON=(
  --cities-dir "$D/input/So2Sat-LCZ42/v4/cities" --cities all
  --year 2017 --label-source gpkg
  --split-mode global --val-inner-cities 6 --buffer-km 1.3
  --family unet --preset medium
  --normalize channel --dice-weight 0.0 --monitor val_kappa --tta
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3
  --label-smoothing 0.1 --class-weights sqrt_inv_freq
  --max-epochs 50 --early-stopping-patience 10
  --batch-size 16 --num-workers 4
  --no-inference
  --output-dir "$D/output/lcz-segmentation/dl"
)

run () {
  local name=$1; shift
  if already_complete "$LOGS/$name.log"; then
    echo "=== $(date -Is) skipping $name (already finished) ==="; return 0
  fi
  wait_for_gpu "$name"
  echo "=== $(date -Is) starting $name ==="
  python src/semantic_segmentation.py "${COMMON[@]}" --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

# Row 4 — AlphaEarth coop, held to Tessera's ground.
run "seg-row4-coop-global-medium" \
  --output-name AlphaEarthCoop \
  --embedding-name alpha_earth_coop \
  --embedding-dir "$D/input/Google/AlphaEarth/coop" \
  --require-embeddings GeoTessera_v1.1_global

# Row 5 — Tessera v1.1 global, same ground.
run "seg-row5-tessera-global-medium" \
  --output-name GeoTessera_v1.1_global \
  --embedding-name tesserav1.1_global \
  --embedding-dir /tessera/v1.1 \
  --require-embeddings AlphaEarthCoop

echo "=== $(date -Is) seg ladder rows 4-5 (medium) complete ==="
