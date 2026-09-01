#!/usr/bin/env bash
# Segmentation ablation ladder, rows 4 and 5 — the first real seg results.
#
#   row 4  U-Net on AlphaEarth coop 2017     (primary)
#   row 5  U-Net on Tessera v1.1 global 2017 (second embedding)
#
# The two rows differ ONLY in the embedding. That is the entire point, so both
# carry --require-embeddings naming the other: the 2017 archives do not cover
# the same ground (Tessera is short 952 tiles, 32% of Istanbul and 37% of
# Qingdao), and without the restriction a gap between the rows would partly be a
# gap in which cities each model saw. Held to the intersection, both train and
# score on 13,719 identical tiles.
#
# Schedule: lr 5e-4 + 3-epoch warmup, which Task 2.1c stage 1 independently
# picked as the optimum for the patch task (0.6304, against 0.6183 at 2.5e-4 and
# 0.6202 at 1e-3). It is evidence from a sibling pipeline rather than a seg
# sweep, so it is recorded here as an assumption, not a result.
#
# Mixup is absent by construction (LCZUNetModule has none) and --dice-weight 0
# per the plan: Dice on sparsely-labelled tiles optimises a quantity whose
# denominator is the labelled subset.
#
# --no-inference: the 51-city ROI rasters cost far more than training and are
# not needed for the ladder numbers. Generate them later from the checkpoints
# with infer_roi.py --normalize auto (the stats travel in the checkpoint).
#
# Calibrated at ~5m50s/epoch, so each row is roughly 5 hours at 50 epochs.
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
  --family unet --preset small
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
run "seg-row4-coop-global-small" \
  --output-name AlphaEarthCoop \
  --embedding-name alpha_earth_coop \
  --embedding-dir "$D/input/Google/AlphaEarth/coop" \
  --require-embeddings GeoTessera_v1.1_global

# Row 5 — Tessera v1.1 global, same ground (a no-op filter today, but stated so
# the two commands are symmetric and the intent survives an archive update).
run "seg-row5-tessera-global-small" \
  --output-name GeoTessera_v1.1_global \
  --embedding-name tesserav1.1_global \
  --embedding-dir /tessera/v1.1 \
  --require-embeddings AlphaEarthCoop

echo "=== $(date -Is) seg ladder rows 4-5 complete ==="
