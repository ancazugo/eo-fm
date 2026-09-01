#!/usr/bin/env bash
# Segmentation timing calibration — the first GPU run of the seg pipeline.
#
# Purpose is a NUMBER, not a result: seconds per epoch and peak GPU memory for
# the intended row-4 configuration at full scale, so the real ablation can be
# scheduled instead of guessed. Every timing so far is from nano-preset CPU
# smoke tests and says nothing about GPU cost.
#
# So the config is the intended headline command with two changes, and only two:
#   --max-epochs 3   enough epochs to see a steady-state epoch time (epoch 1
#                    carries page-cache warming, so it is discarded)
#   --no-inference   the 51-city ROI rasters cost far more than 3 epochs and
#                    are irrelevant to the question
#
# It deliberately runs on ALL cities under --split-mode global, because a subset
# would have to be extrapolated and the item-building pass over 51 cities is
# itself part of the cost being measured.
#
# --require-embeddings GeoTessera_v1.1_global holds this to the same tiles the
# Tessera arm can supply, matching what row 4 will actually run.
set -uo pipefail

cd "$(dirname "$0")"
source ./run_phase2_gpu_gate.sh
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

NAME=seg-calib-coop-global-small
LOG=$LOGS/$NAME.log

if already_complete "$LOG"; then
  echo "=== $(date -Is) skipping $NAME (already finished) ==="
  exit 0
fi

wait_for_gpu "$NAME"
echo "=== $(date -Is) starting $NAME ==="

python src/semantic_segmentation.py \
  --cities-dir "$D/input/So2Sat-LCZ42/v4/cities" --cities all \
  --output-name AlphaEarthCoop --year 2017 --label-source gpkg \
  --split-mode global --val-inner-cities 6 --buffer-km 1.3 \
  --require-embeddings GeoTessera_v1.1_global \
  --family unet --preset small \
  --normalize channel --dice-weight 0.0 --monitor val_kappa \
  --batch-size 16 --num-workers 4 \
  --max-epochs 3 --early-stopping-patience 10 \
  --no-inference --no-wandb \
  --embedding-name alpha_earth_coop \
  --embedding-dir "$D/input/Google/AlphaEarth/coop" \
  --run-name "$NAME" \
  --output-dir "$D/output/lcz-segmentation/dl" \
  > "$LOG" 2>&1
echo "=== $(date -Is) finished $NAME (exit $?) ==="
