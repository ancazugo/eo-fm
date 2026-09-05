#!/usr/bin/env bash
# fcn8/large on Tessera v1.1 global 2017, cultural split — the matched-capacity
# comparison against unet/small.
#
# fcn8/large is 1,818,653 params against unet/small's 1,963,537 (within 7%), and
# every preset from `small` up is depth 3, so this is the first fcn8 run that
# both matches the reference's capacity AND performs the real three-scale
# stride-8 score fusion. fcn8/nano's (2, 8) payload is depth 2 — two scales from
# stride 4 — so its -14.7 kappa against unet/small conflated architecture with a
# 110x capacity gap and a degenerate fusion depth. This run separates them.
#
# Identical to run_seg_ladder.sh row 5 in every other respect, including the
# default --seed 42, which is what keeps the six val-inner cities (Cairo,
# Istanbul, London, Melbourne, Rio_De_Janeiro, Vancouver) identical: the role
# assignment is seeded from args.seed, so changing it would resample the split
# rather than just the weight init.
#
# Memory: measured 4,171 MiB allocator peak at the real training shape (batch 16,
# 129x129, 128ch, forward+backward+Adam). nvidia-smi will show ~5.2 GB once the
# CUDA context and caching allocator are counted -- fcn8/nano measured 490 MiB
# but showed 1,440 MiB in its actual run -- so the gate is raised from 5000 to
# 6000 to avoid starting into a card that cannot quite hold it.
set -uo pipefail

cd "$(dirname "$0")"
export GPU_NEED_MIB=6000
source ./run_phase2_gpu_gate.sh
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

NAME=newarch-seg-fcn8-large-tessera

# The gate's already_complete greps "Run complete. Outputs in", which a
# --no-inference seg run NEVER prints: semantic_segmentation.py logs
# "--no-inference: skipping city rasters. Outputs in ..." and returns instead.
# Using the gate's version here would leave this guard permanently dead -- which
# is exactly the live bug in run_seg_ladder.sh, whose finished rows 4 and 5
# contain zero occurrences of the marker and would be re-trained on a re-run.
seg_complete () {
  local log=$1
  [[ -f $log ]] && grep -qE "Run complete\. Outputs in|--no-inference: skipping city rasters\. Outputs in" "$log"
}

if seg_complete "$LOGS/$NAME.log"; then
  echo "=== $(date -Is) skipping $NAME (already finished) ==="
  exit 0
fi

wait_for_gpu "$NAME"
echo "=== $(date -Is) starting $NAME ==="
python src/semantic_segmentation.py --run-name "$NAME" \
  --cities-dir "$D/input/So2Sat-LCZ42/v4/cities" --cities all \
  --year 2017 --label-source gpkg \
  --split-mode global --val-inner-cities 6 --buffer-km 1.3 \
  --family fcn8 --preset large \
  --normalize channel --dice-weight 0.0 --monitor val_kappa --tta \
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 \
  --label-smoothing 0.1 --class-weights sqrt_inv_freq \
  --max-epochs 50 --early-stopping-patience 10 \
  --batch-size 16 --num-workers 4 \
  --no-inference \
  --output-name GeoTessera_v1.1_global \
  --embedding-name tesserav1.1_global \
  --embedding-dir /tessera/v1.1 \
  --require-embeddings AlphaEarthCoop \
  --output-dir "$D/output/lcz-segmentation/dl" \
  > "$LOGS/$NAME.log" 2>&1
echo "=== $(date -Is) finished $NAME (exit $?) ==="
