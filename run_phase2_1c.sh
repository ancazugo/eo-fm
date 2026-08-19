#!/usr/bin/env bash
# PLAN-v3 Phase 2, Task 2.1c — learning-rate and schedule audit.
#
# Two of three anchor seeds selected their best checkpoint at epoch 2 with
# --warmup-epochs 3, i.e. BEFORE warmup finished, at a low learning rate, after
# which val_kappa degraded while training loss kept falling. The opt3 recipe was
# inherited from grid-split work where memorising city-specific features is
# rewarded; on the cultural split it may simply be too aggressive.
#
# CONFIG IS TASK 2.2 ARM A, not the anchor. Phase 1 defaults (--normalize channel
# --nodata-mode mask, both CLI defaults) plus the Task 2.0 manifest, so whatever
# this sweep picks transfers directly to Task 2.2's baseline. Running it on the
# anchor's pre-fix flags would answer a question about a configuration no later
# task uses.
#
# Usage:
#   ./run_phase2_1c.sh stage1              # LR sweep, 5 LRs x 3 seeds
#   ./run_phase2_1c.sh stage2 <best-lr>    # warmup 0 and 1 at the winning LR
#
# Stage 2 is deliberately NOT launched blind: it needs stage 1's answer, and
# warmup 3 at the best LR already exists in stage 1 rather than being re-run.
set -uo pipefail

cd "$(dirname "$0")"
source ./run_phase2_gpu_gate.sh
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

MANIFEST=diagnostics/patch_manifest_v1.parquet
[[ -f $MANIFEST ]] || { echo "FATAL: manifest not found at $MANIFEST"; exit 1; }

COMMON=(
  --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017
  --family resnet --preset small --patch-size 32
  --batch-size 256 --num-workers 8
  --weight-decay 1e-3 --mixup-alpha 0.4
  --label-smoothing 0.1 --class-weights sqrt_inv_freq --monitor val_kappa --tta
  --max-epochs 50 --early-stopping-patience 10
  --output-name GeoTessera_v1.1_global
  --embedding-name tesserav1.1_global
  --embedding-dir /tessera/v1.1
  --patch-manifest "$MANIFEST"
  --output-dir "$D/output/lcz-classification/dl"
)
# --lr and --warmup-epochs are swept, so they are NOT in COMMON.
# --normalize / --nodata-mode are left at their defaults (channel / mask) on
# purpose: those ARE Task 2.2 Arm A. Naming them here would invite someone to
# change them and quietly turn this into a different experiment.

run () {
  local name=$1; shift
  if already_complete "$LOGS/$name.log"; then
    echo "=== $(date -Is) skipping $name (already finished) ==="; return 0
  fi
  wait_for_gpu "$name"
  echo "=== $(date -Is) starting $name ==="
  python src/patch_classification.py "${COMMON[@]}" --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

stage=${1:-stage1}

case "$stage" in
  stage1)
    for lr in 5e-5 1e-4 2.5e-4 5e-4 1e-3; do
      for s in 0 1 2; do
        run "p2-1c-lr${lr}-w3-seed$s" --lr "$lr" --warmup-epochs 3 --seed "$s"
      done
    done
    ;;
  stage2)
    best_lr=${2:?stage2 needs the winning LR from stage 1, e.g. ./run_phase2_1c.sh stage2 1e-4}
    for w in 0 1; do
      for s in 0 1 2; do
        run "p2-1c-lr${best_lr}-w${w}-seed$s" --lr "$best_lr" --warmup-epochs "$w" --seed "$s"
      done
    done
    ;;
  *)
    echo "unknown stage: $stage (expected stage1 or stage2)"; exit 2 ;;
esac

echo "=== $(date -Is) 2.1c $stage complete ==="
