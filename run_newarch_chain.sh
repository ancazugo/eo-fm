#!/usr/bin/env bash
# The two new architecture families on Tessera v1.1 global 2017, cultural split.
#
#   run 1  shallow_cnn/nano  — patch classification, --global-split
#   run 2  fcn8/nano         — semantic segmentation, --split-mode global
#
# Each run mirrors the established reference recipe for its pipeline and changes
# ONLY --family/--preset, so each number lands next to an existing one:
#
#   shallow_cnn/nano  vs  p2-1c-lr5e-4-w3-seed0 (resnet/small, Task 2.1c stage-1
#                         winner, kappa 0.6304). Task 2.2 Arm A config: the
#                         --normalize channel / --nodata-mode mask defaults plus
#                         the Task 2.0 manifest, lr 5e-4, warmup 3, seed 0.
#   fcn8/nano         vs  seg-row5-tessera-global-small (unet/small, seg ladder
#                         row 5). Same split, same schedule, and the same
#                         --require-embeddings AlphaEarthCoop restriction, so
#                         both models are held to the identical 13,719 tiles.
#
# Caveat, stated up front: both reference recipes were tuned for models 80-1000x
# larger (resnet/small ~21M params, unet/small ~1.9M, against shallow_cnn/nano
# 23,697 and fcn8/nano 17,295). Mixup 0.4 + label smoothing 0.1 + weight decay
# 1e-3 may underfit at this capacity. Holding the recipe fixed is the point — it
# isolates the architecture. A capacity-matched schedule sweep is a separate
# experiment, and these runs are the evidence for whether one is worth doing.
#
# One shared T4 (and other users on it), so the runs are sequential and gated on
# free GPU memory. Resumable: already_complete keys on the completion marker, so
# re-running the script skips a finished run and redoes an interrupted one.
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

run () {
  local name=$1 script=$2; shift 2
  if already_complete "$LOGS/$name.log"; then
    echo "=== $(date -Is) skipping $name (already finished) ==="; return 0
  fi
  wait_for_gpu "$name"
  echo "=== $(date -Is) starting $name ==="
  python "$script" --run-name "$name" "$@" > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

# ── Run 1: shallow_cnn/nano, patch classification ────────────────────────────
run "newarch-cls-shallow-cnn-nano-tessera" src/patch_classification.py \
  --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017 \
  --family shallow_cnn --preset nano --patch-size 32 \
  --batch-size 256 --num-workers 8 \
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4 \
  --label-smoothing 0.1 --class-weights sqrt_inv_freq --monitor val_kappa --tta \
  --max-epochs 50 --early-stopping-patience 10 --seed 0 \
  --output-name GeoTessera_v1.1_global \
  --embedding-name tesserav1.1_global \
  --embedding-dir /tessera/v1.1 \
  --patch-manifest "$MANIFEST" \
  --output-dir "$D/output/lcz-classification/dl"

# ── Run 2: fcn8/nano, semantic segmentation ──────────────────────────────────
# Seed left at the CLI default (42), matching row 5.
run "newarch-seg-fcn8-nano-tessera" src/semantic_segmentation.py \
  --cities-dir "$D/input/So2Sat-LCZ42/v4/cities" --cities all \
  --year 2017 --label-source gpkg \
  --split-mode global --val-inner-cities 6 --buffer-km 1.3 \
  --family fcn8 --preset nano \
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
  --output-dir "$D/output/lcz-segmentation/dl"

echo "=== $(date -Is) new-architecture chain complete ==="
