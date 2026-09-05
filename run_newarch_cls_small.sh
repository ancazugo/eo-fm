#!/usr/bin/env bash
# shallow_cnn/small on Tessera v1.1 global 2017, cultural split — the capacity
# rung above the nano run, chained to start when the fcn8 seg run finishes.
#
# Separate script rather than an edit to run_newarch_chain.sh: bash reads a
# script lazily by byte offset, so editing a file that is currently executing
# makes the running shell resume at a stale offset and execute garbage. The
# chain is mid-fcn8, so that file is off limits until it exits.
#
# Identical to the nano run in every respect except --preset small
# (scnn_32-64, 56,593 params at 128 input channels, against nano's 23,697), so
# the pair isolates capacity. Same --seed 0, so both also line up with
# p2-1c-lr5e-4-w3-seed0 (resnet/small, ~21M params, kappa 0.6145).
#
# Waits on the fcn8 process itself rather than on a log marker, so it starts
# however that run ends — completed, crashed or killed. The GPU gate still
# guards admission afterwards.
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

NAME=newarch-cls-shallow-cnn-small-tessera
SEG_RUN=newarch-seg-fcn8-nano-tessera

# ── Wait for the fcn8 run to exit ────────────────────────────────────────────
waited=0
while pgrep -f "run-name $SEG_RUN" > /dev/null 2>&1; do
  if (( waited % 1800 == 0 )); then
    echo "$(date -Is) waiting for $SEG_RUN to finish (${waited}s so far)"
  fi
  sleep 120
  waited=$(( waited + 120 ))
done
echo "$(date -Is) $SEG_RUN is no longer running (waited ${waited}s)"
if already_complete "$LOGS/$SEG_RUN.log"; then
  echo "$(date -Is) $SEG_RUN completed normally"
else
  echo "$(date -Is) WARNING: $SEG_RUN did not print its completion marker — it crashed or was killed"
fi

# ── Run: shallow_cnn/small, patch classification ─────────────────────────────
if already_complete "$LOGS/$NAME.log"; then
  echo "=== $(date -Is) skipping $NAME (already finished) ==="
  exit 0
fi
wait_for_gpu "$NAME"
echo "=== $(date -Is) starting $NAME ==="
python src/patch_classification.py --run-name "$NAME" \
  --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017 \
  --family shallow_cnn --preset small --patch-size 32 \
  --batch-size 256 --num-workers 8 \
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4 \
  --label-smoothing 0.1 --class-weights sqrt_inv_freq --monitor val_kappa --tta \
  --max-epochs 50 --early-stopping-patience 10 --seed 0 \
  --output-name GeoTessera_v1.1_global \
  --embedding-name tesserav1.1_global \
  --embedding-dir /tessera/v1.1 \
  --patch-manifest "$MANIFEST" \
  --output-dir "$D/output/lcz-classification/dl" \
  > "$LOGS/$NAME.log" 2>&1
echo "=== $(date -Is) finished $NAME (exit $?) ==="
