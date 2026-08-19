#!/usr/bin/env bash
# PLAN-v3 Phase 2, Task 2.1b-iii — four more seeds of the anchor.
#
# Tasks 2.1b-i and 2.1b-ii excluded both mechanisms Rev C named for the anchor's
# ~0.8-point shortfall, which unblocks this: seeds 3-6 take the new side to n=7,
# for roughly 70% power against the fixed historical n=3.
#
# The question is NOT only whether the mean shifted. The anchor's three seeds peak
# at epochs 2, 2 and 15 — two selecting their best checkpoint during warmup at a
# low LR. If peak epoch is bimodal, three draws from it against a historical three
# explains the gap with no code mechanism, and nothing further is chased.
#
# CONFIG IS THE TASK 2.1 ANCHOR, UNCHANGED. The COMMON array below is copied
# verbatim from run_phase2_anchor.sh; the only difference is --seed. Anything else
# would make these four runs incomparable with the three they extend, which is the
# entire point of running them.
set -uo pipefail

cd "$(dirname "$0")"
source ./run_phase2_gpu_gate.sh
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

COMMON=(
  --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017
  --family resnet --preset small --patch-size 32
  --batch-size 256 --num-workers 8
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4
  --label-smoothing 0.1 --class-weights sqrt_inv_freq --monitor val_kappa --tta
  --max-epochs 50 --early-stopping-patience 10
  --output-name GeoTessera_v1.1_global
  --embedding-name tesserav1.1_global
  --embedding-dir /tessera/v1.1
  --normalize none --nodata-mode zero --max-invalid-frac 1.0
  --output-dir "$D/output/lcz-classification/dl"
)

run () {
  local name=$1; shift
  wait_for_gpu "$name"
  echo "=== $(date -Is) starting $name ==="
  python src/patch_classification.py "${COMMON[@]}" --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

for s in 3 4 5 6; do
  run "p2-anchor-tessera-seed$s" --seed "$s"
done

echo "=== $(date -Is) 2.1b-iii chain complete ==="
