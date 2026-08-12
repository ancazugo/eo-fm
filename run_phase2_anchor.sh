#!/usr/bin/env bash
# PLAN-v3 Phase 2, Task 2.1 — reproduce the pre-fix baseline (the anchor).
#
# Three seeds of the original opt3 recipe on tesserav1.1_global, with the
# PRE-Phase-1 flags: --normalize none --nodata-mode zero, no coverage filter and
# no manifest. Every later Phase 2 number is measured against the seed band this
# produces, and the existing opt3 kappa 0.6190 has to fall inside it.
#
# The Phase 1 RNG change means seed-level, not bit-level, agreement is the
# standard (tests/test_augment_distribution.py pins that the augmentation
# DISTRIBUTION is unchanged; only the draw order moved).
#
# Deliberately NOT set, and why:
#   --patch-manifest      Task 2.0's manifest would change the training and test
#                         populations. filter_by_manifest(items, None) returns
#                         the input list object, so its absence is a provable
#                         no-op rather than an assumed one.
#   --min-native-frac     Not a flag on this CLI; Rev A's "0.0" is inert. It
#                         arrives with Arm B in Task 2.2, where it first does work.
#   --normalize channel   The anchor predates it. With --normalize none the 0.05
#                         noise sigma is ABSOLUTE, which is the old behaviour and
#                         cell C of Task 2.3's factorial. The CLI warns about the
#                         cross-family confound here; that is expected and correct.
set -uo pipefail

cd "$(dirname "$0")"
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
  echo "=== $(date -Is) starting $name ==="
  python src/patch_classification.py "${COMMON[@]}" --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

# One T4, so the seeds run sequentially.
for s in 0 1 2; do
  run "p2-anchor-tessera-seed$s" --seed "$s"
done

echo "=== $(date -Is) anchor chain complete ==="
