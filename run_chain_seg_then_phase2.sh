#!/usr/bin/env bash
# Sequential chain: segmentation calibration, then the paused Phase 2 revC work.
#
# Order is deliberate. The calibration is minutes and answers "what does a seg
# epoch cost", which is needed before any seg run can be scheduled. Phase 2 is
# days and was paused only for GPU contention, so it takes the card afterwards
# and keeps it.
#
# Each step is independently resumable — the chain scripts skip runs whose log
# carries the completion marker — so re-running this after an interruption
# continues rather than restarting.
set -uo pipefail
cd "$(dirname "$0")"

# Keep the chain log beside the per-run logs rather than in the repo.
source .env 2>/dev/null || true
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGDIR=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)
CHAIN_LOG=$LOGDIR/chain-${STAMP}.log
echo "chain log: $CHAIN_LOG"

step () {
  local label=$1; shift
  echo "############################################################" | tee -a "$CHAIN_LOG"
  echo "### $(date -Is) START $label" | tee -a "$CHAIN_LOG"
  echo "############################################################" | tee -a "$CHAIN_LOG"
  "$@" 2>&1 | tee -a "$CHAIN_LOG"
  local rc=${PIPESTATUS[0]}
  echo "### $(date -Is) END $label (exit $rc)" | tee -a "$CHAIN_LOG"
  return $rc
}

# A failing step must not silently take the rest of the chain down with it:
# Phase 2 is the long-pole work and does not depend on the calibration.
step "seg calibration"   ./run_seg_calibration.sh || echo "!!! seg calibration failed — continuing to Phase 2" | tee -a "$CHAIN_LOG"
step "phase2 2.1b-iii"   ./run_phase2_1b_iii.sh   || echo "!!! 2.1b-iii failed" | tee -a "$CHAIN_LOG"
step "phase2 2.1c stage1" ./run_phase2_1c.sh stage1 || echo "!!! 2.1c stage1 failed" | tee -a "$CHAIN_LOG"

echo "=== $(date -Is) CHAIN COMPLETE ===" | tee -a "$CHAIN_LOG"
