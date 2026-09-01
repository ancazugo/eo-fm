#!/usr/bin/env bash
# Sequential chain: Phase 2 Task 2.1c stage 2, then segmentation ladder rows 4-5.
#
# Stage 2 first because it closes the schedule question stage 1 opened -- it
# tests warmup 0 and 1 against the winning LR, and warmup 3 at that LR already
# exists in stage 1 rather than being re-run. The seg rows inherit 5e-4/warmup 3
# from stage 1; if stage 2 overturns warmup 3, the seg rows are the thing to
# re-run, which is why they come second.
#
# Both steps are independently resumable (runs whose log carries the completion
# marker are skipped), so re-running this after an interruption continues.
set -uo pipefail
cd "$(dirname "$0")"

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

step "phase2 2.1c stage2" ./run_phase2_1c.sh stage2 5e-4 \
  || echo "!!! 2.1c stage2 failed — continuing to the seg ladder" | tee -a "$CHAIN_LOG"
step "seg ladder rows 4-5" ./run_seg_ladder.sh \
  || echo "!!! seg ladder failed" | tee -a "$CHAIN_LOG"

echo "=== $(date -Is) CHAIN COMPLETE ===" | tee -a "$CHAIN_LOG"
