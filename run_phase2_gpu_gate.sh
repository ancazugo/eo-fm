#!/usr/bin/env bash
# Shared GPU admission gate for the Phase 2 chains.
#
# The machine has ONE Tesla T4 (15,360 MiB) and it is shared. A patch_classification
# run at --preset small / --batch-size 256 peaks at 3,681 MiB and holds the SMs at
# ~93% (measured: W&B system metrics for the anchor run p3x6xm0v), so a chain that
# launches into a busy card either OOMs at startup or crawls.
#
# Sourced by the chain scripts. wait_for_gpu blocks until the card has room, which
# is also what lets a second chain start early if the card frees up — concurrency
# when it is actually available, rather than assumed.

GPU_NEED_MIB=${GPU_NEED_MIB:-5000}     # 3,681 measured + ~1.3 GB headroom
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-120}

wait_for_gpu () {
  local label=$1 waited=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ -z "$free" ]]; then
      echo "$(date -Is) [$label] nvidia-smi gave no reading — proceeding anyway"
      return 0
    fi
    if (( free >= GPU_NEED_MIB )); then
      echo "$(date -Is) [$label] GPU has ${free} MiB free (need ${GPU_NEED_MIB}) — starting after ${waited}s of waiting"
      return 0
    fi
    if (( waited % 1800 == 0 )); then
      echo "$(date -Is) [$label] waiting for GPU: ${free} MiB free, need ${GPU_NEED_MIB} (${waited}s so far)"
    fi
    sleep "$GPU_POLL_SECONDS"
    waited=$(( waited + GPU_POLL_SECONDS ))
  done
}
