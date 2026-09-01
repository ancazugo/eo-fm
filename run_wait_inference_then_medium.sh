#!/usr/bin/env bash
# Waits for the 6 city-inference maps (rows 4/5 x Nairobi/Cairo/London) to
# finish, then launches the medium-preset seg ladder. Lives in tmux so it
# survives the launching Claude session ending.
set -uo pipefail
cd "$(dirname "$0")"

LOGDIR="/tmp/claude-1016/-home-acz25-repos-eo-fm/6edf2714-360b-47c9-b6ae-d52cb8d27c4c/scratchpad"
LOGS=(
  "$LOGDIR/infer_row4_Nairobi.log" "$LOGDIR/infer_row4_Cairo.log" "$LOGDIR/infer_row4_London.log"
  "$LOGDIR/infer_row5_Nairobi.log" "$LOGDIR/infer_row5_Cairo.log" "$LOGDIR/infer_row5_London.log"
)

echo "$(date -Is) waiting for ${#LOGS[@]} inference runs to finish …"
for f in "${LOGS[@]}"; do
  while [[ ! -f "$f" ]] || ! grep -q 'exit=' "$f"; do
    sleep 15
  done
  echo "$(date -Is)   done: $(basename "$f")"
done
echo "$(date -Is) all inference maps finished — starting the medium seg ladder"

./run_seg_ladder_medium.sh
