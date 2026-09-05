#!/usr/bin/env bash
# The four new-architecture configurations repeated on AlphaEarth coop 2017,
# cultural split — the embedding arm of the comparison whose Tessera arm is
# run_newarch_chain.sh + run_newarch_cls_small.sh + run_newarch_seg_large.sh.
#
#   1  shallow_cnn/nano   patch classification   (Tessera: kappa 0.6330)
#   2  shallow_cnn/small  patch classification   (Tessera: kappa 0.6342)
#   3  fcn8/nano          semantic segmentation  (Tessera: kappa 0.3955)
#   4  fcn8/large         semantic segmentation  (Tessera: running)
#
# Every flag matches its Tessera counterpart except the embedding triple, so
# each pair isolates the embedding exactly.
#
# The classification runs keep --patch-manifest: `in_manifest` is the three-way
# coop-tessera-seamless intersection (389,484 of 400,673 patches, verified), so
# it is what puts both embeddings on identical data. Coop actually holds all
# 400,673 patches while Tessera is short 9,993 — without the manifest the coop
# arm would train and score on ~9.4k patches Tessera never saw, and the
# difference would partly be a difference in the test set.
#
# The segmentation runs mirror run_seg_ladder.sh row 4, including
# --require-embeddings GeoTessera_v1.1_global, holding coop to Tessera's ground
# (the 2017 archives differ: Tessera is short 952 tiles, 32% of Istanbul and 37%
# of Qingdao). Default --seed 42 throughout, which keeps the six val-inner cities
# identical — assign_city_roles is seeded from args.seed, so changing it would
# resample the split rather than just the init.
#
# Nearest existing classification reference is opt3-coop (resnet/small, kappa
# 0.5195/0.5288/0.5177), but those predate this config (no manifest, --normalize
# none), so cross-architecture coop comparison is approximate; the coop-vs-Tessera
# comparison within these runs is exact. Segmentation has an exact reference:
# seg-row4-coop-global-small (unet/small, kappa 0.4533).
#
# Waits for the Tessera fcn8/large run to finish first. One shared T4, so all
# four are sequential and gated on free GPU memory. Coop is 64 channels against
# Tessera's 128, so memory is strictly lower than the 4,171 MiB measured there.
set -uo pipefail

cd "$(dirname "$0")"
export GPU_NEED_MIB=6000
source ./run_phase2_gpu_gate.sh
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
LOGS=$D/output/lcz-classification/dl/_experiment_logs
mkdir -p "$LOGS"

MANIFEST=diagnostics/patch_manifest_v1.parquet
[[ -f $MANIFEST ]] || { echo "FATAL: manifest not found at $MANIFEST"; exit 1; }

COOP=(--output-name AlphaEarthCoop --embedding-name alpha_earth_coop
      --embedding-dir "$D/input/Google/AlphaEarth/coop")

# The gate's already_complete only knows the classification completion marker.
# A --no-inference seg run never prints it: semantic_segmentation.py logs
# "--no-inference: skipping city rasters. Outputs in ..." and returns. This
# accepts both, so the skip-if-done guard actually works for seg rows -- unlike
# run_seg_ladder.sh, whose finished rows would be re-trained on a re-run.
run_complete () {
  local log=$1
  [[ -f $log ]] && grep -qE "Run complete\. Outputs in|--no-inference: skipping city rasters\. Outputs in" "$log"
}

run () {
  local name=$1 script=$2; shift 2
  if run_complete "$LOGS/$name.log"; then
    echo "=== $(date -Is) skipping $name (already finished) ==="; return 0
  fi
  wait_for_gpu "$name"
  echo "=== $(date -Is) starting $name ==="
  python "$script" --run-name "$name" "$@" > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) finished $name (exit $?) ==="
}

# ── Wait for the Tessera fcn8/large run to finish ────────────────────────────
PRIOR=newarch-seg-fcn8-large-tessera
waited=0
while pgrep -f "run-name $PRIOR" > /dev/null 2>&1; do
  if (( waited % 1800 == 0 )); then
    echo "$(date -Is) waiting for $PRIOR to finish (${waited}s so far)"
  fi
  sleep 120
  waited=$(( waited + 120 ))
done
echo "$(date -Is) $PRIOR is no longer running (waited ${waited}s)"
run_complete "$LOGS/$PRIOR.log" \
  && echo "$(date -Is) $PRIOR completed normally" \
  || echo "$(date -Is) WARNING: $PRIOR did not print a completion marker — it crashed or was killed"

CLS_COMMON=(
  --so2sat-dir "$D/input/So2Sat-LCZ42/v4" --global-split --year 2017
  --patch-size 32 --batch-size 256 --num-workers 8
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3 --mixup-alpha 0.4
  --label-smoothing 0.1 --class-weights sqrt_inv_freq --monitor val_kappa --tta
  --max-epochs 50 --early-stopping-patience 10 --seed 0
  --patch-manifest "$MANIFEST"
  --output-dir "$D/output/lcz-classification/dl"
)

SEG_COMMON=(
  --cities-dir "$D/input/So2Sat-LCZ42/v4/cities" --cities all
  --year 2017 --label-source gpkg
  --split-mode global --val-inner-cities 6 --buffer-km 1.3
  --normalize channel --dice-weight 0.0 --monitor val_kappa --tta
  --lr 5e-4 --warmup-epochs 3 --weight-decay 1e-3
  --label-smoothing 0.1 --class-weights sqrt_inv_freq
  --max-epochs 50 --early-stopping-patience 10
  --batch-size 16 --num-workers 4 --no-inference
  --require-embeddings GeoTessera_v1.1_global
  --output-dir "$D/output/lcz-segmentation/dl"
)

run "newarch-cls-shallow-cnn-nano-coop"  src/patch_classification.py \
  "${CLS_COMMON[@]}" "${COOP[@]}" --family shallow_cnn --preset nano

run "newarch-cls-shallow-cnn-small-coop" src/patch_classification.py \
  "${CLS_COMMON[@]}" "${COOP[@]}" --family shallow_cnn --preset small

run "newarch-seg-fcn8-nano-coop"  src/semantic_segmentation.py \
  "${SEG_COMMON[@]}" "${COOP[@]}" --family fcn8 --preset nano

run "newarch-seg-fcn8-large-coop" src/semantic_segmentation.py \
  "${SEG_COMMON[@]}" "${COOP[@]}" --family fcn8 --preset large

echo "=== $(date -Is) coop chain complete ==="
