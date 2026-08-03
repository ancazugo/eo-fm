#!/usr/bin/env bash
# Qualitative LCZ prediction maps for 12 non-So2Sat cities (all 3 embeddings),
# matching the _tmp_best_predictions_20260715 pattern:
#   3 classification models x (320m + 160m stride) + 2 seg-distill UNets x 10m.
# Cities chosen 2026-07-16 (diverse dozen, all tessera+coop+seamless covered;
# availability table in the session scratchpad city_availability.csv).
set -uo pipefail

cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
DL=$D/output/lcz-classification/dl
LOGS=$DL/_experiment_logs
OUT=$DL/_tmp_best_predictions_20260715
mkdir -p "$LOGS" "$OUT"

# smod_id|filename_token (pretty name for --city-name derived from token)
CITIES=(
  "30_4732|Cambridge"
  "30_4693|Oxford"
  "30_10403|Utrecht"
  "30_3455|Bonn"
  "30_3527|Potsdam"
  "30_228|Geelong"
  "30_9136|Ongata_Rongai"
  "30_5715|Vasai-Virar"
  "30_1676|Changping"
  "30_3778|Kafr_Al_Dayr"
  "30_955|Ribeirao_Pires"
  "30_12979|Somerset_West"
)

infer () {
  local smod=$1 token=$2 run=$3 tag=$4 suffix=$5; shift 5
  local out="$OUT/${run}_${tag}-prediction_${token}${suffix}.tif"
  local log="$LOGS/infer_${run}_${token}${suffix}_20260716.log"
  if [ -f "$out" ]; then echo "--- skip (exists): $(basename "$out")"; return 0; fi
  echo "--- $(date -Is) $run -> $token$suffix"
  python src/infer_roi.py \
    --smod-id "$smod" --bounds-csv data/guppd_bounds.csv --year 2017 \
    --city-name "${token//_/ }" \
    --output "$out" "$@" \
    > "$log" 2>&1
  echo "    exit $? ($(basename "$out"))"
}

for entry in "${CITIES[@]}"; do
  smod=${entry%%|*}; token=${entry##*|}
  echo "=== $(date -Is) CITY $token ($smod) ==="

  # tessera classification (student-noisy-v3), 320m + 160m
  T=(--model-type resnet --preset small
     --checkpoint "$DL/student-noisy-v3/resnet_small_GeoTessera_v1.1_global_global-best.pt"
     --embedding-name tesserav1.1_global --embedding-dir /tessera/v1.1)
  infer "$smod" "$token" student-noisy-v3 resnet-small-classification "" "${T[@]}"
  infer "$smod" "$token" student-noisy-v3 resnet-small-classification "_160m" "${T[@]}" --patch-physical-stride 160

  # coop classification (opt3-coop-seed1), 320m + 160m
  C=(--model-type resnet --preset small
     --checkpoint "$DL/opt3-coop-seed1/resnet_small_AlphaEarthCoop_global-best.pt"
     --embedding-name alpha_earth_coop --embedding-dir "$D/input/Google/AlphaEarth/coop")
  infer "$smod" "$token" opt3-coop-seed1 resnet-small-classification "" "${C[@]}"
  infer "$smod" "$token" opt3-coop-seed1 resnet-small-classification "_160m" "${C[@]}" --patch-physical-stride 160

  # seamless classification (student-seamless-v1), 320m + 160m
  S=(--model-type resnet --preset small
     --checkpoint "$DL/student-seamless-v1/resnet_small_EmbeddedSeamless_global-best.pt"
     --embedding-name seamless --embedding-dir "$D/input/EmbeddedSeamlessData/2017")
  infer "$smod" "$token" student-seamless-v1 resnet-small-classification "" "${S[@]}"
  infer "$smod" "$token" student-seamless-v1 resnet-small-classification "_160m" "${S[@]}" --patch-physical-stride 160

  # tessera segmentation UNets, 10m
  for seg in seg-distill-unet-large-v1 seg-distill-unet-large-v2-origsplit; do
    infer "$smod" "$token" "$seg" unet-large-segmentation "" \
      --model-type unet --preset large \
      --checkpoint "$DL/$seg/"*-best.pt \
      --embedding-name tesserav1.1_global --embedding-dir /tessera/v1.1
  done
done

echo "=== $(date -Is) city-prediction chain complete ==="
