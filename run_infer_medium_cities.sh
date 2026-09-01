#!/usr/bin/env bash
# Full-ROI inference maps for the medium-preset seg checkpoints (rows 4/5),
# same three cities as the small-preset batch: Nairobi, Cairo, London.
# Standard naming/output-dir convention (run_dir + prediction tif/png),
# --normalize auto reproduces the checkpoint's training-time stats exactly.
#
# --bounds-csv is the 51-city So2Sat file, NOT the generic guppd_bounds.csv —
# that one has two "London" rows (Canada + UK) and would silently pick the
# wrong bbox.
set -uo pipefail
cd "$(dirname "$0")"
source /maps/acz25/envs/eo_fm-env/bin/activate
source .env
D=${DATA_DIR:-/maps/acz25/phd-thesis-data}
BOUNDS=data/so2sat_guppd_bounds.csv
CITIES="Nairobi Cairo London"

run_one () {
  local run_dir=$1 preset=$2 ckpt=$3 emb_name=$4 emb_dir=$5 city=$6
  local out="$run_dir/$(basename $run_dir)_unet-${preset}-segmentation-prediction_${city}.tif"
  echo "=== $(date -Is) starting $(basename $run_dir) / $city ==="
  python src/infer_roi.py \
    --model-type unet --preset "$preset" --checkpoint "$ckpt" \
    --embedding-name "$emb_name" --embedding-dir "$emb_dir" --year 2017 \
    --city "$city" --bounds-csv "$BOUNDS" --city-name "$city" \
    --normalize auto --output "$out"
  echo "=== $(date -Is) finished $(basename $run_dir) / $city (exit $?) ==="
}

# Row 4 medium (coop) — fast, small tile counts.
for city in $CITIES; do
  run_one "$D/output/lcz-segmentation/dl/seg-row4-coop-global-medium" medium \
    "$D/output/lcz-segmentation/dl/seg-row4-coop-global-medium/unet_medium_AlphaEarthCoop_Amsterdam_Beijing_Berlin-best.pt" \
    alpha_earth_coop "$D/input/Google/AlphaEarth/coop" "$city"
done

# Row 5 medium (Tessera) — London alone intersects ~340 small archive tiles
# over a slow mount; this is the long pole (~9.5h last time), which is why
# the whole batch runs in tmux.
for city in $CITIES; do
  run_one "$D/output/lcz-segmentation/dl/seg-row5-tessera-global-medium" medium \
    "$D/output/lcz-segmentation/dl/seg-row5-tessera-global-medium/unet_medium_GeoTessera_v1.1_global_Amsterdam_Beijing_Berlin-best.pt" \
    tesserav1.1_global /tessera/v1.1 "$city"
done

echo "=== $(date -Is) medium inference batch complete ==="
