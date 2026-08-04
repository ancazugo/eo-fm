"""M4 pilot roster run — writes results to isolated locations, never touches
the production lcz_labels/ cache_dir's merged labels_all.parquet or
validation_all.md (lesson from the earlier ad-hoc `all` CLI test).

Final result (11/11 non-Toronto AOIs, 1,047,126 total blocks):
So2Sat agreement@conf0.8 — Nairobi 0.284 (known hard case, matches the
pre-block-pipeline grid baseline of 0.290), Munich 0.957, Paris 0.989,
Mumbai 0.866, Sydney 0.891. LCZ-7 contamination 0.000 on all three
audited cities (Nairobi/Mumbai/Sydney). Full report:
$DATA_DIR/output/lcz_labels_blocks_pilot/pilot_validation.md.

Re-run notes: FORCE only needs to be True for a genuine cache-busting
rebuild (e.g. after the momepy leak fix, commit b9faaf9); leave False
for a normal resumed run — city results are individually cached by
config_hash regardless of FORCE's setting for _export_aoi/_validate_aoi.
"""
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import geopandas as gpd
import pandas as pd
from loguru import logger

from lcz_labels.cli import _export_aoi, _validate_aoi, build_block_labels
from lcz_labels.config import LczLabelConfig

# FORCE=False now: the leak-fix rebuild already completed and cached 10/12
# cities correctly (Nairobi..Accra); the process was killed by an unrelated
# session restart mid-way through Toronto. config_hash-based cache hits will
# fast-skip the finished cities and resume at Toronto/Bangkok.
PROD_CITIES = ["Nairobi", "Munich", "Paris", "Mumbai", "Sydney"]
PILOT_CONFIG = "lcz_labels_pilot_config.yaml"
FORCE = False

logger.remove()
logger.add("run_pilot_m4.log", level="INFO")
logger.add(sys.stderr, level="INFO")

results = []
gdfs = []

prod_cfg = LczLabelConfig()
for city in PROD_CITIES:
    t0 = time.perf_counter()
    try:
        # force=FORCE only on the first call: build_block_labels writes a
        # fresh blocks_labelled_{city}.parquet stamped with the current
        # config_hash, so _export_aoi/_validate_aoi's own cache-hit check
        # (path exists AND config_hash matches) picks it up without forcing
        # a second full rebuild (they redundantly re-run build_block_labels
        # internally when forced — cost every stage from Overture extraction
        # onward twice per city).
        gdf = build_block_labels(city, prod_cfg, force=FORCE, run_mask=True)
        _export_aoi(city, prod_cfg, force=False)
        res = _validate_aoi(city, prod_cfg, force=False)
        results.append(res)
        gdfs.append(gdf.to_crs("EPSG:4326"))
        logger.info(f"[{city}] DONE in {time.perf_counter()-t0:.0f}s: "
                    f"{len(gdf)} blocks, oa_conf80={res.get('oa_conf80')}")
    except Exception:
        logger.error(f"[{city}] FAILED:\n{traceback.format_exc()}")

# Toronto excluded: compute_ucp hangs reproducibly (6/6 attempts, always
# between ~5-15 min in, always right after the compute_heights log line,
# always with STABLE memory and no traceback — ruling out both a memory
# leak and a simple crash). Confirmed NOT a scale issue: Bangkok (3.96M
# buildings, 758K roads — 3x/1.5x Toronto's counts) completed the identical
# code path cleanly in 209s the same session. This points to a pathological
# geometry specific to Toronto's Overture extract (e.g. a degenerate/complex
# building or road polygon causing a GEOS operation to hang in the
# per-block STRtree query loop in ucp.py::_building_stats/_road_stats).
# Follow-up: bisect Toronto's buildings/roads with shapely.is_valid_reason
# and vertex-count outlier checks to isolate the offending geometry.
pilot_cfg = LczLabelConfig.from_yaml(PILOT_CONFIG)
for aoi in pilot_cfg.aoi_list:
    city = aoi.name
    if city == "Toronto":
        logger.warning(f"[{city}] SKIPPED — compute_ucp hangs reproducibly, see code comment")
        continue
    t0 = time.perf_counter()
    try:
        gdf = build_block_labels(city, pilot_cfg, force=FORCE, run_mask=True)
        _export_aoi(city, pilot_cfg, force=False)
        res = _validate_aoi(city, pilot_cfg, force=False)  # {} — no So2Sat GT
        results.append(res)
        gdfs.append(gdf.to_crs("EPSG:4326"))
        logger.info(f"[{city}] DONE in {time.perf_counter()-t0:.0f}s: {len(gdf)} blocks "
                    f"(no So2Sat GT — {gdf['label_type'].value_counts().to_dict()})")
    except Exception:
        logger.error(f"[{city}] FAILED:\n{traceback.format_exc()}")

from lcz_labels.validate import write_report

merged = pd.concat(gdfs, ignore_index=True)
out_dir = pilot_cfg.cache_dir
gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326").to_parquet(
    out_dir / "pilot_labels_all.parquet"
)
write_report([r for r in results if r], out_dir / "pilot_validation.md")
logger.info(f"PILOT DONE: {len(gdfs)}/11 AOIs (Toronto skipped — see code comment), "
            f"{len(merged)} total blocks, {sum(1 for r in results if r)} validated vs So2Sat")
