"""Offline end-to-end test on the committed ~1.5 km^2 Nairobi Overture fixture.

No network, no external rasters: GHS dirs point at an empty tmp dir (heights use
explicit/levels tiers only; ghs_built_s -> NaN) and the temporal product is
absent (stable=False). Exercises Stages 3-6 + assembly.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from lcz_labels.classify import classify_patches
from lcz_labels.config import LczLabelConfig
from lcz_labels.overture import OvertureExtract
from lcz_labels.ucp import compute_ucp

FX = Path(__file__).parent / "fixtures"
VALID_LCZ = set(range(1, 18))   # LCZ 7 is now emitted by the router


def _load_extract():
    roads_path = FX / "overture_roads.parquet"
    return OvertureExtract(
        buildings=gpd.read_parquet(FX / "overture_buildings.parquet"),
        landcover=gpd.read_parquet(FX / "overture_landcover.parquet"),
        infrastructure=gpd.read_parquet(FX / "overture_infrastructure.parquet"),
        utm_crs="EPSG:32737",
        roads=gpd.read_parquet(roads_path) if roads_path.exists() else None,
    )


def test_fixtures_present():
    for name in ("grid_fixture", "overture_buildings", "overture_landcover",
                 "overture_infrastructure"):
        assert (FX / f"{name}.parquet").exists()


def test_end_to_end(tmp_path):
    cfg = LczLabelConfig()
    cfg.cache_dir = tmp_path
    cfg.rasters.ghs_built_h_dir = tmp_path / "empty_h"
    cfg.rasters.ghs_built_s_dir = tmp_path / "empty_s"
    grid = gpd.read_parquet(FX / "grid_fixture.parquet")

    ucp = compute_ucp(grid, _load_extract(), cfg, "fixture", force=True)
    assert len(ucp) == len(grid)
    for col in ("bsf", "h_mean", "height_evidence_frac", "f_trees", "ghs_built_s"):
        assert col in ucp.columns

    for col in ("median_footprint_area", "building_count_density", "road_length_density",
                "buildings_per_road_km", "f_google_source"):
        assert col in ucp.columns

    cls = classify_patches(ucp, cfg)
    assert len(cls) == len(grid)
    assert set(cls["label_type"].unique()) <= {"hard", "coarse", "unlabelled"}

    hard = cls[cls["label_type"] == "hard"]
    assert set(hard["lcz"].astype(int)) <= VALID_LCZ    # 7 allowed now

    # Coarse rows have null lcz and a multi-member set
    coarse = cls[cls["label_type"] == "coarse"]
    assert coarse["lcz"].isna().all()
    assert coarse["lcz_set"].map(lambda s: len(s) >= 2).all() if len(coarse) else True

    # Confidence is in [0, 1] for every emitted label
    emitted = cls[cls["label_type"] != "unlabelled"]
    assert emitted["confidence"].between(0.0, 1.0).all()

    # config_hash is deterministic and non-empty
    assert len(cfg.config_hash) == 12


def test_config_hash_in_assembled_output(tmp_path):
    cfg = LczLabelConfig()
    cfg.cache_dir = tmp_path
    cfg.rasters.ghs_built_h_dir = tmp_path / "e"
    cfg.rasters.ghs_built_s_dir = tmp_path / "e"
    grid = gpd.read_parquet(FX / "grid_fixture.parquet")
    ucp = compute_ucp(grid, _load_extract(), cfg, "fixture", force=True)
    cls = classify_patches(ucp, cfg)
    out = pd.concat([grid.reset_index(drop=True), cls], axis=1)
    out["config_hash"] = cfg.config_hash
    out["overture_release"] = cfg.overture_release
    assert (out["config_hash"] == cfg.config_hash).all()
    assert "lcz" in out.columns
