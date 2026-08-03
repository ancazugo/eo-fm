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


def test_block_chain_end_to_end(tmp_path):
    """Fixture -> blocks -> ucp -> classify -> zones -> mask -> rasters ->
    patch transfer -> validation. Fully offline."""
    import numpy as np
    import rasterio

    from lcz_labels.blocks import build_adjacency, build_blocks
    from lcz_labels.change_mask import compute_change_mask
    from lcz_labels.classify import classify_blocks, form_zones
    from lcz_labels.config import AOI
    from lcz_labels.export import patch_transfer, write_blocks_parquet, write_rasters
    from lcz_labels.validate import block_homogeneity, validate_patch_transfer

    cfg = LczLabelConfig(cache_dir=tmp_path)
    cfg.rasters.ghs_built_h_dir = tmp_path / "empty_h"
    cfg.rasters.ghs_built_s_dir = tmp_path / "empty_s"
    grid = gpd.read_parquet(FX / "grid_fixture.parquet")
    cfg.aoi_list = [AOI(name="fixture", bbox=tuple(grid.total_bounds))]
    ex = _load_extract()

    blocks = build_blocks("fixture", ex, cfg)
    adjacency = build_adjacency(blocks, cfg, "fixture")
    ucp = compute_ucp(blocks, ex, cfg, "fixture")
    assert "block_id" in ucp.columns and "compactness" in ucp.columns
    assert ucp["elongation"].between(0, 1).all()

    cls = classify_blocks(ucp, cfg)
    cls.insert(0, "block_id", ucp["block_id"].to_numpy())
    zones = form_zones(cls, blocks, adjacency, cfg)
    change = compute_change_mask(blocks, cfg, "fixture")
    assert not change["stable_2017_to_label_year"].any()  # no temporal product

    labels = (blocks.merge(ucp, on="block_id", validate="1:1")
              .merge(cls, on="block_id", validate="1:1")
              .merge(zones, on="block_id", validate="1:1")
              .merge(change, on="block_id", validate="1:1"))
    labels["label_year"] = cfg.label_year
    path = write_blocks_parquet(labels, "fixture", cfg)
    assert path.exists()

    # Acceptance criterion 4: never sparse/natural where GHS says built but
    # Overture is empty (here ghs is NaN everywhere, so just assert the rule
    # never emitted natural labels without positive evidence).
    nat = labels[labels["lcz"].isin(range(11, 18))]
    assert (nat[["f_water", "f_trees", "f_lowplants", "f_shrub", "f_sand",
                 "f_bare_rock", "f_paved_infra"]].max(axis=1) > 0).all() if len(nat) else True

    paths = write_rasters(labels, "fixture", cfg)
    with rasterio.open(paths["bitmask"]) as src:
        bm = src.read(1)
        assert src.res == (cfg.export.raster_res_m, cfg.export.raster_res_m)
    with rasterio.open(paths["block_id"]) as src:
        bi = src.read(1)
    assert (bm != 0).any()                      # some supervision exists
    assert bi.max() <= len(blocks)
    assert np.isin(np.unique(bi), np.r_[0, blocks["block_idx"].to_numpy()]).all()

    patches = patch_transfer(labels, grid, "fixture", cfg)
    assert len(patches) == len(grid)
    frac_cols = [f"f_lcz_{c}" for c in range(1, 18)]
    total = patches[frac_cols].sum(axis=1) + patches["unlabelled_frac"]
    assert ((total > 0.98) & (total < 1.02)).all()
    # The fixture grid is unlabelled (dataset="unlabeled", no LCZ_class) so
    # validation legitimately has nothing to match against.
    assert validate_patch_transfer(patches, "fixture", min_dominant=0.0) == {}

    # Re-run the transfer against a synthetic So2Sat ground truth to exercise
    # the validation + homogeneity path end-to-end.
    grid_gt = grid.copy()
    grid_gt["LCZ_class"] = 1 + (np.arange(len(grid_gt)) % 17)
    patches_gt = patch_transfer(labels, grid_gt, "fixture", cfg)
    result = validate_patch_transfer(patches_gt, "fixture", min_dominant=0.0)
    assert result and result["n_matched"] > 0
    assert "lcz7_audit" in result

    hom = block_homogeneity(labels, grid_gt)
    assert "n_blocks" not in hom or hom["n_blocks"] >= 0  # runs without error


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
