"""Stage 1 — configuration for the LCZ pseudo-labelling pipeline.

Everything the pipeline does is driven by :class:`LczLabelConfig`: the pinned
Overture release, the list of AOIs, all classification thresholds (Stage 5), all
confidence thresholds (Stage 6), and raster/path settings. The config is
serialisable to/from YAML and produces a stable ``config_hash`` that is stamped
into every output row and used as the per-stage cache key, so a threshold change
invalidates caches deterministically.

No thresholds are hard-coded anywhere else in the module — the classifier reads
them from the config object it is handed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Mirror src/utils/constants.py: pick up DATA_DIR (and friends) from the repo .env.
load_dotenv()

# Pinned Overture monthly release. Overture only retains ~60 days of releases,
# so this tag will eventually 404 on S3 — override it in the YAML config (or set
# a newer one here) when that happens. See docs.overturemaps.org/release-calendar.
DEFAULT_OVERTURE_RELEASE = "2026-06-17.0"
DEFAULT_LABEL_YEAR = 2025


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "."))


# ── Sub-models ────────────────────────────────────────────────────────────────

class AOI(BaseModel):
    """An area of interest: a name plus a WGS84 bounding box.

    ``bbox`` is (minx, miny, maxx, maxy) in EPSG:4326. If omitted, the AOI is
    resolved from ``data/so2sat_guppd_bounds.csv`` by name (So2Sat cities).
    ``equal_area_crs`` overrides the auto-selected local UTM used for all area
    computations; leave ``None`` to auto-estimate.
    """

    name: str
    bbox: tuple[float, float, float, float] | None = None
    equal_area_crs: str | None = None


class ClassificationThresholds(BaseModel):
    """Stage 5 decision-rule thresholds (Stewart & Oke morphology cut-offs)."""

    # Built-signal gates for the natural branch
    natural_bsf_max: float = 0.05
    natural_ghs_built_s_max: float = 0.10

    # Natural / non-built land-cover fractions
    water_min: float = 0.75          # G
    trees_dense_min: float = 0.75    # A
    trees_scatter_min: float = 0.35  # B lower bound
    trees_scatter_veg_min: float = 0.75   # B: f_lowplants + f_trees
    shrub_min: float = 0.60          # C
    lowplants_min: float = 0.75      # D
    bare_rock_min: float = 0.60      # E (rock)
    paved_infra_min: float = 0.60    # E (paved)
    sand_min: float = 0.60           # F

    # Special built classes (checked before the height x density matrix)
    heavy_industry_lu_min: float = 0.50   # LCZ 10
    heavy_industry_poi_min: int = 1
    large_lowrise_bsf_min: float = 0.25   # LCZ 8
    large_lowrise_h_max: float = 10.0
    large_lowrise_footprint_min: float = 800.0   # m^2 mean footprint
    large_lowrise_type_frac: float = 0.50        # majority of built area

    # Height classes (metres)
    height_high_min: float = 25.0
    height_mid_min: float = 10.0
    tower_count_min: int = 3         # h_max>=high with >=N tower-type buildings

    # Density classes (building surface fraction)
    bsf_compact_min: float = 0.40
    bsf_open_min: float = 0.20
    bsf_sparse_min: float = 0.05

    # LCZ 9 (sparsely built) vegetation gate
    lcz9_veg_min: float = 0.40       # f_lowplants + f_trees


class ConfidenceThresholds(BaseModel):
    """Stage 6 confidence / purity filter parameters."""

    height_evidence_full: float = 0.50   # >= this -> full height confidence
    raster_only_cap: float = 0.60        # cap when heights are raster-tier only
    height_none_max_area: float = 0.50   # if "none"-tier over > this area -> unlabelled

    completeness_ghs_min: float = 0.15   # bsf<natural_bsf_max but ghs>this -> under-mapped
    boundary_band: float = 0.03          # +/- bsf band around a density threshold
    boundary_penalty: float = 0.70

    suspect_ml_ghs_max: float = 0.02     # GHS-BUILT-S ~ 0
    suspect_ml_penalty: float = 0.50

    min_emit: float = 0.0                # confidence floor below which -> unlabelled


class Lcz7RouterThresholds(BaseModel):
    """Stage 5b — compact-lowrise router (LCZ 3 vs 7 vs coarse {3,7}).

    Informal fabric (LCZ 7) shows tiny footprints, extreme building-count
    density, high footprint-size irregularity, and a mismatch between dense
    buildings and sparse mapped roads. These thresholds turn those morphology
    signals into routing booleans. The safe direction on ambiguity is always
    toward the coarse {3,7} label, never a wrong hard 3.
    """

    # informal_morphology
    median_footprint_informal_max: float = 60.0     # m^2 median single footprint
    count_density_informal_min: float = 6000.0      # buildings / km^2
    footprint_cv_informal_min: float = 0.7          # std/mean of footprint areas

    # road_deficit
    buildings_per_road_km_informal_min: float = 250.0
    road_density_deficit_max: float = 5.0           # km road / km^2

    # formal_morphology
    median_footprint_formal_min: float = 100.0
    formal_height_evidence_min: float = 0.3

    # mn_informal (Million Neighborhoods corroboration)
    mn_informal_frac_min: float = 0.5

    # blob_suspect — dense fabric merged into few blobs by non-Google ML footprints
    blob_google_frac_max: float = 0.5
    blob_count_density_max: float = 2000.0
    blob_bsf_min: float = 0.4

    # LCZ 7 confidence caps + diagnostics
    lcz7_confidence_cap: float = 0.60
    lcz7_confidence_cap_mn: float = 0.75            # when MN corroborates morphology
    lcz7_strong_bsf: float = 0.60                   # Stewart&Oke 7 lower BSF (diagnostic only)

    # buildings_per_road_km sentinel when roads=0 but buildings>0
    buildings_per_road_km_sentinel: float = 1.0e4


class RasterPaths(BaseModel):
    """Locations of the auxiliary rasters (Stage 3/7).

    Defaults point at the repo's existing ``aux_struct`` tile tree
    (``{dir}/{prefix}_{lon}_{lat}.tif`` on a 0.5-degree grid, matching
    ``src/extract_aux_features.py``). ``google_temporal_dir`` is optional: when
    absent, Stage 7 degrades gracefully (every patch ``stable=False``).
    """

    ghs_built_h_dir: Path = Field(default_factory=lambda: _data_dir() / "input/aux_struct/ghs_built_h")
    ghs_built_s_dir: Path = Field(default_factory=lambda: _data_dir() / "input/aux_struct/ghs_built_s")
    google_temporal_dir: Path | None = None
    # Optional Million Neighborhoods (Mansueto) block layer — informality/low-access
    # scores. When None/absent, the router degrades gracefully (mn_informal_frac=NaN).
    million_neighborhoods_path: Path | None = None
    tile_size_deg: float = 0.5


# ── Top-level config ──────────────────────────────────────────────────────────

class LczLabelConfig(BaseModel):
    """Full pipeline configuration.

    Defaults are chosen so that a bare ``LczLabelConfig()`` runs on any So2Sat
    city given the repo's standard ``DATA_DIR`` layout.
    """

    overture_release: str = DEFAULT_OVERTURE_RELEASE
    label_year: int = DEFAULT_LABEL_YEAR
    change_baseline_year: int = 2017     # embeddings epoch to test stability against
    patch_size_m: float = 320.0

    aoi_list: list[AOI] = Field(default_factory=list)

    # Repo integration paths
    so2sat_dir: Path = Field(default_factory=lambda: _data_dir() / "input/So2Sat-LCZ42/v4")
    cities_subdir: str = "cities"
    city_bounds_csv: Path = Path("data/so2sat_guppd_bounds.csv")
    cache_dir: Path = Field(default_factory=lambda: _data_dir() / "output/lcz_labels")

    rasters: RasterPaths = Field(default_factory=RasterPaths)
    classification: ClassificationThresholds = Field(default_factory=ClassificationThresholds)
    confidence: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    router: Lcz7RouterThresholds = Field(default_factory=Lcz7RouterThresholds)

    # Height model
    metres_per_floor: float = 3.2
    height_min_sane: float = 2.0
    height_max_sane: float = 1000.0

    # Overture source datasets treated as trustworthy for heights/semantics
    trusted_source_datasets: list[str] = Field(
        default_factory=lambda: ["OpenStreetMap", "Esri Community Maps"]
    )
    # Google Open Buildings source string (used for f_google_source / blob_suspect)
    google_source_dataset: str = "Google Open Buildings"

    # transportation/segment road classes: motorized count toward road density;
    # non-motorized are excluded (kept only as a separate count).
    road_motorized_classes: list[str] = Field(default_factory=lambda: [
        "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
        "residential", "living_street", "service", "road", "track", "unknown",
    ])
    road_nonmotorized_classes: list[str] = Field(default_factory=lambda: [
        "footway", "path", "steps", "cycleway", "pedestrian", "bridleway", "corridor",
    ])

    model_config = {"extra": "forbid"}

    # ── hashing / serialisation ──────────────────────────────────────────────

    def _canonical(self) -> dict:
        """JSON-canonical dict for hashing (paths -> str, deterministic order)."""
        return json.loads(self.model_dump_json())

    @property
    def config_hash(self) -> str:
        """Stable 12-char hash of the full config (stamped into every output)."""
        blob = json.dumps(self._canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self._canonical(), fh, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LczLabelConfig":
        with Path(path).open() as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def cities_dir(self) -> Path:
        return self.so2sat_dir / self.cities_subdir

    def aoi(self, name: str) -> AOI:
        for a in self.aoi_list:
            if a.name == name:
                return a
        # Not explicitly configured: treat the name as a So2Sat city (bbox
        # resolved later from the bounds CSV in grid.py).
        return AOI(name=name)
