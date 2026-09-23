"""Configuration for the WUDAPT / LCZ-Generator label source.

Mirrors :mod:`lcz_labels.config` conventions exactly — pydantic, ``extra="forbid"``,
YAML round-trip, and a stable 12-char ``config_hash`` stamped into every output
row and used as the per-stage cache key.

Two coupling decisions are deliberate and load-bearing:

* :class:`LczLabelConfig` is held as a **field**, not a base class, so
  ``lcz_labels.export.raster_grid`` produces a provably identical canonical grid
  for both label sources. Forking the grid definition is the one invariant that
  must not fork.
* ``cache_dir`` defaults somewhere *different* from ``lcz_labels``'. Both
  packages write ``{cache_dir}/{aoi}/blocks_labelled_{aoi}.parquet`` and would
  silently collide on the 50 city names they share.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from lcz_labels.config import LczLabelConfig

# Mirror lcz_labels.config / src.utils.constants: DATA_DIR comes from the repo .env.
load_dotenv()

# The LCZ-Generator submission database release this config is pinned to. The
# filename carries the export date; repinning changes polygon counts, so it is
# part of the config hash rather than a bare default buried in the loader.
DEFAULT_WUDAPT_RELEASE = "2024-10-01"

# Layer name inside the gpkg. The file also carries a `layer_styles` layer that
# pyogrio will warn about and that must never be read as data.
DEFAULT_WUDAPT_LAYER = "LCZ-Generator_training_areas_2024-10-01"


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "."))


# ── Sub-models ────────────────────────────────────────────────────────────────

class QualityGates(BaseModel):
    """Pre-harmonisation cleaning gates (H2).

    Defaults here are deliberately permissive placeholders. The real operating
    point is *calibrated* by ``lcz_wudapt audit`` against So2Sat agreement (gate
    G0.4) — the whole point of Stage 0 is that these numbers come from the data
    rather than from priors.
    """

    # Submission-level
    require_qc_step1: bool = True
    require_qc_step2: bool = False
    require_qc_step3: bool = False
    min_oa: float = 0.0
    min_oau: float = 0.0          # applied only to built classes 1-10
    min_class_f1: float = 0.0     # per-class f1_{c} for the class each polygon carries

    # Polygon-level. 5% of polygons are < 0.0012 km2 — a dozen 10 m pixels, which
    # carries no LCZ information at any embedding resolution we use.
    min_area_km2: float = 0.0012
    max_area_km2: float = 100.0
    min_vertices: int = 4

    model_config = {"extra": "forbid"}


class ConsensusParams(BaseModel):
    """Multi-annotator consensus parameters (H3)."""

    # Decision thresholds: accumulate classes by descending posterior until
    # tau_mass is reached; |S| == 1 -> hard, 2..max_set_size -> coarse, else drop.
    tau_mass: float = 0.65
    max_set_size: int = 3
    min_p_top: float = 0.35

    # Dirichlet prior strength, in units of ONE AVERAGE ANNOTATOR of this AOI
    # (alpha = prior_alpha * mean positive polygon weight). An absolute alpha is
    # wrong: real annotator weights average ~0.3-0.6, so a fixed alpha = 1.0 is a
    # virtual annotator that outvotes every real one, which drove the hard-label
    # fraction to 0.6% in Nairobi — a city that agrees with So2Sat at 0.77.
    # Scale-free means the prior stays "one chance-level opinion" whatever the
    # weight model does.
    prior_alpha: float = 1.0

    # Weight model. w = w_qc * w_acc * w_time * w_size.
    qc_fail_soft_penalty: float = 0.6     # qc_step2/3 False (step1 False is fatal)
    acc_floor: float = 0.05
    acc_oa_min: float = 0.40              # oa at/below this maps to acc_floor
    acc_oa_span: float = 0.50
    time_decay_years: float = 3.0         # exp(-|rep_year - target_year| / this)

    # Consensus depth -> confidence. depth = clip(base + per_annotator * n_eff, 0, 1),
    # deliberately CONTINUOUS in n_eff: a branch at n_eff == 1 (lone annotator
    # scored differently from a hair above one) is a cliff that float32 vs float64
    # rounding flips, and it made confidence depend on arithmetic precision.
    # A lone annotator is already penalised by the Dirichlet prior, which pulls
    # p_top toward the class prior when their weight is small, so depth does not
    # need a separate single-annotator reliability term.
    depth_base: float = 0.40              # n_eff = 1 -> 0.55
    depth_per_annotator: float = 0.15     # n_eff >= 4 -> 1.0
    deep_n_eff: float = 3.0               # block_kind deep/shallow boundary
    small_region_penalty: float = 0.75    # region smaller than one So2Sat patch

    model_config = {"extra": "forbid"}


class QcRules(BaseModel):
    """LCZ-Generator / WUDAPT training-area quality control, as published.

    Every default is the value stated in the sources, not a guess:

    * ``min_area_km2`` / ``max_shape`` — Generator QC step 1. Verified against
      the 2024-10-01 release: ``qc_step1 == True`` is *exactly*
      ``area >= 0.04 AND shape < 3`` with **zero exceptions in 630,311 rows**,
      so the rule is reconstructible and can be re-derived at our thresholds
      instead of trusting a flag computed on Web-Mercator area.
      ``shape = perimeter**2 / (4*pi*area)`` — 1.0 for a circle, 1.273 for a
      square. Note ``qc_step1 == True`` means PASSED, not "flagged".
    * ``min_oa`` — the 0.50 floor of Bechtel et al. (2019a), applied by both the
      Generator and the ESSD global map.
    * ``neighbour_buffer_m``, ``min/max_polys_per_class`` — the WUDAPT
      digitizing guide ("leave a buffer of > 100 m between LCZs", "several
      examples (5-15) of each LCZ").
    * ``oversize_km2`` / ``oversize_core_radius_m`` — the Generator reduces
      polygons > 1.5 km2 to a ~350 m radius before classifying.
    """

    # ── Generator QC step 1 ──────────────────────────────────────────────────
    min_area_km2: float = 0.04
    max_shape: float = 3.0

    # ── Submission-level accuracy gates ──────────────────────────────────────
    # oau is used for built classes 1-10 and oa for natural 11-17, mirroring how
    # the Generator reports them. Only min_oa is on by default; the other two are
    # off so the operating point is chosen by measurement, not by prior.
    min_oa: float = 0.50
    min_oau: float = 0.0
    min_class_f1: float = 0.0

    # ── So2Sat geometry target ───────────────────────────────────────────────
    # A So2Sat label is a 320 m square of one pure class. Requiring a polygon to
    # *contain* one is the >200 m narrowest-width rule retargeted to 320 m, and
    # it is far stricter than an area test: 39.8% of QC-passing polygons pass it
    # against 61.4% passing step 1, and containment is class-biased (LCZ 1 22.4%,
    # water 55.6%) which is what drove water to 42% of an earlier patch pool.
    patch_size_m: float = 320.0

    # ── Relationship to other labels ─────────────────────────────────────────
    # Measured per city, the fraction of candidate patches NOT within 100 m of a
    # differently-labelled polygon tracks So2Sat agreement better than any
    # shipped accuracy column (Tehran 0.19, Delhi 0.23 vs Berlin 0.80,
    # Bogota 0.97). It is therefore emitted as a swept column; the default keeps
    # every patch and lets the sweep choose. Hard-thresholding here at 100 m
    # would delete most of Tehran and Delhi outright.
    neighbour_buffer_m: float = 100.0
    drop_neighbour_conflicts: bool = False

    # Raster-side erosion before burning the segmentation labels. NOT
    # neighbour_buffer_m: the median polygon is 0.048 km2, about 219 m across, so
    # eroding 100 m from every side would leave 19 m and delete most of the
    # dataset. The >100 m inter-LCZ rule is honoured instead by writing nodata
    # wherever two classes claim the same pixel, plus the soft nbr_dist_m weight.
    # 20 m matches the segmentation CLI's existing --erode-px 2 at 10 m.
    # The ESSD duplicate-priority rule ("keep the submission with the highest
    # overall accuracy") resolves every different-class overlap by a TOTAL rank
    # order, so exactly one member of each contested pair survives. That is
    # faithful to the global-map paper, but it is aggressive -- it removes 59% of
    # Tehran -- so it is switchable for the sweep. With it off, contested pixels
    # fall through to the raster's nodata rule instead.
    use_conflict_priority: bool = True

    raster_erode_m: float = 20.0
    raster_res_m: float = 10.0
    raster_max_pixels: int = 120_000_000

    # ── Per-class / per-polygon caps (WUDAPT 5-15 guidance) ──────────────────
    # The WUDAPT "5-15 examples per class" guidance is advice to *annotators*
    # about how many training areas to draw, not a cap on how much supervision a
    # model may see. Capping patches at 15 per class per AOI would discard most
    # of Beijing and Sao Paulo for no methodological reason, so the cap is OFF by
    # default and class balance is left to the trainers' existing
    # --class-weights / --sampler. min_polys_per_class is used to *flag*
    # under-represented AOI-class cells, not to drop them.
    min_polys_per_class: int = 5
    max_patches_per_class: int | None = None

    # This one IS load-bearing. A handful of huge lakes and forests supplied 42%
    # of an earlier patch pool; capping patches per source polygon is what breaks
    # that, and it is the same intent as the Generator's own >1.5 km2 reduction.
    max_patches_per_polygon: int = 4

    # ── Oversize reduction ───────────────────────────────────────────────────
    oversize_km2: float = 1.5
    oversize_core_radius_m: float = 350.0

    # ── Generator QC steps 2/3, adapted to embedding space ───────────────────
    # The Generator runs DBSCAN over 33 Landsat/Sentinel features; we have better
    # features, so the same test runs over mean embeddings. Optional: it needs
    # embeddings on disk, so it is off until the roster is resolved.
    dbscan_eps: float = 0.3
    dbscan_minpts_divisor: int = 10
    use_embedding_outliers: bool = False

    # ── Soft temporal weight ─────────────────────────────────────────────────
    # w_time = exp(-|label_year - embedding_year| / time_decay_years).
    #
    # tau = 8.0, NOT the 3.0 used by the superseded consensus path. Measured over
    # all 630,311 polygons against a fixed 2017: median lag is 4 years and only
    # 2.0% are from 2017 itself, so tau=3 puts 80.1% of the corpus below weight
    # 0.5 — a hard filter wearing a soft filter's clothes. tau=8 leaves 70.4%
    # above 0.5. LCZs change slowly; age should tilt the weighting, not gut it.
    #
    # Set to None to disable temporal weighting entirely (the tau = inf arm).
    time_decay_years: float | None = 8.0

    # Embedding years available to match against. The residual lag after
    # year-matching is what w_time actually penalises: with coop's {2017, 2025}
    # it never exceeds 4 years, against a median of 4 and a max of 27 versus a
    # fixed 2017. Year-matching is the primary correction; the weight is the
    # remainder.
    embedding_years: tuple[int, ...] = (2017, 2025)

    # CAUTION when tuning: `oa` declines with recency (2019: 0.768, 2022: 0.638,
    # 2023: 0.615), so w_time and w_acc partly cancel. Fit and report jointly.
    acc_floor: float = 0.05
    acc_oa_min: float = 0.40
    acc_oa_span: float = 0.50

    model_config = {"extra": "forbid"}


class RegionParams(BaseModel):
    """Consensus-region formation (H6) — what plays the role of a `block`."""

    # Matches lcz_labels.config.BlockParams.min_block_area_m2 so the two label
    # sources produce comparably-sized training units.
    min_region_area_m2: float = 2500.0
    connectivity: int = 1                 # scipy.ndimage.label: 1 = 4-connectivity

    model_config = {"extra": "forbid"}


# ── Top-level config ──────────────────────────────────────────────────────────

class WudaptConfig(BaseModel):
    """Full WUDAPT label-source configuration.

    A bare ``WudaptConfig()`` runs against the repo's standard ``DATA_DIR``
    layout with no arguments.
    """

    wudapt_release: str = DEFAULT_WUDAPT_RELEASE
    gpkg_path: Path = Field(
        default_factory=lambda: _data_dir()
        / f"input/WUDAPT/LCZ-Generator_training_areas_{DEFAULT_WUDAPT_RELEASE}.gpkg"
    )
    layer: str = DEFAULT_WUDAPT_LAYER

    # Global GUPPD urban-area table (5,558 rows). Note JRC_NAME_MAIN is NOT
    # unique — every AOI key is SMOD_ID-qualified, see ingest.aoi_key.
    bounds_csv: Path = Path("data/guppd_bounds.csv")

    # MUST differ from LczLabelConfig.cache_dir: both write
    # {cache_dir}/{aoi}/blocks_labelled_{aoi}.parquet.
    cache_dir: Path = Field(default_factory=lambda: _data_dir() / "output/lcz_wudapt")

    # The Overture-path config, held as a field so raster_grid() is shared.
    labels: LczLabelConfig = Field(default_factory=LczLabelConfig)

    qc: QcRules = Field(default_factory=QcRules)
    quality: QualityGates = Field(default_factory=QualityGates)
    consensus: ConsensusParams = Field(default_factory=ConsensusParams)
    regions: RegionParams = Field(default_factory=RegionParams)

    # Blank-name policy. 106,793 polygons (16.8%) ship with empty firstname AND
    # lastname. Keying each on its own submission invents 1,764 pseudo-annotators
    # — more than the 1,491 real named authors, and in Wuhan 92 against 40 — which
    # inflates n_eff and therefore inflates confidence exactly where consensus is
    # least trustworthy. "collapse_per_aoi" instead treats all blank-name rows in
    # one AOI as a single annotator: it may merge distinct people, but understating
    # consensus only costs coverage, whereas overstating it corrupts labels.
    # The audit reports G0 under both settings.
    blank_name_policy: Literal["collapse_per_aoi", "per_submission"] = "collapse_per_aoi"

    # Label epochs outside this range are typos (2029, 2117, 2323 all appear) and
    # are treated as missing rather than clamped.
    min_label_year: int = 1990
    max_label_year: int = 2025

    # So2Sat sparsity admission rule (H5). A So2Sat city admits WUDAPT only when
    # its own labels are too sparse or too degenerate to supervise anything.
    # Measured from patches_reference_rxr.gpkg, this selects the 11 single-class
    # cities (Salvador 1 patch ... Karachi 1140) and provably cannot select any
    # of the 10 held-out test cities, which all have >= 4798 patches and >= 13
    # classes. leakage.assert_no_test_city_admitted() enforces that in code.
    sparse_city_max_patches: int = 2000
    sparse_city_max_classes: int = 8

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

    @property
    def ingest_hash(self) -> str:
        """Hash of ONLY the inputs that shape ingestion (H1).

        The cleaned per-AOI parquet keys on this rather than ``config_hash`` so
        that tuning a consensus or quality threshold never re-reads and
        re-validates all 630k polygons.
        """
        blob = json.dumps(
            {
                "release": self.wudapt_release,
                "gpkg": str(self.gpkg_path),
                "layer": self.layer,
                "bounds_csv": str(self.bounds_csv),
                "blank_name_policy": self.blank_name_policy,
                "min_label_year": self.min_label_year,
                "max_label_year": self.max_label_year,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def aoi_dir(self, aoi: str) -> Path:
        """Per-AOI output directory, created on demand."""
        d = self.cache_dir / aoi
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── YAML round-trip ──────────────────────────────────────────────────────

    def to_yaml(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self._canonical(), sort_keys=True))
        return path

    @classmethod
    def from_yaml(cls, path: Path) -> WudaptConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})
