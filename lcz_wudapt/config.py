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
