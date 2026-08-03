"""Overture/OSM-fused LCZ pseudo-labelling module.

Generates global Local Climate Zone (LCZ) pseudo-labels from Overture Maps
(OSM-fused) vector data, aligned onto the existing So2Sat 320 m embedding patch
grid. Overture/OSM is a *semantic* database; LCZ is *morphology* — so this module
never maps tags to LCZ directly. Instead it extracts geometric/height evidence,
computes Urban Canopy Parameters (UCPs) per patch, and thresholds them into LCZ
classes per Stewart & Oke (2012), attaching confidence and temporal-stability
flags to every record.

LCZ 7 (informal / lightweight low-rise): recovered via a footprint-morphology +
road-topology router (Stage 5b) that emits hard 7, hard 3, or a first-class
**coarse {3,7}** label when the morphology is unreadable — never a guessed hard
class. Coarse labels train under a marginalised loss over ``lcz_set``.

The module reuses repo code under ``src/`` (patch grid, LCZ table, GHSL sampling
formulas); it puts ``src/`` on ``sys.path`` at import time following the repo's
existing convention (scripts self-add their parent dir).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: <repo>/lcz_labels/__init__.py and <repo>/src/...
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["config", "grid", "overture", "heights", "blocks", "ucp", "classify",
           "change_mask", "export", "validate", "cli"]
