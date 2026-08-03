"""Training harness for block-based LCZ pseudo-labels.

Consumes ONLY the ``lcz_labels`` Stage 8 export contract — the
``lcz_bitmask/confidence/block_id`` 10 m rasters, ``blocks_labelled_*.parquet``
and the block adjacency — plus the repo's embedding tiles (via ``src/``,
untouched). Two formulations share one label contract:

* **A (dense)**: per-pixel heads over embedding windows, masked marginalised
  cross-entropy on eroded block rasters.
* **B (block)**: block-as-sample classification on pooled embeddings, with an
  optional block-adjacency GNN (torch_geometric, optional ``gnn`` extra).

Both are evaluated at block level on city-held-out, region-stratified splits.
Foundation-model embeddings are frozen throughout; every model is a light head.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: <repo>/lcz_train/__init__.py and <repo>/src/...
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["config", "splits", "mosaics", "datasets", "losses", "models",
           "train", "eval", "postprocess", "experiments", "cli"]
