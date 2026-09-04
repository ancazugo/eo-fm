"""WUDAPT / LCZ-Generator community training areas as a Stage 8 label source.

Consumes ``LCZ-Generator_training_areas_*.gpkg`` (the community submission
database) and emits the same export contract as :mod:`lcz_labels` — uint32 LCZ
bitmask + uint8 confidence + uint32 block-id rasters plus a ``blocks_labelled``
parquet — so :mod:`lcz_train` consumes community labels with no change to its
loss or dataset layer.

The pipeline is deliberately split so the cheap, falsifiable measurements run
before any raster is written::

    ingest  -> clean + spatial AOI assignment      (lcz_wudapt.ingest)
    quality -> author collapse + submission gates  (lcz_wudapt.quality)
    audit   -> the G0 gate measurements            (lcz_wudapt.audit)

Stage 0 (the three modules above) needs no embeddings and no GPU.
"""

from __future__ import annotations

__all__ = ["WudaptConfig"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name == "WudaptConfig":
        from .config import WudaptConfig

        return WudaptConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
