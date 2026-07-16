"""Data layer: embedding metadata, tile access, and torch datasets.

Modules (import them directly, e.g. ``from datasets.tiles import crop_patch``):

    registry     EMBEDDING_REGISTRY metadata per embedding source
                 (in_channels, tile filename patterns) + find_tiles_for_roi
    tiles        raw source-tile access: build_tile_index / open_tile /
                 crop_patch / numpy_mosaic (handles south-up Google tiles,
                 Tessera v1.1 int8+scales, seamless, aux rasters)
    so2sat       per-patch items and PatchDataset/PatchDataModule for the
                 classification pipeline (incl. fusion + pseudo-label items)
    grid_tiles   grid-tile items and GridSegDataset/GridSegDataModule for the
                 segmentation pipeline
    downloaders  GEE / geotessera / source.coop tile downloads

Label conventions: classification LCZ 1-17 → 0-16; segmentation masks
1-17 → 0-16 with nodata 0 → -1 (ignore_index=-1 everywhere).
Dequantization is auto-applied for alpha_earth_coop and seamless via
``utils.runtime.resolve_dequantize`` (seamless expands 13→72 channels).
"""
