"""Registry of embedding types: metadata used across the pipelines.

``in_channels`` feeds model construction; the ``zarr_*`` filename metadata
drives tile discovery without opening files (datasets.tiles.build_tile_index,
find_tiles_for_roi). CLI ``--embedding-name`` choices are derived from the
registry keys.

``nodata_predicate`` marks per-pixel missing data. Each family stores nodata
differently and none of them uses NaN, so the rule cannot be global — see
:func:`get_nodata_predicate` for the measurements behind each entry.
"""

import re
from pathlib import Path

import numpy as np

EMBEDDING_REGISTRY: dict[str, dict] = {
    "tessera": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera 128-band Sentinel-1/2 embeddings",
        # Zarr fast-path: tiles are named by center coords, 0.1° × 0.1° grid.
        # e.g. grid_0.15_52.05_2024.zarr → center (0.15, 52.05)
        "zarr_tile_size": 0.1,
        "zarr_filename_pattern": r"grid_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)_\d+\.(zarr|tif)",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": True,
    },
    "alpha_earth": {
        "in_channels": 64,
        "resolution": 10,
        "description": "AlphaEarth 64-band satellite embeddings",
        # Zarr fast-path: tiles are named by bottom-left corner, 0.1° × 0.1° grid.
        # e.g. gse_2.2_48.8_2021.zarr → bottom-left (2.2, 48.8)
        "zarr_tile_size": 0.1,
        "zarr_filename_pattern": r"gse_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)_\d+\.(zarr|tif)",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": False,
    },
    "seamless": {
        "in_channels": 72,  # 12 temporal months × 6 VQ levels (band 13 is QA, excluded)
        "resolution": 30,
        "description": "Embedded Seamless Data (ESD) quantized embeddings, dequantized to 72 channels",
    },
    "alpha_earth_coop": {
        "in_channels": 64,
        "resolution": 10,
        "description": "AlphaEarth coop GeoTIFF tiles (source.coop), 64-band Int8 quantized",
        "nodata_all_channels_eq": -128.0,
    },
    "tesserav1.1": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v1.1 int8+scales tiles, dequantized to 128-band float32",
        "nodata_all_channels_eq": 0.0,
    },
    "tesserav1.1_global": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v1.1 global tiles (global_0.1_degree_representation + tiff_all), 128-band float32",
        "nodata_all_channels_eq": 0.0,
    },
    "tesserav2": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v2 global tiles (large_student), int8+scales dequantized to 128-band float32",
        "nodata_all_channels_eq": 0.0,
    },
    "osm_evidence": {
        "in_channels": 15,
        "resolution": 10,
        "description": "Binary OSM evidence layers (buildings + height buckets, "
                       "roads, rail, landuse groups, vegetation, water, bare, "
                       "completeness mask) rasterized by build_osm_rasters.py. "
                       "--embedding-dir = .../osm_evidence/tiles",
        # 0.5° tiles named by bottom-left corner: osm_{lon}_{lat}.tif
        "zarr_tile_size": 0.5,
        "zarr_filename_pattern": r"osm_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)\.(zarr|tif)",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": False,
    },
    "aux_struct": {
        "in_channels": 4,
        "resolution": 10,
        "description": "Auxiliary structural bands, normalised to ~[0,1]: GHSL ANBH/50, "
                       "built fraction, non-residential built fraction, ETH canopy height/50. "
                       "--embedding-dir = .../aux_struct/merged_aux (precompute_aux_tiles.py)",
        # 0.5° precomputed tiles named by bottom-left corner: aux_{lon}_{lat}.tif
        "zarr_tile_size": 0.5,
        "zarr_filename_pattern": r"aux_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)\.(zarr|tif)",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": False,
    },
}


def get_nodata_predicate(name: str):
    """Return ``fn(arr) -> (H, W) bool`` marking invalid pixels, or None.

    ``arr`` is the ``(C, H, W)`` array exactly as it comes off disk — before
    ``nan_to_num`` and before any dequantization — so the sentinel is tested in
    the units it is actually stored in.

    Per-family rules, each measured over the So2Sat training split in Phase 0
    (see RESULTS.md, "Incidental findings" and the GATE 0 amendments):

    * ``alpha_earth_coop`` — invalid where all 64 int8 channels equal -128.
      0.41-0.68% of pixels. -128 NEVER occurs in a single channel alone, so a
      per-channel test would be wrong; and these are not NaN, so ``nan_to_num``
      never caught them. Under the (correct) dequantization each becomes a
      vector of L2 norm 8.06 instead of 1.0.
    * ``tesserav1.1`` / ``tesserav1.1_global`` / ``tesserav2`` — invalid where
      all 128 channels are exactly 0 (0.05-0.15% of pixels). Tessera also has
      ~0.9% *per-channel* quantization zeros affecting 61% of pixels, which are
      valid data: testing per-channel here would discard most of the dataset.
    * ``seamless`` — none. Measured 0.0000% all-zero pixels.
    * ``sentinel1`` / ``sentinel2`` — none. The GeoTIFFs set ``nodata=None``,
      carry no NaNs and have no all-zero pixels in any of the three splits, so
      they are treated as fully valid rather than searched for a mask at read
      time.

    NaN in any channel counts as invalid for every family, even though no
    family currently contains any — it is the one rule that is always right.
    """
    sentinel = EMBEDDING_REGISTRY.get(name, {}).get("nodata_all_channels_eq")

    def predicate(arr: np.ndarray) -> np.ndarray:
        invalid = np.isnan(arr).any(axis=0)
        if sentinel is not None:
            invalid |= (arr == sentinel).all(axis=0)
        return invalid

    return predicate


def get_in_channels(name: str) -> int:
    """Return the number of input channels for a given embedding name."""
    if name not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding '{name}'. Choose from: {list(EMBEDDING_REGISTRY)}")
    return EMBEDDING_REGISTRY[name]["in_channels"]


def find_tiles_for_roi(
    directory: str | Path,
    roi: tuple[float, float, float, float],
    embedding_name: str,
) -> list[Path]:
    """Return paths of tiles in *directory* that overlap *roi*.

    Bounds are derived from filenames — no file is opened.

    Args:
        directory: Folder containing ``.zarr`` or ``.tif`` tile files.
        roi: ``(west, south, east, north)`` bounding box in EPSG:4326.
        embedding_name: Key in :data:`EMBEDDING_REGISTRY` that carries
            filename-pattern metadata (``zarr_filename_pattern`` etc.).

    Returns:
        Sorted list of :class:`~pathlib.Path` objects for matching tiles.

    Raises:
        ValueError: If *embedding_name* is unknown or has no filename-pattern
            metadata.
    """
    if embedding_name not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding '{embedding_name}'. Choose from: {list(EMBEDDING_REGISTRY)}")

    meta = EMBEDDING_REGISTRY[embedding_name]
    pattern = meta.get("zarr_filename_pattern")
    tile_size = meta.get("zarr_tile_size")
    is_center = meta.get("zarr_filename_is_center")

    if pattern is None or tile_size is None or is_center is None:
        supported = sorted(
            k for k, m in EMBEDDING_REGISTRY.items() if m.get("zarr_filename_pattern")
        )
        raise ValueError(
            f"Embedding '{embedding_name}' has no filename-pattern metadata. "
            f"Embeddings supporting find_tiles_for_roi: {supported}."
        )

    directory = Path(directory)
    regex = re.compile(pattern)
    half = tile_size / 2

    qminx, qminy, qmaxx, qmaxy = roi
    matching: list[Path] = []

    for path in sorted(directory.glob("*.zarr")) + sorted(directory.glob("*.tif")):
        m = regex.match(path.name)
        if m is None:
            continue
        lon = float(m.group("lon"))
        lat = float(m.group("lat"))
        if is_center:
            tminx, tminy, tmaxx, tmaxy = lon - half, lat - half, lon + half, lat + half
        else:
            tminx, tminy, tmaxx, tmaxy = lon, lat, lon + tile_size, lat + tile_size
        if tmaxx > qminx and tminx < qmaxx and tmaxy > qminy and tminy < qmaxy:
            matching.append(path)

    return sorted(matching)
