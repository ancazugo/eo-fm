"""Registry of embedding types: metadata used across the pipelines.

``in_channels`` feeds model construction; the ``zarr_*`` filename metadata
drives tile discovery without opening files (datasets.tiles.build_tile_index,
find_tiles_for_roi). CLI ``--embedding-name`` choices are derived from the
registry keys.
"""

import re
from pathlib import Path

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
    },
    "tesserav1.1": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v1.1 int8+scales tiles, dequantized to 128-band float32",
    },
    "tesserav1.1_global": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v1.1 global tiles (global_0.1_degree_representation + tiff_all), 128-band float32",
    },
    "tesserav2": {
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera v2 global tiles (large_student), int8+scales dequantized to 128-band float32",
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
