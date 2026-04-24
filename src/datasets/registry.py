"""Registry mapping embedding names to torchgeo dataset classes and metadata."""

import re
from pathlib import Path

from torchgeo.datasets import (
    EmbeddedSeamlessData,
    GoogleSatelliteEmbedding,
    TesseraEmbeddings,
)
from torchgeo.datasets.geo import GeoDataset, RasterDataset


class GeoTiffEmbedding(RasterDataset):
    """Generic multi-band GeoTIFF embedding dataset (any .tif/.tiff file)."""
    filename_glob = "*.tif*"
    is_image = True


EMBEDDING_REGISTRY: dict[str, dict] = {
    "tessera": {
        "class": TesseraEmbeddings,
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
        "class": GoogleSatelliteEmbedding,
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
    "tessera_tiff": {
        "class": GeoTiffEmbedding,
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera 128-band GeoTIFF mosaic embedding",
    },
    "seamless": {
        "class": EmbeddedSeamlessData,
        "in_channels": 128,
        "resolution": 30,
        "description": "Embedded Seamless Data 128-band quantized embeddings",
    },
}


def get_embedding_class(name: str):
    """Return the torchgeo dataset class for a given embedding name."""
    if name not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding '{name}'. Choose from: {list(EMBEDDING_REGISTRY)}")
    return EMBEDDING_REGISTRY[name]["class"]


def get_in_channels(name: str) -> int:
    """Return the number of input channels for a given embedding name."""
    if name not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding '{name}'. Choose from: {list(EMBEDDING_REGISTRY)}")
    return EMBEDDING_REGISTRY[name]["in_channels"]


def create_embedding_dataset(
    name: str,
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
) -> GeoDataset:
    """Create an embedding dataset, auto-detecting Zarr vs GeoTIFF format.

    If the path contains .zarr stores, returns a ZarrGeoDataset.
    Otherwise falls back to the torchgeo built-in class (GeoTIFF).

    Args:
        name: Embedding name from the registry.
        path: Path to the embedding data directory.
        bbox: Optional ``(west, south, east, north)`` bounding box in EPSG:4326.
            Passed to ZarrGeoDataset so CRS detection uses a tile from the
            correct region (important when the directory spans multiple UTM zones).

    Returns:
        A GeoDataset instance for the embeddings.
    """
    path = Path(path)
    zarr_stores = list(path.glob("*.zarr"))

    if zarr_stores:
        from datasets.zarr_dataset import ZarrGeoDataset

        meta = EMBEDDING_REGISTRY.get(name, {})
        return ZarrGeoDataset(
            paths=path,
            tile_size=meta.get("zarr_tile_size"),
            filename_pattern=meta.get("zarr_filename_pattern"),
            filename_crs=meta.get("zarr_filename_crs"),
            filename_is_center=meta.get("zarr_filename_is_center", False),
            bbox=bbox,
        )

    embedding_cls = get_embedding_class(name)
    return embedding_cls(paths=str(path))


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
        embedding_name: Key in :data:`EMBEDDING_REGISTRY` with filename
            metadata (``"tessera"`` or ``"alpha_earth"``).

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
        raise ValueError(
            f"Embedding '{embedding_name}' has no filename-pattern metadata. "
            "Only 'tessera' and 'alpha_earth' support find_tiles_for_roi."
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
