"""Registry mapping embedding names to torchgeo dataset classes and metadata."""

from pathlib import Path

from torchgeo.datasets import (
    EmbeddedSeamlessData,
    GoogleSatelliteEmbedding,
    TesseraEmbeddings,
)
from torchgeo.datasets.geo import GeoDataset

EMBEDDING_REGISTRY: dict[str, dict] = {
    "tessera": {
        "class": TesseraEmbeddings,
        "in_channels": 128,
        "resolution": 10,
        "description": "Tessera 128-band Sentinel-1/2 embeddings",
        # Zarr fast-path: tiles are named by center coords, 0.1° × 0.1° grid.
        # e.g. grid_0.15_52.05_2024.zarr → center (0.15, 52.05)
        "zarr_tile_size": 0.1,
        "zarr_filename_pattern": r"grid_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)_\d+\.zarr",
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
        "zarr_filename_pattern": r"gse_(?P<lon>[-\d.]+)_(?P<lat>[-\d.]+)_\d+\.zarr",
        "zarr_filename_crs": "EPSG:4326",
        "zarr_filename_is_center": False,
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
