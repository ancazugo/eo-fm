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
    },
    "google_satellite": {
        "class": GoogleSatelliteEmbedding,
        "in_channels": 64,
        "resolution": 10,
        "description": "Google Satellite Embedding (AlphaEarth) 64-band",
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


def create_embedding_dataset(name: str, path: str | Path) -> GeoDataset:
    """Create an embedding dataset, auto-detecting Zarr vs GeoTIFF format.

    If the path contains .zarr stores, returns a ZarrGeoDataset.
    Otherwise falls back to the torchgeo built-in class (GeoTIFF).

    Args:
        name: Embedding name from the registry.
        path: Path to the embedding data directory.

    Returns:
        A GeoDataset instance for the embeddings.
    """
    path = Path(path)
    zarr_stores = list(path.glob("*.zarr"))

    if zarr_stores:
        from datasets.zarr_dataset import ZarrGeoDataset

        return ZarrGeoDataset(paths=path)

    embedding_cls = get_embedding_class(name)
    return embedding_cls(paths=str(path))
