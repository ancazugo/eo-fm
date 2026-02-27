"""Label datasets for the eo-fm pipeline."""

from pathlib import Path
from typing import Any, Callable, Sequence

from torchgeo.datasets import RasterDataset
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox

class LCZLabelDataset(RasterDataset):
    """Local Climate Zone label dataset backed by GeoTIFF files.

    Subclasses torchgeo's RasterDataset with is_image=False so samples
    return a "mask" key instead of "image".
    """

    is_image = False
    filename_glob = "*.tif"

    def __init__(
        self,
        paths: Path | list[Path] = "data",
        crs: Any | None = None,
        res: float | tuple[float, float] | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable | None = None,
        cache: bool = True,
        remap: dict[int, int] | None = None,
    ) -> None:
        """Initialize LCZLabelDataset.

        Args:
            paths: Path(s) to directory containing label GeoTIFFs.
            crs: Coordinate reference system to warp to.
            res: Resolution to resample to.
            bands: Bands to use.
            transforms: Transforms to apply to each sample.
            cache: Whether to cache file handles.
            remap: Optional dict mapping old class values to new class values.
        """
        super().__init__(
            paths=paths, crs=crs, res=res, bands=bands,
            transforms=transforms, cache=cache,
        )
        self.remap = remap

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve a label sample and apply optional class remapping.

        Args:
            query: Spatiotemporal bounding box query.

        Returns:
            Dict with "mask" key containing the (optionally remapped) label tensor.
        """
        sample = super().__getitem__(query)
        if self.remap:
            mask = sample["mask"]
            remapped = mask.clone()
            for old_val, new_val in self.remap.items():
                remapped[mask == old_val] = new_val
            sample["mask"] = remapped
        return sample

    @classmethod
    def from_gee(
        cls,
        bbox: list[float],
        output_dir: str | Path,
        scale: int = 100,
        ee_path: str = "RUB/RUBCLIM/LCZ/global_lcz_map/latest",
        band: str = "LCZ_Filter",
        **kwargs: Any,
    ) -> "LCZLabelDataset":
        """Download LCZ labels from GEE and return a dataset instance.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326.
            output_dir: Directory to save the downloaded GeoTIFFs.
            scale: Resolution in meters for the GEE export.
            ee_path: GEE ImageCollection path.
            band: Band name to select from the image.
            **kwargs: Additional arguments passed to LCZLabelDataset.__init__.
        """
        from datasets.downloaders import download_gee_dataset

        download_gee_dataset(
            bbox=bbox,
            output_dir=output_dir,
            dataset="demuzere_lcz",
            ee_path=ee_path,
            bands=[band],
            scale=scale,
        )

        return cls(paths=output_dir, **kwargs)


class VectorPatchLabelDataset(GeoDataset):
    """Label dataset backed by a GeoPackage (or any fiona-readable vector file).

    Each polygon feature is a labeled spatial patch. For a given bounding box
    query, returns a constant-value mask tensor with the label of the first
    intersecting feature. The mask shape (1, 1, 1) is compatible with
    EmbeddingLabelDataModule's collate_fn for both classification (majority
    vote) and segmentation (broadcast by UNetTask._prepare_mask).

    Uses the torchgeo 0.9 geopandas-backed index (same as ZarrGeoDataset).

    Args:
        path: Path to the GeoPackage (or any fiona-readable vector file).
        label_col: Column name containing integer class labels (1-based; 0 = nodata).
        crs: Target CRS. Pass the embedding dataset's CRS so spatial queries align.
    """

    is_image = False

    def __init__(
        self,
        path: str | Path,
        label_col: str,
        crs: Any | None = None,
    ) -> None:
        import geopandas as gpd
        import pandas as pd
        from datetime import datetime

        super().__init__()
        self.label_col = label_col

        gdf = gpd.read_file(path)
        if crs is not None:
            gdf = gdf.to_crs(crs)
        gdf = gdf.reset_index(drop=True)

        self._res = 0.0  # No native raster resolution; sampler uses embedding's res

        # Build geopandas GeoDataFrame index (torchgeo 0.9 API, matches ZarrGeoDataset)
        mint = datetime(1900, 1, 1)
        maxt = datetime(2100, 12, 31)
        datetimes = [(mint, maxt)] * len(gdf)
        time_index = pd.IntervalIndex.from_tuples(datetimes, closed="both", name="datetime")
        self.index = gpd.GeoDataFrame(
            {"label": gdf[label_col].astype(int).values},
            index=time_index,
            geometry=gdf.geometry.values,
            crs=gdf.crs,
        )

    def __getitem__(self, index: Any) -> dict[str, Any]:
        import torch

        x, y, t = self._disambiguate_slice(index)

        interval = __import__("pandas").Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            return {"mask": torch.zeros(1, 1, 1, dtype=torch.long)}

        label_val = int(df.iloc[0]["label"])
        return {"mask": torch.full((1, 1, 1), label_val, dtype=torch.long)}
