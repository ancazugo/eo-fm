"""Label datasets for the eo-fm pipeline."""

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox


def rasterize_gdf(gdf, label_col: str, out_path: str | Path, res: float, nodata: int = 0) -> Path:
    """Burn all polygon labels into a single north-up GeoTIFF.

    All polygons are burned before any train/val/test splitting so the same pixel
    always gets the same label value regardless of split. This avoids the ambiguity
    that arises when overlapping polygons are rasterised independently per split.

    Label encoding:
    - Values 1–17 = LCZ classes (1-indexed, matching lcz_dict)
    - Value 0      = nodata (pixels not covered by any polygon)

    At training time shift by -1: classes become 0–16 and nodata becomes -1
    (used as ignore_index in losses).

    Args:
        gdf: GeoDataFrame with geometry and label_col in the target CRS.
        label_col: Column with 1-based integer class values.
        out_path: Output GeoTIFF path (.tif).
        res: Pixel size in CRS units (e.g. metres for UTM CRS).
        nodata: Fill value for pixels not covered by any polygon.

    Returns:
        Path to the written GeoTIFF.
    """
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_origin

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = gdf.total_bounds
    width  = max(1, int(np.ceil((maxx - minx) / res)))
    height = max(1, int(np.ceil((maxy - miny) / res)))
    maxy_snap = miny + height * res
    transform = from_origin(minx, maxy_snap, res, res)

    shapes = (
        (geom, int(val))
        for geom, val in zip(gdf.geometry, gdf[label_col])
    )
    burned = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=np.uint8,
    )

    with rasterio.open(
        str(out_path), "w",
        driver="GTiff",
        height=height, width=width,
        count=1, dtype="uint8",
        crs=gdf.crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(burned, 1)

    return out_path


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

    @classmethod
    def from_gdf(cls, gdf, label_col: str) -> "VectorPatchLabelDataset":
        """Create a VectorPatchLabelDataset directly from a pre-split GeoDataFrame.

        Bypasses file I/O; the GDF must already have the structure produced by
        VectorPatchLabelDataset.__init__ (pd.IntervalIndex, 'label' column, CRS set).

        Args:
            gdf: Pre-split GeoDataFrame (e.g. from split_label_gdf()).
            label_col: Column name for class labels (stored for reference).
        """
        obj = cls.__new__(cls)
        super(VectorPatchLabelDataset, obj).__init__()
        obj.label_col = label_col
        obj._res = 0.0
        obj.index = gdf
        return obj

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
