"""GeoDataset subclass that reads Zarr stores instead of GeoTIFFs."""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import shapely
import torch
import xarray as xr
from pyproj import CRS, Transformer
from torchgeo.datasets.geo import GeoDataset


class ZarrGeoDataset(GeoDataset):
    """GeoDataset backed by Zarr stores.

    Unlike RasterDataset (which uses rasterio/GeoTIFF), this dataset reads
    xarray-compatible Zarr stores. Each Zarr store should contain an
    'embedding' variable with dims (band, y, x) and rioxarray CRS metadata.
    """

    is_image = True

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """Initialize ZarrGeoDataset.

        Args:
            paths: Path(s) to directory containing .zarr stores, or list of .zarr paths.
            crs: CRS to warp data to. Defaults to the CRS of the first Zarr store.
            res: Resolution in CRS units. Defaults to the resolution of the first store.
            transforms: Transforms to apply to each sample.
        """
        self.transforms = transforms

        # Collect .zarr directories
        if isinstance(paths, str | Path):
            root = Path(paths)
            zarr_paths = sorted(root.glob("*.zarr"))
        else:
            zarr_paths = [Path(p) for p in paths]

        if not zarr_paths:
            raise FileNotFoundError(f"No .zarr stores found in {paths}")

        filepaths = []
        datetimes = []
        geometries = []

        for zpath in zarr_paths:
            ds = xr.open_zarr(str(zpath))
            da = ds["embedding"]

            # rioxarray may not pick up CRS from Zarr; fall back to spatial_ref attrs
            rio_crs = da.rio.crs
            if rio_crs is None and "spatial_ref" in ds:
                crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
                if crs_wkt:
                    da = da.rio.write_crs(crs_wkt)

            store_crs = CRS.from_user_input(da.rio.crs)
            bounds = da.rio.bounds()  # (left, bottom, right, top)

            if crs is None:
                crs = store_crs

            if res is None:
                # Compute resolution from coordinate spacing
                x_coords = da.coords["x"].values
                y_coords = da.coords["y"].values
                xres = abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 1.0
                yres = abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 1.0
                res = (xres, yres)

            # Transform bounds to target CRS if needed
            if store_crs != crs:
                transformer = Transformer.from_crs(store_crs, crs, always_xy=True)
                left, bottom = transformer.transform(bounds[0], bounds[1])
                right, top = transformer.transform(bounds[2], bounds[3])
                bounds = (left, bottom, right, top)

            geometries.append(shapely.box(*bounds))
            filepaths.append(str(zpath))
            # Use a wide time range since embeddings are generally timeless
            mint = datetime(1900, 1, 1)
            maxt = datetime(2100, 12, 31)
            datetimes.append((mint, maxt))

            ds.close()

        if isinstance(res, int | float):
            res = (res, res)
        self._res = res

        data = {"filepath": filepaths}
        index = pd.IntervalIndex.from_tuples(datetimes, closed="both", name="datetime")
        self.index = gpd.GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    def __getitem__(self, index: Any) -> dict[str, Any]:
        """Retrieve an image sample for the given spatiotemporal query.

        Args:
            index: GeoSlice with [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres].

        Returns:
            Dict with 'image', 'bounds', and 'transform' keys.
        """
        x, y, t = self._disambiguate_slice(index)

        # Temporal filtering
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]

        # Spatial filtering
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        target_crs = self.index.crs

        arrays = []
        for filepath in df.filepath:
            ds = xr.open_zarr(filepath)
            da = ds["embedding"]

            # rioxarray may not pick up CRS from Zarr; fall back to spatial_ref attrs
            if da.rio.crs is None and "spatial_ref" in ds:
                crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
                if crs_wkt:
                    da = da.rio.write_crs(crs_wkt)

            store_crs = CRS.from_user_input(da.rio.crs)

            # Reproject if CRS doesn't match
            if store_crs != target_crs:
                da = da.rio.reproject(str(target_crs), resolution=self._res)

            # Clip to query bbox
            da = da.rio.clip_box(
                minx=x.start, miny=y.start, maxx=x.stop, maxy=y.stop,
                allow_one_dimensional_raster=True,
            )
            arrays.append(da)

        # Merge multiple tiles if needed
        if len(arrays) == 1:
            merged = arrays[0]
        else:
            from rioxarray.merge import merge_arrays

            merged = merge_arrays(arrays)

        # Convert to tensor (band, H, W)
        data = merged.values.astype(np.float32)
        tensor = torch.from_numpy(data)

        # Ensure consistent spatial dimensions (clip_box can be off by 1 pixel)
        target_h = round((y.stop - y.start) / abs(y.step))
        target_w = round((x.stop - x.start) / abs(x.step))
        if tensor.ndim == 3:
            _, h, w = tensor.shape
            if h != target_h or w != target_w:
                tensor = tensor[:, :target_h, :target_w]
                # Pad if clipped result was smaller than expected
                if tensor.shape[1] < target_h or tensor.shape[2] < target_w:
                    padded = torch.zeros(tensor.shape[0], target_h, target_w, dtype=tensor.dtype)
                    padded[:, :tensor.shape[1], :tensor.shape[2]] = tensor
                    tensor = padded

        import rasterio.transform

        transform = rasterio.transform.from_origin(
            x.start, y.stop, x.step, y.step,
        )

        sample = {
            "bounds": self._slice_to_tensor(index),
            "transform": torch.tensor(transform),
            "image": tensor,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
