"""GeoDataset subclass that reads Zarr stores instead of GeoTIFFs."""

import functools
import re
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

    When ``tile_size`` and ``filename_pattern`` are provided, bounds are parsed
    directly from filenames (O(1) file opens at init instead of O(N)), making
    initialisation much faster for large tile collections. One store is still
    opened to auto-detect CRS and resolution.
    """

    is_image = True

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable | None = None,
        tile_size: float | None = None,
        filename_pattern: str | None = None,
        filename_crs: str | None = None,
        filename_is_center: bool = False,
        bbox: tuple[float, float, float, float] | None = None,
        cache: bool = True,
        cache_size: int = 64,
    ) -> None:
        """Initialize ZarrGeoDataset.

        Args:
            paths: Path(s) to directory containing .zarr stores, or list of
                .zarr paths.
            crs: CRS to warp data to. Defaults to the CRS of the first store.
            res: Resolution in CRS units. Defaults to the resolution of the
                first store.
            transforms: Transforms to apply to each sample.
            tile_size: Tile size in units of ``filename_crs`` (usually degrees).
                When provided together with ``filename_pattern``, bounds are
                parsed from filenames instead of opening each store — reducing
                init from O(N) to O(1) file opens.
            filename_pattern: Regex with named groups ``lon`` and ``lat`` that
                extract tile coordinates from the filename stem. Example::

                    r"gse_(?P<lon>[-\\d.]+)_(?P<lat>[-\\d.]+)_\\d+\\.zarr"

            filename_crs: CRS string for the coordinates encoded in the
                filename (e.g. ``"EPSG:4326"``). Defaults to EPSG:4326.
            filename_is_center: If ``True``, the ``lon``/``lat`` groups in the
                filename refer to the tile *center*; bounds are computed as
                ``[lon ± tile_size/2, lat ± tile_size/2]``.  If ``False``
                (default) they are the bottom-left corner and bounds are
                ``[lon, lat, lon + tile_size, lat + tile_size]``.
            bbox: Optional bounding box ``(west, south, east, north)`` in the
                ``filename_crs`` (EPSG:4326 by default). When provided with
                the fast-path filename approach, only tiles that overlap this
                bbox are included in the index, and CRS/resolution are detected
                from a tile within the bbox rather than the first tile
                alphabetically. This is important when the store directory
                contains tiles from multiple UTM zones (e.g. global tessera
                downloads) — without it, the wrong UTM zone is picked for CRS.
            cache: If ``True``, wrap ``xr.open_zarr`` with an LRU cache so
                that repeatedly accessed tiles are not re-opened on each
                ``__getitem__`` call.
            cache_size: Maximum number of open Zarr stores held in the LRU
                cache.
        """
        self.transforms = transforms

        # Per-instance LRU cache for open Zarr store handles.
        if cache:
            self._open_zarr_store = functools.lru_cache(maxsize=cache_size)(
                self._open_zarr_store_raw
            )
        else:
            self._open_zarr_store = self._open_zarr_store_raw

        # Collect .zarr directories.
        if isinstance(paths, str | Path):
            root = Path(paths)
            zarr_paths = sorted(root.glob("*.zarr"))
        else:
            zarr_paths = [Path(p) for p in paths]

        if not zarr_paths:
            raise FileNotFoundError(f"No .zarr stores found in {paths}")

        filepaths: list[str] = []
        datetimes: list[tuple[datetime, datetime]] = []
        geometries: list = []

        # Embeddings are effectively timeless; use a wide sentinel interval.
        mint = datetime(1900, 1, 1)
        maxt = datetime(2100, 12, 31)

        use_fast_path = tile_size is not None and filename_pattern is not None

        if use_fast_path:
            self._init_from_filenames(
                zarr_paths=zarr_paths,
                tile_size=tile_size,  # type: ignore[arg-type]
                filename_pattern=filename_pattern,  # type: ignore[arg-type]
                filename_crs=filename_crs,
                filename_is_center=filename_is_center,
                bbox=bbox,
                crs=crs,
                res=res,
                mint=mint,
                maxt=maxt,
                filepaths=filepaths,
                datetimes=datetimes,
                geometries=geometries,
            )
            # Retrieve CRS/res that _init_from_filenames detected.
            crs = self._detected_crs
            res = self._detected_res
        else:
            self._init_from_files(
                zarr_paths=zarr_paths,
                crs=crs,
                res=res,
                mint=mint,
                maxt=maxt,
                filepaths=filepaths,
                datetimes=datetimes,
                geometries=geometries,
            )
            crs = self._detected_crs
            res = self._detected_res

        if isinstance(res, int | float):
            res = (res, res)
        self._res = res

        data = {"filepath": filepaths}
        index = pd.IntervalIndex.from_tuples(datetimes, closed="both", name="datetime")
        self.index = gpd.GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    # ------------------------------------------------------------------
    # Private init helpers
    # ------------------------------------------------------------------

    def _read_crs_res(self, zpath: Path) -> tuple[CRS | None, tuple[float, float] | None]:
        """Open one Zarr store and return its CRS and pixel resolution."""
        ds = xr.open_zarr(str(zpath))
        da = ds["embedding"]
        if da.rio.crs is None and "spatial_ref" in ds:
            crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
            if crs_wkt:
                da = da.rio.write_crs(crs_wkt)
        detected_crs = CRS.from_user_input(da.rio.crs) if da.rio.crs is not None else None
        x_coords = da.coords["x"].values
        y_coords = da.coords["y"].values
        xres = abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 1.0
        yres = abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 1.0
        ds.close()
        return detected_crs, (xres, yres)

    def _init_from_filenames(
        self,
        zarr_paths: list[Path],
        tile_size: float,
        filename_pattern: str,
        filename_crs: str | None,
        filename_is_center: bool,
        bbox: tuple[float, float, float, float] | None,
        crs: CRS | None,
        res: tuple[float, float] | None,
        mint: datetime,
        maxt: datetime,
        filepaths: list,
        datetimes: list,
        geometries: list,
    ) -> None:
        """Fast-path init: parse bounds from filenames, open only one store."""
        regex = re.compile(filename_pattern)
        fname_crs = CRS.from_user_input(filename_crs) if filename_crs else CRS.from_epsg(4326)
        half = tile_size / 2 if filename_is_center else None

        def _tile_lon_lat_bounds(zpath: Path) -> tuple[float, float, float, float] | None:
            match = regex.match(zpath.name)
            if match is None:
                return None
            lon = float(match.group("lon"))
            lat = float(match.group("lat"))
            if filename_is_center:
                return lon - half, lat - half, lon + half, lat + half  # type: ignore[operator]
            return lon, lat, lon + tile_size, lat + tile_size

        def _overlaps_bbox(tile_bounds: tuple, query: tuple) -> bool:
            tminx, tminy, tmaxx, tmaxy = tile_bounds
            qminx, qminy, qmaxx, qmaxy = query
            return tmaxx > qminx and tminx < qmaxx and tmaxy > qminy and tminy < qmaxy

        # Open one store to detect CRS/resolution. If bbox is provided, prefer
        # a tile inside it so we get the right UTM zone instead of whatever tile
        # happens to sort first alphabetically.
        if crs is None or res is None:
            candidates = zarr_paths if bbox is None else [
                zp for zp in zarr_paths
                if (b := _tile_lon_lat_bounds(zp)) is not None and _overlaps_bbox(b, bbox)
            ]
            for zpath in (candidates or zarr_paths):
                if regex.match(zpath.name):
                    detected_crs, detected_res = self._read_crs_res(zpath)
                    if crs is None:
                        crs = detected_crs
                    if res is None:
                        res = detected_res
                    break

        self._detected_crs = crs
        self._detected_res = res

        # Build transformer from filename CRS → dataset CRS (if they differ).
        transformer: Transformer | None = None
        if crs is not None and fname_crs != crs:
            transformer = Transformer.from_crs(fname_crs, crs, always_xy=True)

        for zpath in zarr_paths:
            tile_bounds = _tile_lon_lat_bounds(zpath)
            if tile_bounds is None:
                continue  # filename doesn't match pattern

            if bbox is not None and not _overlaps_bbox(tile_bounds, bbox):
                continue  # outside the region of interest

            minx, miny, maxx, maxy = tile_bounds
            if transformer is not None:
                left, bottom = transformer.transform(minx, miny)
                right, top = transformer.transform(maxx, maxy)
                bounds: tuple[float, float, float, float] = (left, bottom, right, top)
            else:
                bounds = (minx, miny, maxx, maxy)

            geometries.append(shapely.box(*bounds))
            filepaths.append(str(zpath))
            datetimes.append((mint, maxt))

    def _init_from_files(
        self,
        zarr_paths: list[Path],
        crs: CRS | None,
        res: tuple[float, float] | None,
        mint: datetime,
        maxt: datetime,
        filepaths: list,
        datetimes: list,
        geometries: list,
    ) -> None:
        """Slow-path init: open each Zarr store to read its bounds."""
        for zpath in zarr_paths:
            ds = xr.open_zarr(str(zpath))
            da = ds["embedding"]

            if da.rio.crs is None and "spatial_ref" in ds:
                crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
                if crs_wkt:
                    da = da.rio.write_crs(crs_wkt)

            store_crs = CRS.from_user_input(da.rio.crs)
            bounds = da.rio.bounds()  # (left, bottom, right, top)

            if crs is None:
                crs = store_crs

            if res is None:
                x_coords = da.coords["x"].values
                y_coords = da.coords["y"].values
                xres = abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 1.0
                yres = abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 1.0
                res = (xres, yres)

            if store_crs != crs:
                t = Transformer.from_crs(store_crs, crs, always_xy=True)
                left, bottom = t.transform(bounds[0], bounds[1])
                right, top = t.transform(bounds[2], bounds[3])
                bounds = (left, bottom, right, top)

            geometries.append(shapely.box(*bounds))
            filepaths.append(str(zpath))
            datetimes.append((mint, maxt))

            ds.close()

        self._detected_crs = crs
        self._detected_res = res

    # ------------------------------------------------------------------
    # Zarr store access
    # ------------------------------------------------------------------

    def _open_zarr_store_raw(self, filepath: str) -> xr.Dataset:
        """Open a Zarr store. Optionally wrapped by an LRU cache in ``__init__``."""
        # chunks=False disables dask, reducing per-patch overhead from ~4s to ~0.04s
        # for random 32×32 patch access patterns typical in training.
        return xr.open_zarr(filepath, chunks=False)

    # ------------------------------------------------------------------
    # GeoDataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, index: Any) -> dict[str, Any]:
        """Retrieve an image sample for the given spatiotemporal query.

        Args:
            index: GeoSlice with [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres].

        Returns:
            Dict with ``'image'``, ``'bounds'``, and ``'transform'`` keys.
        """
        x, y, t = self._disambiguate_slice(index)

        # Temporal filtering.
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]

        # Spatial filtering.
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        target_crs = self.index.crs

        arrays = []
        for filepath in df.filepath:
            ds = self._open_zarr_store(filepath)
            da = ds["embedding"]

            if da.rio.crs is None and "spatial_ref" in ds:
                crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
                if crs_wkt:
                    da = da.rio.write_crs(crs_wkt)

            store_crs = CRS.from_user_input(da.rio.crs)

            if store_crs != target_crs:
                da = da.rio.reproject(str(target_crs), resolution=self._res)

            # Clip BEFORE any y-flip.  Zarr does not support negative strides
            # natively, so isel(y=slice(None,None,-1)) on the full tile forces a
            # whole-tile read (~244 MB for south-up tiles like existing Google
            # AlphaEarth stores) even when only a small patch is needed.
            # Clipping first means zarr reads only the overlapping chunks.
            # NoDataInBounds can occur at tile edges where the spatial index
            # reports overlap but pixel alignment means the raster doesn't
            # actually cover the query box — skip such tiles silently.
            try:
                da = da.rio.clip_box(
                    minx=x.start, miny=y.start, maxx=x.stop, maxy=y.stop,
                    allow_one_dimensional_raster=True,
                )
            except Exception as e:
                if "NoDataInBounds" in type(e).__name__:
                    continue
                raise
            arrays.append(da)

        if not arrays:
            raise IndexError(
                f"No zarr tile contained data for query bounds "
                f"x=[{x.start}, {x.stop}] y=[{y.start}, {y.stop}]"
            )

        if len(arrays) == 1:
            merged = arrays[0]
        else:
            # rasterio.merge (used by merge_arrays) requires transform.e < 0
            # (north-up) and isel(y=::-1) leaves a stale cached transform that
            # merge_arrays reads, so bypass rasterio entirely with a numpy mosaic.
            # All clips share the same CRS and pixel size; place each one at its
            # correct position using y/x coordinates.
            res = abs(self._res[1]) if isinstance(self._res, tuple) else abs(self._res)
            xres = abs(self._res[0]) if isinstance(self._res, tuple) else res

            all_y = sorted(set(v for a in arrays for v in a.y.values.tolist()))
            all_x = sorted(set(v for a in arrays for v in a.x.values.tolist()))
            n_bands = arrays[0].sizes["band"]
            out = np.zeros((n_bands, len(all_y), len(all_x)), dtype=np.float32)

            y_idx = {v: i for i, v in enumerate(all_y)}
            x_idx = {v: i for i, v in enumerate(all_x)}

            for a in arrays:
                vals = a.values.astype(np.float32)  # (bands, h, w), triggers zarr read
                ay = [y_idx[v] for v in a.y.values.tolist()]
                ax = [x_idx[v] for v in a.x.values.tolist()]
                out[np.ix_(range(n_bands), ay, ax)] = vals

            # Build the merged DataArray with ascending y (same orientation as clips)
            merged_y = np.array(all_y, dtype=np.float64)
            merged_x = np.array(all_x, dtype=np.float64)
            merged = xr.DataArray(
                out,
                dims=["band", "y", "x"],
                coords={"band": arrays[0].coords["band"], "y": merged_y, "x": merged_x},
            )
            merged = merged.rio.set_spatial_dims(x_dim="x", y_dim="y")
            if arrays[0].rio.crs is not None:
                merged = merged.rio.write_crs(arrays[0].rio.crs)

        # Normalise to north-up (descending y).  This is done on the small
        # merged patch, so only the relevant zarr chunks are read.
        if merged.sizes.get("y", 0) > 1 and float(merged.y.values[0]) < float(merged.y.values[-1]):
            merged = merged.isel(y=slice(None, None, -1))

        data = merged.values.astype(np.float32)
        tensor = torch.from_numpy(data)

        # Ensure consistent spatial dimensions (clip_box can be off by 1 pixel).
        target_h = round((y.stop - y.start) / abs(y.step))
        target_w = round((x.stop - x.start) / abs(x.step))
        if tensor.ndim == 3:
            _, h, w = tensor.shape
            if h != target_h or w != target_w:
                tensor = tensor[:, :target_h, :target_w]
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
