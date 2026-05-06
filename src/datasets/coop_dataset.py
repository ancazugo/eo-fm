"""GeoDataset for AlphaEarth coop GeoTIFF tiles (source.coop)."""

import functools
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import rasterio.windows
import torch
from pyproj import CRS
from rasterio.enums import Resampling
from torchgeo.datasets.geo import GeoDataset

_S3_PREFIX = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/"


def _s3_to_local(s3_path: str, root: Path) -> Path:
    """Convert an S3 URI to a local path under *root*, mirroring the S3 layout.

    Example::

        s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2017/37S/tile.tiff
        → root/2017/37S/tile.tiff
    """
    return root / s3_path.removeprefix(_S3_PREFIX)


class CoopEmbeddingDataset(GeoDataset):
    """GeoDataset backed by locally-downloaded AlphaEarth coop GeoTIFF tiles.

    Tiles are indexed via the ``aef_index.gpkg`` registry (available alongside
    the coop data on source.coop) and read from locally-downloaded GeoTIFF
    files.  The GeoPackage is filtered at load time by year and optional bbox,
    so only a small fraction of the 302,466-row index is read into memory.

    Expected directory layout (mirroring the S3 bucket structure)::

        root/
            aef_index.gpkg
            2017/
                37S/
                    xdfh15put9tmg8j6r-0000000000-0000000000.tiff
                    xdfh15put9tmg8j6r-0000000000-0000000000.vrt

    Use :func:`~datasets.downloaders.download_alpha_earth_coop` to populate this
    layout from S3 before calling this class.

    Args:
        root: Local directory containing ``aef_index.gpkg`` and downloaded tiles.
        year: Year of embeddings to load (2017–2025).
        bbox: ``(west, south, east, north)`` in EPSG:4326 to restrict which
            tiles are loaded.  Required when the coop directory covers multiple
            UTM zones (globally) — omitting it loads all zones for the year.
        dequantize: If ``True``, apply the AlphaEarth dequantisation formula
            ``sign(v) × (|v| / 127.5)²`` before returning tensors.  Defaults
            to ``False`` — raw Int8 values cast to float32 are returned.  Pass
            ``dequantize=True`` when feeding embeddings directly into a model.
        cache_size: Max open rasterio file handles kept in the per-instance LRU
            cache.  Increase for workloads that access many tiles simultaneously.
        transforms: Optional callable applied to each sample dict after loading.
    """

    is_image = True

    def __init__(
        self,
        root: str | Path,
        year: int,
        bbox: tuple[float, float, float, float] | None = None,
        dequantize: bool = False,
        cache_size: int = 8,
        transforms: Callable | None = None,
    ) -> None:
        self.transforms = transforms
        self._dequantize = dequantize
        root = Path(root)
        index_path = root / "aef_index.gpkg"

        # Push bbox and year filters down to the GPKG reader to avoid loading
        # all 302 k rows.  The 'where' arg is an OGR SQL attribute filter.
        read_kwargs: dict = {"where": f"year = {year}"}
        if bbox is not None:
            read_kwargs["bbox"] = bbox  # (minx, miny, maxx, maxy) = (W, S, E, N)

        gdf = gpd.read_file(index_path, **read_kwargs)

        if len(gdf) == 0:
            raise ValueError(f"No coop tiles found for year={year}, bbox={bbox}")

        # Use the most common CRS among filtered tiles as the dataset CRS.
        # For a small city bbox all tiles share one UTM zone.
        dominant_crs_str = gdf["crs"].value_counts().index[0]

        # Project tile geometries (stored as WGS84 in the GPKG) to dataset CRS.
        gdf_utm = gdf.to_crs(dominant_crs_str)

        local_paths = [str(_s3_to_local(row["path"], root)) for _, row in gdf.iterrows()]
        tile_crses = gdf["crs"].tolist()

        # Assign each tile a temporal interval spanning the full year.
        mint = datetime(year, 1, 1)
        maxt = datetime(year, 12, 31, 23, 59, 59)
        n = len(gdf_utm)

        data = {"local_path": local_paths, "tile_crs": tile_crses}
        time_index = pd.IntervalIndex.from_tuples(
            [(mint, maxt)] * n, closed="both", name="datetime"
        )
        self.index = gpd.GeoDataFrame(
            data,
            index=time_index,
            geometry=gdf_utm.geometry.values,
            crs=dominant_crs_str,
        )

        self._res = (10.0, 10.0)

        # Per-instance LRU cache so tiles accessed frequently are not re-opened.
        self._open_tiff = functools.lru_cache(maxsize=cache_size)(self._open_tiff_raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_tiff_raw(self, path: str) -> rasterio.DatasetReader:
        return rasterio.open(path)

    # ------------------------------------------------------------------
    # GeoDataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, index: Any) -> dict[str, Any]:
        """Return an embedding patch for a spatiotemporal query.

        Args:
            index: GeoSlice ``(x_slice, y_slice, t_slice)`` as produced by a
                GeoSampler (e.g. ``RandomBatchGeoSampler`` or
                ``GridGeoSampler``).

        Returns:
            Dict with keys:

            * ``"image"`` — float32 tensor of shape ``(64, H, W)``.
            * ``"bounds"`` — GeoSlice encoded as a tensor.
            * ``"transform"`` — rasterio Affine as a 6-element tensor.
        """
        x, y, t = self._disambiguate_slice(index)

        # Temporal filter.
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]

        # Spatial filter.
        df = df.cx[x.start : x.stop, y.start : y.stop]

        res = abs(x.step)  # pixel size in CRS units (metres for UTM)
        target_h = round((y.stop - y.start) / res)
        target_w = round((x.stop - x.start) / res)
        n_bands = 64
        output = np.zeros((n_bands, target_h, target_w), dtype=np.float32)

        for _, row in df.iterrows():
            local_path = row["local_path"]
            if not Path(local_path).exists():
                continue

            src = self._open_tiff(local_path)
            tb = src.bounds  # rasterio BoundingBox(left, bottom, right, top)

            # Coop tiles are south-up (transform.e > 0): in rasterio's BoundingBox,
            # 'top' holds the smaller y (geographic south) and 'bottom' holds the
            # larger y (geographic north).  Normalise to geographic y extents.
            tile_y_min = min(tb.top, tb.bottom)  # geographic south
            tile_y_max = max(tb.top, tb.bottom)  # geographic north

            # Intersect query bbox with tile geographic bbox.
            ix_min = max(x.start, tb.left)
            ix_max = min(x.stop, tb.right)
            iy_min = max(y.start, tile_y_min)
            iy_max = min(y.stop, tile_y_max)

            if ix_min >= ix_max or iy_min >= iy_max:
                continue

            out_h = max(1, round((iy_max - iy_min) / res))
            out_w = max(1, round((ix_max - ix_min) / res))

            is_south_up = src.transform.e > 0
            if is_south_up:
                # For south-up transforms, rasterio's from_bounds expects 'bottom'
                # (the row-nrows y value = geographic north) and 'top' (row-0 y
                # value = geographic south), so bottom and top are swapped vs
                # the north-up convention.
                window = rasterio.windows.from_bounds(
                    ix_min, iy_max, ix_max, iy_min, src.transform
                )
            else:
                window = rasterio.windows.from_bounds(
                    ix_min, iy_min, ix_max, iy_max, src.transform
                )

            data = src.read(
                window=window,
                out_shape=(n_bands, out_h, out_w),
                resampling=Resampling.nearest,
            ).astype(np.float32)

            # Zero out nodata pixels (nodata = -128 for coop tiles).
            if src.nodata is not None:
                data[data == src.nodata] = 0.0

            # South-up tiles arrive row-0=south; flip to north-up for placement.
            if is_south_up:
                data = data[:, ::-1, :].copy()

            # Pixel offsets in the north-up output array.
            col = max(0, round((ix_min - x.start) / res))
            row_off = max(0, round((y.stop - iy_max) / res))
            r_end = min(row_off + out_h, target_h)
            c_end = min(col + out_w, target_w)
            output[:, row_off:r_end, col:c_end] = data[:, : r_end - row_off, : c_end - col]

        if self._dequantize:
            from dequantize_embeddings import dequantize_alphaearth_embeddings
            output = dequantize_alphaearth_embeddings(output)

        transform = rasterio.transform.from_origin(x.start, y.stop, res, res)

        sample = {
            "bounds": self._slice_to_tensor(index),
            "transform": torch.tensor(list(transform)),
            "image": torch.from_numpy(output),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
