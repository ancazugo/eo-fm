"""Tessera v1.1 dClimate embeddings, streamed from the published Zarr store.

geotessera 0.11 publishes the complete global v1.1 run (the ``dclimate``
variant, 2017-2025) as one Zarr v3 store on Source Cooperative, laid out as one
group per UTM zone — ``utm01`` … ``utm60`` — each holding int8 ``embeddings``
``[time, band, y, x]`` and float32 ``scales`` ``[time, y, x]`` on a 10 m grid
that spans both hemispheres in the zone's northern CRS (``proj:code``).

The rest of the pipeline thinks in tiles: picklable ``Path`` objects that
:func:`datasets.tiles.open_tile` turns into a ``(band, y, x)`` DataArray. A
UTM zone is far too big to be one (``lcz_train/mosaics.py`` and
``embedding_rgb.py`` read whole tiles), so this module exposes **virtual 0.1°
cells** named exactly like Tessera tiles — ``grid_{lon}_{lat}`` by cell centre —
under ``<embedding_dir>/dclimate_cells/<year>/``. Nothing exists at that path:
it only carries the cache dir, year and cell into worker processes.

A cell opens **lazily**. Its values sit behind an xarray ``BackendArray``, so
``rio.clip_box`` in ``crop_patch``/``open_and_clip`` fetches only the clipped
window. That matters: the store's inner chunks are 32×32×128, so a So2Sat
patch costs ~4 chunk reads, whereas loading whole cells to cover So2Sat would
pull ~1.25 TB.

Nodata: the store sets ``geotessera:mask_source = source_nodata`` and no
landmask, so water is embedded and NaN (non-finite scales) marks a real gap.

The index (:class:`CellGrid` / :class:`CellPaths`) is arithmetic rather than an
STRtree over 6.5M global cells. It covers every cell, ocean included; ocean
cells read as all-NaN and so become nodata, but ``check_embedding_coverage.py``
will over-report coverage for this embedding.
"""

from __future__ import annotations

import functools
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from shapely.geometry import box

CELL_DIRNAME = "dclimate_cells"
CELL_SIZE = 0.1
# Global 0.1° lattice: column 0 starts at -180°, row 0 at -90°. Edges are
# computed as integer tenths (k / 10) so they equal the decimal literals —
# -180 + col * 0.1 drifts off them and breaks touching-cell predicates.
PER_DEGREE = 10
N_COLS = 360 * PER_DEGREE
N_ROWS = 180 * PER_DEGREE

# Override the store location (URL or local path) — the offline tests point
# this at a synthetic store; it also serves a local mirror.
STORE_ENV = "TESSERA_DCLIMATE_STORE"

# zarr's CacheStore keeps byte-range reads — every sharded chunk read — in
# MEMORY, under the same budget as the on-disk full-key cache (metadata). The
# budget is therefore per-process RAM, multiplied by the extraction pool size.
CACHE_MAX_BYTES = 512 * 1024**2

# Chunk fetches in flight per read; geotessera uses 32 for the same reason
# (remote reads are latency-bound).
READ_CONCURRENCY = 32

_CELL_RE = re.compile(r"grid_(?P<lon>-?[\d.]+)_(?P<lat>-?[\d.]+)$")


# ── Virtual cell paths ───────────────────────────────────────────────────────

def is_cell_path(path: Path) -> bool:
    """True for a virtual dClimate cell path built by :func:`cell_path`."""
    return path.parent.parent.name == CELL_DIRNAME


def cell_path(embedding_dir: Path, year: int | str, col: int, row: int) -> Path:
    """Virtual path of cell (*col*, *row*) on the global 0.1° lattice."""
    lon = (col - N_COLS // 2 + 0.5) / PER_DEGREE
    lat = (row - N_ROWS // 2 + 0.5) / PER_DEGREE
    return Path(embedding_dir) / CELL_DIRNAME / str(year) / f"grid_{lon:.2f}_{lat:.2f}"


def parse_cell_path(path: Path) -> tuple[Path, int, float, float]:
    """``(embedding_dir, year, lon, lat)`` of a virtual cell path (centre coords)."""
    m = _CELL_RE.match(path.name)
    if m is None or not is_cell_path(path):
        raise ValueError(f"Not a dClimate cell path: {path}")
    return path.parent.parent.parent, int(path.parent.name), float(m["lon"]), float(m["lat"])


def _cell_box(index: int):
    row, col = divmod(int(index), N_COLS)
    w, e = (col - N_COLS // 2) / PER_DEGREE, (col - N_COLS // 2 + 1) / PER_DEGREE
    s, n = (row - N_ROWS // 2) / PER_DEGREE, (row - N_ROWS // 2 + 1) / PER_DEGREE
    return box(w, s, e, n)


class CellPaths(Sequence):
    """``paths[i]`` for :class:`CellGrid` index *i*, built on demand."""

    def __init__(self, embedding_dir: Path, year: int | str):
        self.embedding_dir = Path(embedding_dir)
        self.year = year

    def __len__(self) -> int:
        return N_COLS * N_ROWS

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        row, col = divmod(int(i), N_COLS)
        return cell_path(self.embedding_dir, self.year, col, row)


class _CellGeometries:
    """``tree.geometries[idx]`` for an int or an index array."""

    def __len__(self) -> int:
        return N_COLS * N_ROWS

    def __getitem__(self, idx):
        if np.ndim(idx) == 0:
            return _cell_box(idx)
        out = np.empty(len(idx), dtype=object)
        out[:] = [_cell_box(i) for i in np.asarray(idx)]
        return out


class CellGrid:
    """Duck-typed stand-in for the ``STRtree`` that :func:`build_tile_index` returns.

    Supports what callers use: ``query(geom)`` → 1-D indices,
    ``query(geoms, predicate=...)`` → ``(2, n)`` ``[input, cell]`` pairs (the
    shapely convention), and ``geometries[idx]``. Candidates come from the
    query bounds; a predicate is then tested against the real cell box.
    Without a predicate, as with STRtree, the bounds test is the answer.
    """

    geometries = _CellGeometries()

    def _candidates(self, geom) -> np.ndarray:
        w, s, e, n = geom.bounds
        c0 = max(0, math.floor((w + 180.0) * PER_DEGREE))
        c1 = min(N_COLS - 1, math.floor((e + 180.0) * PER_DEGREE))
        r0 = max(0, math.floor((s + 90.0) * PER_DEGREE))
        r1 = min(N_ROWS - 1, math.floor((n + 90.0) * PER_DEGREE))
        if c1 < c0 or r1 < r0:
            return np.empty(0, dtype=np.intp)
        rows, cols = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
        return (rows * N_COLS + cols).ravel().astype(np.intp)

    def _query_one(self, geom, predicate) -> np.ndarray:
        idx = self._candidates(geom)
        if predicate is None or idx.size == 0:
            return idx
        keep = [i for i in idx if getattr(geom, predicate)(_cell_box(i))]
        return np.asarray(keep, dtype=np.intp)

    def query(self, geom, predicate: str | None = None):
        if isinstance(geom, (list, tuple, np.ndarray)):
            pairs = [
                (k, i) for k, g in enumerate(geom) for i in self._query_one(g, predicate)
            ]
            if not pairs:
                return np.empty((2, 0), dtype=np.intp)
            return np.asarray(pairs, dtype=np.intp).T
        return self._query_one(geom, predicate)


# ── Store access (per process: forked workers must not share the client) ────

def store_location() -> str:
    """The dClimate store: ``$TESSERA_DCLIMATE_STORE`` or geotessera's published URL."""
    override = os.environ.get(STORE_ENV)
    if override:
        return override
    from geotessera.registry import zarr_store_url

    return zarr_store_url("v1.1", "dclimate")


@functools.lru_cache(maxsize=4)
def _root(location: str, cache_dir: str, pid: int):
    """Root group of the store. *pid* keys the cache so a forked child re-opens."""
    import zarr
    from geotessera.store import zarr_store

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    store = zarr_store(location, cache_dir=cache_dir, cache_max_size=CACHE_MAX_BYTES)
    logger.info(f"dClimate store: {location} (cache {cache_dir}, pid {pid})")
    return zarr.open_group(store, mode="r")


@functools.lru_cache(maxsize=64)
def _zone(location: str, cache_dir: str, pid: int, zone: int) -> dict:
    """Geometry and arrays of one zone group, from its attrs (no coord arrays read)."""
    group = _root(location, cache_dir, pid)[f"utm{zone:02d}"]
    attrs = dict(group.attrs)
    a, _, x0, _, e, y0 = attrs["spatial:transform"][:6]
    if a != -e:
        raise ValueError(f"utm{zone:02d}: non-square pixels {attrs['spatial:transform']}")
    return {
        # Array handles, not the group: a group lookup re-opens the array
        # metadata on every read.
        "embeddings": group["embeddings"],
        "scales": group["scales"],
        "crs": attrs["proj:code"],
        "x0": float(x0),
        "y0": float(y0),
        "px": float(a),
        "shape": tuple(group["scales"].shape[1:]),
        "years": [int(t) for t in group["time"][:]],
        "incomplete": attrs.get("geotessera:years_incomplete", {}),
    }


@functools.lru_cache(maxsize=4)
def _io_pool(pid: int):
    """Reader threads; keyed by *pid* because a forked child inherits no threads."""
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="dclimate")


def _utm_zone(lon: float) -> int:
    return min(60, max(1, int((lon + 180.0) // 6) + 1))


# ── Lazy cell array ─────────────────────────────────────────────────────────

class _CellBackend(xr.backends.BackendArray):
    """``(band, y, x)`` float32 view of one cell's window; only indexed windows load."""

    def __init__(self, zone: dict, ti: int, rows: tuple[int, int], cols: tuple[int, int]):
        self.zone = zone
        self.ti = ti
        self.r0, r1 = rows
        self.c0, c1 = cols
        n_bands = zone["embeddings"].shape[1]
        self.shape = (n_bands, r1 - self.r0, c1 - self.c0)
        self.dtype = np.dtype("float32")

    def __getitem__(self, key):
        from xarray.core import indexing

        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._read
        )

    def _read(self, key: tuple) -> np.ndarray:
        """Read the bounding range of *key* per axis, then index locally."""
        import zarr
        from geotessera.store import TesseraAccessor

        spans, local = [], []
        for k, n in zip(key, self.shape):
            if isinstance(k, (int, np.integer)):
                k = int(k) + n if k < 0 else int(k)
                spans.append((k, k + 1))
                local.append(0)
            else:
                idx = np.arange(*k.indices(n))
                if idx.size == 0:
                    spans.append((0, 0))
                    local.append(slice(0, 0))
                    continue
                lo = int(idx.min())
                spans.append((lo, int(idx.max()) + 1))
                step_one = k.step in (None, 1)
                local.append(slice(None) if step_one else idx - lo)

        (b0, b1), (y0, y1), (x0, x1) = spans
        if b1 <= b0 or y1 <= y0 or x1 <= x0:
            out = np.empty((b1 - b0, y1 - y0, x1 - x0), np.float32)
        else:
            ys = slice(self.r0 + y0, self.r0 + y1)
            xs = slice(self.c0 + x0, self.c0 + x1)
            # Latency-bound: fetch both arrays at once rather than in turn.
            with zarr.config.set({"async.concurrency": READ_CONCURRENCY}):
                emb_f = _io_pool(os.getpid()).submit(
                    self.zone["embeddings"].__getitem__, (self.ti, slice(b0, b1), ys, xs)
                )
                scales = self.zone["scales"][self.ti, ys, xs]
                emb = emb_f.result()
            out = TesseraAccessor.dequantise(emb, scales).transpose(2, 0, 1)

        # Fancy (stepped) axes first, then drop the integer-indexed axes.
        for axis, loc in enumerate(local):
            if isinstance(loc, np.ndarray):
                out = np.take(out, loc, axis=axis)
        return out[tuple(0 if isinstance(loc, int) else slice(None) for loc in local)]


def cell_window(zone: dict, lon: float, lat: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """``((row0, row1), (col0, col1))`` of the zone grid covering the 0.1° cell.

    The window is the zone-lattice envelope of the cell's lon/lat box, so the
    box (the index footprint) is always inside it; neighbouring windows
    overlap by a pixel or so on the same lattice, which is harmless.
    """
    from pyproj import Transformer

    h = CELL_SIZE / 2
    t = Transformer.from_crs("EPSG:4326", zone["crs"], always_xy=True)
    minx, miny, maxx, maxy = t.transform_bounds(lon - h, lat - h, lon + h, lat + h, densify_pts=21)
    px, (n_rows, n_cols) = zone["px"], zone["shape"]
    c0 = max(0, math.floor((minx - zone["x0"]) / px))
    c1 = min(n_cols, math.ceil((maxx - zone["x0"]) / px))
    r0 = max(0, math.floor((zone["y0"] - maxy) / px))
    r1 = min(n_rows, math.ceil((zone["y0"] - miny) / px))
    return (r0, r1), (c0, c1)


def open_dclimate_cell(path: Path) -> xr.DataArray:
    """Open a virtual cell as a lazy ``(band, y, x)`` float32 DataArray, CRS set.

    North-up, pixel-centre coords in the zone's UTM CRS — the ``open_tile``
    contract. Raises ``ValueError`` if the store lacks the year.
    """
    import rioxarray  # noqa: F401 — registers .rio
    from xarray.core import indexing

    cache_dir, year, lon, lat = parse_cell_path(path)
    zone_no = _utm_zone(lon)
    zone = _zone(store_location(), str(cache_dir), os.getpid(), zone_no)
    if year not in zone["years"]:
        raise ValueError(
            f"Year {year} is not in the dClimate store (utm{zone_no:02d} has {zone['years']})."
        )
    for group, years in zone["incomplete"].items():
        if year in years:
            logger.warning(f"dClimate {group} has no complete {year}; gaps read as nodata")
    ti = zone["years"].index(year)

    (r0, r1), (c0, c1) = cell_window(zone, lon, lat)
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"{path.name} lies outside utm{zone_no:02d}'s grid")
    px = zone["px"]
    backend = _CellBackend(zone, ti, (r0, r1), (c0, c1))
    var = xr.Variable(("band", "y", "x"), indexing.LazilyIndexedArray(backend))
    da = xr.DataArray(
        var,
        coords={
            "band": np.arange(backend.shape[0]),
            "y": zone["y0"] - (np.arange(r0, r1) + 0.5) * px,
            "x": zone["x0"] + (np.arange(c0, c1) + 0.5) * px,
        },
    )
    return da.rio.write_crs(zone["crs"])
