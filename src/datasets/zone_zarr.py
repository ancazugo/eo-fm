"""Embeddings streamed from per-UTM-zone Zarr stores (Tessera dClimate, OlmoEarth).

Both stores are published on Source Cooperative with the same layout: one
Zarr v3 group per UTM zone — ``utm01`` … ``utm60`` — each holding int8
``embeddings [time, band, y, x]`` on a 10 m grid that spans both hemispheres
in the zone's northern CRS (``proj:code``), with the grid in the zone's
``spatial:transform`` attr. They differ only in how values are dequantised
and in how much of each year is filled, which :class:`ZoneZarrSource`
captures per registry key:

* ``tesserav1.1_dclimate`` — geotessera 0.11's v1.1 run. int8 × per-pixel
  float32 ``scales`` (NaN/inf = not embedded). No landmask, so water is
  embedded.
* ``olmoearth_v1.3`` / ``olmoearth_v1.3_d64`` — Ai2's OlmoEarth v1.3,
  ``(|q| / 127.5)² · sign(q)`` with -128 = nodata (the coop formula). Bands
  0:64 are a complete Matryoshka embedding stored in their own chunks, so the
  d64 key reads half the bytes. Never mix the two widths in one analysis.

The rest of the pipeline thinks in tiles: picklable ``Path`` objects that
:func:`datasets.tiles.open_tile` turns into a ``(band, y, x)`` DataArray. A
UTM zone is far too big to be one (``lcz_train/mosaics.py`` and
``embedding_rgb.py`` read whole tiles), so this module exposes **virtual 0.1°
cells** named exactly like Tessera tiles — ``grid_{lon}_{lat}`` by cell centre —
under ``<embedding_dir>/<cell_dirname>/<year>/``. Nothing exists at that path:
it only carries the source, cache dir, year and cell into worker processes.

A cell opens **lazily**. Its values sit behind an xarray ``BackendArray``, so
``rio.clip_box`` in ``crop_patch``/``open_and_clip`` fetches only the clipped
window. Covering So2Sat by loading whole cells would pull over a terabyte.

The index (:class:`CellGrid` / :class:`CellPaths`) is arithmetic rather than an
STRtree over 6.5M global cells. Without a coverage mask it covers every cell,
ocean included (dClimate: ocean reads as all-NaN, and
``check_embedding_coverage.py`` over-reports). OlmoEarth publishes per-year
provenance, so its index is masked to the cells its written windows touch —
2017-2022 are only partly computed yet, and an unmasked index would extract
all-NaN patches that a later ``--skip-existing`` top-up could never replace.
"""

from __future__ import annotations

import functools
import math
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from shapely.geometry import box

CELL_SIZE = 0.1
# Global 0.1° lattice: column 0 starts at -180°, row 0 at -90°. Edges are
# computed as integer tenths (k / 10) so they equal the decimal literals —
# -180 + col * 0.1 drifts off them and breaks touching-cell predicates.
PER_DEGREE = 10
N_COLS = 360 * PER_DEGREE
N_ROWS = 180 * PER_DEGREE

# zarr's CacheStore keeps byte-range reads — every sharded chunk read — in
# MEMORY, under the same budget as the on-disk full-key cache (metadata). The
# budget is therefore per-process RAM, multiplied by the extraction pool size.
CACHE_MAX_BYTES = 512 * 1024**2

# Chunk fetches in flight per read; geotessera uses 32 for the same reason
# (remote reads are latency-bound).
READ_CONCURRENCY = 32

# zarr >= 3.3 merges nearby chunk ranges into one request and downloads the
# gaps between them (1 MB by default) — far more bytes than a patch read needs.
# Both stores' READMEs turn it off.
READ_CONFIG = {
    "async.concurrency": READ_CONCURRENCY,
    "array.sharding_coalesce_max_gap_bytes": 0,
}

_CELL_RE = re.compile(r"grid_(?P<lon>-?[\d.]+)_(?P<lat>-?[\d.]+)$")

SOURCE_COOP_S3 = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
OLMOEARTH_ROOT = f"{SOURCE_COOP_S3}/olmoearth/olmoearth-v1-3-embeddings"


# ── Sources ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneZarrSource:
    """What differs between per-UTM-zone stores; everything else is shared."""

    key: str                       # registry key
    cell_dirname: str              # virtual-path marker, one per key
    store_env: str                 # env var overriding the store (URL or path)
    default_store: Callable[[], str]
    n_bands: int                   # bands 0:n_bands are read
    read: Callable                 # (zone, ti, b0, b1, ys, xs) -> float32 (B, H, W), NaN = nodata
    coverage: Callable | None = None   # (cache_dir, year) -> bool (N_ROWS, N_COLS) mask
    extract_chunksize: int = 256   # tasks per worker hand-out in extract_so2sat_embeddings

    def store_location(self) -> str:
        return os.environ.get(self.store_env) or self.default_store()


def _dclimate_store() -> str:
    from geotessera.registry import zarr_store_url

    return zarr_store_url("v1.1", "dclimate")


def _read_dclimate(zone, ti, b0, b1, ys, xs) -> np.ndarray:
    import zarr
    from geotessera.store import TesseraAccessor

    # Latency-bound: fetch both arrays at once rather than in turn.
    with zarr.config.set(READ_CONFIG):
        emb_f = _io_pool(os.getpid()).submit(
            zone["embeddings"].__getitem__, (ti, slice(b0, b1), ys, xs)
        )
        scales = zone["group"]["scales"][ti, ys, xs]
        emb = emb_f.result()
    return TesseraAccessor.dequantise(emb, scales).transpose(2, 0, 1)


# Per-process LRU of raw int8 blocks, aligned to the store's inner chunks.
# zarr's CacheStore never sees sharded inner-chunk reads (measured: only the
# metadata requests reached it), so without this every patch re-downloads the
# 256² chunks it sits in (~6 MB at 128 bands) although ~60 neighbouring
# patches share each one. The budget is RAM per process, times pool size.
BLOCK_CACHE_BYTES = 256 * 1024**2
_blocks: OrderedDict = OrderedDict()
_blocks_nbytes = 0


def _cached_window(zone, ti, b0, b1, ys: slice, xs: slice) -> np.ndarray:
    """``emb[ti, b0:b1, ys, xs]`` assembled from chunk-aligned cached blocks.

    Missing blocks are fetched as one aligned window in a single zarr call
    (re-reading any cached ones inside it — simpler, and rare for patch-sized
    requests), then split into the cache.
    """
    import zarr

    global _blocks_nbytes
    emb = zone["embeddings"]
    bh, bw = emb.chunks[2:]
    by0, by1 = ys.start // bh, (ys.stop - 1) // bh
    bx0, bx1 = xs.start // bw, (xs.stop - 1) // bw
    keys = [(zone["id"], ti, b0, b1, by, bx)
            for by in range(by0, by1 + 1) for bx in range(bx0, bx1 + 1)]
    if not all(k in _blocks for k in keys):
        n_rows, n_cols = zone["shape"]
        wy = slice(by0 * bh, min((by1 + 1) * bh, n_rows))
        wx = slice(bx0 * bw, min((bx1 + 1) * bw, n_cols))
        with zarr.config.set(READ_CONFIG):
            win = emb[ti, b0:b1, wy, wx]
        for (_, _, _, _, by, bx) in keys:
            blk = win[:, (by - by0) * bh:(by - by0 + 1) * bh, (bx - bx0) * bw:(bx - bx0 + 1) * bw]
            k = (zone["id"], ti, b0, b1, by, bx)
            if k not in _blocks:
                _blocks[k] = blk.copy()
                _blocks_nbytes += blk.nbytes
    out = np.empty((b1 - b0, ys.stop - ys.start, xs.stop - xs.start), dtype=np.int8)
    for k in keys:
        _blocks.move_to_end(k)
        by, bx = k[4], k[5]
        blk = _blocks[k]
        # Intersection of this block with the request, in zone pixel coords.
        y0, x0 = max(ys.start, by * bh), max(xs.start, bx * bw)
        y1, x1 = min(ys.stop, by * bh + blk.shape[1]), min(xs.stop, bx * bw + blk.shape[2])
        out[:, y0 - ys.start:y1 - ys.start, x0 - xs.start:x1 - xs.start] = \
            blk[:, y0 - by * bh:y1 - by * bh, x0 - bx * bw:x1 - bx * bw]
    while _blocks_nbytes > BLOCK_CACHE_BYTES and len(_blocks) > len(keys):
        _, old = _blocks.popitem(last=False)
        _blocks_nbytes -= old.nbytes
    return out


def _read_olmoearth(zone, ti, b0, b1, ys, xs) -> np.ndarray:
    from dequantize_embeddings import dequantize_alphaearth_embeddings

    q = _cached_window(zone, ti, b0, b1, ys, xs)
    v = q.astype(np.float32)
    v[q == -128] = np.nan
    return dequantize_alphaearth_embeddings(v)


def _olmoearth_coverage(cache_dir: Path, year: int) -> np.ndarray:
    """0.1° cells touched by a window OlmoEarth has written for *year*.

    From ``provenance/windows/{year}.parquet``, cached under the cache dir and
    re-fetched when its ETag changes — so the 2017-2022 backfill is picked up
    without a flag. Offline, the cached copy is used with a warning.
    """
    import urllib.request

    root = os.environ.get("OLMOEARTH_PROVENANCE", f"{OLMOEARTH_ROOT}/provenance")
    url = f"{root}/windows/{year}.parquet"
    d = Path(cache_dir) / "olmoearth_provenance"
    d.mkdir(parents=True, exist_ok=True)
    parquet, mask_npy, etag_file = (
        d / f"windows_{year}.parquet", d / f"windows_{year}.mask.npy", d / f"windows_{year}.etag",
    )

    cached_etag = etag_file.read_text() if etag_file.exists() else None
    try:
        if "://" in url:
            req = urllib.request.Request(url, method="HEAD")
            etag = urllib.request.urlopen(req, timeout=60).headers["ETag"]
        else:
            etag = str(Path(url).stat().st_mtime_ns)
    except Exception as e:  # noqa: BLE001 — offline: fall back to the cache
        if not parquet.exists():
            raise RuntimeError(f"OlmoEarth coverage for {year} unavailable ({e}) and not cached") from e
        logger.warning(f"OlmoEarth provenance unreachable ({e}); using cached {parquet.name}")
        etag = cached_etag

    if etag == cached_etag and mask_npy.exists():
        return np.load(mask_npy)

    if etag != cached_etag or not parquet.exists():
        logger.info(f"Fetching OlmoEarth coverage: {url}")
        if "://" in url:
            urllib.request.urlretrieve(url, parquet)
        else:
            parquet.write_bytes(Path(url).read_bytes())

    mask = coverage_mask_from_windows(parquet)
    np.save(mask_npy, mask)
    if etag is not None:
        etag_file.write_text(etag)
    logger.info(f"OlmoEarth {year}: {int(mask.sum())} covered 0.1° cells")
    return mask


def coverage_mask_from_windows(parquet: Path) -> np.ndarray:
    """Rasterise the written windows' WGS84 bounds onto the 0.1° lattice."""
    import geopandas as gpd
    import shapely

    w = gpd.read_parquet(parquet, columns=["written", "geometry"])
    b = shapely.bounds(w.loc[w["written"].astype(bool), "geometry"].values)
    c0 = np.clip(np.floor((b[:, 0] + 180.0) * PER_DEGREE), 0, N_COLS - 1).astype(int)
    c1 = np.clip(np.floor((b[:, 2] + 180.0) * PER_DEGREE), 0, N_COLS - 1).astype(int)
    r0 = np.clip(np.floor((b[:, 1] + 90.0) * PER_DEGREE), 0, N_ROWS - 1).astype(int)
    r1 = np.clip(np.floor((b[:, 3] + 90.0) * PER_DEGREE), 0, N_ROWS - 1).astype(int)
    mask = np.zeros((N_ROWS, N_COLS), dtype=bool)
    for a, bb, c, d in zip(r0, r1, c0, c1):
        mask[a:bb + 1, c:d + 1] = True
    return mask


def _olmoearth_store() -> str:
    return f"{OLMOEARTH_ROOT}/embeddings.zarr"


SOURCES: dict[str, ZoneZarrSource] = {
    s.key: s for s in (
        # Small inner chunks (32²): each patch is a few latency-bound reads, so
        # small hand-outs keep every worker busy (256 left a 2.4k-patch split
        # on 10 of 32 workers).
        ZoneZarrSource(
            "tesserav1.1_dclimate", "dclimate_cells", "TESSERA_DCLIMATE_STORE",
            _dclimate_store, 128, _read_dclimate, extract_chunksize=16,
        ),
        # 256² inner chunks hold ~60 So2Sat patches each, so runs of same-cell
        # patches on one worker reuse them through the block cache. Measured
        # on 2.8k patches x 32 workers: chunksize 64 ~= 256 (102 vs 104 s),
        # 16 slower (114 s, blocks re-fetched by more workers); 64 also
        # starves fewer workers on single-city runs.
        ZoneZarrSource(
            "olmoearth_v1.3", "olmoearth_v1.3_cells", "OLMOEARTH_STORE",
            _olmoearth_store, 128, _read_olmoearth, _olmoearth_coverage, extract_chunksize=64,
        ),
        ZoneZarrSource(
            "olmoearth_v1.3_d64", "olmoearth_v1.3_d64_cells", "OLMOEARTH_STORE",
            _olmoearth_store, 64, _read_olmoearth, _olmoearth_coverage, extract_chunksize=64,
        ),
    )
}
_BY_DIRNAME = {s.cell_dirname: s for s in SOURCES.values()}


def extract_chunksize(key: str, default: int = 256) -> int:
    """Tasks per worker hand-out for *key* (``default`` for non-zone sources)."""
    return SOURCES[key].extract_chunksize if key in SOURCES else default


# ── Virtual cell paths ───────────────────────────────────────────────────────

def is_cell_path(path: Path) -> bool:
    """True for a virtual cell path built by :func:`cell_path`."""
    return path.parent.parent.name in _BY_DIRNAME


def cell_path(source: ZoneZarrSource, embedding_dir: Path, year: int | str, col: int, row: int) -> Path:
    """Virtual path of cell (*col*, *row*) on the global 0.1° lattice."""
    lon = (col - N_COLS // 2 + 0.5) / PER_DEGREE
    lat = (row - N_ROWS // 2 + 0.5) / PER_DEGREE
    return Path(embedding_dir) / source.cell_dirname / str(year) / f"grid_{lon:.2f}_{lat:.2f}"


def parse_cell_path(path: Path) -> tuple[ZoneZarrSource, Path, int, float, float]:
    """``(source, embedding_dir, year, lon, lat)`` of a virtual cell path (centre coords)."""
    m = _CELL_RE.match(path.name)
    if m is None or not is_cell_path(path):
        raise ValueError(f"Not a zone-zarr cell path: {path}")
    return (
        _BY_DIRNAME[path.parent.parent.name], path.parent.parent.parent,
        int(path.parent.name), float(m["lon"]), float(m["lat"]),
    )


def _cell_box(index: int):
    row, col = divmod(int(index), N_COLS)
    w, e = (col - N_COLS // 2) / PER_DEGREE, (col - N_COLS // 2 + 1) / PER_DEGREE
    s, n = (row - N_ROWS // 2) / PER_DEGREE, (row - N_ROWS // 2 + 1) / PER_DEGREE
    return box(w, s, e, n)


class CellPaths(Sequence):
    """``paths[i]`` for :class:`CellGrid` index *i*, built on demand.

    *cells* (sorted global cell indices) restricts the sequence to covered
    cells, so ``len(paths)`` and random draws over it — e.g. the
    ``embedding_rgb fit`` rural supplement — only ever see real data.
    """

    def __init__(self, source: ZoneZarrSource, embedding_dir: Path, year: int | str,
                 cells: np.ndarray | None = None):
        self.source = source
        self.embedding_dir = Path(embedding_dir)
        self.year = year
        self.cells = cells

    def __len__(self) -> int:
        return N_COLS * N_ROWS if self.cells is None else len(self.cells)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        g = int(i) if self.cells is None else int(self.cells[int(i)])
        row, col = divmod(g, N_COLS)
        return cell_path(self.source, self.embedding_dir, self.year, col, row)


class _CellGeometries:
    """``tree.geometries[idx]`` for an int or an index array (compact indices)."""

    def __init__(self, cells: np.ndarray | None = None):
        self.cells = cells

    def __len__(self) -> int:
        return N_COLS * N_ROWS if self.cells is None else len(self.cells)

    def _global(self, i) -> int:
        return int(i) if self.cells is None else int(self.cells[int(i)])

    def __getitem__(self, idx):
        if np.ndim(idx) == 0:
            return _cell_box(self._global(idx))
        out = np.empty(len(idx), dtype=object)
        out[:] = [_cell_box(self._global(i)) for i in np.asarray(idx)]
        return out


class CellGrid:
    """Duck-typed stand-in for the ``STRtree`` that :func:`build_tile_index` returns.

    Supports what callers use: ``query(geom)`` → 1-D indices,
    ``query(geoms, predicate=...)`` → ``(2, n)`` ``[input, cell]`` pairs (the
    shapely convention), and ``geometries[idx]``. Candidates come from the
    query bounds, minus cells outside *mask* when one is given; a predicate
    is then tested against the real cell box. Without a predicate, as with
    STRtree, the bounds test is the answer.

    With a mask, indices are positions in ``cells`` (the sorted covered cell
    indices) rather than global cell numbers, matching :class:`CellPaths`.
    """

    def __init__(self, mask: np.ndarray | None = None):
        self.mask = None if mask is None else mask.ravel()
        self.cells = None if mask is None else np.flatnonzero(self.mask)
        self.geometries = _CellGeometries(self.cells)

    def _candidates(self, geom) -> np.ndarray:
        w, s, e, n = geom.bounds
        c0 = max(0, math.floor((w + 180.0) * PER_DEGREE))
        c1 = min(N_COLS - 1, math.floor((e + 180.0) * PER_DEGREE))
        r0 = max(0, math.floor((s + 90.0) * PER_DEGREE))
        r1 = min(N_ROWS - 1, math.floor((n + 90.0) * PER_DEGREE))
        if c1 < c0 or r1 < r0:
            return np.empty(0, dtype=np.intp)
        rows, cols = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
        idx = (rows * N_COLS + cols).ravel().astype(np.intp)
        return idx if self.mask is None else idx[self.mask[idx]]

    def _query_one(self, geom, predicate) -> np.ndarray:
        idx = self._candidates(geom)  # global cell numbers
        if predicate is not None and idx.size:
            idx = np.asarray([i for i in idx if getattr(geom, predicate)(_cell_box(i))], dtype=np.intp)
        if self.cells is None or idx.size == 0:
            return idx
        return np.searchsorted(self.cells, idx).astype(np.intp)

    def query(self, geom, predicate: str | None = None):
        if isinstance(geom, (list, tuple, np.ndarray)):
            pairs = [
                (k, i) for k, g in enumerate(geom) for i in self._query_one(g, predicate)
            ]
            if not pairs:
                return np.empty((2, 0), dtype=np.intp)
            return np.asarray(pairs, dtype=np.intp).T
        return self._query_one(geom, predicate)


def build_index(key: str, embedding_dir: Path, year) -> tuple[CellPaths, CellGrid]:
    """``(paths, tree)`` for :func:`datasets.tiles.build_tile_index`.

    Opens no store, so a forking caller holds no client yet; only a source
    with a coverage mask reads (and caches) its provenance.
    """
    source = SOURCES[key]
    mask = None if source.coverage is None else source.coverage(Path(embedding_dir), int(year))
    tree = CellGrid(mask)
    return CellPaths(source, embedding_dir, year, tree.cells), tree


# ── Store access (per process: forked workers must not share the client) ────

@functools.lru_cache(maxsize=4)
def _root(location: str, cache_dir: str, pid: int):
    """Root group of the store. *pid* keys the cache so a forked child re-opens."""
    import zarr
    from geotessera.store import zarr_store

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # geotessera's opener: obstore HTTPStore with retries + zarr CacheStore.
    store = zarr_store(location, cache_dir=cache_dir, cache_max_size=CACHE_MAX_BYTES)
    logger.info(f"Zone-zarr store: {location} (cache {cache_dir}, pid {pid})")
    return zarr.open_group(store, mode="r")


@functools.lru_cache(maxsize=64)
def _zone(location: str, cache_dir: str, pid: int, zone: int) -> dict:
    """Geometry and arrays of one zone group, from its attrs (no coord arrays read)."""
    group = _root(location, cache_dir, pid)[f"utm{zone:02d}"]
    attrs = dict(group.attrs)
    a, _, x0, _, e, y0 = attrs["spatial:transform"][:6]
    if a != -e:
        raise ValueError(f"utm{zone:02d}: non-square pixels {attrs['spatial:transform']}")
    emb = group["embeddings"]
    return {
        # Array handles, not the group: a group lookup re-opens the array
        # metadata on every read. The group stays for source-specific arrays.
        "id": (location, zone),  # cache key: two stores can share a zone CRS
        "group": group,
        "embeddings": emb,
        "crs": attrs["proj:code"],
        "x0": float(x0),
        "y0": float(y0),
        "px": float(a),
        "shape": tuple(emb.shape[2:]),
        "years": [int(t) for t in group["time"][:]],
        "incomplete": attrs.get("geotessera:years_incomplete", {}),
    }


@functools.lru_cache(maxsize=4)
def _io_pool(pid: int):
    """Reader threads; keyed by *pid* because a forked child inherits no threads."""
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="zone_zarr")


def _utm_zone(lon: float) -> int:
    return min(60, max(1, int((lon + 180.0) // 6) + 1))


# ── Lazy cell array ─────────────────────────────────────────────────────────

class _CellBackend(xr.backends.BackendArray):
    """``(band, y, x)`` float32 view of one cell's window; only indexed windows load."""

    def __init__(self, source: ZoneZarrSource, zone: dict, ti: int,
                 rows: tuple[int, int], cols: tuple[int, int]):
        self.source = source
        self.zone = zone
        self.ti = ti
        self.r0, r1 = rows
        self.c0, c1 = cols
        self.shape = (source.n_bands, r1 - self.r0, c1 - self.c0)
        self.dtype = np.dtype("float32")

    def __getitem__(self, key):
        from xarray.core import indexing

        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._read
        )

    def _read(self, key: tuple) -> np.ndarray:
        """Read the bounding range of *key* per axis, then index locally."""
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
            out = self.source.read(self.zone, self.ti, b0, b1, ys, xs)

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


def open_cell(path: Path) -> xr.DataArray:
    """Open a virtual cell as a lazy ``(band, y, x)`` float32 DataArray, CRS set.

    North-up, pixel-centre coords in the zone's UTM CRS — the ``open_tile``
    contract. Raises ``ValueError`` if the store lacks the year.
    """
    import rioxarray  # noqa: F401 — registers .rio
    from xarray.core import indexing

    source, cache_dir, year, lon, lat = parse_cell_path(path)
    zone_no = _utm_zone(lon)
    zone = _zone(source.store_location(), str(cache_dir), os.getpid(), zone_no)
    if year not in zone["years"]:
        raise ValueError(
            f"Year {year} is not in the {source.key} store (utm{zone_no:02d} has {zone['years']})."
        )
    for group, years in zone["incomplete"].items():
        if year in years:
            logger.warning(f"{source.key} {group} has no complete {year}; gaps read as nodata")
    ti = zone["years"].index(year)

    (r0, r1), (c0, c1) = cell_window(zone, lon, lat)
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"{path.name} lies outside utm{zone_no:02d}'s grid")
    px = zone["px"]
    backend = _CellBackend(source, zone, ti, (r0, r1), (c0, c1))
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
