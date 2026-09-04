"""H3 — multi-annotator consensus on the canonical raster grid.

The same ground is labelled by many independent annotators who frequently
disagree: pooled class agreement between overlapping polygons is ~0.71 by area,
and 0.44 in Guangzhou. This module turns that pile of overlapping opinions into
a per-pixel posterior, then into the Stage 8 label contract.

Why a per-pixel vote rather than a polygon merge: WUDAPT polygons overlap by
construction (Wuhan's overlap area is 6.03x its union), and a ``block_id``
raster is single-valued, so overlapping polygons simply cannot be represented in
the export contract. Voting on the grid computes the planar arrangement of all
annotators implicitly, with no sliver explosion.

Why a *set* rather than a forced hard label: ``lcz_train.losses.marginalized_ce``
scores ``-log sum_{c in S} p_c``, so an ambiguous label costs nothing when the
truth is in the set. Forcing a hard label over the contested ~29% of ground
would inject on the order of 15% wrong-class supervision instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin

from lcz_labels.grid import local_utm_crs

from .config import WudaptConfig
from .ingest import N_LCZ
from .quality import burn_order

__all__ = ["ConsensusGrid", "ConsensusResult", "consensus_for_aoi", "footprint_grid"]


@dataclass(frozen=True)
class ConsensusGrid:
    """Canonical raster grid for one AOI."""

    transform: Affine
    crs: str
    shape: tuple[int, int]
    res_m: float

    @property
    def n_pixels(self) -> int:
        return int(self.shape[0]) * int(self.shape[1])


@dataclass
class ConsensusResult:
    """Per-pixel consensus, stored sparsely over the labelled pixels only."""

    grid: ConsensusGrid
    index: np.ndarray          # flat pixel indices into (H*W), int64
    bitmask: np.ndarray        # uint32, lcz_set encoded bit c-1
    confidence: np.ndarray     # float32 in [0, 1]
    top_class: np.ndarray      # int16, argmax class (1-17); 0 where unlabelled
    p_top: np.ndarray          # float32
    p_set: np.ndarray          # float32, posterior mass on the emitted set
    n_eff: np.ndarray          # float32, effective independent annotators
    set_size: np.ndarray       # int8

    def __len__(self) -> int:
        return int(self.index.size)


def footprint_grid(gdf: gpd.GeoDataFrame, config: WudaptConfig, *,
                   res_m: float | None = None, halo_m: float = 100.0) -> ConsensusGrid:
    """Grid covering the *labelled footprint*, snapped like ``export.raster_grid``.

    Deliberately the label footprint rather than the full GUPPD bbox: WUDAPT
    covers a small fraction of most city boxes, and a full-bbox grid wastes
    memory (and, later, mosaic disk) on ground that carries no label. Snapping to
    resolution multiples keeps it deterministic and alignable.
    """
    res = float(res_m if res_m is not None else config.labels.export.raster_res_m)
    bbox4326 = tuple(gdf.to_crs("EPSG:4326").total_bounds)
    utm = local_utm_crs(bbox4326, None)
    minx, miny, maxx, maxy = gdf.to_crs(utm).total_bounds
    minx, miny = minx - halo_m, miny - halo_m
    maxx, maxy = maxx + halo_m, maxy + halo_m
    minx = np.floor(minx / res) * res
    miny = np.floor(miny / res) * res
    maxx = np.ceil(maxx / res) * res
    maxy = np.ceil(maxy / res) * res
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    return ConsensusGrid(from_origin(minx, maxy, res, res), utm, (height, width), res)


def _accumulate(gdf: gpd.GeoDataFrame, weights: np.ndarray, grid: ConsensusGrid,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate per-class weight, total weight and sum of squared weights.

    One rasterisation pass *per annotator*, because a vote is per annotator: an
    author who drew the same pixel in five submissions must contribute once, not
    five times. Within an annotator the burn order (oldest first) means the
    newest revision wins on self-overlap.
    """
    h, w = grid.shape
    S = np.zeros((N_LCZ, h, w), dtype=np.float32)
    W1 = np.zeros((h, w), dtype=np.float32)
    W2 = np.zeros((h, w), dtype=np.float32)

    g = gdf.to_crs(grid.crs)
    kw = dict(out_shape=(h, w), transform=grid.transform, fill=0, all_touched=False)

    for _, idx in g.groupby(g["annotator_id"], sort=False).indices.items():
        sub = g.iloc[idx]
        sub_w = weights[idx]
        keep = sub_w > 0
        if not keep.any():
            continue
        sub, sub_w = sub.iloc[keep], sub_w[keep]

        # Painter's algorithm: last write wins, so the class and weight planes
        # stay consistent with each other.
        cls_plane = rasterize(zip(sub.geometry.values, sub["class"].astype("uint8")),
                              dtype="uint8", **kw)
        w_plane = rasterize(zip(sub.geometry.values, sub_w.astype("float32")),
                            dtype="float32", **kw)
        ti = np.flatnonzero(cls_plane.ravel())
        if ti.size == 0:
            continue
        cv = cls_plane.ravel()[ti].astype(np.intp) - 1
        wv = w_plane.ravel()[ti]
        # Within one annotator every pixel carries exactly one class, so the
        # (class, pixel) index pairs are unique and fancy-index += is exact
        # (no duplicate-index accumulation hazard, and far faster than a
        # per-class full-grid mask loop).
        S.reshape(N_LCZ, -1)[cv, ti] += wv
        W1.ravel()[ti] += wv
        W2.ravel()[ti] += wv * wv
    return S, W1, W2


def _decide(S: np.ndarray, W1: np.ndarray, W2: np.ndarray, config: WudaptConfig,
            grid: ConsensusGrid) -> ConsensusResult:
    """Posterior -> (bitmask, confidence) over the labelled pixels only."""
    c = config.consensus
    h, w = grid.shape
    flat_valid = np.flatnonzero(W1.ravel() > 0)
    if flat_valid.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return ConsensusResult(
            grid, empty_i, np.zeros(0, "uint32"), np.zeros(0, "float32"),
            np.zeros(0, "int16"), np.zeros(0, "float32"), np.zeros(0, "float32"),
            np.zeros(0, "float32"), np.zeros(0, "int8"),
        )

    Sv = S.reshape(N_LCZ, -1)[:, flat_valid]                      # (17, n) float32
    tot = Sv.sum(axis=0, dtype=np.float32)

    # Dirichlet prior: one virtual annotator drawn from the AOI's own class mix,
    # so a single annotator's posterior is finite instead of exactly 1.0.
    prior = Sv.sum(axis=1, dtype=np.float64)
    prior = prior / prior.sum() if prior.sum() > 0 else np.full(N_LCZ, 1.0 / N_LCZ)
    # Prior strength in units of one average annotator of this AOI (see
    # ConsensusParams.prior_alpha). `tot` is the TOTAL weight on a pixel, summed
    # over everyone who labelled it, so it is not one annotator's weight wherever
    # annotators overlap — and overlap is the norm (Wuhan covers its own area
    # 6x). Dividing by n_eff recovers the per-annotator scale, so the prior stays
    # one opinion instead of growing with the evidence it is meant to temper.
    w2 = W2.ravel()[flat_valid]
    n_eff = np.divide(tot ** 2, w2, out=np.zeros_like(tot), where=w2 > 0)
    per_annotator = np.divide(tot, n_eff, out=np.zeros_like(tot), where=n_eff > 0)
    mean_w = float(np.mean(per_annotator[per_annotator > 0])) if np.any(per_annotator > 0) else 1.0
    alpha = np.float32(c.prior_alpha * max(mean_w, 1e-6))
    P = (Sv + alpha * prior[:, None].astype(np.float32)) / (tot + alpha)

    # Only the top (max_set_size + 1) classes can affect the decision: the set
    # never exceeds max_set_size, and one more column is enough to detect that
    # the mass threshold was NOT reached within it. Partial selection instead of
    # a full 17-way argsort is ~10x faster on the ~10^6-pixel grids here.
    k = int(min(c.max_set_size + 1, N_LCZ))
    part = np.argpartition(-P, k - 1, axis=0)[:k]
    vals = np.take_along_axis(P, part, axis=0)
    rank_k = np.argsort(-vals, axis=0)
    order = np.take_along_axis(part, rank_k, axis=0)              # (k, n) best first
    P_sorted = np.take_along_axis(vals, rank_k, axis=0)
    cum = np.cumsum(P_sorted, axis=0, dtype=np.float32)
    # Smallest j with cumulative mass >= tau_mass, capped at k (which is how a
    # "too diffuse" pixel is detected below).
    set_size = (cum < c.tau_mass).sum(axis=0) + 1
    set_size = np.minimum(set_size, k).astype(np.int64)

    rank = np.arange(k)[:, None]
    in_set = rank < set_size[None, :]
    bits = np.where(in_set, np.left_shift(np.uint32(1), order.astype(np.uint32)),
                    np.uint32(0)).astype(np.uint32)
    bitmask = np.bitwise_or.reduce(bits, axis=0)

    p_top = P_sorted[0]
    p_set = np.take_along_axis(cum, (set_size - 1)[None, :], axis=0)[0]
    # Reject: too diffuse to be a usable set, or no class stands out at all.
    reject = (set_size > c.max_set_size) | (p_top < c.min_p_top)
    bitmask[reject] = 0

    depth = np.clip(c.depth_base + c.depth_per_annotator * n_eff, 0.0, 1.0)
    conf = np.clip(p_set * depth, 0.0, 1.0)
    conf[reject] = 0.0

    top_class = (order[0] + 1).astype(np.int16)
    top_class[reject] = 0

    keep = bitmask > 0
    return ConsensusResult(
        grid,
        flat_valid[keep],
        bitmask[keep],
        conf[keep].astype(np.float32),
        top_class[keep],
        p_top[keep].astype(np.float32),
        p_set[keep].astype(np.float32),
        n_eff[keep].astype(np.float32),
        set_size[keep].astype(np.int8),
    )


def consensus_for_aoi(gdf: gpd.GeoDataFrame, weights: np.ndarray, config: WudaptConfig,
                      *, res_m: float | None = None, grid: ConsensusGrid | None = None,
                      max_pixels: int = 200_000_000) -> ConsensusResult:
    """Full H3 pass for one AOI's gated, weighted polygons.

    Pass ``grid`` to pin the raster instead of deriving it from this frame's own
    footprint. Callers that compare several subsets of one AOI — leave-one-author
    -out especially — MUST do this: ``ConsensusResult.index`` holds flat indices
    into its own grid, so results computed on differently-derived grids are not
    comparable and cross-indexing them silently reads the wrong pixels.

    ``max_pixels`` guards the ``(17, H, W) float32`` accumulator; grids beyond it
    are coarsened rather than allowed to exhaust memory, and the coarsening is
    logged so an audit number is never silently computed at a different scale.
    """
    if len(gdf) == 0:
        raise ValueError("consensus_for_aoi got an empty frame")
    gdf = burn_order(gdf.assign(_w=weights))
    weights = gdf.pop("_w").to_numpy(dtype="float64")

    if grid is not None:
        S, W1, W2 = _accumulate(gdf, weights, grid)
        return _decide(S, W1, W2, config, grid)

    grid = footprint_grid(gdf, config, res_m=res_m)
    if grid.n_pixels > max_pixels:
        factor = float(np.ceil(np.sqrt(grid.n_pixels / max_pixels)))
        coarse = grid.res_m * factor
        logger.warning(f"grid {grid.shape} exceeds {max_pixels:,} px; coarsening "
                       f"{grid.res_m:g} m -> {coarse:g} m")
        grid = footprint_grid(gdf, config, res_m=coarse)

    S, W1, W2 = _accumulate(gdf, weights, grid)
    res = _decide(S, W1, W2, config, grid)
    if len(res):
        hard = int((res.set_size == 1).sum())
        logger.info(
            f"consensus: {len(res):,} labelled px on {grid.shape} @ {grid.res_m:g} m "
            f"| hard {100 * hard / len(res):.1f}% | mean conf {res.confidence.mean():.3f} "
            f"| mean n_eff {res.n_eff.mean():.2f}"
        )
    return res
