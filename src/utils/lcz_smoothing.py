"""Resolution-coarsening and denoising filters for finished LCZ prediction maps.

Both filters operate on an already-classified (argmaxed) hard-label raster —
1-indexed LCZ classes 1-17, ``nodata`` (default 0) elsewhere — not on raw model
probabilities. ``infer_roi.py`` never carries softmax past the sliding-window
argmax, and Demuzere et al. 2020 (*Sci Data*, "A global map of local climate
zones") define their Gaussian-likelihood smoothing the same way: on the
classified map's per-class binary membership masks, not on raw probabilities.

``majority_pool`` is a plain block-mode filter (same algorithm as
``training.evaluate._mode_pool``, ported to numpy with this module's
1-indexed/``nodata=0`` convention rather than that function's torch/batched/
``-1``-ignore-indexed one). ``gaussian_likelihood_filter`` is the Demuzere-style
per-class Gaussian-weighted vote, which additionally anti-alias-smooths before
any resolution change instead of just mode-pooling already-hard labels.
"""

from __future__ import annotations

import numpy as np

# Per-class Gaussian sigma (metres), from Demuzere et al. 2020's stated groups:
# LCZ1=100, LCZ2-6=150, LCZ8/10=250, water(17)=25, other natural(11-16)=75.
# Classes 7 (Lightweight Low-Rise) and 9 (Sparsely Built) aren't named
# individually in the paper (it groups "other urban classes 2-6" at 150m and
# singles out 1/8/10) -- folded into the 150m bucket as the nearest reasonable
# default. Override via a single global sigma if this assumption is wrong for
# a given use case.
DEFAULT_SIGMA_BY_CLASS: dict[int, float] = {
    1: 100.0,
    2: 150.0, 3: 150.0, 4: 150.0, 5: 150.0, 6: 150.0, 7: 150.0, 9: 150.0,
    8: 250.0, 10: 250.0,
    11: 75.0, 12: 75.0, 13: 75.0, 14: 75.0, 15: 75.0, 16: 75.0,
    17: 25.0,
}


def repair_seams(
    raster: np.ndarray,
    conf_raster: np.ndarray | None = None,
    max_dist_px: int = 2,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fill thin nodata seams left by independently-reprojected adjacent tiles.

    Adjacent source tiles (e.g. different Tessera UTM zones) are reprojected
    independently (nearest-neighbour) into a shared output grid, which can
    leave a thin band of destination pixels unclaimed by either tile right at
    the boundary. Filled via nearest-VALID-pixel lookup -- categorical-safe,
    unlike ``rasterio.fill.fillnodata``'s IDW interpolation, which would
    average integer class labels into meaningless values.

    Thresholding raw per-pixel distance-to-valid at ``max_dist_px`` is NOT
    enough to tell a thin seam apart from a genuinely large nodata region (no
    tile covers this area at all): every region's own *outer rim* is close to
    valid data by definition, regardless of how large the region is, so a
    naive per-pixel cutoff would erode that rim on every nodata blob, not
    just close true seams. Instead, nodata pixels are grouped into connected
    components, and a whole component is only filled when its OWN maximum
    distance-to-valid (i.e. the distance at its deepest interior point) is
    ``<= max_dist_px`` -- meaning the component itself is narrow everywhere,
    not merely that its edge happens to be near valid data. A genuinely large
    region is left untouched in full, boundary included.
    """
    from scipy import ndimage

    nodata = raster == 0
    if not nodata.any():
        return raster, conf_raster
    dist, (iy, ix) = ndimage.distance_transform_edt(
        nodata, return_distances=True, return_indices=True
    )
    labeled, n_components = ndimage.label(nodata)
    if n_components == 0:
        return raster, conf_raster
    component_ids = np.arange(1, n_components + 1)
    max_dist_per_component = ndimage.maximum(dist, labeled, index=component_ids)
    thin_ids = component_ids[max_dist_per_component <= max_dist_px]
    if thin_ids.size == 0:
        return raster, conf_raster
    fill = np.isin(labeled, thin_ids)
    raster = raster.copy()
    raster[fill] = raster[iy[fill], ix[fill]]
    if conf_raster is not None:
        conf_raster = conf_raster.copy()
        conf_raster[fill] = conf_raster[iy[fill], ix[fill]]
    return raster, conf_raster


def _block_average(x: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool the last two axes of ``x`` over non-overlapping blocks.

    Trailing rows/cols that don't fill a block are dropped, matching
    ``_mode_pool``'s truncation convention (training/evaluate.py:185).
    """
    if factor <= 1:
        return x
    *lead, H, W = x.shape
    Hc, Wc = (H // factor) * factor, (W // factor) * factor
    cropped = x[..., :Hc, :Wc]
    reshaped = cropped.reshape(*lead, Hc // factor, factor, Wc // factor, factor)
    return reshaped.mean(axis=(-3, -1))


def majority_pool(
    labels: np.ndarray,
    factor: int,
    nodata: int = 0,
    num_classes: int = 17,
) -> np.ndarray:
    """Block-mode (majority vote) pooling over non-overlapping ``factor`` blocks.

    A block that is entirely ``nodata`` stays ``nodata``; otherwise the
    majority is taken among the block's non-nodata pixels only. Trailing
    rows/cols that don't fill a block are dropped.
    """
    if factor <= 1:
        return labels.copy()
    H, W = labels.shape
    Hc, Wc = (H // factor) * factor, (W // factor) * factor
    cropped = labels[:Hc, :Wc]
    blocks = cropped.reshape(Hc // factor, factor, Wc // factor, factor)
    counts = np.zeros((Hc // factor, Wc // factor, num_classes), dtype=np.int32)
    for c in range(1, num_classes + 1):
        counts[..., c - 1] = (blocks == c).sum(axis=(1, 3))
    pooled = (counts.argmax(axis=-1) + 1).astype(labels.dtype)
    pooled[counts.sum(axis=-1) == 0] = nodata
    return pooled


def gaussian_likelihood_filter(
    labels: np.ndarray,
    native_res_m: float,
    out_res_m: float | None = None,
    sigma_by_class: dict[int, float] | float = DEFAULT_SIGMA_BY_CLASS,
    nodata: int = 0,
    min_coverage: float = 0.5,
    num_classes: int = 17,
) -> np.ndarray:
    """Per-class Gaussian-likelihood smoothing filter (Demuzere et al. 2020).

    For each class, convolves its binary membership mask with a Gaussian
    kernel (class-specific sigma, converted from metres to pixels via
    ``native_res_m``) to get a likelihood surface, then argmaxes per pixel
    over all classes' likelihoods. ``nodata`` pixels contribute to no class's
    mask, so they never win a vote on their own -- but neighbouring valid
    pixels' kernels can still reach across them.

    If ``out_res_m`` is coarser than ``native_res_m``, the per-class
    likelihood volume is block-*averaged* down to the coarse grid (a proper
    anti-aliased decimation of a continuous field) before the final argmax,
    rather than mode-pooling already-hard labels. ``out_res_m=None`` (or equal
    to ``native_res_m``) denoises at native resolution only, matching the
    paper's own use of this filter exactly.

    ``min_coverage``: output cells whose fraction of real (non-nodata) input
    pixels falls below this are set to ``nodata`` rather than assigned an
    arbitrary near-zero-confidence class -- the same ``weight_sum > 0`` idiom
    ``_sliding_window_seg`` already uses for its own blending.
    """
    from scipy import ndimage

    if out_res_m is not None and out_res_m < native_res_m:
        raise ValueError(
            f"out_res_m ({out_res_m}) must be >= native_res_m ({native_res_m}) "
            "-- this filter coarsens, it does not upsample."
        )
    if isinstance(sigma_by_class, (int, float)):
        sigma_by_class = {c: float(sigma_by_class) for c in range(1, num_classes + 1)}

    H, W = labels.shape
    valid = labels != nodata
    likelihood = np.zeros((num_classes, H, W), dtype=np.float32)
    for c in range(1, num_classes + 1):
        mask = (labels == c).astype(np.float32)
        sigma_px = sigma_by_class.get(c, 100.0) / native_res_m
        likelihood[c - 1] = ndimage.gaussian_filter(
            mask, sigma=sigma_px, mode="constant", cval=0.0
        )

    factor = max(1, round(out_res_m / native_res_m)) if out_res_m else 1
    if factor > 1:
        likelihood = _block_average(likelihood, factor)
        coverage = _block_average(valid.astype(np.float32), factor)
    else:
        coverage = valid.astype(np.float32)

    pooled = (likelihood.argmax(axis=0) + 1).astype(np.uint8)
    pooled[coverage < min_coverage] = nodata
    return pooled
