"""Seamless ROI inference from raw source embedding tiles.

Instead of using pre-extracted grid npy files, this script:
1. Finds source embedding tiles (zarr/tif) that cover the requested bbox.
2. For each tile, clips the embedding to the ROI (+ an optional margin for context).
3. Runs the model with a sliding window — Hanning-weighted logit blending
   (seg) or softmax accumulation over overlapping patches (cls) — keeping the
   per-pixel PROBABILITY volume rather than argmaxing it away.
4. Sums that volume, and its validity mask, onto the output grid with
   rasterio.warp.reproject(Resampling.sum), so an output cell is a genuine
   pixel-count-weighted pool of the fine predictions under it — across tile
   boundaries as well as within a cell.
5. Divides, argmaxes, and saves a GeoTIFF + PNG at ``--target-res``, plus the
   native-resolution map alongside.

This eliminates all grid-tile artifacts and UTM-zone boundary artefacts.

Resolution is the point of --target-res: LCZ is a ~100 m urban-climate concept.
A classification model predicts one class per 320 m patch (the So2Sat unit) and
a segmentation model one per 10 m pixel, and neither is the resolution the map
should be read at. --target-res sets that independently of the model's own
sampling, and --aggregate chooses how the fine predictions are pooled into it
(soft probability average / majority vote / Demuzere-style Gaussian).

Supports:
- Every model family in the models registry: segmentation families (unet,
  resnet_unet, fcn8, attention_unet) run per-pixel with Hanning-blended logits;
  classification families (resnet, mlp, aspp, vit, shallow_cnn, linear_probe,
  ...) run patch-wise with soft-voted softmax.
- Legacy linear-probe checkpoints from the retired linear_probe.py script
  (fc-only state dict): pass ``--stats-file`` with the training-set mean/std
  NPZ and they are converted on load.
- All embedding types in datasets.registry (tessera, tesserav1.1,
  alpha_earth, alpha_earth_coop, seamless, ...)

Example (segmentation):
    python src/infer_roi.py \\
        --model-type unet --preset small \\
        --checkpoint /path/to/unet-small-best.pt \\
        --embedding-name alpha_earth_coop \\
        --embedding-dir /maps/.../coop \\
        --year 2017 \\
        --bbox "-0.51,51.28,0.33,51.69" \\
        --output /maps/.../London_seg.tif \\
        --num-classes 17 --patch-size 64 --overlap 32
    # -> London_seg.tif @ 100 m (primary) + London_seg_10m.tif (native)

Example (classification):
    python src/infer_roi.py \\
        --model-type resnet --preset base \\
        --checkpoint /path/to/resnet-base-best.pt \\
        --embedding-name alpha_earth_coop \\
        --embedding-dir /maps/.../coop \\
        --year 2017 \\
        --bbox "-0.51,51.28,0.33,51.69" \\
        --output /maps/.../London_cls.tif \\
        --num-classes 17 --patch-size 32 --target-res 100
    # -> London_cls.tif @ 100 m, voting among the overlapping 320 m patches
    #    (omit --target-res for the default one-cell-per-patch 320 m map)
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from datasets.registry import EMBEDDING_REGISTRY, get_nodata_predicate
from datasets.tiles import build_coop_valid_bbox_map, build_tile_index, open_tile
from utils.lcz_smoothing import (
    DEFAULT_SIGMA_BY_CLASS,
    repair_seams,
    smooth_class_volume,
)

# LCZ is a ~100 m urban-climate concept, not a 10 m one: a segmentation model
# predicts per 10 m pixel, but the map that means something is the pooled one.
DEFAULT_SEG_TARGET_RES_M = 100.0

AGGREGATE_METHODS = ("soft", "majority", "gaussian")

# Classes reprojected per warp call. The accumulator is (num_classes, H, W)
# float32 — 1.7 GB for a city-sized 10 m grid — and reprojecting all of it in
# one call needs a second array that size per tile. Chunking bounds the extra
# to this many bands at the cost of a few more warp setups.
REPROJECT_BAND_CHUNK = 4


def _is_segmentation(model_type: str) -> bool:
    """Whether a model family runs per-pixel segmentation (vs patch cls)."""
    from models import MODEL_REGISTRY
    fam = MODEL_REGISTRY.get(model_type)
    return fam is not None and fam.pipeline == "segmentation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hanning_2d(h: int, w: int, eps: float = 1e-6) -> np.ndarray:
    """2D Hanning taper window of shape (h, w), float32.

    Floored at ``eps`` rather than the raw np.hanning, whose first/last sample
    is exactly 0: _patch_positions guarantees the tile's outer edge is covered
    by exactly one patch, at that patch's own zero-weight border, so an
    unfloored window leaves weight_sum == 0 (and the prediction unset) along
    the full 1px perimeter of every tile. eps is far below the smallest
    legitimate nonzero Hanning sample even at large patch sizes, so it only
    ever matters at that single-covering-patch edge case.
    """
    win_h = np.maximum(np.hanning(h).astype(np.float32), eps)
    win_w = np.maximum(np.hanning(w).astype(np.float32), eps)
    return win_h[:, None] * win_w[None, :]


def _repair_and_log(raster, conf_raster, label: str):
    """``repair_seams`` plus a line saying how much it actually had to fill.

    The count is the diagnostic that matters: thin seams between
    independently-reprojected adjacent tiles are exactly what this is for, so a
    nonzero fill on the summed output grid would mean the coverage-weighted
    accumulation is NOT closing tile boundaries the way it should, and a large
    remaining nodata count means the ROI genuinely reaches past the tiles.
    """
    before = int((raster == 0).sum())
    raster, conf_raster = repair_seams(raster, conf_raster)
    after = int((raster == 0).sum())
    logger.info(
        f"Seam repair ({label}): filled {before - after} px; "
        f"{after} nodata px remain ({after / raster.size:.3%})"
    )
    return raster, conf_raster


# These geometry helpers moved to datasets.tiles so that torch-free tools
# (src/embedding_rgb.py) can use them without importing this module, which
# pulls in torch. Re-exported here for existing callers.
from datasets.tiles import (  # noqa: E402
    open_and_clip as _open_and_clip,
    setup_output as _setup_output,
    meters_to_out_res as _meters_to_out_res,
)


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------

def _patch_positions(H: int, W: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """All (row, col) top-left corners for a sliding window."""
    rows = list(range(0, max(1, H - patch_size + 1), stride))
    cols = list(range(0, max(1, W - patch_size + 1), stride))
    if not rows or rows[-1] + patch_size < H:
        rows.append(max(0, H - patch_size))
    if not cols or cols[-1] + patch_size < W:
        cols.append(max(0, W - patch_size))
    return [(r, c) for r in rows for c in cols]


def _sliding_window_seg(
    model: nn.Module,
    arr: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device,
    num_classes: int,
    batch_size: int,
) -> np.ndarray:
    """Run segmentation model with Hanning-blended sliding window.

    Args:
        model: U-Net (takes (B, C, H, W) → (B, num_classes, H, W)).
        arr: (C, H, W) float32 embedding.

    Returns:
        (num_classes, H, W) float32 per-pixel softmax probabilities. Pixels no
        window covered are left all-zero, which is how callers tell them apart
        from a genuine prediction (a real softmax column sums to 1) — the
        volume doubles as its own validity mask.

        Logits are Hanning-blended first and softmaxed once at the end, not
        the other way round: averaging probabilities across overlapping
        windows would apply the taper after the nonlinearity and lose the
        blend's intended meaning.
    """
    C, H, W = arr.shape
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, H_pad, W_pad = arr.shape

    logit_sum = np.zeros((num_classes, H_pad, W_pad), dtype=np.float32)
    weight_sum = np.zeros((H_pad, W_pad), dtype=np.float32)
    hann = _hanning_2d(patch_size, patch_size)

    positions = _patch_positions(H_pad, W_pad, patch_size, stride)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_pos = positions[i:i + batch_size]
            batch = np.stack([arr[:, r:r + patch_size, c:c + patch_size] for r, c in batch_pos])
            logits = model(torch.from_numpy(batch).to(device)).cpu().numpy()
            for (r, c), patch_logits in zip(batch_pos, logits):
                logit_sum[:, r:r + patch_size, c:c + patch_size] += patch_logits * hann[None]
                weight_sum[r:r + patch_size, c:c + patch_size] += hann

    probs = np.zeros((num_classes, H_pad, W_pad), dtype=np.float32)
    valid = weight_sum > 0
    if valid.any():
        blended = logit_sum[:, valid] / weight_sum[None, valid]
        blended -= blended.max(axis=0, keepdims=True)   # softmax, stably
        np.exp(blended, out=blended)
        probs[:, valid] = blended / blended.sum(axis=0, keepdims=True)

    return probs[:, :H, :W]


def _sliding_window_cls(
    model: nn.Module,
    arr: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device,
    num_classes: int,
    batch_size: int,
    extract_size: int | None = None,
    model_input_size: int | None = None,
) -> np.ndarray:
    """Run classification model with softmax-voting sliding window.

    Each patch's softmax probabilities are accumulated over its full footprint
    and the accumulation is renormalised per pixel. With overlapping patches
    (stride < extract_size) this soft-votes among all patches covering a pixel;
    with non-overlapping patches it reduces to the plain per-patch softmax.

    Args:
        model: ResNet (takes (B, C, H, W) → (B, num_classes)).
        arr: (C, H, W) float32 embedding.
        extract_size: Embedding pixels to extract per patch (default: patch_size).
            Set to round(patch_physical_res_m / embedding_res_m) so each patch
            covers the same physical area as the training patches.
        model_input_size: Model's expected square input side (default: patch_size).
            Extracted patches are bilinearly resized to this size when it differs
            from extract_size.

    Returns:
        (num_classes, H, W) float32 soft-voted probabilities, renormalised so
        each covered pixel sums to 1. Uncovered pixels are left all-zero and
        so act as their own validity mask, matching ``_sliding_window_seg``.
    """
    import torch.nn.functional as F

    extract_size = extract_size or patch_size
    model_input_size = model_input_size or patch_size

    C, H, W = arr.shape
    pad_h = max(0, extract_size - H)
    pad_w = max(0, extract_size - W)
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, H_pad, W_pad = arr.shape

    prob_sum = np.zeros((num_classes, H_pad, W_pad), dtype=np.float32)

    positions = _patch_positions(H_pad, W_pad, extract_size, stride)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_pos = positions[i:i + batch_size]
            patches = [arr[:, r:r + extract_size, c:c + extract_size] for r, c in batch_pos]
            batch = torch.from_numpy(np.stack(patches)).to(device)
            if extract_size != model_input_size:
                batch = F.interpolate(
                    batch, size=(model_input_size, model_input_size),
                    mode="bilinear", align_corners=False,
                )
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
            for (r, c), p in zip(batch_pos, probs):
                prob_sum[:, r:r + extract_size, c:c + extract_size] += p[:, None, None]

    total = prob_sum.sum(axis=0, keepdims=True)
    probs = np.zeros_like(prob_sum)
    np.divide(prob_sum, total, out=probs, where=total > 0)
    return probs[:, :H, :W]


# ---------------------------------------------------------------------------
# Per-tile probability source
# ---------------------------------------------------------------------------

class TileProbSource:
    """Per-source-tile probability volumes for an ROI, with their geometry.

    Owns everything between "which tiles cover this bbox" and "here is a
    ``(num_classes, H, W)`` softmax volume for one of them, plus the CRS and
    affine that place it": the tile index, the coop valid-bbox map, the clip,
    the sliding window, and the classification stride/extract arithmetic that
    depends on the embedding's own pixel size.

    Split out of ``infer_roi`` so that a caller wanting to pool the SAME
    predictions several ways — ``coarsen_bakeoff.py``, sweeping ``aggregate`` ×
    ``target_res`` — pays for one GPU pass rather than one per combination.
    Iterating twice re-runs the model; iterate once and fan out.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        embedding_name: str,
        embedding_dir: Path,
        bbox: tuple[float, float, float, float],
        *,
        year: str | None = None,
        num_classes: int = 17,
        patch_size: int = 64,
        overlap: int = 0,
        batch_size: int = 8,
        device: torch.device | None = None,
        dequantize_fn=None,
        normalize: tuple | None = None,
        margin_m: float = 200.0,
        patch_physical_res_m: float = 320.0,
        patch_physical_stride_m: float | None = None,
        target_res_m: float | None = None,
        roi_geom_4326=None,
    ):
        from shapely.geometry import box

        if device is None:
            from utils.runtime import resolve_device
            device = resolve_device("auto")

        self.model = model
        self.bbox = bbox
        self.device = device
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.margin_m = margin_m
        self.dequantize_fn = dequantize_fn
        self.normalize = normalize

        self.is_seg = _is_segmentation(model_type)
        self.stride = patch_size - overlap
        if self.stride <= 0:
            raise ValueError(f"overlap ({overlap}) must be < patch_size ({patch_size})")

        # ── Tile spatial index ───────────────────────────────────────────────
        tile_paths, tree = build_tile_index(embedding_dir, embedding_name, year=year)
        idxs = tree.query(box(*bbox))
        if roi_geom_4326 is not None:
            # Keep only tiles touching the ROI geometry itself: a bbox around
            # scattered cells can cover several times their area.
            idxs = [i for i in idxs if tree.geometries[i].intersects(roi_geom_4326)]
        if len(idxs) == 0:
            raise RuntimeError(f"No embedding tiles found for bbox {bbox}")
        self.matched_paths = [tile_paths[i] for i in idxs]
        logger.info(f"Found {len(self.matched_paths)} tile(s) intersecting the ROI")

        # For coop tiles, intersecting with the reported valid bounds in
        # _open_and_clip discards data overshooting the tile's UTM zone.
        self.path_to_valid_bbox: dict[Path, tuple[float, float, float, float]] = {}
        if embedding_name == "alpha_earth_coop":
            self.path_to_valid_bbox = build_coop_valid_bbox_map(
                embedding_dir, year, self.matched_paths
            )

        # The valid bbox above is a lon/lat rectangle, but the zone edge it
        # stands for is a meridian, which runs diagonally across a UTM tile: the
        # clip still keeps a wedge of the neighbouring zone, filled with the
        # -128 sentinel. Dequantized and normalised, that fill looks like real
        # data and wins the vote over the other zone's clean predictions (a
        # visible sliver along 0 deg over London). So sentinel pixels are masked
        # out of every tile's probability volume, whatever the embedding.
        self.nodata_predicate = (get_nodata_predicate(embedding_name)
                                 if EMBEDDING_REGISTRY.get(embedding_name, {})
                                 .get("nodata_all_channels_eq") is not None else None)

        # ── Probe the first valid tile for CRS + pixel size ──────────────────
        first_result = None
        for path in self.matched_paths:
            r = _open_and_clip(path, bbox, margin_m=0.0, dequantize_fn=None,
                               valid_bbox_4326=self.path_to_valid_bbox.get(path))
            if r is not None:
                first_result = r
                break
        if first_result is None:
            raise RuntimeError("No valid data in any matched tile")

        _, self.first_crs, self.first_transform = first_result
        self.embedding_res_m = abs(self.first_transform.a)

        # ── Classification stride / extract window ───────────────────────────
        # `default_target_m` is the output cell size the caller gets if it does
        # not ask for one; the stride is what the model actually slides by, and
        # the two are deliberately separable.
        self.extract_px = self.cls_stride = None
        if not self.is_seg:
            # The stride follows target_res_m unless set explicitly, so asking
            # for a 100 m map really does slide 100 m and vote, rather than
            # resampling a 320 m blocky field onto a 100 m grid.
            stride_m = patch_physical_stride_m or target_res_m or patch_physical_res_m
            if not 0 < stride_m <= patch_physical_res_m:
                raise ValueError(
                    f"patch_physical_stride_m ({stride_m}) must be in "
                    f"(0, patch_physical_res_m={patch_physical_res_m}] — a larger "
                    "stride would leave uncovered pixels."
                )
            self.extract_px = max(1, round(patch_physical_res_m / self.embedding_res_m))
            self.cls_stride = max(1, round(stride_m / self.embedding_res_m))
            self.default_target_m = self.cls_stride * self.embedding_res_m
        else:
            self.default_target_m = DEFAULT_SEG_TARGET_RES_M

        self.n_done = self.n_skip = 0

    def __len__(self) -> int:
        return len(self.matched_paths)

    def __iter__(self):
        """Yield ``(probs, tile_crs, tile_transform)`` per covering tile.

        ``probs`` is ``(num_classes, H, W)`` float32; pixels no window covered
        are all-zero, so ``probs.sum(axis=0) > 0`` is the validity mask.
        """
        self.n_done = self.n_skip = 0
        for i, tile_path in enumerate(self.matched_paths):
            logger.info(f"Tile {i + 1}/{len(self.matched_paths)}: {tile_path.name}")

            result = _open_and_clip(
                tile_path, self.bbox, margin_m=self.margin_m,
                dequantize_fn=self.dequantize_fn, normalize=self.normalize,
                valid_bbox_4326=self.path_to_valid_bbox.get(tile_path),
            )
            if result is None:
                logger.warning("  Skipped — no valid data after clip")
                self.n_skip += 1
                continue

            arr, tile_crs, tile_transform = result
            C, H_tile, W_tile = arr.shape
            logger.info(f"  {C}ch × {H_tile}×{W_tile} px  crs={tile_crs}")
            if H_tile < 1 or W_tile < 1:
                self.n_skip += 1
                continue

            invalid = None
            if self.nodata_predicate is not None:
                raw = _open_and_clip(
                    tile_path, self.bbox, margin_m=self.margin_m,
                    valid_bbox_4326=self.path_to_valid_bbox.get(tile_path),
                )
                invalid = self.nodata_predicate(raw[0])
                if invalid.shape != arr.shape[1:] or not invalid.any():
                    invalid = None
                else:
                    # Windows reaching into the fill would still see it, and the
                    # pixels beside the edge would inherit their vote. Extend the
                    # real data into the fill instead (nearest valid pixel, the
                    # same idea as the reflect padding at a tile edge); the fill
                    # pixels' own predictions are zeroed below.
                    from scipy.ndimage import distance_transform_edt
                    iy, ix = distance_transform_edt(invalid, return_distances=False,
                                                    return_indices=True)
                    arr = arr[:, iy, ix]
                    del iy, ix

            if self.is_seg:
                probs = _sliding_window_seg(
                    self.model, arr, self.patch_size, self.stride, self.device,
                    self.num_classes, self.batch_size,
                )
            else:
                probs = _sliding_window_cls(
                    self.model, arr, self.patch_size, self.cls_stride, self.device,
                    self.num_classes, self.batch_size,
                    extract_size=self.extract_px, model_input_size=self.patch_size,
                )
            if invalid is not None:
                probs[:, invalid] = 0.0     # all-zero = "no prediction here"
                logger.info(f"  masked {invalid.mean():.1%} nodata px")
            self.n_done += 1
            yield probs, tile_crs, tile_transform

        logger.info(f"Tiles processed: {self.n_done} done, {self.n_skip} skipped")


def build_aggregate_volume(
    probs: np.ndarray,
    aggregate: str,
    native_res_m: float,
    sigma=None,
) -> np.ndarray:
    """The per-class volume that gets summed onto the output grid.

    The three ``--aggregate`` modes differ ONLY here; everything downstream —
    the ``Resampling.sum`` accumulation, the coverage test, the argmax — is
    shared, which is why they are directly comparable.

    ``soft``      the probabilities themselves: a confidence-weighted vote.
    ``majority``  a one-hot of the fine argmax, so summing counts votes and the
                  cell's winner is its plain mode — the same operator
                  ``training.evaluate._mode_pool`` uses for the 100 m metrics,
                  with confidence deliberately discarded.
    ``gaussian``  Demuzere et al. 2020's per-class kernel, applied to the real
                  probabilities rather than to one-hotted hard labels.
    """
    if aggregate not in AGGREGATE_METHODS:
        raise ValueError(
            f"Unknown aggregate {aggregate!r} — choose from {sorted(AGGREGATE_METHODS)}"
        )
    if aggregate == "soft":
        return probs

    valid = probs.sum(axis=0) > 0
    if aggregate == "majority":
        vol = (np.arange(probs.shape[0])[:, None, None]
               == probs.argmax(axis=0)[None]).astype(np.float32)
    else:
        vol = smooth_class_volume(
            probs, native_res_m, sigma if sigma is not None else DEFAULT_SIGMA_BY_CLASS
        )
    vol *= valid  # the kernel leaks mass into pixels no window covered
    return vol


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def infer_roi(
    model: nn.Module,
    model_type: str,
    embedding_name: str,
    embedding_dir: Path,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    num_classes: int = 17,
    patch_size: int = 64,
    overlap: int = 0,
    batch_size: int = 8,
    device: torch.device | None = None,
    dequantize_fn=None,
    normalize: tuple | None = None,
    out_crs: str | None = None,
    out_res: float | None = None,
    year: str | None = None,
    city_name: str = "ROI",
    title: str | None = None,
    margin_m: float = 200.0,
    patch_physical_res_m: float = 320.0,
    patch_physical_stride_m: float | None = None,
    save_confidence: bool = False,
    target_res_m: float | None = None,
    aggregate: str = "soft",
    write_native: bool = True,
    min_coverage: float = 0.5,
    coarsen_to_m: float | None = None,
    coarsen_method: str = "gaussian",
    gaussian_sigma: float | None = None,
    roi_geom_4326=None,
) -> Path:
    """Run model inference over a bbox directly from raw source embedding tiles.

    Bypasses pre-extracted grid npy files entirely.  Tiles are clipped on-the-fly,
    the model runs as a sliding window over each clipped tile, and results are
    assembled via rasterio.warp.reproject so UTM zone boundaries produce no artefacts.

    Args:
        model: Loaded nn.Module, already on ``device``.
        model_type: Model family name from the models registry. Segmentation
            families run per-pixel; everything else runs patch classification.
        embedding_name: Any key in ``datasets.registry.EMBEDDING_REGISTRY``
            (e.g. ``"tesserav1.1"``, ``"alpha_earth_coop"``, ``"seamless"``).
        embedding_dir: Directory containing source tile files.
        bbox: ``(west, south, east, north)`` in EPSG:4326.
        output_path: Output GeoTIFF path (PNG is saved alongside).
        num_classes: Number of LCZ classes (default 17).
        patch_size: Sliding-window patch side in pixels.
        overlap: Overlap between adjacent patches in pixels (0 = no overlap).
        batch_size: GPU batch size for the sliding window.
        device: Torch device; auto-selected if None.
        dequantize_fn: Optional per-tile dequantize function (see
            ``utils.runtime.resolve_dequantize`` — coop int8 / seamless ESD).
        out_crs: Output CRS (auto-detected from first tile if None).
        out_res: Output pixel size in ``out_crs`` units (auto-detected if None).
        year: Year string, required for ``"alpha_earth_coop"``.
        city_name: City/area name used in the default PNG title.
        title: Explicit PNG title; overrides the default ``LCZ <MODEL> — <city>``.
        margin_m: Extra metres clipped around the bbox per tile for edge context.
        patch_physical_res_m: Physical side length of one patch in metres (resnet only).
            Determines how many embedding pixels to extract per patch.
            Default 320 m = 32 px × 10 m/px (So2Sat patch size). Ignored
            for unet (segmentation always outputs at embedding resolution).
        patch_physical_stride_m: Distance in metres between patch origins
            (classification only). Defaults to ``target_res_m``, else to
            ``patch_physical_res_m`` (non-overlapping, one prediction per
            patch). Smaller values slide overlapping patches and soft-vote
            their softmax probabilities per pixel — e.g. 100 m votes among the
            ~9 overlapping 320 m patches covering each pixel.
        target_res_m: Resolution in metres of the PRIMARY output map — the grid
            the per-pixel probability volume is pooled onto. Defaults to the
            classification stride (so 320 m by default), and to
            ``DEFAULT_SEG_TARGET_RES_M`` (100 m) for segmentation, whose
            per-pixel 10 m output is far finer than LCZ means anything at.
            Independent of the stride: the stride sets how many patches vote,
            this sets how big an output cell is.
        aggregate: How the fine probability volume is pooled into an output
            cell. ``"soft"`` (default) averages the probabilities — a
            confidence-weighted vote; ``"majority"`` counts fine argmax votes,
            matching ``training.evaluate._mode_pool``'s 100 m metrics;
            ``"gaussian"`` applies Demuzere et al. 2020's per-class Gaussian to
            the probabilities first (see ``gaussian_sigma``).
        write_native: Also write ``<output_stem>_<res>m.tif`` (+PNG) at the
            embedding's own resolution, via the plain per-tile-argmax +
            nearest-neighbour + ``repair_seams`` path. Skipped automatically
            when the primary output already is native resolution.
        min_coverage: An output cell is nodata unless this fraction of it is
            backed by real fine pixels — the same idiom
            ``gaussian_likelihood_filter`` uses, so ROI-edge slivers are not
            handed a near-arbitrary class.
        save_confidence: Also write a float32 sidecar ``<output>_conf.tif``
            with the pooled score of the winning class per cell (0 = nodata) —
            a probability under ``aggregate="soft"``, an area-weighted vote
            fraction under ``"majority"``, and a mean smoothed likelihood —
            NOT a probability, since the kernel loses mass off the tile edge —
            under ``"gaussian"``. Available for both pipelines now that
            segmentation also carries a real softmax volume.
        coarsen_to_m: If given, also write a second, coarser GeoTIFF+PNG
            (``<output_stem>_<res>m_<method>.tif``) at this resolution in
            metres — LCZ is a ~100 m concept, not a 10 m one. Built from the
            finished native-resolution raster via
            ``utils.lcz_smoothing.majority_pool`` or
            ``gaussian_likelihood_filter`` (see ``coarsen_method``); never
            replaces the native-resolution output.
        coarsen_method: ``"gaussian"`` (default, matching Demuzere et al.
            2020's per-class Gaussian-likelihood filter) or ``"majority"``
            (plain block-mode vote, faster and blockier). Only used when
            ``coarsen_to_m`` is given.
        gaussian_sigma: Optional single sigma (metres) overriding the
            per-class default table for the ``"gaussian"`` method.
        roi_geom_4326: Optional shapely geometry (EPSG:4326). Only tiles that
            intersect it are run; the bbox still sets the output grid.

    Returns:
        Path to the saved GeoTIFF.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    source = TileProbSource(
        model, model_type, embedding_name, embedding_dir, bbox,
        year=year, num_classes=num_classes, patch_size=patch_size, overlap=overlap,
        batch_size=batch_size, device=device, dequantize_fn=dequantize_fn,
        normalize=normalize, margin_m=margin_m,
        patch_physical_res_m=patch_physical_res_m,
        patch_physical_stride_m=patch_physical_stride_m,
        target_res_m=target_res_m, roi_geom_4326=roi_geom_4326,
    )
    first_crs = source.first_crs
    resolved_crs = out_crs or first_crs
    embedding_res_m = source.embedding_res_m
    lat_c = (bbox[1] + bbox[3]) / 2
    lon_c = (bbox[0] + bbox[2]) / 2

    # ── Resolutions ───────────────────────────────────────────────────────────
    # The sliding windows produce a per-class probability volume at the
    # EMBEDDING's own resolution. `target_m` is the grid that volume is summed
    # onto, and is independent of the classification stride: the stride sets
    # how many overlapping patches vote for a pixel, the target sets how big an
    # output cell is. Conflating the two is what made `--patch-physical-stride
    # 100` sample one arbitrary 10 m pixel per 100 m cell instead of pooling it.
    target_m = source.default_target_m if target_res_m is None else float(target_res_m)
    if target_m < embedding_res_m:
        logger.warning(
            f"--target-res {target_m} is finer than the embedding's {embedding_res_m} m "
            "— clamped; this coarsens, it does not upsample."
        )
        target_m = embedding_res_m

    def _to_out_units(res_m: float) -> float:
        if resolved_crs == first_crs:
            return res_m
        return _meters_to_out_res(res_m, first_crs, resolved_crs, lon_c, lat_c)

    native_res = _to_out_units(embedding_res_m)
    if out_res is None:
        resolved_res = _to_out_units(target_m)
    else:
        # An explicit --out-res wins outright; report the metric resolution it
        # actually implies rather than the target it overrode, so the PNG title
        # and the log do not disagree with the file.
        resolved_res = out_res
        target_m = embedding_res_m * resolved_res / native_res

    if aggregate not in AGGREGATE_METHODS:
        raise ValueError(
            f"Unknown aggregate {aggregate!r} — choose from {sorted(AGGREGATE_METHODS)}"
        )

    logger.info(
        f"Output CRS: {resolved_crs}, resolution: {resolved_res:.8g} units/px "
        f"(~{target_m:.4g} m), aggregate={aggregate}"
    )

    # ── Output grid + probability accumulators ────────────────────────────────
    out_transform, out_H, out_W = _setup_output(bbox, resolved_crs, resolved_res)
    logger.info(f"Output raster: {out_H}×{out_W} px")

    # Accumulated at the OUTPUT resolution, so this is ~18 MB for a
    # London-sized 100 m grid rather than the ~1.8 GB a 10 m volume would cost.
    prob_accum = np.zeros((num_classes, out_H, out_W), dtype=np.float32)
    weight_accum = np.zeros((out_H, out_W), dtype=np.float32)
    chunk = max(1, min(num_classes, REPROJECT_BAND_CHUNK))
    warp_buf = np.zeros((chunk, out_H, out_W), dtype=np.float32)
    mask_buf = np.zeros((1, out_H, out_W), dtype=np.float32)

    # ── Native-resolution sidecar ─────────────────────────────────────────────
    # The pre-existing nearest-neighbour + last-write-wins + repair_seams path,
    # kept verbatim, so nothing that depended on the fine map loses it and the
    # seam repair stays exercised where seams actually arise. Pointless when
    # the primary output already IS native resolution (e.g. classification at
    # --patch-physical-stride 10, which is what generate_seg_pseudo_rasters.py
    # asks for), so it is skipped there rather than written twice.
    write_native_now = write_native and resolved_res > native_res * 1.01
    nat_raster = nat_conf = None
    if write_native_now:
        nat_transform, nat_H, nat_W = _setup_output(bbox, resolved_crs, native_res)
        nat_raster = np.zeros((nat_H, nat_W), dtype=np.uint8)
        nat_conf = np.zeros((nat_H, nat_W), dtype=np.float32) if save_confidence else None
        logger.info(f"Native sidecar: {nat_H}×{nat_W} px @ {native_res:.8g} units/px")

    # ── Process each tile ─────────────────────────────────────────────────────
    for probs, tile_crs, tile_transform in source:
        valid = probs.sum(axis=0) > 0
        vol = build_aggregate_volume(probs, aggregate, embedding_res_m, gaussian_sigma)

        # Resampling.sum gives the exact sum of contributing source pixels per
        # output cell, so summing the volume AND the validity mask and dividing
        # at the end is a true pixel-count-weighted average — across tiles as
        # well as within a cell. That replaces the old nearest-neighbour
        # sampling (which picked one arbitrary fine pixel per cell) and the old
        # last-write-wins tile arbitration (which let iteration order decide
        # overlaps) in one step.
        warp_kwargs = dict(
            src_transform=tile_transform, src_crs=tile_crs,
            dst_transform=out_transform, dst_crs=resolved_crs,
            resampling=Resampling.sum,
        )
        for c0 in range(0, num_classes, chunk):
            c1 = min(c0 + chunk, num_classes)
            view = warp_buf[: c1 - c0]
            view[:] = 0.0
            reproject(source=vol[c0:c1], destination=view, **warp_kwargs)
            prob_accum[c0:c1] += view
        mask_buf[:] = 0.0
        reproject(source=valid.astype(np.float32)[None], destination=mask_buf, **warp_kwargs)
        weight_accum += mask_buf[0]

        if nat_raster is not None:
            pred_1idx = np.where(valid, probs.argmax(axis=0) + 1, 0).astype(np.uint8)
            tmp = np.zeros((1, nat_H, nat_W), dtype=np.uint8)
            reproject(
                source=pred_1idx[None],
                destination=tmp,
                src_transform=tile_transform,
                src_crs=tile_crs,
                dst_transform=nat_transform,
                dst_crs=resolved_crs,
                resampling=Resampling.nearest,
                dst_nodata=0,
            )
            np.copyto(nat_raster, tmp[0], where=tmp[0] > 0)
            if nat_conf is not None:
                tmp_conf = np.zeros((1, nat_H, nat_W), dtype=np.float32)
                reproject(
                    source=probs.max(axis=0)[None],
                    destination=tmp_conf,
                    src_transform=tile_transform,
                    src_crs=tile_crs,
                    dst_transform=nat_transform,
                    dst_crs=resolved_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0.0,
                )
                # Same last-write-wins mask as the class raster so the two stay aligned
                np.copyto(nat_conf, tmp_conf[0], where=tmp[0] > 0)

    # ── Resolve the accumulators into a map ───────────────────────────────────
    # weight_accum counts contributing fine pixels, so the count a fully
    # covered cell should have is (cell side in fine pixels)^2. Cells below
    # min_coverage of that are edge slivers with almost no data behind them and
    # become nodata rather than being handed a near-arbitrary class.
    cell_px = max(1.0, resolved_res / native_res)
    full_count = cell_px * cell_px
    covered = weight_accum >= min_coverage * full_count

    raster = np.zeros((out_H, out_W), dtype=np.uint8)
    conf_raster = np.zeros((out_H, out_W), dtype=np.float32) if save_confidence else None
    if covered.any():
        mean_probs = prob_accum[:, covered] / weight_accum[None, covered]
        raster[covered] = (mean_probs.argmax(axis=0) + 1).astype(np.uint8)
        if conf_raster is not None:
            conf_raster[covered] = mean_probs.max(axis=0)
    else:
        logger.warning("No output cell reached min_coverage — the map is empty")

    raster, conf_raster = _repair_and_log(raster, conf_raster, f"{target_m:g} m map")

    # ── Save GeoTIFF ─────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from utils.plot_lcz import save_lcz_map
    map_title = title if title is not None else f"LCZ {model_type.upper()} — {city_name}"

    def _write(arr_out, transform, path, subtitle):
        with rasterio.open(
            str(path), "w", driver="GTiff",
            height=arr_out.shape[0], width=arr_out.shape[1], count=1, dtype="uint8",
            crs=resolved_crs, transform=transform, nodata=0,
        ) as dst_ds:
            dst_ds.write(arr_out, 1)
        logger.info(f"Saved GeoTIFF: {path}")
        png = path.with_suffix(".png")
        save_lcz_map(arr_out, subtitle, png, extent=bbox)
        logger.info(f"Saved PNG: {png}")

    def _write_conf(arr_out, transform, path):
        with rasterio.open(
            str(path), "w", driver="GTiff",
            height=arr_out.shape[0], width=arr_out.shape[1], count=1, dtype="float32",
            crs=resolved_crs, transform=transform, nodata=0.0,
        ) as dst_ds:
            dst_ds.write(arr_out, 1)
        logger.info(f"Saved confidence GeoTIFF: {path}")

    _write(raster, out_transform, output_path,
           f"{map_title} @ {int(round(target_m))}m ({aggregate})")
    if conf_raster is not None:
        _write_conf(conf_raster, out_transform,
                    output_path.with_name(output_path.stem + "_conf.tif"))

    if nat_raster is not None:
        nat_raster, nat_conf = _repair_and_log(nat_raster, nat_conf, "native map")
        nat_m = int(round(embedding_res_m))
        nat_path = output_path.with_name(f"{output_path.stem}_{nat_m}m.tif")
        _write(nat_raster, nat_transform, nat_path,
               f"{map_title} @ {nat_m}m (native)")
        if nat_conf is not None:
            _write_conf(nat_conf, nat_transform,
                        nat_path.with_name(nat_path.stem + "_conf.tif"))

    # ── Coarsened / smoothed output (optional) ──────────────────────────────
    if coarsen_to_m is not None:
        from rasterio.transform import Affine

        from utils.lcz_smoothing import gaussian_likelihood_filter, majority_pool

        if coarsen_to_m < resolved_res:
            raise ValueError(
                f"coarsen_to_m ({coarsen_to_m}) must be >= the native output "
                f"resolution ({resolved_res:.4g}) — this coarsens, it does "
                "not upsample."
            )
        factor = max(1, round(coarsen_to_m / resolved_res))

        if coarsen_method == "majority":
            coarse_raster = majority_pool(raster, factor, num_classes=num_classes)
        elif coarsen_method == "gaussian":
            kwargs = {"sigma_by_class": gaussian_sigma} if gaussian_sigma is not None else {}
            coarse_raster = gaussian_likelihood_filter(
                raster, native_res_m=resolved_res, out_res_m=coarsen_to_m,
                num_classes=num_classes, **kwargs,
            )
        else:
            raise ValueError(f"Unknown coarsen_method: {coarsen_method!r}")

        actual_res_m = resolved_res * factor
        coarse_transform = Affine(
            actual_res_m, 0.0, out_transform.c, 0.0, -actual_res_m, out_transform.f
        )
        coarse_H, coarse_W = coarse_raster.shape
        suffix = f"_{int(round(coarsen_to_m))}m_{coarsen_method}"
        coarse_path = output_path.with_name(output_path.stem + suffix + ".tif")
        with rasterio.open(
            str(coarse_path), "w", driver="GTiff",
            height=coarse_H, width=coarse_W, count=1, dtype="uint8",
            crs=resolved_crs, transform=coarse_transform, nodata=0,
        ) as dst:
            dst.write(coarse_raster, 1)
        logger.info(f"Saved coarsened GeoTIFF ({coarsen_method}, {actual_res_m:.4g}m): {coarse_path}")

        coarse_png_path = coarse_path.with_suffix(".png")
        save_lcz_map(
            coarse_raster, f"{map_title} @ {int(round(coarsen_to_m))}m ({coarsen_method})",
            coarse_png_path, extent=bbox,
        )
        logger.info(f"Saved coarsened PNG: {coarse_png_path}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    from datasets.registry import available_embeddings
    from models import MODEL_REGISTRY

    p = argparse.ArgumentParser(
        description="Seamless ROI inference from raw source embedding tiles."
    )
    p.add_argument("--model-type", required=True, choices=sorted(MODEL_REGISTRY),
                   help="Model family (must match training).")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to .pt checkpoint file.")
    p.add_argument("--preset", default="small",
                   choices=["nano", "small", "base", "medium", "large"],
                   help="Model size preset (must match training).")
    p.add_argument("--arch", default=None,
                   help="Arch override, e.g. a timm model name (must match training).")
    p.add_argument("--depth", type=int, default=None,
                   help="U-Net depth override (must match training).")
    p.add_argument("--base-features", type=int, default=None,
                   help="U-Net base_features override (must match training).")
    p.add_argument("--bottleneck-dropout", type=float, default=0.3,
                   help="U-Net bottleneck dropout (must match training).")
    p.add_argument("--stats-file", type=Path, default=None,
                   help="Legacy linear_probe checkpoints only (fc-only state dict): "
                        "NPZ with train-set 'mean'/'std' arrays. Probes trained via "
                        "patch_classification.py carry their stats in the checkpoint.")
    p.add_argument("--num-classes", type=int, default=17,
                   help="Number of LCZ classes (must match training).")
    p.add_argument("--embedding-name", required=True,
                   choices=available_embeddings(),
                   help="Embedding type key. Deprecated and pending entries are "
                        "excluded (PLAN-v3).")
    p.add_argument("--embedding-dir", required=True, type=Path,
                   help="Directory containing source tile files (.zarr or .tif).")
    p.add_argument("--year", default=None,
                   help="Year filter (required for alpha_earth_coop).")
    p.add_argument("--bbox", default=None,
                   help="ROI bounding box 'west,south,east,north' in EPSG:4326. "
                        "Mutually exclusive with --city / --smod-id.")
    p.add_argument("--city", default=None,
                   help="Look up bbox by JRC_NAME_MAIN from --bounds-csv (case-insensitive).")
    p.add_argument("--smod-id", default=None,
                   help="Look up bbox by SMOD_ID from --bounds-csv (e.g. '30_4716').")
    p.add_argument("--bounds-csv", type=Path,
                   default=Path(__file__).parent.parent / "data" / "guppd_bounds.csv",
                   help="CSV with city bboxes used by --city / --smod-id. "
                        "Default: data/guppd_bounds.csv")
    p.add_argument("--output", required=True, type=Path,
                   help="Output GeoTIFF path.")
    p.add_argument("--patch-size", type=int, default=None,
                   help="Model input size in pixels. Classification: the size the "
                        "model was TRAINED at -- taken from the checkpoint when it "
                        "records one, else 32 (the training default); a rebuild at "
                        "another size loads cleanly but is a different network. "
                        "Segmentation: the sliding-window size (default: 64).")
    p.add_argument("--patch-physical-res", type=float, default=320.0,
                   help="Physical side length of one patch in metres, resnet only "
                        "(default: 320 = 32 px × 10 m/px, So2Sat standard). "
                        "Controls extraction window size and output resolution.")
    p.add_argument("--patch-physical-stride", type=float, default=None,
                   help="Stride between patch origins in metres, classification only "
                        "(default: --target-res, else --patch-physical-res, i.e. "
                        "non-overlapping). Smaller values soft-vote overlapping "
                        "patches per pixel, e.g. 100 votes among the ~9 overlapping "
                        "320 m patches covering each pixel.")
    p.add_argument("--target-res", type=float, default=None,
                   help="Resolution in metres of the PRIMARY output map. Default: "
                        "the classification stride (320 m unless changed), and "
                        f"{DEFAULT_SEG_TARGET_RES_M:g} m for segmentation — LCZ is a "
                        "~100 m concept, not a 10 m one. For classification this also "
                        "sets the sliding stride unless --patch-physical-stride is given.")
    p.add_argument("--aggregate", choices=list(AGGREGATE_METHODS), default="soft",
                   help="How the per-pixel probability volume is pooled into an output "
                        "cell. 'soft' (default): average the probabilities — a "
                        "confidence-weighted vote. 'majority': count fine argmax votes, "
                        "matching the test_*_100m eval metrics. 'gaussian': Demuzere "
                        "et al. 2020's per-class Gaussian applied to the probabilities.")
    p.add_argument("--no-native", dest="write_native", action="store_false",
                   help="Skip the native-resolution sidecar <output>_<res>m.tif, which "
                        "is otherwise written alongside the primary map.")
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="An output cell is nodata unless this fraction of it is backed "
                        "by real fine pixels (default: 0.5).")
    p.add_argument("--overlap", type=int, default=None,
                   help="Overlap between adjacent patches in pixels "
                        "(default: patch_size // 2).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="GPU batch size for inference (default: 8).")
    p.add_argument("--margin-m", type=float, default=200.0,
                   help="Extra metres clipped around the bbox per tile for edge context "
                        "(default: 200).")
    p.add_argument("--normalize", choices=["auto", "none", "channel"], default="auto",
                   help="Input normalisation. 'auto' (default) takes the mode and "
                        "statistics from the checkpoint and errors if it has none; "
                        "'none' reproduces the pre-Phase-1 unnormalised path.")
    p.add_argument("--dequantize", action="store_true",
                   help="Dequantize embeddings on-the-fly. "
                        "Function is selected from --embedding-name: "
                        "seamless → ESD (72-ch), alpha_earth_coop → AlphaEarth int8.")
    p.add_argument("--out-crs", default=None,
                   help="Output CRS (e.g. 'EPSG:4326'). Auto-detected from tiles if omitted.")
    p.add_argument("--out-res", type=float, default=None,
                   help="Output pixel size in out-crs units. Auto-detected if omitted.")
    p.add_argument("--save-confidence", action="store_true",
                   help="Also write <output>_conf.tif with the pooled probability of "
                        "the winning class per cell (both pipelines).")
    p.add_argument("--city-name", default="ROI",
                   help="City/area name for the PNG title.")
    p.add_argument("--accelerator", default="auto",
                   choices=["auto", "gpu", "cpu"],
                   help="Device: auto, gpu, or cpu (default: auto).")
    p.add_argument("--coarsen-to", type=float, default=None,
                   help="Also write a THIRD, coarser GeoTIFF+PNG at this resolution "
                        "in metres, post-processed from the finished primary map "
                        "(e.g. 300 on top of a 100 m primary). For the primary map's "
                        "own resolution use --target-res.")
    p.add_argument("--coarsen-method", choices=["gaussian", "majority"], default="gaussian",
                   help="'gaussian' (default): per-class Gaussian-likelihood filter "
                        "(Demuzere et al. 2020). 'majority': plain block-mode vote, "
                        "faster and blockier. Only used with --coarsen-to.")
    p.add_argument("--gaussian-sigma", type=float, default=None,
                   help="Single sigma in metres overriding the per-class default "
                        "table, for --aggregate gaussian and --coarsen-method gaussian.")
    return p.parse_args()


# Training-time defaults: patch_classification.py --patch-size and the
# segmentation sliding window. Only consulted when neither the caller nor the
# checkpoint says otherwise.
DEFAULT_CLS_PATCH_SIZE = 32
DEFAULT_SEG_PATCH_SIZE = 64


def default_patch_size(checkpoint: Path, model_type: str) -> int:
    """The model input size to use when the caller did not choose one.

    Classification checkpoints written since 2026-10-06 record the size they
    were trained at; older ones were all trained at the 32 px default.
    """
    if _is_segmentation(model_type):
        return DEFAULT_SEG_PATCH_SIZE
    ckpt = torch.load(checkpoint, map_location="cpu")
    stored = ckpt.get("patch_size") if isinstance(ckpt, dict) else None
    return int(stored) if stored is not None else DEFAULT_CLS_PATCH_SIZE


def load_model_and_normalize(
    checkpoint: Path,
    model_type: str,
    embedding_name: str,
    device: torch.device,
    *,
    preset: str = "small",
    arch: str | None = None,
    num_classes: int = 17,
    patch_size: int = 32,
    depth: int | None = None,
    base_features: int | None = None,
    bottleneck_dropout: float = 0.3,
    stats_file: Path | None = None,
    normalize_mode: str = "auto",
) -> tuple[nn.Module, tuple | None]:
    """Rebuild a trained model from a checkpoint, with its normalisation stats.

    Returns ``(model_on_device, normalize)`` where ``normalize`` is the
    ``(mean, std)`` pair to hand to ``TileProbSource``/``infer_roi``, or None
    for the unnormalised path. Raises ``SystemExit`` on the user-facing errors
    (wrong product, missing stats, legacy probe without ``--stats-file``) so a
    CLI reports them as messages rather than tracebacks.

    Shared by this module's own CLI and by ``coarsen_bakeoff.py``: the
    provenance check, the arch/preset reconstruction and the normalisation
    contract all have to agree exactly, or a map is silently wrong.
    """
    # ── Build model from the registry ─────────────────────────────────────────
    from datasets.registry import check_checkpoint_provenance, get_in_channels
    from models import get_family, resolve_arch
    from training.tasks import LCZResNetModule, LCZUNetModule

    in_channels = get_in_channels(embedding_name)
    family = get_family(model_type)

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    # Before anything is built: a checkpoint from another product loads cleanly
    # whenever the channel counts agree, and then predicts confident nonsense.
    # Re-raised as SystemExit so the CLI reports it like the other user errors
    # here, rather than as a traceback out of the registry.
    if isinstance(ckpt, dict):
        try:
            check_checkpoint_provenance(ckpt, embedding_name)
        except ValueError as e:
            raise SystemExit(str(e)) from None

    if model_type == "linear_probe" and "norm.running_mean" not in state:
        # Legacy probe from the retired linear_probe.py script: fc-only state
        # dict + external stats npz. Converted into a self-contained
        # LinearProbeModel so it runs through the standard path below.
        from models.linear_probe import load_legacy_linear_probe

        if stats_file is None:
            raise SystemExit(
                "Legacy linear_probe checkpoint (no normalisation stats in the "
                "state dict) — pass --stats-file with the training-set mean/std NPZ."
            )
        logger.info(f"Legacy linear_probe checkpoint — loading stats from {stats_file}")
        model = load_legacy_linear_probe(
            state, stats_file, in_channels, num_classes
        ).to(device)
    else:
        arch = resolve_arch(model_type, preset, arch)

        build_kwargs: dict = {}
        if family.pipeline == "segmentation":
            build_kwargs["bottleneck_dropout"] = bottleneck_dropout
            # Every (depth, base_features) family honours the overrides, not
            # just unet: fcn8 and attention_unet carry the same tuple payload,
            # and rebuilding them at the preset depth after training with
            # --depth would silently load the wrong structure. resnet_unet's
            # payload is a backbone name, so the isinstance check skips it.
            if isinstance(arch, tuple):
                d, bf = arch
                if depth is not None:
                    d = depth
                if base_features is not None:
                    bf = base_features
                arch = (d, bf)
        else:
            # Classification families: img_size drives both the ViT token grid
            # and the conv stem adaptation, so the rebuilt structure matches
            # what training produced -- but only at the size training used.
            # A different size changes strides/pooling, not parameter shapes,
            # so load_state_dict cannot catch it.
            trained = ckpt.get("patch_size") if isinstance(ckpt, dict) else None
            if trained is not None and int(trained) != int(patch_size):
                raise SystemExit(
                    f"{checkpoint} was trained at patch size {int(trained)}, but "
                    f"{patch_size} was requested. Rebuilding at another size "
                    "yields a different network that still loads cleanly; "
                    f"pass --patch-size {int(trained)}."
                )
            build_kwargs["img_size"] = patch_size

        # This calls family.build directly rather than build_model, so the
        # signature filtering build_model does has to happen here too --
        # img_size is meaningless to mlp/aspp/shallow_cnn/linear_probe and
        # bottleneck_dropout to a seg family that does not take it.
        sig = inspect.signature(family.build)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            build_kwargs = {k: v for k, v in build_kwargs.items() if k in sig.parameters}

        logger.info(f"Building {model_type}: preset={preset}, arch={arch}")
        net = family.build(arch, in_channels=in_channels, num_classes=num_classes,
                           **build_kwargs)

        task_cls = LCZUNetModule if family.pipeline == "segmentation" else LCZResNetModule
        task = task_cls(net, num_classes=num_classes)
        task.model.load_state_dict(state)
        model = task.model.to(device)

    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint}")

    # ── Input normalisation, read from the checkpoint ─────────────────────────
    # It has to match training exactly or every map produced is wrong, so the
    # stats travel inside the checkpoint. Checkpoints written before Phase 1
    # carry no normalisation keys; rather than silently assuming "none" (which
    # would quietly mis-scale a normalised model), demand --normalize none.
    normalize = None
    ckpt_norm = ckpt.get("normalize") if isinstance(ckpt, dict) else None
    if normalize_mode == "auto":
        if ckpt_norm is None:
            raise SystemExit(
                f"{checkpoint} carries no normalisation metadata (trained "
                "before Phase 1, or from a script that does not record it). Pass "
                "--normalize none to reproduce the pre-Phase-1 unnormalised path, "
                "or retrain so the stats are stored in the checkpoint."
            )
        mode = ckpt_norm
    else:
        mode = normalize_mode
        if ckpt_norm is not None and ckpt_norm != mode:
            logger.warning(
                f"--normalize {mode} overrides the checkpoint's '{ckpt_norm}' — "
                "predictions will not match the trained model unless this is "
                "a deliberate reproduction of the old path."
            )

    if mode == "channel":
        mean, std = ckpt.get("channel_mean"), ckpt.get("channel_std")
        if mean is None or std is None:
            raise SystemExit(
                f"{checkpoint} says normalize='channel' but has no "
                "channel_mean/channel_std arrays."
            )
        normalize = (np.asarray(mean, dtype=np.float32),
                     np.asarray(std, dtype=np.float32))
        logger.info(
            f"Input normalisation: channel (median std {np.median(normalize[1]):.4f})"
        )
    else:
        logger.info("Input normalisation: none")

    return model, normalize


def main() -> None:
    args = _parse_args()
    from utils.cli import resolve_overlap
    if args.patch_size is not None:
        resolve_overlap(args)

    # ── Resolve bbox (from --bbox, --city, or --smod-id) ─────────────────────
    n_sources = sum(x is not None for x in [args.bbox, args.city, args.smod_id])
    if n_sources == 0:
        raise SystemExit("Provide one of --bbox, --city, or --smod-id.")
    if n_sources > 1:
        raise SystemExit("--bbox, --city, and --smod-id are mutually exclusive.")

    if args.bbox is not None:
        parts = [float(v) for v in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be 'west,south,east,north'")
        bbox = (parts[0], parts[1], parts[2], parts[3])
    else:
        import pandas as pd
        if not args.bounds_csv.exists():
            raise SystemExit(f"--bounds-csv not found: {args.bounds_csv}")
        df = pd.read_csv(args.bounds_csv)
        if args.city is not None:
            mask = df["JRC_NAME_MAIN"].str.lower() == args.city.lower()
            col, val = "JRC_NAME_MAIN", args.city
        else:
            mask = df["SMOD_ID"].astype(str) == str(args.smod_id)
            col, val = "SMOD_ID", args.smod_id
        matches = df[mask]
        if matches.empty:
            raise SystemExit(f"No city found for {col}='{val}' in {args.bounds_csv}")
        row = matches.iloc[0]
        bbox = (float(row["minx"]), float(row["miny"]), float(row["maxx"]), float(row["maxy"]))
        if args.city_name == "ROI":
            args.city_name = str(row["JRC_NAME_MAIN"])
        logger.info(f"Resolved bbox for '{row['JRC_NAME_MAIN']}': {bbox}")

    # ── Device ────────────────────────────────────────────────────────────────
    from utils.runtime import resolve_device, resolve_dequantize

    device = resolve_device(args.accelerator)
    logger.info(f"Device: {device}")

    if args.patch_size is None:
        args.patch_size = default_patch_size(args.checkpoint, args.model_type)
        logger.info(f"--patch-size not given: using {args.patch_size}")
    if args.overlap is None:
        resolve_overlap(args)

    model, normalize = load_model_and_normalize(
        args.checkpoint, args.model_type, args.embedding_name, device,
        preset=args.preset, arch=args.arch, num_classes=args.num_classes,
        patch_size=args.patch_size, depth=args.depth,
        base_features=args.base_features,
        bottleneck_dropout=args.bottleneck_dropout,
        stats_file=args.stats_file, normalize_mode=args.normalize,
    )

    # ── Dequantize function (auto-applied for coop/seamless) ──────────────────
    dequantize_fn, _ = resolve_dequantize(args.embedding_name, force=args.dequantize)

    # ── Inference ─────────────────────────────────────────────────────────────
    infer_roi(
        model=model,
        model_type=args.model_type,
        embedding_name=args.embedding_name,
        embedding_dir=args.embedding_dir,
        bbox=bbox,
        output_path=args.output,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=device,
        dequantize_fn=dequantize_fn,
        normalize=normalize,
        out_crs=args.out_crs,
        out_res=args.out_res,
        year=args.year,
        city_name=args.city_name,
        margin_m=args.margin_m,
        patch_physical_res_m=args.patch_physical_res,
        patch_physical_stride_m=args.patch_physical_stride,
        save_confidence=args.save_confidence,
        target_res_m=args.target_res,
        aggregate=args.aggregate,
        write_native=args.write_native,
        min_coverage=args.min_coverage,
        coarsen_to_m=args.coarsen_to,
        coarsen_method=args.coarsen_method,
        gaussian_sigma=args.gaussian_sigma,
    )


if __name__ == "__main__":
    main()
