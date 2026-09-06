"""Seamless ROI inference from raw source embedding tiles.

Instead of using pre-extracted grid npy files, this script:
1. Finds source embedding tiles (zarr/tif) that cover the requested bbox.
2. For each tile, clips the embedding to the ROI (+ an optional margin for context).
3. Runs the model with a sliding window + Hanning-weighted logit blending (seg)
   or majority vote (cls).
4. Projects the per-tile prediction into the output raster via rasterio.warp.reproject.
5. Saves a GeoTIFF + PNG.

This eliminates all grid-tile artifacts and UTM-zone boundary artefacts.

Supports:
- Every model family in the models registry: segmentation families (unet,
  resnet_unet, fcn8) run per-pixel with Hanning-blended logits; classification
  families (resnet, mlp, aspp, vit, shallow_cnn, linear_probe, ...) run
  patch-wise with majority vote.
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

Example (classification):
    python src/infer_roi.py \\
        --model-type resnet --preset base \\
        --checkpoint /path/to/resnet-base-best.pt \\
        --embedding-name alpha_earth_coop \\
        --embedding-dir /maps/.../coop \\
        --year 2017 \\
        --bbox "-0.51,51.28,0.33,51.69" \\
        --output /maps/.../London_cls.tif \\
        --num-classes 17 --patch-size 32
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

from datasets.tiles import build_coop_valid_bbox_map, build_tile_index, open_tile


def _is_segmentation(model_type: str) -> bool:
    """Whether a model family runs per-pixel segmentation (vs patch cls)."""
    from models import MODEL_REGISTRY
    fam = MODEL_REGISTRY.get(model_type)
    return fam is not None and fam.pipeline == "segmentation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hanning_2d(h: int, w: int) -> np.ndarray:
    """2D Hanning taper window of shape (h, w), float32."""
    win_h = np.hanning(h).astype(np.float32)
    win_w = np.hanning(w).astype(np.float32)
    return win_h[:, None] * win_w[None, :]


def _open_and_clip(
    path: Path,
    roi_4326: tuple[float, float, float, float],
    margin_m: float = 0.0,
    dequantize_fn=None,
    normalize: tuple | None = None,
    valid_bbox_4326: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, str, Any] | None:
    """Open a source tile, clip to roi_4326 + margin, return (arr, crs, transform).

    arr is float32 (C, H, W), north-up.  transform is rasterio Affine.
    Returns None if no valid data remains after clipping.

    valid_bbox_4326: if given, the clip region is intersected with this box
        before any margin is applied.  Use for coop tiles whose actual pixel
        extent overshoots the reported UTM-zone boundary: the index clips
        reported bounds to the zone edge (E=0° for UTM30, W=0° for UTM31),
        and intersecting here discards the contaminated overhang so that
        UTM31 tile predictions never overwrite UTM30 clean predictions west
        of lon=0° (and vice-versa).
    """
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    try:
        da = open_tile(path)
    except Exception as e:
        logger.warning(f"Failed to open tile {path.name}: {e} — skipping")
        return None
    if da.rio.crs is None:
        return None

    tile_crs_str = da.rio.crs.to_string()

    t = Transformer.from_crs("EPSG:4326", tile_crs_str, always_xy=True)

    # Restrict clip region to tile's reported valid extent (removes contaminated overhang).
    clip_geom_4326 = box(*roi_4326)
    if valid_bbox_4326 is not None:
        clip_geom_4326 = clip_geom_4326.intersection(box(*valid_bbox_4326))
        if clip_geom_4326.is_empty:
            return None

    roi_in_tile = shapely_transform(t.transform, clip_geom_4326)
    minx, miny, maxx, maxy = roi_in_tile.bounds

    if margin_m > 0.0:
        if da.rio.crs.is_geographic:
            m = margin_m / 111320.0
        else:
            m = margin_m
        minx -= m; miny -= m; maxx += m; maxy += m

    try:
        clipped = da.rio.clip_box(minx, miny, maxx, maxy)
    except Exception:
        return None

    if clipped.size == 0 or clipped.sizes.get("x", 0) < 1 or clipped.sizes.get("y", 0) < 1:
        return None

    # North-up
    if clipped.sizes.get("y", 0) > 1 and float(clipped.y.values[0]) < float(clipped.y.values[-1]):
        clipped = clipped.isel(y=slice(None, None, -1))

    arr = clipped.values.astype(np.float32)
    if dequantize_fn is not None:
        arr = dequantize_fn(arr)
    if normalize is not None:
        mean, std = normalize
        arr = (arr - mean[:, None, None]) / (std[:, None, None] + 1e-6)

    # Compute Affine manually from coordinate arrays.
    # clipped.rio.transform() uses step = y[1]-y[0] and returns e = -step,
    # but after isel(y=::-1) the step is negative → e=+10 (south-up), wrong.
    from rasterio.transform import Affine
    x_vals = clipped.x.values
    y_vals = clipped.y.values  # descending (north-up) after the flip above
    res_x = float(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else float(abs(clipped.rio.resolution()[0]))
    res_y = float(y_vals[1] - y_vals[0]) if len(y_vals) > 1 else -float(abs(clipped.rio.resolution()[1]))
    # Top-left corner = centre of top-left pixel ± half pixel
    transform = Affine(
        res_x, 0.0, float(x_vals[0]) - res_x / 2,
        0.0, res_y, float(y_vals[0]) - res_y / 2,
    )
    return arr, tile_crs_str, transform


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
        (H, W) uint8 with 0-indexed class predictions.
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

    valid = weight_sum > 0
    result = np.zeros((H_pad, W_pad), dtype=np.uint8)
    if valid.any():
        result[valid] = (logit_sum[:, valid] / weight_sum[None, valid]).argmax(axis=0).astype(np.uint8)

    return result[:H, :W]


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

    Each patch's softmax probabilities are accumulated over its full footprint;
    the per-pixel argmax of the accumulated probabilities is returned. With
    overlapping patches (stride < extract_size) this soft-votes among all
    patches covering a pixel; with non-overlapping patches it reduces to the
    plain per-patch argmax.

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
        (pred, conf): (H, W) uint8 0-indexed class predictions and (H, W)
        float32 normalized confidence (max accumulated prob / total accumulated
        prob, i.e. the soft-voted probability of the winning class).
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

    pred = prob_sum.argmax(axis=0).astype(np.uint8)
    total = prob_sum.sum(axis=0)
    conf = np.zeros_like(total, dtype=np.float32)
    np.divide(prob_sum.max(axis=0), total, out=conf, where=total > 0)
    return pred[:H, :W], conf[:H, :W]


# ---------------------------------------------------------------------------
# Output raster setup
# ---------------------------------------------------------------------------

def _setup_output(
    bbox_4326: tuple[float, float, float, float],
    out_crs: str,
    out_res: float,
) -> tuple[Any, int, int]:
    """Return (transform, H, W) for the output raster in out_crs at out_res."""
    from pyproj import Transformer
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform

    t = Transformer.from_crs("EPSG:4326", out_crs, always_xy=True)
    roi_out = shapely_transform(t.transform, box(*bbox_4326))
    minx, miny, maxx, maxy = roi_out.bounds

    W = max(1, int(round((maxx - minx) / out_res)))
    H = max(1, int(round((maxy - miny) / out_res)))
    transform = from_origin(minx, maxy, out_res, out_res)
    return transform, H, W


def _meters_to_out_res(
    tile_res_m: float,
    first_crs: str,
    out_crs: str,
    bbox_center_lon: float,
    bbox_center_lat: float,
) -> float:
    """Convert tile resolution in meters to out_crs units at bbox centre."""
    from pyproj import Transformer

    t_to_tile = Transformer.from_crs("EPSG:4326", first_crs, always_xy=True)
    t_to_out = Transformer.from_crs(first_crs, out_crs, always_xy=True)
    x0, y0 = t_to_tile.transform(bbox_center_lon, bbox_center_lat)
    ox0, oy0 = t_to_out.transform(x0, y0)
    ox1, oy1 = t_to_out.transform(x0 + tile_res_m, y0)
    return abs(ox1 - ox0)


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
            (classification only). Defaults to ``patch_physical_res_m``
            (non-overlapping, one prediction per patch). Smaller values slide
            overlapping patches and soft-vote their softmax probabilities per
            pixel, producing a finer output grid — e.g. 160 m yields a 160 m map
            where each pixel averages the 4 overlapping 320 m patches. Also sets
            the output resolution (unless ``out_res`` is given).
        save_confidence: Classification only — also write a float32 sidecar
            ``<output>_conf.tif`` with the soft-voted probability of the
            winning class per pixel (0 = nodata). Ignored for segmentation.

    Returns:
        Path to the saved GeoTIFF.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from shapely.geometry import box

    if device is None:
        from utils.runtime import resolve_device
        device = resolve_device("auto")

    is_seg = _is_segmentation(model_type)
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < patch_size ({patch_size})")

    # ── Tile spatial index ────────────────────────────────────────────────────
    tile_paths, tree = build_tile_index(embedding_dir, embedding_name, year=year)
    roi_geom = box(*bbox)
    idxs = tree.query(roi_geom)
    if len(idxs) == 0:
        raise RuntimeError(f"No embedding tiles found for bbox {bbox}")

    matched_paths = [tile_paths[i] for i in idxs]
    logger.info(f"Found {len(matched_paths)} tile(s) intersecting the ROI")

    # ── For coop tiles, build path → reported-valid-bbox map ──────────────────
    # Intersecting with the reported bounds in _open_and_clip discards data
    # that overshoots the tile's UTM zone boundary.
    path_to_valid_bbox: dict[Path, tuple[float, float, float, float]] = {}
    if embedding_name == "alpha_earth_coop":
        path_to_valid_bbox = build_coop_valid_bbox_map(embedding_dir, year, matched_paths)

    # ── Auto-detect output CRS / resolution from first valid tile ─────────────
    first_result = None
    for p in matched_paths:
        r = _open_and_clip(p, bbox, margin_m=0.0, dequantize_fn=None,
                           valid_bbox_4326=path_to_valid_bbox.get(p))
        if r is not None:
            first_result = r
            break
    if first_result is None:
        raise RuntimeError("No valid data in any matched tile")

    _, first_crs, first_transform = first_result
    resolved_crs = out_crs or first_crs

    embedding_res_m = abs(first_transform.a)
    lat_c = (bbox[1] + bbox[3]) / 2
    lon_c = (bbox[0] + bbox[2]) / 2

    if not is_seg:
        # Classification: each output pixel = one stride cell (patch_physical_stride_m,
        # default = patch_physical_res_m → non-overlapping, one pixel per patch).
        # Compute how many embedding pixels span those physical distances, then derive
        # output resolution so reproject downsamples to exactly one pixel per stride.
        stride_m = patch_physical_stride_m or patch_physical_res_m
        if not 0 < stride_m <= patch_physical_res_m:
            raise ValueError(
                f"patch_physical_stride_m ({stride_m}) must be in "
                f"(0, patch_physical_res_m={patch_physical_res_m}] — a larger stride "
                "would leave uncovered pixels."
            )
        extract_px = max(1, round(patch_physical_res_m / embedding_res_m))
        cls_stride = max(1, round(stride_m / embedding_res_m))
        if out_res is not None:
            resolved_res = out_res
        elif resolved_crs == first_crs:
            resolved_res = cls_stride * embedding_res_m
        else:
            resolved_res = _meters_to_out_res(
                cls_stride * embedding_res_m, first_crs, resolved_crs, lon_c, lat_c
            )
    else:
        if out_res is not None:
            resolved_res = out_res
        elif resolved_crs == first_crs:
            resolved_res = embedding_res_m
        else:
            resolved_res = _meters_to_out_res(embedding_res_m, first_crs, resolved_crs, lon_c, lat_c)

    logger.info(f"Output CRS: {resolved_crs}, resolution: {resolved_res:.8g} units/px")

    # ── Output raster ─────────────────────────────────────────────────────────
    out_transform, out_H, out_W = _setup_output(bbox, resolved_crs, resolved_res)
    raster = np.zeros((out_H, out_W), dtype=np.uint8)
    logger.info(f"Output raster: {out_H}×{out_W} px")

    if save_confidence and is_seg:
        logger.warning("save_confidence is classification-only — ignoring for segmentation")
        save_confidence = False
    conf_raster = np.zeros((out_H, out_W), dtype=np.float32) if save_confidence else None

    # ── Process each tile ─────────────────────────────────────────────────────
    n_done = n_skip = 0
    for i, tile_path in enumerate(matched_paths):
        logger.info(f"Tile {i + 1}/{len(matched_paths)}: {tile_path.name}")

        result = _open_and_clip(tile_path, bbox, margin_m=margin_m, dequantize_fn=dequantize_fn,
                               normalize=normalize,
                               valid_bbox_4326=path_to_valid_bbox.get(tile_path))
        if result is None:
            logger.warning("  Skipped — no valid data after clip")
            n_skip += 1
            continue

        arr, tile_crs, tile_transform = result
        C, H_tile, W_tile = arr.shape
        logger.info(f"  {C}ch × {H_tile}×{W_tile} px  crs={tile_crs}")

        if H_tile < 1 or W_tile < 1:
            n_skip += 1
            continue

        # Sliding window
        if is_seg:
            pred = _sliding_window_seg(
                model, arr, patch_size, stride, device, num_classes, batch_size
            )
        else:
            pred, conf = _sliding_window_cls(
                model, arr, patch_size, cls_stride, device, num_classes, batch_size,
                extract_size=extract_px, model_input_size=patch_size,
            )

        # Convert 0-indexed → 1-indexed (1-17, 0=nodata)
        pred_1idx = (pred.astype(np.uint16) + 1).astype(np.uint8)

        # Reproject tile prediction into output raster (last-write-wins for overlaps)
        tmp = np.zeros((1, out_H, out_W), dtype=np.uint8)
        reproject(
            source=pred_1idx[None],
            destination=tmp,
            src_transform=tile_transform,
            src_crs=tile_crs,
            dst_transform=out_transform,
            dst_crs=resolved_crs,
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
        np.copyto(raster, tmp[0], where=tmp[0] > 0)

        if conf_raster is not None:
            tmp_conf = np.zeros((1, out_H, out_W), dtype=np.float32)
            reproject(
                source=conf[None],
                destination=tmp_conf,
                src_transform=tile_transform,
                src_crs=tile_crs,
                dst_transform=out_transform,
                dst_crs=resolved_crs,
                resampling=Resampling.nearest,
                dst_nodata=0.0,
            )
            # Same last-write-wins mask as the class raster so the two stay aligned
            np.copyto(conf_raster, tmp_conf[0], where=tmp[0] > 0)
        n_done += 1

    logger.info(f"Tiles processed: {n_done} done, {n_skip} skipped")

    # ── Save GeoTIFF ─────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        str(output_path), "w", driver="GTiff",
        height=out_H, width=out_W, count=1, dtype="uint8",
        crs=resolved_crs, transform=out_transform, nodata=0,
    ) as dst:
        dst.write(raster, 1)
    logger.info(f"Saved GeoTIFF: {output_path}")

    if conf_raster is not None:
        conf_path = output_path.with_name(output_path.stem + "_conf.tif")
        with rasterio.open(
            str(conf_path), "w", driver="GTiff",
            height=out_H, width=out_W, count=1, dtype="float32",
            crs=resolved_crs, transform=out_transform, nodata=0.0,
        ) as dst:
            dst.write(conf_raster, 1)
        logger.info(f"Saved confidence GeoTIFF: {conf_path}")

    # ── Save PNG ──────────────────────────────────────────────────────────────
    from utils.plot_lcz import save_lcz_map
    png_path = output_path.with_suffix(".png")
    map_title = title if title is not None else f"LCZ {model_type.upper()} — {city_name}"
    save_lcz_map(raster, map_title, png_path, extent=bbox)
    logger.info(f"Saved PNG: {png_path}")

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
    p.add_argument("--patch-size", type=int, default=64,
                   help="Sliding window patch size in pixels (default: 64).")
    p.add_argument("--patch-physical-res", type=float, default=320.0,
                   help="Physical side length of one patch in metres, resnet only "
                        "(default: 320 = 32 px × 10 m/px, So2Sat standard). "
                        "Controls extraction window size and output resolution.")
    p.add_argument("--patch-physical-stride", type=float, default=None,
                   help="Stride between patch origins in metres, classification only "
                        "(default: --patch-physical-res, i.e. non-overlapping). "
                        "Smaller values soft-vote overlapping patches per pixel and "
                        "set the output resolution, e.g. 160 gives a 160 m map "
                        "averaging the softmax of the 4 overlapping 320 m patches.")
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
                   help="Classification only: also write <output>_conf.tif with the "
                        "soft-voted probability of the winning class per pixel.")
    p.add_argument("--city-name", default="ROI",
                   help="City/area name for the PNG title.")
    p.add_argument("--accelerator", default="auto",
                   choices=["auto", "gpu", "cpu"],
                   help="Device: auto, gpu, or cpu (default: auto).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from utils.cli import resolve_overlap
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

    # ── Build model from the registry ─────────────────────────────────────────
    from datasets.registry import check_checkpoint_provenance, get_in_channels
    from models import get_family, resolve_arch
    from training.tasks import LCZResNetModule, LCZUNetModule

    in_channels = get_in_channels(args.embedding_name)
    family = get_family(args.model_type)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    # Before anything is built: a checkpoint from another product loads cleanly
    # whenever the channel counts agree, and then predicts confident nonsense.
    # Re-raised as SystemExit so the CLI reports it like the other user errors
    # here, rather than as a traceback out of the registry.
    if isinstance(ckpt, dict):
        try:
            check_checkpoint_provenance(ckpt, args.embedding_name)
        except ValueError as e:
            raise SystemExit(str(e)) from None

    if args.model_type == "linear_probe" and "norm.running_mean" not in state:
        # Legacy probe from the retired linear_probe.py script: fc-only state
        # dict + external stats npz. Converted into a self-contained
        # LinearProbeModel so it runs through the standard path below.
        from models.linear_probe import load_legacy_linear_probe

        if args.stats_file is None:
            raise SystemExit(
                "Legacy linear_probe checkpoint (no normalisation stats in the "
                "state dict) — pass --stats-file with the training-set mean/std NPZ."
            )
        logger.info(f"Legacy linear_probe checkpoint — loading stats from {args.stats_file}")
        model = load_legacy_linear_probe(
            state, args.stats_file, in_channels, args.num_classes
        ).to(device)
    else:
        arch = resolve_arch(args.model_type, args.preset, args.arch)

        build_kwargs: dict = {}
        if family.pipeline == "segmentation":
            build_kwargs["bottleneck_dropout"] = args.bottleneck_dropout
            if args.model_type == "unet":
                d, bf = arch
                if args.depth is not None:
                    d = args.depth
                if args.base_features is not None:
                    bf = args.base_features
                arch = (d, bf)
        else:
            # Classification families: img_size drives both the ViT token grid
            # and the conv stem adaptation, so the rebuilt structure matches
            # what training produced.
            build_kwargs["img_size"] = args.patch_size

        # This calls family.build directly rather than build_model, so the
        # signature filtering build_model does has to happen here too --
        # img_size is meaningless to mlp/aspp/shallow_cnn/linear_probe and
        # bottleneck_dropout to a seg family that does not take it.
        sig = inspect.signature(family.build)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            build_kwargs = {k: v for k, v in build_kwargs.items() if k in sig.parameters}

        logger.info(f"Building {args.model_type}: preset={args.preset}, arch={arch}")
        net = family.build(arch, in_channels=in_channels, num_classes=args.num_classes,
                           **build_kwargs)

        task_cls = LCZUNetModule if family.pipeline == "segmentation" else LCZResNetModule
        task = task_cls(net, num_classes=args.num_classes)
        task.model.load_state_dict(state)
        model = task.model.to(device)

    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Input normalisation, read from the checkpoint ─────────────────────────
    # It has to match training exactly or every map produced is wrong, so the
    # stats travel inside the checkpoint. Checkpoints written before Phase 1
    # carry no normalisation keys; rather than silently assuming "none" (which
    # would quietly mis-scale a normalised model), demand --normalize none.
    normalize = None
    ckpt_norm = ckpt.get("normalize") if isinstance(ckpt, dict) else None
    if args.normalize == "auto":
        if ckpt_norm is None:
            raise SystemExit(
                f"{args.checkpoint} carries no normalisation metadata (trained "
                "before Phase 1, or from a script that does not record it). Pass "
                "--normalize none to reproduce the pre-Phase-1 unnormalised path, "
                "or retrain so the stats are stored in the checkpoint."
            )
        mode = ckpt_norm
    else:
        mode = args.normalize
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
                f"{args.checkpoint} says normalize='channel' but has no "
                "channel_mean/channel_std arrays."
            )
        normalize = (np.asarray(mean, dtype=np.float32),
                     np.asarray(std, dtype=np.float32))
        logger.info(
            f"Input normalisation: channel (median std {np.median(normalize[1]):.4f})"
        )
    else:
        logger.info("Input normalisation: none")

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
    )


if __name__ == "__main__":
    main()
