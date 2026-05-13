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
- UNet (segmentation) from train_unet.py checkpoints
- ResNet (classification) from train_resnet.py checkpoints
- All embedding types: tessera, alpha_earth, alpha_earth_coop

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
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from extract_so2sat_embeddings import _build_tile_index, _open_tile


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

    da = _open_tile(path)
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
    """Run classification model with majority-vote sliding window.

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
        (H, W) uint8 with 0-indexed class predictions.
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

    vote_sum = np.zeros((num_classes, H_pad, W_pad), dtype=np.int32)

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
            preds = model(batch).argmax(dim=1).cpu().numpy()
            for (r, c), cls in zip(batch_pos, preds):
                vote_sum[int(cls), r:r + extract_size, c:c + extract_size] += 1

    return vote_sum.argmax(axis=0).astype(np.uint8)[:H, :W]


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
    out_crs: str | None = None,
    out_res: float | None = None,
    year: str | None = None,
    city_name: str = "ROI",
    margin_m: float = 200.0,
    patch_physical_res_m: float = 320.0,
) -> Path:
    """Run model inference over a bbox directly from raw source embedding tiles.

    Bypasses pre-extracted grid npy files entirely.  Tiles are clipped on-the-fly,
    the model runs as a sliding window over each clipped tile, and results are
    assembled via rasterio.warp.reproject so UTM zone boundaries produce no artefacts.

    Args:
        model: Loaded nn.Module (UNet or ResNet), already on ``device``.
        model_type: ``"unet"`` (per-pixel segmentation) or ``"resnet"`` (patch cls).
        embedding_name: Registry key — ``"tessera"``, ``"alpha_earth"``,
            ``"alpha_earth_coop"``.
        embedding_dir: Directory containing source tile files.
        bbox: ``(west, south, east, north)`` in EPSG:4326.
        output_path: Output GeoTIFF path (PNG is saved alongside).
        num_classes: Number of LCZ classes (default 17).
        patch_size: Sliding-window patch side in pixels.
        overlap: Overlap between adjacent patches in pixels (0 = no overlap).
        batch_size: GPU batch size for the sliding window.
        device: Torch device; auto-selected if None.
        dequantize: Apply AlphaEarth coop dequantization.
        out_crs: Output CRS (auto-detected from first tile if None).
        out_res: Output pixel size in ``out_crs`` units (auto-detected if None).
        year: Year string, required for ``"alpha_earth_coop"``.
        city_name: Title string for the PNG.
        margin_m: Extra metres clipped around the bbox per tile for edge context.
        patch_physical_res_m: Physical side length of one patch in metres (resnet only).
            Determines how many embedding pixels to extract per patch and the output
            resolution. Default 320 m = 32 px × 10 m/px (So2Sat patch size). Ignored
            for unet (segmentation always outputs at embedding resolution).

    Returns:
        Path to the saved GeoTIFF.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from shapely.geometry import box

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < patch_size ({patch_size})")

    # ── Tile spatial index ────────────────────────────────────────────────────
    tile_paths, tree = _build_tile_index(embedding_dir, embedding_name, year=year)
    roi_geom = box(*bbox)
    idxs = tree.query(roi_geom)
    if len(idxs) == 0:
        raise RuntimeError(f"No embedding tiles found for bbox {bbox}")

    matched_paths = [tile_paths[i] for i in idxs]
    logger.info(f"Found {len(matched_paths)} tile(s) intersecting the ROI")

    # ── For coop tiles, build path → reported-valid-bbox map ──────────────────
    # Coop tile data physically overshoots the UTM zone boundary (e.g. a UTM30
    # tile extends ~0.5° into UTM31 territory) but the index clips the reported
    # WGS84 bounds to the zone edge.  Intersecting with those bounds in
    # _open_and_clip discards the contaminated overhang region.
    path_to_valid_bbox: dict[Path, tuple[float, float, float, float]] = {}
    if embedding_name == "alpha_earth_coop":
        import geopandas as _gpd
        _idx = _gpd.read_file(
            embedding_dir / "aef_index.gpkg",
            where=f"year = {int(year)}" if year else "",
        )
        _name_to_bounds: dict[str, tuple] = {
            Path(r["path"]).name: (
                r["wgs84_west"], r["wgs84_south"],
                r["wgs84_east"], r["wgs84_north"],
            )
            for _, r in _idx.iterrows()
        }
        for p in matched_paths:
            if p.name in _name_to_bounds:
                path_to_valid_bbox[p] = _name_to_bounds[p.name]

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

    if model_type != "unet":
        # Classification: each output pixel = one physical patch (patch_physical_res_m).
        # Compute how many embedding pixels span that physical distance, then derive
        # output resolution so reproject downsamples to exactly one pixel per patch.
        extract_px = max(1, round(patch_physical_res_m / embedding_res_m))
        cls_stride = extract_px  # non-overlapping: one patch per output pixel
        if out_res is not None:
            resolved_res = out_res
        elif resolved_crs == first_crs:
            resolved_res = extract_px * embedding_res_m
        else:
            resolved_res = _meters_to_out_res(
                extract_px * embedding_res_m, first_crs, resolved_crs, lon_c, lat_c
            )
    else:
        extract_px = patch_size
        cls_stride = stride  # unused for unet
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

    # ── Process each tile ─────────────────────────────────────────────────────
    n_done = n_skip = 0
    for i, tile_path in enumerate(matched_paths):
        logger.info(f"Tile {i + 1}/{len(matched_paths)}: {tile_path.name}")

        result = _open_and_clip(tile_path, bbox, margin_m=margin_m, dequantize_fn=dequantize_fn,
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
        if model_type == "unet":
            pred = _sliding_window_seg(
                model, arr, patch_size, stride, device, num_classes, batch_size
            )
        else:
            pred = _sliding_window_cls(
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

    # ── Save PNG ──────────────────────────────────────────────────────────────
    from utils.plot_lcz import save_lcz_map
    png_path = output_path.with_suffix(".png")
    save_lcz_map(raster, f"LCZ {model_type.upper()} — {city_name}", png_path)
    logger.info(f"Saved PNG: {png_path}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seamless ROI inference from raw source embedding tiles."
    )
    p.add_argument("--model-type", required=True, choices=["unet", "resnet"],
                   help="Model architecture type.")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to .pt checkpoint file.")
    p.add_argument("--preset", default="small",
                   choices=["nano", "small", "base", "medium", "large"],
                   help="Model size preset (must match training).")
    p.add_argument("--arch", default=None,
                   help="timm arch override for ResNet (must match training).")
    p.add_argument("--depth", type=int, default=None,
                   help="U-Net depth override (must match training).")
    p.add_argument("--base-features", type=int, default=None,
                   help="U-Net base_features override (must match training).")
    p.add_argument("--bottleneck-dropout", type=float, default=0.3,
                   help="U-Net bottleneck dropout (must match training).")
    p.add_argument("--num-classes", type=int, default=17,
                   help="Number of LCZ classes (must match training).")
    p.add_argument("--embedding-name", required=True,
                   choices=["tessera", "tesserav1.1", "alpha_earth", "alpha_earth_coop", "seamless"],
                   help="Embedding type key.")
    p.add_argument("--embedding-dir", required=True, type=Path,
                   help="Directory containing source tile files (.zarr or .tif).")
    p.add_argument("--year", default=None,
                   help="Year filter (required for alpha_earth_coop).")
    p.add_argument("--bbox", required=True,
                   help="ROI bounding box 'west,south,east,north' in EPSG:4326.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output GeoTIFF path.")
    p.add_argument("--patch-size", type=int, default=64,
                   help="Sliding window patch size in pixels (default: 64).")
    p.add_argument("--patch-physical-res", type=float, default=320.0,
                   help="Physical side length of one patch in metres, resnet only "
                        "(default: 320 = 32 px × 10 m/px, So2Sat standard). "
                        "Controls extraction window size and output resolution.")
    p.add_argument("--overlap", type=int, default=None,
                   help="Overlap between adjacent patches in pixels "
                        "(default: patch_size // 2).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="GPU batch size for inference (default: 8).")
    p.add_argument("--margin-m", type=float, default=200.0,
                   help="Extra metres clipped around the bbox per tile for edge context "
                        "(default: 200).")
    p.add_argument("--dequantize", action="store_true",
                   help="Dequantize embeddings on-the-fly. "
                        "Function is selected from --embedding-name: "
                        "seamless → ESD (72-ch), alpha_earth_coop → AlphaEarth int8.")
    p.add_argument("--out-crs", default=None,
                   help="Output CRS (e.g. 'EPSG:4326'). Auto-detected from tiles if omitted.")
    p.add_argument("--out-res", type=float, default=None,
                   help="Output pixel size in out-crs units. Auto-detected if omitted.")
    p.add_argument("--city-name", default="ROI",
                   help="City/area name for the PNG title.")
    p.add_argument("--accelerator", default="auto",
                   choices=["auto", "gpu", "cpu"],
                   help="Device: auto, gpu, or cpu (default: auto).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.overlap is None:
        args.overlap = args.patch_size // 2

    # ── Parse bbox ────────────────────────────────────────────────────────────
    parts = [float(v) for v in args.bbox.split(",")]
    if len(parts) != 4:
        raise SystemExit("--bbox must be 'west,south,east,north'")
    bbox = (parts[0], parts[1], parts[2], parts[3])

    # ── Device ────────────────────────────────────────────────────────────────
    if args.accelerator == "cpu":
        device = torch.device("cpu")
    elif args.accelerator == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    from datasets.registry import get_in_channels
    in_channels = get_in_channels(args.embedding_name)

    if args.model_type == "unet":
        from train_unet import UNet, LCZUNetModule

        d, bf = UNet.PRESETS.get(args.preset, (3, 32))
        if args.depth is not None:
            d = args.depth
        if args.base_features is not None:
            bf = args.base_features

        logger.info(f"Building U-Net: preset={args.preset}, depth={d}, base_features={bf}")
        unet = UNet(
            in_channels=in_channels,
            num_classes=args.num_classes,
            depth=d,
            base_features=bf,
            bottleneck_dropout=args.bottleneck_dropout,
        )
        task = LCZUNetModule(unet, num_classes=args.num_classes)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        task.model.load_state_dict(state)
        model = task.model.to(device)

    else:  # resnet
        from train_resnet import RESNET_PRESETS, LCZResNetModule, build_resnet

        arch_name = args.arch or RESNET_PRESETS.get(args.preset, "resnet50")
        logger.info(f"Building ResNet: preset={args.preset}, arch={arch_name}")
        resnet = build_resnet(
            arch=arch_name,
            in_channels=in_channels,
            num_classes=args.num_classes,
        )
        task = LCZResNetModule(resnet, num_classes=args.num_classes)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        task.model.load_state_dict(state)
        model = task.model.to(device)

    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Dequantize function ───────────────────────────────────────────────────
    dequantize_fn = None
    if args.dequantize:
        if args.embedding_name == "seamless":
            from dequantize_embeddings import dequantize_esd
            dequantize_fn = dequantize_esd
        else:
            from dequantize_embeddings import dequantize_alphaearth_embeddings
            dequantize_fn = dequantize_alphaearth_embeddings

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
        out_crs=args.out_crs,
        out_res=args.out_res,
        year=args.year,
        city_name=args.city_name,
        margin_m=args.margin_m,
        patch_physical_res_m=args.patch_physical_res,
    )


if __name__ == "__main__":
    main()
