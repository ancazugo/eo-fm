"""ROI prediction map from a linear probe (GAP → nn.Linear) checkpoint.

Slides a window over raw source embedding tiles, applies global average pooling
(or mean+std) to each patch, z-score normalises with training-set stats, then
runs the frozen linear layer to get a class prediction per patch.  Results are
assembled via rasterio.warp.reproject into a single GeoTIFF + PNG.

Example (coop, London):
    python src/infer_roi_linear_probe.py \\
        --checkpoint .../pious-violet-240/linear_probe_best.pt \\
        --stats-file .../cache/global_AlphaEarthCoop_gap_stats.npz \\
        --embedding-name alpha_earth_coop \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \\
        --year 2017 --pooling gap \\
        --smod-id 30_4716 \\
        --output .../pious-violet-240/London_linear_probe.tif

Example (seamless, Nairobi):
    python src/infer_roi_linear_probe.py \\
        --checkpoint .../sleek-flower-241/linear_probe_best.pt \\
        --stats-file .../cache/global_EmbeddedSeamless_gap_stats.npz \\
        --embedding-name seamless \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/EmbeddedSeamlessData/2017 \\
        --year 2017 --pooling gap \\
        --smod-id 30_9135 \\
        --output .../sleek-flower-241/Nairobi_linear_probe.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from infer_roi import (
    _open_and_clip,
    _patch_positions,
    _setup_output,
    _meters_to_out_res,
)
from extract_so2sat_embeddings import _build_tile_index
from linear_probe import LinearProbe


# ---------------------------------------------------------------------------
# Sliding window — linear probe
# ---------------------------------------------------------------------------

def _sliding_window_lp(
    model: LinearProbe,
    stats_mean: np.ndarray,
    stats_std: np.ndarray,
    arr: np.ndarray,
    extract_size: int,
    device: torch.device,
    num_classes: int,
    batch_size: int,
    pooling: str,
) -> np.ndarray:
    """Slide a non-overlapping window; predict one class per patch via GAP + linear.

    Returns:
        (H, W) uint8 with 0-indexed class predictions, at patch resolution.
    """
    C, H, W = arr.shape
    out_H = max(1, (H + extract_size - 1) // extract_size)
    out_W = max(1, (W + extract_size - 1) // extract_size)
    result = np.zeros((out_H, out_W), dtype=np.uint8)

    positions = _patch_positions(H, W, extract_size, extract_size)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_pos = positions[i : i + batch_size]
            patches = []
            for r, c in batch_pos:
                patch = arr[:, r : r + extract_size, c : c + extract_size]
                mean = patch.mean(axis=(1, 2))
                if pooling == "gap":
                    feat = mean
                else:
                    feat = np.concatenate([mean, patch.std(axis=(1, 2))])
                feat = (feat - stats_mean) / (stats_std + 1e-8)
                patches.append(feat)

            X = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)
            preds = model(X).argmax(dim=1).cpu().numpy()

            for k, (r, c) in enumerate(batch_pos):
                pr = r // extract_size
                pc = c // extract_size
                if pr < out_H and pc < out_W:
                    result[pr, pc] = preds[k]

    return result


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def infer_roi_lp(
    checkpoint: Path,
    stats_file: Path,
    embedding_name: str,
    embedding_dir: Path,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    pooling: str = "gap",
    num_classes: int = 17,
    patch_physical_res_m: float = 320.0,
    batch_size: int = 256,
    device: torch.device | None = None,
    dequantize_fn=None,
    out_crs: str | None = None,
    out_res: float | None = None,
    year: str | None = None,
    city_name: str = "ROI",
    margin_m: float = 200.0,
) -> Path:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from shapely.geometry import box

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model + stats ────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt
    feature_dim = state["fc.weight"].shape[1]
    model = LinearProbe(feature_dim, num_classes)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint.name}  feature_dim={feature_dim}")

    stats = np.load(stats_file)
    feat_mean = stats["mean"].astype(np.float32)
    feat_std = stats["std"].astype(np.float32)
    logger.info(f"Loaded stats: mean/std dim={feat_mean.shape[0]}")

    # ── Tile spatial index ────────────────────────────────────────────────────
    tile_paths, tree = _build_tile_index(embedding_dir, embedding_name, year=year)
    roi_geom = box(*bbox)
    idxs = tree.query(roi_geom)
    if len(idxs) == 0:
        raise RuntimeError(f"No embedding tiles found for bbox {bbox}")
    matched_paths = [tile_paths[i] for i in idxs]
    logger.info(f"Found {len(matched_paths)} tile(s) intersecting the ROI")

    # ── coop valid-bbox map (prevent UTM zone overhang contamination) ─────────
    path_to_valid_bbox: dict[Path, tuple] = {}
    if embedding_name == "alpha_earth_coop":
        import geopandas as _gpd
        _idx = _gpd.read_file(
            embedding_dir / "aef_index.gpkg",
            where=f"year = {int(year)}" if year else "",
        )
        _name_to_bounds = {
            Path(r["path"]).name: (
                r["wgs84_west"], r["wgs84_south"],
                r["wgs84_east"], r["wgs84_north"],
            )
            for _, r in _idx.iterrows()
        }
        for p in matched_paths:
            if p.name in _name_to_bounds:
                path_to_valid_bbox[p] = _name_to_bounds[p.name]

    # ── Auto-detect output CRS / resolution ───────────────────────────────────
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

    extract_px = max(1, round(patch_physical_res_m / embedding_res_m))
    if out_res is not None:
        resolved_res = out_res
    elif resolved_crs == first_crs:
        resolved_res = extract_px * embedding_res_m
    else:
        resolved_res = _meters_to_out_res(
            extract_px * embedding_res_m, first_crs, resolved_crs, lon_c, lat_c
        )
    logger.info(
        f"embedding_res={embedding_res_m}m  extract_px={extract_px}  "
        f"output_res={resolved_res:.4g}  output_crs={resolved_crs}"
    )

    # ── Output raster ─────────────────────────────────────────────────────────
    out_transform, out_H, out_W = _setup_output(bbox, resolved_crs, resolved_res)
    raster = np.zeros((out_H, out_W), dtype=np.uint8)
    logger.info(f"Output raster: {out_H}×{out_W} px")

    # ── Process each tile ─────────────────────────────────────────────────────
    n_done = n_skip = 0
    for i, tile_path in enumerate(matched_paths):
        logger.info(f"Tile {i + 1}/{len(matched_paths)}: {tile_path.name}")
        result = _open_and_clip(
            tile_path, bbox, margin_m=margin_m,
            dequantize_fn=dequantize_fn,
            valid_bbox_4326=path_to_valid_bbox.get(tile_path),
        )
        if result is None:
            logger.warning("  Skipped — no valid data after clip")
            n_skip += 1
            continue

        arr, tile_crs, tile_transform = result
        C, H_tile, W_tile = arr.shape
        logger.info(f"  {C}ch × {H_tile}×{W_tile} px  crs={tile_crs}")

        pred = _sliding_window_lp(
            model, feat_mean, feat_std, arr,
            extract_px, device, num_classes, batch_size, pooling,
        )

        # Derive tile-resolution transform for the patch-level output grid
        from rasterio.transform import Affine
        patch_res = extract_px * abs(tile_transform.a)
        patch_transform = Affine(
            patch_res, 0.0, tile_transform.c,
            0.0, -patch_res, tile_transform.f,
        )

        pred_1idx = (pred.astype(np.uint16) + 1).astype(np.uint8)
        tmp = np.zeros((1, out_H, out_W), dtype=np.uint8)
        reproject(
            source=pred_1idx[None],
            destination=tmp,
            src_transform=patch_transform,
            src_crs=tile_crs,
            dst_transform=out_transform,
            dst_crs=resolved_crs,
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
        np.copyto(raster, tmp[0], where=tmp[0] > 0)
        n_done += 1

    logger.info(f"Tiles processed: {n_done} done, {n_skip} skipped")

    # ── Save GeoTIFF ──────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(output_path), "w", driver="GTiff",
        height=out_H, width=out_W, count=1, dtype="uint8",
        crs=resolved_crs, transform=out_transform, nodata=0,
    ) as dst:
        dst.write(raster, 1)
    logger.info(f"Saved GeoTIFF: {output_path}")

    from utils.plot_lcz import save_lcz_map
    png_path = output_path.with_suffix(".png")
    save_lcz_map(raster, f"LCZ Linear Probe — {city_name}", png_path)
    logger.info(f"Saved PNG: {png_path}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ROI prediction map from a linear probe checkpoint."
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--stats-file", required=True, type=Path,
                   help="NPZ file with 'mean' and 'std' arrays (training-set stats).")
    p.add_argument("--embedding-name", required=True,
                   choices=["tessera", "tesserav1.1", "tesserav1.1_global",
                            "alpha_earth", "alpha_earth_coop", "seamless"])
    p.add_argument("--embedding-dir", required=True, type=Path)
    p.add_argument("--year", default=None)
    p.add_argument("--pooling", choices=["gap", "mean_std"], default="gap",
                   help="Must match the pooling used during training.")
    p.add_argument("--num-classes", type=int, default=17)
    p.add_argument("--patch-physical-res", type=float, default=320.0,
                   help="Physical side of one training patch in metres (default 320).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dequantize", action="store_true",
                   help="Auto-applied for alpha_earth_coop and seamless.")
    p.add_argument("--out-crs", default=None)
    p.add_argument("--out-res", type=float, default=None)
    p.add_argument("--margin-m", type=float, default=200.0)

    loc = p.add_mutually_exclusive_group(required=True)
    loc.add_argument("--bbox", default=None,
                     help="west,south,east,north in EPSG:4326")
    loc.add_argument("--city", default=None)
    loc.add_argument("--smod-id", default=None)

    p.add_argument("--bounds-csv", type=Path,
                   default=Path(__file__).parent.parent / "data" / "guppd_bounds.csv")
    p.add_argument("--city-name", default=None)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--accelerator", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.bbox is not None:
        parts = [float(v) for v in args.bbox.split(",")]
        bbox = tuple(parts)
        city_name = args.city_name or "ROI"
    else:
        import pandas as pd
        df = pd.read_csv(args.bounds_csv)
        if args.city is not None:
            mask = df["JRC_NAME_MAIN"].str.lower() == args.city.lower()
        else:
            mask = df["SMOD_ID"].astype(str) == str(args.smod_id)
        row = df[mask].iloc[0]
        bbox = (float(row["minx"]), float(row["miny"]), float(row["maxx"]), float(row["maxy"]))
        city_name = args.city_name or str(row["JRC_NAME_MAIN"])
        logger.info(f"Resolved bbox for '{city_name}': {bbox}")

    if args.accelerator == "cpu":
        device = torch.device("cpu")
    elif args.accelerator == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    need_dequant = args.dequantize or args.embedding_name in {"alpha_earth_coop", "seamless"}
    dequantize_fn = None
    if need_dequant:
        if args.embedding_name == "seamless":
            from dequantize_embeddings import dequantize_esd
            dequantize_fn = dequantize_esd
        else:
            from dequantize_embeddings import dequantize_alphaearth_embeddings
            dequantize_fn = dequantize_alphaearth_embeddings

    infer_roi_lp(
        checkpoint=args.checkpoint,
        stats_file=args.stats_file,
        embedding_name=args.embedding_name,
        embedding_dir=args.embedding_dir,
        bbox=bbox,
        output_path=args.output,
        pooling=args.pooling,
        num_classes=args.num_classes,
        patch_physical_res_m=args.patch_physical_res,
        batch_size=args.batch_size,
        device=device,
        dequantize_fn=dequantize_fn,
        out_crs=args.out_crs,
        out_res=args.out_res,
        year=args.year,
        city_name=city_name,
        margin_m=args.margin_m,
    )


if __name__ == "__main__":
    main()
