"""Semantic segmentation (U-Net) trained on grid tile embeddings.

For each city the script:
  1. Reads {city}/{output_name}/{year}/{split}/{city}_{grid_id}.npy
  2. Loads the per-tile label mask from:
       gpkg (default): rasterises patches_reference_{city}_split.gpkg polygons
                       that fall in this tile → (H, W) label mask
       tif:            clips patches_reference_{city}.tif to tile bounds and
                       resizes to (H, W) by nearest-neighbour
  3. Trains a U-Net with the grid-based train/val/test split and logs to WandB.

Label convention (matching train_unet.py):
  raw 1-17 → 0-16 (class index), raw 0 (nodata) → -1 (ignore_index)

Example (single city, AlphaEarth, labels from GeoPackage):
    python src/semantic_segmentation.py \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi \\
        --output-name AlphaEarth --year 2017 \\
        --label-source gpkg \\
        --preset large --batch-size 16 --num-workers 4 \\
        --max-epochs 50 \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl

Example (multiple cities, GeoTessera, labels from raster TIF):
    python src/semantic_segmentation.py \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi Paris Berlin \\
        --output-name GeoTessera --year 2017 \\
        --label-source tif \\
        --preset base --batch-size 8 --num-workers 4 \\
        --max-epochs 50 \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.transform import from_origin as rio_from_origin
from pyproj import Transformer
from shapely import from_wkt
from shapely.ops import transform as shapely_transform
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, JaccardIndex

# ── src/ must be on sys.path (run from repo root) ────────────────────────────

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train_unet import UNet, LCZUNetModule, _run_unet_training_loop, augment_batch
from utils.plot_lcz import save_lcz_map

# ── Constants ─────────────────────────────────────────────────────────────────

_FILENAME_RE = re.compile(r"^(.+)_(\d+)\.npy$")   # {city}_{grid_id}.npy


# ── Label helpers ─────────────────────────────────────────────────────────────

def _rasterize_polys(tile_geom, polys: list, out_shape: tuple[int, int]) -> np.ndarray:
    """Burn shapely polygon/class pairs into a (H, W) uint8 label mask.

    tile_geom: shapely geometry in tile CRS (provides burn extent).
    polys: list of (shapely_geom_in_tile_crs, lcz_class_int).
    Returns array with 1-17 (class) or 0 (nodata).
    """
    H, W = out_shape
    if not polys:
        return np.zeros((H, W), dtype=np.uint8)
    minx, miny, maxx, maxy = tile_geom.bounds
    transform = rio_from_bounds(minx, miny, maxx, maxy, W, H)
    shapes = [(g, int(c)) for g, c in polys]
    return rio_rasterize(
        shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8,
    )


def _clip_tif(tile_geom, tile_crs: str, tif_path: Path, out_shape: tuple[int, int]) -> np.ndarray:
    """Clip the label TIF to the tile footprint, resize to out_shape (nearest-neighbour).

    Returns (H, W) uint8 with 1-17 (class) or 0 (nodata).
    """
    import rasterio
    import rasterio.mask
    from PIL import Image

    H, W = out_shape
    with rasterio.open(tif_path) as src:
        tif_crs = str(src.crs)
        geom_tif = tile_geom
        if tif_crs != tile_crs:
            t = Transformer.from_crs(tile_crs, tif_crs, always_xy=True)
            geom_tif = shapely_transform(t.transform, tile_geom)
        try:
            out, _ = rasterio.mask.mask(src, [geom_tif], crop=True, nodata=0)
            data = out[0].astype(np.uint8)
        except Exception:
            return np.zeros((H, W), dtype=np.uint8)

    if data.shape != (H, W):
        data = np.array(Image.fromarray(data).resize((W, H), Image.NEAREST))
    return data


# ── Per-city item builder ─────────────────────────────────────────────────────

def _build_city_items(
    city_dir: Path,
    output_name: str,
    year: str,
    label_source: str,
    label_col: str,
) -> tuple[list, dict]:
    """Scan npy files for one city and build per-tile item tuples.

    Returns:
        items: list of (npy_path, tile_geom, tile_crs, polys_or_none, tif_path_or_none)
        split_map: dict {npy_path: split} for DataModule split filtering
    """
    city = city_dir.name
    emb_base = city_dir / output_name / year
    grid_gpkg = city_dir / f"{city}_grid.gpkg"
    split_gpkg = city_dir / f"patches_reference_{city}_split.gpkg"
    tif_path = city_dir / f"patches_reference_{city}.tif"

    if not grid_gpkg.exists():
        logger.warning(f"  {city}: {grid_gpkg.name} missing — skipping")
        return [], {}
    if not emb_base.exists():
        logger.warning(f"  {city}: {emb_base} missing — skipping")
        return [], {}

    grid_gdf = gpd.read_file(grid_gpkg)
    tile_crs = str(grid_gdf.crs)
    # Only train/val/test on tiles within the coverage threshold (is_valid=True)
    valid_gdf = grid_gdf[grid_gdf["is_valid"]] if "is_valid" in grid_gdf.columns else grid_gdf
    id2geom = {int(r["grid_id"]): r.geometry for _, r in valid_gdf.iterrows()}

    # Build label payload
    if label_source == "gpkg":
        if not split_gpkg.exists():
            logger.warning(f"  {city}: {split_gpkg.name} missing — skipping")
            return [], {}
        sdf = gpd.read_file(split_gpkg)
        if str(sdf.crs) != tile_crs:
            sdf = sdf.to_crs(tile_crs)
        id2polys: dict[int, list] = {}
        for _, r in sdf.iterrows():
            gid = int(r["grid_id"])
            id2polys.setdefault(gid, []).append((r.geometry, int(r[label_col])))
        tif_ref = None
    else:
        if not tif_path.exists():
            logger.warning(f"  {city}: {tif_path.name} missing — skipping")
            return [], {}
        id2polys = {}
        tif_ref = tif_path

    items = []
    split_map = {}
    for split in ("train", "val", "test"):
        split_dir = emb_base / split
        if not split_dir.exists():
            continue
        for npy_path in sorted(split_dir.glob(f"{city}_*.npy")):
            m = _FILENAME_RE.match(npy_path.name)
            if m is None:
                continue
            grid_id = int(m.group(2))
            geom = id2geom.get(grid_id)
            if geom is None:
                continue
            polys = id2polys.get(grid_id, []) if label_source == "gpkg" else []
            items.append((npy_path, geom, tile_crs, polys, tif_ref))
            split_map[npy_path] = split

    n_valid = len(valid_gdf)
    n_total = len(grid_gdf)
    logger.info(f"  {city}: {len(items)} tiles ({n_valid}/{n_total} valid)")
    return items, split_map


# ── Dataset ───────────────────────────────────────────────────────────────────

def _dequantize(arr: np.ndarray) -> np.ndarray:
    """AlphaEarth coop dequantisation: sign(v) × (|v| / 127.5)²."""
    return np.sign(arr) * (np.abs(arr) / 127.5) ** 2


class GridSegDataset(Dataset):
    """Grid tile dataset for U-Net segmentation.

    Returns {"image": (C, H, W) float32, "mask": (H, W) long}
    Label convention: raw 1-17 → 0-16, raw 0 → -1 (ignore_index=-1).
    """

    def __init__(
        self,
        items: list,          # (npy_path, tile_geom, tile_crs, polys, tif_path_or_None)
        label_source: str,    # "gpkg" or "tif"
        dequantize: bool = False,
    ) -> None:
        self.items = items
        self.label_source = label_source
        self.dequantize = dequantize

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        npy_path, tile_geom, tile_crs, polys, tif_ref = self.items[idx]

        arr = np.load(npy_path).astype(np.float32)   # (C, H, W)
        if self.dequantize:
            arr = _dequantize(arr)
        _, H, W = arr.shape
        image = torch.from_numpy(arr)

        if self.label_source == "gpkg":
            raw = _rasterize_polys(tile_geom, polys, (H, W))
        else:
            raw = _clip_tif(tile_geom, tile_crs, tif_ref, (H, W))

        # 0→-1, 1-17→0-16
        mask = torch.from_numpy(raw.astype(np.int64)) - 1

        return {"image": image, "mask": mask}


# ── DataModule ────────────────────────────────────────────────────────────────

class GridSegDataModule:
    """Minimal DataModule for _run_unet_training_loop compatibility."""

    def __init__(
        self,
        all_items: list,
        split_map: dict,       # {npy_path: split}
        label_source: str,
        batch_size: int,
        num_workers: int,
        dequantize: bool = False,
    ) -> None:
        self.all_items = all_items
        self.split_map = split_map
        self.label_source = label_source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dequantize = dequantize

    def setup(self) -> None:
        def _for_split(s):
            return [it for it in self.all_items if self.split_map.get(it[0]) == s]

        self._train_ds = GridSegDataset(_for_split("train"), self.label_source, self.dequantize)
        self._val_ds   = GridSegDataset(_for_split("val"),   self.label_source, self.dequantize)
        self._test_ds  = GridSegDataset(_for_split("test"),  self.label_source, self.dequantize)
        logger.info(
            f"Dataset sizes — train: {len(self._train_ds)}, "
            f"val: {len(self._val_ds)}, test: {len(self._test_ds)}"
        )

    @staticmethod
    def _pad_batch(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad images and masks to the largest (H, W) in the batch.

        Some edge tiles are 1-pixel smaller than interior tiles due to the grid
        boundary falling at a sub-pixel position. Padding with 0 (embedding)
        and -1 (nodata ignore_index) keeps the shapes consistent.
        """
        # Use a common square size so that rot90 never produces mismatched shapes
        max_hw = max(
            max(b["image"].shape[-2] for b in batch),
            max(b["image"].shape[-1] for b in batch),
        )
        imgs, masks = [], []
        for b in batch:
            img, msk = b["image"], b["mask"]
            ph = max_hw - img.shape[-2]
            pw = max_hw - img.shape[-1]
            if ph or pw:
                img = F.pad(img,  (0, pw, 0, ph), value=0)
                msk = F.pad(msk,  (0, pw, 0, ph), value=-1)
            imgs.append(img)
            masks.append(msk)
        return torch.stack(imgs), torch.stack(masks)

    @staticmethod
    def _collate(batch: list) -> dict:
        images, masks = GridSegDataModule._pad_batch(batch)
        return {"image": images, "mask": masks}

    @staticmethod
    def _train_collate(batch: list) -> dict:
        images, masks = GridSegDataModule._pad_batch(batch)
        images, masks = augment_batch(images, masks)
        return {"image": images, "mask": masks}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._train_collate, drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate,
        )


# ── Inference helpers ────────────────────────────────────────────────────────

def _build_roi_meta(grid_gdf, sample_npy_path: Path):
    """Compute full-ROI raster dimensions and transform from the grid GDF.

    Returns (out_H, out_W, res, roi_minx, roi_maxy, transform, crs).
    """
    minx, miny, maxx, maxy = grid_gdf.total_bounds
    row = grid_gdf.iloc[0]
    b = row.geometry.bounds   # (tile_minx, tile_miny, tile_maxx, tile_maxy)
    arr = np.load(sample_npy_path, mmap_mode="r")
    _, H, W = arr.shape
    res_x = (b[2] - b[0]) / W
    res_y = (b[3] - b[1]) / H
    res = (res_x + res_y) / 2
    out_W = int(round((maxx - minx) / res))
    out_H = int(round((maxy - miny) / res))
    transform = rio_from_origin(minx, maxy, res, res)
    crs = str(grid_gdf.crs)
    return out_H, out_W, res, minx, maxy, transform, crs


def _save_geotiff(raster: np.ndarray, crs: str, transform, path: Path) -> None:
    """Write a uint8 raster to a GeoTIFF with nodata=0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=raster.shape[0], width=raster.shape[1],
        count=1, dtype="uint8", crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(raster, 1)


def _infer_city_unet(
    model,
    city_dir: Path,
    output_name: str,
    year: str,
    device,
    batch_size: int = 8,
    dequantize: bool = False,
) -> tuple[np.ndarray, str, object]:
    """Run U-Net inference over all grid tiles for one city.

    Returns (uint8 raster with values 1-17, CRS string, rasterio transform).
    """
    city = city_dir.name
    grid_gpkg = city_dir / f"{city}_grid.gpkg"
    if not grid_gpkg.exists():
        logger.warning(f"  {city}: {grid_gpkg.name} not found — skipping inference")
        return None, None, None

    grid_gdf = gpd.read_file(grid_gpkg)
    id2geom = {int(r["grid_id"]): r.geometry for _, r in grid_gdf.iterrows()}

    # Collect all npy paths across all splits
    emb_base = city_dir / output_name / year
    all_tiles: list[tuple[Path, int]] = []
    for split in ("train", "val", "test"):
        split_dir = emb_base / split
        if not split_dir.exists():
            continue
        for p in sorted(split_dir.glob(f"{city}_*.npy")):
            m = _FILENAME_RE.match(p.name)
            if m is None:
                continue
            gid = int(m.group(2))
            if gid in id2geom:
                all_tiles.append((p, gid))

    if not all_tiles:
        logger.warning(f"  {city}: no npy tiles found under {emb_base}")
        return None, None, None

    out_H, out_W, res, roi_minx, roi_maxy, transform, crs = _build_roi_meta(
        grid_gdf, all_tiles[0][0]
    )
    raster = np.zeros((out_H, out_W), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_tiles), batch_size):
            batch_tiles = all_tiles[i : i + batch_size]
            imgs, geoms = [], []
            for p, gid in batch_tiles:
                arr = np.load(p).astype(np.float32)
                if dequantize:
                    arr = _dequantize(arr)
                imgs.append(torch.from_numpy(arr))
                geoms.append(id2geom[gid])

            # Pad to uniform (H, W) within this mini-batch
            max_h = max(t.shape[1] for t in imgs)
            max_w = max(t.shape[2] for t in imgs)
            padded = torch.zeros(len(imgs), imgs[0].shape[0], max_h, max_w)
            for j, t in enumerate(imgs):
                padded[j, :, : t.shape[1], : t.shape[2]] = t
            padded = padded.to(device)

            logits = model(padded)   # (B, num_classes, H, W)
            preds = logits.argmax(dim=1).cpu().numpy()   # (B, H, W), 0-indexed

            for j, (_, gid) in enumerate(batch_tiles):
                geom = geoms[j]
                tx0, ty0, tx1, ty1 = geom.bounds
                ph, pw = imgs[j].shape[1], imgs[j].shape[2]
                pred = (preds[j, :ph, :pw] + 1).astype(np.uint8)  # 0-idx → 1-17
                col_off = int(round((tx0 - roi_minx) / res))
                row_off = int(round((roi_maxy - ty1) / res))
                r0 = max(0, row_off)
                c0 = max(0, col_off)
                r1 = min(out_H, r0 + ph)
                c1 = min(out_W, c0 + pw)
                raster[r0:r1, c0:c1] = pred[: r1 - r0, : c1 - c0]

    return raster, crs, transform


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a U-Net segmentation model on grid tile embeddings."
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Data")
    g.add_argument("--cities-dir", required=True, type=Path,
                   help="Root directory with one subfolder per city.")
    g.add_argument("--cities", nargs="+", default=None,
                   help="City names to include (default: all with grid GDF).")
    g.add_argument("--output-name", required=True,
                   help="Embedding folder name inside each city dir (e.g. AlphaEarth).")
    g.add_argument("--year", required=True,
                   help="Year subfolder (e.g. 2017).")
    g.add_argument("--label-source", choices=["gpkg", "tif"], default="gpkg",
                   help="Label source: 'gpkg' (rasterise polygon GeoPackage) or "
                        "'tif' (clip raster TIF). Default: gpkg.")
    g.add_argument("--label-col", default="LCZ_class",
                   help="Column name in the GeoPackage for LCZ class (default: LCZ_class).")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Model")
    g.add_argument("--preset", choices=["nano","small","base","medium","large"],
                   default="large",
                   help="U-Net size preset (default: large).")
    g.add_argument("--num-classes", type=int, default=17,
                   help="Number of output classes (default: 17).")
    g.add_argument("--bottleneck-dropout", type=float, default=0.3,
                   help="Bottleneck dropout probability (default: 0.3).")

    # ── Loss ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Loss")
    g.add_argument("--dice-weight", type=float, default=0.5,
                   help="Weight of Dice loss (0 = CE-only, 1 = Dice-only). Default: 0.5.")

    # ── Training ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Training")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--max-epochs", type=int, default=50)
    g.add_argument("--early-stopping-patience", type=int, default=10)
    g.add_argument("--seed", type=int, default=411)
    g.add_argument("--accelerator", choices=["auto","cpu","cuda","mps"], default="auto")

    # ── Logging ───────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Logging")
    g.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for checkpoints and WandB run folders.")
    g.add_argument("--wandb-project", default="lcz-classification-dl")
    g.add_argument("--wandb-entity", default="phd-thesis-team")
    g.add_argument("--no-wandb", action="store_true",
                   help="Disable WandB logging.")
    g.add_argument("--run-name", default=None,
                   help="Optional WandB run name override.")
    g.add_argument("--dequantize", action="store_true",
                   help="Apply AlphaEarth dequantisation (sign(v)×(|v|/127.5)²) when loading npy tiles.")
    g.add_argument("--checkpoint", type=Path, default=None,
                   help="Load model weights from this .pt file and skip training (inference only).")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.accelerator == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.accelerator)
    logger.info(f"Device: {device}")

    # ── Collect cities ────────────────────────────────────────────────────────
    cities_dir = args.cities_dir
    city_dirs = sorted(d for d in cities_dir.iterdir() if d.is_dir())
    if args.cities:
        city_dirs = [d for d in city_dirs if d.name in args.cities]
        if not city_dirs:
            logger.error(f"None of {args.cities} found in {cities_dir}")
            raise SystemExit(1)

    # ── Build item lists ──────────────────────────────────────────────────────
    all_items: list = []
    split_map: dict = {}
    for city_dir in city_dirs:
        items, sm = _build_city_items(
            city_dir, args.output_name, args.year,
            args.label_source, args.label_col,
        )
        all_items.extend(items)
        split_map.update(sm)

    if not all_items:
        logger.error("No items found. Check --cities-dir, --output-name, --year.")
        raise SystemExit(1)
    logger.info(f"Total tiles: {len(all_items)}")

    # ── Detect in_channels from first npy ─────────────────────────────────────
    in_channels = int(np.load(all_items[0][0], mmap_mode="r").shape[0])
    logger.info(f"Detected in_channels = {in_channels}")

    # ── Model ─────────────────────────────────────────────────────────────────
    depth, base_features = UNet.PRESETS[args.preset]
    model = UNet(
        in_channels, args.num_classes,
        depth=depth, base_features=base_features,
        bottleneck_dropout=args.bottleneck_dropout,
    )
    task = LCZUNetModule(
        model, args.num_classes,
        lr=args.lr, weight_decay=args.weight_decay,
        dice_weight=args.dice_weight, max_epochs=args.max_epochs,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"UNet '{args.preset}': depth={depth}, base_features={base_features}, "
                f"params={n_params:,}")

    # ── DataModule ────────────────────────────────────────────────────────────
    datamodule = GridSegDataModule(
        all_items, split_map, args.label_source,
        args.batch_size, args.num_workers,
        dequantize=args.dequantize,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    city_names = [d.name for d in city_dirs]
    run_cfg = dict(
        task="segmentation",
        embedding=args.output_name,
        cities=city_names,
        year=args.year,
        label_source=args.label_source,
        preset=args.preset,
        in_channels=in_channels,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dice_weight=args.dice_weight,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        n_params=n_params,
        data_source="grid_tiles",
    )

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=str(args.output_dir),
            config=run_cfg,
            name=args.run_name,
        )
        run_dir = args.output_dir / wandb.run.name
    else:
        run_name = args.run_name or f"unet_{args.preset}_{'_'.join(city_names[:3])}"
        run_dir = args.output_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"unet_{args.preset}_{args.output_name}_{'_'.join(city_names[:3])}"

    # ── Train (or load checkpoint) ────────────────────────────────────────────
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        task.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        task = task.to(device)
        ckpt_path = args.checkpoint
    else:
        task, ckpt_path = _run_unet_training_loop(
            task_module=task,
            datamodule=datamodule,
            device=device,
            max_epochs=args.max_epochs,
            early_stopping_patience=args.early_stopping_patience,
            run_dir=run_dir,
            model_name=model_name,
        )
    logger.info(f"Best checkpoint: {ckpt_path}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    task.eval()
    metric_kw = dict(task="multiclass", num_classes=args.num_classes, ignore_index=-1)
    test_miou = JaccardIndex(**metric_kw, average="macro").to(device)
    test_acc  = Accuracy(**metric_kw).to(device)
    test_loss_total = 0.0
    n_test_batches  = 0

    datamodule.setup()   # re-create test dataloader after training
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if (masks != -1).sum() == 0:
                continue
            logits = task(imgs)
            loss, _, _ = task._loss(logits, masks)
            preds = logits.argmax(dim=1)
            test_miou(preds, masks)
            test_acc(preds, masks)
            test_loss_total += loss.item()
            n_test_batches  += 1

    if n_test_batches > 0:
        miou = test_miou.compute().item()
        acc  = test_acc.compute().item()
        avg_loss = test_loss_total / n_test_batches
        logger.info(f"Test — mIoU: {miou:.4f}  Acc: {acc:.4f}  Loss: {avg_loss:.4f}")
        if not args.no_wandb and wandb.run:
            wandb.log({"test_miou": miou, "test_acc": acc, "test_loss": avg_loss})
    else:
        logger.warning("No test batches with valid labels found.")

    if not args.no_wandb and wandb.run:
        wandb.finish()

    # ── Per-city inference raster + map ──────────────────────────────────────
    logger.info("Running full-ROI inference …")
    task.model.eval()
    for city_dir in city_dirs:
        city = city_dir.name
        raster, crs, transform = _infer_city_unet(
            task.model, city_dir, args.output_name, args.year, device, args.batch_size,
            dequantize=args.dequantize,
        )
        if raster is None:
            continue
        tif_path = run_dir / f"{run_dir.name}_unet-{args.preset}-segmentation-prediction_{city}.tif"
        png_path = run_dir / f"{run_dir.name}_unet-{args.preset}-segmentation-prediction_{city}.png"
        _save_geotiff(raster, crs, transform, tif_path)
        save_lcz_map(
            raster,
            f"U-Net {args.preset} — {city} ({args.output_name} {args.year})",
            png_path,
        )
        logger.info(f"  {city}: saved {tif_path.name}")

    logger.info(f"Run complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
