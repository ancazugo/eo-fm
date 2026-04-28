"""Patch-level classification (ResNet) on So2Sat patches with grid-based split.

For each city the script:
  1. Builds a patch index by scanning the So2Sat embedding folders:
       {so2sat_dir}/{training,validation,testing}/{output_name}/{year}/patch_{id}.npy
  2. Loads patches_reference_{city}_split.gpkg (produced by create_city_grids.py)
     which has: patch_id, LCZ_class (1-17), grid_id, split (train/val/test)
  3. For each patch in the GDF, looks up its npy file in the index.
     The grid-based 'split' column (not the original So2Sat split) determines
     train / val / test assignment.
  4. Trains a ResNet patch classifier and logs to WandB.

Label convention (matching train_resnet.py):
  LCZ_class 1-17 → 0-16 (class index)

Example (single city, AlphaEarth):
    python src/patch_classification.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi \\
        --output-name AlphaEarth --year 2017 \\
        --preset large --patch-size 32 \\
        --batch-size 64 --num-workers 4 --max-epochs 50 \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl

Example (multiple cities, GeoTessera):
    python src/patch_classification.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --cities-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4/cities \\
        --cities Nairobi Paris Berlin \\
        --output-name GeoTessera --year 2017 \\
        --preset base --patch-size 32 \\
        --batch-size 64 --num-workers 4 --max-epochs 50 \\
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
from rasterio.transform import from_origin as rio_from_origin
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── src/ must be on sys.path (run from repo root) ────────────────────────────

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train_resnet import (
    RESNET_PRESETS,
    LCZResNetModule,
    build_resnet,
    augment_images,
    _run_resnet_training_loop,
)
from utils.constants import lcz_dict
from utils.plot_lcz import save_lcz_map


# ── Patch index builder ───────────────────────────────────────────────────────

def _build_patch_index(
    so2sat_dir: Path,
    output_name: str,
    year: str,
) -> dict[str, Path]:
    """Scan all three original So2Sat split dirs and return {patch_id: Path}.

    The original So2Sat split (training/validation/testing) is irrelevant here;
    we collect every patch npy once so we can re-split by the grid assignment.
    """
    index: dict[str, Path] = {}
    for orig_split in ("training", "validation", "testing"):
        d = so2sat_dir / orig_split / output_name / year
        if not d.exists():
            continue
        for p in sorted(d.glob("patch_*.npy")):
            pid = p.stem[len("patch_"):]   # "000004"
            index[pid] = p
    logger.info(f"Patch index: {len(index)} npy files found under {so2sat_dir}")
    return index


# ── Per-city item builder ─────────────────────────────────────────────────────

def _build_city_items(
    cities_dir: Path,
    city: str,
    patch_index: dict[str, Path],
    label_col: str = "LCZ_class",
) -> list[tuple]:
    """Build (npy_path, label_int, split) tuples for one city.

    Uses patches_reference_{city}_split.gpkg as the authoritative source of
    patch_ids and their grid-based train/val/test split assignment.
    """
    split_gpkg = cities_dir / city / f"patches_reference_{city}_split.gpkg"
    if not split_gpkg.exists():
        logger.warning(f"  {city}: patches_reference_{city}_split.gpkg not found — skipping")
        return []

    sdf = gpd.read_file(split_gpkg)
    items: list[tuple] = []
    n_missing = 0

    for _, row in sdf.iterrows():
        pid = str(row["patch_id"])
        path = patch_index.get(pid)
        if path is None:
            n_missing += 1
            continue
        label = int(row[label_col]) - 1   # 1-17 → 0-16
        items.append((path, label, str(row["split"])))

    logger.info(
        f"  {city}: {len(items)} patches matched "
        f"({n_missing} patch_ids had no npy)"
    )
    return items


# ── Dataset ───────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """So2Sat patch dataset for ResNet classification.

    Returns {"image": (C, patch_size, patch_size) float32, "label": scalar long}
    Label: 0-16 (valid LCZ class).

    When sub_patch_size is set, each parent patch is tiled into sub-patches of
    that size (with sub_patch_stride step). Each sub-patch inherits the parent
    label. The full-patch path (sub_patch_size=None) is identical to before.
    """

    def __init__(
        self,
        items: list,        # (npy_path, label_int, split)
        patch_size: int,
        sub_patch_size: int | None = None,
        sub_patch_stride: int | None = None,
        dequantize: bool = False,
    ) -> None:
        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.sub_patch_stride = sub_patch_stride or sub_patch_size
        self.dequantize = dequantize

        if sub_patch_size is None:
            self.expanded = [(path, label, None, None) for path, label, _ in items]
        else:
            stride = self.sub_patch_stride
            self.expanded = []
            for path, label, _ in items:
                _, H, W = np.load(path, mmap_mode="r").shape
                for r in range(0, H - sub_patch_size + 1, stride):
                    for c in range(0, W - sub_patch_size + 1, stride):
                        self.expanded.append((path, label, r, c))

    def __len__(self) -> int:
        return len(self.expanded)

    def __getitem__(self, idx: int) -> dict:
        path, label, r, c = self.expanded[idx]

        arr = np.load(path).astype(np.float32)   # (C, H, W)
        arr = np.nan_to_num(arr, nan=0.0)
        if self.dequantize:
            arr = np.sign(arr) * (np.abs(arr) / 127.5) ** 2

        if r is None:
            image = torch.from_numpy(arr)
        else:
            image = torch.from_numpy(arr[:, r:r + self.sub_patch_size, c:c + self.sub_patch_size])

        # Resize to fixed patch size if needed
        if image.shape[-2:] != (self.patch_size, self.patch_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.patch_size, self.patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ── DataModule ────────────────────────────────────────────────────────────────

class PatchDataModule:
    """Minimal DataModule for _run_resnet_training_loop compatibility."""

    def __init__(
        self,
        all_items: list,
        patch_size: int,
        batch_size: int,
        num_workers: int,
        sub_patch_size: int | None = None,
        sub_patch_stride: int | None = None,
        dequantize: bool = False,
    ) -> None:
        self.all_items = all_items
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sub_patch_size = sub_patch_size
        self.sub_patch_stride = sub_patch_stride
        self.dequantize = dequantize

    def setup(self) -> None:
        def _for_split(s: str) -> list:
            return [it for it in self.all_items if it[2] == s]

        kw = dict(sub_patch_size=self.sub_patch_size, sub_patch_stride=self.sub_patch_stride,
                  dequantize=self.dequantize)
        self._train_ds = PatchDataset(_for_split("train"), self.patch_size, **kw)
        self._val_ds   = PatchDataset(_for_split("val"),   self.patch_size, **kw)
        self._test_ds  = PatchDataset(_for_split("test"),  self.patch_size, **kw)
        logger.info(
            f"Dataset sizes — train: {len(self._train_ds)}, "
            f"val: {len(self._val_ds)}, test: {len(self._test_ds)}"
        )

    @staticmethod
    def _collate(batch: list) -> dict:
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        }

    def _train_collate(self, batch: list) -> dict:
        images = torch.stack([b["image"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        images = augment_images(images)
        return {"image": images, "label": labels}

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

_TILE_RE = re.compile(r"^(.+)_(\d+)\.npy$")


def _build_roi_meta(grid_gdf, sample_npy_path: Path):
    """Compute full-ROI raster dimensions and transform from the grid GDF.

    Returns (out_H, out_W, res, roi_minx, roi_maxy, transform, crs).
    """
    minx, miny, maxx, maxy = grid_gdf.total_bounds
    row = grid_gdf.iloc[0]
    b = row.geometry.bounds
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


def _infer_city_resnet(
    model,
    city_dir: Path,
    output_name: str,
    year: str,
    patch_size: int,
    device,
    batch_size: int = 64,
    sub_patch_size: int | None = None,
    sub_patch_stride: int | None = None,
    dequantize: bool = False,
) -> tuple[np.ndarray, str, object]:
    """Run ResNet inference over all grid tiles for one city (sliding window).

    Each window gets a single class prediction. Window size is sub_patch_size
    when set, otherwise patch_size. Stride defaults to window size.
    Returns (uint8 raster with values 1-17, CRS string, rasterio transform).
    """
    city = city_dir.name
    grid_gpkg = city_dir / f"{city}_grid.gpkg"
    if not grid_gpkg.exists():
        logger.warning(f"  {city}: {grid_gpkg.name} not found — skipping inference")
        return None, None, None

    grid_gdf = gpd.read_file(grid_gpkg)
    id2geom = {int(r["grid_id"]): r.geometry for _, r in grid_gdf.iterrows()}

    emb_base = city_dir / output_name / year
    all_tiles: list[tuple[Path, int]] = []
    for split in ("train", "val", "test"):
        split_dir = emb_base / split
        if not split_dir.exists():
            continue
        for p in sorted(split_dir.glob(f"{city}_*.npy")):
            m = _TILE_RE.match(p.name)
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

    win_size   = sub_patch_size if sub_patch_size is not None else patch_size
    win_stride = sub_patch_stride if sub_patch_stride is not None else win_size

    model.eval()
    with torch.no_grad():
        for npy_path, gid in all_tiles:
            arr = np.load(npy_path).astype(np.float32)  # (C, H, W)
            if dequantize:
                arr = np.sign(arr) * (np.abs(arr) / 127.5) ** 2
            C, H, W = arr.shape
            pred_tile = np.zeros((H, W), dtype=np.uint8)

            # Collect all windows; windows are always exactly win_size × win_size
            windows = []
            for r in range(0, H - win_size + 1, win_stride):
                for c in range(0, W - win_size + 1, win_stride):
                    windows.append((r, c))

            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i : i + batch_size]
                batch_imgs = []
                for r, c in batch_windows:
                    patch = arr[:, r : r + win_size, c : c + win_size]
                    batch_imgs.append(torch.from_numpy(patch))

                imgs_tensor = torch.stack(batch_imgs).to(device)
                # Resize to patch_size if needed (matches training resize)
                if imgs_tensor.shape[-1] != patch_size or imgs_tensor.shape[-2] != patch_size:
                    imgs_tensor = torch.nn.functional.interpolate(
                        imgs_tensor, size=(patch_size, patch_size), mode="bilinear",
                        align_corners=False,
                    )
                logits = model(imgs_tensor)        # (B, num_classes)
                preds = logits.argmax(dim=1).cpu().numpy()  # (B,) 0-indexed

                for (r, c), cls in zip(batch_windows, preds):
                    pred_tile[r : r + win_size, c : c + win_size] = int(cls) + 1  # 1-17

            geom = id2geom[gid]
            tx0, ty0, tx1, ty1 = geom.bounds
            col_off = int(round((tx0 - roi_minx) / res))
            row_off = int(round((roi_maxy - ty1) / res))
            r0 = max(0, row_off)
            c0 = max(0, col_off)
            r1 = min(out_H, r0 + H)
            c1 = min(out_W, c0 + W)
            raster[r0:r1, c0:c1] = pred_tile[: r1 - r0, : c1 - c0]

    return raster, crs, transform


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ResNet patch classifier on So2Sat patches "
                    "using the grid-based train/val/test split."
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Data")
    g.add_argument("--so2sat-dir", required=True, type=Path,
                   help="Root So2Sat directory that contains the "
                        "training/validation/testing subfolders with patch npy files.")
    g.add_argument("--cities-dir", required=True, type=Path,
                   help="Directory containing one subfolder per city "
                        "(each must have patches_reference_{city}_split.gpkg).")
    g.add_argument("--cities", nargs="+", default=None,
                   help="City names to include (default: all with split GDF).")
    g.add_argument("--output-name", required=True,
                   help="Embedding name used as subfolder in the So2Sat patch dirs "
                        "(e.g. AlphaEarth or GeoTessera).")
    g.add_argument("--year", required=True,
                   help="Year subfolder (e.g. 2017).")
    g.add_argument("--label-col", default="LCZ_class",
                   help="Column name for LCZ class in the split GDF (default: LCZ_class).")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Model")
    g.add_argument("--preset", choices=list(RESNET_PRESETS.keys()), default="large",
                   help="ResNet size preset (default: large → resnet152).")
    g.add_argument("--arch", default=None,
                   help="Override with any timm model name (e.g. resnet50).")
    g.add_argument("--num-classes", type=int, default=17,
                   help="Number of output classes (default: 17).")
    g.add_argument("--patch-size", type=int, default=32,
                   help="Resize patches to this square size before feeding ResNet (default: 32).")
    g.add_argument("--sub-patch-size", type=int, default=None,
                   help="If set, sample sub-patches of this size (pixels) from each parent patch "
                        "instead of using the full patch. Sub-patches inherit the parent label.")
    g.add_argument("--sub-patch-stride", type=int, default=None,
                   help="Stride (pixels) for sub-patch sampling (default: sub-patch-size, "
                        "i.e. non-overlapping).")
    g.add_argument("--head-dropout", type=float, default=0.0,
                   help="Dropout before the final FC layer (default: 0.0).")

    # ── Training ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Training")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--max-epochs", type=int, default=50)
    g.add_argument("--early-stopping-patience", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
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
                   help="Apply AlphaEarth dequantisation (sign(v)×(|v|/127.5)²) when loading npy patches.")
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

    # ── Build global patch index (scan all original split dirs once) ──────────
    patch_index = _build_patch_index(args.so2sat_dir, args.output_name, args.year)
    if not patch_index:
        logger.error(
            f"No patch npy files found under {args.so2sat_dir} "
            f"for output_name={args.output_name}, year={args.year}"
        )
        raise SystemExit(1)

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
    for city_dir in city_dirs:
        items = _build_city_items(
            cities_dir, city_dir.name, patch_index, args.label_col,
        )
        all_items.extend(items)

    if not all_items:
        logger.error("No items found. Check --so2sat-dir, --cities-dir, --output-name, --year.")
        raise SystemExit(1)

    split_counts = {s: sum(1 for _, _, sp in all_items if sp == s)
                    for s in ("train", "val", "test")}
    logger.info(f"Total patches: {len(all_items)}  splits: {split_counts}")

    # ── Detect in_channels from first npy ────────────────────────────────────
    in_channels = int(np.load(all_items[0][0], mmap_mode="r").shape[0])
    logger.info(f"Detected in_channels = {in_channels}")

    # ── Model ─────────────────────────────────────────────────────────────────
    arch = args.arch or RESNET_PRESETS[args.preset]
    model = build_resnet(
        arch=arch,
        in_channels=in_channels,
        num_classes=args.num_classes,
        head_dropout=args.head_dropout,
    )
    task = LCZResNetModule(
        model=model,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ResNet '{args.preset}' ({arch}): params={n_params:,}")

    # ── DataModule ────────────────────────────────────────────────────────────
    datamodule = PatchDataModule(
        all_items, args.patch_size, args.batch_size, args.num_workers,
        sub_patch_size=args.sub_patch_size,
        sub_patch_stride=args.sub_patch_stride,
        dequantize=args.dequantize,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    city_names = [d.name for d in city_dirs]
    run_cfg = dict(
        task="patch_classification",
        embedding=args.output_name,
        cities=city_names,
        year=args.year,
        preset=args.preset,
        arch=arch,
        in_channels=in_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        sub_patch_size=args.sub_patch_size,
        sub_patch_stride=args.sub_patch_stride,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        n_params=n_params,
        data_source="so2sat_patches",
        split_source="grid",
        **{f"{s}_patches": split_counts[s] for s in ("train", "val", "test")},
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
        run_name = args.run_name or f"resnet_{args.preset}_{'_'.join(city_names[:3])}"
        run_dir = args.output_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"resnet_{args.preset}_{args.output_name}_{'_'.join(city_names[:3])}"

    # ── Train (or load checkpoint) ────────────────────────────────────────────
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        task.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        task = task.to(device)
        ckpt_path = args.checkpoint
    else:
        task, ckpt_path = _run_resnet_training_loop(
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
    metric_kw  = dict(task="multiclass", num_classes=args.num_classes, ignore_index=-1)
    test_acc   = Accuracy(**metric_kw).to(device)
    test_f1    = MulticlassF1Score(
        num_classes=args.num_classes, average="macro", ignore_index=-1
    ).to(device)
    test_loss_total = 0.0
    n_test_batches  = 0
    all_preds, all_labels = [], []

    datamodule.setup()
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            if (labels != -1).sum() == 0:
                continue
            logits = task(imgs)
            loss   = task.ce_loss(logits, labels)
            preds  = logits.argmax(dim=1)
            test_acc(preds, labels)
            test_f1(preds, labels)
            test_loss_total += loss.item()
            n_test_batches  += 1
            valid = labels != -1
            all_preds.append((preds[valid] + 1).cpu().numpy())
            all_labels.append((labels[valid] + 1).cpu().numpy())

    if n_test_batches > 0:
        acc      = test_acc.compute().item()
        f1       = test_f1.compute().item()
        avg_loss = test_loss_total / n_test_batches
        logger.info(f"Test — Acc: {acc:.4f}  F1: {f1:.4f}  Loss: {avg_loss:.4f}")
        if not args.no_wandb and wandb.run:
            wandb.log({"test_acc": acc, "test_f1": f1, "test_loss": avg_loss})

        if all_preds:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
            display_labels = [lcz_dict.get(l, {}).get("name", str(l)) for l in present]
            cm = confusion_matrix(y_true, y_pred, labels=present)
            fig, ax = plt.subplots(figsize=(12, 10))
            ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(
                ax=ax, colorbar=True, xticks_rotation=45
            )
            ax.set_title(f"Test Confusion Matrix — {', '.join(city_names[:3])}")
            plt.tight_layout()
            cm_path = run_dir / "test_confusion_matrix.png"
            fig.savefig(cm_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Confusion matrix saved to {cm_path}")
    else:
        logger.warning("No test batches with valid labels found.")

    if not args.no_wandb and wandb.run:
        wandb.finish()

    # ── Per-city inference raster + map ──────────────────────────────────────
    logger.info("Running full-ROI inference …")
    task.model.eval()
    for city_dir in city_dirs:
        city = city_dir.name
        raster, crs, transform = _infer_city_resnet(
            task.model, city_dir, args.output_name, args.year,
            args.patch_size, device, args.batch_size,
            sub_patch_size=args.sub_patch_size,
            sub_patch_stride=args.sub_patch_stride,
            dequantize=args.dequantize,
        )
        if raster is None:
            continue
        tif_path = run_dir / f"{run_dir.name}_resnet-{args.preset}-classification-prediction_{city}.tif"
        png_path = run_dir / f"{run_dir.name}_resnet-{args.preset}-classification-prediction_{city}.png"
        _save_geotiff(raster, crs, transform, tif_path)
        save_lcz_map(
            raster,
            f"ResNet {args.preset} — {city} ({args.output_name} {args.year})",
            png_path,
        )
        logger.info(f"  {city}: saved {tif_path.name}")

    logger.info(f"Run complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
