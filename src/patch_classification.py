"""Patch-level classification (ResNet) on So2Sat patches.

Two split modes:

  Per-city (default): specify --cities-dir and --cities.
    Uses patches_reference_{city}_split.gpkg (grid-based split column: train/val/test).

  Global (--global-split): uses patches_reference_rxr.gpkg directly.
    The 'dataset' column (training/validation/testing) defines the split — no
    city selection needed, all 400 k+ patches across 51 cities are included.

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

Example (global split, AlphaEarthCoop):
    python src/patch_classification.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --global-split \\
        --output-name AlphaEarthCoop --year 2017 \\
        --preset large --patch-size 32 \\
        --batch-size 256 --num-workers 8 --max-epochs 50 \\
        --embedding-name alpha_earth_coop \\
        --embedding-dir /maps/acz25/phd-thesis-data/input/Google/AlphaEarth/coop \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/dl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassCohenKappa, MulticlassF1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── src/ must be on sys.path (run from repo root) ────────────────────────────

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from aspp import build_aspp
from infer_roi import infer_roi
from train_resnet import (
    MODEL_PRESETS,
    RESNET_PRESETS,
    LCZResNetModule,
    build_resnet,
    build_mlp,
    augment_images,
    _run_resnet_training_loop,
)
from utils.constants import lcz_dict


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


# ── Global item builder ───────────────────────────────────────────────────────

def _build_global_items(
    patches_gpkg: Path,
    patch_index: dict[str, Path],
    label_col: str = "LCZ_class",
) -> list[tuple]:
    """Build (npy_path, label_int, split) tuples from the global So2Sat GPKG.

    Uses the 'dataset' column ('training'/'validation'/'testing') and maps it
    to the 'train'/'val'/'test' strings expected by PatchDataModule.
    """
    _SPLIT_MAP = {"training": "train", "validation": "val", "testing": "test"}
    gdf = gpd.read_file(patches_gpkg)
    items: list[tuple] = []
    n_missing = 0
    for _, row in gdf.iterrows():
        pid = str(row["patch_id"])
        path = patch_index.get(pid)
        if path is None:
            n_missing += 1
            continue
        split = _SPLIT_MAP.get(str(row["dataset"]))
        if split is None:
            continue
        label = int(row[label_col]) - 1   # 1-17 → 0-16
        items.append((path, label, split))
    logger.info(
        f"Global split: {len(items)} patches matched "
        f"({n_missing} patch_ids had no npy)"
    )
    return items


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
        dequantize_fn=None,
    ) -> None:
        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.sub_patch_stride = sub_patch_stride or sub_patch_size
        self.dequantize_fn = dequantize_fn

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
        if self.dequantize_fn is not None:
            arr = self.dequantize_fn(arr)

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
        dequantize_fn=None,
    ) -> None:
        self.all_items = all_items
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sub_patch_size = sub_patch_size
        self.sub_patch_stride = sub_patch_stride
        self.dequantize_fn = dequantize_fn

    def setup(self) -> None:
        def _for_split(s: str) -> list:
            return [it for it in self.all_items if it[2] == s]

        kw = dict(sub_patch_size=self.sub_patch_size, sub_patch_stride=self.sub_patch_stride,
                  dequantize_fn=self.dequantize_fn)
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
    g.add_argument("--global-split", action="store_true",
                   help="Use the global patches_reference_rxr.gpkg with its original "
                        "training/validation/testing split instead of per-city split GeoPackages.")
    g.add_argument("--global-gpkg", type=Path, default=None,
                   help="Path to global patches GPKG "
                        "(default: {so2sat_dir}/patches_reference_rxr.gpkg). "
                        "Only used with --global-split.")
    g.add_argument("--cities-dir", required=False, default=None, type=Path,
                   help="Directory containing one subfolder per city "
                        "(each must have patches_reference_{city}_split.gpkg). "
                        "Required unless --global-split is set.")
    g.add_argument("--cities", nargs="+", default=None,
                   help="City names to include. In per-city mode: selects cities for training. "
                        "In global-split mode: selects cities for post-training inference only.")
    g.add_argument("--output-name", required=True,
                   help="Embedding name used as subfolder in the So2Sat patch dirs "
                        "(e.g. AlphaEarth or GeoTessera).")
    g.add_argument("--year", required=True,
                   help="Year subfolder (e.g. 2017).")
    g.add_argument("--label-col", default="LCZ_class",
                   help="Column name for LCZ class in the split GDF (default: LCZ_class).")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Model")
    g.add_argument("--family", choices=list(MODEL_PRESETS), default="resnet",
                   help="Model family (default: resnet).")
    g.add_argument("--preset", choices=["nano", "small", "base", "medium", "large"], default="large",
                   help="Size preset (default: large).")
    g.add_argument("--arch", default=None,
                   help="Override: any timm model name.")
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
                   help="Dequantize embeddings when loading npy patches. "
                        "The function is selected from --embedding-name: "
                        "seamless → ESD (72-ch), alpha_earth_coop → AlphaEarth int8.")
    g.add_argument("--checkpoint", type=Path, default=None,
                   help="Load model weights from this .pt file and skip training (inference only).")

    # ── Inference ─────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Inference")
    g.add_argument("--embedding-name", required=True,
                   choices=["tessera", "tesserav1.1", "tesserav1.1_global", "alpha_earth", "alpha_earth_coop", "seamless"],
                   help="Embedding registry key for infer_roi.")
    g.add_argument("--embedding-dir", required=True, type=Path,
                   help="Directory containing raw source embedding tiles (.zarr or .tif).")
    g.add_argument("--overlap", type=int, default=None,
                   help="Overlap between adjacent patches in pixels for inference "
                        "(default: patch_size // 2).")
    g.add_argument("--margin-m", type=float, default=200.0,
                   help="Extra metres clipped around city bbox per tile for edge context (default: 200).")

    args = parser.parse_args()
    if args.overlap is None:
        args.overlap = args.patch_size // 2

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

    # ── Build item lists ──────────────────────────────────────────────────────
    if args.global_split:
        gpkg = args.global_gpkg or (args.so2sat_dir / "patches_reference_rxr.gpkg")
        if not gpkg.exists():
            logger.error(f"Global GPKG not found: {gpkg}")
            raise SystemExit(1)
        all_items = _build_global_items(gpkg, patch_index, args.label_col)
        city_dirs = []
        # --cities in global mode selects cities for post-training inference only
        if args.cities and args.cities_dir:
            city_dirs = [
                args.cities_dir / c
                for c in args.cities
                if (args.cities_dir / c).is_dir()
            ]
    else:
        if args.cities_dir is None:
            logger.error("--cities-dir is required when not using --global-split")
            raise SystemExit(1)
        cities_dir = args.cities_dir
        city_dirs = sorted(d for d in cities_dir.iterdir() if d.is_dir())
        if args.cities:
            city_dirs = [d for d in city_dirs if d.name in args.cities]
            if not city_dirs:
                logger.error(f"None of {args.cities} found in {cities_dir}")
                raise SystemExit(1)
        all_items = []
        for city_dir in city_dirs:
            all_items.extend(
                _build_city_items(cities_dir, city_dir.name, patch_index, args.label_col)
            )

    if not all_items:
        logger.error("No items found. Check --so2sat-dir, --output-name, --year.")
        raise SystemExit(1)

    split_counts = {s: sum(1 for _, _, sp in all_items if sp == s)
                    for s in ("train", "val", "test")}
    logger.info(f"Total patches: {len(all_items)}  splits: {split_counts}")

    # ── Detect in_channels from first npy ────────────────────────────────────
    in_channels = int(np.load(all_items[0][0], mmap_mode="r").shape[0])
    # dequantize_esd expands 13 raw bands → 72 channels; override so the model
    # is built with the post-dequantization channel count.
    if args.dequantize and args.embedding_name == "seamless":
        in_channels = 72
    logger.info(f"Detected in_channels = {in_channels}")

    # ── Model ─────────────────────────────────────────────────────────────────
    arch = args.arch or MODEL_PRESETS[args.family][args.preset]
    if args.family == "aspp":
        model = build_aspp(arch=arch, in_channels=in_channels, num_classes=args.num_classes)
    elif args.family == "mlp":
        model = build_mlp(
            arch=arch,
            in_channels=in_channels,
            num_classes=args.num_classes,
            head_dropout=args.head_dropout,
        )
    else:
        model = build_resnet(
            arch=arch,
            in_channels=in_channels,
            num_classes=args.num_classes,
            family=args.family,
            head_dropout=args.head_dropout,
            img_size=args.patch_size if args.family == "vit" else None,
        )
    task = LCZResNetModule(
        model=model,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"'{args.family}/{args.preset}' ({arch}): params={n_params:,}")

    # ── Dequantize function (selected by embedding type) ──────────────────────
    dequantize_fn = None
    if args.dequantize:
        if args.embedding_name == "seamless":
            from dequantize_embeddings import dequantize_esd
            dequantize_fn = dequantize_esd
        else:
            from dequantize_embeddings import dequantize_alphaearth_embeddings
            dequantize_fn = dequantize_alphaearth_embeddings

    # ── DataModule ────────────────────────────────────────────────────────────
    datamodule = PatchDataModule(
        all_items, args.patch_size, args.batch_size, args.num_workers,
        sub_patch_size=args.sub_patch_size,
        sub_patch_stride=args.sub_patch_stride,
        dequantize_fn=dequantize_fn,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    city_names = [d.name for d in city_dirs]
    run_cfg = dict(
        task="patch_classification",
        embedding=args.output_name,
        cities="all_so2sat" if args.global_split else city_names,
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
        split_source="global_so2sat" if args.global_split else "grid",
        **{f"{s}_patches": split_counts[s] for s in ("train", "val", "test")},
    )

    _run_label = "global" if args.global_split else "_".join(city_names[:3])
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
        run_name = args.run_name or f"{args.family}_{args.preset}_{_run_label}"
        run_dir = args.output_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{args.family}_{args.preset}_{args.output_name}_{_run_label}"

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
    test_acc          = Accuracy(**metric_kw).to(device)
    test_acc_macro    = Accuracy(**metric_kw, average="macro").to(device)
    test_acc_per_class = Accuracy(**metric_kw, average="none").to(device)
    test_f1           = MulticlassF1Score(num_classes=args.num_classes, average="macro", ignore_index=-1).to(device)
    test_f1_micro     = MulticlassF1Score(num_classes=args.num_classes, average="micro", ignore_index=-1).to(device)
    test_f1_per_class = MulticlassF1Score(num_classes=args.num_classes, average="none", ignore_index=-1).to(device)
    test_kappa        = MulticlassCohenKappa(num_classes=args.num_classes, ignore_index=-1).to(device)
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
            test_acc_macro(preds, labels)
            test_acc_per_class(preds, labels)
            test_f1(preds, labels)
            test_f1_micro(preds, labels)
            test_f1_per_class(preds, labels)
            test_kappa(preds, labels)
            test_loss_total += loss.item()
            n_test_batches  += 1
            valid = labels != -1
            all_preds.append((preds[valid] + 1).cpu().numpy())
            all_labels.append((labels[valid] + 1).cpu().numpy())

    if n_test_batches > 0:
        acc          = test_acc.compute().item()
        acc_macro    = test_acc_macro.compute().item()
        f1           = test_f1.compute().item()
        f1_micro     = test_f1_micro.compute().item()
        kappa        = test_kappa.compute().item()
        per_cls_acc  = test_acc_per_class.compute().cpu().numpy()
        per_cls_f1   = test_f1_per_class.compute().cpu().numpy()
        avg_loss     = test_loss_total / n_test_batches
        logger.info(
            f"Test — OA: {acc:.4f}  Acc_macro: {acc_macro:.4f}"
            f"  F1_macro: {f1:.4f}  F1_micro: {f1_micro:.4f}"
            f"  Kappa: {kappa:.4f}  Loss: {avg_loss:.4f}"
        )
        if not args.no_wandb and wandb.run:
            wandb.log({
                "test_acc":       acc,
                "test_acc_macro": acc_macro,
                "test_f1":        f1,
                "test_f1_micro":  f1_micro,
                "test_kappa":     kappa,
                "test_loss":      avg_loss,
            })
            from utils.wandb import log_per_class_metrics
            log_per_class_metrics(per_cls_acc, per_cls_f1, args.num_classes, prefix="test")

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
            ax.set_title(f"Test Confusion Matrix — {_run_label}")
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
        grid_gpkg = city_dir / f"{city}_grid.gpkg"
        if not grid_gpkg.exists():
            logger.warning(f"  {city}: {grid_gpkg.name} not found — skipping inference")
            continue
        grid_gdf = gpd.read_file(grid_gpkg)
        west, south, east, north = grid_gdf.to_crs("EPSG:4326").total_bounds
        bbox = (west, south, east, north)

        tif_path = run_dir / f"{run_dir.name}_{args.family}-{args.preset}-classification-prediction_{city}.tif"
        infer_roi(
            model=task.model,
            model_type=args.family,
            embedding_name=args.embedding_name,
            embedding_dir=args.embedding_dir,
            bbox=bbox,
            output_path=tif_path,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            device=device,
            dequantize_fn=dequantize_fn,
            year=args.year,
            city_name=city,
            margin_m=args.margin_m,
        )
        logger.info(f"  {city}: saved {tif_path.name}")

    logger.info(f"Run complete. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
