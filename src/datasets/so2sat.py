"""So2Sat pre-extracted patch data layer for the classification pipeline.

Item tuples are ``(npy_path, label_int, split)`` with labels 0-16
(LCZ_class 1-17 shifted by -1) and split ∈ {"train", "val", "test"}.

Two split modes:
  Per-city: patches_reference_{city}_split.gpkg (grid-based split column).
  Global:   patches_reference_rxr.gpkg ('dataset' column, all 400k+ patches).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from training.augment import augment_images


# ── Item builders ─────────────────────────────────────────────────────────────

def build_patch_index(
    so2sat_dir: Path,
    output_name: str,
    year: str,
) -> dict[str, dict[str, Path]]:
    """Scan the three original So2Sat split dirs and return
    {orig_split: {patch_id: Path}}.

    patch_id is NOT unique across the original splits (each of
    training/validation/testing restarts at 000000), so lookups must always
    be qualified by the patch's 'dataset' value from the reference GPKG.
    A flat {patch_id: Path} index silently resolves validation/testing ids
    to the wrong files.
    """
    index: dict[str, dict[str, Path]] = {}
    for orig_split in ("training", "validation", "testing"):
        d = so2sat_dir / orig_split / output_name / year
        if not d.exists():
            continue
        index[orig_split] = {
            p.stem[len("patch_"):]: p for p in sorted(d.glob("patch_*.npy"))
        }
    n_total = sum(len(v) for v in index.values())
    logger.info(f"Patch index: {n_total} npy files found under {so2sat_dir}")
    return index


def build_global_items(
    patches_gpkg: Path,
    patch_index: dict[str, dict[str, Path]],
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
        path = patch_index.get(str(row["dataset"]), {}).get(pid)
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


# Hybrid (orig_test) mode: original train/val patches get their grid split, but
# grid-test-cell patches fold into train (the test set comes from the original split).
_GRID_FOLD = {"train": "train", "test": "train", "val": "val"}


def build_city_items(
    cities_dir: Path,
    city: str,
    patch_index: dict[str, dict[str, Path]],
    label_col: str = "LCZ_class",
    *,
    orig_test: bool = False,
) -> list[tuple]:
    """Build (npy_path, label_int, split) tuples for one city.

    Uses patches_reference_{city}_split.gpkg as the authoritative source of
    patch_ids and their grid-based train/val/test split assignment; the
    'dataset' column disambiguates which original So2Sat dir holds each npy.

    With ``orig_test=True`` the test set is taken from the original So2Sat split
    ('dataset' == 'testing'); the remaining (original train+val) patches get their
    grid split, with grid-test-cell patches folded into train (see _GRID_FOLD).
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
        path = patch_index.get(str(row["dataset"]), {}).get(pid)
        if path is None:
            n_missing += 1
            continue
        label = int(row[label_col]) - 1   # 1-17 → 0-16
        if orig_test:
            split = ("test" if str(row["dataset"]) == "testing"
                     else _GRID_FOLD[str(row["split"])])
        else:
            split = str(row["split"])
        items.append((path, label, split))

    logger.info(
        f"  {city}: {len(items)} patches matched "
        f"({n_missing} patch_ids had no npy)"
    )
    return items


def build_so2sat_items(
    so2sat_dir: Path,
    output_name: str,
    year: str,
    *,
    global_split: bool,
    global_gpkg: Path | None = None,
    cities_dir: Path | None = None,
    cities: list[str] | None = None,
    label_col: str = "LCZ_class",
    orig_test: bool = False,
) -> tuple[list[tuple], list[Path]]:
    """Build the full item list plus the city dirs used for per-city inference.

    Per-city mode: ``cities_dir`` is required; ``cities`` filters which cities
    are used for training AND inference.
    Global mode: all patches from the global GPKG are used for training;
    ``cities`` (with ``cities_dir``) selects cities for inference only.
    Hybrid mode (``orig_test``): trains on ALL cities' grid splits but keeps the
    original So2Sat testing patches as the test set; like global mode, ``cities``
    selects cities for inference only.

    Raises SystemExit on missing inputs (CLI-friendly).
    """
    if orig_test and global_split:
        logger.error("--orig-test and --global-split are mutually exclusive")
        raise SystemExit(1)

    patch_index = build_patch_index(so2sat_dir, output_name, year)
    if not patch_index:
        logger.error(
            f"No patch npy files found under {so2sat_dir} "
            f"for output_name={output_name!r}, year={year!r}"
        )
        raise SystemExit(1)

    if orig_test:
        if cities_dir is None:
            logger.error("--cities-dir is required when using --orig-test")
            raise SystemExit(1)
        all_cities = sorted(d for d in cities_dir.iterdir() if d.is_dir())
        all_items = []
        for city_dir in all_cities:
            all_items.extend(
                build_city_items(cities_dir, city_dir.name, patch_index,
                                 label_col, orig_test=True)
            )
        # --cities selects cities for post-training inference only
        city_dirs = []
        if cities:
            city_dirs = [cities_dir / c for c in cities if (cities_dir / c).is_dir()]
    elif global_split:
        gpkg = global_gpkg or (so2sat_dir / "patches_reference_rxr.gpkg")
        if not gpkg.exists():
            logger.error(f"Global GPKG not found: {gpkg}")
            raise SystemExit(1)
        all_items = build_global_items(gpkg, patch_index, label_col)
        # --cities in global mode selects cities for post-training inference only
        city_dirs: list[Path] = []
        if cities and cities_dir:
            city_dirs = [
                cities_dir / c for c in cities if (cities_dir / c).is_dir()
            ]
    else:
        if cities_dir is None:
            logger.error("--cities-dir is required when not using --global-split")
            raise SystemExit(1)
        city_dirs = sorted(d for d in cities_dir.iterdir() if d.is_dir())
        if cities:
            city_dirs = [d for d in city_dirs if d.name in cities]
            if not city_dirs:
                logger.error(f"None of {cities} found in {cities_dir}")
                raise SystemExit(1)
        all_items = []
        for city_dir in city_dirs:
            all_items.extend(
                build_city_items(cities_dir, city_dir.name, patch_index, label_col)
            )

    if not all_items:
        logger.error("No items found. Check --so2sat-dir, --output-name, --year.")
        raise SystemExit(1)

    return all_items, city_dirs


# ── Dataset ───────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """So2Sat patch dataset for patch-level classification.

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
    """Minimal DataModule for run_training_loop compatibility."""

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
