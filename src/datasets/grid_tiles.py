"""Grid-tile data layer for the segmentation pipeline.

Each city directory contains {output_name}/{year}/{split}/{city}_{grid_id}.npy
embedding tiles. Labels come either from rasterised polygons
(patches_reference_{city}_split.gpkg) or a clipped raster
(patches_reference_{city}.tif).

Label convention: raw 1-17 → 0-16 (class index), raw 0 (nodata) → -1
(ignore_index).
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from pyproj import Transformer
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds as rio_from_bounds
from shapely.ops import transform as shapely_transform
from torch.utils.data import DataLoader, Dataset

from training.augment import augment_batch

_FILENAME_RE = re.compile(r"^(.+)_(\d+)\.npy$")   # {city}_{grid_id}.npy


# ── Label helpers ─────────────────────────────────────────────────────────────

def rasterize_polys(tile_geom, polys: list, out_shape: tuple[int, int]) -> np.ndarray:
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


def clip_label_tif(tile_geom, tile_crs: str, tif_path: Path, out_shape: tuple[int, int]) -> np.ndarray:
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

def build_city_tile_items(
    city_dir: Path,
    output_name: str,
    year: str,
    label_source: str,
    label_col: str,
    label_tif_dir: Path | None = None,
) -> tuple[list, dict]:
    """Scan npy files for one city and build per-tile item tuples.

    label_tif_dir: with label_source="tif", read the label raster from
        {label_tif_dir}/pseudo_seg_{city}.tif instead of the So2Sat
        patches_reference_{city}.tif (dense pseudo-label distillation).

    Returns:
        items: list of (npy_path, tile_geom, tile_crs, polys_or_none, tif_path_or_none)
        split_map: dict {npy_path: split} for DataModule split filtering
    """
    city = city_dir.name
    emb_base = city_dir / output_name / year
    grid_gpkg = city_dir / f"{city}_grid.gpkg"
    split_gpkg = city_dir / f"patches_reference_{city}_split.gpkg"
    if label_source == "tif" and label_tif_dir is not None:
        tif_path = label_tif_dir / f"pseudo_seg_{city}.tif"
    else:
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

class GridSegDataset(Dataset):
    """Grid tile dataset for segmentation.

    Returns {"image": (C, H, W) float32, "mask": (H, W) long}
    Label convention: raw 1-17 → 0-16, raw 0 → -1 (ignore_index=-1).

    Fusion: when an item's npy_path is a tuple (one npy per embedding source,
    same grid cell), the sources are loaded, resized to source 0's grid if
    off-by-a-pixel (nearest-neighbour — extra sources are categorical/aux
    rasters), and concatenated along channels. ``dequantize_fn`` applies to
    source 0 only: the segmentation CLI takes a single --embedding-name, so
    extra sources must be stored ready-to-use (unlike PatchDataset, which
    accepts a per-source dequantize list for the classification pipeline).
    """

    def __init__(
        self,
        items: list,          # (npy_path, tile_geom, tile_crs, polys, tif_path_or_None)
        label_source: str,    # "gpkg" or "tif"
        dequantize_fn=None,
    ) -> None:
        self.items = items
        self.label_source = label_source
        self.dequantize_fn = dequantize_fn

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: Path, dequantize: bool) -> np.ndarray:
        arr = np.load(path).astype(np.float32)   # (C, H, W)
        if dequantize and self.dequantize_fn is not None:
            arr = self.dequantize_fn(arr)
        return arr

    def __getitem__(self, idx: int) -> dict:
        npy_path, tile_geom, tile_crs, polys, tif_ref = self.items[idx]

        if isinstance(npy_path, tuple):
            arrs = [self._load(p, dequantize=(i == 0))
                    for i, p in enumerate(npy_path)]
            base_hw = arrs[0].shape[1:]
            for i in range(1, len(arrs)):
                if arrs[i].shape[1:] != base_hw:
                    arrs[i] = F.interpolate(
                        torch.from_numpy(arrs[i]).unsqueeze(0),
                        size=base_hw, mode="nearest",
                    ).squeeze(0).numpy()
            arr = np.concatenate(arrs, axis=0)
        else:
            arr = self._load(npy_path, dequantize=True)
        _, H, W = arr.shape
        image = torch.from_numpy(arr)

        if self.label_source == "gpkg":
            raw = rasterize_polys(tile_geom, polys, (H, W))
        else:
            raw = clip_label_tif(tile_geom, tile_crs, tif_ref, (H, W))

        # 0→-1, 1-17→0-16
        mask = torch.from_numpy(raw.astype(np.int64)) - 1

        return {"image": image, "mask": mask}


# ── DataModule ────────────────────────────────────────────────────────────────

class GridSegDataModule:
    """Minimal DataModule for run_training_loop compatibility.

    eval_items/eval_label_source: optional separate item list + label source
    for the val/test splits — used when training on dense pseudo-label rasters
    so that early stopping and test metrics run against the ground-truth gpkg
    labels instead of the pseudo labels.

    aux_dropout_p/aux_channel_start: with fused (multi-source) items, zero out
    channels >= aux_channel_start for a random aux_dropout_p fraction of the
    TRAIN samples per batch — keeps the model functional without the aux
    modality and damps OSM-completeness identity learning.
    """

    def __init__(
        self,
        all_items: list,
        split_map: dict,       # {npy_path: split} (source-0 path for fused items)
        label_source: str,
        batch_size: int,
        num_workers: int,
        dequantize_fn=None,
        eval_items: list | None = None,
        eval_label_source: str | None = None,
        aux_dropout_p: float = 0.0,
        aux_channel_start: int | None = None,
    ) -> None:
        self.all_items = all_items
        self.split_map = split_map
        self.label_source = label_source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dequantize_fn = dequantize_fn
        self.eval_items = eval_items
        self.eval_label_source = eval_label_source
        self.aux_dropout_p = aux_dropout_p
        self.aux_channel_start = aux_channel_start

    def setup(self) -> None:
        def _key(it):
            return it[0][0] if isinstance(it[0], tuple) else it[0]

        def _for_split(items, s):
            return [it for it in items if self.split_map.get(_key(it)) == s]

        eval_items = self.eval_items if self.eval_items is not None else self.all_items
        eval_source = self.eval_label_source or self.label_source
        self._train_ds = GridSegDataset(_for_split(self.all_items, "train"), self.label_source, self.dequantize_fn)
        self._val_ds   = GridSegDataset(_for_split(eval_items, "val"),  eval_source, self.dequantize_fn)
        self._test_ds  = GridSegDataset(_for_split(eval_items, "test"), eval_source, self.dequantize_fn)
        logger.info(
            f"Dataset sizes — train: {len(self._train_ds)} ({self.label_source}), "
            f"val: {len(self._val_ds)}, test: {len(self._test_ds)} ({eval_source})"
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

    def _train_collate(self, batch: list) -> dict:
        images, masks = GridSegDataModule._pad_batch(batch)
        images, masks = augment_batch(images, masks)
        if self.aux_dropout_p > 0 and self.aux_channel_start is not None:
            drop = torch.rand(images.shape[0]) < self.aux_dropout_p
            images[drop, self.aux_channel_start:] = 0.0
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
