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

def _eroded_shapes(tile_geom, polys: list, out_shape: tuple[int, int],
                   erode_px: float) -> tuple[list, object]:
    """Shrink each patch footprint by ``erode_px`` pixels and build the transform.

    So2Sat patch edges inherit the polygon-digitisation slop that WUDAPT
    explicitly tolerates -- buffers over 100 m between LCZs, with boundary
    geometric accuracy deemed non-critical -- so edge pixels are unreliable by
    construction, not by accident. Eroding costs a ring of pixels per patch and
    removes a systematic label error.

    A patch that erosion would erase entirely is kept un-eroded rather than
    dropped: losing the smallest patches would bias the class distribution.
    """
    H, W = out_shape
    minx, miny, maxx, maxy = tile_geom.bounds
    transform = rio_from_bounds(minx, miny, maxx, maxy, W, H)
    if erode_px <= 0:
        return list(polys), transform
    px = (maxx - minx) / W if W else 0.0
    if px <= 0:
        return list(polys), transform
    out = []
    for entry in polys:
        geom = entry[0]
        shrunk = geom.buffer(-erode_px * px)
        out.append((geom if shrunk.is_empty else shrunk, *entry[1:]))
    return out, transform


def rasterize_polys(tile_geom, polys: list, out_shape: tuple[int, int],
                    erode_px: float = 0.0) -> np.ndarray:
    """Burn shapely polygon/class pairs into a (H, W) uint8 label mask.

    tile_geom: shapely geometry in tile CRS (provides burn extent).
    polys: list of (geom_in_tile_crs, lcz_class_int) or
        (geom, lcz_class_int, patch_uid) — extra fields are ignored here.
    Returns array with 1-17 (class) or 0 (nodata).
    """
    H, W = out_shape
    if not polys:
        return np.zeros((H, W), dtype=np.uint8)
    shapes, transform = _eroded_shapes(tile_geom, polys, out_shape, erode_px)
    return rio_rasterize(
        [(s[0], int(s[1])) for s in shapes],
        out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8,
    )


def rasterize_patch_uids(tile_geom, polys: list, out_shape: tuple[int, int],
                         erode_px: float = 0.0) -> np.ndarray:
    """Burn patch identity into an aligned (H, W) int32 raster, -1 = nodata.

    This is what makes patch-level aggregation of a dense prediction possible,
    and it is the piece that is cheap to emit now and expensive to retrofit:
    without it there is no way to pool per-pixel softmax back onto So2Sat patch
    footprints, and therefore no way to put a segmentation number on the same
    axis as the patch-classification ladder.

    ``polys`` entries must be ``(geom, lcz_class, patch_uid)``. UIDs are burned
    as ``uid + 1`` and shifted back, so uid 0 survives the ``fill=0`` nodata
    convention.
    """
    H, W = out_shape
    if not polys or len(polys[0]) < 3:
        return np.full((H, W), -1, dtype=np.int32)
    shapes, transform = _eroded_shapes(tile_geom, polys, out_shape, erode_px)
    burned = rio_rasterize(
        [(s[0], int(s[2]) + 1) for s in shapes],
        out_shape=(H, W), transform=transform, fill=0, dtype=np.int32,
    )
    return burned.astype(np.int32) - 1


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
    *,
    split_mode: str = "grid",
    city_role: str | None = None,
    buffer_km: float = 1.3,
    min_labelled_frac: float = 0.01,
    uid_registry: dict | None = None,
) -> tuple[list, dict]:
    """Scan npy files for one city and build per-tile item tuples.

    label_tif_dir: with label_source="tif", read the label raster from
        {label_tif_dir}/pseudo_seg_{city}.tif instead of the So2Sat
        patches_reference_{city}.tif (dense pseudo-label distillation).

    split_mode:
        "grid"   — the within-city macro-block split carried by the ``split``
                   column. Quarantined for headline numbers (campaign §7), kept
                   for per-city demos.
        "global" — the So2Sat culture-10 assignment carried by the ``dataset``
                   column, resolved to one split per *tile* by
                   :mod:`utils.city_split`, which also enforces tile purity, the
                   proximity buffer and the minimum labelled fraction.
                   ``city_role`` is then required.

    uid_registry: shared ``{(city, dataset, patch_id): uid}`` map. Pass the same
        dict for every city so patch UIDs are unique across the run — bare
        ``patch_id`` restarts at 000000 in each of So2Sat's three original
        splits, so it is not an identity on its own.

    Returns:
        items: list of (npy_path, tile_geom, tile_crs, polys_or_none, tif_path_or_none)
            where polys entries are (geom, lcz_class, patch_uid)
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
        if uid_registry is None:
            uid_registry = {}
        id2polys: dict[int, list] = {}
        id2patches: dict[int, list] = {}
        has_dataset = "dataset" in sdf.columns and "patch_id" in sdf.columns
        for _, r in sdf.iterrows():
            gid = int(r["grid_id"])
            dataset = str(r["dataset"]) if has_dataset else "training"
            pid = str(r["patch_id"]) if has_dataset else f"{gid}"
            uid = uid_registry.setdefault((city, dataset, pid), len(uid_registry))
            id2polys.setdefault(gid, []).append(
                (r.geometry, int(r[label_col]), uid)
            )
            id2patches.setdefault(gid, []).append(
                (r.geometry, int(r[label_col]), dataset, pid)
            )
        tif_ref = None
    else:
        if not tif_path.exists():
            logger.warning(f"  {city}: {tif_path.name} missing — skipping")
            return [], {}
        id2polys = {}
        id2patches = {}
        tif_ref = tif_path

    # Global mode resolves the split per grid cell up front; cells that fail
    # purity, the buffer or the labelled-fraction floor are simply absent.
    gid2split: dict[int, str] | None = None
    if split_mode == "global":
        if label_source != "gpkg":
            raise ValueError(
                "--split-mode global needs the polygon labels: the culture-10 "
                "assignment lives in the split GeoPackage's `dataset` column, "
                "and a label raster does not carry it."
            )
        if city_role is None:
            raise ValueError("--split-mode global requires a city_role")
        from utils.city_split import tile_splits_for_city

        gid2split, drops = tile_splits_for_city(
            city, city_role, id2patches, id2geom,
            buffer_km=buffer_km, min_labelled_frac=min_labelled_frac,
        )
        dropped = ", ".join(f"{k}={v}" for k, v in drops.items() if v)
        logger.info(
            f"  {city} [{city_role}]: {len(gid2split)} tiles kept"
            + (f" — dropped {dropped}" if dropped else "")
        )

    items = []
    split_map = {}
    for split in ("train", "val", "test"):
        split_dir = emb_base / split
        if not split_dir.exists():
            continue
        # NOT glob(f"{city}_*.npy"): three So2Sat cities carry brackets in their
        # directory names — Osaka_[Kyoto], Quezon_City_[Manila],
        # Rawalpindi_[Islamabad] — and glob reads "[Kyoto]" as a character
        # class, so the pattern matches nothing and the city is dropped from
        # the run in silence. Match the filename with the regex instead, which
        # is literal.
        for npy_path in sorted(split_dir.iterdir()):
            if npy_path.suffix != ".npy":
                continue
            m = _FILENAME_RE.match(npy_path.name)
            if m is None or m.group(1) != city:
                continue
            grid_id = int(m.group(2))
            geom = id2geom.get(grid_id)
            if geom is None:
                continue
            if gid2split is not None:
                resolved = gid2split.get(grid_id)
                if resolved is None:
                    continue
            else:
                resolved = split
            polys = id2polys.get(grid_id, []) if label_source == "gpkg" else []
            items.append((npy_path, geom, tile_crs, polys, tif_ref))
            split_map[npy_path] = resolved

    n_valid = len(valid_gdf)
    n_total = len(grid_gdf)
    if split_mode != "global":
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
        erode_px: float = 0.0,
        emit_patch_uids: bool = False,
        normalize: str = "none",
        channel_mean: np.ndarray | None = None,
        channel_std: np.ndarray | None = None,
        nodata_predicate=None,
        emit_valid: bool = False,
    ) -> None:
        if normalize not in ("none", "channel"):
            raise ValueError(f"normalize must be 'none' or 'channel', got {normalize!r}")
        self.items = items
        self.label_source = label_source
        self.dequantize_fn = dequantize_fn
        self.normalize = normalize
        # Predicate(s) run on the RAW array, before dequantization, because the
        # sentinel is defined in the units it is stored in: coop's invalid
        # pixels are all-64-channels == -128, which dequantization turns into an
        # ordinary-looking vector of L2 norm 8.06. A sequence marks per-source
        # predicates for fused items.
        self.nodata_predicate = nodata_predicate
        self.emit_valid = emit_valid
        if normalize == "channel":
            if channel_mean is None or channel_std is None:
                raise ValueError(
                    "normalize='channel' requires channel_mean and channel_std"
                )
            self._norm_mean = torch.from_numpy(
                np.asarray(channel_mean, dtype=np.float32)
            )[:, None, None]
            self._norm_std = torch.from_numpy(
                np.asarray(channel_std, dtype=np.float32)
            )[:, None, None]
        self.erode_px = erode_px
        # Patch UIDs are an evaluation-time concern only. Keeping them out of
        # the training path means the augmentation stack (flips/rot90, which
        # would have to transform the UID raster in lockstep) needs no changes.
        self.emit_patch_uids = emit_patch_uids

    def __len__(self) -> int:
        return len(self.items)

    def _predicate_for(self, source_idx: int):
        pred = self.nodata_predicate
        if pred is None:
            return None
        if isinstance(pred, (list, tuple)):
            return pred[source_idx] if source_idx < len(pred) else None
        return pred if source_idx == 0 else None

    def _load(self, path: Path, dequantize: bool,
              source_idx: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
        arr = np.load(path).astype(np.float32)   # (C, H, W)
        pred = self._predicate_for(source_idx)
        invalid = pred(arr) if pred is not None else None    # (H, W) bool, RAW units
        if dequantize and self.dequantize_fn is not None:
            arr = self.dequantize_fn(arr)
        return np.nan_to_num(arr, copy=False), invalid

    def __getitem__(self, idx: int) -> dict:
        npy_path, tile_geom, tile_crs, polys, tif_ref = self.items[idx]

        if isinstance(npy_path, tuple):
            loaded = [self._load(p, dequantize=(i == 0), source_idx=i)
                      for i, p in enumerate(npy_path)]
            arrs = [a for a, _ in loaded]
            base_hw = arrs[0].shape[1:]
            for i in range(1, len(arrs)):
                if arrs[i].shape[1:] != base_hw:
                    arrs[i] = F.interpolate(
                        torch.from_numpy(arrs[i]).unsqueeze(0),
                        size=base_hw, mode="nearest",
                    ).squeeze(0).numpy()
            arr = np.concatenate(arrs, axis=0)
            # A pixel invalid in ANY source is invalid for the fused tensor:
            # the concatenated channels are one sample, so a sentinel in one
            # source contaminates the whole vector.
            invalid = None
            for a, inv in loaded:
                if inv is None:
                    continue
                if inv.shape != base_hw:
                    inv = np.asarray(F.interpolate(
                        torch.from_numpy(inv[None, None].astype(np.float32)),
                        size=base_hw, mode="nearest",
                    ).squeeze().numpy() > 0.5)
                invalid = inv if invalid is None else (invalid | inv)
        else:
            arr, invalid = self._load(npy_path, dequantize=True, source_idx=0)
        _, H, W = arr.shape
        image = torch.from_numpy(arr)

        if self.normalize == "channel":
            image = (image - self._norm_mean) / (self._norm_std + 1e-6)

        if self.label_source == "gpkg":
            raw = rasterize_polys(tile_geom, polys, (H, W), erode_px=self.erode_px)
        else:
            raw = clip_label_tif(tile_geom, tile_crs, tif_ref, (H, W))

        # 0→-1, 1-17→0-16
        mask = torch.from_numpy(raw.astype(np.int64)) - 1
        out = {"image": image, "mask": mask}

        if self.emit_patch_uids and self.label_source == "gpkg":
            uids = rasterize_patch_uids(
                tile_geom, polys, (H, W), erode_px=self.erode_px
            )
            out["patch_uid"] = torch.from_numpy(uids.astype(np.int64))
        if self.emit_valid:
            out["valid"] = (
                torch.ones((H, W), dtype=torch.bool) if invalid is None
                else torch.from_numpy(~invalid)
            )
        return out


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
        noise_sigma: float = 0.05,
        noise_prob: float = 0.5,
        erode_px: float = 0.0,
        emit_patch_uids: bool = False,
        normalize: str = "none",
        channel_mean: np.ndarray | None = None,
        channel_std: np.ndarray | None = None,
        nodata_predicate=None,
    ) -> None:
        self.erode_px = erode_px
        self.emit_patch_uids = emit_patch_uids
        self.normalize = normalize
        self.channel_mean = channel_mean
        self.channel_std = channel_std
        self.nodata_predicate = nodata_predicate
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
        self.noise_sigma = noise_sigma
        self.noise_prob = noise_prob

    def setup(self) -> None:
        def _key(it):
            return it[0][0] if isinstance(it[0], tuple) else it[0]

        def _for_split(items, s):
            return [it for it in items if self.split_map.get(_key(it)) == s]

        eval_items = self.eval_items if self.eval_items is not None else self.all_items
        eval_source = self.eval_label_source or self.label_source

        norm_kw = dict(
            normalize=self.normalize, channel_mean=self.channel_mean,
            channel_std=self.channel_std, nodata_predicate=self.nodata_predicate,
        )

        def _eval_ds(split):
            return GridSegDataset(
                _for_split(eval_items, split), eval_source, self.dequantize_fn,
                erode_px=self.erode_px, emit_patch_uids=self.emit_patch_uids,
                **norm_kw,
            )

        self._train_ds = GridSegDataset(
            _for_split(self.all_items, "train"), self.label_source,
            self.dequantize_fn, erode_px=self.erode_px, emit_patch_uids=False,
            **norm_kw,
        )
        self._val_ds = _eval_ds("val")
        self._test_ds = _eval_ds("test")
        # The culture cities' validation tiles. Not used for early stopping —
        # that is what the val-inner cities are for — but kept addressable
        # because ensemble_stacking.py --city-holdout fits its combiner weights
        # on exactly these patches.
        self._culture_val_ds = _eval_ds("culture_val")
        extra = (f", culture_val: {len(self._culture_val_ds)}"
                 if len(self._culture_val_ds) else "")
        logger.info(
            f"Dataset sizes — train: {len(self._train_ds)} ({self.label_source}), "
            f"val: {len(self._val_ds)}, test: {len(self._test_ds)} ({eval_source})"
            f"{extra}"
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
        imgs, masks, uids = [], [], []
        has_uid = "patch_uid" in batch[0]
        for b in batch:
            img, msk = b["image"], b["mask"]
            uid = b.get("patch_uid")
            ph = max_hw - img.shape[-2]
            pw = max_hw - img.shape[-1]
            if ph or pw:
                img = F.pad(img,  (0, pw, 0, ph), value=0)
                msk = F.pad(msk,  (0, pw, 0, ph), value=-1)
                if uid is not None:
                    uid = F.pad(uid, (0, pw, 0, ph), value=-1)
            imgs.append(img)
            masks.append(msk)
            if uid is not None:
                uids.append(uid)
        stacked_uids = torch.stack(uids) if has_uid and uids else None
        return torch.stack(imgs), torch.stack(masks), stacked_uids

    @staticmethod
    def _collate(batch: list) -> dict:
        images, masks, uids = GridSegDataModule._pad_batch(batch)
        out = {"image": images, "mask": masks}
        if uids is not None:
            out["patch_uid"] = uids
        return out

    def _train_collate(self, batch: list) -> dict:
        images, masks, _ = GridSegDataModule._pad_batch(batch)
        images, masks = augment_batch(
            images, masks, noise_sigma=self.noise_sigma, noise_prob=self.noise_prob
        )
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

    def culture_val_dataloader(self) -> DataLoader | None:
        """Loader over the culture cities' validation tiles, or None if empty.

        Only ``--split-mode global`` populates this split.
        """
        if not len(self._culture_val_ds):
            return None
        return DataLoader(
            self._culture_val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate,
        )
