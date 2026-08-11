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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
    for orig_split in ("training", "validation", "testing", "unlabeled"):
        d = so2sat_dir / orig_split / output_name / year
        if not d.exists():
            continue
        index[orig_split] = {
            p.stem[len("patch_"):]: p for p in sorted(d.glob("patch_*.npy"))
        }
    n_total = sum(len(v) for v in index.values())
    logger.info(f"Patch index: {n_total} npy files found under {so2sat_dir}")
    return index


def merge_patch_indexes(indexes: list[dict]) -> dict:
    """Intersect per-source patch indexes into one tuple-valued index.

    Input: one ``{orig_split: {patch_id: Path}}`` index per embedding source.
    Output: same structure but values are tuples of Paths (one per source, in
    input order); only patch_ids present in EVERY source survive.
    """
    merged: dict[str, dict[str, tuple]] = {}
    n_dropped = 0
    for orig_split in indexes[0]:
        if not all(orig_split in ix for ix in indexes):
            continue
        common = set(indexes[0][orig_split])
        for ix in indexes[1:]:
            common &= set(ix[orig_split])
        n_dropped += max(len(ix.get(orig_split, {})) for ix in indexes) - len(common)
        merged[orig_split] = {
            pid: tuple(ix[orig_split][pid] for ix in indexes) for pid in common
        }
    n_total = sum(len(v) for v in merged.values())
    logger.info(
        f"Fused patch index: {n_total} patches present in all "
        f"{len(indexes)} sources ({n_dropped} dropped)"
    )
    return merged


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


def build_pseudo_items(
    pseudo_gpkg: Path,
    patch_index: dict,
    label_col: str = "LCZ_class",
    weight_col: str = "weight",
    weight_scale: float = 1.0,
) -> list[tuple]:
    """Build weighted train items from a pseudo-label GeoPackage
    (generate_pseudo_labels.py output).

    Returns 4-tuples ``(npy_path, label_int, "train", weight)``; the weight
    (scaled by ``weight_scale``) flows through PatchDataset into the
    per-sample weighted CE loss.
    """
    gdf = gpd.read_file(pseudo_gpkg)
    items: list[tuple] = []
    n_missing = 0
    for _, row in gdf.iterrows():
        pid = str(row["patch_id"])
        dataset = str(row.get("dataset", "unlabeled"))
        path = patch_index.get(dataset, {}).get(pid)
        if path is None:
            n_missing += 1
            continue
        label = int(row[label_col]) - 1   # 1-17 → 0-16
        weight = float(row[weight_col]) * weight_scale
        items.append((path, label, "train", weight))
    logger.info(
        f"Pseudo items: {len(items)} weighted train patches from "
        f"{pseudo_gpkg.name} ({n_missing} had no npy)"
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
    output_name: str | list[str],
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

    ``output_name`` may be a list of embedding names (fusion): the per-source
    patch indexes are intersected and item paths become tuples of npy paths,
    one per source in input order.

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

    names = [output_name] if isinstance(output_name, str) else list(output_name)
    indexes = []
    for name in names:
        index = build_patch_index(so2sat_dir, name, year)
        if not index:
            logger.error(
                f"No patch npy files found under {so2sat_dir} "
                f"for output_name={name!r}, year={year!r}"
            )
            raise SystemExit(1)
        indexes.append(index)
    patch_index = indexes[0] if len(indexes) == 1 else merge_patch_indexes(indexes)

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

    Fusion: when item paths are tuples (one npy per embedding source), each
    source is loaded, dequantized (``dequantize_fn`` must then be a matching
    sequence), resized to patch_size and concatenated along channels. Sources
    may have different native resolutions. Incompatible with sub_patch_size.
    """

    def __init__(
        self,
        items: list,        # (npy_path, label_int, split)
        patch_size: int,
        sub_patch_size: int | None = None,
        sub_patch_stride: int | None = None,
        dequantize_fn=None,
        nodata_mode: str = "zero",
        nodata_predicate=None,
        normalize: str = "none",
        channel_mean=None,
        channel_std=None,
    ) -> None:
        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.sub_patch_stride = sub_patch_stride or sub_patch_size
        self.dequantize_fn = dequantize_fn
        if nodata_mode not in ("zero", "mask"):
            raise ValueError(f"nodata_mode must be 'zero' or 'mask', got {nodata_mode!r}")
        self.nodata_mode = nodata_mode
        self.nodata_predicate = nodata_predicate

        if normalize not in ("none", "channel"):
            raise ValueError(f"normalize must be 'none' or 'channel', got {normalize!r}")
        self.normalize = normalize
        self.channel_mean = None
        self.channel_std = None
        if normalize == "channel":
            if channel_mean is None or channel_std is None:
                raise ValueError("normalize='channel' requires channel_mean and channel_std")
            self.channel_mean = np.asarray(channel_mean, dtype=np.float32)
            self.channel_std = np.asarray(channel_std, dtype=np.float32)

        # Value written into invalid pixels, in decoded units: the per-channel
        # mean when we know it, so masked pixels normalise to exactly 0.
        self.fill_value: float | np.ndarray = (
            0.0 if self.channel_mean is None else self.channel_mean[:, None]
        )
        if self.channel_mean is not None:
            self._norm_mean = torch.from_numpy(self.channel_mean)[:, None, None]
            self._norm_std = torch.from_numpy(self.channel_std)[:, None, None]

        if items and isinstance(items[0][0], tuple) and sub_patch_size is not None:
            raise ValueError("sub_patch_size is not supported with fused (multi-source) items")

        # Items are (path, label, split) or (path, label, split, weight);
        # the "weight" batch key is only emitted when any item carries one,
        # so unweighted runs keep their exact previous batch format.
        self.has_weights = any(len(it) > 3 for it in items)

        def _w(it) -> float:
            return float(it[3]) if len(it) > 3 else 1.0

        if sub_patch_size is None:
            self.expanded = [(it[0], it[1], None, None, _w(it)) for it in items]
        else:
            stride = self.sub_patch_stride
            self.expanded = []
            for it in items:
                path, label = it[0], it[1]
                _, H, W = np.load(path, mmap_mode="r").shape
                for r in range(0, H - sub_patch_size + 1, stride):
                    for c in range(0, W - sub_patch_size + 1, stride):
                        self.expanded.append((path, label, r, c, _w(it)))

    def __len__(self) -> int:
        return len(self.expanded)

    def _resize(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.patch_size, self.patch_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.patch_size, self.patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image

    def _resize_valid(self, valid: torch.Tensor) -> torch.Tensor:
        """Resize a (1, H, W) float validity mask through the image's own call.

        Bilinear-interpolates the mask and then demands a full weight of 1, so
        any output pixel whose interpolation touched an invalid input pixel is
        itself invalid. The mask therefore never claims validity the resized
        image cannot back.
        """
        if valid.shape[-2:] == (self.patch_size, self.patch_size):
            return valid
        return (self._resize(valid) >= 1.0 - 1e-6).float()

    def _load_source(self, path: Path, dequantize_fn, predicate=None, fill_offset: int = 0):
        """Load one source as ``(C, H, W)`` float32 plus its validity mask.

        The nodata predicate is applied to the array exactly as stored, before
        ``nan_to_num`` and before dequantization, because each family's sentinel
        is defined in its own stored units (see
        ``datasets.registry.get_nodata_predicate``). Invalid pixels are then
        overwritten with ``fill_value`` in *decoded* units so the sentinel does
        not bleed into its neighbours through the bilinear resize.

        Returns ``(arr, valid)``; ``valid`` is None in "zero" mode, which keeps
        the pre-Phase-1 behaviour byte for byte.
        """
        raw = np.load(path).astype(np.float32)   # (C, H, W)

        invalid = None
        if self.nodata_mode == "mask" and predicate is not None:
            invalid = predicate(raw)             # (H, W) bool, stored units

        arr = np.nan_to_num(raw, nan=0.0)
        if dequantize_fn is not None:
            arr = dequantize_fn(arr)

        if invalid is not None and invalid.any():
            fill = self.fill_value
            if isinstance(fill, np.ndarray):
                # Fused sources each take their own slice of the channel means.
                fill = fill[fill_offset:fill_offset + arr.shape[0]]
            arr[:, invalid] = fill

        valid = None
        if self.nodata_mode == "mask":
            valid = (
                np.ones((1, *arr.shape[1:]), dtype=np.float32) if invalid is None
                else (~invalid)[None].astype(np.float32)
            )
        return arr, valid

    def _predicate_for(self, i: int):
        """Per-source nodata predicate (a sequence for fused multi-source items)."""
        p = self.nodata_predicate
        if isinstance(p, (list, tuple)):
            return p[i]
        return p

    def __getitem__(self, idx: int) -> dict:
        path, label, r, c, weight = self.expanded[idx]

        if isinstance(path, tuple):
            # Fusion: resize each source to the common grid, then concat channels
            fns = (self.dequantize_fn if isinstance(self.dequantize_fn, (list, tuple))
                   else [self.dequantize_fn] * len(path))
            images, valids = [], []
            offset = 0
            for i, (p, fn) in enumerate(zip(path, fns)):
                arr, v = self._load_source(p, fn, self._predicate_for(i), offset)
                offset += arr.shape[0]
                images.append(self._resize(torch.from_numpy(arr)))
                if v is not None:
                    valids.append(self._resize_valid(torch.from_numpy(v)))
            image = torch.cat(images, dim=0)
            # A pixel is usable only where every source has data.
            valid = torch.stack(valids).amin(dim=0) if valids else None
        else:
            arr, v = self._load_source(path, self.dequantize_fn, self._predicate_for(0))
            if r is None:
                image = torch.from_numpy(arr)
                valid = None if v is None else torch.from_numpy(v)
            else:
                sl = (slice(None), slice(r, r + self.sub_patch_size),
                      slice(c, c + self.sub_patch_size))
                image = torch.from_numpy(arr[sl])
                valid = None if v is None else torch.from_numpy(v[sl])
            image = self._resize(image)
            valid = None if valid is None else self._resize_valid(valid)

        if self.normalize == "channel":
            # Applied here, after the resize: bilinear interpolation is affine
            # with weights summing to 1, so normalising before or after it gives
            # the same result, and doing it once covers fused sources too.
            image = (image - self._norm_mean) / (self._norm_std + 1e-6)

        out = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
        if valid is not None:
            out["valid"] = valid
        if self.has_weights:
            out["weight"] = torch.tensor(weight, dtype=torch.float32)
        return out


# ── DataModule ────────────────────────────────────────────────────────────────

class PatchDataModule:
    """Minimal DataModule for run_training_loop compatibility.

    ``sampler``: "none" (shuffle), or "balanced"/"sqrt_balanced" for a
    WeightedRandomSampler with per-sample weights 1/count (resp. 1/sqrt(count))
    of the sample's class in the train split (with replacement, one epoch =
    len(train) draws).
    """

    def __init__(
        self,
        all_items: list,
        patch_size: int,
        batch_size: int,
        num_workers: int,
        sub_patch_size: int | None = None,
        sub_patch_stride: int | None = None,
        dequantize_fn=None,
        sampler: str = "none",
        nodata_mode: str = "zero",
        nodata_predicate=None,
        normalize: str = "none",
        channel_mean=None,
        channel_std=None,
    ) -> None:
        self.all_items = all_items
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sub_patch_size = sub_patch_size
        self.sub_patch_stride = sub_patch_stride
        self.dequantize_fn = dequantize_fn
        self.sampler = sampler
        self.nodata_mode = nodata_mode
        self.nodata_predicate = nodata_predicate
        self.normalize = normalize
        self.channel_mean = channel_mean
        self.channel_std = channel_std

    def setup(self) -> None:
        def _for_split(s: str) -> list:
            return [it for it in self.all_items if it[2] == s]

        kw = dict(sub_patch_size=self.sub_patch_size, sub_patch_stride=self.sub_patch_stride,
                  dequantize_fn=self.dequantize_fn, nodata_mode=self.nodata_mode,
                  nodata_predicate=self.nodata_predicate, normalize=self.normalize,
                  channel_mean=self.channel_mean, channel_std=self.channel_std)
        self._train_ds = PatchDataset(_for_split("train"), self.patch_size, **kw)
        self._val_ds   = PatchDataset(_for_split("val"),   self.patch_size, **kw)
        self._test_ds  = PatchDataset(_for_split("test"),  self.patch_size, **kw)
        logger.info(
            f"Dataset sizes — train: {len(self._train_ds)}, "
            f"val: {len(self._val_ds)}, test: {len(self._test_ds)}"
        )

    @staticmethod
    def _collate(batch: list) -> dict:
        out = {
            "image": torch.stack([b["image"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        }
        if "valid" in batch[0]:
            out["valid"] = torch.stack([b["valid"] for b in batch])
        if "weight" in batch[0]:
            out["weight"] = torch.stack([b["weight"] for b in batch])
        return out

    def _train_collate(self, batch: list) -> dict:
        out = self._collate(batch)
        out["image"] = augment_images(out["image"])
        return out

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.sampler != "none":
            labels = np.array([it[1] for it in self._train_ds.expanded])
            counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
            class_w = 1.0 / np.maximum(counts, 1)
            if self.sampler == "sqrt_balanced":
                class_w = np.sqrt(class_w)
            sampler = WeightedRandomSampler(
                torch.as_tensor(class_w[labels], dtype=torch.double),
                num_samples=len(labels), replacement=True,
            )
            logger.info(f"Train sampler: {self.sampler} over {len(counts)} classes")
        return DataLoader(
            self._train_ds, batch_size=self.batch_size,
            shuffle=sampler is None, sampler=sampler,
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
