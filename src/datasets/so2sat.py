"""So2Sat pre-extracted patch data layer for the classification pipeline.

Items are :class:`PatchItem` records with labels 0-16 (LCZ_class 1-17 shifted
by -1) and split ∈ {"train", "val", "test"}.

Two split modes:
  Per-city: patches_reference_{city}_split.gpkg (grid-based split column).
  Global:   patches_reference_rxr.gpkg ('dataset' column, all 400k+ patches).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler



@dataclass(frozen=True)
class PatchItem:
    """One labelled patch.

    Replaces the ``(path, label, split)`` / ``(path, label, split, weight)``
    tuples the builders used to return. Those were unpacked positionally in
    several places (``for path, label, _ in items``, ``it[2] == split``), so
    adding a field would have broken them silently — with named access the
    breakage is loud and local.

    Attributes:
        path: The patch ``.npy`` path, or a tuple of paths for fused sources.
        label: Class index 0-16.
        split: "train" | "val" | "test".
        city: Source city, when known. The per-city builder knows it directly;
            the global builder derives it from the So2Sat city bounds, since
            patches_reference_rxr.gpkg carries no city column. Needed by
            per-city normalization in Phase 4.
        weight: Per-sample loss weight, or None for ordinary (unweighted)
            patches. Only pseudo-labelled items set it, and the "weight" batch
            key is emitted only when some item carries one, so unweighted runs
            keep their exact previous batch format.
    """

    path: Path | tuple
    label: int
    split: str
    city: str | None = None
    weight: float | None = None


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


def assign_cities(gdf, city_bounds: Path):
    """Attach a ``city`` column to a patch GeoDataFrame by spatial join.

    patches_reference_rxr.gpkg has only patch_id / dataset / LCZ_class /
    geometry — no city — so for the global split the city has to come from the
    So2Sat city bounds (JRC_NAME_MAIN in so2sat_guppd_bounds.gpkg). Patches
    outside every city box get None.
    """
    bounds = gpd.read_file(city_bounds, layer="so2sat_guppd_bounds_gdf")
    bounds = bounds[["JRC_NAME_MAIN", "geometry"]].rename(
        columns={"JRC_NAME_MAIN": "city"}
    ).to_crs(gdf.crs)

    pts = gdf[["geometry"]].copy()
    with warnings.catch_warnings():
        # Centroids in a geographic CRS are approximate, which geopandas warns
        # about. It cannot matter here: a So2Sat patch is 320 m across and the
        # error is sub-metre, far too small to move a patch into another city.
        warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")
        pts["geometry"] = pts.geometry.centroid
    joined = gpd.sjoin(pts, bounds, how="left", predicate="within")
    # Overlapping city boxes can match a patch twice; keep the first.
    return joined[~joined.index.duplicated(keep="first")]["city"]


def build_global_items(
    patches_gpkg: Path,
    patch_index: dict[str, dict[str, Path]],
    label_col: str = "LCZ_class",
    city_bounds: Path | None = None,
) -> list[PatchItem]:
    """Build PatchItems from the global So2Sat GPKG.

    Uses the 'dataset' column ('training'/'validation'/'testing') and maps it
    to the 'train'/'val'/'test' strings expected by PatchDataModule.

    ``city_bounds`` (default: so2sat_guppd_bounds.gpkg beside the patches GPKG)
    supplies the city per patch; without it the items carry city=None and
    Phase 4's per-city normalization cannot run on the global split.
    """
    _SPLIT_MAP = {"training": "train", "validation": "val", "testing": "test"}
    gdf = gpd.read_file(patches_gpkg)

    cities = None
    bounds_path = city_bounds or (patches_gpkg.parent / "so2sat_guppd_bounds.gpkg")
    if bounds_path.exists():
        try:
            cities = assign_cities(gdf, bounds_path)
            logger.info(
                f"Global split: city assigned from {bounds_path.name} "
                f"({cities.notna().sum()}/{len(gdf)} patches matched a city)"
            )
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"City assignment failed ({e}) — items will carry city=None")
    else:
        logger.warning(
            f"No city bounds at {bounds_path} — global items carry city=None, "
            "so per-city normalization (Phase 4) is unavailable for this split."
        )

    items: list[PatchItem] = []
    n_missing = 0
    for i, row in gdf.iterrows():
        pid = str(row["patch_id"])
        path = patch_index.get(str(row["dataset"]), {}).get(pid)
        if path is None:
            n_missing += 1
            continue
        split = _SPLIT_MAP.get(str(row["dataset"]))
        if split is None:
            continue
        label = int(row[label_col]) - 1   # 1-17 → 0-16
        city = None if cities is None else cities.get(i)
        items.append(PatchItem(path, label, split,
                               city=None if city is None or city != city else str(city)))
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
) -> list[PatchItem]:
    """Build weighted train items from a pseudo-label GeoPackage
    (generate_pseudo_labels.py output).

    Returns train PatchItems carrying a ``weight`` (scaled by ``weight_scale``)
    that flows through PatchDataset into the per-sample weighted CE loss.
    """
    gdf = gpd.read_file(pseudo_gpkg)
    items: list[PatchItem] = []
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
        items.append(PatchItem(path, label, "train", weight=weight))
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
) -> list[PatchItem]:
    """Build PatchItems for one city.

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
    items: list[PatchItem] = []
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
        items.append(PatchItem(path, label, split, city=city))

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
    max_invalid_frac: float = 1.0,
    embedding_names: list[str] | None = None,
    invalid_frac_parquet: Path | None = None,
    patch_manifest: Path | None = None,
) -> tuple[list[PatchItem], list[Path]]:
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

    ``max_invalid_frac`` (with ``embedding_names`` and ``invalid_frac_parquet``)
    applies the Task 1.75.1 nodata drop policy to the finished item list, so it
    behaves identically in all three modes. The default of 1.0 is a no-op.

    ``patch_manifest`` restricts the items to the Task 2.0 common manifest — the
    patches every in-scope family holds — so a cross-family comparison is run on
    one population. The default of None is a no-op.

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

    all_items = filter_by_invalid_fraction(
        all_items, max_invalid_frac, embedding_names, invalid_frac_parquet
    )
    all_items = filter_by_manifest(all_items, patch_manifest)

    return all_items, city_dirs


# ── Drop policy (Task 1.75.1) ────────────────────────────────────────────────

DEFAULT_INVALID_FRAC_PARQUET = (
    Path(__file__).resolve().parents[2] / "diagnostics" / "invalid_fraction.parquet"
)

_SCAN_COMMAND = "python src/diagnostics/nodata_population.py"
_MANIFEST_COMMAND = "python src/diagnostics/patch_manifest.py"


def patch_key(path: Path | tuple) -> tuple[str, str]:
    """``(dataset, patch_id)`` for a patch path, the only key that is unique.

    Extraction writes ``{so2sat_dir}/{split}/{output_name}/{year}/patch_{id}.npy``
    and each original split restarts patch_id at 000000, so the id alone
    resolves validation and testing patches to the wrong rows. Fused items carry
    a tuple of paths, one per source; they all describe the same patch, so the
    first is enough.
    """
    p = Path(path[0] if isinstance(path, tuple) else path)
    return p.parents[2].name, p.stem[len("patch_"):]


def filter_by_invalid_fraction(
    items: list[PatchItem],
    max_invalid_frac: float,
    embedding_names: list[str] | None,
    parquet: Path | None = None,
) -> list[PatchItem]:
    """Drop patches whose nodata fraction exceeds *max_invalid_frac*.

    Applied to training *and* evaluation alike: a threshold that filtered only
    the training set would report accuracy on data the model was never allowed
    to learn from, which is a different experiment from the one being run.

    The fractions come from the Task 1.75.1 parquet rather than being measured
    here. Measuring them would mean reading all 352,366 training files at the
    start of every run; the parquet turns that into one lookup. A missing
    parquet is an error naming the command that writes it, never a silent
    fallback to no filtering.

    For a fused run the **maximum** fraction across the requested sources
    decides, since fusion concatenates them and one bad source contaminates the
    whole sample.

    ``max_invalid_frac >= 1.0`` returns *items* itself, unchanged and in order —
    the default has to be a provable no-op because every Phase 2 baseline
    depends on it.
    """
    if max_invalid_frac >= 1.0:
        return items

    parquet = Path(parquet) if parquet is not None else DEFAULT_INVALID_FRAC_PARQUET
    if not parquet.exists():
        logger.error(
            f"--max-invalid-frac {max_invalid_frac} needs per-patch nodata "
            f"fractions, but {parquet} does not exist. Write it with:\n"
            f"    {_SCAN_COMMAND}"
        )
        raise SystemExit(1)

    df = pd.read_parquet(parquet, columns=["family", "dataset", "patch_id",
                                           "invalid_frac"])
    if embedding_names:
        known = set(df["family"].unique())
        missing = [n for n in embedding_names if n not in known]
        if missing:
            logger.error(
                f"{parquet.name} has no rows for {missing}. It covers "
                f"{sorted(known)}; re-run `{_SCAN_COMMAND} --families "
                f"{' '.join(embedding_names)}`."
            )
            raise SystemExit(1)
        df = df[df["family"].isin(embedding_names)]

    frac = (df.groupby(["dataset", "patch_id"])["invalid_frac"].max().to_dict())

    kept: list[PatchItem] = []
    dropped: dict[str, int] = {}
    n_unknown = 0
    for it in items:
        f = frac.get(patch_key(it.path))
        if f is None:
            # Not in the audit (e.g. an unlabeled or pseudo-labelled patch).
            # Keeping it is the conservative choice: the filter must not become
            # an accidental second coverage restriction.
            n_unknown += 1
            kept.append(it)
        elif f <= max_invalid_frac:
            kept.append(it)
        else:
            dropped[it.split] = dropped.get(it.split, 0) + 1

    n_dropped = sum(dropped.values())
    detail = ", ".join(f"{s} {n}" for s, n in sorted(dropped.items())) or "none"
    logger.info(
        f"--max-invalid-frac {max_invalid_frac}: dropped {n_dropped} of "
        f"{len(items)} patches ({detail})"
    )
    if n_unknown:
        logger.warning(
            f"{n_unknown} patches had no entry in {parquet.name} and were kept."
        )
    return kept


def manifest_sha256(path: Path | None) -> str | None:
    """SHA-256 of a manifest file, for the run config and the checkpoint.

    Recording the path alone would not survive the file being rebuilt with
    different thresholds, which is exactly the change a later reader would most
    need to detect.
    """
    if path is None:
        return None
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def filter_by_manifest(
    items: list[PatchItem],
    manifest: Path | None,
) -> list[PatchItem]:
    """Restrict *items* to the Task 2.0 common manifest.

    The three in-scope families do not hold the same patches — on the training
    split `tesserav1.1_global` is missing 9,422 that the other two have, and it
    evaluates on 330 fewer test patches — so a cross-family table built on each
    family's own coverage compares numbers computed on different data. The
    manifest is the intersection that removes that confound.

    Three outcomes per item, and the third is the one that needs care:

    * in the manifest as a member — kept;
    * in the manifest as a non-member — dropped, from training *and* evaluation,
      since restricting only the training set would report accuracy on patches
      the model was never allowed to learn from;
    * **absent from the manifest entirely** — kept, with a warning. The manifest
      covers the three So2Sat reference splits; unlabeled and pseudo-labelled
      pools are outside its universe and dropping them would make this filter a
      silent second coverage restriction. Same rule as
      ``filter_by_invalid_fraction``.

    ``manifest is None`` returns *items* itself, unchanged and in order. Task
    2.1's anchor and every Arm B run depend on that being a provable no-op, not
    merely an equal-valued one.
    """
    if manifest is None:
        return items

    manifest = Path(manifest)
    if not manifest.exists():
        logger.error(
            f"--patch-manifest {manifest} does not exist. Write it with:\n"
            f"    {_MANIFEST_COMMAND}"
        )
        raise SystemExit(1)

    df = pd.read_parquet(manifest, columns=["dataset", "patch_id", "in_manifest"])
    member = dict(zip(zip(df["dataset"].astype(str), df["patch_id"].astype(str)),
                      df["in_manifest"].to_numpy()))

    kept: list[PatchItem] = []
    dropped: dict[str, int] = {}
    n_unknown = 0
    for it in items:
        m = member.get(patch_key(it.path))
        if m is None:
            n_unknown += 1
            kept.append(it)
        elif m:
            kept.append(it)
        else:
            dropped[it.split] = dropped.get(it.split, 0) + 1

    n_dropped = sum(dropped.values())
    detail = ", ".join(f"{s} {n}" for s, n in sorted(dropped.items())) or "none"
    logger.info(
        f"--patch-manifest {manifest.name}: dropped {n_dropped} of {len(items)} "
        f"patches ({detail})"
    )
    if n_unknown:
        logger.warning(
            f"{n_unknown} patches were outside {manifest.name}'s universe and "
            "were kept."
        )
    return kept


class PackedPatchStore:
    """Reader for the memory-mapped shards written by ``src/pack_patches.py``.

    Opening one file per patch across ~400k files dominates input-pipeline time;
    the shards turn that into a memmap slice. Shards are opened lazily and cached
    per worker process, so each DataLoader worker keeps only what it touches.
    """

    def __init__(self, packed_dir: Path) -> None:
        import json
        self.dir = Path(packed_dir)
        index_path = self.dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index.json in {self.dir} — run src/pack_patches.py first."
            )
        self.index = json.loads(index_path.read_text())
        self.prefix = self.index.get("prefix", "patch_")
        # {patch_id: (split, shard, row)} across every packed split
        self.lookup: dict[str, tuple[str, int, int]] = {}
        for split, entry in self.index["splits"].items():
            for pid, (shard, row) in entry["patches"].items():
                self.lookup[f"{split}/{pid}"] = (split, shard, row)
        self._shards: dict[tuple[str, int], np.ndarray] = {}

    def _key(self, path: Path) -> str:
        """Map an original patch path to its packed key ({split}/{patch_id}).

        The original So2Sat splits restart patch_id at 000000, so the id alone
        is ambiguous and the split has to qualify it (same rule as
        build_patch_index).
        """
        pid = path.stem[len(self.prefix):]
        for part in path.parts[::-1]:
            if part in ("training", "validation", "testing", "unlabeled"):
                return f"{part}/{pid}"
        raise KeyError(f"Cannot infer split for {path}")

    def get(self, path: Path) -> np.ndarray | None:
        """Return the patch array, or None when it is not in the pack."""
        try:
            key = self._key(path)
        except KeyError:
            return None
        hit = self.lookup.get(key)
        if hit is None:
            return None
        split, shard, row = hit
        mm = self._shards.get((split, shard))
        if mm is None:
            mm = np.load(self.dir / f"{split}_{shard:04d}.npy", mmap_mode="r")
            self._shards[(split, shard)] = mm
        return np.asarray(mm[row], dtype=np.float32)


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
        items: list[PatchItem],
        patch_size: int,
        sub_patch_size: int | None = None,
        sub_patch_stride: int | None = None,
        dequantize_fn=None,
        nodata_mode: str = "zero",
        nodata_predicate=None,
        normalize: str = "none",
        channel_mean=None,
        channel_std=None,
        packed_dir: Path | None = None,
    ) -> None:
        self.packed = PackedPatchStore(packed_dir) if packed_dir else None
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

        if items and isinstance(items[0].path, tuple) and sub_patch_size is not None:
            raise ValueError("sub_patch_size is not supported with fused (multi-source) items")

        # The "weight" batch key is only emitted when some item carries one,
        # so unweighted runs keep their exact previous batch format.
        self.has_weights = any(it.weight is not None for it in items)

        def _w(it: PatchItem) -> float:
            return 1.0 if it.weight is None else float(it.weight)

        if sub_patch_size is None:
            self.expanded = [(it.path, it.label, None, None, _w(it)) for it in items]
        else:
            stride = self.sub_patch_stride
            self.expanded = []
            for it in items:
                path, label = it.path, it.label
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
        raw = None
        if self.packed is not None:
            raw = self.packed.get(path)          # memmap slice, no per-file open
        if raw is None:
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
        noise_sigma: float = 0.05,
        noise_prob: float = 0.5,
        packed_dir: Path | None = None,
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
        self.noise_sigma = noise_sigma
        self.noise_prob = noise_prob
        self.packed_dir = packed_dir

    def setup(self) -> None:
        def _for_split(s: str) -> list:
            return [it for it in self.all_items if it.split == s]

        kw = dict(sub_patch_size=self.sub_patch_size, sub_patch_stride=self.sub_patch_stride,
                  dequantize_fn=self.dequantize_fn, nodata_mode=self.nodata_mode,
                  nodata_predicate=self.nodata_predicate, normalize=self.normalize,
                  channel_mean=self.channel_mean, channel_std=self.channel_std,
                  packed_dir=self.packed_dir)
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
            num_workers=self.num_workers, collate_fn=self._collate, drop_last=True,
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
