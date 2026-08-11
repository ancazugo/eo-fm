"""Pack per-patch files into memory-mapped shards (Task 1.8).

The classification pipeline opens one file per patch across ~400k files per
split. That is a syscall and a header parse per sample, and it is the binding
constraint on how many experiments the plan can afford. Packing rewrites each
split as a handful of contiguous ``.npy`` shards plus a JSON index; the dataset
then memory-maps a shard and slices it, so a patch costs a memcpy.

Format-agnostic at pack time (dispatch on file suffix), so the same script
covers the raw Sentinel-1/2 GeoTIFFs needed by Task 3.4 — which benefit far
more than the ``.npy`` files, since each TIFF carries its own header and
compression.

Patches must share a common (C, H, W); the packer verifies this and reports the
offenders rather than writing a corrupt shard. Reading stays optional:
``PatchDataset`` uses per-file paths unless ``--packed-dir`` is given.

Example (embeddings):

    python src/pack_patches.py \\
        --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \\
        --output-name AlphaEarthCoop --year 2017 \\
        --out-dir ${DATA_DIR}/packed/AlphaEarthCoop_2017

Example (raw Sentinel-1, no year nesting, different prefix):

    python src/pack_patches.py \\
        --so2sat-dir ${DATA_DIR}/input/So2Sat-LCZ42/v4 \\
        --subdir sentinel1 --prefix sen1_patch_ --suffix .tif --no-year \\
        --out-dir ${DATA_DIR}/packed/sentinel1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.constants import DATA_DIR  # noqa: E402

SPLITS = ("training", "validation", "testing", "unlabeled")


def read_patch(path: Path) -> np.ndarray:
    """Read one patch as ``(C, H, W)`` float32, dispatching on suffix."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)
    raise ValueError(f"Unsupported patch format: {path.suffix} ({path})")


def pack_split(
    files: list[Path],
    out_dir: Path,
    split: str,
    prefix: str,
    shard_bytes: int,
) -> dict:
    """Write one split's patches into shards; return its index entry."""
    if not files:
        return {}

    sample = read_patch(files[0])
    shape = sample.shape
    per_patch = int(np.prod(shape)) * 4                      # float32
    per_shard = max(1, shard_bytes // per_patch)
    logger.info(
        f"[{split}] {len(files)} patches, shape={shape}, "
        f"{per_patch / 1e3:.1f} kB each → {per_shard} per shard"
    )

    entries: dict[str, list[int]] = {}
    bad: list[str] = []
    n_shards = (len(files) + per_shard - 1) // per_shard

    for si in range(n_shards):
        chunk = files[si * per_shard:(si + 1) * per_shard]
        shard_path = out_dir / f"{split}_{si:04d}.npy"
        arr = np.lib.format.open_memmap(
            shard_path, mode="w+", dtype=np.float32, shape=(len(chunk), *shape)
        )
        for j, f in enumerate(tqdm(chunk, desc=f"{split} shard {si}", leave=False)):
            a = read_patch(f)
            if a.shape != shape:
                bad.append(f"{f.name}: {a.shape} != {shape}")
                continue
            arr[j] = a
            entries[f.stem[len(prefix):]] = [si, j]
        arr.flush()
        del arr

    if bad:
        logger.warning(
            f"[{split}] {len(bad)} patches skipped for shape mismatch; first few: "
            + "; ".join(bad[:5])
        )
    return {"shape": list(shape), "n_shards": n_shards, "patches": entries,
            "n_skipped": len(bad)}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pack per-patch files into memory-mapped shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--output-name", default=None,
                   help="Embedding dir under each split (e.g. AlphaEarthCoop). "
                        "Mutually exclusive with --subdir.")
    p.add_argument("--subdir", default=None,
                   help="Literal subdir under each split (e.g. sentinel1), for "
                        "sources that are not nested by output-name/year.")
    p.add_argument("--year", default="2017")
    p.add_argument("--no-year", action="store_true",
                   help="The source has no {year} level (raw Sentinel patches).")
    p.add_argument("--prefix", default="patch_")
    p.add_argument("--suffix", default=".npy")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--shard-gb", type=float, default=2.0,
                   help="Target uncompressed size per shard.")
    args = p.parse_args()

    if (args.output_name is None) == (args.subdir is None):
        p.error("pass exactly one of --output-name or --subdir")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {
        "source": args.output_name or args.subdir,
        "prefix": args.prefix, "suffix": args.suffix,
        "year": None if args.no_year else args.year,
        "splits": {},
    }

    for split in args.splits:
        d = args.so2sat_dir / split / (args.output_name or args.subdir)
        if not args.no_year and args.output_name:
            d = d / args.year
        if not d.exists():
            logger.info(f"[{split}] {d} not found — skipping")
            continue
        files = sorted(d.glob(f"{args.prefix}*{args.suffix}"))
        entry = pack_split(files, args.out_dir, split, args.prefix,
                           int(args.shard_gb * 1e9))
        if entry:
            index["splits"][split] = entry

    (args.out_dir / "index.json").write_text(json.dumps(index))
    total = sum(len(v["patches"]) for v in index["splits"].values())
    logger.info(f"Packed {total} patches into {args.out_dir}")


if __name__ == "__main__":
    main()
