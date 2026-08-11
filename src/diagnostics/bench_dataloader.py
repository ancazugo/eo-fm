"""Task 1.8 — patches/sec benchmark for the classification input pipeline.

Measures the two costs the training loop actually pays: reading and decoding
patches (``__getitem__`` across ~400k small ``.npy`` files) and augmenting the
batch. Run it before and after a pipeline change with the same flags; the
numbers are only comparable within one machine and one page-cache state, so
always report a matched pair.

Example:

    python src/diagnostics/bench_dataloader.py \\
        --output-name AlphaEarthCoop --embedding-name alpha_earth_coop \\
        --cities Nairobi --batches 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.registry import get_nodata_predicate  # noqa: E402
from datasets.so2sat import PatchDataModule, build_so2sat_items  # noqa: E402
from training.augment import augment_images  # noqa: E402
from utils.constants import DATA_DIR  # noqa: E402
from utils.runtime import resolve_dequantize, resolve_device  # noqa: E402


def bench_augment(
    batch: torch.Tensor, device: torch.device, repeats: int = 50
) -> dict[str, float]:
    """Time the augmentation alone, on CPU (collate) and on GPU (train_step)."""
    out = {}
    x = batch.clone()
    t0 = time.perf_counter()
    for _ in range(repeats):
        augment_images(x)
    out["augment_cpu_patches_per_s"] = repeats * x.shape[0] / (time.perf_counter() - t0)

    if device.type == "cuda":
        xg = batch.to(device)
        for _ in range(3):                       # warm up kernels
            augment_images(xg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            augment_images(xg)
        torch.cuda.synchronize()
        out["augment_gpu_patches_per_s"] = repeats * xg.shape[0] / (time.perf_counter() - t0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark the patch input pipeline.")
    p.add_argument("--so2sat-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4")
    p.add_argument("--cities-dir", type=Path,
                   default=DATA_DIR / "input" / "So2Sat-LCZ42" / "v4" / "cities")
    p.add_argument("--cities", nargs="+", default=["Nairobi"])
    p.add_argument("--global-split", action="store_true")
    p.add_argument("--output-name", default="AlphaEarthCoop")
    p.add_argument("--embedding-name", default="alpha_earth_coop")
    p.add_argument("--year", default="2017")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--repeats", type=int, default=3,
                   help="Alternating A/B rounds; results are averaged.")
    p.add_argument("--packed-dir", type=Path, default=None)
    p.add_argument("--nodata-mode", choices=["zero", "mask"], default="mask")
    p.add_argument("--accelerator", default="auto")
    args = p.parse_args()

    device = resolve_device(args.accelerator)
    items, _ = build_so2sat_items(
        args.so2sat_dir, args.output_name, args.year,
        global_split=args.global_split, cities_dir=args.cities_dir,
        cities=args.cities,
    )
    dequantize_fn, _ = resolve_dequantize(args.embedding_name)

    dm = PatchDataModule(
        items, args.patch_size, args.batch_size, args.num_workers,
        dequantize_fn=dequantize_fn, nodata_mode=args.nodata_mode,
        nodata_predicate=get_nodata_predicate(args.embedding_name),
        packed_dir=args.packed_dir,
    )
    dm.setup()
    loader = dm.train_dataloader()

    def run_once(augment_in_collate: bool) -> tuple[int, float, torch.Tensor]:
        """One pass over --batches, optionally augmenting inside the collate.

        ``augment_in_collate=True`` reproduces the pre-Task-1.8 pipeline, where
        the per-sample Python augmentation ran in the DataLoader worker.
        """
        dm._collate_augment = augment_in_collate
        ld = dm.train_dataloader()
        i = iter(ld)
        first_batch = next(i)                     # warm workers, exclude startup
        n = 0
        t0 = time.perf_counter()
        for _ in range(args.batches):
            try:
                b = next(i)
            except StopIteration:
                break
            n += b["image"].shape[0]
        return n, time.perf_counter() - t0, first_batch["image"]

    # Alternate the two arms so page-cache state cannot favour either: caching
    # effects between separate invocations dwarf the difference being measured.
    base_collate = dm._collate
    dm._collate_augment = False

    def collate(batch):
        out = base_collate(batch)
        if getattr(dm, "_collate_augment", False):
            out["image"] = augment_images(out["image"])
        return out

    dm._collate = collate

    results: dict[bool, list[float]] = {True: [], False: []}
    first = None
    for _ in range(args.repeats):
        for mode in (True, False):
            n, dt, img = run_once(mode)
            results[mode].append(n / dt)
            first = img

    logger.info(f"Device: {device}  workers={args.num_workers}  bs={args.batch_size}  "
                f"packed={'yes' if dm.packed_dir else 'no'}")
    for mode, label in ((True, "augment in collate (pre-1.8)"),
                        (False, "augment in train_step (post-1.8)")):
        r = results[mode]
        logger.info(f"Loader, {label:34s}: {sum(r) / len(r):8,.0f} patches/s "
                    f"(runs: {', '.join(f'{v:,.0f}' for v in r)})")
    for k, v in bench_augment(first, device).items():
        logger.info(f"{k}: {v:,.0f}")


if __name__ == "__main__":
    main()
