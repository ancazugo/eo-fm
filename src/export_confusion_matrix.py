"""Export the saved test confusion matrices to a tidy CSV for the R figure stack.

Every classification run already writes a dense 17x17 `test_confusion_matrix.npy`
next to its checkpoint (`training/evaluate.py::dense_confusion_matrix`), but R has
no `.npy` reader in the `r-environment` conda env. This caches those matrices as
one long CSV, exactly as `src/export_run_metrics.py` caches the test metrics and
`R/prepare_city_data.R` caches the city summary -- R reads only the CSV.

The run set is taken from `data/model_metrics.csv`, so the confusion matrices and
the results table are always the same cells: the run directory name on disk is
exactly the `run_name` column.

**Counts only.** Row-normalisation is one `group_by(run_name, true_code)` in R,
and exporting only the raw counts keeps the CSV the single source of truth --
there is no second, derived number that can drift out of step with it.

`cm[i, j]` is *true* i, *predicted* j, 0-indexed over ALL 17 classes whether or
not a class appears. The CSV re-indexes to the 1-indexed LCZ codes the rest of
the project uses (`utils.constants.lcz_dict`, `R/constants.R::LCZ_TABLE`).

Read-only: reads run artefacts under $DATA_DIR, touches neither W&B nor a model.

Example:

    python src/export_confusion_matrix.py
    python src/export_confusion_matrix.py --matrix-name test_confusion_matrix_patch.npy
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from loguru import logger

# Carried through from data/model_metrics.csv so the CSV is self-describing: R
# can facet or title a matrix without joining back to the metrics table.
CONTEXT_KEYS = ["embedding", "split_source", "family", "arch", "preset"]

FIELDNAMES = ["run_name", *CONTEXT_KEYS, "true_code", "pred_code", "count"]

NUM_CLASSES = 17


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-csv", type=Path, default=Path("data/model_metrics.csv"),
                   help="run set to export, one row per run (default: the results table)")
    p.add_argument("--run-root", type=Path, default=None,
                   help="directory holding the run dirs "
                        "(default: $DATA_DIR/output/lcz-classification/dl)")
    p.add_argument("--matrix-name", default="test_confusion_matrix.npy",
                   help="matrix filename inside each run dir")
    p.add_argument("--matrix", action="append", default=[], metavar="NAME=PATH",
                   help="export this .npy directly, as run NAME. Repeatable. "
                        "Given at least once, --metrics-csv is not read at all: "
                        "the run set is exactly what is named here. For matrices "
                        "that belong to no row of the results table -- a "
                        "segmentation run, an ad-hoc evaluation -- pair it with "
                        "its own --output so the classification cache is left "
                        "alone.")
    p.add_argument("--output", type=Path, default=Path("data/confusion_matrices.csv"))
    return p.parse_args()


def named_matrices(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse NAME=PATH arguments, splitting on the FIRST '=' only.

    A run name never contains '=' but a path may, and getting this backwards
    would silently truncate the path rather than fail.
    """
    out = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--matrix wants NAME=PATH, got {spec!r}")
        out.append((name, Path(path)))
    return out


def main() -> None:
    args = parse_args()

    run_root = args.run_root
    if run_root is None:
        from utils.paths import OUTPUT_DIR
        run_root = OUTPUT_DIR / "lcz-classification" / "dl"

    # Two ways to name a run set. --matrix gives (name, path) outright and is
    # the escape hatch for a matrix with no results-table row; otherwise the run
    # set is the table, and the path is assembled per run.
    if args.matrix:
        named = named_matrices(args.matrix)
        runs = [{"run_name": n, "__path": p} for n, p in named]
        logger.info(f"{len(runs)} matrix/matrices named on the command line")
    else:
        with args.metrics_csv.open() as fh:
            runs = list(csv.DictReader(fh))
        logger.info(f"{len(runs)} runs in {args.metrics_csv}; "
                    f"matrices under {run_root}")

    rows: list[dict] = []
    missing: list[str] = []

    for run in runs:
        name = run["run_name"]
        path = run.get("__path") or run_root / name / args.matrix_name
        if not path.exists():
            # Skip, never fail: a still-training cell must not block the export.
            missing.append(name)
            continue

        # float is what a normalised or averaged matrix comes back as; the counts
        # are still integral, and int() below is exact for anything under 2^53.
        cm = np.load(path)
        if cm.shape != (NUM_CLASSES, NUM_CLASSES):
            logger.warning(f"{name}: expected {NUM_CLASSES}x{NUM_CLASSES}, "
                           f"got {cm.shape} -- skipped")
            missing.append(name)
            continue

        context = {k: run.get(k, "") for k in CONTEXT_KEYS}
        total = int(cm.sum())
        expected = run.get("test_patches")
        if expected and total != int(float(expected)):
            # Worth shouting about: the matrix and the metrics row should describe
            # the same evaluation, so a mismatch means one of them is stale.
            logger.warning(f"{name}: matrix sums to {total:,} but test_patches "
                           f"is {int(float(expected)):,}")

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                rows.append({"run_name": name, **context,
                             "true_code": i + 1, "pred_code": j + 1,
                             "count": int(cm[i, j])})

        logger.debug(f"{name}: {total:,} samples")

    for name in missing:
        logger.warning(f"no usable '{args.matrix_name}' for '{name}' -- skipped")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    kept = len(runs) - len(missing)
    logger.success(f"Wrote {len(rows):,} rows ({kept} runs) to {args.output}")


if __name__ == "__main__":
    main()
