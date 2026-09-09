"""Export the 2026-09 architecture-campaign test metrics from W&B to a CSV.

The R figure stack (`R/constants.R`, `R/plotting.R`, ...) draws every summary
figure for the paper and poster, but R has no W&B client, so the numbers have to
be cached first. This mirrors `R/prepare_city_data.R`, which caches
`data/so2sat_city_summary.csv` for `R/plotting.R` to read.

The campaign compares **GeoTessera v2** against **AlphaEarth (coop)** across the
culture-10 (`global_so2sat`) and hybrid (`grid_orig_test`) splits for several
small architectures. Runs are selected by *config*, not by name: the naming is
inconsistent across the campaign (`v2-cultural-*`, `newarch-cls-*-coop`,
`stemfix-cls-*-coop`), while `(embedding, split_source, family, preset)`
identifies a cell exactly.

Where two runs fill the same cell the **newest finished run wins** — that is what
supersedes the pre-stem-fix timm numbers, which the stem-fix ledger records as
void. Every superseded run is logged, so a supersede is never silent.

Read-only against W&B. Credentials come from ~/.netrc (or WANDB_API_KEY).

Example:

    python src/export_run_metrics.py
    python src/export_run_metrics.py --since 2026-06-01 --embeddings GeoTessera_v1.1_global
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loguru import logger

import wandb

# The four leading columns of the table plus every test metric the runs log.
# All ten metrics are exported even though the figure shows four: re-picking a
# column then costs no W&B round-trip.
METRIC_KEYS = [
    "test_acc",
    "test_oau",
    "test_oabu",
    "test_oaw",
    "test_acc_macro",
    "test_f1",
    "test_f1_micro",
    "test_kappa",
    "test_kappa_w",
    "test_loss",
]

CONFIG_KEYS = [
    "embedding",
    "embedding_name",
    "split_source",
    "family",
    "arch",
    "preset",
    "n_params",
    "seed",
    "test_patches",
]

# (embedding, split_source, family, preset) — one row of the table.
CELL_KEYS = ("embedding", "split_source", "family", "preset")

FIELDNAMES = ["run_name", "run_id", "created_at"] + CONFIG_KEYS + METRIC_KEYS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default="phd-thesis-team")
    p.add_argument("--project", default="lcz-classification-dl")
    p.add_argument("--task", default="patch_classification",
                   help="config['task'] to keep; 'any' disables the filter.")
    p.add_argument("--embeddings", nargs="+",
                   default=["GeoTessera_v2", "AlphaEarthCoop"],
                   help="config['embedding'] values to keep.")
    p.add_argument("--split-sources", nargs="+",
                   default=["global_so2sat", "grid_orig_test"],
                   help="config['split_source'] values to keep.")
    p.add_argument("--since", default="2026-09-05",
                   help="ISO date; runs created before it are dropped.")
    p.add_argument("--output", type=Path, default=Path("data/model_metrics.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    logger.info(f"Scanning {path} (since {args.since}, embeddings {args.embeddings})")

    # One pass over the project; the filtering below is cheap next to the fetch.
    cells: dict[tuple, "wandb.apis.public.Run"] = {}
    superseded: list[tuple[str, str]] = []
    n_seen = 0

    for run in api.runs(path, per_page=500):
        n_seen += 1
        cfg = run.config
        if args.task != "any" and cfg.get("task") != args.task:
            continue
        if cfg.get("embedding") not in args.embeddings:
            continue
        if cfg.get("split_source") not in args.split_sources:
            continue
        if run.created_at < args.since:
            continue
        if run.state != "finished":
            logger.debug(f"skip {run.name}: state={run.state}")
            continue

        key = tuple(cfg.get(k) for k in CELL_KEYS)
        incumbent = cells.get(key)
        if incumbent is None:
            cells[key] = run
        elif run.created_at > incumbent.created_at:
            cells[key] = run
            superseded.append((incumbent.name, run.name))
        else:
            superseded.append((run.name, incumbent.name))

    logger.info(f"{n_seen} runs scanned, {len(cells)} cells kept")
    for dropped, kept in superseded:
        logger.warning(f"superseded: '{dropped}' dropped in favour of '{kept}'")

    rows = []
    for key in sorted(cells, key=lambda k: tuple(str(v) for v in k)):
        run = cells[key]
        row = {"run_name": run.name, "run_id": run.id, "created_at": run.created_at}
        row.update({k: run.config.get(k) for k in CONFIG_KEYS})
        row.update({k: run.summary.get(k) for k in METRIC_KEYS})
        rows.append(row)
        missing = [k for k in METRIC_KEYS if row[k] is None]
        if missing:
            logger.warning(f"{run.name}: no summary value for {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    logger.success(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
