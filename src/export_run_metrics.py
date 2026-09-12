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

The **segmentation** campaign is the same table with different numbers, so it is
the same export with different flags rather than a second script: `--task
segmentation`, the `*_patch_exact` metric names (a segmentation run is scored on
the So2Sat patches it covers exactly, which is what makes it comparable to a
classification run at all) and `--metrics` to name them.

Example:

    python src/export_run_metrics.py
    python src/export_run_metrics.py --since 2026-06-01 --embeddings GeoTessera_v1.1_global
    python src/export_run_metrics.py --task segmentation --output data/seg_metrics.csv \
        --metrics test_acc_patch_exact test_oau_patch_exact ... --include-unfinished
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

# The same four table columns for a segmentation run, plus mIoU. A segmentation
# model predicts pixels, so it has no "test_acc" that means what a classifier's
# does: `_patch_exact` is the evaluation restricted to the So2Sat patches the
# prediction covers exactly, which is the only footing on which the two campaigns
# compare. mIoU has no patch-exact form and is reported at the native 10 m grid.
SEG_METRIC_KEYS = [
    "test_acc_patch_exact",
    "test_oau_patch_exact",
    "test_oabu_patch_exact",
    "test_oaw_patch_exact",
    "test_acc_macro_patch_exact",
    "test_f1_patch_exact",
    "test_f1_micro_patch_exact",
    "test_kappa_patch_exact",
    "test_kappa_w_patch_exact",
    "test_miou",
    "test_n_patch_exact",
    "test_coverage_patch_exact",
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

LEAD_KEYS = ["run_name", "run_id", "created_at", "state"]

# How a segmentation run says which split it was trained on. Classification
# writes `split_source` into the config; the segmentation pipeline takes it as a
# CLI flag, so it is recovered from the recorded argv and mapped into the SAME
# vocabulary, which is what lets one R script label both tables.
SPLIT_MODE_ARG = "--split-mode"
SPLIT_MODE_TO_SOURCE = {"global": "global_so2sat", "orig_test": "grid_orig_test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default="phd-thesis-team")
    p.add_argument("--project", default="lcz-classification-dl")
    p.add_argument("--task", default="patch_classification",
                   help="config['task'] to keep; 'any' disables the filter. "
                        "'segmentation' also switches the default metric set to "
                        "SEG_METRIC_KEYS, unless --metrics says otherwise.")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="summary keys to export, in column order "
                        "(default: the metric set for --task)")
    p.add_argument("--include-unfinished", action="store_true",
                   help="also emit a row for a run that is still going, with "
                        "every metric blank. The cell is real -- it is in the "
                        "ladder and its size is already known -- so the table "
                        "should show it pending rather than not at all. A "
                        "finished run always wins the cell over an unfinished "
                        "one, whichever is newer.")
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


def split_source_of(run, cfg: dict) -> str | None:
    """The run's split, from the config if it is there and argv if it is not."""
    if cfg.get("split_source"):
        return cfg["split_source"]
    argv = (run.metadata or {}).get("args") or []
    if SPLIT_MODE_ARG in argv:
        mode = argv[argv.index(SPLIT_MODE_ARG) + 1]
        return SPLIT_MODE_TO_SOURCE.get(mode, mode)
    return None


def arch_of(cfg: dict) -> str | None:
    """The architecture string the table labels a row by.

    Classification configs carry the timm name (`resnet34`); the segmentation
    pipeline has no such field because the family *is* the architecture and the
    preset is its capacity, so the two are joined into one key the label lookup
    can resolve (`unet_small`).
    """
    if cfg.get("arch"):
        return cfg["arch"]
    if cfg.get("family") and cfg.get("preset"):
        return f"{cfg['family']}_{cfg['preset']}"
    return None


def main() -> None:
    args = parse_args()

    metric_keys = args.metrics or (
        SEG_METRIC_KEYS if args.task == "segmentation" else METRIC_KEYS)
    fieldnames = LEAD_KEYS + CONFIG_KEYS + metric_keys

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    logger.info(f"Scanning {path} (since {args.since}, embeddings {args.embeddings})")

    # One pass over the project; the filtering below is cheap next to the fetch.
    cells: dict[tuple, "wandb.apis.public.Run"] = {}
    superseded: list[tuple[str, str]] = []
    n_seen = 0

    for run in api.runs(path, per_page=500):
        n_seen += 1
        cfg = dict(run.config)
        if args.task != "any" and cfg.get("task") != args.task:
            continue
        if cfg.get("embedding") not in args.embeddings:
            continue
        cfg["split_source"] = split_source_of(run, cfg)
        cfg["arch"] = arch_of(cfg)
        if cfg["split_source"] not in args.split_sources:
            continue
        if run.created_at < args.since:
            continue
        if run.state != "finished" and not (args.include_unfinished
                                            and run.state == "running"):
            logger.debug(f"skip {run.name}: state={run.state}")
            continue

        key = tuple(cfg.get(k) for k in CELL_KEYS)
        incumbent = cells.get(key)
        if incumbent is None:
            cells[key] = (run, cfg)
        else:
            # Finished beats unfinished whatever the dates say: a running run has
            # no test metrics, so letting it supersede on recency alone would
            # blank a cell that is already measured.
            old_run = incumbent[0]
            better = (run.state == "finished", run.created_at) > \
                     (old_run.state == "finished", old_run.created_at)
            winner, loser = (run, old_run) if better else (old_run, run)
            if better:
                cells[key] = (run, cfg)
            superseded.append((loser.name, winner.name))

    logger.info(f"{n_seen} runs scanned, {len(cells)} cells kept")
    for dropped, kept in superseded:
        logger.warning(f"superseded: '{dropped}' dropped in favour of '{kept}'")

    rows = []
    for key in sorted(cells, key=lambda k: tuple(str(v) for v in k)):
        run, cfg = cells[key]
        row = {"run_name": run.name, "run_id": run.id,
               "created_at": run.created_at, "state": run.state}
        row.update({k: cfg.get(k) for k in CONFIG_KEYS})
        row.update({k: run.summary.get(k) for k in metric_keys})
        rows.append(row)
        missing = [k for k in metric_keys if row[k] is None]
        if missing and run.state == "finished":
            logger.warning(f"{run.name}: no summary value for {', '.join(missing)}")
        elif missing:
            logger.info(f"{run.name}: {run.state}, exported as a pending cell")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.success(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
