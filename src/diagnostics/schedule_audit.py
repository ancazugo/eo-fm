"""PLAN-v3 Phase 2, Tasks 2.1b-iii and 2.1c — peak epoch, not just the metric.

Rev C asks for "the epoch of best val_kappa for every run, not just the metric",
because the schedule question is not answered by the metric alone. Two of the
three Task 2.1 anchor seeds selected their best checkpoint at epoch 2 with
``--warmup-epochs 3`` — before warmup finished, at a low learning rate — while
the third peaked at 15. A recipe whose peak epoch is bimodal across seeds is
sitting on an unstable point, and that is visible only if peak epoch is reported.

Peak epoch comes from the W&B history rather than the checkpoint: the checkpoint
records the epoch it was written at, which is the same number, but the history
also gives the *stop* epoch and the whole val_kappa trace, so a run that peaked
early and then degraded is distinguishable from one that plateaued.

W&B counts history steps from 0 and the training loop logs epochs from 1, so
peak epoch is ``argmax(val_kappa) + 1``. Verified against the anchor logs, which
independently give 2, 2 and 15.

Reads W&B only. Writes ``diagnostics/schedule_audit.{json,md}``.

    python src/diagnostics/schedule_audit.py --prefix p2-anchor-tessera
    python src/diagnostics/schedule_audit.py --prefix p2-1c- --group-by lr warmup_epochs
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# macro-F1 first, then OA, then kappa — Amendment B3's ordering.
METRICS = [("macro_f1", "test_f1"), ("oa", "test_acc"), ("kappa", "test_kappa")]


def collect(entity: str, project: str, prefix: str) -> list[dict]:
    """Every finished run whose name starts with *prefix*, with its peak epoch."""
    import wandb

    api = wandb.Api()
    runs = [r for r in api.runs(f"{entity}/{project}", per_page=200)
            if r.name.startswith(prefix)]
    logger.info(f"{len(runs)} runs matching '{prefix}*' in {entity}/{project}")

    rows = []
    for run in runs:
        if run.state != "finished":
            logger.warning(f"{run.name}: state={run.state} — skipped")
            continue
        history = run.history(keys=["val_kappa"], pandas=True)
        if history.empty:
            logger.warning(f"{run.name}: no val_kappa history — skipped")
            continue
        trace = history["val_kappa"].tolist()
        peak = max(range(len(trace)), key=trace.__getitem__)
        cfg = run.config
        rows.append({
            "run_name": run.name,
            "run_id": run.id,
            "seed": cfg.get("seed"),
            "lr": cfg.get("lr"),
            "warmup_epochs": cfg.get("warmup_epochs"),
            "patch_manifest": cfg.get("patch_manifest"),
            "normalize": cfg.get("normalize"),
            "peak_epoch": peak + 1,           # history is 0-based, epochs are 1-based
            "stop_epoch": len(trace),
            "best_val_kappa": trace[peak],
            "peaked_during_warmup": peak + 1 <= (cfg.get("warmup_epochs") or 0),
            **{name: run.summary.get(key) for name, key in METRICS},
        })
    return sorted(rows, key=lambda r: (str(r["lr"]), r["warmup_epochs"] or 0,
                                       r["seed"] if r["seed"] is not None else -1))


def _stats(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def summarise(rows: list[dict], group_by: list[str]) -> list[dict]:
    """Per-group mean +- sd for each metric, and the peak-epoch distribution."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in group_by)].append(r)

    out = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        entry = {
            "group": dict(zip(group_by, key)),
            "runs": len(members),
            "peak_epochs": sorted(m["peak_epoch"] for m in members),
            "stop_epochs": sorted(m["stop_epoch"] for m in members),
            "peaked_during_warmup": sum(m["peaked_during_warmup"] for m in members),
            "peak_epoch_stats": _stats([m["peak_epoch"] for m in members]),
        }
        for name, _ in METRICS:
            entry[name] = _stats([m[name] for m in members])
        out.append(entry)
    return out


def bimodality(rows: list[dict]) -> dict:
    """Is peak epoch bimodal, or is it one cluster?

    Not a formal test — n is far too small. It reports the gap structure that
    Task 2.1b-iii's pre-registration is about: the anchor's 2, 2, 15 either
    persists as two clusters or fills in. The largest gap between consecutive
    sorted peak epochs, against the spread, is the honest summary of that.
    """
    peaks = sorted(r["peak_epoch"] for r in rows)
    if len(peaks) < 3:
        return {"n": len(peaks), "peak_epochs": peaks, "verdict": "too few runs"}
    gaps = [(peaks[i + 1] - peaks[i], i) for i in range(len(peaks) - 1)]
    widest, at = max(gaps)
    low, high = peaks[: at + 1], peaks[at + 1:]
    return {
        "n": len(peaks),
        "peak_epochs": peaks,
        "widest_gap": widest,
        "span": peaks[-1] - peaks[0],
        "cluster_low": low,
        "cluster_high": high,
        "gap_fraction_of_span": widest / (peaks[-1] - peaks[0]) if peaks[-1] > peaks[0] else 0.0,
    }


def markdown_report(payload: dict) -> str:
    rows, groups, meta = payload["runs"], payload["groups"], payload["meta"]
    L = [f"### Schedule audit — `{meta['prefix']}*`", "",
         f"{len(rows)} finished runs. Generated {meta['generated']}.", "",
         "Peak epoch is the epoch of best `val_kappa`, which is the checkpoint that",
         "gets evaluated. A peak inside the warmup window means the best model was",
         "selected before the learning rate finished ramping.", "",
         "#### Per run", "",
         "| run | lr | warmup | seed | peak epoch | stop epoch | best val_kappa | macro-F1 | OA | kappa |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = "**" if r["peaked_during_warmup"] else ""
        L.append(
            f"| `{r['run_name']}` | {r['lr']} | {r['warmup_epochs']} | {r['seed']} | "
            f"{mark}{r['peak_epoch']}{mark} | {r['stop_epoch']} | {r['best_val_kappa']:.4f} | "
            + " | ".join(f"{r[n]:.4f}" if r[n] is not None else "—" for n, _ in METRICS) + " |")
    L += ["", "Bold peak epoch = the best checkpoint was chosen during warmup.", ""]

    if groups:
        keys = list(groups[0]["group"])
        L += ["#### Grouped", "",
              "| " + " | ".join(keys) + " | n | peak epochs | in warmup | macro-F1 | OA | kappa |",
              "|" + "---|" * (len(keys) + 6)]
        for g in groups:
            cells = [str(g["group"][k]) for k in keys]
            metrics = " | ".join(
                f"{g[n]['mean']:.4f} ± {g[n]['sd']:.4f}" if g[n]["n"] else "—"
                for n, _ in METRICS)
            L.append(f"| {' | '.join(cells)} | {g['runs']} | "
                     f"{', '.join(map(str, g['peak_epochs']))} | "
                     f"{g['peaked_during_warmup']}/{g['runs']} | {metrics} |")
        L.append("")

    b = payload["bimodality"]
    L += ["#### Peak-epoch distribution", "",
          f"n = {b['n']}, peak epochs {b['peak_epochs']}"]
    if "widest_gap" in b:
        L += ["",
              f"Widest gap between consecutive peaks: **{b['widest_gap']} epochs** over a span "
              f"of {b['span']} ({b['gap_fraction_of_span']:.0%} of the span), splitting "
              f"{b['cluster_low']} from {b['cluster_high']}.", "",
              "This is a description, not a test — n is far too small for one."]
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Peak epoch and metrics for a family of W&B runs (Tasks 2.1b-iii, 2.1c).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--entity", default="phd-thesis-team")
    p.add_argument("--project", default="lcz-classification-dl")
    p.add_argument("--prefix", required=True, help="Run-name prefix to collect.")
    p.add_argument("--group-by", nargs="*", default=["lr", "warmup_epochs"],
                   help="Config fields to aggregate over.")
    p.add_argument("--output-json", type=Path, default=Path("diagnostics/schedule_audit.json"))
    p.add_argument("--output-md", type=Path, default=Path("diagnostics/schedule_audit.md"))
    args = p.parse_args()

    rows = collect(args.entity, args.project, args.prefix)
    if not rows:
        raise SystemExit(f"No finished runs matching '{args.prefix}*'.")

    payload = {
        "meta": {
            "prefix": args.prefix,
            "entity": args.entity,
            "project": args.project,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_runs": len(rows),
        },
        "runs": rows,
        "groups": summarise(rows, args.group_by) if args.group_by else [],
        "bimodality": bimodality(rows),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, default=str))
    args.output_md.write_text(markdown_report(payload))
    logger.info(f"Wrote {args.output_json} and {args.output_md}")
    print(markdown_report(payload))


if __name__ == "__main__":
    main()
