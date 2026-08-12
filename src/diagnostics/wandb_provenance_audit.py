"""Task 1.5.4 — label every historical W&B run with the product it actually used.

Task 1.5.1 established that `tesserav1.1` and `tesserav1.1_global` are different
feature bases, not two samples of one product. Every Tessera number recorded so
far therefore names a family it cannot distinguish, and `RESULTS.md` has no
column that resolves it. This script goes back through the W&B project and
resolves each run.

**No run can have mixed the two.** A run takes one `--output-name` /
`--embedding-name` pair and `infer_roi` reads its embedding from the CLI, so
there is no path by which a single run trained on one product and evaluated on
the other. The audit's job is labelling, not damage assessment.

Provenance is resolved in two ways, and the report says which was used per run:

* `explicit` — the run's config carries `embedding_name` (every run from Task
  1.5.3 onward carries the full triple directly).
* `inferred` — older runs record only `embedding`, the free-text `--output-name`
  label. LEGACY_OUTPUT_NAMES below maps the labels actually used in this project
  onto registry keys. Anything unrecognised is reported as `unresolved` rather
  than guessed.

Read-only: it queries the W&B API and writes a JSON + markdown artefact.

Example:

    python src/diagnostics/wandb_provenance_audit.py \\
        --entity phd-thesis-team --project lcz-classification-dl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.registry import EMBEDDING_REGISTRY, provenance  # noqa: E402

# --output-name labels used in this project, mapped onto registry keys. Only for
# runs predating the `embedding_name` config field; anything not listed here is
# left unresolved rather than guessed, since guessing is exactly the failure
# this task exists to remove.
LEGACY_OUTPUT_NAMES: dict[str, str] = {
    "GeoTessera": "tessera",
    "GeoTessera_v1.1": "tesserav1.1",
    # Six runs from 2026-05-05/06 used this spelling; no directory of that name
    # survives on disk, so the extraction was renamed to GeoTessera_v1.1
    # afterwards. Resolved to the per-city archive on dating rather than on the
    # label: these are per-city grid-split runs, and the global 0.1-degree
    # extraction did not exist yet.
    "GeoTesserav1.1": "tesserav1.1",
    "GeoTessera_v1.1_global": "tesserav1.1_global",
    "GeoTessera_v2": "tesserav2",
    "AlphaEarth": "alpha_earth",
    "AlphaEarthCoop": "alpha_earth_coop",
    "EmbeddedSeamless": "seamless",
    "AuxStruct": "aux_struct",
}

METRICS = ("test_kappa", "test_f1", "test_acc", "test_miou")


def resolve(config: dict) -> tuple[str | None, str]:
    """Return ``(embedding_name, how)`` for one run config."""
    name = config.get("embedding_name")
    if isinstance(name, list):
        name = "+".join(name)
    if name and all(part in EMBEDDING_REGISTRY for part in str(name).split("+")):
        return str(name), "explicit"

    label = config.get("embedding")
    if isinstance(label, str):
        parts = [LEGACY_OUTPUT_NAMES.get(p) for p in label.split("+")]
        if all(parts):
            return "+".join(parts), "inferred"
    return None, "unresolved"


def audit(entity: str, project: str) -> list[dict]:
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=200)
    logger.info(f"{len(runs)} runs in {entity}/{project}")

    out: list[dict] = []
    for run in runs:
        cfg = run.config
        name, how = resolve(cfg)
        row = {
            "run_name": run.name,
            "run_id": run.id,
            "state": run.state,
            "created": str(run.created_at),
            "task": cfg.get("task"),
            "output_name": cfg.get("embedding"),
            "embedding_name": name,
            "resolved_by": how,
            "split_source": cfg.get("split_source"),
            "year": cfg.get("year"),
            "family": cfg.get("family"),
            "preset": cfg.get("preset"),
            "seed": cfg.get("seed"),
            **{m: run.summary.get(m) for m in METRICS},
        }
        if name:
            row.update({k: v for k, v in provenance(name.split("+")).items()
                        if k != "embedding_name"})
        out.append(row)
    return out


def markdown_report(rows: list[dict], meta: dict) -> str:
    by_res = Counter(r["resolved_by"] for r in rows)
    by_emb: Counter = Counter(r["embedding_name"] or "unresolved" for r in rows)

    lines = [
        "### Task 1.5.4 — W&B provenance audit",
        "",
        f"{len(rows)} runs in `{meta['entity']}/{meta['project']}`, read "
        f"{meta['generated']}. Resolution: "
        + ", ".join(f"{k} {v}" for k, v in sorted(by_res.items())) + ".",
        "",
        "| embedding | product | version | source | runs |",
        "|---|---|---|---|---|",
    ]
    seen: dict[str, dict] = {}
    for r in rows:
        key = r["embedding_name"] or "unresolved"
        seen.setdefault(key, r)
    for key, count in by_emb.most_common():
        r = seen[key]
        lines.append(
            f"| `{key}` | {r.get('product', '—')} | {r.get('version', '—')} | "
            f"{r.get('source', '—')} | {count} |"
        )

    tess = [r for r in rows if r.get("product", "").startswith("tessera")]
    lines += [
        "",
        f"#### Tessera runs by product ({len(tess)} runs)",
        "",
        "| source | runs | best test_kappa | best run |",
        "|---|---|---|---|",
    ]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in tess:
        groups[r["source"]].append(r)
    for source, rs in sorted(groups.items()):
        scored = [r for r in rs if isinstance(r.get("test_kappa"), (int, float))]
        best = max(scored, key=lambda r: r["test_kappa"], default=None)
        lines.append(
            f"| `{source}` | {len(rs)} | "
            f"{best['test_kappa']:.4f} | `{best['run_name']}` ({best['run_id']}) |"
            if best else f"| `{source}` | {len(rs)} | — | — |"
        )

    # Split by split_source: the grid-split numbers are autocorrelation-inflated
    # (0.93-0.97) and would otherwise swamp the cultural-split runs that carry
    # every headline claim in the chapter.
    for split, title in (
        ("global_so2sat", "cultural split (`global_so2sat`) — the headline numbers"),
        ("grid", "per-city grid split — autocorrelation-inflated, not comparable"),
    ):
        top = sorted(
            (r for r in tess
             if r.get("split_source") == split
             and isinstance(r.get("test_kappa"), (int, float))),
            key=lambda r: r["test_kappa"], reverse=True,
        )[:10]
        if not top:
            continue
        lines += [
            "",
            f"Top Tessera runs on the {title}:",
            "",
            "| run | id | source | test_kappa | test_f1 | resolved by |",
            "|---|---|---|---|---|---|",
        ]
        for r in top:
            f1 = r.get("test_f1")
            f1_cell = f"{f1:.4f}" if isinstance(f1, (int, float)) else "—"
            lines.append(
                f"| `{r['run_name']}` | {r['run_id']} | `{r['source']}` | "
                f"{r['test_kappa']:.4f} | {f1_cell} | {r['resolved_by']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Task 1.5.4 — W&B provenance audit.")
    p.add_argument("--entity", default="phd-thesis-team")
    p.add_argument("--project", default="lcz-classification-dl")
    p.add_argument("--output-json", type=Path,
                   default=Path("diagnostics/wandb_provenance_audit.json"))
    p.add_argument("--output-md", type=Path,
                   default=Path("diagnostics/wandb_provenance_audit.md"))
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
        os.environ.setdefault("WANDB_API_KEY", os.environ.get("WANDB_KEY", ""))
    except ImportError:                                          # pragma: no cover
        pass

    rows = audit(args.entity, args.project)
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entity": args.entity, "project": args.project, "n_runs": len(rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"meta": meta, "runs": rows}, indent=2))
    args.output_md.write_text(markdown_report(rows, meta))

    unresolved = [r for r in rows if r["resolved_by"] == "unresolved"]
    if unresolved:
        logger.warning(
            f"{len(unresolved)} runs unresolved: "
            + ", ".join(sorted({str(r['output_name']) for r in unresolved}))
        )
    logger.info(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
