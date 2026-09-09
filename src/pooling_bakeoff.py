"""Which patch representation is right? Measure it, don't argue it.

A So2Sat patch is 320 m of ground — roughly 33x33 px at 10 m — and the LCZ label
describes the *whole* neighbourhood's built form. Every pooled model in this repo
collapses that to a channel mean (``gap``). This script asks whether that loses
anything, by scoring the recipes in ``utils.pooling_features`` against each other
on the same patches, the same splits and the same subsample.

The recipes divide into three kinds, and the distinction is the point:

* ``center`` — one focal pixel. The "one pixel" representation, ~1/1100 of the
  area the label describes. Included as the low end of the scale.
* ``gap`` / ``mean_std`` / ``quantile`` — order statistics. All are
  **permutation-invariant**: shuffle the patch's pixels and the feature is
  bit-identical. They describe the distribution of pixel embeddings, so they can
  capture heterogeneity but never layout.
* ``ring`` / ``rich`` — include centre-vs-surround means, which are **not**
  permutation-invariant. If within-patch arrangement carries signal, this is the
  family that can see it.

**Criterion.** Linear-probe kappa is primary; cosine-kNN kappa is secondary, for
continuity with the existing baselines. Both are reported at native
dimensionality *and* after PCA to a common width, because a 1024-d feature that
beats a 128-d one may simply have won on capacity.

The silhouette / kNN-agreement diagnostics are computed and written, but they are
labelled **descriptive, non-decisional**: this project has already recorded a case
where they invert the downstream ranking (Seamless has the best LCZ silhouette of
the three embeddings and ranks last in accuracy).

**Held-out cities come for free.** The So2Sat global split is city-disjoint — the
only overlap between train and val/test cities is Guangzhou, which is the
documented GUPPD Guangzhou/Shenzhen merge artefact rather than a leak. So val and
test kappa already measure cross-city generalisation, the failure mode that
actually matters here, and no separate leave-one-city-out fold is needed.

Example:
    python src/pooling_bakeoff.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --embedding tesserav2:GeoTessera_v2 \\
        --embedding alpha_earth_coop:AlphaEarthCoop \\
        --year 2017 --recipes gap mean_std center quantile ring rich \\
        --masked both --common-patches --workers 16 \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/embedding_viz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Cap BLAS before numpy/sklearn import: OpenBLAS on this 256-core host is built
# for <=128 threads and aborts with "tried to allocate too many memory regions".
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import cohen_kappa_score, f1_score  # noqa: E402

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from datasets.registry import EMBEDDING_REGISTRY, get_nodata_predicate  # noqa: E402
from datasets.so2sat import build_so2sat_items, patch_key  # noqa: E402
from knn_baseline import normalize, run_knn  # noqa: E402
from utils import embedding_metrics as em  # noqa: E402
from utils.constants import lcz_dict  # noqa: E402
from utils.geo_lookup import CITY_TO_CONTINENT  # noqa: E402
from utils.pooling_features import (  # noqa: E402
    POOLING_RECIPES,
    cap_blas_threads,
    compose_features,
    extract_blocks_and_cache,
    masked_cache_key,
    recipe_blocks,
)

SPLITS = ("train", "val", "test")


def _parse_embedding(spec: str) -> tuple[str, str]:
    """``registry_name:output_folder`` -> the pair, folder defaulting sensibly."""
    if ":" in spec:
        name, folder = spec.split(":", 1)
    else:
        name, folder = spec, spec
    if name not in EMBEDDING_REGISTRY:
        raise ValueError(f"unknown embedding {name!r}; known: {sorted(EMBEDDING_REGISTRY)}")
    return name, folder


def ordered_meta(items: list) -> pd.DataFrame:
    """Per-row metadata in feature-cache order (train, then val, then test).

    ``extract_blocks_and_cache`` groups by split in this order and preserves item
    order within each split, so replicating it here keeps metadata aligned to the
    feature rows without another pass over the data.
    """
    rows = []
    for s in SPLITS:
        for it in items:
            if it.split != s:
                continue
            ds, pid = patch_key(Path(it.path))
            rows.append((ds, pid, s, int(it.label), it.city))
    meta = pd.DataFrame(rows, columns=["dataset", "patch_id", "split", "label", "city"])
    meta["uid"] = meta["dataset"] + "/" + meta["patch_id"]
    meta["LCZ_class"] = meta["label"] + 1
    meta["lcz_name"] = meta["LCZ_class"].map(
        lambda c: f"LCZ {int(c)}: {lcz_dict[int(c)]['name']}")
    meta["continent"] = meta["city"].map(CITY_TO_CONTINENT)
    return meta


def _stratified_subsample(y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Class-stratified index subsample, capped at ``cap`` rows total."""
    if cap <= 0 or len(y) <= cap:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    # Proportional allocation, but never fewer than a handful of a rare class.
    share = np.maximum((counts / counts.sum() * cap).astype(int), np.minimum(counts, 25))
    keep = []
    for c, n in zip(classes, share):
        idx = np.flatnonzero(y == c)
        keep.append(rng.choice(idx, min(n, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    ok = y_pred >= 0
    return {
        "kappa": float(cohen_kappa_score(y_true[ok], y_pred[ok])),
        "macro_f1": float(f1_score(y_true[ok], y_pred[ok], average="macro")),
        "oa": float((y_true[ok] == y_pred[ok]).mean()),
    }


def evaluate_cell(
    X: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    *,
    fit_idx: np.ndarray,
    knn_k: int,
    knn_ref_max: int,
    knn_jobs: int,
    probe_max_iter: int,
    seed: int,
) -> list[dict]:
    """Linear probe + cosine kNN on one feature set, scored on val and test."""
    mean, std = X["train"][fit_idx].mean(axis=0), X["train"][fit_idx].std(axis=0)
    Xtr = normalize(X["train"][fit_idx], mean, std)
    ytr = y["train"][fit_idx]

    # kNN cost is linear in the reference count and it saturates quickly, so a
    # capped reference set buys most of the signal for a fraction of the time.
    rng = np.random.default_rng(seed)
    ref = (rng.choice(len(Xtr), knn_ref_max, replace=False)
           if knn_ref_max and len(Xtr) > knn_ref_max else np.arange(len(Xtr)))

    out = []
    # n_jobs was dropped from LogisticRegression in sklearn 1.8.
    probe = LogisticRegression(max_iter=probe_max_iter, C=1.0,
                               random_state=seed).fit(Xtr, ytr)
    for split in ("val", "test"):
        if not len(y[split]):
            continue
        Xs = normalize(X[split], mean, std)
        for metric, value in _scores(y[split], probe.predict(Xs)).items():
            out.append({"classifier": "linear_probe", "split": split,
                        "metric": metric, "value": value})
        knn_pred = run_knn(Xtr[ref], ytr[ref], Xs, k=knn_k, n_jobs=knn_jobs)
        for metric, value in _scores(y[split], knn_pred).items():
            out.append({"classifier": "knn", "split": split,
                        "metric": metric, "value": value})
    return out


def _flush(rows: list[dict], out: Path) -> None:
    """Append-safe incremental write, so partial runs are still usable."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    rows.clear()


def run_embedding(args, name: str, folder: str, keep_uids: set | None) -> pd.DataFrame:
    """Extract blocks once, then score every recipe x masking variant."""
    from utils.runtime import resolve_dequantize

    dequantize_fn, _ = resolve_dequantize(name, force=args.dequantize)
    items, _ = build_so2sat_items(
        args.so2sat_dir, folder, args.year, global_split=True,
        global_gpkg=args.global_gpkg, label_col=args.label_col,
    )
    if keep_uids is not None:
        before = len(items)
        items = [it for it in items
                 if "/".join(patch_key(Path(it.path))) in keep_uids]
        logger.info(f"{name}: restricted to common patches, {before} -> {len(items)}")

    meta = ordered_meta(items)
    logger.info(f"{name}: {len(meta)} patches, "
                f"{meta.groupby('split').size().to_dict()}")

    needed = sorted({b for r in args.recipes for b in recipe_blocks(r)})
    base_key = f"global_{folder}"
    rows: list[dict] = []
    written: list[dict] = []

    for masked in args.masked_variants:
        key = masked_cache_key(base_key, masked)
        logger.info(f"── {name} | masked={masked} | blocks={needed}")
        extract_blocks_and_cache(
            items, needed, dequantize_fn, args.cache_dir, key,
            nodata_predicate=get_nodata_predicate(name) if masked else None,
            workers=args.workers, no_cache=args.no_cache, splits=SPLITS,
        )

        y = {s: meta.loc[meta["split"] == s, "label"].to_numpy() for s in SPLITS}
        fit_idx = _stratified_subsample(y["train"], args.fit_max, args.seed)
        logger.info(f"   fit subsample: {len(fit_idx)} of {len(y['train'])} train rows")

        for recipe in args.recipes:
            X = {s: compose_features(args.cache_dir, key, recipe, s) for s in SPLITS}
            dim = X["train"].shape[1]
            logger.info(f"   {recipe}: dim={dim}")

            variants = {"native": X}
            if args.pca_dim and dim > args.pca_dim:
                # Common-width control: if a wide recipe only wins at native
                # dimensionality, it won on capacity rather than information.
                sub = X["train"][fit_idx]
                pca = PCA(n_components=args.pca_dim, random_state=args.seed).fit(sub)
                variants[f"pca{args.pca_dim}"] = {s: pca.transform(X[s]) for s in SPLITS}

            for space, Xv in variants.items():
                cells = evaluate_cell(
                    Xv, y, fit_idx=fit_idx, knn_k=args.knn_k,
                    knn_ref_max=args.knn_ref_max, knn_jobs=args.workers,
                    probe_max_iter=args.probe_max_iter, seed=args.seed,
                )
                for c in cells:
                    written.append({
                        "embedding": name, "output_name": folder, "pooling": recipe,
                        "masked": masked, "feature_dim": Xv["train"].shape[1],
                        "space": space, "decisional": True, **c,
                    })
                    rows.append(written[-1])
                    logger.info(
                        f"      {space:8s} {c['classifier']:12s} {c['split']:4s} "
                        f"{c['metric']:8s} {c['value']:.4f}")

            _flush(written, args.output_dir / args.out_csv)

            if not args.no_diagnostics:
                # Descriptive only. Recorded because they are cheap and
                # interesting, NOT because they decide anything.
                idx = _stratified_subsample(y["train"], args.metric_cap, args.seed)
                sep = em.separability(
                    X["train"][idx],
                    meta.loc[meta["split"] == "train"].iloc[idx].reset_index(drop=True),
                    cap=args.metric_cap, k=args.metric_k, seed=args.seed,
                )
                for _, r in sep.iterrows():
                    for metric in ("silhouette", "knn_agreement"):
                        rows.append({
                            "embedding": name, "output_name": folder, "pooling": recipe,
                            "masked": masked, "feature_dim": dim, "space": "native",
                            "decisional": False, "classifier": f"separability_{r['facet']}",
                            "split": "train", "metric": metric, "value": float(r[metric]),
                        })

    _flush(written, args.output_dir / args.out_csv)
    return pd.DataFrame(rows)


def common_uids(args, pairs: list[tuple[str, str]]) -> set:
    """Patch uids present in every embedding, so the cells are strictly paired."""
    sets = []
    for _, folder in pairs:
        items, _ = build_so2sat_items(
            args.so2sat_dir, folder, args.year, global_split=True,
            global_gpkg=args.global_gpkg, label_col=args.label_col,
        )
        sets.append({"/".join(patch_key(Path(it.path))) for it in items})
        logger.info(f"{folder}: {len(sets[-1])} patches")
    common = set.intersection(*sets)
    logger.info(f"common to all {len(pairs)} embeddings: {len(common)}")
    return common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("Data")
    g.add_argument("--so2sat-dir", required=True, type=Path)
    g.add_argument("--embedding", dest="embeddings", action="append", required=True,
                   metavar="NAME[:FOLDER]",
                   help="Registry name, optionally with its extraction folder "
                        "(e.g. tesserav2:GeoTessera_v2). Repeatable.")
    g.add_argument("--year", required=True)
    g.add_argument("--label-col", default="LCZ_class")
    g.add_argument("--global-gpkg", type=Path, default=None)
    g.add_argument("--dequantize", action="store_true")
    g.add_argument("--common-patches", action="store_true",
                   help="Restrict every embedding to the patches all of them have, "
                        "so the comparison is strictly paired.")

    g = p.add_argument_group("Features")
    g.add_argument("--recipes", nargs="+", default=["gap", "mean_std", "center",
                                                    "quantile", "ring", "rich"],
                   choices=sorted(POOLING_RECIPES))
    g.add_argument("--masked", choices=["off", "on", "both"], default="both",
                   help="Nodata masking variant(s) to score. For alpha_earth_coop the "
                        "-128 sentinel dequantizes to L2 norm 8.06 against a normal "
                        "pixel's 1.0, so this is not a cosmetic switch.")
    g.add_argument("--cache-dir", type=Path, default=None)
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--workers", type=int, default=16)

    g = p.add_argument_group("Scoring")
    g.add_argument("--fit-max", type=int, default=80_000,
                   help="Class-stratified train subsample shared by every cell, so "
                        "recipes are compared on identical rows.")
    g.add_argument("--pca-dim", type=int, default=128,
                   help="Common-width control. 0 disables.")
    g.add_argument("--knn-k", type=int, default=20)
    g.add_argument("--knn-ref-max", type=int, default=40_000,
                   help="Cap on kNN reference rows. Cost is linear in this and the "
                        "signal saturates early.")
    g.add_argument("--probe-max-iter", type=int, default=300,
                   help="lbfgs converges in ~250 here; 1000 just burns time.")
    g.add_argument("--metric-cap", type=int, default=20_000)
    g.add_argument("--metric-k", type=int, default=20)
    g.add_argument("--no-diagnostics", action="store_true")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", required=True, type=Path)
    g.add_argument("--out-csv", default="pooling_bakeoff.csv")
    g.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.cache_dir or (args.output_dir / "cache")
    args.masked_variants = {"off": [False], "on": [True], "both": [False, True]}[args.masked]
    cap_blas_threads(16)

    pairs = [_parse_embedding(s) for s in args.embeddings]
    keep = common_uids(args, pairs) if args.common_patches and len(pairs) > 1 else None

    frames = [run_embedding(args, name, folder, keep) for name, folder in pairs]
    df = pd.concat(frames, ignore_index=True)

    out = args.output_dir / args.out_csv
    # Rewrite once at the end so the file is a clean single-header CSV even
    # though it was appended to along the way.
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df)} rows → {out}")

    # The headline table: primary criterion only.
    head = df[(df["classifier"] == "linear_probe") & (df["metric"] == "kappa")
              & (df["split"] == "test") & (df["space"] == "native")]
    if not head.empty:
        logger.info("\nLinear-probe test kappa (native dim):\n"
                    + head.pivot_table(index=["embedding", "masked"],
                                       columns="pooling", values="value").to_string())


if __name__ == "__main__":
    main()
