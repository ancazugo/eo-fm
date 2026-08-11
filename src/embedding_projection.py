"""Project pooled So2Sat patch embeddings to 2-D (PCA / UMAP / t-SNE) and diagnose.

Pools every patch embedding to a vector (GAP or mean+std — reusing the
``knn_baseline`` feature cache), then StandardScaler → PCA(~50) → UMAP (full) and
t-SNE (balanced subsample). Saves a parquet of 2-D coords + metadata plus static
PNG and interactive Plotly HTML scatters coloured by LCZ class, city, country,
continent and train/val/test split. Also runs separability / train-test-shift /
entanglement diagnostics (utils.embedding_metrics).

The motivating question: do these embeddings cluster by *LCZ class* (good) or by
*geography* (explains weak cross-city generalisation)? — and what does the
train↔test distribution shift imply for the next training step on the so2sat
global split.

Example (Tessera v1.1 global, full ~400k):
    python src/embedding_projection.py \\
        --so2sat-dir /maps/acz25/phd-thesis-data/input/So2Sat-LCZ42/v4 \\
        --global-split \\
        --output-name GeoTessera_v1.1_global --year 2017 \\
        --embedding-name tesserav1.1_global \\
        --pooling gap --methods umap tsne \\
        --output-dir /maps/acz25/phd-thesis-data/output/lcz-classification/embedding_viz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Cap BLAS / numba thread pools BEFORE numpy / sklearn / numba import. On
# many-core hosts (this one has 256) OpenBLAS (built for ≤128 threads) aborts
# with "tried to allocate too many memory regions" during t-SNE / UMAP.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from datasets.registry import EMBEDDING_REGISTRY
from datasets.so2sat import build_so2sat_items
from knn_baseline import extract_and_cache
from utils.constants import lcz_dict
from utils.geo_lookup import CITY_TO_CONTINENT, CITY_TO_COUNTRY, assign_city
from utils.runtime import init_run, resolve_dequantize
from utils import embedding_metrics as em

_SPLITS = ("train", "val", "test")
_DATASET_TO_SPLIT = {"training": "train", "validation": "val", "testing": "test"}
FACETS = ("lcz_name", "city", "country", "continent", "split")


# ── Metadata (aligned to the cached feature-row order) ─────────────────────────

def build_metadata(all_items: list[tuple], gpkg: Path, bounds_csv: Path) -> pd.DataFrame:
    """Build a per-row metadata frame matching the order of the feature cache.

    ``extract_and_cache`` concatenates splits in the order ('train','val','test'),
    preserving the original item order within each split (knn_baseline.py:138-147).
    We replicate that grouping, then join the global GPKG (by dataset + patch_id)
    for geometry centroids, and derive city / country / continent.
    """
    # Replicate extract_and_cache's split grouping + ordering.
    grouped: dict[str, list[tuple]] = {s: [] for s in _SPLITS}
    for it in all_items:
        path, label, sp = it.path, it.label, it.split
        if sp in grouped:
            grouped[sp].append((path, label))

    datasets, patch_ids, splits, labels = [], [], [], []
    for s in _SPLITS:
        for path, label in grouped[s]:
            path = Path(path)
            datasets.append(path.parents[2].name)          # training/validation/testing
            stem = path.stem
            patch_ids.append(stem[len("patch_"):] if stem.startswith("patch_") else stem)
            splits.append(s)
            labels.append(int(label))
    ordered = pd.DataFrame({
        "dataset": datasets, "patch_id": patch_ids, "split": splits, "label": labels,
    })

    # GPKG centroid lookup (keyed by dataset + patch_id).
    gdf = gpd.read_file(gpkg)
    cent = gdf.geometry.centroid
    glook = pd.DataFrame({
        "dataset": gdf["dataset"].astype(str),
        "patch_id": gdf["patch_id"].astype(str),
        "_cx": cent.x.to_numpy(),
        "_cy": cent.y.to_numpy(),
    })
    meta = ordered.merge(glook, on=["dataset", "patch_id"], how="left", sort=False)

    n_missing = int(meta["_cx"].isna().sum())
    if n_missing:
        logger.warning(f"{n_missing} rows had no GPKG centroid match")

    coords = meta[["_cx", "_cy"]].to_numpy()
    meta["city"] = assign_city(coords, bounds_csv)
    meta["country"] = meta["city"].map(CITY_TO_COUNTRY)
    meta["continent"] = meta["city"].map(CITY_TO_CONTINENT)
    meta["LCZ_class"] = meta["label"] + 1
    meta["lcz_name"] = meta["LCZ_class"].map(
        lambda c: f"LCZ {int(c)}: {lcz_dict[int(c)]['name']}"
    )
    return meta


# ── Plotting ──────────────────────────────────────────────────────────────────

def _palette(categories: list[str], facet: str) -> dict[str, str]:
    if facet == "lcz_name":
        return {f"LCZ {k}: {v['name']}": v["color"] for k, v in lcz_dict.items()}
    cmap = plt.get_cmap("tab20" if len(categories) <= 20 else "hsv")
    return {c: matplotlib.colors.to_hex(cmap(i / max(len(categories), 1)))
            for i, c in enumerate(categories)}


def _scatter_ax(ax, df: pd.DataFrame, x: str, y: str, facet: str,
                *, s: float = 2.0, legend: bool = True) -> None:
    """Draw a categorical scatter onto ``ax``; legend only for low-cardinality facets."""
    cats = sorted(df[facet].dropna().astype(str).unique())
    colors = _palette(cats, facet)
    fvals = df[facet].astype(str)
    for c in cats:
        m = fvals == c
        ax.scatter(df.loc[m, x], df.loc[m, y], s=s, alpha=0.5,
                   color=colors.get(c, "#888888"), label=c, linewidths=0)
    if legend and len(cats) <= 18:
        ax.legend(markerscale=4, fontsize=6, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
    ax.set_xlabel(x, fontsize=8)
    ax.set_ylabel(y, fontsize=8)


def save_png(df: pd.DataFrame, x: str, y: str, facet: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_ax(ax, df, x, y, facet)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_panel(df: pd.DataFrame, x: str, y: str, facets: tuple[str, ...],
               title: str, path: Path) -> None:
    """One overview figure: the same projection coloured by each facet, side by side."""
    n = len(facets)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, facet in zip(axes, facets):
        _scatter_ax(ax, df, x, y, facet, s=1.5, legend=True)
        nc = df[facet].nunique()
        ax.set_title(f"by {facet} ({nc})", fontsize=10)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Diagnostic plots ──────────────────────────────────────────────────────────

def plot_separability(sep_df: pd.DataFrame, path: Path) -> None:
    """Grouped bars: silhouette + kNN-label-agreement per facet."""
    d = sep_df.set_index("facet")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    d["silhouette"].plot.bar(ax=a1, color="#4c72b0")
    a1.axhline(0, color="grey", lw=0.8)
    a1.set(title="Silhouette (cosine) — higher = clusters by facet", ylabel="silhouette")
    d["knn_agreement"].plot.bar(ax=a2, color="#dd8452")
    a2.set(title="kNN label agreement", ylabel="fraction shared")
    for ax in (a1, a2):
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density(shift: dict, path: Path) -> None:
    """Overlaid histograms: train↔train baseline vs test→train distances + threshold."""
    tr = shift["train_density"]["mean_cosine_dist_to_train"].to_numpy()
    te = shift["test_density"]["mean_cosine_dist_to_train"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(tr.max(), te.max()), 60)
    ax.hist(tr, bins=bins, density=True, alpha=0.6, label="train→train (baseline)", color="#4c72b0")
    ax.hist(te, bins=bins, density=True, alpha=0.6, label="test→train", color="#c44e52")
    ax.axvline(shift["density_threshold"], ls="--", color="black",
               label=f"95th-pct train ({shift['density_threshold']:.3f})")
    ax.set(title=(f"Local train-density — domain AUC {shift['domain_auc']:.3f}, "
                  f"{shift['frac_test_in_sparse_regions']:.0%} of test in sparse regions"),
           xlabel="mean cosine distance to k nearest train points", ylabel="density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_drift(drift_df: pd.DataFrame, label_col: str, path: Path) -> None:
    """Horizontal bars of per-class train↔test centroid distance."""
    d = drift_df.sort_values("centroid_dist")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(d))))
    ax.barh(d[label_col].astype(str), d["centroid_dist"], color="#55a868")
    ax.set(title="Per-class train↔test centroid drift (PCA space)",
           xlabel="L2 distance between train & test centroids")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entanglement(mat: pd.DataFrame, title: str, path: Path) -> None:
    """Heatmap of the row-normalised kNN cross-neighbour matrix."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(mat, cmap="magma", annot=False, square=True, ax=ax,
                cbar_kws={"label": "fraction of kNN in group"})
    ax.set(title=title, xlabel="neighbour group", ylabel="point group")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_html(df: pd.DataFrame, x: str, y: str, facet: str, title: str, path: Path) -> None:
    import plotly.express as px

    cats = sorted(df[facet].dropna().astype(str).unique())
    cmap = _palette(cats, facet)
    fig = px.scatter(
        df, x=x, y=y, color=df[facet].astype(str), title=title,
        color_discrete_map=cmap, opacity=0.6,
        hover_data=["patch_id", "city", "lcz_name", "split"],
        width=1000, height=750,
    )
    fig.update_traces(marker=dict(size=3))
    fig.write_html(path, include_plotlyjs="cdn")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA/UMAP/t-SNE projection + diagnostics for pooled So2Sat embeddings."
    )

    g = parser.add_argument_group("Data")
    g.add_argument("--so2sat-dir", required=True, type=Path)
    g.add_argument("--global-split", action="store_true",
                   help="Use patches_reference_rxr.gpkg global split (recommended for this tool).")
    g.add_argument("--global-gpkg", type=Path, default=None,
                   help="Global GPKG (default: {so2sat_dir}/patches_reference_rxr.gpkg).")
    g.add_argument("--cities-dir", type=Path, default=None)
    g.add_argument("--cities", nargs="+", default=None)
    g.add_argument("--output-name", required=True,
                   help="Embedding subfolder name (e.g. GeoTessera_v1.1_global).")
    g.add_argument("--year", required=True)
    g.add_argument("--label-col", default="LCZ_class")
    g.add_argument("--embedding-name", required=True, choices=sorted(EMBEDDING_REGISTRY),
                   help="Embedding type — controls auto-dequantization.")
    g.add_argument("--dequantize", action="store_true",
                   help="Force dequantize (auto for alpha_earth_coop and seamless).")
    g.add_argument("--bounds-csv", type=Path,
                   default=_src.parent / "data" / "so2sat_guppd_bounds.csv",
                   help="GUPPD city bounds CSV for nearest-centroid city assignment.")

    g = parser.add_argument_group("Features")
    g.add_argument("--pooling", choices=["gap", "mean_std"], default="gap")
    g.add_argument("--cache-dir", type=Path, default=None,
                   help="Feature cache directory (default: {output_dir}/cache).")
    g.add_argument("--no-cache", action="store_true")

    g = parser.add_argument_group("Projection")
    g.add_argument("--methods", nargs="+", default=["umap"],
                   choices=["pca", "umap", "tsne"])
    g.add_argument("--pca-dim", type=int, default=50)
    g.add_argument("--umap-n-neighbors", type=int, default=30)
    g.add_argument("--umap-min-dist", type=float, default=0.1)
    g.add_argument("--tsne-perplexity", type=float, default=40.0)
    g.add_argument("--tsne-max-per-class", type=int, default=2000,
                   help="Balanced subsample size per LCZ class for t-SNE (it doesn't scale).")
    g.add_argument("--plot-max-points", type=int, default=80_000,
                   help="Random subsample for rendering only (parquet keeps all rows).")

    g = parser.add_argument_group("Diagnostics")
    g.add_argument("--no-diagnostics", action="store_true")
    g.add_argument("--metric-cap", type=int, default=20_000,
                   help="Subsample cap for silhouette / kNN-graph metrics.")
    g.add_argument("--metric-k", type=int, default=20)

    g = parser.add_argument_group("Output")
    g.add_argument("--output-dir", required=True, type=Path)
    g.add_argument("--run-name", default=None)
    g.add_argument("--wandb-project", default="lcz-classification-dl")
    g.add_argument("--wandb-entity", default="phd-thesis-team")
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dequantize_fn, _ = resolve_dequantize(args.embedding_name, force=args.dequantize)

    all_items, _ = build_so2sat_items(
        args.so2sat_dir, args.output_name, args.year,
        global_split=args.global_split, global_gpkg=args.global_gpkg,
        cities_dir=args.cities_dir, cities=args.cities, label_col=args.label_col,
    )
    split_counts = {s: sum(1 for _, _, sp in all_items if sp == s) for s in _SPLITS}
    logger.info(f"Total patches: {len(all_items)}  splits: {split_counts}")

    # ── Features (reuse knn_baseline cache) ───────────────────────────────────
    run_label = "global" if args.global_split else "_".join(
        sorted({sp for _, _, sp in all_items}))
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    cache_key = f"{run_label}_{args.output_name}_{args.pooling}"
    feats, _, _, _ = extract_and_cache(
        all_items, args.pooling, dequantize_fn, cache_dir, cache_key, no_cache=args.no_cache,
    )
    present = [s for s in _SPLITS if feats[s].shape[0] > 0]
    X = np.concatenate([feats[s] for s in present], axis=0)
    logger.info(f"Feature matrix: {X.shape}")

    # ── Metadata aligned to X rows ────────────────────────────────────────────
    gpkg = args.global_gpkg or (args.so2sat_dir / "patches_reference_rxr.gpkg")
    meta = build_metadata(all_items, gpkg, args.bounds_csv)
    assert len(meta) == X.shape[0], (
        f"metadata/feature row mismatch: {len(meta)} vs {X.shape[0]}")
    logger.info(
        f"Metadata: {meta['city'].nunique()} cities, "
        f"{meta['country'].nunique()} countries, {meta['continent'].nunique()} continents")

    # ── Reduce: StandardScaler → PCA → UMAP / t-SNE ───────────────────────────
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(args.pca_dim, Xs.shape[1], Xs.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=args.seed)
    X_pca = pca.fit_transform(Xs)
    cumvar = float(np.cumsum(pca.explained_variance_ratio_)[-1] * 100)
    logger.info(f"PCA → {n_comp} comps; cumulative variance {cumvar:.1f}%")
    meta["pca_x"], meta["pca_y"] = X_pca[:, 0], X_pca[:, 1]

    if "umap" in args.methods:
        import umap
        logger.info("Running UMAP on full set …")
        reducer = umap.UMAP(n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
                            n_components=2, random_state=args.seed, verbose=True)
        XY = reducer.fit_transform(X_pca)
        meta["umap_x"], meta["umap_y"] = XY[:, 0], XY[:, 1]

    if "tsne" in args.methods:
        rng = np.random.default_rng(args.seed)
        keep = []
        for _, grp in meta.groupby("LCZ_class"):
            idx = grp.index.to_numpy()
            keep.append(rng.choice(idx, args.tsne_max_per_class, replace=False)
                        if len(idx) > args.tsne_max_per_class else idx)
        keep = np.sort(np.concatenate(keep))
        logger.info(f"Running t-SNE on balanced subsample ({len(keep)} pts) …")
        tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity,
                    random_state=args.seed, verbose=1)
        XY = tsne.fit_transform(X_pca[keep])
        meta["tsne_x"] = np.nan
        meta["tsne_y"] = np.nan
        meta.loc[keep, "tsne_x"] = XY[:, 0]
        meta.loc[keep, "tsne_y"] = XY[:, 1]

    # ── Output dir + persist ──────────────────────────────────────────────────
    run_cfg = dict(task="embedding_projection", embedding=args.output_name,
                   pooling=args.pooling, methods=args.methods, pca_dim=n_comp,
                   n_patches=int(X.shape[0]), split=run_label)
    run_dir = init_run(args.output_dir, run_cfg, args.run_name,
                       default_name=f"proj_{args.output_name}_{run_label}_{args.pooling}",
                       wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                       no_wandb=args.no_wandb)

    parquet_path = run_dir / f"projection_{cache_key}.parquet"
    meta.drop(columns=["_cx", "_cy"]).to_parquet(parquet_path)
    logger.info(f"Saved projection table → {parquet_path}")

    # ── Plots (render on a subsample for speed/size) ──────────────────────────
    if len(meta) > args.plot_max_points:
        rng = np.random.default_rng(args.seed)
        plot_df = meta.iloc[np.sort(rng.choice(len(meta), args.plot_max_points, replace=False))]
    else:
        plot_df = meta
    method_axes = {"pca": ("pca_x", "pca_y"), "umap": ("umap_x", "umap_y"),
                   "tsne": ("tsne_x", "tsne_y")}
    for method in args.methods:
        x, y = method_axes[method]
        if x not in meta.columns:
            continue
        sub = plot_df.dropna(subset=[x, y])
        for facet in FACETS:
            title = f"{method.upper()} — {args.output_name} — by {facet}"
            png = run_dir / f"{method}_{facet}.png"
            html = run_dir / f"{method}_{facet}.html"
            save_png(sub, x, y, facet, title, png)
            save_html(sub, x, y, facet, title, html)
            if not args.no_wandb:
                import wandb
                wandb.log({f"{method}/{facet}": wandb.Image(str(png))})
        # Overview panel: one figure, all facets side by side.
        panel = run_dir / f"{method}_panel.png"
        save_panel(sub, x, y, FACETS, f"{method.upper()} — {args.output_name}", panel)
        if not args.no_wandb:
            import wandb
            wandb.log({f"{method}/panel": wandb.Image(str(panel))})
    logger.info(f"Saved scatter PNG/HTML + overview panel for {args.methods} × {len(FACETS)} facets")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if not args.no_diagnostics:
        sep = em.separability(X_pca, meta, cap=args.metric_cap, k=args.metric_k, seed=args.seed)
        sep.to_csv(run_dir / "separability.csv", index=False)
        plot_separability(sep, run_dir / "diag_separability.png")

        shift = em.train_test_shift(X_pca, meta, k=args.metric_k,
                                    cap=max(args.metric_cap, 30_000), seed=args.seed)
        shift["centroid_drift"].to_csv(run_dir / "shift_centroid_drift.csv", index=False)
        plot_density(shift, run_dir / "diag_train_test_density.png")
        plot_centroid_drift(shift["centroid_drift"], "lcz_name",
                            run_dir / "diag_centroid_drift.png")

        ent = {f: em.entanglement(X_pca, meta, facet=f, cap=args.metric_cap,
                                  k=args.metric_k, seed=args.seed)
               for f in ("lcz_name", "country")}
        for f, df in ent.items():
            df.to_csv(run_dir / f"entanglement_{f}.csv", index=False)
        ent_mat = em.entanglement_matrix(X_pca, meta, facet="lcz_name",
                                         cap=args.metric_cap, k=args.metric_k, seed=args.seed)
        ent_mat.to_csv(run_dir / "entanglement_lcz_matrix.csv")
        plot_entanglement(ent_mat, f"LCZ kNN entanglement — {args.output_name}",
                          run_dir / "diag_entanglement_lcz.png")

        if not args.no_wandb:
            import wandb
            wandb.log({
                "diag/domain_auc": shift["domain_auc"],
                "diag/frac_test_in_sparse_regions": shift["frac_test_in_sparse_regions"],
                "diag/separability": wandb.Table(dataframe=sep),
                "diag/centroid_drift": wandb.Table(dataframe=shift["centroid_drift"]),
                "diag/separability_plot": wandb.Image(str(run_dir / "diag_separability.png")),
                "diag/density_plot": wandb.Image(str(run_dir / "diag_train_test_density.png")),
                "diag/centroid_drift_plot": wandb.Image(str(run_dir / "diag_centroid_drift.png")),
                "diag/entanglement_plot": wandb.Image(str(run_dir / "diag_entanglement_lcz.png")),
            })

    if not args.no_wandb:
        import wandb
        if wandb.run:
            wandb.finish()
    logger.info(f"Done → {run_dir}")


if __name__ == "__main__":
    main()
