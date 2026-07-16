"""Quantitative diagnostics for pooled patch embeddings.

Operates on a PCA-reduced feature matrix ``X (N, D)`` plus a metadata DataFrame
(columns: ``split``, ``LCZ_class``, ``lcz_name``, ``city``, ``country``,
``continent``). Three families of diagnostics, each returning a tidy DataFrame so
the projection script and the notebook can share them:

  * separability    — how well the embedding clusters by each facet
  * train_test_shift — distribution shift between the so2sat train and test splits
  * entanglement     — which classes / regions bleed into each other in feature space

Metrics that scale poorly (silhouette, kNN graphs) run on a capped random
subsample; pass a generous ``cap`` for the final report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

FACETS = ("lcz_name", "city", "country", "continent")


def _subsample(n: int, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= cap:
        return np.arange(n)
    return rng.choice(n, cap, replace=False)


def _knn_label_agreement(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Mean fraction of each point's k nearest neighbours sharing its label."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=4)
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]                       # drop self
    neigh_labels = labels[idx]             # (N, k)
    same = (neigh_labels == labels[:, None]).mean(axis=1)
    return float(same.mean())


# ── Separability ──────────────────────────────────────────────────────────────

def separability(
    X: np.ndarray,
    meta: pd.DataFrame,
    facets: tuple[str, ...] = FACETS,
    *,
    cap: int = 20_000,
    k: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Silhouette (cosine) and kNN-label-agreement per facet, on a subsample.

    Higher silhouette / agreement ⇒ the embedding clusters by that facet. If a
    geography facet (city/country/continent) scores above ``lcz_name``, the
    embedding encodes location more than LCZ semantics.
    """
    sel = _subsample(len(X), cap, seed)
    Xs = X[sel]
    rows = []
    for facet in facets:
        codes = pd.Categorical(meta.iloc[sel][facet]).codes
        n_groups = len(np.unique(codes))
        if n_groups < 2:
            continue
        sil = float(silhouette_score(Xs, codes, metric="cosine"))
        agr = _knn_label_agreement(Xs, codes, k)
        rows.append(dict(facet=facet, n_groups=n_groups,
                         silhouette=sil, knn_agreement=agr))
    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    logger.info(f"Separability (cap={len(sel)}, k={k}):\n{df.to_string(index=False)}")
    return df


# ── Train/test distribution shift ─────────────────────────────────────────────

def train_test_shift(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    label_col: str = "lcz_name",
    k: int = 20,
    cap: int = 30_000,
    low_density_pct: float = 95.0,
    seed: int = 42,
) -> dict:
    """Quantify train↔test shift in feature space.

    Returns a dict with:
      * ``centroid_drift``  — per-class train↔test centroid L2 distance (DataFrame)
      * ``test_density``    — DataFrame of per-test-point mean cosine distance to
        its k nearest *train* points, plus the fraction of test points beyond the
        ``low_density_pct`` percentile of train↔train distances (sparse regions)
      * ``domain_auc``      — 5-fold logistic-regression AUC separating train vs
        test points; ~0.5 = no shift, →1.0 = strongly separable (shift)
    """
    is_train = (meta["split"] == "train").to_numpy()
    is_test = (meta["split"] == "test").to_numpy()
    Xtr_all, Xte_all = X[is_train], X[is_test]

    # (a) per-class centroid drift
    rows = []
    labels = meta[label_col].to_numpy()
    for cls in pd.unique(labels):
        m = labels == cls
        tr, te = X[m & is_train], X[m & is_test]
        if len(tr) and len(te):
            rows.append(dict(
                **{label_col: cls},
                n_train=len(tr), n_test=len(te),
                centroid_dist=float(np.linalg.norm(tr.mean(0) - te.mean(0))),
            ))
    centroid_drift = (pd.DataFrame(rows)
                      .sort_values("centroid_dist", ascending=False)
                      .reset_index(drop=True))

    # (b) test-point local train-density (cosine distance to k nearest train pts)
    rng = np.random.default_rng(seed)
    tr_idx = (rng.choice(len(Xtr_all), cap, replace=False)
              if len(Xtr_all) > cap else np.arange(len(Xtr_all)))
    Xtr = Xtr_all[tr_idx]
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=4)
    nn.fit(Xtr)
    # train↔train baseline (k+1, drop self) → threshold for "sparse"
    d_tt, _ = nn.kneighbors(Xtr, n_neighbors=k + 1)
    tt_density = d_tt[:, 1:].mean(axis=1)
    thresh = float(np.percentile(tt_density, low_density_pct))
    te_idx = (rng.choice(len(Xte_all), cap, replace=False)
              if len(Xte_all) > cap else np.arange(len(Xte_all)))
    d_te, _ = nn.kneighbors(Xte_all[te_idx])
    te_density = d_te.mean(axis=1)
    frac_sparse = float((te_density > thresh).mean())
    test_density = pd.DataFrame({"mean_cosine_dist_to_train": te_density})

    # (c) domain classifier AUC (train vs test)
    n = min(len(Xtr_all), len(Xte_all), cap)
    Xd = np.vstack([Xtr_all[rng.choice(len(Xtr_all), n, replace=False)],
                    Xte_all[rng.choice(len(Xte_all), n, replace=False)]])
    yd = np.r_[np.zeros(n), np.ones(n)]
    clf = LogisticRegression(max_iter=1000)
    domain_auc = float(cross_val_score(clf, Xd, yd, cv=5, scoring="roc_auc").mean())

    logger.info(
        f"Train/test shift: domain_auc={domain_auc:.3f} "
        f"(0.5=no shift), {frac_sparse:.1%} of test points in sparse "
        f"(>{low_density_pct:.0f}th pct) train regions; "
        f"top class drift:\n{centroid_drift.head(5).to_string(index=False)}"
    )
    return dict(
        centroid_drift=centroid_drift,
        test_density=test_density,
        train_density=pd.DataFrame({"mean_cosine_dist_to_train": tt_density}),
        density_threshold=thresh,
        frac_test_in_sparse_regions=frac_sparse,
        domain_auc=domain_auc,
    )


# ── Entanglement ──────────────────────────────────────────────────────────────

def entanglement_matrix(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    facet: str = "lcz_name",
    cap: int = 20_000,
    k: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Full row-normalised kNN cross-neighbour matrix for a facet.

    Entry [a, b] = fraction of group-a points' k nearest neighbours that belong
    to group b (diagonal kept = self-neighbour fraction). Returned as a
    DataFrame indexed/columned by group name.
    """
    sel = _subsample(len(X), cap, seed)
    Xs = X[sel]
    groups = pd.Categorical(meta.iloc[sel][facet])
    codes = groups.codes
    names = np.array(groups.categories)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=4)
    nn.fit(Xs)
    _, idx = nn.kneighbors(Xs)
    neigh = codes[idx[:, 1:]]              # (N, k)

    n_groups = len(names)
    counts = np.zeros((n_groups, n_groups), dtype=np.int64)
    for a in range(n_groups):
        m = codes == a
        if m.any():
            np.add.at(counts[a], neigh[m].ravel(), 1)
    totals = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float),
                     where=totals > 0)
    return pd.DataFrame(frac, index=names, columns=names)


def entanglement(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    facet: str = "lcz_name",
    cap: int = 20_000,
    k: int = 20,
    top: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Most-entangled ordered group pairs for a facet (off-diagonal of the matrix).

    For each point, look at its k nearest neighbours; count how often a point of
    group A has neighbours of a *different* group B. Returns the ``top`` ordered
    (group_a → group_b) pairs by cross-neighbour fraction — i.e. which LCZ
    classes (or cities/countries) the embedding confuses.
    """
    mat = entanglement_matrix(X, meta, facet=facet, cap=cap, k=k, seed=seed)
    frac = mat.to_numpy().copy()
    names = mat.index.to_numpy()
    np.fill_diagonal(frac, 0.0)           # ignore self-group

    rows = []
    for a in range(len(names)):
        for b in range(len(names)):
            if frac[a, b] > 0:
                rows.append(dict(group_a=names[a], group_b=names[b],
                                 cross_neighbour_frac=float(frac[a, b])))
    df = (pd.DataFrame(rows)
          .sort_values("cross_neighbour_frac", ascending=False)
          .head(top).reset_index(drop=True))
    logger.info(f"Entanglement ({facet}, top {top}):\n{df.to_string(index=False)}")
    return df
