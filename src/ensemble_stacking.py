"""Weighted-average and stacked ensembling on cached ensemble_eval probs.

Fits ensemble weights / a stacker on the VAL split probs and evaluates on the
TEST split probs (both produced by ensemble_eval.py --split val/test with the
same --model specs). Reports:
  - equal-weight softmax average (baseline, = ensemble_eval's full-ensemble row)
  - weight grid search on the probability simplex (val-kappa-selected)
  - multinomial logistic-regression stacker on concatenated per-model probs
  - with --temperature-scale: equal-weight + weighted on per-model
    temperature-calibrated probs (T fit on val by NLL)
  - with --city-holdout: leave-one-city-out weighted + stacked-LR — fit on val
    patches of the other 9 cities, evaluate on test patches of the held-out
    city, pool. So2Sat val/test share the same 10 cities, so the plain val-fit
    stacker reads city-conditional structure back off test; LOCO is the honest
    protocol.

Example:
    python src/ensemble_stacking.py \\
        --val-npz  .../ensemble_3models_val/probs.npz \\
        --test-npz .../ensemble_3models_test/probs.npz \\
        --temperature-scale \\
        --city-holdout --global-gpkg ${SO2SAT}/patches_reference_rxr.gpkg
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

_src = Path(__file__).parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from utils.geo_lookup import assign_cities


def load_npz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Returns (model_names, probs (M, N, C), labels (N,), patch_ids (N,))."""
    data = np.load(path, allow_pickle=False)
    names = [k for k in data.files if k not in ("labels", "patch_ids")]
    return names, np.stack([data[n] for n in names]), data["labels"], data["patch_ids"]


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = probs.argmax(axis=1)
    return {
        "acc":   float(accuracy_score(labels, preds)),
        "f1":    float(f1_score(labels, preds, average="macro")),
        "kappa": float(cohen_kappa_score(labels, preds)),
    }


def simplex_grid(n: int, step: float):
    """All weight vectors of length n on the simplex with the given step."""
    ticks = int(round(1.0 / step))
    for cuts in itertools.combinations_with_replacement(range(ticks + 1), n - 1):
        w, prev = [], 0
        for c in sorted(cuts):
            w.append((c - prev) * step)
            prev = c
        w.append((ticks - prev) * step)
        for perm in set(itertools.permutations(w)):
            yield np.array(perm)


def search_weights(pv: np.ndarray, yv: np.ndarray, step: float) -> np.ndarray:
    """Grid search on the simplex, selecting on val kappa."""
    best_w, best_k = None, -1.0
    for w in simplex_grid(pv.shape[0], step):
        k = cohen_kappa_score(yv, np.tensordot(w, pv, axes=1).argmax(axis=1))
        if k > best_k:
            best_k, best_w = k, w
    return best_w


def apply_temperature(probs: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """Per-model temperature on (M, N, C) probs: softmax(log p / T), i.e. p^(1/T) renormalised."""
    q = np.clip(probs, 1e-12, None) ** (1.0 / temps[:, None, None])
    return q / q.sum(axis=2, keepdims=True)


def fit_temperatures(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit one temperature per model by minimising val NLL."""
    temps = []
    for p in probs:
        def nll(log_t: float) -> float:
            q = np.clip(p, 1e-12, None) ** (1.0 / np.exp(log_t))
            q /= q.sum(axis=1, keepdims=True)
            return -np.mean(np.log(q[np.arange(len(labels)), labels]))
        res = minimize_scalar(nll, bounds=(np.log(0.1), np.log(10.0)), method="bounded")
        temps.append(float(np.exp(res.x)))
    return np.array(temps)



def load_aux(paths: list[Path], patch_ids: np.ndarray, split: str) -> tuple[np.ndarray, list[str]]:
    """Aux features aligned to the npz patch order. Returns (N, F) with NaN for misses."""
    import pandas as pd

    dataset = "validation" if split == "val" else "testing"
    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        df = df[df["dataset"] == dataset].drop(columns=["dataset"])
        df["patch_id"] = df["patch_id"].astype(str)
        dfs.append(df.set_index("patch_id"))
    df = pd.concat(dfs, axis=1)
    df = df.reindex(np.asarray(patch_ids, dtype=str))
    n_miss = int(df.isna().all(axis=1).sum())
    if n_miss:
        logger.warning(f"{split}: {n_miss} patches without aux features (median-imputed per fold)")
    return df.to_numpy(dtype=float), list(df.columns)


def impute_scale(X_fit: np.ndarray, *arrays: np.ndarray) -> list[np.ndarray]:
    """Median-impute + standardize using statistics of the fit fold only."""
    med = np.nanmedian(X_fit, axis=0)
    med = np.where(np.isnan(med), 0.0, med)  # column entirely NaN in the fit fold
    out = []
    for X in (X_fit, *arrays):
        Z = np.where(np.isnan(X), med, X)
        out.append(Z)
    mu, sd = out[0].mean(axis=0), out[0].std(axis=0) + 1e-8
    return [(Z - mu) / sd for Z in out]


def log_probs(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-12, None))


def fit_offset_corrector(
    logp: np.ndarray, aux: np.ndarray, y: np.ndarray, lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit logits = logp + A @ aux + b by CE + lam*||A||^2 (logp coefficient frozen
    at 1, so the corrector can only add what the aux features justify)."""
    from scipy.optimize import minimize

    n, c = logp.shape
    f = aux.shape[1]

    def objective(theta: np.ndarray):
        A = theta[: c * f].reshape(c, f)
        b = theta[c * f:]
        logits = logp + aux @ A.T + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(np.clip(p[np.arange(n), y], 1e-12, None)))
        loss += lam * np.sum(A * A)
        g = p.copy()
        g[np.arange(n), y] -= 1.0
        g /= n
        grad_A = g.T @ aux + 2.0 * lam * A
        grad_b = g.sum(axis=0)
        return loss, np.concatenate([grad_A.ravel(), grad_b])

    res = minimize(objective, np.zeros(c * f + c), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    theta = res.x
    return theta[: c * f].reshape(c, f), theta[c * f:]


def offset_corrector_predict(
    logp: np.ndarray, aux: np.ndarray, A: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    return (logp + aux @ A.T + b).argmax(axis=1)


def fit_stacker(Xv: np.ndarray, yv: np.ndarray, select_groups: np.ndarray | None = None) -> LogisticRegression:
    """Fit stacked LR; C selected city-grouped (fit on half the cities, score kappa
    on the other half) when groups are given, else on training kappa."""
    best_C, best_k = None, -1.0
    for C in (0.1, 1.0, 10.0):
        if select_groups is not None:
            cities = np.unique(select_groups)
            half = np.isin(select_groups, cities[: len(cities) // 2])
            lr = LogisticRegression(max_iter=2000, C=C).fit(Xv[half], yv[half])
            k = cohen_kappa_score(yv[~half], lr.predict(Xv[~half]))
        else:
            lr = LogisticRegression(max_iter=2000, C=C).fit(Xv, yv)
            k = cohen_kappa_score(yv, lr.predict(Xv))
        if k > best_k:
            best_k, best_C = k, C
    return LogisticRegression(max_iter=2000, C=best_C).fit(Xv, yv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted/stacked ensembling on cached probs.")
    parser.add_argument("--val-npz", required=True, type=Path)
    parser.add_argument("--test-npz", required=True, type=Path)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--temperature-scale", action="store_true",
                        help="Also report equal-weight/weighted on per-model T-calibrated probs.")
    parser.add_argument("--city-holdout", action="store_true",
                        help="Also report leave-one-city-out weighted + stacked-LR.")
    parser.add_argument("--global-gpkg", type=Path, default=None,
                        help="patches_reference_rxr.gpkg (required with --city-holdout).")
    parser.add_argument("--city-bounds", type=Path, default=Path("data/so2sat_guppd_bounds.csv"))
    parser.add_argument("--aux-parquet", type=Path, nargs="+", default=None,
                        help="Aux feature parquet(s) from extract_aux_features.py / "
                             "extract_osm_features.py; adds LOCO aux-gate results "
                             "(requires --city-holdout).")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.city_holdout and args.global_gpkg is None:
        parser.error("--city-holdout requires --global-gpkg")
    if args.aux_parquet and not args.city_holdout:
        parser.error("--aux-parquet requires --city-holdout (the gate is LOCO-only)")

    names_v, pv, yv, ids_v = load_npz(args.val_npz)
    names_t, pt, yt, ids_t = load_npz(args.test_npz)
    assert names_v == names_t, f"model sets differ: {names_v} vs {names_t}"
    logger.info(f"Models: {names_v}")
    logger.info(f"val N={len(yv)}  test N={len(yt)}")

    results: dict[str, dict] = {}

    # ── Equal-weight baseline ────────────────────────────────────────────────
    results["equal_weight"] = {
        "val":  metrics(pv.mean(axis=0), yv),
        "test": metrics(pt.mean(axis=0), yt),
        "weights": [1 / len(names_v)] * len(names_v),
    }

    # ── Weight grid search (select on val kappa) ─────────────────────────────
    best_w = search_weights(pv, yv, args.weight_step)
    results["weighted"] = {
        "val":  metrics(np.tensordot(best_w, pv, axes=1), yv),
        "test": metrics(np.tensordot(best_w, pt, axes=1), yt),
        "weights": best_w.round(4).tolist(),
    }

    # ── Temperature-calibrated variants (T fit on val by NLL) ────────────────
    if args.temperature_scale:
        temps = fit_temperatures(pv, yv)
        logger.info(f"Fitted temperatures: "
                    + ", ".join(f"{n}={t:.3f}" for n, t in zip(names_v, temps)))
        pv_c, pt_c = apply_temperature(pv, temps), apply_temperature(pt, temps)
        results["equal_weight_calibrated"] = {
            "val":  metrics(pv_c.mean(axis=0), yv),
            "test": metrics(pt_c.mean(axis=0), yt),
            "temperatures": temps.round(4).tolist(),
        }
        w_c = search_weights(pv_c, yv, args.weight_step)
        results["weighted_calibrated"] = {
            "val":  metrics(np.tensordot(w_c, pv_c, axes=1), yv),
            "test": metrics(np.tensordot(w_c, pt_c, axes=1), yt),
            "weights": w_c.round(4).tolist(),
            "temperatures": temps.round(4).tolist(),
        }

    # ── Logistic-regression stacker on concatenated probs ────────────────────
    Xv = np.concatenate(list(pv), axis=1)   # (N, M*C)
    Xt = np.concatenate(list(pt), axis=1)
    best_lr = fit_stacker(Xv, yv)
    results["stacked_lr"] = {
        "val":  metrics(best_lr.predict_proba(Xv), yv),
        "test": metrics(best_lr.predict_proba(Xt), yt),
        "C": best_lr.C,
    }

    # ── Leave-one-city-out (honest: eval city never seen by weight/stacker fit) ──
    if args.city_holdout:
        cities_v = assign_cities(ids_v, "val", args.global_gpkg, args.city_bounds)
        cities_t = assign_cities(ids_t, "test", args.global_gpkg, args.city_bounds)
        cities = sorted(set(cities_v) | set(cities_t))
        logger.info(f"City-holdout over {len(cities)} cities: {cities}")

        aux_v = aux_t = None
        if args.aux_parquet:
            aux_v, aux_names = load_aux(args.aux_parquet, ids_v, "val")
            aux_t, _ = load_aux(args.aux_parquet, ids_t, "test")
            logger.info(f"Aux features ({len(aux_names)}): {aux_names}")
            preds_aux = np.full(len(yt), -1)
            preds_ctrl = np.full(len(yt), -1)
            preds_auxonly = np.full(len(yt), -1)
            preds_offset = np.full(len(yt), -1)

        preds_w = np.full(len(yt), -1)
        preds_s = np.full(len(yt), -1)
        per_city: dict[str, dict] = {}
        for city in cities:
            fit = cities_v != city
            hold = cities_t == city
            if not hold.any():
                continue
            w = search_weights(pv[:, fit], yv[fit], args.weight_step)
            preds_w[hold] = np.tensordot(w, pt[:, hold], axes=1).argmax(axis=1)
            lr = fit_stacker(Xv[fit], yv[fit], select_groups=cities_v[fit])
            preds_s[hold] = lr.predict(Xt[hold])
            per_city[city] = {
                "n_test": int(hold.sum()),
                "weights": w.round(4).tolist(),
                "C": lr.C,
                "kappa_weighted": float(cohen_kappa_score(yt[hold], preds_w[hold])),
                "kappa_stacked": float(cohen_kappa_score(yt[hold], preds_s[hold])),
            }
            if aux_v is not None:
                # aux gate: log weighted-ensemble probs (+/- aux) -> small LR,
                # everything (weights, impute/scale stats, C) fit on the 9 cities
                lp_v = log_probs(np.tensordot(w, pv[:, fit], axes=1))
                lp_t = log_probs(np.tensordot(w, pt[:, hold], axes=1))
                Av, At = impute_scale(aux_v[fit], aux_t[hold])
                grp = cities_v[fit]
                lr_aux = fit_stacker(np.hstack([lp_v, Av]), yv[fit], select_groups=grp)
                preds_aux[hold] = lr_aux.predict(np.hstack([lp_t, At]))
                lr_ctrl = fit_stacker(lp_v, yv[fit], select_groups=grp)
                preds_ctrl[hold] = lr_ctrl.predict(lp_t)
                lr_ao = fit_stacker(Av, yv[fit], select_groups=grp)
                preds_auxonly[hold] = lr_ao.predict(At)
                # offset corrector: probs pathway frozen; lam and a confidence
                # gate (apply the correction only below max-prob tau; tau=1 =
                # ungated) selected city-grouped on the fit cities
                half = np.isin(grp, np.unique(grp)[: len(np.unique(grp)) // 2])
                conf_v = np.exp(lp_v).max(axis=1)
                conf_t = np.exp(lp_t).max(axis=1)
                best_lam, best_tau, best_k = None, None, -1.0
                for lam in (0.0, 0.00001, 0.0001, 0.001):
                    A_, b_ = fit_offset_corrector(lp_v[half], Av[half], yv[fit][half], lam)
                    corr = Av[~half] @ A_.T + b_
                    for tau in (0.15, 0.2, 0.3, 0.4, 0.5):
                        logits = lp_v[~half] + np.where(conf_v[~half, None] < tau, corr, 0.0)
                        k = cohen_kappa_score(yv[fit][~half], logits.argmax(axis=1))
                        if k > best_k:
                            best_k, best_lam, best_tau = k, lam, tau
                A_, b_ = fit_offset_corrector(lp_v, Av, yv[fit], best_lam)
                corr_t = At @ A_.T + b_
                preds_offset[hold] = (
                    lp_t + np.where(conf_t[:, None] < best_tau, corr_t, 0.0)
                ).argmax(axis=1)
                per_city[city].update({
                    "kappa_probs_aux": float(cohen_kappa_score(yt[hold], preds_aux[hold])),
                    "kappa_probs_ctrl": float(cohen_kappa_score(yt[hold], preds_ctrl[hold])),
                    "kappa_aux_only": float(cohen_kappa_score(yt[hold], preds_auxonly[hold])),
                    "kappa_aux_offset": float(cohen_kappa_score(yt[hold], preds_offset[hold])),
                    "offset_lam": best_lam,
                    "offset_tau": best_tau,
                })
        loco_methods = [("weighted_loco", preds_w, ("kappa_weighted", "weights")),
                        ("stacked_lr_loco", preds_s, ("kappa_stacked", "C"))]
        if aux_v is not None:
            loco_methods += [("probs_aux_offset_loco", preds_offset, ("kappa_aux_offset", "offset_lam", "offset_tau")),
                             ("probs_aux_lr_loco", preds_aux, ("kappa_probs_aux",)),
                             ("probs_ctrl_lr_loco", preds_ctrl, ("kappa_probs_ctrl",)),
                             ("aux_only_lr_loco", preds_auxonly, ("kappa_aux_only",))]
        for method, preds, keys in loco_methods:
            results[method] = {
                "test": {
                    "acc":   float(accuracy_score(yt, preds)),
                    "f1":    float(f1_score(yt, preds, average="macro")),
                    "kappa": float(cohen_kappa_score(yt, preds)),
                },
                "per_city": {c: {k: v for k, v in d.items() if k in ("n_test", *keys)}
                             for c, d in per_city.items()},
            }

        # ── Aux gate: confusion-pair flips vs the LOCO-weighted baseline ─────
        if aux_v is not None:
            # 0-indexed pairs: LCZ 3<->6, 8<->10, 6<->9, C<->D, 4<->5
            pairs = {"3-6": (2, 5), "8-10": (7, 9), "6-9": (5, 8),
                     "C-D": (12, 13), "4-5": (3, 4)}
            flips = {}
            for name, (a, b) in pairs.items():
                on_pair = np.isin(yt, (a, b))
                base_err = on_pair & (preds_w != yt) & np.isin(preds_w, (a, b))
                flips[name] = {
                    "baseline_pair_errors": int(base_err.sum()),
                    "fixed_by_aux": int((base_err & (preds_offset == yt)).sum()),
                    "new_errors_on_pair": int((on_pair & (preds_w == yt)
                                               & (preds_offset != yt)).sum()),
                }
            results["aux_pair_flips"] = flips
            logger.info("── Aux gate pair flips (vs weighted_loco) ──")
            for name, d in flips.items():
                logger.info(f"  LCZ {name}: {d['baseline_pair_errors']} errors, "
                            f"{d['fixed_by_aux']} fixed, {d['new_errors_on_pair']} new")

    logger.info("── Test-split results (fit on val) ──")
    for method, res in results.items():
        if "test" not in res:
            continue
        t = res["test"]
        extra = ""
        if "weights" in res:
            extra += f" weights={res['weights']}"
        if "temperatures" in res:
            extra += f" T={res['temperatures']}"
        if "C" in res:
            extra += f" C={res['C']}"
        logger.info(f"  kappa={t['kappa']:.4f}  OA={t['acc']:.4f}  F1={t['f1']:.4f}  | {method}{extra}")

    out = args.output_json or args.test_npz.parent / "stacking_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out}")


if __name__ == "__main__":
    main()
