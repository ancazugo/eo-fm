"""G0 — the label-only gate. No embeddings, no GPU, no training.

Everything downstream of Stage 0 is expensive, so the cheapest falsifiable
questions are asked first, against the one independent reference available in
the So2Sat cities:

* **G0.1** How many *annotators* are there, really, as opposed to submissions?
* **G0.2** Leave-one-author-out — does the consensus of the other annotators
  predict a held-out annotator better than picking one other annotator at
  random? If not, there is no consensus signal to harvest.
* **G0.3** Does consensus beat a naive union (largest-polygon-wins) against
  So2Sat? **This is the core claim of the whole approach.**
* **G0.4** Does any shipped quality signal (``oa``, ``oau``, QC flags) actually
  predict So2Sat agreement? If none does, the weight model is uniform and should
  say so rather than pretending to be informative.
* **G0.5** Is confidence monotone against agreement? Confidence that does not
  rank correctness is worse than no confidence, because ``min_conf`` and
  ``conf_gamma`` both trust it.

Measured baselines these are scored against: pooled agreement between
overlapping WUDAPT polygons is 0.71 by area; WUDAPT-vs-So2Sat area-weighted
agreement runs Paris 0.98, Berlin 0.94, Sao Paulo 0.86, Nairobi 0.77, Beijing
0.74, Mumbai 0.66, Guangzhou 0.59, Tehran 0.45.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from rasterio.features import rasterize

from lcz_labels.grid import _CITY_NAME_OVERRIDES

from .config import WudaptConfig
from .consensus import ConsensusGrid, consensus_for_aoi, footprint_grid
from .ingest import N_LCZ, aoi_key
from .quality import apply_gates, burn_order, polygon_weights, submission_accuracy

__all__ = ["AuditResult", "audit_aoi", "so2sat_aoi_map", "run_audit"]

# The 8 cities with independently measured WUDAPT-vs-So2Sat agreement, spanning
# the full range from near-perfect to near-chance.
AUDIT_CITIES = ["Paris", "Berlin", "Sao_Paulo", "Nairobi",
                "Beijing", "Mumbai", "Guangzhou", "Tehran"]

_CONF_BINS = (0.0, 0.2, 0.4, 0.6, 0.8)


def so2sat_aoi_map(config: WudaptConfig) -> dict[str, str]:
    """``{aoi_key: So2Sat city directory name}`` for the 51 So2Sat cities.

    The directory names use underscores while ``JRC_NAME_MAIN`` uses spaces and
    occasionally a different script entirely (``Dongying`` vs ``东营区``), so the
    mapping reuses ``lcz_labels.grid._CITY_NAME_OVERRIDES`` rather than guessing.
    """
    bounds = pd.read_csv(config.labels.city_bounds_csv)
    by_name = {r.JRC_NAME_MAIN: r.SMOD_ID for r in bounds.itertuples()}
    cities_dir = Path(config.labels.so2sat_dir) / config.labels.cities_subdir
    out: dict[str, str] = {}
    if not cities_dir.exists():
        return out
    for d in sorted(p for p in cities_dir.iterdir() if p.is_dir()):
        jrc = _CITY_NAME_OVERRIDES.get(d.name.replace("_", " "), d.name.replace("_", " "))
        smod = by_name.get(jrc)
        if smod is not None:
            out[aoi_key(jrc, smod)] = d.name
    return out


def so2sat_raster(city: str, grid: ConsensusGrid, config: WudaptConfig) -> np.ndarray:
    """So2Sat ground-truth classes (1-17, 0 = no label) burned on the AOI grid."""
    path = (Path(config.labels.so2sat_dir) / config.labels.cities_subdir / city
            / f"patches_reference_{city}.gpkg")
    if not path.exists():
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path).to_crs(grid.crs)
    gdf = gdf[gdf["LCZ_class"].between(1, N_LCZ)]
    if gdf.empty:
        return np.zeros(grid.shape, dtype=np.uint8)
    return rasterize(
        zip(gdf.geometry.values, gdf["LCZ_class"].astype("uint8")),
        out_shape=grid.shape, transform=grid.transform, fill=0,
        dtype="uint8", all_touched=False,
    )


def _set_agreement(bitmask: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-pixel correctness: a set label is correct if the truth is in the set.

    Mirrors ``lcz_train.eval``'s coarse-scoring convention, so an audit number
    means the same thing as a training-time number.
    """
    bit = np.left_shift(np.uint32(1), (truth.astype(np.uint32) - 1))
    return (bitmask & bit) != 0


def naive_union_labels(gdf: gpd.GeoDataFrame, grid: ConsensusGrid) -> np.ndarray:
    """Baseline for G0.3: largest polygon wins, no consensus, no weighting.

    This is what "just use WUDAPT" looks like. Larger polygons are burned first
    so smaller, more specific ones overwrite them — the most favourable reading
    of a naive union.
    """
    g = gdf.to_crs(grid.crs).assign(_a=gdf["area_km2"].to_numpy())
    g = g.sort_values("_a", ascending=False)
    return rasterize(
        zip(g.geometry.values, g["class"].astype("uint8")),
        out_shape=grid.shape, transform=grid.transform, fill=0,
        dtype="uint8", all_touched=False,
    )


@dataclass
class AuditResult:
    aoi: str
    city: str
    n_polys: int
    n_submissions: int
    n_annotators: int
    n_polys_gated: int
    px_consensus: int
    px_compared: int
    hard_frac: float
    mean_conf: float
    mean_n_eff: float
    oa_consensus: float = float("nan")
    oa_naive_union: float = float("nan")
    oa_hard_only: float = float("nan")
    loao_consensus: float = float("nan")
    loao_random_peer: float = float("nan")
    conf_sweep: dict = field(default_factory=dict)
    conf_monotone: bool = False
    quality_signal: dict = field(default_factory=dict)

    @property
    def g0_3_margin(self) -> float:
        return self.oa_consensus - self.oa_naive_union

    @property
    def g0_2_margin(self) -> float:
        return self.loao_consensus - self.loao_random_peer


def _confidence_sweep(bitmask: np.ndarray, confidence: np.ndarray,
                      truth: np.ndarray) -> tuple[dict, bool]:
    """Agreement at rising confidence thresholds; monotone is the G0.5 gate."""
    ok = _set_agreement(bitmask, truth)
    sweep: dict[str, float] = {}
    vals = []
    for t in _CONF_BINS:
        sel = confidence >= t
        v = float(ok[sel].mean()) if sel.any() else float("nan")
        sweep[f"conf>={t:.1f}"] = v
        sweep[f"n@{t:.1f}"] = int(sel.sum())
        if np.isfinite(v) and sel.sum() >= 100:
            vals.append(v)
    monotone = len(vals) >= 3 and all(b >= a - 1e-3 for a, b in zip(vals, vals[1:]))
    return sweep, monotone


def _quality_signal(gdf: gpd.GeoDataFrame, weights: np.ndarray, grid: ConsensusGrid,
                    truth: np.ndarray) -> dict:
    """G0.4 — does any shipped metadata field rank annotator correctness?

    Each polygon is scored by how well its own class matches So2Sat over its own
    footprint, then correctness is aggregated by tercile of each candidate
    signal. A signal that is informative shows a rising sequence.
    """
    g = gdf.to_crs(grid.crs)
    per_poly_ok, per_poly_n = [], []
    shapes = list(zip(g.geometry.values, np.arange(1, len(g) + 1)))
    idx = rasterize(shapes, out_shape=grid.shape, transform=grid.transform,
                    fill=0, dtype="int32", all_touched=False)
    # Note: overlapping polygons mean only the last-burned one is scored here.
    # That is fine for a *relative* ranking of quality signals.
    flat_idx, flat_truth = idx.ravel(), truth.ravel()
    sel = (flat_idx > 0) & (flat_truth > 0)
    if not sel.any():
        return {}
    poly_i = flat_idx[sel] - 1
    hit = (g["class"].to_numpy()[poly_i] == flat_truth[sel])
    n = np.bincount(poly_i, minlength=len(g)).astype(float)
    k = np.bincount(poly_i, weights=hit.astype(float), minlength=len(g))
    scored = n > 0
    if scored.sum() < 30:
        return {}

    out: dict[str, list] = {}
    candidates = {
        "oa": pd.to_numeric(gdf["oa"], errors="coerce").to_numpy(),
        "oau_or_oa": submission_accuracy(gdf),
        "weight": weights,
        "area_km2": gdf["area_km2"].to_numpy(),
    }
    for name, sig in candidates.items():
        s = sig[scored]
        if not np.isfinite(s).any() or np.nanstd(s) == 0:
            continue
        try:
            tier = pd.qcut(pd.Series(s), 3, labels=False, duplicates="drop").to_numpy()
        except ValueError:
            continue
        kk, nn = k[scored], n[scored]
        vals = [float(kk[tier == t].sum() / max(nn[tier == t].sum(), 1)) for t in range(3)
                if (tier == t).any()]
        if len(vals) == 3:
            out[name] = [round(v, 4) for v in vals]
    return out


def _loao(gdf: gpd.GeoDataFrame, weights: np.ndarray, config: WudaptConfig,
          grid: ConsensusGrid, *, max_authors: int = 8,
          seed: int = 0) -> tuple[float, float]:
    """G0.2 — leave-one-author-out, with a random-peer baseline.

    Needs no ground truth at all, which is what makes it available in the ~800
    cities that have no So2Sat. Returns ``(consensus_score, random_peer_score)``,
    both area-weighted over the held-out author's own footprint.
    """
    rng = np.random.default_rng(seed)
    authors = gdf["annotator_id"].value_counts()
    authors = authors[authors >= 5].index.tolist()
    if len(authors) < 3:
        return float("nan"), float("nan")
    rng.shuffle(authors)
    authors = authors[:max_authors]

    cons_hits = cons_n = peer_hits = peer_n = 0
    for a in authors:
        held = gdf["annotator_id"] == a
        others = gdf.loc[~held]
        if others.empty or others["annotator_id"].nunique() < 2:
            continue
        w_others = weights[(~held).to_numpy()]
        if not (w_others > 0).any():
            continue
        try:
            # Pin the AOI grid across folds: ConsensusResult.index is relative to
            # its own grid, so a per-fold footprint grid would make `truth[res.index]`
            # read entirely different pixels.
            res = consensus_for_aoi(others, w_others, config, grid=grid)
        except ValueError:
            continue
        if len(res) == 0:
            continue

        truth = rasterize(
            zip(gdf.loc[held].to_crs(grid.crs).geometry.values,
                gdf.loc[held, "class"].astype("uint8")),
            out_shape=grid.shape, transform=grid.transform, fill=0,
            dtype="uint8", all_touched=False,
        ).ravel()
        t = truth[res.index]
        m = t > 0
        if not m.any():
            continue
        cons_hits += int(_set_agreement(res.bitmask[m], t[m]).sum())
        cons_n += int(m.sum())

        # Baseline: one randomly chosen *other* annotator, over the same pixels.
        peer = rng.choice(others["annotator_id"].unique())
        psub = others[others["annotator_id"] == peer].to_crs(grid.crs)
        praster = rasterize(
            zip(psub.geometry.values, psub["class"].astype("uint8")),
            out_shape=grid.shape, transform=grid.transform, fill=0,
            dtype="uint8", all_touched=False,
        ).ravel()
        p = praster[res.index][m]
        pm = p > 0
        if pm.any():
            peer_hits += int((p[pm] == t[m][pm]).sum())
            peer_n += int(pm.sum())

    cons = cons_hits / cons_n if cons_n else float("nan")
    peer = peer_hits / peer_n if peer_n else float("nan")
    return cons, peer


def audit_aoi(gdf: gpd.GeoDataFrame, config: WudaptConfig, *, aoi: str, city: str | None,
              res_m: float = 20.0, seed: int = 0) -> AuditResult:
    """Run every G0 measurement for one AOI."""
    n_polys, n_sub, n_ann = len(gdf), gdf["submission_id"].nunique(), gdf["annotator_id"].nunique()
    gated = apply_gates(gdf, config)
    if gated.empty:
        raise ValueError(f"[{aoi}] quality gates removed every polygon")
    weights = polygon_weights(gated, config)["weight"].to_numpy()

    res = consensus_for_aoi(gated, weights, config, res_m=res_m)
    grid = res.grid
    out = AuditResult(
        aoi=aoi, city=city or "", n_polys=n_polys, n_submissions=n_sub, n_annotators=n_ann,
        n_polys_gated=len(gated), px_consensus=len(res), px_compared=0,
        hard_frac=float((res.set_size == 1).mean()) if len(res) else float("nan"),
        mean_conf=float(res.confidence.mean()) if len(res) else float("nan"),
        mean_n_eff=float(res.n_eff.mean()) if len(res) else float("nan"),
    )

    gated_sorted = burn_order(gated.assign(_w=weights))
    w_sorted = gated_sorted.pop("_w").to_numpy()
    # LOAO refits consensus once per held-out author, so it runs on a coarser
    # grid than the headline numbers. It is a *relative* comparison (consensus
    # vs random peer) over the same pixels, so resolution cancels.
    loao_grid = footprint_grid(gated_sorted, config, res_m=res_m * 2)
    out.loao_consensus, out.loao_random_peer = _loao(
        gated_sorted, w_sorted, config, loao_grid, seed=seed
    )

    if city:
        truth = so2sat_raster(city, grid, config)
        flat = truth.ravel()
        t = flat[res.index]
        m = t > 0
        out.px_compared = int(m.sum())
        if m.any():
            out.oa_consensus = float(_set_agreement(res.bitmask[m], t[m]).mean())
            hard = m & (res.set_size == 1)
            if hard.any():
                th = flat[res.index][hard]
                out.oa_hard_only = float((res.top_class[hard] == th).mean())
            naive = naive_union_labels(gated, grid).ravel()
            nm = (naive > 0) & (flat > 0)
            out.oa_naive_union = float((naive[nm] == flat[nm]).mean()) if nm.any() else float("nan")
            out.conf_sweep, out.conf_monotone = _confidence_sweep(
                res.bitmask[m], res.confidence[m], t[m]
            )
            out.quality_signal = _quality_signal(gated, weights, grid, truth)
    return out


def run_audit(config: WudaptConfig, *, cities: list[str] | None = None,
              res_m: float = 20.0, seed: int = 0) -> pd.DataFrame:
    """Audit the So2Sat overlap cities and return one row per AOI."""
    clean_path = Path(config.cache_dir) / f"wudapt_clean_{config.ingest_hash}.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(f"run `lcz_wudapt ingest` first: {clean_path}")
    gdf = gpd.read_parquet(clean_path)
    amap = so2sat_aoi_map(config)
    wanted = set(cities or AUDIT_CITIES)

    rows = []
    for aoi, city in amap.items():
        if city not in wanted:
            continue
        sub = gdf[gdf["aoi"] == aoi]
        if sub.empty:
            logger.warning(f"[{aoi}] no WUDAPT polygons; skipped")
            continue
        logger.info(f"── audit {city} ({aoi}): {len(sub):,} polygons")
        try:
            rows.append(asdict(audit_aoi(sub, config, aoi=aoi, city=city,
                                         res_m=res_m, seed=seed)))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(f"[{aoi}] skipped: {exc}")
    if not rows:
        raise RuntimeError("audit produced no rows")
    return pd.DataFrame(rows)
