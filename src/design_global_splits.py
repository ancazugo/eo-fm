"""Design city-group splits for global LCZ training across So2Sat and WUDAPT.

Answers four questions with numbers rather than by hand:

1. **What is a unit?** AOIs are grouped into *blocks* by complete-linkage
   clustering of their centroids at ``--block-km``. Complete linkage bounds
   a block's diameter, which matters here: grouping by "bounding boxes touch
   after a buffer" chains whole megaregions together (measured on GUPPD: one
   25 km-buffered component swallows 661 urban areas). Leakage between
   neighbouring blocks is handled separately, by ``--buffer-km``.

2. **What should a test set look like?** Like the places the map will be used.
   The target is the 5,558 GUPPD urban areas, described by region x Koppen
   main group (A/B/C/D/E), weighted by bounding-box area or by count. A test
   set is scored on three things against that target:

   * stratum mix, as the L1 distance between shares;
   * distance to training labels: the distribution of test-to-nearest-train
     distances, compared by KS statistic with the distribution of
     target-to-nearest-train distances. This is the nearest-neighbour
     distance matching idea (Milà et al. 2022; Linnenbrink et al. 2024), so a
     test set is neither easier (too close) nor harder (too far) than the
     real map;
   * coverage: every LCZ class and every label-year bin gets a minimum
     labelled area.

3. **Which blocks may be tested on?** Only trustworthy ones (``--max-conflict``,
   ``--min-test-km2``, ``--min-test-classes``, ``--quarantine``) and none that
   contain a So2Sat training city. The So2Sat culture-10 are fixed as test A
   and never move.

4. **Time.** Every AOI carries labelled area per label-year bin. Test sets are
   required to cover each bin. Polygons inside val/test blocks that were drawn
   by different submissions in different years over the same ground are
   written out as *revisit pairs*: a reference for temporal consistency
   (same class) and for change (class differs).

Outputs (``--out-dir``):

* ``aoi_table.parquet``  -- one row per AOI: block, split, fold, stratum, km2,
  class and year-bin areas, quality
* ``blocks.parquet``     -- one row per block
* ``polygon_split.parquet`` -- per WUDAPT polygon: split, plus whether the
  buffer drops it from training
* ``revisit_pairs.parquet`` -- temporal reference pairs inside val/test
* ``design_report.md``   -- the tables the choice should be judged on

Example:
    python src/design_global_splits.py \\
        --wudapt-clean ${DATA_DIR}/output/lcz_wudapt/wudapt_clean_<hash>.parquet \\
        --aoi-quality  ${DATA_DIR}/output/lcz_wudapt/suitability_aoi.csv \\
        --koppen-tif   ${DATA_DIR}/input/koppen/koppen_geiger_0p00833333.tif \\
        --out-dir data/global_split_design
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import AgglomerativeClustering

REPO = Path(__file__).resolve().parent.parent
EARTH_R_KM = 6371.0088
N_LCZ = 17
RURAL_AOI = "_rural"
PATCH_KM2 = 0.32 * 0.32
YEAR_BINS = ((-10_000, 2016, "<=2016"), (2017, 2018, "2017-18"),
             (2019, 2021, "2019-21"), (2022, 10_000, ">=2022"))
YEAR_LABELS = tuple(b[2] for b in YEAR_BINS)
KG_GROUPS = ("A", "B", "C", "D", "E")

# Fixed benchmark: never trained on, never moved (lcz_wudapt.leakage).
SO2SAT_TEST_CITIES = ("Santiago", "San_Jose", "Tehran", "Nairobi", "Sydney",
                      "Moscow", "Jakarta", "Munich", "Mumbai", "Guangzhou")


# ── configuration ────────────────────────────────────────────────────────────

@dataclass
class DesignConfig:
    block_km: float = 50.0          # complete-linkage diameter bound of a block
    buffer_km: float = 20.0         # train polygons this close to val/test are dropped
    test_frac: float = 0.15         # share of WUDAPT labelled area in test B
    val_frac: float = 0.10
    folds: int = 5
    min_test_km2: float = 5.0       # a test block must carry at least this much label
    min_test_classes: int = 8
    max_conflict: float = 0.40      # nbr_conflict ceiling for val/test eligibility
    max_stratum_holdout: float = 0.40  # val+test may take at most this share of any
                                       # region x Koppen cell's labels (test A counts)
    min_class_km2: float = 2.0      # coverage floor per class in a test set
    min_year_km2: float = 5.0       # coverage floor per label-year bin in a test set
    so2sat_train_min_patches: int = 500
    quarantine: tuple[str, ...] = ("tehran",)   # AOI-key substrings: never val/test
    force_test: tuple[str, ...] = ()            # AOI-key substrings: always test B
    force_train: tuple[str, ...] = ()           # AOI-key substrings: always train
    target_weight: str = "area"     # "area" (bbox km2) or "count" of GUPPD areas
    restarts: int = 8
    swap_iters: int = 400
    n_candidates: int = 64          # blocks scored per greedy step
    seed: int = 0
    # objective weights
    w_strata: float = 1.0
    w_dist: float = 1.0
    w_class: float = 0.5
    w_year: float = 0.5
    w_size: float = 2.0
    extra: dict = field(default_factory=dict)


# ── small geometry helpers ───────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Pairwise great-circle distance (km) between two point sets."""
    p1, l1 = np.radians(np.asarray(lat1))[:, None], np.radians(np.asarray(lon1))[:, None]
    p2, l2 = np.radians(np.asarray(lat2))[None, :], np.radians(np.asarray(lon2))[None, :]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def weighted_centroid(lat, lon, w) -> tuple[float, float]:
    """Area-weighted centroid on the sphere (safe across the antimeridian)."""
    lat, lon, w = np.radians(lat), np.radians(lon), np.asarray(w, float)
    x = np.sum(w * np.cos(lat) * np.cos(lon))
    y = np.sum(w * np.cos(lat) * np.sin(lon))
    z = np.sum(w * np.sin(lat))
    return float(np.degrees(np.arctan2(z, np.hypot(x, y)))), float(np.degrees(np.arctan2(y, x)))


def year_bin(years: pd.Series) -> pd.Series:
    y = pd.to_numeric(years, errors="coerce")
    out = pd.Series(pd.NA, index=y.index, dtype="object")
    for lo, hi, lab in YEAR_BINS:
        out[(y >= lo) & (y <= hi)] = lab
    return out


# ── Koppen and region lookups ────────────────────────────────────────────────

class Koppen:
    """Koppen class at points: a GeoTIFF if given (e.g. Beck et al. 2023, 1 km),
    else kgcpy's bundled Rubel et al. 2016 raster (~3 km)."""

    _BECK = {1: "Af", 2: "Am", 3: "Aw", 4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk", 8: "Csa",
             9: "Csb", 10: "Csc", 11: "Cwa", 12: "Cwb", 13: "Cwc", 14: "Cfa", 15: "Cfb",
             16: "Cfc", 17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd", 21: "Dwa", 22: "Dwb",
             23: "Dwc", 24: "Dwd", 25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd", 29: "ET", 30: "EF"}

    def __init__(self, tif: Path | None = None):
        self.tif = tif
        if tif is None:
            import kgcpy
            self._arr = np.array(kgcpy.img)
            self._names = dict(zip(kgcpy.kg_zoneNum_df.zoneNum, kgcpy.kg_zoneNum_df.kg_zone))
            self._ocean = {0, 32}

    def __call__(self, lon, lat) -> np.ndarray:
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        if self.tif is not None:
            import rasterio
            with rasterio.open(self.tif) as ds:
                vals = np.array([v[0] for v in ds.sample(zip(lon, lat))])
            return np.array([self._BECK.get(int(v), "Ocean") for v in vals], dtype=object)
        a = self._arr
        col = np.clip(np.round((lon + 180) * a.shape[1] / 360 - 0.5).astype(int), 0, a.shape[1] - 1)
        row = np.clip(np.round(-(lat - 90) * a.shape[0] / 180 - 0.5).astype(int), 0, a.shape[0] - 1)
        vals = a[row, col]
        return np.array(["Ocean" if v in self._ocean else self._names[int(v)] for v in vals], dtype=object)


def region_table() -> dict[str, str]:
    """ISO3 -> region, read from lcz_wudapt.splits without importing the package."""
    src = (REPO / "lcz_wudapt" / "splits.py").read_text()
    start, end = src.index("ISO_TO_REGION: dict"), src.index("def region_for")
    ns: dict = {}
    exec("import numpy as np\n" + src[start:end], ns)  # the table is literal data
    return ns["ISO_TO_REGION"]


def modal_land_class(classes: np.ndarray, weights: np.ndarray) -> str:
    keep = classes != "Ocean"
    if not keep.any():
        return "Ocean"
    s = pd.Series(weights[keep]).groupby(classes[keep]).sum()
    return str(s.idxmax())


# ── loading ──────────────────────────────────────────────────────────────────

def load_wudapt(path: Path, require_qc: bool) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    need = {"aoi", "class", "area_km2", "label_year", "annotator_id", "iso"}
    missing = need - set(gdf.columns)
    if missing:
        raise SystemExit(f"{path} lacks columns {sorted(missing)}")
    if require_qc:
        qc = [c for c in ("qc_step1", "qc_step2", "qc_step3") if c in gdf.columns]
        ok = np.ones(len(gdf), bool)
        for c in qc:
            ok &= ~(gdf[c] == False).fillna(False).to_numpy()  # noqa: E712 -- NA is not a fail
        logger.info(f"QC filter keeps {ok.sum():,} of {len(gdf):,} polygons")
        gdf = gdf[ok]
    gdf = gdf[gdf["class"].between(1, N_LCZ)].copy()
    pts = gdf.geometry.representative_point()
    gdf["lon"], gdf["lat"] = pts.x.to_numpy(), pts.y.to_numpy()
    gdf["ybin"] = year_bin(gdf["label_year"])
    gdf["pid"] = np.arange(len(gdf))
    return gdf


def snap_rural(gdf: gpd.GeoDataFrame, snap_km: float = 30.0) -> gpd.GeoDataFrame:
    """Give `_rural` polygons the nearest AOI within snap_km, else a 1-degree cell.

    Rural polygons are 62% natural classes (lcz_wudapt README). Dropping them
    strips the classes So2Sat under-samples; leaving them unassigned lets them
    reach train from right next to a test city.
    """
    out = gdf.copy()
    rural = out["aoi"] == RURAL_AOI
    if not rural.any():
        return out
    urb = out[~rural].groupby("aoi").apply(
        lambda d: pd.Series(weighted_centroid(d.lat, d.lon, d.area_km2), index=["lat", "lon"]),
        include_groups=False)
    r = out[rural]
    if len(urb):
        d = haversine_km(r.lat, r.lon, urb.lat, urb.lon)
        j = d.argmin(axis=1)
        near = d[np.arange(len(r)), j] <= snap_km
    else:
        near, j = np.zeros(len(r), bool), np.zeros(len(r), int)
    cell = ("rural:" + np.floor(r.lon).astype(int).astype(str) + "_" + np.floor(r.lat).astype(int).astype(str))
    out.loc[rural, "aoi"] = np.where(near, urb.index.to_numpy()[j] if len(urb) else cell, cell)
    logger.info(f"rural polygons: {int(rural.sum()):,} | snapped to an AOI {int(near.sum()):,} | "
                f"own 1-degree cell {int((~near).sum()):,}")
    return out


def aoi_table_wudapt(gdf: gpd.GeoDataFrame, regions: dict[str, str], kop: Koppen,
                     quality: pd.DataFrame | None) -> pd.DataFrame:
    gdf = gdf.copy()
    gdf["koppen"] = kop(gdf.lon.to_numpy(), gdf.lat.to_numpy())
    rows = []
    for aoi, d in gdf.groupby("aoi", sort=True):
        w = d.area_km2.to_numpy()
        lat, lon = weighted_centroid(d.lat, d.lon, w)
        iso = d.iso.dropna().mode()
        iso = str(iso.iloc[0]) if len(iso) else None
        rec = {"aoi": aoi, "source": "wudapt", "lat": lat, "lon": lon, "iso": iso,
               "region": regions.get(iso or "", "Unknown"),
               "koppen": modal_land_class(d.koppen.to_numpy(dtype=object), w),
               "km2": float(w.sum()), "n_polys": len(d),
               "n_annotators": int(d.annotator_id.nunique()),
               "n_submissions": int(d.submission_id.nunique()) if "submission_id" in d else np.nan,
               "jrc_name": d.jrc_name.dropna().iloc[0] if "jrc_name" in d and d.jrc_name.notna().any() else aoi}
        cls = d.groupby("class").area_km2.sum()
        for c in range(1, N_LCZ + 1):
            rec[f"c{c}"] = float(cls.get(c, 0.0))
        yb = d.groupby("ybin").area_km2.sum()
        for lab in YEAR_LABELS:
            rec[f"y{lab}"] = float(yb.get(lab, 0.0))
        rows.append(rec)
    t = pd.DataFrame(rows)
    if quality is not None:
        t = t.merge(quality, on="aoi", how="left")
    for col in ("nbr_conflict", "agree_on_overlap"):
        if col not in t:
            t[col] = np.nan
    return t


def aoi_table_so2sat(summary: Path, class_counts: Path, regions: dict[str, str],
                     kop: Koppen) -> pd.DataFrame:
    s = pd.read_csv(summary)
    cc = pd.read_csv(class_counts)
    iso_by_country = {
        "Netherlands": "NLD", "China": "CHN", "Germany": "DEU", "Colombia": "COL", "Argentina": "ARG",
        "Egypt": "EGY", "South Africa": "ZAF", "Venezuela": "VEN", "United States": "USA",
        "Bangladesh": "BGD", "Turkey": "TUR", "Indonesia": "IDN", "Pakistan": "PAK", "Peru": "PER",
        "Portugal": "PRT", "United Kingdom": "GBR", "Spain": "ESP", "Australia": "AUS", "Italy": "ITA",
        "Russia": "RUS", "India": "IND", "Kenya": "KEN", "Japan": "JPN", "France": "FRA",
        "Philippines": "PHL", "Brazil": "BRA", "Chile": "CHL", "Iran": "IRN", "Canada": "CAN",
        "Switzerland": "CHE"}
    rows = []
    for r in s.itertuples(index=False):
        iso = iso_by_country.get(r.country)
        rec = {"aoi": f"so2sat:{r.city_dir}", "source": "so2sat", "lat": r.lat, "lon": r.lon,
               "iso": iso, "region": regions.get(iso or "", "Unknown"),
               "koppen": str(kop(np.array([r.lon]), np.array([r.lat]))[0]),
               "km2": r.n_total * PATCH_KM2, "n_polys": int(r.n_total), "n_annotators": np.nan,
               "n_submissions": np.nan, "jrc_name": r.city_label,
               "so2sat_role": r.role, "so2sat_patches": int(r.n_total),
               "so2sat_city": r.city_dir, "nbr_conflict": np.nan, "agree_on_overlap": np.nan}
        sub = cc[cc.city_dir == r.city_dir].set_index("LCZ_class").n
        for c in range(1, N_LCZ + 1):
            rec[f"c{c}"] = float(sub.get(c, 0)) * PATCH_KM2
        for lab in YEAR_LABELS:
            rec[f"y{lab}"] = rec["km2"] if lab == "2017-18" else 0.0   # So2Sat imagery is 2016-18
        rows.append(rec)
    return pd.DataFrame(rows)


def target_table(guppd: Path, regions: dict[str, str], kop: Koppen, n_grid: int = 5) -> pd.DataFrame:
    g = pd.read_csv(guppd)
    xs = np.linspace(0, 1, n_grid)
    gx, gy = np.meshgrid(xs, xs)
    lon = g.minx.to_numpy()[:, None] + gx.ravel()[None, :] * (g.maxx - g.minx).to_numpy()[:, None]
    lat = g.miny.to_numpy()[:, None] + gy.ravel()[None, :] * (g.maxy - g.miny).to_numpy()[:, None]
    kc = kop(lon.ravel(), lat.ravel()).reshape(lon.shape)
    g["koppen"] = [modal_land_class(k, np.ones(k.size)) for k in kc]
    g["region"] = g.ISO.map(regions).fillna("Unknown")
    g["lat"], g["lon"] = (g.miny + g.maxy) / 2, (g.minx + g.maxx) / 2
    from shapely.geometry import box
    boxes = gpd.GeoSeries([box(*b) for b in g[["minx", "miny", "maxx", "maxy"]].itertuples(index=False)], crs=4326)
    g["bbox_km2"] = boxes.to_crs("EPSG:6933").area.to_numpy() / 1e6
    return g


def stratum(region: pd.Series, koppen: pd.Series) -> pd.Series:
    return region.astype(str) + "|" + koppen.astype(str).str[0]


# ── blocks ───────────────────────────────────────────────────────────────────

def make_blocks(aois: pd.DataFrame, block_km: float) -> np.ndarray:
    """Complete-linkage clusters of AOI centroids: max pairwise distance <= block_km."""
    if len(aois) == 1:
        return np.zeros(1, int)
    d = haversine_km(aois.lat, aois.lon, aois.lat, aois.lon)
    cl = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="complete",
                                 distance_threshold=block_km)
    return cl.fit_predict(d)


# ── objective ────────────────────────────────────────────────────────────────

@dataclass
class Problem:
    aois: pd.DataFrame          # all AOIs, with block, stratum, km2, c*, y*
    target_share: pd.Series     # stratum -> share
    target_lat: np.ndarray
    target_lon: np.ndarray
    target_w: np.ndarray
    d_aoi: np.ndarray           # AOI x AOI km
    d_tgt: np.ndarray           # target x AOI km
    cfg: DesignConfig

    def __post_init__(self):
        a = self.aois
        self.blocks = np.sort(a.block.unique())
        self.blk_of = a.block.to_numpy()
        self.km2 = a.km2.to_numpy()
        self.cls = a[[f"c{c}" for c in range(1, N_LCZ + 1)]].to_numpy()
        self.yr = a[[f"y{lab}" for lab in YEAR_LABELS]].to_numpy()
        self.strata = a.stratum.to_numpy()
        self.wudapt_km2 = float(self.km2[a.source.to_numpy() == "wudapt"].sum())
        # Stratum codes aligned with the target share, so a stratum mix is one bincount.
        keys = sorted(set(self.strata) | set(self.target_share.index))
        code = {k: i for i, k in enumerate(keys)}
        self.scode = np.array([code[s] for s in self.strata])
        self.tvec = self.target_share.reindex(keys, fill_value=0.0).to_numpy()
        self.n_strata = len(keys)
        self.aoi_idx = {b: np.flatnonzero(self.blk_of == b) for b in self.blocks}
        self.blk_km2 = {b: float(self.km2[i].sum()) for b, i in self.aoi_idx.items()}
        # Per-block labelled area by stratum, and each stratum's total, for the holdout cap.
        self.blk_str = {b: np.bincount(self.scode[i], weights=self.km2[i], minlength=self.n_strata)
                        for b, i in self.aoi_idx.items()}
        self.str_total = np.bincount(self.scode, weights=self.km2, minlength=self.n_strata)
        # Nearest-train lookups are the hot path (thousands of objective calls):
        # keep each row's K nearest AOIs sorted, fall back to a full scan only
        # when all K are excluded from training.
        self._tgt = self._topk(self.d_tgt)
        self._aoi = self._topk(self.d_aoi)

    def held_by_stratum(self, mask: np.ndarray) -> np.ndarray:
        return np.bincount(self.scode[mask], weights=self.km2[mask], minlength=self.n_strata)

    def within_cap(self, held: np.ndarray, add: np.ndarray) -> bool:
        """True if holding `add` on top of `held` keeps every stratum's held-out
        share of labelled area <= cfg.max_stratum_holdout, so training always
        keeps the rest of every climate/region cell (strata with no labels are
        unconstrained)."""
        new = held + add
        touched = add > 0
        frac = new[touched] / np.maximum(self.str_total[touched], 1e-12)
        return bool((frac <= self.cfg.max_stratum_holdout + 1e-9).all())

    @staticmethod
    def _topk(d: np.ndarray, k: int = 64):
        k = min(k, d.shape[1])
        idx = np.argsort(d, axis=1)[:, :k]
        return idx, np.take_along_axis(d, idx, axis=1), d

    @staticmethod
    def _nearest(tk, rows: np.ndarray | slice, train_mask: np.ndarray) -> np.ndarray:
        idx, dist, full = tk
        idx, dist = idx[rows], dist[rows]
        ok = train_mask[idx]
        out = dist[np.arange(len(idx)), ok.argmax(axis=1)].copy()
        miss = ~ok.any(axis=1)
        if miss.any():
            r = np.arange(full.shape[0])[rows][miss]
            out[miss] = full[np.ix_(r, train_mask)].min(axis=1)
        return out

    def nearest_train_targets(self, train_mask: np.ndarray) -> np.ndarray:
        return self._nearest(self._tgt, slice(None), train_mask)

    def nearest_train_aois(self, rows: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        return self._nearest(self._aoi, rows, train_mask)

    def members(self, chosen) -> np.ndarray:
        return (np.concatenate([self.aoi_idx[b] for b in chosen]) if chosen
                else np.zeros(0, int))


def ks_weighted(x: np.ndarray, wx: np.ndarray, y: np.ndarray, wy: np.ndarray) -> float:
    """Two-sample KS statistic with sample weights."""
    if len(x) == 0 or len(y) == 0:
        return 1.0
    grid = np.sort(np.concatenate([x, y]))
    def cdf(v, w):
        o = np.argsort(v)
        cw = np.cumsum(w[o]) / w.sum()
        return np.concatenate([[0.0], cw])[np.searchsorted(v[o], grid, side="right")]
    return float(np.max(np.abs(cdf(x, wx) - cdf(y, wy))))


def score(p: Problem, chosen, frac: float, fixed_out: np.ndarray) -> tuple[float, dict]:
    """Lower is better. `chosen` are the candidate set's blocks; `fixed_out` AOIs
    already held out elsewhere (they are neither in the set nor training)."""
    cfg = p.cfg
    rows = p.members(chosen)
    if rows.size == 0:
        return np.inf, {}
    train_mask = ~fixed_out.copy()
    train_mask[rows] = False
    if not train_mask.any():
        return np.inf, {}
    km2 = p.km2[rows]
    tot = km2.sum()
    share = np.bincount(p.scode[rows], weights=km2, minlength=p.n_strata) / max(tot, 1e-12)
    l1 = float(np.abs(share - p.tvec).sum()) / 2
    # distance matching: set AOIs vs map targets, both to their nearest training AOI
    dt = p.nearest_train_aois(rows, train_mask)
    dg = p.nearest_train_targets(train_mask)
    ks = ks_weighted(dt, km2, dg, p.target_w)
    cls_def = float(np.mean(np.clip(1 - p.cls[rows].sum(axis=0) / cfg.min_class_km2, 0, 1)))
    yr_def = float(np.mean(np.clip(1 - p.yr[rows].sum(axis=0) / cfg.min_year_km2, 0, 1)))
    size = abs(tot / max(p.wudapt_km2, 1e-9) - frac) / frac
    total = (cfg.w_strata * l1 + cfg.w_dist * ks + cfg.w_class * cls_def
             + cfg.w_year * yr_def + cfg.w_size * size)
    return total, {"strata_l1": l1, "dist_ks": ks, "class_deficit": cls_def,
                   "year_deficit": yr_def, "size_err": float(size), "km2": float(tot)}


def select(p: Problem, eligible: list, frac: float, fixed_out: np.ndarray,
           rng: np.random.Generator, must: set | None = None) -> tuple[set, dict]:
    """Greedy build-up over sampled candidates, then swap search; best of `restarts`.

    Blocks in `must` start in the set and are never swapped out.
    """
    cfg = p.cfg
    must = set(must or ())
    best, best_s, best_info = set(), np.inf, {}
    goal = frac * p.wudapt_km2

    for r in range(cfg.restarts):
        chosen: set = set(must)
        pool = [b for b in eligible if b not in must]
        area = sum(p.blk_km2[b] for b in must)
        cur = score(p, chosen, frac, fixed_out)[0] if chosen else np.inf
        held = p.held_by_stratum(fixed_out) + sum((p.blk_str[b] for b in must), np.zeros(p.n_strata))
        while pool and area < goal:
            pool = [b for b in pool if p.within_cap(held, p.blk_str[b])]
            if not pool:
                break
            k = min(len(pool), cfg.n_candidates)
            cand = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
            s, b = min(((score(p, chosen | {b}, frac, fixed_out)[0], b) for b in cand),
                       key=lambda t: t[0])
            chosen.add(b)
            pool.remove(b)
            area += p.blk_km2[b]
            held = held + p.blk_str[b]
            cur = s
        outside = [b for b in eligible if b not in chosen]
        for _ in range(cfg.swap_iters if (chosen - must) and outside else 0):
            free = sorted(chosen - must)
            b_out = free[rng.integers(len(free))]
            b_in = outside[rng.integers(len(outside))]
            if not p.within_cap(held - p.blk_str[b_out], p.blk_str[b_in]):
                continue
            trial = (chosen - {b_out}) | {b_in}
            s = score(p, trial, frac, fixed_out)[0]
            if s < cur:
                chosen, cur = trial, s
                outside[outside.index(b_in)] = b_out
                held = held - p.blk_str[b_out] + p.blk_str[b_in]
        s, info = score(p, chosen, frac, fixed_out)
        logger.info(f"  restart {r}: objective {s:.4f} {json.dumps({k: round(v, 3) for k, v in info.items()})}")
        if s < best_s:
            best, best_s, best_info = set(chosen), s, info
    return best, {"objective": best_s, **best_info}


def assign_folds(p: Problem, blocks: list, k: int, rng: np.random.Generator) -> dict:
    """Balance train blocks into k folds on stratum and class area (greedy, largest first)."""
    feats = []
    strata = sorted(set(p.strata))
    for b in blocks:
        m = p.blk_of == b
        st = pd.Series(p.km2[m]).groupby(p.strata[m]).sum().reindex(strata, fill_value=0).to_numpy()
        feats.append(np.concatenate([st, p.cls[m].sum(axis=0)]))
    feats = np.array(feats) if feats else np.zeros((0, len(strata) + N_LCZ))
    total = feats.sum(axis=0) + 1e-9
    order = np.argsort(-feats.sum(axis=1) + rng.random(len(blocks)) * 1e-6)
    load = np.zeros((k, feats.shape[1]))
    fold = {}
    for i in order:
        # put the block where it most reduces imbalance relative to the 1/k ideal
        cost = [np.abs((load[f] + feats[i]) / total - 1 / k).sum() - np.abs(load[f] / total - 1 / k).sum()
                for f in range(k)]
        f = int(np.argmin(cost))
        load[f] += feats[i]
        fold[blocks[i]] = f
    return fold


# ── buffer and revisits ──────────────────────────────────────────────────────

def buffer_drop(gdf: gpd.GeoDataFrame, held_aois: set, buffer_km: float) -> np.ndarray:
    """True for polygons outside held AOIs that lie within buffer_km of any held AOI's hull."""
    if not held_aois:
        return np.zeros(len(gdf), bool)
    m = gdf.to_crs("EPSG:6933")
    held = m[m.aoi.isin(held_aois)]
    hulls = held.dissolve(by="aoi").convex_hull.buffer(buffer_km * 1000)
    cand = m[~m.aoi.isin(held_aois)]
    hit = gpd.sjoin(cand[["geometry"]], gpd.GeoDataFrame(geometry=hulls.values, crs=m.crs),
                    predicate="intersects", how="inner")
    drop = np.zeros(len(gdf), bool)
    drop[np.flatnonzero(gdf.index.isin(hit.index.unique()))] = True
    return drop


def revisit_pairs(gdf: gpd.GeoDataFrame, aois: set, min_overlap: float = 0.5) -> pd.DataFrame:
    """Overlapping polygons from different submissions and different label years.

    One AOI at a time: overlap is heavy in places (Wuhan's polygons cover its
    ground ~6 times over), and pairs never cross AOIs anyway.
    """
    parts = [_revisit_pairs_one(d, min_overlap)
             for _, d in gdf[gdf.aoi.isin(aois) & gdf.label_year.notna()].groupby("aoi")]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _revisit_pairs_one(sub: gpd.GeoDataFrame, min_overlap: float) -> pd.DataFrame:
    if len(sub) < 2:
        return pd.DataFrame()
    m = sub.to_crs("EPSG:6933").reset_index(drop=True)
    left, right = m.sindex.query(m.geometry, predicate="intersects")
    keep = left < right
    left, right = left[keep], right[keep]
    sid = m["submission_id"].to_numpy() if "submission_id" in m else m["annotator_id"].to_numpy()
    yrs = pd.to_numeric(m.label_year).to_numpy()
    ok = (sid[left] != sid[right]) & (yrs[left] != yrs[right])
    left, right = left[ok], right[ok]
    if len(left) == 0:
        return pd.DataFrame()
    inter = m.geometry.values[left].intersection(m.geometry.values[right]).area
    smaller = np.minimum(m.geometry.values[left].area, m.geometry.values[right].area)
    frac = inter / np.maximum(smaller, 1e-9)
    sel = frac >= min_overlap
    a, b = left[sel], right[sel]
    early = np.where(yrs[a] <= yrs[b], a, b)
    late = np.where(yrs[a] <= yrs[b], b, a)
    return pd.DataFrame({
        "aoi": m.aoi.to_numpy()[early], "pid_early": m.pid.to_numpy()[early],
        "pid_late": m.pid.to_numpy()[late], "year_early": yrs[early], "year_late": yrs[late],
        "class_early": m["class"].to_numpy()[early], "class_late": m["class"].to_numpy()[late],
        "overlap_frac": frac[sel]}).assign(changed=lambda d: d.class_early != d.class_late)


# ── driver ───────────────────────────────────────────────────────────────────

def design(wudapt: gpd.GeoDataFrame, so2sat: pd.DataFrame | None, target: pd.DataFrame,
           cfg: DesignConfig, kop: Koppen | None = None, regions: dict | None = None,
           quality: pd.DataFrame | None = None) -> dict:
    """Pure core (no file IO beyond what callers pass in): returns all tables."""
    rng = np.random.default_rng(cfg.seed)
    regions = regions or region_table()
    kop = kop or Koppen()
    wudapt = snap_rural(wudapt)
    aw = aoi_table_wudapt(wudapt, regions, kop, quality)
    aois = pd.concat([aw, so2sat], ignore_index=True) if so2sat is not None else aw
    aois["stratum"] = stratum(aois.region, aois.koppen)
    aois["block"] = make_blocks(aois, cfg.block_km)

    w = target.bbox_km2.to_numpy() if cfg.target_weight == "area" else np.ones(len(target))
    tshare = pd.Series(w).groupby(stratum(target.region, target.koppen).to_numpy()).sum()
    tshare = tshare / tshare.sum()
    d_aoi = haversine_km(aois.lat, aois.lon, aois.lat, aois.lon)
    d_tgt = haversine_km(target.lat, target.lon, aois.lat, aois.lon)
    p = Problem(aois, tshare, target.lat.to_numpy(), target.lon.to_numpy(), w, d_aoi, d_tgt, cfg)

    # Test A: blocks holding a culture-10 city. Fixed.
    is_c10 = aois.get("so2sat_city", pd.Series(index=aois.index, dtype=object)).isin(SO2SAT_TEST_CITIES)
    blocks_a = set(aois.loc[is_c10, "block"])
    # Never test on a block that holds a So2Sat training city (its labels belong to train).
    s2s_train = ((aois.source == "so2sat") & (aois.get("so2sat_role") == "train")
                 & (aois.get("so2sat_patches", 0) >= cfg.so2sat_train_min_patches))
    blocks_s2s_train = set(aois.loc[s2s_train, "block"])

    blk = aois.groupby("block").agg(
        km2=("km2", "sum"), n_aoi=("aoi", "size"),
        n_classes=("aoi", lambda s: int((aois.loc[s.index, [f"c{c}" for c in range(1, N_LCZ + 1)]]
                                         .sum() > 0.25).sum())),
        conflict=("nbr_conflict", "max"),
        names=("jrc_name", lambda s: ", ".join(map(str, s.head(4)))))
    def blocks_matching(patterns) -> set:
        """Blocks holding an AOI whose key contains any pattern (case-insensitive).

        Matches are logged by name: a bare city name is ambiguous ("lagos" also
        hits Lagos de Moreno, Mexico), so pass the full `{slug}__{SMOD_ID}` key
        when it matters.
        """
        pats = [q.lower() for q in patterns]
        if not pats:
            return set()
        for q in pats:
            hits = aois.loc[aois.aoi.str.lower().str.contains(q, regex=False), "aoi"].tolist()
            logger.info(f"pattern {q!r} matches {len(hits)} AOI(s): {hits[:8]}")
        hit = aois.aoi.str.lower().apply(lambda a: any(q in a for q in pats))
        return set(aois.loc[hit, "block"])

    quarantined = blocks_matching(cfg.quarantine)
    forced_train = blocks_matching(cfg.force_train)
    forced_test = blocks_matching(cfg.force_test) - blocks_a - blocks_s2s_train
    eligible = [b for b in blk.index
                if b not in blocks_a | blocks_s2s_train | quarantined | forced_train
                and blk.loc[b, "km2"] >= cfg.min_test_km2
                and blk.loc[b, "n_classes"] >= cfg.min_test_classes
                and not (blk.loc[b, "conflict"] > cfg.max_conflict)]
    eligible = sorted(set(eligible) | forced_test)
    logger.info(f"{len(aois):,} AOIs -> {len(blk):,} blocks | test-A blocks {len(blocks_a)} | "
                f"eligible for test-B/val {len(eligible)} (quarantined {len(quarantined)}, "
                f"forced train {len(forced_train)}, forced test {len(forced_test)})")

    fixed = np.isin(p.blk_of, list(blocks_a))
    logger.info("selecting test B")
    test_b, info_b = select(p, eligible, cfg.test_frac, fixed, rng, must=forced_test)
    fixed_v = fixed | np.isin(p.blk_of, list(test_b))
    logger.info("selecting validation")
    val, info_v = select(p, [b for b in eligible if b not in test_b], cfg.val_frac, fixed_v, rng)

    split = pd.Series("train", index=blk.index)
    split[list(blocks_a)] = "test_A"
    split[list(test_b)] = "test_B"
    split[list(val)] = "val"
    train_blocks = list(split.index[split == "train"])
    folds = assign_folds(p, train_blocks, cfg.folds, rng)
    aois["split"] = aois.block.map(split)
    aois["fold"] = aois.block.map(folds).astype("Int64")
    blk["split"] = split
    blk["fold"] = pd.Series(folds).reindex(blk.index).astype("Int64")

    held = set(aois.loc[aois.split != "train", "aoi"])
    poly = wudapt[["pid", "aoi", "class", "label_year", "ybin", "area_km2"]].copy()
    poly["split"] = poly.aoi.map(aois.set_index("aoi").split)
    poly["buffer_drop"] = buffer_drop(wudapt, held, cfg.buffer_km) & (poly.split == "train").to_numpy()
    rev = revisit_pairs(wudapt, set(aois.loc[aois.split.isin(["test_B", "val"]), "aoi"]))

    # distance matching diagnostics for the final design
    trm = (aois.split == "train").to_numpy()
    diag = {}
    for name in ("test_A", "test_B", "val"):
        m = (aois.split == name).to_numpy()
        if m.any() and trm.any():
            diag[name] = np.percentile(d_aoi[np.ix_(m, trm)].min(axis=1), [10, 50, 90]).round(0).tolist()
    diag["map_targets"] = np.percentile(p.nearest_train_targets(trm), [10, 50, 90]).round(0).tolist()

    return {"aois": aois, "blocks": blk.reset_index(), "polygons": poly, "revisits": rev,
            "target_share": tshare, "info": {"test_B": info_b, "val": info_v},
            "dist_pct": diag, "cfg": cfg}


def report(res: dict) -> str:
    a, cfg = res["aois"], res["cfg"]
    lines = ["# Global split design", "",
             f"block diameter <= {cfg.block_km:g} km, train buffer {cfg.buffer_km:g} km, "
             f"test-B {cfg.test_frac:.0%} / val {cfg.val_frac:.0%} of WUDAPT labelled area, "
             f"{cfg.folds} CV folds, seed {cfg.seed}", ""]
    g = a.groupby("split").agg(aois=("aoi", "size"), blocks=("block", "nunique"), km2=("km2", "sum"))
    lines += ["## Size", "", g.round(1).to_markdown(), ""]
    sh = (a.pivot_table(index="stratum", columns="split", values="km2", aggfunc="sum", fill_value=0))
    sh = sh / sh.sum()
    sh["map_target"] = res["target_share"]
    lines += ["## Stratum shares (region | Koppen group)", "",
              (sh.fillna(0) * 100).round(1).sort_values("map_target", ascending=False).to_markdown(), ""]
    cls = a.groupby("split")[[f"c{c}" for c in range(1, N_LCZ + 1)]].sum().T
    lines += ["## Labelled km2 per LCZ class", "", cls.round(1).to_markdown(), ""]
    yr = a.groupby("split")[[f"y{lab}" for lab in YEAR_LABELS]].sum().T
    lines += ["## Labelled km2 per label-year bin", "", yr.round(1).to_markdown(), ""]
    lines += ["## Distance to nearest training AOI (km, p10/p50/p90)", "",
              "\n".join(f"- {k}: {v}" for k, v in res["dist_pct"].items()), ""]
    lines += ["## Objective of the chosen sets", "", "```", json.dumps(res["info"], indent=2, default=float), "```", ""]
    p = res["polygons"]
    lines += [f"Buffer drops {int(p.buffer_drop.sum()):,} training polygons "
              f"({p.loc[p.buffer_drop, 'area_km2'].sum():.1f} km2).", ""]
    r = res["revisits"]
    if len(r):
        lines += [f"Revisit pairs in val/test: {len(r):,} ({int(r.changed.sum()):,} with a class change).", ""]
    held = a[a.split.isin(["test_A", "test_B", "val"])].sort_values(["split", "km2"], ascending=[True, False])
    lines += ["## Held-out AOIs", "",
              held[["split", "jrc_name", "region", "koppen", "km2", "n_annotators", "nbr_conflict"]]
              .round(2).to_markdown(index=False), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wudapt-clean", type=Path, required=True)
    ap.add_argument("--so2sat-summary", type=Path, default=REPO / "data/so2sat_city_summary.csv")
    ap.add_argument("--so2sat-class-counts", type=Path, default=REPO / "data/so2sat_city_class_counts.csv")
    ap.add_argument("--guppd-bounds", type=Path, default=REPO / "data/guppd_bounds.csv")
    ap.add_argument("--aoi-quality", type=Path, help="csv: aoi, nbr_conflict[, agree_on_overlap]")
    ap.add_argument("--koppen-tif", type=Path)
    ap.add_argument("--no-qc", action="store_true", help="keep polygons failing LCZ-Generator QC")
    ap.add_argument("--out-dir", type=Path, required=True)
    for f, v in DesignConfig().__dict__.items():
        if f == "extra":
            continue
        if isinstance(v, tuple):
            ap.add_argument(f"--{f.replace('_', '-')}", nargs="*", default=list(v))
        else:
            ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    cfg = DesignConfig(**{f: (tuple(getattr(args, f)) if isinstance(v, tuple) else getattr(args, f))
                          for f, v in DesignConfig().__dict__.items() if f != "extra"})
    kop = Koppen(args.koppen_tif)
    regions = region_table()
    wudapt = load_wudapt(args.wudapt_clean, require_qc=not args.no_qc)
    so2sat = aoi_table_so2sat(args.so2sat_summary, args.so2sat_class_counts, regions, kop)
    target = target_table(args.guppd_bounds, regions, kop)
    quality = pd.read_csv(args.aoi_quality) if args.aoi_quality else None
    res = design(wudapt, so2sat, target, cfg, kop=kop, regions=regions, quality=quality)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    res["aois"].to_parquet(out / "aoi_table.parquet", index=False)
    res["blocks"].to_parquet(out / "blocks.parquet", index=False)
    res["polygons"].to_parquet(out / "polygon_split.parquet", index=False)
    res["revisits"].to_parquet(out / "revisit_pairs.parquet", index=False)
    (out / "design_report.md").write_text(report(res))
    (out / "design_config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=list))
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
