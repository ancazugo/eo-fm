"""H1 — ingest and clean the LCZ-Generator submission database.

The shipped gpkg needs real cleaning before any of it can be trusted. Each step
below fixes a defect verified in the 2024-10-01 release (630,311 polygons):

* 4,885 invalid geometries, and every geometry is ``Polygon Z``.
* ``class`` carries 580 rows of 18 and 53 of 19, outside the LCZ 1-17 scheme.
* ``qc_step1/2/3`` are mixed-encoded — ``True``/``False`` *and* ``T``/``F``
  strings. Filtering the raw column silently misclassifies ~34k rows.
* ``area`` is Web Mercator km², inflated by 1/cos²(lat) (median ratio 1.35 vs
  geodesic). It is never read; area is recomputed on an equal-area CRS.
* ``city`` is free-text and multilingual ("北京", "wuhan", "guangzhoushi", "..").
  It is never read either; AOI assignment is purely spatial.

AOI keys are SMOD_ID-qualified because ``JRC_NAME_MAIN`` is not unique in
``data/guppd_bounds.csv`` (Aurangabad ×3, León ×3, San Cristóbal ×3), and the
So2Sat bounds gpkg disagrees with its own CSV on at least one name (``东营区``
vs ``Dongying``). Joining on name would silently merge distinct cities.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from loguru import logger

from .config import WudaptConfig

# LCZ 1-17 (built 1-10, natural 11-17 == A-G). See src/utils/constants.py::lcz_dict.
N_LCZ = 17
BUILT_CLASSES = frozenset(range(1, 11))

# Equal-area CRS for area computation: Lambert Cylindrical Equal Area on the
# WGS84 ellipsoid. Exact (not spherical-approximate) and valid to +/-86 deg;
# WUDAPT spans -43.1 to +69.1, comfortably inside. Chosen over a per-AOI local
# UTM so that the rural partition — which has no AOI — is measured identically.
EQUAL_AREA_CRS = "EPSG:6933"

# The QC columns ship as strings with two different encodings for the same value.
_TRUE_TOKENS = frozenset({"true", "t", "1", "yes", "y"})
_FALSE_TOKENS = frozenset({"false", "f", "0", "no", "n"})

RURAL_AOI = "_rural"

__all__ = [
    "aoi_key",
    "assign_aoi",
    "clean",
    "ingest",
    "load_raw",
    "normalise_qc",
    "resolve_annotators",
    "slug",
]


def slug(value: str) -> str:
    """Filesystem- and join-safe token from a possibly non-ASCII place name.

    Keeps unicode letters (so ``东营区`` does not collapse to an empty string and
    collide with every other name that transliterates away) but strips path
    separators, whitespace runs and punctuation.
    """
    s = unicodedata.normalize("NFKC", str(value)).strip()
    s = re.sub(r"[\s/\\]+", "_", s)
    s = re.sub(r"[^\w.-]", "", s, flags=re.UNICODE)
    return s.strip("_.") or "unnamed"


def aoi_key(jrc_name: str, smod_id: str) -> str:
    """Canonical AOI key: ``{slug(name)}__{SMOD_ID}``.

    SMOD_ID-qualified because JRC_NAME_MAIN is not unique — see module docstring.
    """
    return f"{slug(jrc_name)}__{smod_id}"


def load_raw(config: WudaptConfig, columns: list[str] | None = None) -> gpd.GeoDataFrame:
    """Read the submission gpkg.

    ``force_2d`` because the geometries are ``Polygon Z`` and the Z coordinate is
    meaningless here; carrying it through doubles memory and breaks some shapely
    set operations.
    """
    path = Path(config.gpkg_path)
    if not path.exists():
        raise FileNotFoundError(f"WUDAPT gpkg not found: {path}")
    gdf = pyogrio.read_dataframe(path, layer=config.layer, columns=columns, force_2d=True)
    logger.info(f"read {len(gdf):,} polygons from {path.name} [{config.layer}]")
    return gdf


def normalise_qc(values: pd.Series) -> pd.Series:
    """Mixed ``True``/``T`` (and ``False``/``F``) strings -> nullable boolean.

    Returns ``pd.NA`` for anything unrecognised rather than guessing, so an
    unexpected token surfaces as a NA count in the ingest log instead of being
    silently read as False.
    """
    if values.dtype == "bool":
        return values.astype("boolean")
    lowered = values.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=values.index, dtype="boolean")
    out[lowered.isin(_TRUE_TOKENS)] = True
    out[lowered.isin(_FALSE_TOKENS)] = False
    return out


def _equal_area_km2(geometry: gpd.GeoSeries) -> np.ndarray:
    """Polygon areas in km² on an equal-area CRS (vectorised)."""
    return geometry.to_crs(EQUAL_AREA_CRS).area.to_numpy() / 1e6


def _annotator_id(df: pd.DataFrame) -> pd.Series:
    """Stable per-author key.

    ``lastname|firstname`` where available, else the ``reference``/``cite_as``
    study string. 106,793 polygons (16.8%) have neither and get a provisional
    ``blank:{submission_id}`` key plus ``annotator_src == "blank"``;
    :func:`resolve_annotators` then applies ``config.blank_name_policy`` once the
    AOI is known. Returns ``(annotator_id, annotator_src)``.
    """
    last = df["lastname"].fillna("").astype(str).str.strip().str.lower()
    first = df["firstname"].fillna("").astype(str).str.strip().str.lower()
    named = (last != "") | (first != "")

    out = pd.Series("", index=df.index, dtype="object")
    src = pd.Series("", index=df.index, dtype="object")
    out[named] = "name:" + last[named].map(slug) + "|" + first[named].map(slug)
    src[named] = "name"

    study = df["reference"].fillna(df.get("cite_as", "")).fillna("").astype(str).str.strip()
    has_study = (~named) & (study != "")
    out[has_study] = "study:" + study[has_study].map(slug)
    src[has_study] = "study"

    orphan = out == ""
    out[orphan] = "blank:" + df.loc[orphan, "submission_id"].astype(str)
    src[orphan] = "blank"
    return out, src


def clean(gdf: gpd.GeoDataFrame, config: WudaptConfig) -> gpd.GeoDataFrame:
    """Apply every H1 correction. Order matters — see inline notes."""
    n0 = len(gdf)
    df = gdf.copy()

    # 1. Class domain. A nonzero count here in a future release is a schema alarm,
    #    so it is logged rather than dropped quietly.
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    bad_class = ~df["class"].between(1, N_LCZ)
    if int(bad_class.sum()):
        counts = df.loc[bad_class, "class"].value_counts().to_dict()
        logger.warning(f"dropping {int(bad_class.sum()):,} polygons outside LCZ 1-{N_LCZ}: {counts}")
    df = df[~bad_class].copy()
    df["class"] = df["class"].astype("int16")

    # 2. Geometry repair, then explode: make_valid can return GeometryCollections,
    #    and downstream rasterisation and area both need single polygons.
    invalid = ~df.geometry.is_valid
    if int(invalid.sum()):
        logger.info(f"repairing {int(invalid.sum()):,} invalid geometries")
        df.loc[invalid, "geometry"] = shapely.make_valid(df.loc[invalid, "geometry"].values)
    df = df.explode(index_parts=False, ignore_index=True)
    non_poly = df.geometry.geom_type != "Polygon"
    if int(non_poly.sum()):
        logger.info(f"dropping {int(non_poly.sum()):,} non-polygonal fragments from make_valid")
        df = df[~non_poly].copy()
    df = df[~df.geometry.is_empty].reset_index(drop=True)

    # 3. QC flags -> nullable boolean.
    for col in ("qc_step1", "qc_step2", "qc_step3"):
        df[col] = normalise_qc(df[col])
        n_na = int(df[col].isna().sum())
        if n_na:
            logger.warning(f"{col}: {n_na:,} unrecognised values -> NA")

    # 4. Area. The shipped column is Web Mercator km² (median 1.35x geodesic) and
    #    is preserved only so a reader can see why it is not used.
    df = df.rename(columns={"area": "area_km2_mercator_raw"})
    df["area_km2"] = _equal_area_km2(df.geometry)

    # 5. Label epoch. representative_date is the imagery the annotator worked
    #    from — the epoch the label actually describes — so it takes precedence
    #    over the (later) submission date.
    rep = pd.to_datetime(df["representative_date"], errors="coerce", utc=True)
    sub = pd.to_datetime(df["submission_date"], errors="coerce", utc=True)

    def _sane(years: pd.Series) -> pd.Series:
        # 2029, 2117 and 2323 all appear. Treat typos as missing rather than
        # clamping them, so they fall through to the submission date instead of
        # silently becoming a real-looking epoch.
        ok = years.between(config.min_label_year, config.max_label_year)
        return years.where(ok).astype("Int16")

    df["rep_year"] = _sane(rep.dt.year)
    df["sub_year"] = _sane(sub.dt.year)
    n_bad = int(rep.dt.year.notna().sum() - df["rep_year"].notna().sum())
    if n_bad:
        logger.warning(f"{n_bad:,} representative_date years outside "
                       f"[{config.min_label_year}, {config.max_label_year}] -> NA")
    df["label_year"] = df["rep_year"].fillna(df["sub_year"]).astype("Int16")

    # 6. Annotator identity. This is the unit of the vote, not submission_id:
    #    8,827 submissions collapse to ~1,498 authors (Wuhan 388 -> 41).
    df["annotator_id"], df["annotator_src"] = _annotator_id(df)

    # 7. `city` is multilingual free text and is never used for grouping.
    df = df.rename(columns={"city": "city_freetext"})

    logger.info(
        f"cleaned {n0:,} -> {len(df):,} polygons | "
        f"{df.annotator_id.nunique():,} annotators, {df.submission_id.nunique():,} submissions"
    )
    return df


def _load_bounds(config: WudaptConfig) -> gpd.GeoDataFrame:
    df = pd.read_csv(config.bounds_csv)
    geom = shapely.box(df.minx.values, df.miny.values, df.maxx.values, df.maxy.values)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    gdf["aoi"] = [aoi_key(n, s) for n, s in zip(gdf.JRC_NAME_MAIN, gdf.SMOD_ID)]
    if gdf["aoi"].duplicated().any():
        dupes = gdf.loc[gdf["aoi"].duplicated(keep=False), "aoi"].unique()[:5]
        raise ValueError(f"AOI keys are not unique in {config.bounds_csv}: {list(dupes)}")
    gdf["_bbox_area"] = gdf.to_crs(EQUAL_AREA_CRS).area.to_numpy()
    return gdf


def assign_aoi(gdf: gpd.GeoDataFrame, config: WudaptConfig) -> gpd.GeoDataFrame:
    """Attach ``aoi`` (and GUPPD metadata) by spatial join against GUPPD bboxes.

    Polygons in more than one bbox go to the **smaller** bbox — GUPPD boxes nest
    around conurbations, so the smaller box is the more specific city. Ties break
    on SMOD_ID so the assignment is deterministic across runs.

    Polygons inside no bbox are **retained** in the ``_rural`` partition, not
    discarded. They are 27.4% of the database and 61.8% natural classes (LCZ
    11-17) against 35.1% inside — dropping them would systematically strip the
    very classes So2Sat already under-samples.
    """
    bounds = _load_bounds(config)
    pts = gpd.GeoDataFrame(
        {"_row": np.arange(len(gdf))},
        geometry=gdf.geometry.representative_point(),
        crs=gdf.crs,
    )
    cols = ["aoi", "JRC_NAME_MAIN", "SMOD_ID", "ISO", "CNTRY_NAME", "_bbox_area", "geometry"]
    hit = gpd.sjoin(pts, bounds[cols], predicate="within", how="inner")

    # Deterministic pick among overlapping bboxes.
    hit = hit.sort_values(["_row", "_bbox_area", "SMOD_ID"]).drop_duplicates("_row", keep="first")

    out = gdf.copy()
    meta = ["aoi", "JRC_NAME_MAIN", "SMOD_ID", "ISO", "CNTRY_NAME"]
    for col in meta:
        out[col] = pd.Series(hit.set_index("_row")[col], index=pd.RangeIndex(len(out)))
    out["aoi"] = out["aoi"].fillna(RURAL_AOI)
    out = out.rename(columns={"JRC_NAME_MAIN": "jrc_name", "SMOD_ID": "smod_id",
                              "ISO": "iso", "CNTRY_NAME": "cntry_name"})

    n_rural = int((out["aoi"] == RURAL_AOI).sum())
    n_urban = len(out) - n_rural
    nat = out["class"].isin(range(11, N_LCZ + 1))
    logger.info(
        f"AOI assignment: {n_urban:,} urban across {out.loc[out.aoi != RURAL_AOI, 'aoi'].nunique():,} "
        f"areas | {n_rural:,} rural ({100 * n_rural / len(out):.1f}%), "
        f"natural-class share rural {100 * nat[out.aoi == RURAL_AOI].mean():.1f}% "
        f"vs urban {100 * nat[out.aoi != RURAL_AOI].mean():.1f}%"
    )
    return out


def resolve_annotators(gdf: gpd.GeoDataFrame, config: WudaptConfig) -> gpd.GeoDataFrame:
    """Apply ``blank_name_policy`` to the rows that shipped with no author name.

    Must run after :func:`assign_aoi`, because the conservative policy collapses
    per AOI. Under ``collapse_per_aoi`` all blank-name rows in one AOI become a
    single annotator; under ``per_submission`` each submission is its own. The
    former can merge two real people (costing coverage), the latter can invent
    independent annotators that do not exist (corrupting confidence via n_eff),
    so the former is the default.
    """
    out = gdf.copy()
    blank = out["annotator_src"] == "blank"
    if not bool(blank.any()):
        return out

    before = out.loc[blank, "annotator_id"].nunique()
    if config.blank_name_policy == "collapse_per_aoi":
        out.loc[blank, "annotator_id"] = "blank:" + out.loc[blank, "aoi"].astype(str)
    after = out.loc[blank, "annotator_id"].nunique()
    logger.info(
        f"blank-name policy '{config.blank_name_policy}': "
        f"{int(blank.sum()):,} polygons, {before:,} -> {after:,} annotator keys "
        f"(named authors: {out.loc[~blank, 'annotator_id'].nunique():,})"
    )
    return out


def ingest(config: WudaptConfig, *, force: bool = False) -> tuple[Path, Path]:
    """Run H1 end to end and cache the results.

    Writes two artefacts under ``cache_dir``, both keyed on ``ingest_hash`` so
    that tuning a quality or consensus threshold never re-reads all 630k
    polygons:

    * ``wudapt_clean_{ingest_hash}.parquet`` — one row per cleaned polygon.
    * ``aoi_index_{ingest_hash}.parquet`` — one row per AOI touched, carrying
      ``aoi, jrc_name, iso, cntry_name, smod_id`` for split/region resolution.

    Returns ``(clean_path, aoi_index_path)``.
    """
    cache = Path(config.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    clean_path = cache / f"wudapt_clean_{config.ingest_hash}.parquet"
    index_path = cache / f"aoi_index_{config.ingest_hash}.parquet"

    if clean_path.exists() and index_path.exists() and not force:
        logger.info(f"ingest cache hit: {clean_path.name}")
        return clean_path, index_path

    gdf = resolve_annotators(assign_aoi(clean(load_raw(config), config), config), config)
    gdf.to_parquet(clean_path)

    urban = gdf[gdf["aoi"] != RURAL_AOI]
    index = (
        urban.groupby("aoi")
        .agg(
            jrc_name=("jrc_name", "first"),
            smod_id=("smod_id", "first"),
            iso=("iso", "first"),
            cntry_name=("cntry_name", "first"),
            n_polys=("class", "size"),
            n_annotators=("annotator_id", "nunique"),
            n_submissions=("submission_id", "nunique"),
            n_classes=("class", "nunique"),
        )
        .reset_index()
        .sort_values("n_polys", ascending=False)
    )
    index.to_parquet(index_path, index=False)

    logger.info(f"wrote {clean_path.name} ({len(gdf):,} rows) and {index_path.name} ({len(index):,} AOIs)")
    return clean_path, index_path
