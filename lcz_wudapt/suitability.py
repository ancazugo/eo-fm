"""Is the WUDAPT QC label set actually usable, city by city and region by region?

Producing labels is not the same as producing *useful* labels, and the answer
differs sharply between the two training pathways. Three questions decide it:

**How much supervision is there?** For patch classification that is the number of
320 m patches and how many classes reach a usable count. For segmentation it is
labelled km² and the number of 1280 m grid cells carrying enough label to train
on. These diverge far more than they sound like they should: roughly 60% of
QC-passing polygons cannot contain a 320 m square, so the patch arm discards
them outright while segmentation keeps every interior pixel. Buenos Aires ends up
with **4 patches but 23 km² across 15 classes**.

**Can it be trusted?** Where So2Sat exists, the class agreement on overlapping
ground is a direct per-city trust score, and it has a cliff in it — Moscow 1.00
and Munich 0.98 against Tehran 0.46 and Mumbai 0.62. Where So2Sat does not exist
(the ~1,100 AOIs that are the whole point), there is no ground truth to compare
against, so ``nbr_conflict`` — the fraction of patches within 100 m of a
differently-labelled polygon — stands in for it. It was validated against So2Sat
agreement at Spearman +0.929 on the eight audit cities, which is what licenses
using it as a proxy elsewhere.

**Would using it leak?** In the 51 So2Sat cities, WUDAPT patches physically
overlap So2Sat patches, and the So2Sat culture-10 carry the benchmark every
headline number is measured on. :func:`city_suitability` reports the overlapping
count so it can be excluded, rather than leaving the leak to be discovered after
a run.

Everything here is read-only over artefacts written by :mod:`lcz_wudapt.qc` and
:mod:`lcz_wudapt.so2sat_shape`. See ``docs/wudapt_label_suitability.md`` for the
measured tables and the resulting recommendations.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio import features

from .config import WudaptConfig

__all__ = [
    "REGION_GROUPS",
    "TIER1_CULTURE",
    "TIER2_SPARSE",
    "city_suitability",
    "labelled_km2",
    "region_suitability",
    "resolve_city_aois",
    "tile_footprint",
]

# The So2Sat culture-10: the only cities carrying validation/testing patches, and
# the ones every kappa in docs/global_lcz_campaign_2026-07.md is measured on.
TIER1_CULTURE = (
    "Guangzhou", "Tehran", "Jakarta", "Mumbai", "Sydney",
    "Moscow", "Munich", "Santiago", "San_Jose", "Nairobi",
)

# So2Sat cities whose own labels are degenerate — every one is single-class, and
# Salvador contributes exactly one patch. These are where WUDAPT has the most to
# add per unit of effort.
TIER2_SPARSE = ("Chicago", "Lima", "Bogota", "Caracas", "Buenos_Aires", "Salvador")

# West Africa is an ISO set rather than one of splits.ISO_TO_REGION's coarse
# buckets: it sits inside "Africa" but behaves nothing like North or Southern
# Africa in this dataset, carrying the highest LCZ 7 share and the lowest
# conflict rate anywhere.
WEST_AFRICA_ISO = frozenset(
    "NGA GHA SEN CIV MLI BFA BEN TGO GIN SLE LBR GMB GNB MRT NER CPV".split()
)

REGION_GROUPS: dict[str, object] = {
    "Central America": lambda df: df["region"] == "America-Central",
    "West Africa": lambda df: df["iso"].isin(WEST_AFRICA_ISO),
    "India": lambda df: df["iso"] == "IND",
    "Southeast Asia": lambda df: df["region"] == "Asia-Southeast",
}

# Tessera and GeoTessera name tiles grid_{lon}_{lat} on a 0.1 degree lattice with
# lon/lat at the CENTRE of the cell, so a request is counted in centres.
_TILE_DEG = 0.1


def labelled_km2(tif: Path) -> tuple[float, int]:
    """Labelled ground area and class count in a segmentation label raster.

    The rasters are EPSG:4326, so pixel area varies with latitude; it is
    corrected at the raster's centre latitude. Exact enough to compare cities —
    these numbers size an experiment, they do not go in a results table.
    """
    with rasterio.open(tif) as r:
        arr = r.read(1)
        centre_lat = float(r.transform.f - r.height * abs(r.transform.e) / 2)
        px_km2 = (
            abs(r.transform.a * r.transform.e)
            * (111_320.0**2)
            * np.cos(np.radians(centre_lat))
            / 1e6
        )
    labelled = arr > 0
    return float(labelled.sum() * px_km2), int(len(np.unique(arr[labelled])))


def _cells_with_label(tif: Path, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Labelled fraction per grid cell, in ONE rasterisation pass.

    The obvious implementation masks the raster per cell, which is O(cells) reads
    and takes minutes on a city like Guangzhou. Burning the cell index once and
    using ``np.bincount`` is the same answer in one pass.
    """
    with rasterio.open(tif) as r:
        lab = r.read(1)
        transform, shape, crs = r.transform, lab.shape, r.crs

    g = grid.to_crs(crs).reset_index(drop=True)
    idx = features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(g.geometry)),
        out_shape=shape, transform=transform, fill=0, dtype="int32",
    )
    inside = idx > 0
    cell, val = idx[inside], lab[inside]
    total = np.bincount(cell, minlength=len(g) + 1)[1:]
    labelled = np.bincount(cell[val > 0], minlength=len(g) + 1)[1:]
    frac = np.divide(labelled, total, out=np.zeros(len(g), float), where=total > 0)
    return pd.DataFrame({"split": g.get("split", pd.Series(["?"] * len(g))).values,
                         "frac": frac, "labelled_px": labelled})


def tile_footprint(gdf: gpd.GeoDataFrame, deg: float = _TILE_DEG) -> set[tuple[float, float]]:
    """0.1-degree tile centres covering ``gdf``, for sizing an embedding request."""
    out: set[tuple[float, float]] = set()
    for b in gdf.geometry.bounds.itertuples():
        lons = np.arange(np.floor(b.minx / deg) * deg, b.maxx + 1e-9, deg)
        lats = np.arange(np.floor(b.miny / deg) * deg, b.maxy + 1e-9, deg)
        for lon in lons:
            for lat in lats:
                out.add((round(lon + deg / 2, 2), round(lat + deg / 2, 2)))
    return out


def resolve_city_aois(
    config: WudaptConfig,
    cities: tuple[str, ...],
    *,
    cities_root: Path | None = None,
) -> dict[str, str]:
    """Map So2Sat city directory names to WUDAPT AOI keys.

    Matched on the slug before the ``__SMOD_ID`` suffix rather than hard-coded,
    so the mapping survives a change of GUPPD release. Names are compared with
    underscores folded to spaces because So2Sat directories use underscores
    (``San_Jose``, ``Buenos_Aires``) and ``JRC_NAME_MAIN`` uses spaces.
    """
    root = Path(cities_root) if cities_root else Path(config.gpkg_path).parent / "cities"
    if not root.exists():
        return {}
    available = [d.name for d in root.iterdir() if d.is_dir()]

    def _key(name: str) -> str:
        return name.replace("_", " ").strip().lower()

    by_slug: dict[str, list[str]] = {}
    for aoi in available:
        by_slug.setdefault(_key(aoi.rsplit("__", 1)[0]), []).append(aoi)

    out: dict[str, str] = {}
    for city in cities:
        hits = by_slug.get(_key(city), [])
        if not hits:  # fall back to a prefix match for name variants
            hits = [a for k, v in by_slug.items() if k.startswith(_key(city)[:5]) for a in v]
        if len(hits) == 1:
            out[city] = hits[0]
        elif hits:
            logger.warning(f"{city}: ambiguous AOI match {sorted(hits)}; taking the first")
            out[city] = sorted(hits)[0]
        else:
            logger.warning(f"{city}: no WUDAPT AOI found")
    return out


def city_suitability(
    config: WudaptConfig,
    cities: dict[str, str],
    *,
    cities_root: Path | None = None,
    so2sat_cities: Path | None = None,
) -> pd.DataFrame:
    """Per-city capacity, trust and leakage, for both pathways.

    ``cities`` maps a So2Sat city directory name to its WUDAPT AOI key. Columns
    prefixed ``w_`` are WUDAPT, ``s_`` are So2Sat, ``seg_`` are the segmentation
    raster.

    ``agree_on_overlap`` is the trust score and ``w_clean`` the leakage-safe
    patch count — the two numbers that decide whether a city is usable and how
    much of it you may actually train on.
    """
    root = Path(cities_root) if cities_root else Path(config.gpkg_path).parent / "cities"
    so2sat = Path(so2sat_cities) if so2sat_cities else (
        Path(config.labels.so2sat_dir) / config.labels.cities_subdir
    )

    rows = []
    for city, aoi in cities.items():
        wdir, sdir = root / aoi, so2sat / city
        gpkg, tif = wdir / f"patches_reference_{aoi}.gpkg", wdir / f"patches_reference_{aoi}.tif"
        rec: dict = {"city": city, "aoi": aoi, "w_patches": 0}
        if not gpkg.exists():
            rows.append(rec)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = gpd.read_file(gpkg)
        counts = w["LCZ_class"].value_counts()
        rec.update(
            w_patches=len(w),
            w_classes=int(w["LCZ_class"].nunique()),
            median_per_class=int(counts.median()),
            classes_ge5=int((counts >= 5).sum()),
            mean_weight=round(float(w["weight"].mean()), 3),
            nbr_conflict=round(float(w["nbr_conflict"].mean()), 3),
        )

        s_gpkg = sdir / f"patches_reference_{city}.gpkg"
        if s_gpkg.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = gpd.read_file(s_gpkg)
            utm = w.estimate_utm_crs()
            hit = gpd.sjoin(
                w[["patch_id", "LCZ_class", "geometry"]].to_crs(utm),
                s[["patch_id", "dataset", "LCZ_class", "geometry"]]
                 .rename(columns={"patch_id": "s_id", "LCZ_class": "s_lcz"}).to_crs(utm),
                predicate="intersects", how="inner",
            )
            overlapping = int(hit["patch_id"].nunique())
            clean = w[~w["patch_id"].isin(hit["patch_id"])]
            rec.update(
                s_patches=len(s), s_classes=int(s["LCZ_class"].nunique()),
                w_overlapping=overlapping,
                w_clean=len(w) - overlapping,
                clean_classes=int(clean["LCZ_class"].nunique()),
                agree_on_overlap=(
                    round(float((hit["LCZ_class"] == hit["s_lcz"]).mean()), 3) if len(hit) else np.nan
                ),
                overlap_pairs=len(hit),
            )

        if tif.exists():
            km2, ncls = labelled_km2(tif)
            rec.update(seg_km2=round(km2, 2), seg_classes=ncls)
            grid = sdir / f"{city}_grid.gpkg"
            if grid.exists():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    per = _cells_with_label(tif, gpd.read_file(grid))
                train = per[per["split"] == "train"]
                rec.update(
                    cells_train_ge1=int((train["frac"] >= 0.01).sum()),
                    cells_train_ge5=int((train["frac"] >= 0.05).sum()),
                    cells_test=int((per["split"] == "test").sum()),
                )
        rows.append(rec)
    return pd.DataFrame(rows)


def region_suitability(
    config: WudaptConfig,
    *,
    patches_gpkg: Path | None = None,
    cities_root: Path | None = None,
    groups: dict | None = None,
) -> pd.DataFrame:
    """Capacity per region for the AOIs So2Sat does not cover at all.

    No trust score is available here — there is no ground truth to agree with —
    so ``nbr_conflict`` carries that weight, on the strength of its +0.929
    correlation with So2Sat agreement measured on the audit cities.
    """
    root = Path(cities_root) if cities_root else Path(config.gpkg_path).parent / "cities"
    gpkg = Path(patches_gpkg) if patches_gpkg else root.parent / "patches_wudapt_rxr.gpkg"
    groups = groups or REGION_GROUPS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patches = gpd.read_file(gpkg)

    cache = Path(config.cache_dir)
    index = pd.read_parquet(cache / f"aoi_index_{config.ingest_hash}.parquet")
    if "iso" not in patches.columns:
        patches = patches.merge(index[["aoi", "iso"]], on="aoi", how="left")

    # The segmentation universe is LARGER than the patch universe, and the gap is
    # the whole point: an AOI whose polygons all fail the 320 m containment test
    # yields zero patches while still producing a perfectly good label raster.
    # Counting rasters only over AOIs that have patches undercounts segmentation
    # capacity — measured on India, 92 AOIs with patches against 132 with a
    # raster. The AOI roster therefore comes from the index, not from the patches.
    universe = index[["aoi", "iso"]].copy()
    if "region" in patches.columns:
        universe = universe.merge(
            patches[["aoi", "region"]].drop_duplicates(), on="aoi", how="left"
        )
    if "region" not in universe.columns or universe["region"].isna().any():
        from .splits import region_for
        fallback = universe["iso"].map(region_for)
        universe["region"] = (
            universe["region"].fillna(fallback) if "region" in universe.columns else fallback
        )

    rows = []
    for name, sel in groups.items():
        sub = patches[sel(patches)]
        region_aois = set(universe[sel(universe)]["aoi"])
        if not len(sub) and not region_aois:
            rows.append({"region": name, "patches": 0})
            continue
        counts = sub["LCZ_class"].value_counts()
        per_aoi = sub.groupby("aoi")

        km2, seg_classes, with_raster, big = 0.0, set(), 0, 0
        for aoi in sorted(region_aois):
            tif = root / aoi / f"patches_reference_{aoi}.tif"
            if not tif.exists():
                continue
            a, n = labelled_km2(tif)
            km2 += a
            with_raster += 1
            big += int(a >= 5.0)
            if n:
                seg_classes.add(n)

        rows.append({
            "region": name,
            "aois_patches": int(sub["aoi"].nunique()),
            "aois_raster": with_raster,
            "countries": int(sub["iso"].nunique()),
            "patches": len(sub),
            "classes": int(sub["LCZ_class"].nunique()),
            "classes_ge50": int((counts >= 50).sum()),
            "aois_ge50_patches": int((per_aoi.size() >= 50).sum()),
            "aois_ge8_classes": int((per_aoi["LCZ_class"].nunique() >= 8).sum()),
            "mean_weight": round(float(sub["weight"].mean()), 3),
            "nbr_conflict": round(float(sub["nbr_conflict"].mean()), 3),
            "seg_km2": round(km2, 1),
            "aois_ge5km2": big,
            "tiles_0p1deg": len(tile_footprint(sub)),
            **{f"n_{s}": int((sub["wudapt_split"] == s).sum())
               for s in ("train", "val", "test") if "wudapt_split" in sub},
        })
    return pd.DataFrame(rows)


def class_mix(
    config: WudaptConfig,
    *,
    patches_gpkg: Path | None = None,
    groups: dict | None = None,
) -> pd.DataFrame:
    """Per-region class shares against So2Sat's, as percentages.

    The point of the whole exercise: which morphologies these regions supply that
    So2Sat does not.
    """
    root = Path(config.gpkg_path).parent / "cities"
    gpkg = Path(patches_gpkg) if patches_gpkg else root.parent / "patches_wudapt_rxr.gpkg"
    groups = groups or REGION_GROUPS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patches = gpd.read_file(gpkg)
    index = pd.read_parquet(Path(config.cache_dir) / f"aoi_index_{config.ingest_hash}.parquet")
    if "iso" not in patches.columns:
        patches = patches.merge(index[["aoi", "iso"]], on="aoi", how="left")

    out = {}
    for name, sel in groups.items():
        sub = patches[sel(patches)]
        if len(sub):
            out[name] = (sub["LCZ_class"].value_counts(normalize=True) * 100).round(1)

    table = pd.DataFrame(out).reindex(range(1, 18)).fillna(0.0)
    ref = Path(config.labels.so2sat_dir) / "patches_reference_rxr.gpkg"
    if ref.exists():
        import pyogrio
        s = pyogrio.read_dataframe(ref, columns=["LCZ_class"], read_geometry=False)
        table["So2Sat"] = (s["LCZ_class"].value_counts(normalize=True) * 100).round(1)
    return table.fillna(0.0)
