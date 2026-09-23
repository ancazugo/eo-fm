"""City-disjoint WUDAPT train/val/test splits that leave So2Sat's frozen.

The adaptation protocol is: take a model trained on So2Sat, then give it more
geographic context from WUDAPT. That imposes two non-negotiable constraints.

**The So2Sat culture-10 split is frozen.** Every headline number in
``docs/global_lcz_campaign_2026-07.md`` (single-model kappa 0.6497, LOCO-weighted
ensemble 0.6871) is measured on those ten held-out cities. WUDAPT covers them
heavily -- 29,376 polygons in Guangzhou, 6,547 in Tehran -- so training on WUDAPT
without care would contaminate the very benchmark the adaptation is judged
against. :func:`assign_splits` therefore **forces all ten into WUDAPT-test**,
never train or val, and :func:`assert_split_integrity` fails the run rather than
letting it through quietly.

**Splits are city-disjoint, and stratified by region.** Splitting patches rather
than cities leaks through spatial autocorrelation, which the campaign already
measured at roughly +22 kappa on the ``--orig-test`` mode. And WUDAPT is
extremely unbalanced by country -- China alone is a large minority of all
polygons -- so an unstratified city split would put most of the test set on one
continent.

The output is a single ``wudapt_split`` column, which the existing
``--global-split --split-col`` path in ``src/datasets/so2sat.py`` consumes with
no new split code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .config import WudaptConfig
from .leakage import SO2SAT_TEST_CITIES

__all__ = [
    "ISO_TO_REGION",
    "assert_split_integrity",
    "assign_splits",
    "region_for",
]

# ISO3 -> coarse region, covering all 173 country codes present in
# data/guppd_bounds.csv. Keyed on ISO rather than CNTRY_NAME because
# lcz_train.splits.city_regions() already demonstrated that name-keyed lookups
# raise on unmapped countries the moment the roster grows beyond So2Sat's 51
# cities -- and here it grows to 1,251 areas across 173 countries.
ISO_TO_REGION: dict[str, str] = {}


def _add(region: str, codes: str) -> None:
    for c in codes.split():
        ISO_TO_REGION[c] = region


_add("Africa", """
AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM DJI DZA EGY ERI ESH ETH GAB GHA GIN
GMB GNB GNQ KEN LBR LBY LSO MAR MDG MLI MOZ MRT MUS MWI NAM NER NGA RWA SDN SEN
SLE SOM SSD STP SWZ TCD TGO TUN TZA UGA ZAF ZMB ZWE
""")
_add("Asia-East", "CHN JPN KOR PRK MNG TWN HKG MAC")
_add("Asia-South", "AFG BGD BTN IND LKA MDV NPL PAK IRN")
_add("Asia-Southeast", "IDN KHM LAO MMR MYS PHL SGP THA TLS VNM BRN")
_add("Asia-West", """
ARE ARM AZE BHR CYP GEO IRQ ISR JOR KWT LBN OMN PSE QAT SAU SYR TUR YEM
KAZ KGZ TJK TKM UZB
""")
_add("Europe", """
ALB AUT BEL BGR BIH BLR CHE CZE DEU DNK ESP EST FIN FRA GBR GRC HRV HUN IRL ISL
ITA LTU LUX LVA MDA MKD MLT MNE NLD NOR POL PRT ROU RUS SRB SVK SVN SWE UKR XKO
""")
_add("America-North", "CAN USA MEX")
_add("America-Central", """
BHS BRB CRI CUB CUW DOM GTM HND HTI JAM NIC PAN PRI SLV TTO
""")
_add("America-South", "ARG BOL BRA CHL COL ECU GUY PER PRY SUR URY VEN")
_add("Oceania", "AUS FJI NZL PNG SLB VUT WSM")


def region_for(iso: str | None) -> str:
    """Region for an ISO3 code. Unknown codes get their own bucket, loudly.

    Deliberately does not raise: an unmapped country should not stop a 1,251-AOI
    build, but it must not silently join a real region either, because that would
    quietly break the stratification it was added to preserve.
    """
    if iso is None or (isinstance(iso, float) and np.isnan(iso)):
        return "Unknown"
    return ISO_TO_REGION.get(str(iso).upper(), "Unknown")


def _stable_fraction(key: str, salt: str) -> float:
    """Deterministic uniform in [0, 1) from a city key.

    Hash-based rather than RNG-shuffled so that adding an AOI to the roster does
    not reshuffle the split of every other AOI -- otherwise a re-run after new
    Tessera tiles land silently moves cities between train and test, and no two
    experiments are comparable.
    """
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def assign_splits(
    aoi_index: pd.DataFrame,
    *,
    so2sat_aoi_map: dict[str, str] | None = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    salt: str = "wudapt-v1",
) -> pd.DataFrame:
    """Assign ``wudapt_split`` per AOI, region-stratified and city-disjoint.

    ``aoi_index`` is ``ingest.ingest()``'s second artefact: one row per AOI with
    ``aoi, jrc_name, iso, n_polys``. ``so2sat_aoi_map`` maps AOI keys to So2Sat
    city directory names (see :func:`lcz_wudapt.audit.so2sat_aoi_map`); it is how
    the culture-10 are recognised, since GUPPD names and So2Sat directory names
    disagree (``东营区`` vs ``Dongying``).

    Returns ``aoi_index`` plus ``region``, ``wudapt_split`` and
    ``forced_test`` columns.
    """
    out = aoi_index.copy()
    out["region"] = [region_for(i) for i in out["iso"]]

    # Which AOIs are held-out So2Sat cities? Match on the So2Sat directory name
    # where we have the mapping, and fall back to the GUPPD name.
    mapping = so2sat_aoi_map or {}
    culture = {c.replace(" ", "_") for c in SO2SAT_TEST_CITIES} | set(SO2SAT_TEST_CITIES)
    forced = []
    for rec in out.itertuples():
        so2sat_name = mapping.get(rec.aoi)
        name = str(getattr(rec, "jrc_name", "") or "")
        forced.append(bool((so2sat_name in culture) if so2sat_name else (name in culture)))
    out["forced_test"] = forced

    frac = np.array([_stable_fraction(a, salt) for a in out["aoi"]])
    split = np.full(len(out), "train", dtype=object)

    # Stratify within region so no continent lands wholly in one split.
    for region, idx in out.groupby("region").groups.items():
        pos = out.index.get_indexer(list(idx))
        f = frac[pos]
        order = np.argsort(f)
        n = len(pos)
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        sel = pos[order]
        split[sel[:n_test]] = "test"
        split[sel[n_test:n_test + n_val]] = "val"

    out["wudapt_split"] = split
    # The forced assignment wins over the stratified draw, always.
    out.loc[out["forced_test"], "wudapt_split"] = "test"

    counts = out["wudapt_split"].value_counts().to_dict()
    logger.info(
        f"WUDAPT splits: {counts} across {out['region'].nunique()} regions | "
        f"{int(out['forced_test'].sum())} So2Sat culture cities forced to test"
    )
    unknown = int((out["region"] == "Unknown").sum())
    if unknown:
        isos = sorted(set(out.loc[out.region == "Unknown", "iso"].dropna().astype(str)))
        logger.warning(f"{unknown} AOIs have an unmapped ISO ({isos[:10]}); add them to ISO_TO_REGION")
    return out


def assert_split_integrity(splits: pd.DataFrame) -> None:
    """Fail loudly if any held-out So2Sat city reached WUDAPT train or val.

    This is the one error that would silently invalidate every number the
    adaptation is judged against, so it is an assertion and not a log line.
    """
    bad = splits[(splits["forced_test"]) & (splits["wudapt_split"] != "test")]
    if len(bad):
        raise AssertionError(
            f"So2Sat held-out cities leaked into WUDAPT {sorted(set(bad.wudapt_split))}: "
            f"{sorted(bad.aoi)}. Every headline kappa in "
            "docs/global_lcz_campaign_2026-07.md is measured on these cities. "
            "Refusing to proceed."
        )
    missing = set(splits["wudapt_split"]) - {"train", "val", "test"}
    if missing:
        raise AssertionError(f"unexpected split labels: {sorted(missing)}")
    for name in ("train", "val", "test"):
        if not (splits["wudapt_split"] == name).any():
            raise AssertionError(f"WUDAPT split '{name}' is empty; --split-col would reject it")
    logger.info(
        f"split integrity OK: {int(splits.forced_test.sum())} culture cities in test, "
        f"{splits.wudapt_split.value_counts().to_dict()}"
    )


def write_splits(config: WudaptConfig, splits: pd.DataFrame) -> Path:
    """Persist the split table next to the other AOI-level artefacts."""
    path = Path(config.cache_dir) / f"wudapt_splits_{config.config_hash}.parquet"
    splits.to_parquet(path, index=False)
    logger.info(f"wrote {path.name} ({len(splits):,} AOIs)")
    return path
