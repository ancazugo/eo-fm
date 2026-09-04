"""Leakage control — which So2Sat cities may admit WUDAPT labels.

So2Sat is authoritative wherever it exists, so WUDAPT is admitted inside a
So2Sat city only where So2Sat's own labels are too sparse or too degenerate to
supervise anything. Measured from ``patches_reference_rxr.gpkg``, 11 of the 51
cities are single-class and tiny — Salvador has **1** patch, Philadelphia 2,
Buenos Aires 5, Bogota 8, Caracas 12, Dhaka 30, Chicago 48, Lima 48, Quezon City
384, Karachi 1140 — while every one of the 10 held-out test cities has >= 4798
patches and >= 13 classes.

So the sparsity rule cannot select a test city. That is a happy structural
property, not a guarantee, so :func:`assert_no_test_city_admitted` enforces it in
code: if the rule or the data ever changes so that a test city qualifies, the
run fails loudly instead of quietly contaminating the benchmark.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from .config import WudaptConfig
from .ingest import N_LCZ

__all__ = [
    "SO2SAT_TEST_CITIES",
    "assert_no_test_city_admitted",
    "city_label_inventory",
    "sparse_cities",
]

# The So2Sat "culture-10" held-out cities: the only ones carrying `testing` /
# `validation` patches in the global split. Every headline kappa in
# docs/global_lcz_campaign_2026-07.md is measured on these.
SO2SAT_TEST_CITIES = (
    "Santiago", "San Jose", "Tehran", "Nairobi", "Sydney",
    "Moscow", "Jakarta", "Munich", "Mumbai", "Guangzhou",
)


def city_label_inventory(config: WudaptConfig) -> pd.DataFrame:
    """Per-So2Sat-city patch count, class count and split membership.

    Read from the authoritative global gpkg (``patches_reference_rxr.gpkg``)
    rather than the per-city files, so the numbers are the ones the benchmark
    actually trains and tests on.
    """
    so2sat_dir = Path(config.labels.so2sat_dir)
    patches = gpd.read_file(so2sat_dir / "patches_reference_rxr.gpkg")
    bounds = gpd.read_file(so2sat_dir / "so2sat_guppd_bounds.gpkg")

    pts = patches.copy()
    pts["geometry"] = patches.geometry.representative_point()
    join = gpd.sjoin(
        pts[["dataset", "LCZ_class", "geometry"]],
        bounds[["JRC_NAME_MAIN", "geometry"]],
        predicate="within", how="inner",
    )
    inv = join.groupby("JRC_NAME_MAIN").agg(
        n_patches=("LCZ_class", "size"),
        n_classes=("LCZ_class", "nunique"),
    )
    splits = join.pivot_table(index="JRC_NAME_MAIN", columns="dataset",
                              values="LCZ_class", aggfunc="size").fillna(0).astype(int)
    inv = inv.join(splits).reset_index().rename(columns={"JRC_NAME_MAIN": "city"})
    inv["is_test_city"] = inv["city"].isin(SO2SAT_TEST_CITIES)
    return inv.sort_values("n_patches")


def sparse_cities(inventory: pd.DataFrame, config: WudaptConfig) -> list[str]:
    """So2Sat cities whose own labels are too thin to supervise anything."""
    sel = (
        (inventory["n_patches"] < config.sparse_city_max_patches)
        | (inventory["n_classes"] < config.sparse_city_max_classes)
    )
    return inventory.loc[sel, "city"].tolist()


def assert_no_test_city_admitted(admitted: list[str]) -> None:
    """Hard guard: no held-out test city may ever admit WUDAPT labels.

    Mirrors ``lcz_train.splits.assert_no_leakage`` in spirit — the one error that
    would silently invalidate every headline number gets an explicit assertion
    rather than a comment.
    """
    bad = sorted(set(admitted) & set(SO2SAT_TEST_CITIES))
    if bad:
        raise AssertionError(
            f"WUDAPT would be admitted into held-out So2Sat test cities {bad}. "
            "This contaminates the benchmark reported in "
            "docs/global_lcz_campaign_2026-07.md. Refusing to proceed."
        )
    logger.info(f"leakage guard OK: {len(admitted)} sparse cities admitted, none held-out")
