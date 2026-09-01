"""City-level (global) split assignment for the segmentation pipeline.

The segmentation pipeline has only ever had the within-city macro-block split
built by :mod:`utils.grid_split` -- train, val and test cells interleaved on a
3x3 checkerboard inside every city. Section 7 of
``docs/global_lcz_campaign_2026-07.md`` quarantines every number produced that
way as spatial-autocorrelation-inflated, so it cannot carry a headline result.

This module supplies the alternative: assign the split by **city**, inheriting
the So2Sat culture-10 assignment that the patch pipeline already uses, so
segmentation numbers land on the same footing as the 0.6497 / 0.6871 / 0.7055
patch ladder.

Three rules make it honest, and all three are enforced here rather than left to
the caller:

**Tile purity.** A tile whose patches carry more than one split is dropped, not
majority-assigned. This is the segmentation-specific leak with no analogue in
the patch pipeline: a 1.28 km tile is a single training example, so one test
patch inside it contaminates the whole thing.

**A proximity buffer.** The original plan called for buffering an east/west
"dividing meridian" per culture city. There is no such meridian -- the So2Sat
validation and testing patches interleave, and in Santiago the validation
x-range sits entirely inside the testing one. So the buffer is implemented as
what it was actually for: a tile is dropped when it comes within
``buffer_km`` of a patch assigned to a *different* split. That is the same
protection without assuming a geometry the data does not have.

**Culture cities contribute no training tiles.** Only Guangzhou is affected --
it is the one city carrying both ``training`` (5,540) and ``testing`` /
``validation`` patches, so "the 42 training cities" and "the 10 culture cities"
genuinely overlap by one. Its training patches sit ~24 km east of its val/test
patches and would probably survive the buffer, but a city that appears in both
train and test needs a decision recorded rather than an accident, and dropping
1.6 % of the training pool keeps all 10 test cities and therefore keeps the
comparison against the published ladder exact.
"""

from __future__ import annotations

import unicodedata

from loguru import logger

from utils.geo_lookup import CITY_TO_CONTINENT

# The 10 So2Sat "culture" cities -- the only ones carrying validation/testing
# patches. Spelled as the city *directory* names under
# ``${DATA_DIR}/input/So2Sat-LCZ42/v4/cities``. Mirrors
# ``lcz_wudapt/leakage.py::SO2SAT_TEST_CITIES``, which uses space-separated
# spellings; the two are checked against each other in the test suite. Not
# imported from there because ``python src/semantic_segmentation.py`` puts
# ``src/`` on sys.path, not the repo root.
SO2SAT_CULTURE_CITIES: tuple[str, ...] = (
    "Guangzhou", "Jakarta", "Moscow", "Mumbai", "Munich",
    "Nairobi", "San_Jose", "Santiago", "Sydney", "Tehran",
)

# So2Sat's original split names -> our loader splits.
DATASET_TO_ROLE = {"training": "train", "validation": "val", "testing": "test"}

# Split labels this module can emit. ``culture_val`` is the culture cities'
# validation patches: deliberately NOT used for early stopping (that is what
# val-inner is for), but kept addressable because
# ``ensemble_stacking.py --city-holdout`` fits its weights on exactly those
# patches, and the LOCO protocol is the only honest way to combine models.
SPLITS = ("train", "val", "test", "culture_val")

DEFAULT_N_VAL_INNER = 6
DEFAULT_BUFFER_KM = 1.3


def _norm(name: str) -> str:
    """Fold a city name to a comparison key.

    City directories are ASCII-ified and underscored (``Sao_Paulo``,
    ``Osaka_[Kyoto]``) while ``geo_lookup`` uses the JRC spellings
    (``São Paulo``, ``Osaka [Kyoto]``, ``东营区``). Comparing them raw silently
    drops cities from the stratification.
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace("_", " ").replace("[", "").replace("]", "").strip()


_CONTINENT_BY_KEY = {_norm(k): v for k, v in CITY_TO_CONTINENT.items()}
# Dongying's JRC name is 东营区, which no folding maps to "dongying".
_CONTINENT_BY_KEY.setdefault("dongying", "Asia")


def continent_of(city_dir_name: str) -> str:
    return _CONTINENT_BY_KEY.get(_norm(city_dir_name), "Unknown")


def assign_city_roles(
    cities: list[str],
    n_val_inner: int = DEFAULT_N_VAL_INNER,
    seed: int = 42,
    city_weights: dict[str, int] | None = None,
) -> dict[str, str]:
    """Map each city directory name to ``"culture"``, ``"val_inner"`` or ``"train"``.

    The val-inner cities are held out of training for early stopping. The
    campaign's own section 5 audit is the argument for having them at all: the
    culture cities' val and test patches are the *same 10 cities*, median 2.55
    km apart, so anything selected on that val set can read city-conditional
    structure off test. Early stopping is a low-degrees-of-freedom fit, but it
    is not zero, and selecting on held-out *training-pool* cities costs almost
    nothing and removes the objection outright.

    Selection is deterministic and continent-stratified -- picking 6 cities at
    random can easily take 4 from Europe, which would make the early-stopping
    signal a European one.

    ``city_weights`` (city -> tile or patch count) breaks the within-continent
    choice towards the *largest* available city. Without it the picker can hand
    back cities like Salvador, which has a single grid tile: stratification
    would look right and the early-stopping signal would be noise.
    """
    import random

    culture = [c for c in cities if c in SO2SAT_CULTURE_CITIES]
    pool = sorted(c for c in cities if c not in SO2SAT_CULTURE_CITIES)
    if n_val_inner > len(pool):
        raise ValueError(
            f"--val-inner-cities {n_val_inner} exceeds the {len(pool)} "
            "non-culture cities available"
        )

    by_continent: dict[str, list[str]] = {}
    for c in pool:
        by_continent.setdefault(continent_of(c), []).append(c)

    rng = random.Random(seed)
    weights = city_weights or {}
    # Round-robin over continents, largest first, so the held-out set spans as
    # many regions as it has slots. Within a continent, take the biggest city;
    # the seeded shuffle only breaks ties when no weights are supplied.
    order = sorted(by_continent, key=lambda k: (-len(by_continent[k]), k))
    for k in order:
        rng.shuffle(by_continent[k])
        by_continent[k].sort(key=lambda c: -weights.get(c, 0))
    picked: list[str] = []
    while len(picked) < n_val_inner:
        progressed = False
        for k in order:
            if by_continent[k] and len(picked) < n_val_inner:
                picked.append(by_continent[k].pop(0))
                progressed = True
        if not progressed:
            break

    roles = {c: "culture" for c in culture}
    for c in pool:
        roles[c] = "val_inner" if c in picked else "train"

    logger.info(
        f"City roles — {len(culture)} culture, {len(picked)} val-inner "
        f"({', '.join(sorted(picked))}), "
        f"{len(pool) - len(picked)} train"
    )
    return roles


def tile_splits_for_city(
    city: str,
    city_role: str,
    id2patches: dict[int, list],
    id2geom: dict[int, object],
    *,
    buffer_km: float = DEFAULT_BUFFER_KM,
    min_labelled_frac: float = 0.01,
) -> tuple[dict[int, str], dict[str, int]]:
    """Assign one split per grid cell for one city, or drop the cell.

    ``id2patches`` maps grid_id -> list of ``(geom, lcz_class, dataset,
    patch_id)`` in the tile CRS (metres). ``id2geom`` maps grid_id -> tile
    geometry in the same CRS.

    Returns ``(grid_id -> split, drop_reason_counts)``. A grid cell absent from
    the returned mapping is dropped.
    """
    counts = {"no_labels": 0, "mixed_split": 0, "buffer": 0,
              "sparse": 0, "culture_train": 0}
    out: dict[int, str] = {}

    # Patch centroids per role, for the proximity buffer.
    from shapely.strtree import STRtree

    role_geoms: dict[str, list] = {}
    for gid, patches in id2patches.items():
        for geom, _cls, dataset, _pid in patches:
            role = DATASET_TO_ROLE.get(dataset)
            if role is None:
                continue
            if city_role == "culture" and role == "train":
                continue  # Guangzhou's training patches; counted below
            role_geoms.setdefault(role, []).append(geom)
    trees = {r: STRtree(g) for r, g in role_geoms.items() if g}

    buffer_m = buffer_km * 1000.0

    for gid, geom in id2geom.items():
        patches = id2patches.get(gid, [])
        roles = set()
        n_dropped_train = 0
        for _g, _c, dataset, _pid in patches:
            role = DATASET_TO_ROLE.get(dataset)
            if role is None:
                continue
            if city_role == "culture" and role == "train":
                n_dropped_train += 1
                continue
            roles.add(role)

        if not roles:
            counts["culture_train" if n_dropped_train else "no_labels"] += 1
            continue
        if len(roles) > 1:
            counts["mixed_split"] += 1
            continue
        role = roles.pop()

        # A training-pool city contributes only training patches, so its role
        # is decided by the city, not by the patches.
        if city_role == "train":
            split = "train"
        elif city_role == "val_inner":
            split = "val"
        else:  # culture city
            split = "test" if role == "test" else "culture_val"

        # Proximity buffer: reject a tile that comes within buffer_km of a
        # patch belonging to a different split.
        if buffer_m > 0 and city_role == "culture":
            other = "val" if role == "test" else "test"
            tree = trees.get(other)
            if tree is not None and len(tree.query(geom.buffer(buffer_m))) > 0:
                counts["buffer"] += 1
                continue

        # Sparse tiles make batches that are mostly ignore_index.
        area = geom.area
        if area > 0 and min_labelled_frac > 0:
            covered = sum(g.intersection(geom).area for g, _c, _d, _p in patches)
            if covered / area < min_labelled_frac:
                counts["sparse"] += 1
                continue

        out[gid] = split

    return out, counts
