"""Task 1.5.2 / 1.5.3 — provenance metadata and the three guards.

Offline: the registry is a literal and the guards are pure functions, so none of
this needs the data mounts.

What these are defending against, concretely: `tesserav1.1` and
`tesserav1.1_global` are different feature bases of the same scene (Task 1.5.1),
yet both declare `in_channels: 128`. A checkpoint from one loads cleanly against
the other and a normalizer fitted on one applies silently to the other. Nothing
about the shapes catches it, so the provenance triple has to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.registry import (  # noqa: E402
    EMBEDDING_REGISTRY,
    PRODUCTS,
    PROVENANCE_FIELDS,
    SOURCES,
    STATUSES,
    UNAVAILABLE_STATUSES,
    VERSIONS,
    available_embeddings,
    check_checkpoint_provenance,
    is_comparable,
    provenance,
)

TESSERA_V11 = "tesserav1.1"
TESSERA_GLOBAL = "tesserav1.1_global"

# PLAN-v3's scope restriction, as data: the only three families any experiment
# from Phase 2 onward may use.
IN_SCOPE = {TESSERA_GLOBAL, "alpha_earth_coop", "seamless"}
OUT_OF_SCOPE = {"tessera", "alpha_earth", TESSERA_V11, "tesserav2"}


# ── Schema ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(EMBEDDING_REGISTRY))
def test_every_entry_declares_a_full_provenance(name):
    """A missing field would compare unequal to everything and block every run."""
    meta = EMBEDDING_REGISTRY[name]
    for field in PROVENANCE_FIELDS:
        assert field in meta, f"{name} is missing '{field}'"


@pytest.mark.parametrize("name", sorted(EMBEDDING_REGISTRY))
def test_provenance_values_come_from_the_vocabularies(name):
    """A typo would create a unique provenance that matches nothing, silently."""
    meta = EMBEDDING_REGISTRY[name]
    assert meta["product"] in PRODUCTS
    assert meta["version"] in VERSIONS
    assert meta["source"] in SOURCES
    assert meta["status"] in STATUSES


def test_exactly_one_canonical_tessera():
    """Phase 2 makes every cross-family claim against one Tessera entry."""
    canonical = [
        n for n, m in EMBEDDING_REGISTRY.items()
        if m["product"] == "tessera" and m["status"] == "canonical"
    ]
    assert canonical == [TESSERA_GLOBAL]


def test_provenance_of_a_fused_run_joins_each_field():
    p = provenance([TESSERA_GLOBAL, "aux_struct"])
    assert p["embedding_name"] == f"{TESSERA_GLOBAL}+aux_struct"
    assert p["product"] == "tessera+aux"
    assert p["version"] == "v1.1+none"        # None renders as a string
    assert p["source"] == "global_0.1deg+local_tif"


def test_provenance_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown embedding"):
        provenance("tesserav9")


# ── is_comparable ────────────────────────────────────────────────────────────

def test_the_two_tessera_v11_extractions_are_not_comparable():
    """The whole point of Task 1.5.2. Same product and version, different source."""
    assert not is_comparable(TESSERA_V11, TESSERA_GLOBAL)


@pytest.mark.parametrize("name", sorted(EMBEDDING_REGISTRY))
def test_every_entry_is_comparable_with_itself(name):
    assert is_comparable(name, name)


def test_tessera_generations_are_not_comparable():
    assert not is_comparable(TESSERA_GLOBAL, "tesserav2")


def test_comparability_ignores_status():
    """Promoting an entry to canonical must not invalidate existing checkpoints."""
    meta = EMBEDDING_REGISTRY[TESSERA_V11]
    original = meta["status"]
    try:
        meta["status"] = "canonical"
        assert is_comparable(TESSERA_V11, TESSERA_V11)
        assert not is_comparable(TESSERA_V11, TESSERA_GLOBAL)
    finally:
        meta["status"] = original


# ── Task 1.75.0: the PLAN-v3 scope restriction ───────────────────────────────

def test_the_four_out_of_scope_families_are_marked_unavailable():
    for name in OUT_OF_SCOPE:
        assert EMBEDDING_REGISTRY[name]["status"] in UNAVAILABLE_STATUSES, name


def test_tesserav2_is_pending_not_deprecated():
    """Deferred, not cancelled — extraction is still running and the version
    comparison runs once it completes."""
    assert EMBEDDING_REGISTRY["tesserav2"]["status"] == "pending"


def test_available_embeddings_offers_the_three_in_scope_families():
    available = set(available_embeddings())
    assert IN_SCOPE <= available
    assert not (OUT_OF_SCOPE & available)


def test_available_embeddings_is_sorted_and_a_subset_of_the_registry():
    available = available_embeddings()
    assert available == sorted(available)
    assert set(available) <= set(EMBEDDING_REGISTRY)


@pytest.mark.parametrize("name", sorted(OUT_OF_SCOPE))
def test_a_deprecated_entry_keeps_its_key_and_its_provenance(name):
    """Hiding an entry from the CLI must not orphan its history: the key still
    resolves, so every W&B run and every checkpoint that names it stays readable."""
    p = provenance(name)
    assert p["embedding_name"] == name
    assert p["product"] and p["source"]


@pytest.mark.parametrize("name", sorted(OUT_OF_SCOPE))
def test_deprecation_does_not_change_comparability(name):
    """status is about trust and scope, never about the data."""
    assert is_comparable(name, name)


# ── Guard 2: checkpoint binding ──────────────────────────────────────────────

def test_checkpoint_guard_raises_on_a_mismatch_and_names_both_sides():
    ckpt = provenance(TESSERA_V11)
    with pytest.raises(ValueError) as exc:
        check_checkpoint_provenance(ckpt, TESSERA_GLOBAL)

    message = str(exc.value)
    assert TESSERA_V11 in message and TESSERA_GLOBAL in message
    assert "percity_geotessera" in message and "global_0.1deg" in message


def test_checkpoint_guard_accepts_a_match():
    check_checkpoint_provenance(provenance(TESSERA_GLOBAL), TESSERA_GLOBAL)


def test_checkpoint_guard_only_warns_when_provenance_is_absent():
    """Pre-1.5.3 checkpoints carry nothing — including the GATE 1 regression one."""
    check_checkpoint_provenance({"model_state_dict": {}}, TESSERA_GLOBAL)


def test_checkpoint_guard_catches_a_cross_product_swap():
    with pytest.raises(ValueError, match="alphaearth"):
        check_checkpoint_provenance(provenance("alpha_earth_coop"), TESSERA_GLOBAL)


# ── Guard 1: normalizer cache key ────────────────────────────────────────────

def test_stats_cache_path_separates_the_two_tessera_products():
    """Everything else identical — only the embedding differs, as in practice.

    Both extractions get passed the same --output-name in a careless re-run, so
    output_name cannot be the thing that keeps their normalizers apart.
    """
    from datasets.channel_stats import stats_cache_path

    args = ("GeoTessera_v1.1", "2017", "global_so2sat")
    a = stats_cache_path(*args, embedding_name=TESSERA_V11)
    b = stats_cache_path(*args, embedding_name=TESSERA_GLOBAL)
    assert a != b
    assert "percity_geotessera" in a.name
    assert "global_0.1deg" in b.name


def test_stats_cache_path_is_stable_for_the_same_embedding():
    from datasets.channel_stats import stats_cache_path

    args = ("AlphaEarthCoop", "2017", "global_so2sat")
    assert (stats_cache_path(*args, embedding_name="alpha_earth_coop")
            == stats_cache_path(*args, embedding_name="alpha_earth_coop"))


def test_stats_cache_digest_also_carries_provenance():
    """With items passed, the digest must move too — not just the filename stem."""
    from datasets.channel_stats import stats_cache_path
    from datasets.so2sat import PatchItem

    items = [PatchItem(Path(f"/x/patch_{i}.npy"), 0, "train") for i in range(8)]
    a = stats_cache_path("Same", "2017", "grid", embedding_name=TESSERA_V11,
                         items=items)
    b = stats_cache_path("Same", "2017", "grid", embedding_name=TESSERA_GLOBAL,
                         items=items)
    assert a.name.split("_")[-1] != b.name.split("_")[-1]


# ── Guard 3: W&B config ──────────────────────────────────────────────────────

def test_wandb_config_and_checkpoint_carry_the_same_five_fields():
    """The five fields RESULTS.md has to be reconstructable from."""
    p = provenance(TESSERA_GLOBAL)
    assert set(p) == {"embedding_name", "product", "version", "source", "status"}
    assert p["embedding_name"] == TESSERA_GLOBAL
    assert p["product"] == "tessera"
    assert p["source"] == "global_0.1deg"
