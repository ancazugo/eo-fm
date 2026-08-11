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
    VERSIONS,
    check_checkpoint_provenance,
    is_comparable,
    provenance,
)

TESSERA_V11 = "tesserav1.1"
TESSERA_GLOBAL = "tesserav1.1_global"


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
        meta["status"] = "deprecated"
        assert is_comparable(TESSERA_V11, TESSERA_V11)
    finally:
        meta["status"] = original


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
