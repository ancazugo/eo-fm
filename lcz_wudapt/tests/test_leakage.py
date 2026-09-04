"""The one error that would silently invalidate every headline number."""

from __future__ import annotations

import pandas as pd
import pytest

from lcz_wudapt.config import WudaptConfig
from lcz_wudapt.leakage import SO2SAT_TEST_CITIES, assert_no_test_city_admitted, sparse_cities


def _inventory() -> pd.DataFrame:
    """The real measured shape: degenerate training cities, healthy test cities."""
    rows = [
        ("Salvador", 1, 1), ("Philadelphia", 2, 1), ("Buenos Aires", 5, 1),
        ("Dhaka", 30, 1), ("Karachi", 1140, 1), ("Tokyo", 884, 10),
        ("Berlin", 19130, 12), ("London", 27340, 16),
    ]
    rows += [(c, 4798, 13) for c in SO2SAT_TEST_CITIES]
    df = pd.DataFrame(rows, columns=["city", "n_patches", "n_classes"])
    df["is_test_city"] = df["city"].isin(SO2SAT_TEST_CITIES)
    return df


def test_sparsity_rule_selects_the_degenerate_training_cities():
    got = sparse_cities(_inventory(), WudaptConfig())
    assert {"Salvador", "Philadelphia", "Buenos Aires", "Dhaka", "Karachi", "Tokyo"} <= set(got)
    assert "Berlin" not in got and "London" not in got


def test_sparsity_rule_cannot_select_a_held_out_test_city():
    """Every test city has >= 4798 patches and >= 13 classes; the rule must miss them all."""
    got = sparse_cities(_inventory(), WudaptConfig())
    assert not set(got) & set(SO2SAT_TEST_CITIES)
    assert_no_test_city_admitted(got)          # must not raise


def test_guard_raises_when_a_test_city_would_be_admitted():
    """A future threshold or data change must fail loudly, not contaminate quietly."""
    with pytest.raises(AssertionError, match="held-out So2Sat test cities"):
        assert_no_test_city_admitted(["Berlin", "Nairobi"])


def test_guard_names_every_offending_city():
    with pytest.raises(AssertionError) as exc:
        assert_no_test_city_admitted(["Tehran", "Mumbai"])
    assert "Mumbai" in str(exc.value) and "Tehran" in str(exc.value)


def test_a_looser_threshold_would_catch_a_test_city_and_be_refused():
    """Pins that the guard is doing real work, not passing vacuously."""
    cfg = WudaptConfig(sparse_city_max_patches=5000)
    got = sparse_cities(_inventory(), cfg)
    assert set(got) & set(SO2SAT_TEST_CITIES)
    with pytest.raises(AssertionError):
        assert_no_test_city_admitted(got)
