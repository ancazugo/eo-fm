"""PatchDataset's item contract.

`PatchItem` replaced the old `(path, label, split)` tuples, but three scripts
kept building tuples and would raise `AttributeError` at `it.path` the moment
they were run: `tta_city_adapt.py`, `ensemble_eval.py`, `generate_pseudo_labels.py`.
These tests pin the contract so the next such drift is caught here rather than
in a run that has already loaded a model.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.so2sat import PatchDataset, PatchItem  # noqa: E402

_SCRIPTS = ("tta_city_adapt.py", "ensemble_eval.py", "generate_pseudo_labels.py")


def test_patch_item_is_not_tuple_indexable():
    """`items[0][0]` used to mean 'the path'. It must now fail loudly, not
    silently return something plausible.
    """
    it = PatchItem(Path("/x.npy"), 0, "train")
    with pytest.raises(TypeError):
        _ = it[0]


def test_patch_dataset_accepts_patch_items(tmp_path):
    p = tmp_path / "patch_000001.npy"
    np.save(p, np.zeros((4, 32, 32), dtype=np.float32))
    ds = PatchDataset([PatchItem(p, 3, "train")], 32)
    out = ds[0]
    assert out["image"].shape == (4, 32, 32)
    assert int(out["label"]) == 3


def test_patch_dataset_rejects_bare_tuples(tmp_path):
    """The exact regression: a 3-tuple must not silently work."""
    p = tmp_path / "patch_000002.npy"
    np.save(p, np.zeros((4, 32, 32), dtype=np.float32))
    with pytest.raises((AttributeError, TypeError)):
        PatchDataset([(p, 3, "train")], 32)[0]


@pytest.mark.parametrize("script", _SCRIPTS)
def test_scripts_build_patch_items_not_tuples(script):
    """Static check: each script imports PatchItem and constructs it.

    Cheaper and more robust than importing these scripts, which pull torch, GEE
    and a checkpoint parser at module scope.
    """
    src = (Path(__file__).resolve().parents[1] / "src" / script).read_text()
    tree = ast.parse(src)
    imports_it = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "datasets.so2sat"
        and any(a.name == "PatchItem" for a in n.names)
        for n in ast.walk(tree)
    )
    constructs_it = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "PatchItem"
        for n in ast.walk(tree)
    )
    assert imports_it, f"{script} does not import PatchItem"
    assert constructs_it, f"{script} does not construct PatchItem"
