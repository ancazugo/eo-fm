"""Segmentation distillation from a patch-classifier ensemble -- offline checks.

**The ensemble teacher must return exactly the mean softmax** once infer_roi
applies its own softmax to the output, with each member normalising the raw
input by its OWN stats (seeds of one recipe store slightly different ones).

**"all_train" puts every valid tile of a pseudo-labelled city in train**, which
is the whole point of distilling: the global split's purity floor would
otherwise keep only the tiles its sparse So2Sat polygons happen to fill.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.grid_tiles import build_city_tile_items  # noqa: E402
from generate_seg_pseudo_rasters import NormalizedEnsemble  # noqa: E402


class _Linear(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(5, 3, generator=g))

    def forward(self, x):
        return x.mean((2, 3)) @ self.w.T


def test_ensemble_returns_mean_softmax_with_per_member_normalisation():
    x = torch.randn(4, 3, 8, 8)
    norms = [(np.array([0.1, -0.2, 0.3]), np.array([1.0, 2.0, 0.5])),
             (np.array([0.0, 0.5, 0.0]), np.array([0.7, 1.0, 1.5]))]
    members = [(_Linear(i), n) for i, n in enumerate(norms)]
    ens = NormalizedEnsemble(members).eval()

    expected = sum(
        torch.softmax(m((x - torch.tensor(mu, dtype=torch.float32).view(1, -1, 1, 1))
                        / torch.tensor(sd, dtype=torch.float32).view(1, -1, 1, 1)), 1)
        for m, (mu, sd) in members
    ) / 2
    with torch.no_grad():
        got = torch.softmax(ens(x), 1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_ensemble_member_without_stats_is_unnormalised():
    x = torch.randn(2, 3, 4, 4)
    m = _Linear(0)
    ens = NormalizedEnsemble([(m, None)]).eval()
    with torch.no_grad():
        assert torch.allclose(torch.softmax(ens(x), 1), torch.softmax(m(x), 1), atol=1e-6)


def test_all_train_puts_every_valid_tile_in_train(tmp_path):
    city = "Testville"
    cdir = tmp_path / city
    grid = gpd.GeoDataFrame(
        {"grid_id": [0, 1, 2], "is_valid": [True, True, False]},
        geometry=[box(0, 0, 1280, 1280), box(1280, 0, 2560, 1280), box(0, 1280, 1280, 2560)],
        crs="EPSG:32631",
    )
    cdir.mkdir()
    grid.to_file(cdir / f"{city}_grid.gpkg", driver="GPKG")
    for split, gid in (("train", 0), ("test", 1), ("val", 2)):
        d = cdir / "Emb" / "2017" / split
        d.mkdir(parents=True)
        np.save(d / f"{city}_{gid}.npy", np.zeros((2, 4, 4), np.float32))
    tifs = tmp_path / "pseudo"
    tifs.mkdir()
    (tifs / f"pseudo_seg_{city}.tif").write_bytes(b"")  # only existence is checked here

    items, split_map = build_city_tile_items(
        cdir, "Emb", "2017", "tif", "LCZ_class",
        label_tif_dir=tifs, split_mode="all_train",
    )
    assert len(items) == 2                       # the invalid tile is still dropped
    assert set(split_map.values()) == {"train"}  # the grid split is ignored
    assert all(it[4] == tifs / f"pseudo_seg_{city}.tif" for it in items)
