"""Warm start, freezing and shot budgets. Offline — no data, no GPU."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.adapt import (  # noqa: E402
    FREEZE_MODES,
    apply_freeze,
    find_head_modules,
    limit_shots_per_class,
    limit_train_items,
    load_init_weights,
)
from models import build_model, families_for  # noqa: E402


# ── Head discovery across every registered family ────────────────────────────

@pytest.mark.parametrize("family", sorted(families_for("classification")))
def test_head_is_found_for_every_classification_family(family):
    model = build_model(family, "nano", in_channels=8, num_classes=17)
    assert find_head_modules(model, 17), family


@pytest.mark.parametrize("family", sorted(families_for("segmentation")))
def test_head_is_found_for_every_segmentation_family(family):
    model = build_model(family, "nano", in_channels=8, num_classes=17)
    assert find_head_modules(model, 17), family


def test_head_discovery_is_structural_not_name_based():
    """The head is 'fc' on ResNet, 'head' on UNet, 'scores.*' on FCN8. Matching
    on output width instead of attribute names is what makes one flag work
    across the registry.
    """
    names = {
        f: set(find_head_modules(build_model(f, "nano", in_channels=8, num_classes=17), 17))
        for f in ("resnet", "unet", "fcn8")
    }
    assert len(set(map(frozenset, names.values()))) == len(names)


# ── Freezing ─────────────────────────────────────────────────────────────────

def test_freeze_none_leaves_everything_trainable():
    m = build_model("resnet", "nano", in_channels=8, num_classes=17)
    rep = apply_freeze(m, "none", 17)
    assert rep["frozen"] == 0
    assert all(p.requires_grad for p in m.parameters())


def test_freeze_backbone_trains_only_the_head():
    m = build_model("resnet", "nano", in_channels=8, num_classes=17)
    rep = apply_freeze(m, "backbone", 17)
    assert rep["trainable"] > 0
    assert rep["frozen"] > rep["trainable"]
    head = m.get_submodule(rep["head_modules"][0])
    assert all(p.requires_grad for p in head.parameters())


def test_freeze_backbone_keep_bn_adds_batchnorm_affine():
    m = build_model("resnet", "nano", in_channels=8, num_classes=17)
    only_head = apply_freeze(m, "backbone", 17)["trainable"]
    m = build_model("resnet", "nano", in_channels=8, num_classes=17)
    with_bn = apply_freeze(m, "backbone_keep_bn", 17)["trainable"]
    assert with_bn > only_head


def test_freeze_rejects_a_head_width_that_does_not_exist():
    """Freezing everything would train nothing; that must fail loudly rather
    than produce a run whose loss never moves.
    """
    m = build_model("resnet", "nano", in_channels=8, num_classes=17)
    with pytest.raises(ValueError, match="no output head"):
        apply_freeze(m, "backbone", 99)


def test_unknown_freeze_mode_is_rejected():
    m = nn.Linear(4, 17)
    with pytest.raises(ValueError, match="unknown freeze mode"):
        apply_freeze(m, "everything", 17)


def test_all_declared_modes_are_implemented():
    m = build_model("resnet", "nano", in_channels=8, num_classes=17)
    for mode in FREEZE_MODES:
        apply_freeze(m, mode, 17)


def test_optimiser_only_sees_trainable_params():
    """requires_grad=False alone does not freeze a weight that Adam still holds:
    weight decay and momentum keep moving it. This pins the filter the training
    loop applies.
    """
    m = build_model("resnet", "nano", in_channels=8, num_classes=17)
    apply_freeze(m, "backbone", 17)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-2, weight_decay=0.5)
    frozen = [p for p in m.parameters() if not p.requires_grad]
    before = [p.detach().clone() for p in frozen]
    for _ in range(3):
        opt.zero_grad()
        loss = sum(p.sum() for p in params)
        loss.backward()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, frozen))


# ── Warm start ───────────────────────────────────────────────────────────────

def test_warm_start_loads_weights_and_reports(tmp_path):
    src = build_model("resnet", "nano", in_channels=8, num_classes=17)
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state_dict": src.state_dict(), "epoch": 3, "val_f1": 0.5}, ckpt)

    dst = build_model("resnet", "nano", in_channels=8, num_classes=17)
    rep = load_init_weights(dst, ckpt, torch.device("cpu"))
    assert rep["missing"] == 0 and rep["unexpected"] == 0
    for a, b in zip(src.state_dict().values(), dst.state_dict().values()):
        assert torch.equal(a, b)


def test_warm_start_accepts_a_bare_state_dict(tmp_path):
    src = build_model("resnet", "nano", in_channels=8, num_classes=17)
    ckpt = tmp_path / "bare.pt"
    torch.save(src.state_dict(), ckpt)
    dst = build_model("resnet", "nano", in_channels=8, num_classes=17)
    assert load_init_weights(dst, ckpt, torch.device("cpu"))["loaded"] > 0


def test_warm_start_refuses_a_checkpoint_from_another_family(tmp_path):
    """A mismatched load produces a model that trains fine and means nothing.
    That is the worst failure mode available here, so it raises.
    """
    ckpt = tmp_path / "unet.pt"
    torch.save(build_model("unet", "nano", in_channels=8, num_classes=17).state_dict(), ckpt)
    dst = build_model("linear_probe", "nano", in_channels=8, num_classes=17)
    with pytest.raises(ValueError, match="zero tensors"):
        load_init_weights(dst, ckpt, torch.device("cpu"))


def test_warm_start_tolerates_a_changed_head_but_says_so(tmp_path, caplog):
    src = build_model("resnet", "nano", in_channels=8, num_classes=17)
    ckpt = tmp_path / "m.pt"
    torch.save(src.state_dict(), ckpt)
    dst = build_model("resnet", "nano", in_channels=8, num_classes=5)
    rep = load_init_weights(dst, ckpt, torch.device("cpu"))
    assert rep["loaded"] > 0
    assert rep["reinit"] == 2          # fc.weight and fc.bias
    # the backbone really did transfer, only the head was re-initialised
    assert torch.equal(src.conv1.weight, dst.conv1.weight)
    assert dst.fc.out_features == 5


# ── Shot budgets ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Item:
    path: str
    label: int
    split: str
    city: str | None = None
    weight: float | None = None


def _items():
    out = []
    for city in ("A", "B"):
        for label in range(3):
            for k in range(10):
                out.append(_Item(f"{city}{label}{k}", label, "train", city))
    out += [_Item("v", 0, "val", "A"), _Item("t", 1, "test", "A")]
    return out


def test_shots_are_counted_per_class_per_city():
    got = limit_shots_per_class(_items(), 2, seed=0)
    train = [i for i in got if i.split == "train"]
    assert len(train) == 2 * 3 * 2          # n x classes x cities
    for city in ("A", "B"):
        for label in range(3):
            assert sum(1 for i in train if i.city == city and i.label == label) == 2


def test_shots_never_touch_val_or_test():
    got = limit_shots_per_class(_items(), 1, seed=0)
    assert sum(1 for i in got if i.split == "val") == 1
    assert sum(1 for i in got if i.split == "test") == 1


def test_shots_none_is_a_no_op():
    src = _items()
    assert limit_shots_per_class(src, None) is src


def test_shots_are_seeded_and_reproducible():
    a = [i.path for i in limit_shots_per_class(_items(), 3, seed=7)]
    b = [i.path for i in limit_shots_per_class(_items(), 3, seed=7)]
    c = [i.path for i in limit_shots_per_class(_items(), 3, seed=8)]
    assert a == b and a != c


def test_shots_larger_than_the_pool_keep_everything():
    src = _items()
    assert len(limit_shots_per_class(src, 999, seed=0)) == len(src)


def test_tile_budget_counts_per_city_from_the_filename():
    """City comes from the filename stem, not a directory level: paths are
    {city_dir}/{output_name}/{year}/{split}/{city}_{grid_id}.npy.
    """
    items, split_map = [], {}
    for city in ("Berlin", "Osaka_[Kyoto]"):
        for gid in range(8):
            p = Path(f"/d/{city}/emb/2017/train/{city}_{gid:02d}.npy")
            items.append((p, None, None, [], None))
            split_map[p] = "train"
    got = limit_train_items(items, split_map, 3, seed=0)
    assert len(got) == 6
    stems = [Path(it[0]).name.rsplit("_", 1)[0] for it in got]
    assert stems.count("Berlin") == 3 and stems.count("Osaka_[Kyoto]") == 3


def test_tile_budget_handles_the_fused_tuple_key():
    p1 = Path("/d/Berlin/a/2017/train/Berlin_01.npy")
    p2 = Path("/d/Berlin/b/2017/train/Berlin_01.npy")
    items = [((p1, p2), None, None, [], None)]
    got = limit_train_items(items, {p1: "train"}, 5, seed=0)
    assert len(got) == 1


def test_tile_budget_leaves_val_and_test_alone():
    items, split_map = [], {}
    for i, split in enumerate(["train"] * 5 + ["val"] * 4 + ["test"] * 3):
        p = Path(f"/d/C/emb/2017/{split}/C_{i:02d}.npy")
        items.append((p, None, None, [], None))
        split_map[p] = split
    got = limit_train_items(items, split_map, 2, seed=0)
    assert sum(1 for it in got if split_map[it[0]] == "train") == 2
    assert sum(1 for it in got if split_map[it[0]] == "val") == 4
    assert sum(1 for it in got if split_map[it[0]] == "test") == 3


# ── The small-N dataloader trap ──────────────────────────────────────────────

def test_partial_batch_is_kept_when_the_shot_budget_is_below_one_batch():
    """drop_last=True drops EVERY batch when the training set is smaller than
    batch_size: the epoch yields nothing, loss prints 0.0000 and the run trains
    on no data. That is exactly --shots-per-class 1 or 5, the interesting end of
    a few-shot curve, so it has to keep the partial batch.

    Measured before the fix: a 61-item 5-shot Nairobi run reported loss 0.0000
    and test kappa 0.0610 (chance). After: loss 2.68 -> 2.55, kappa 0.1538.
    """
    from torch.utils.data import DataLoader, Dataset

    class _Tiny(Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return torch.zeros(2)

    def _drop_last_for(n, batch_size):
        return n >= 2 * batch_size

    for n, bs in [(61, 64), (5, 64), (1, 32)]:
        assert not _drop_last_for(n, bs)
        assert len(list(DataLoader(_Tiny(n), batch_size=bs,
                                   drop_last=_drop_last_for(n, bs)))) > 0
    # and a comfortably large set still drops its ragged tail
    assert _drop_last_for(4828, 64)
