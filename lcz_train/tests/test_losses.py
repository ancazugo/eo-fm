"""T3 — marginalised CE: hard = CE identity, coarse semantics, masks, options."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lcz_labels.export import encode_lcz_set
from lcz_train.losses import bitmask_to_target, marginalized_ce

torch.manual_seed(0)


def _one_hot_sets(labels, n=17):
    t = torch.zeros(len(labels), n, dtype=torch.bool)
    for i, y in enumerate(labels):
        t[i, y] = True
    return t


def test_hard_labels_equal_cross_entropy():
    logits = torch.randn(32, 17)
    y = torch.randint(0, 17, (32,))
    loss = marginalized_ce(logits, _one_hot_sets(y))
    torch.testing.assert_close(loss, F.cross_entropy(logits, y))


def test_hard_labels_with_class_weights_match_torch():
    logits = torch.randn(16, 17)
    y = torch.randint(0, 17, (16,))
    w = torch.rand(17) + 0.1
    loss = marginalized_ce(logits, _one_hot_sets(y), class_weights=w)
    torch.testing.assert_close(loss, F.cross_entropy(logits, y, weight=w))


def test_coarse_loss_depends_only_on_set_mass():
    # {3,7} -> class indices 2 and 6. Two logit rows with identical in-set
    # probability mass must give identical loss: out-of-set logits equal, and
    # b concentrates a's exp-sum exp(1)+exp(1) = exp(1+ln 2) entirely on idx 2.
    target = torch.zeros(1, 17, dtype=torch.bool)
    target[0, [2, 6]] = True
    a = torch.full((1, 17), -2.0)
    a[0, 2], a[0, 6] = 1.0, 1.0
    b = torch.full((1, 17), -2.0)
    b[0, 2], b[0, 6] = 1.0 + float(torch.log(torch.tensor(2.0))), -torch.inf
    pa = torch.softmax(a, 1)[0, [2, 6]].sum()
    pb = torch.softmax(b, 1)[0, [2, 6]].sum()
    torch.testing.assert_close(pa, pb)
    torch.testing.assert_close(marginalized_ce(a, target), marginalized_ce(b, target))


def test_coarse_loss_decreases_when_any_member_prob_rises():
    target = torch.zeros(1, 17, dtype=torch.bool)
    target[0, [2, 6]] = True
    base = torch.zeros(1, 17)
    higher = base.clone()
    higher[0, 6] += 1.0
    assert marginalized_ce(higher, target) < marginalized_ce(base, target)


def test_coarse_never_exceeds_best_hard_member():
    logits = torch.randn(8, 17)
    coarse = torch.zeros(8, 17, dtype=torch.bool)
    coarse[:, [2, 6]] = True
    l_coarse = marginalized_ce(logits, coarse, reduction="none")
    l_hard3 = marginalized_ce(logits, _one_hot_sets([2] * 8), reduction="none")
    l_hard7 = marginalized_ce(logits, _one_hot_sets([6] * 8), reduction="none")
    assert (l_coarse <= torch.minimum(l_hard3, l_hard7) + 1e-6).all()


def test_bitmask_round_trip_matches_export_contract():
    sets = [[3], [3, 7], [8, 10], [17], [1, 2, 3], []]
    bm = torch.from_numpy(encode_lcz_set(sets).astype(np.int64))
    target = bitmask_to_target(bm)
    decoded = [sorted((torch.nonzero(row).flatten() + 1).tolist()) for row in target]
    assert decoded == [sorted(s) for s in sets]


def test_empty_sets_are_excluded_from_the_mean():
    logits = torch.randn(4, 17)
    target = _one_hot_sets([0, 1, 2, 3])
    target[3] = False  # unlabelled
    loss = marginalized_ce(logits, target)
    ref = F.cross_entropy(logits[:3], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(loss, ref)


def test_dense_shape_and_valid_mask():
    logits = torch.randn(2, 17, 4, 4)
    bm = torch.zeros(2, 4, 4, dtype=torch.long)
    bm[0, :2] = 1 << 2   # hard 3 on top half of sample 0
    bm[1] = (1 << 2) | (1 << 6)  # coarse {3,7} everywhere on sample 1
    target = bitmask_to_target(bm).permute(0, 3, 1, 2)
    valid = torch.ones(2, 4, 4, dtype=torch.bool)
    valid[1, 0, 0] = False
    loss = marginalized_ce(logits, target, valid=valid)
    per = marginalized_ce(logits, target, reduction="none")
    keep = (bm != 0) & valid
    torch.testing.assert_close(loss, per[keep].mean())


def test_smoothing_rewards_adjacent_mass_only():
    # True hard 3 (idx 2); model confidently predicts 2 (idx 1, adjacent)
    # vs predicts G (idx 16, not adjacent). Smoothing helps only the former.
    target = _one_hot_sets([2])
    on_adjacent = torch.full((1, 17), -3.0)
    on_adjacent[0, 1] = 4.0
    on_far = torch.full((1, 17), -3.0)
    on_far[0, 16] = 4.0
    plain_adj = marginalized_ce(on_adjacent, target)
    smooth_adj = marginalized_ce(on_adjacent, target, smoothing_eps=0.1)
    plain_far = marginalized_ce(on_far, target)
    smooth_far = marginalized_ce(on_far, target, smoothing_eps=0.1)
    assert smooth_adj < plain_adj
    assert smooth_far >= plain_far - 1e-4
    assert (smooth_adj - plain_adj) < (smooth_far - plain_far)


def test_confidence_gamma_weighting():
    logits = torch.randn(2, 17)
    target = _one_hot_sets([4, 9])
    conf = torch.tensor([1.0, 0.0])
    weighted = marginalized_ce(logits, target, conf=conf, gamma=1.0)
    only_first = F.cross_entropy(logits[:1], torch.tensor([4]))
    torch.testing.assert_close(weighted, only_first)
    # gamma=0 disables confidence weighting entirely
    torch.testing.assert_close(
        marginalized_ce(logits, target, conf=conf, gamma=0.0),
        marginalized_ce(logits, target),
    )
