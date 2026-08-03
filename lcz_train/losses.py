"""T3 — one masked, marginalised cross-entropy for every model (A and B).

With a per-sample/per-pixel label set *S* (from the uint32 bitmask contract),

    loss = -log( sum_{c in S} p_c )

averaged over valid entries. Hard labels are the |S| = 1 case, so there is a
single code path everywhere (never collapse a coarse label to one member!).

Options, all off by default: per-class weights (a sample weighs the mean weight
of its set, matching ``F.cross_entropy(weight=...)`` semantics in the hard
case), label smoothing restricted to morphologically adjacent classes, and
confidence weighting ``loss * confidence**gamma``.
"""

from __future__ import annotations

import torch

N_LCZ = 17

# Morphologically adjacent LCZ pairs (codes 1-17; A-G = 11-17). Smoothing mass
# may only leak along these edges: compact/open ladders, 3<->7, 8<->10, B-C-D.
_ADJACENT_PAIRS = [
    (1, 2), (2, 3), (4, 5), (5, 6), (3, 7), (8, 10), (12, 13), (13, 14),
]

_ADJ = torch.zeros(N_LCZ, N_LCZ)
for _a, _b in _ADJACENT_PAIRS:
    _ADJ[_a - 1, _b - 1] = 1.0
    _ADJ[_b - 1, _a - 1] = 1.0


def bitmask_to_target(bitmask: torch.Tensor, num_classes: int = N_LCZ) -> torch.Tensor:
    """uint32/int bitmask (...,) -> boolean set mask (..., C).

    Bit ``c-1`` of the bitmask marks class code ``c`` (class index ``c-1``).
    0 (unlabelled) yields an all-False row — mask those out via ``valid``.
    """
    bits = torch.arange(num_classes, device=bitmask.device, dtype=torch.long)
    return (bitmask.long().unsqueeze(-1) >> bits) & 1 > 0


def marginalized_ce(
    logits: torch.Tensor,
    target_set: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    smoothing_eps: float = 0.0,
    conf: torch.Tensor | None = None,
    gamma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Marginalised CE over label sets.

    Args:
        logits: ``(B, C)`` or ``(B, C, H, W)``.
        target_set: boolean/0-1 set membership, same shape as ``logits``
            (class dim included) — build from the bitmask raster via
            :func:`bitmask_to_target` (move the class dim to position 1 for
            dense inputs).
        valid: optional bool ``(B,)`` / ``(B, H, W)``; entries with an empty
            set are always dropped regardless.
        class_weights: ``(C,)``; a sample's weight is the mean weight of its
            set (hard case == ``F.cross_entropy(weight=...)``), and the
            weighted mean divides by the summed weights, matching torch.
        smoothing_eps: mass ``eps`` moved from the set onto classes adjacent
            (per Stewart & Oke morphology) to any set member and outside it.
        conf: per-entry confidence in [0, 1]; scales the loss by
            ``conf ** gamma`` (``gamma=0`` disables).
        reduction: "mean" (weighted), "sum" or "none".
    """
    if logits.shape != target_set.shape:
        raise ValueError(f"logits {tuple(logits.shape)} vs target_set {tuple(target_set.shape)}")
    logp = torch.log_softmax(logits, dim=1)
    target = target_set.to(logits.dtype)

    nonempty = target.sum(dim=1) > 0
    valid = nonempty if valid is None else (valid.bool() & nonempty)

    if not bool(nonempty.all()):
        # An all-empty target row (padded/invalid pixels) makes every class's
        # logw = -inf, so logsumexp sees an all -inf row. Its backward needs
        # x - max(x), and max(x) is ALSO -inf there, giving -inf-(-inf) = nan
        # — this poisons the gradient inside logsumexp itself, before the
        # `valid` masking below ever runs, so masking the forward value can't
        # undo it. Substitute a harmless one-hot placeholder (class 0) for
        # empty rows; `valid` still zeroes their contribution to the loss.
        dummy = torch.zeros_like(target)
        dummy.select(1, 0).fill_(1.0)
        target = torch.where(nonempty.unsqueeze(1), target, dummy)

    weights_shape = [1, N_LCZ] + [1] * (logits.dim() - 2)
    w = target
    if smoothing_eps > 0.0:
        adj = _ADJ.to(device=logits.device, dtype=logits.dtype)
        # neighbours of any set member, excluding the set itself
        neigh = (torch.einsum("bc...,cd->bd...", target, adj) > 0).to(logits.dtype)
        neigh = neigh * (1.0 - target)
        n_neigh = neigh.sum(dim=1, keepdim=True).clamp_min(1.0)
        w = target * (1.0 - smoothing_eps) + neigh * (smoothing_eps / n_neigh)
        # sets with no adjacent classes keep full mass
        has_neigh = (neigh.sum(dim=1, keepdim=True) > 0).to(logits.dtype)
        w = w + target * smoothing_eps * (1.0 - has_neigh)

    # -log sum_c w_c p_c, computed stably as -logsumexp(logp + log w)
    logw = torch.where(w > 0, w.clamp_min(1e-12).log(), torch.full_like(w, -torch.inf))
    loss = -torch.logsumexp(logp + logw, dim=1)

    sample_w = torch.ones_like(loss)
    if class_weights is not None:
        cw = class_weights.to(device=logits.device, dtype=logits.dtype).view(weights_shape)
        sample_w = (target * cw).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
    if conf is not None and gamma != 0.0:
        sample_w = sample_w * conf.to(logits.dtype).clamp(0.0, 1.0).pow(gamma)

    loss = loss * sample_w
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    denom = torch.where(valid, sample_w, torch.zeros_like(sample_w)).sum().clamp_min(1e-12)
    return loss.sum() / denom
