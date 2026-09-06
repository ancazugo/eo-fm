"""Attention U-Net segmentation family.

U-Net with additive attention gates on the skip connections (Oktay et al.,
2018, "Attention U-Net: Learning Where to Look for the Pancreas"). Each gate
takes the upsampled decoder feature as a *gating signal* and uses it to produce
a per-pixel weight in (0, 1) that rescales the encoder skip before the
concatenation, so the decoder can suppress skip regions that are irrelevant at
that stage.

Preset payloads are ``(depth, base_features)`` tuples, **identical to**
:data:`models.unet.UNET_PRESETS`. That is deliberate: the class subclasses
:class:`models.unet.UNet` and adds nothing but the gates, so an
``attention_unet`` row and a ``unet`` row at the same preset differ only by the
attention mechanism (plus the gates' own parameter cost, a measured 1.0-1.1%
of the model at every preset on both embeddings). Any
difference in the resulting metrics is therefore attributable to the gates
rather than to capacity or to a differently-tuned decoder.

Why this family exists — and what to expect. Attention gates were designed for
networks trained from raw intensity images, where the shallow skips carry
low-level edges and texture that the decoder must learn to ignore; the gate is
what arbitrates that encoder-decoder semantic gap. This project feeds the
encoder pre-trained foundation-model embeddings (128-ch Tessera, 64-ch
AlphaEarth), so *every* skip, including the first, already carries semantic
features and there is no low-level tier for a gate to suppress. Measured on the
cultural split, decoder capacity itself is already saturated or reversing
(fcn8 tessera banks 93% of its gain by 477k params; coop peaks at ``base`` and
degrades at ``large``), and the only change that has moved segmentation-adjacent
numbers in this project was a resolution fix, not a topology change. The honest
prior is therefore that this lands inside seed noise against ``unet`` at matched
preset. It is registered so that can be *measured* rather than asserted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import ModelFamily, register
from models.unet import UNet

# Same payloads as UNET_PRESETS, by design: matched-capacity rows against unet.
ATTENTION_UNET_PRESETS: dict[str, tuple[int, int]] = {
    "nano":   (2,  8),
    "small":  (3, 32),
    "base":   (3, 48),
    "medium": (4, 32),
    "large":  (4, 48),
}


class AttentionGate(nn.Module):
    """Additive attention gate over one skip connection.

    Projects the skip and the gating signal into a shared ``inter_channels``
    space, adds them, and squeezes the sum to a single-channel logit map which
    a sigmoid turns into per-pixel weights::

        psi = sigmoid(W_psi(relu(W_skip(skip) + W_gate(gate))))
        out = skip * psi

    Both inputs must already be at the same spatial size; the caller is
    responsible for that (:class:`AttentionUNet` resizes the decoder feature to
    the skip's size first, which is also what handles odd-sized inputs).

    Two deviations from the reference implementation, both house style: the
    projections use ``bias=False`` because a BatchNorm follows and its shift
    subsumes the bias, and ``inter_channels`` defaults to half the skip width
    rather than being passed in.

    Note the gate can only attenuate, never amplify — ``psi`` is bounded by
    (0, 1), and the BatchNorm before the sigmoid puts it near 0.5 at
    initialisation, so skips start at roughly half the magnitude a plain U-Net
    would give them. This is benign here: the ``DoubleConv`` that consumes the
    concatenation opens with a conv+BatchNorm, which absorbs a constant rescale.

    Args:
        skip_channels: Channels in the encoder skip being gated.
        gate_channels: Channels in the decoder gating signal.
        inter_channels: Width of the shared projection space
            (default: ``skip_channels // 2``, floored at 1).
    """

    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, skip_channels // 2)

        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        attention = self.psi(self.relu(self.w_skip(skip) + self.w_gate(gate)))
        return skip * attention


class AttentionUNet(UNet):
    """U-Net whose skip connections pass through additive attention gates.

    Encoder, bottleneck, decoder and head are inherited from
    :class:`models.unet.UNet` unchanged. The only structural addition is one
    :class:`AttentionGate` per decoder stage, applied to the skip *after* the
    decoder feature has been upsampled (and, for odd inputs, resized) to the
    skip's resolution — so the gating signal and the skip are always spatially
    aligned, and the odd-size handling stays identical to the parent's.

    Built-in presets (depth, base_features) — the same ladder as ``unet``:
        nano:   (2,  8)
        small:  (3, 32)
        base:   (3, 48)
        medium: (4, 32)
        large:  (4, 48)

    Args:
        in_channels: Number of embedding input channels.
        num_classes: Number of segmentation output classes.
        depth: Number of encoder/decoder stages.
        base_features: Feature maps at the first encoder stage;
            doubles at each subsequent stage.
        bottleneck_dropout: Dropout2d probability at the bottleneck.
    """

    PRESETS = ATTENTION_UNET_PRESETS

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depth: int = 3,
        base_features: int = 32,
        bottleneck_dropout: float = 0.3,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            depth=depth,
            base_features=base_features,
            bottleneck_dropout=bottleneck_dropout,
        )

        # One gate per decoder stage, in decoder order. Widths are read off the
        # upsample layers the parent already built rather than recomputed from
        # base_features, so the gates cannot drift out of step with the parent
        # if its channel schedule ever changes: after up(), the decoder feature
        # carries out_channels, which is also the skip's width.
        self.gates = nn.ModuleList(
            AttentionGate(skip_channels=up.out_channels, gate_channels=up.out_channels)
            for up in self.upsamples
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, gate, dec, skip in zip(
            self.upsamples, self.gates, self.decoders, reversed(skips)
        ):
            x = up(x)
            # Correct for odd-sized inputs (bilinear resize if needed). Must run
            # before the gate: it adds the two tensors pixel-wise.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = gate(skip, x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


def build_attention_unet(
    arch: tuple[int, int],
    in_channels: int,
    num_classes: int,
    bottleneck_dropout: float = 0.3,
) -> nn.Module:
    """``arch`` is the ``(depth, base_features)`` preset payload."""
    depth, base_features = arch
    return AttentionUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        depth=depth,
        base_features=base_features,
        bottleneck_dropout=bottleneck_dropout,
    )


register(ModelFamily(
    name="attention_unet",
    pipeline="segmentation",
    presets=ATTENTION_UNET_PRESETS,
    build=build_attention_unet,
))
