"""T4 — light heads over frozen embeddings. All output 17 logits.

A-family (dense, per-pixel): A1 linear, A2 2-layer MLP, A3 a small dilated-conv
head with a multi-scale receptive field. ``MultiScaleWrapper`` implements the
cheap multi-scale-pooling feature (concat the pixel embedding with 100 m/300 m
mean pools) usable under A1/A2 — historically where most LCZ gains live.

B-family (block-as-sample): B1 mean-pool + MLP, B2 learned attention-pool +
MLP, B3 a 2-layer GNN over block adjacency (``torch_geometric``, optional —
everything else in this module runs without it).

No UCP regression heads in this phase (kept out of scope); a future auxiliary
head can hang off any A/B backbone's pooled/pixel features without touching
these classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LCZ = 17


class A1Linear(nn.Module):
    """Per-pixel linear probe: a single 1x1 conv, C -> 17."""

    def __init__(self, in_channels: int, num_classes: int = N_LCZ):
        super().__init__()
        self.fc = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class A2MLP(nn.Module):
    """Per-pixel 2-layer MLP (1x1 convs), C -> hidden -> 17."""

    def __init__(self, in_channels: int, num_classes: int = N_LCZ, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class A3DilatedConv(nn.Module):
    """Small dilated-conv head: 3x3 convs at dilations 1/2/4, <=~1M params.

    A learned-context alternative to the pooled multi-scale feature — the
    experiment ladder compares the two directly (A2+MS vs A3).
    """

    def __init__(self, in_channels: int, num_classes: int = N_LCZ, width: int = 48):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else width, width, kernel_size=3,
                          padding=d, dilation=d),
                nn.ReLU(inplace=True),
            )
            for i, d in enumerate((1, 2, 4))
        ])
        self.fc = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.fc(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MultiScaleWrapper(nn.Module):
    """Concat the pixel embedding with 100 m and 300 m mean pools, then wrap.

    ``pixel_res_m`` is the embedding's ground resolution (10 m for Tessera/
    AlphaEarth), so a 100 m pool is a ``round(100/pixel_res_m)``-px average.
    Wraps an A1/A2-style head built for ``3 * in_channels`` input channels.
    """

    def __init__(self, base: nn.Module, in_channels: int, pixel_res_m: float = 10.0):
        super().__init__()
        self.base = base
        self.k100 = max(1, round(100.0 / pixel_res_m))
        self.k300 = max(1, round(300.0 / pixel_res_m))

    @staticmethod
    def _pool(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 1:
            return x
        pooled = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2,
                              count_include_pad=False)
        return pooled[..., : x.shape[-2], : x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x, self._pool(x, self.k100), self._pool(x, self.k300)], dim=1)
        return self.base(feat)


class B1MeanPoolMLP(nn.Module):
    """Block head: mean-pooled embedding (+ extra features) -> MLP -> 17."""

    def __init__(self, in_features: int, num_classes: int = N_LCZ, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Learned single-query attention pool over a ragged set of pixel embeddings.

    Input: ``(B, K, C)`` padded pixel-embedding sets + ``(B, K)`` bool mask
    (True = real pixel). Output: ``(B, C)`` pooled embedding.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.query(x).squeeze(-1)                       # (B, K)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)     # (B, K, 1)
        weights = torch.nan_to_num(weights, nan=0.0)              # all-masked rows
        return (weights * x).sum(dim=1)


class B2AttentionPoolMLP(nn.Module):
    """Block head: learned attention-pooled embedding (+ extras) -> MLP -> 17.

    Node feature = ``concat(attention_pool(pixel_embeddings), extra_features)``,
    matching :class:`B1MeanPoolMLP`'s ``in_features`` contract for ``extra``.
    """

    def __init__(self, channels: int, extra_features: int = 0,
                num_classes: int = N_LCZ, hidden: int = 128):
        super().__init__()
        self.pool = AttentionPool(channels)
        self.net = nn.Sequential(
            nn.Linear(channels + extra_features, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, pixels: torch.Tensor, mask: torch.Tensor,
               extra: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.pool(pixels, mask)
        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)
        return self.net(pooled)


def _import_torch_geometric():
    try:
        import torch_geometric.nn as gnn
    except ImportError as e:
        raise ImportError(
            "B3GNN requires torch_geometric — install the optional 'gnn' extra "
            "(uv sync --extra gnn); B1/B2 and all of A run without it."
        ) from e
    return gnn


class B3GNN(nn.Module):
    """Block head: 2-layer GraphSAGE over block adjacency -> 17.

    Optional dependency (``torch_geometric``, the ``gnn`` extra) — importing
    this class without it raises a clear ImportError; every other model in
    this module is unaffected. Built from Stage 4a adjacency restricted to the
    AOI, 2-hop neighbourhoods (edge construction lives in ``datasets.py``).
    """

    def __init__(self, in_features: int, num_classes: int = N_LCZ, hidden: int = 128):
        super().__init__()
        gnn = _import_torch_geometric()
        self.conv1 = gnn.SAGEConv(in_features, hidden)
        self.conv2 = gnn.SAGEConv(hidden, hidden)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x)


def is_gnn_available() -> bool:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True
