"""T4 — model shape/param checks; B3 skips cleanly without torch_geometric."""

import pytest
import torch

from lcz_train.models import (
    A1Linear,
    A2MLP,
    A3DilatedConv,
    AttentionPool,
    B1MeanPoolMLP,
    B2AttentionPoolMLP,
    B3GNN,
    MultiScaleWrapper,
    is_gnn_available,
)

N_LCZ = 17


@pytest.mark.parametrize("Model", [A1Linear, A2MLP, A3DilatedConv])
def test_dense_heads_output_shape(Model):
    x = torch.randn(2, 16, 8, 8)
    model = Model(in_channels=16)
    out = model(x)
    assert out.shape == (2, N_LCZ, 8, 8)


def test_a3_param_budget():
    model = A3DilatedConv(in_channels=128)
    assert model.num_params() <= 1_000_000


def test_multiscale_wrapper_shape_and_gradient():
    x = torch.randn(2, 16, 12, 12, requires_grad=True)
    base = A1Linear(in_channels=16 * 3)
    wrapped = MultiScaleWrapper(base, in_channels=16, pixel_res_m=10.0)
    out = wrapped(x)
    assert out.shape == (2, N_LCZ, 12, 12)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_multiscale_wrapper_pool_sizes():
    wrapped = MultiScaleWrapper(A1Linear(48), in_channels=16, pixel_res_m=10.0)
    assert wrapped.k100 == 10 and wrapped.k300 == 30


def test_multiscale_a2_end_to_end():
    x = torch.randn(1, 8, 16, 16)
    wrapped = MultiScaleWrapper(A2MLP(in_channels=24), in_channels=8, pixel_res_m=10.0)
    assert wrapped(x).shape == (1, N_LCZ, 16, 16)


def test_b1_mean_pool_mlp():
    x = torch.randn(5, 128)
    model = B1MeanPoolMLP(in_features=128)
    assert model(x).shape == (5, N_LCZ)


def test_attention_pool_respects_mask():
    x = torch.zeros(2, 4, 8)
    x[0, 0] = 1.0    # sample 0: only pixel 0 valid, value 1
    x[1, 1] = 5.0    # sample 1: only pixel 1 valid, value 5
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 1] = True
    pool = AttentionPool(channels=8)
    out = pool(x, mask)
    torch.testing.assert_close(out[0], x[0, 0])
    torch.testing.assert_close(out[1], x[1, 1])


def test_attention_pool_all_masked_is_finite():
    x = torch.randn(1, 3, 8)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    pool = AttentionPool(channels=8)
    out = pool(x, mask)
    assert torch.isfinite(out).all()


def test_b2_attention_pool_mlp_with_extra_features():
    pixels = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[0, 2:] = False    # sample 0 has only 2 valid pixels
    extra = torch.randn(3, 4)   # e.g. area, compactness, elongation, +1
    model = B2AttentionPoolMLP(channels=16, extra_features=4)
    out = model(pixels, mask, extra)
    assert out.shape == (3, N_LCZ)


def test_b2_without_extra_features():
    pixels = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    model = B2AttentionPoolMLP(channels=16, extra_features=0)
    assert model(pixels, mask).shape == (2, N_LCZ)


def test_gnn_unavailable_in_this_env():
    # Documents the CI/dev environment state this ladder runs against: the
    # optional 'gnn' extra (torch_geometric) is declared but not installed.
    assert is_gnn_available() is False


def test_b3_raises_clear_error_without_torch_geometric():
    with pytest.raises(ImportError, match="torch_geometric"):
        B3GNN(in_features=128)


def test_all_dense_and_block_heads_construct_without_gnn_extra():
    # B1/B2 and all of A must be usable even though torch_geometric is absent.
    A1Linear(16); A2MLP(16); A3DilatedConv(16)
    B1MeanPoolMLP(128); B2AttentionPoolMLP(128)
