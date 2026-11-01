"""Tests for the EVA transformer blocks used by Primus."""

import pytest
import torch
from torch import nn

# MIST imports.
from mist.models.primus.eva_blocks import (
    DropPath,
    Eva,
    EvaAttention,
    EvaBlock,
    apply_rotary_embedding,
    build_rotary_embedding,
    rotate_pairs,
)


def test_build_rotary_embedding_shape():
    """Embeddings cover every token with sin and cos halves of head_dim each."""
    embedding = build_rotary_embedding((2, 3, 4), rope_dim=16)
    head_dim = 24  # 1.5 * rope_dim
    assert embedding.shape == (2 * 3 * 4, 2 * head_dim)
    sin, cos = embedding.chunk(2, -1)
    assert torch.allclose(sin**2 + cos**2, torch.ones_like(sin))


def test_rotate_pairs():
    """Pairs are rotated as [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(rotate_pairs(x), torch.tensor([-2.0, 1.0, -4.0, 3.0]))


def test_apply_rotary_embedding_preserves_norm():
    """Rotary embeddings rotate pairs, so vector norms are unchanged."""
    embedding = build_rotary_embedding((2, 2, 2), rope_dim=16)
    x = torch.randn(1, 2, 8, 24)
    rotated = apply_rotary_embedding(x, embedding)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_drop_path_identity_in_eval_and_drops_in_train():
    """DropPath is a no-op in eval and zeroes or rescales whole samples in train."""
    layer = DropPath(0.5)
    x = torch.ones(64, 3)
    layer.eval()
    assert torch.equal(layer(x), x)
    layer.train()
    torch.manual_seed(0)
    out = layer(x)
    assert set(out[:, 0].tolist()) <= {0.0, 2.0}
    assert torch.all(out == out[:, :1])


def test_attention_inner_norm_toggle():
    """scale_attn_inner adds a LayerNorm; the key projection has no bias."""
    with_norm = EvaAttention(48, 2, scale_attn_inner=True)
    without_norm = EvaAttention(48, 2, scale_attn_inner=False)
    assert isinstance(with_norm.norm, nn.LayerNorm)
    assert isinstance(without_norm.norm, nn.Identity)
    assert with_norm.k_proj.bias is None
    out = without_norm(torch.randn(2, 8, 48), rope=None)
    assert out.shape == (2, 8, 48)


def test_block_layer_scale_and_drop_path_toggles():
    """LayerScale and DropPath are only built when requested."""
    plain = EvaBlock(48, 2, 4 * 2 / 3, True, drop_path=0.0, init_values=None)
    scaled = EvaBlock(48, 2, 4 * 2 / 3, True, drop_path=0.1, init_values=0.1)
    assert plain.gamma_1 is None and isinstance(plain.drop_path1, nn.Identity)
    assert torch.allclose(scaled.gamma_1, torch.full((48,), 0.1))
    assert isinstance(scaled.drop_path1, DropPath)
    rope = build_rotary_embedding((2, 2, 2), rope_dim=16)
    x = torch.randn(1, 8, 48)
    assert plain(x, rope).shape == scaled(x, rope).shape == (1, 8, 48)


def test_eva_forward_and_rope_not_persisted():
    """Eva keeps token shape; rotary embeddings are not saved in the state dict."""
    eva = Eva(embed_dim=48, depth=2, num_heads=2, grid_shape=(2, 2, 2))
    out = eva(torch.randn(1, 8, 48))
    assert out.shape == (1, 8, 48)
    assert "pos_embed" in eva.state_dict()
    assert not any(k.startswith("rope.") for k in eva.state_dict())


def test_eva_invalid_head_dim_raises():
    """head_dim must be divisible by 6 for 3D rotary embeddings."""
    with pytest.raises(ValueError, match="divisible by 6"):
        Eva(embed_dim=40, depth=1, num_heads=2, grid_shape=(2, 2, 2))
