"""EVA-02 style transformer blocks with 3D rotary position embeddings.

Adapted from timm (``timm.models.eva`` and ``timm.layers``, Apache License 2.0)
and MIC-DKFZ's dynamic-network-architectures (Apache License 2.0), keeping only
what Primus uses: unfused q/k/v projections, scaled-dot-product attention,
axial 3D rotary embeddings for a fixed token grid, a SwiGLU MLP with an inner
norm, LayerScale, and stochastic depth. Module and attribute names mirror the
originals so that state dict keys match nnU-Net and nnssl checkpoints.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# MIST imports.
from mist.models.primus.primus_constants import PrimusConstants as constants


def build_rotary_embedding(
    grid_shape: Sequence[int], rope_dim: int, temperature: float = constants.ROPE_TEMPERATURE
) -> torch.Tensor:
    """Build concatenated sin/cos rotary embeddings for a 3D token grid.

    Each spatial axis gets rope_dim // 4 frequency bands. The result has shape
    (num_tokens, 2 * head_dim), where head_dim = 1.5 * rope_dim, with the sin
    half first. Values are interleaved pairwise to match rotate_pairs().

    Args:
        grid_shape: Number of tokens along each spatial axis.
        rope_dim: Rotary dimension, i.e. head_dim / 1.5.
        temperature: Frequency base.

    Returns:
        Tensor of shape (prod(grid_shape), 3 * rope_dim).
    """
    num_bands = rope_dim // 4
    exponents = torch.arange(0, num_bands, dtype=torch.int64).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exponents)
    # Positions are scaled by size / reference size; both are the grid shape,
    # which keeps the arithmetic identical to the reference implementation.
    positions = [torch.arange(s, dtype=torch.int64).to(torch.float32) / s * s for s in grid_shape]
    grid = torch.stack(torch.meshgrid(positions, indexing="ij"), dim=-1).unsqueeze(-1)
    angles = grid * bands
    num_tokens = math.prod(grid_shape)
    sin = angles.sin().reshape(num_tokens, -1).repeat_interleave(2, -1)
    cos = angles.cos().reshape(num_tokens, -1).repeat_interleave(2, -1)
    return torch.cat([sin, cos], dim=-1)


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]."""
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rotary_embedding(x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    """Apply concatenated sin/cos rotary embeddings to queries or keys."""
    sin, cos = embedding.chunk(2, -1)
    return x * cos + rotate_pairs(x) * sin


class RotaryEmbedding3D(nn.Module):
    """Fixed-grid 3D rotary position embeddings.

    The embeddings are precomputed for the token grid and stored as a
    non-persistent buffer, so they never appear in checkpoints.
    """

    def __init__(self, rope_dim: int, grid_shape: Sequence[int]):
        super().__init__()
        self.register_buffer(
            "pos_embed", build_rotary_embedding(grid_shape, rope_dim), persistent=False
        )

    def get_embed(self) -> torch.Tensor:
        """Return the precomputed embeddings."""
        return self.pos_embed


class DropPath(nn.Module):
    """Stochastic depth: drop the residual branch per sample during training."""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero whole samples and rescale the survivors."""
        if not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        return x * mask / keep_prob


class SwiGLU(nn.Module):
    """SwiGLU MLP with separate gate/value projections and an inner LayerNorm."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1_g = nn.Linear(dim, hidden_dim)
        self.fc1_x = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_dim, eps=constants.LAYER_NORM_EPS)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP."""
        return self.fc2(self.norm(self.act(self.fc1_g(x)) * self.fc1_x(x)))


class EvaAttention(nn.Module):
    """Multi-head self-attention with rotary embeddings and no key bias.

    Attributes:
        q_proj, k_proj, v_proj: Query, key (no bias), and value projections.
        norm: LayerNorm on the attention output if scale_attn_inner is set,
            otherwise identity.
        proj: Output projection.
    """

    def __init__(self, dim: int, num_heads: int, scale_attn_inner: bool):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.norm = (
            nn.LayerNorm(dim, eps=constants.LAYER_NORM_EPS) if scale_attn_inner else nn.Identity()
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None) -> torch.Tensor:
        """Attend over all tokens.

        Args:
            x: Tokens of shape (B, N, C).
            rope: Rotary embeddings of shape (N, 2 * head_dim), or None.

        Returns:
            Tensor of shape (B, N, C).
        """
        b, n, c = x.shape
        q, k, v = (
            proj(x).reshape(b, n, self.num_heads, -1).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        if rope is not None:
            q = apply_rotary_embedding(q, rope).type_as(v)
            k = apply_rotary_embedding(k, rope).type_as(v)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(self.norm(x))


class EvaBlock(nn.Module):
    """Pre-norm transformer block with LayerScale and stochastic depth."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        scale_attn_inner: bool,
        drop_path: float,
        init_values: float | None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=constants.LAYER_NORM_EPS)
        self.attn = EvaAttention(dim, num_heads, scale_attn_inner)
        self.gamma_1 = (
            nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=constants.LAYER_NORM_EPS)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.gamma_2 = (
            nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None) -> torch.Tensor:
        """Apply attention and MLP residual branches."""
        attn = self.attn(self.norm1(x), rope=rope)
        mlp_input = x + self.drop_path1(attn if self.gamma_1 is None else self.gamma_1 * attn)
        mlp = self.mlp(self.norm2(mlp_input))
        return mlp_input + self.drop_path2(mlp if self.gamma_2 is None else self.gamma_2 * mlp)


class Eva(nn.Module):
    """EVA-02 style transformer over a fixed 3D token grid.

    Uses a learned absolute position embedding and axial 3D rotary embeddings.
    Expects already-embedded tokens (no patch embedding, class token, or head).

    Attributes:
        pos_embed: Learned absolute position embedding, (1, num_tokens, dim).
        rope: Rotary position embeddings for the token grid.
        blocks: Transformer blocks.
        norm: Final LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        grid_shape: Sequence[int],
        mlp_ratio: float = constants.MLP_RATIO,
        scale_attn_inner: bool = True,
        drop_path_rate: float = constants.DROP_PATH_RATE,
        init_values: float | None = constants.LAYER_SCALE_INIT,
    ):
        super().__init__()
        rope_dim = round(embed_dim // num_heads / 1.5)
        if rope_dim != embed_dim / num_heads / 1.5 or rope_dim % 4 != 0:
            raise ValueError(
                "embed_dim / num_heads must be divisible by 6 for 3D rotary "
                f"embeddings, but got {embed_dim} / {num_heads}."
            )

        self.pos_embed = nn.Parameter(torch.zeros(1, math.prod(grid_shape), embed_dim))
        self.rope = RotaryEmbedding3D(rope_dim, grid_shape)
        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                EvaBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    scale_attn_inner,
                    drop_path_rates[i],
                    init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=constants.LAYER_NORM_EPS)
        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated normal init, then depth-scaled output projections."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=constants.INIT_STD)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.pos_embed, std=constants.INIT_STD)
        with torch.no_grad():
            for layer_id, block in enumerate(self.blocks, start=1):
                block.attn.proj.weight.div_(math.sqrt(2.0 * layer_id))
                block.mlp.fc2.weight.div_(math.sqrt(2.0 * layer_id))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process tokens of shape (B, N, C) and return the same shape."""
        x = x + self.pos_embed
        rope = self.rope.get_embed()
        for block in self.blocks:
            x = block(x, rope=rope)
        return self.norm(x)
