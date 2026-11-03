"""MIST implementation of Primus and PrimusV2."""

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# MIST imports.
from mist.models.base_model import MISTModel
from mist.models.primus.eva_blocks import Eva
from mist.models.primus.primus_blocks import PatchDecode, PatchEmbed, PatchEmbedDeeper
from mist.models.primus.primus_constants import PrimusConstants as constants

# Encoder submodules, matching nnU-Net/nnssl's "down_projection" stem and "eva".
_ENCODER_PREFIXES = ("down_projection.", "eva.")
_POS_EMBED_KEY = "eva.pos_embed"


class Primus(MISTModel):
    """Transformer-centric 3D segmentation network (Wald et al., 2025).

    The input is split into 8x8x8 tokens, processed by an EVA-02 style
    transformer with learned absolute and 3D rotary position embeddings, and
    decoded back to full resolution by a lightweight transposed-convolution
    decoder. PrimusV2 replaces the single strided-convolution tokenizer with a
    small residual network, which helps on small and fine structures.

    The position embeddings are sized for patch_size, so the model only accepts
    inputs of exactly that size. Encoder parameter names match nnU-Net's
    Primus, so encoder weights from nnU-Net or nnssl checkpoints load by key;
    a learned position embedding pretrained at a different patch size is
    resized by adapt_pretrained_state_dict.

    Attributes:
        down_projection: Tokenizer.
        eva: Transformer encoder.
        up_projection: Patch decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Sequence[int],
        embed_dim: int,
        depth: int,
        num_heads: int,
        use_deeper_tokenizer: bool = False,
        drop_path_rate: float = constants.DROP_PATH_RATE,
    ):
        super().__init__()
        if len(patch_size) != 3:
            raise ValueError(
                f"Primus requires a 3D patch_size, but got {len(patch_size)} dimensions."
            )
        bad_dims = [d for d in patch_size if d % constants.TOKEN_SIZE != 0]
        if bad_dims:
            raise ValueError(
                f"Primus requires all patch_size dimensions to be divisible by "
                f"{constants.TOKEN_SIZE}. Got patch_size={list(patch_size)}, "
                f"offending dimensions: {bad_dims}."
            )

        token_size = (constants.TOKEN_SIZE,) * 3
        self.grid_shape = tuple(d // constants.TOKEN_SIZE for d in patch_size)

        if use_deeper_tokenizer:
            self.down_projection: nn.Module = PatchEmbedDeeper(in_channels, embed_dim)
        else:
            self.down_projection = PatchEmbed(token_size, in_channels, embed_dim)
        self.up_projection = PatchDecode(token_size, embed_dim, out_channels)
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            grid_shape=self.grid_shape,
            drop_path_rate=drop_path_rate,
        )
        # Kept for state dict compatibility with nnU-Net checkpoints, which
        # include the mask token used by masked-image pretraining.
        self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))

        # The transformer initializes itself; the conv layers use He init.
        self.down_projection.apply(self._init_conv)
        self.up_projection.apply(self._init_conv)

    @staticmethod
    def _init_conv(module: nn.Module) -> None:
        """Kaiming normal init with zero bias for (transposed) convolutions."""
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=constants.HE_NEGATIVE_SLOPE)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder weights: tokenizer and transformer."""
        return OrderedDict(
            {k: v for k, v in self.state_dict().items() if k.startswith(_ENCODER_PREFIXES)}
        )

    def adapt_pretrained_state_dict(self, state_dict: dict) -> dict:
        """Resize a pretrained position embedding to this model's token grid.

        Weights pretrained at another patch size have a position embedding for
        a different number of tokens. If the source grid is cubic, the
        embedding is trilinearly resized to this model's grid; otherwise it is
        left unchanged and skipped by the loader for its shape mismatch.

        Args:
            state_dict: Source state dict.

        Returns:
            The state dict with eva.pos_embed resized if needed.
        """
        source = state_dict.get(_POS_EMBED_KEY)
        target_tokens = self.eva.pos_embed.shape[1]
        if source is None or source.shape[1] == target_tokens:
            return state_dict
        side = round(source.shape[1] ** (1 / 3))
        if side**3 != source.shape[1] or source.shape[2] != self.eva.pos_embed.shape[2]:
            return state_dict

        grid = source.reshape(1, side, side, side, -1).permute(0, 4, 1, 2, 3)
        resized = F.interpolate(
            grid.float(), size=self.grid_shape, mode="trilinear", align_corners=False
        )
        state_dict = dict(state_dict)
        state_dict[_POS_EMBED_KEY] = (
            resized.permute(0, 2, 3, 4, 1).reshape(1, target_tokens, -1).to(source.dtype)
        )
        return state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        """Forward pass for Primus.

        Returns:
            In eval mode, the full-resolution prediction. In training mode, a
            dict with the prediction under "prediction" and "deep_supervision"
            set to None (Primus has no deep supervision).
        """
        x = self.down_projection(x)
        b, c, d, h, w = x.shape
        x = self.eva(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(b, c, d, h, w)
        prediction = self.up_projection(x)

        if self.training:
            return {"prediction": prediction, "deep_supervision": None}
        return prediction
