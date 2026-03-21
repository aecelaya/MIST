"""MIST-compatible base implementation of MedNeXt."""
from collections import OrderedDict
from typing import Union, Dict
from collections.abc import Sequence
import torch
import torch.nn as nn

# MIST imports.
from mist.models.base_model import MISTModel
from mist.models.mednext.mednext_blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    MedNeXtOutBlock,
)


class MedNeXt(MISTModel):
    """Base MedNeXt architecture."""
    def __init__(
        self,
        spatial_dims: int=3,
        init_filters: int=32,
        in_channels: int=1,
        out_channels: int=2,
        encoder_expansion_ratio: Union[Sequence[int], int]=2,
        decoder_expansion_ratio: Union[Sequence[int], int]=2,
        bottleneck_expansion_ratio: int=2,
        kernel_size: int=7,
        use_deep_supervision: bool=False,
        use_residual_blocks: bool=False,
        blocks_down: Sequence[int]=(2, 2, 2, 2),
        blocks_bottleneck: int=2,
        blocks_up: Sequence[int]=(2, 2, 2, 2),
        norm_type: str="group",
        global_resp_norm: bool=False,
        use_pocket_model: bool=False,
    ):
        """Initialize the MedNeXt model.

        Args:
            spatial_dims: Number of spatial dimensions (2 or 3).
            init_filters: Number of initial filters.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            encoder_expansion_ratio: Expansion ratio for encoder blocks.
            decoder_expansion_ratio: Expansion ratio for decoder blocks.
            bottleneck_expansion_ratio: Expansion ratio for bottleneck blocks.
            kernel_size: Kernel size for convolutional layers.
            use_deep_supervision: Whether to use deep supervision.
            use_residual_blocks: Whether to use residual connections.
            blocks_down: Number of blocks in each encoder stage.
            blocks_bottleneck: Number of blocks in the bottleneck stage.
            blocks_up: Number of blocks in each decoder stage.
            norm_type: Normalization type (e.g., "group", "batch").
            global_resp_norm: Whether to use global response normalization.
            use_pocket_model: Whether to use the pocket version of MedNeXT.
        """
        super().__init__()
        # Set up basic parameters.
        self.use_deep_supervision = use_deep_supervision
        if spatial_dims not in [2, 3]:
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        spatial_dims_str = f"{spatial_dims}d"
        enc_kernel_size = dec_kernel_size = kernel_size
        filters_multiplier = 1 if use_pocket_model else 2

        if isinstance(encoder_expansion_ratio, int):
            encoder_expansion_ratio = (
                [encoder_expansion_ratio] * len(blocks_down)
            )
        if isinstance(decoder_expansion_ratio, int):
            decoder_expansion_ratio = [decoder_expansion_ratio] * len(blocks_up)

        conv = nn.Conv2d if spatial_dims_str == "2d" else nn.Conv3d
        self.stem = conv(in_channels, init_filters, kernel_size=1)

        # Encoder.
        enc_stages = []
        down_blocks = []
        for i, num_blocks in enumerate(blocks_down):
            enc_stages.append(
                nn.Sequential(*[
                    MedNeXtBlock(
                        in_channels=init_filters * (filters_multiplier ** i),
                        out_channels=init_filters * (filters_multiplier ** i),
                        expansion_ratio=encoder_expansion_ratio[i],
                        kernel_size=enc_kernel_size,
                        use_residual_connection=use_residual_blocks,
                        norm_type=norm_type,
                        dim=spatial_dims_str,
                        global_resp_norm=global_resp_norm,
                    )
                    for _ in range(num_blocks)
                ])
            )
            down_blocks.append(
                MedNeXtDownBlock(
                    in_channels=init_filters * (filters_multiplier ** i),
                    out_channels=init_filters * (filters_multiplier ** (i + 1)),
                    expansion_ratio=encoder_expansion_ratio[i],
                    kernel_size=enc_kernel_size,
                    use_residual_connection=use_residual_blocks,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                )
            )
        self.enc_stages = nn.ModuleList(enc_stages)
        self.down_blocks = nn.ModuleList(down_blocks)

        # Bottleneck.
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=(
                    init_filters * (filters_multiplier ** len(blocks_down))
                ),
                out_channels=(
                    init_filters * (filters_multiplier ** len(blocks_down))
                ),
                expansion_ratio=bottleneck_expansion_ratio,
                kernel_size=dec_kernel_size,
                use_residual_connection=use_residual_blocks,
                norm_type=norm_type,
                dim=spatial_dims_str,
                global_resp_norm=global_resp_norm,
            )
            for _ in range(blocks_bottleneck)
        ])

        # Decoder.
        up_blocks = []
        dec_stages = []
        for i, num_blocks in enumerate(blocks_up):
            up_blocks.append(
                MedNeXtUpBlock(
                    in_channels=(
                        init_filters *
                        (filters_multiplier ** (len(blocks_up) - i))
                    ),
                    out_channels=(
                        init_filters *
                        (filters_multiplier ** (len(blocks_up) - i - 1))
                    ),
                    expansion_ratio=decoder_expansion_ratio[i],
                    kernel_size=dec_kernel_size,
                    use_residual_connection=use_residual_blocks,
                    norm_type=norm_type,
                    dim=spatial_dims_str,
                    global_resp_norm=global_resp_norm,
                )
            )
            dec_stages.append(
                nn.Sequential(*[
                    MedNeXtBlock(
                        in_channels=(
                            init_filters *
                            (filters_multiplier ** (len(blocks_up) - i - 1))
                        ),
                        out_channels=(
                            init_filters *
                            (filters_multiplier ** (len(blocks_up) - i - 1))
                        ),
                        expansion_ratio=decoder_expansion_ratio[i],
                        kernel_size=dec_kernel_size,
                        use_residual_connection=use_residual_blocks,
                        norm_type=norm_type,
                        dim=spatial_dims_str,
                        global_resp_norm=global_resp_norm,
                    )
                    for _ in range(num_blocks)
                ])
            )
        self.up_blocks = nn.ModuleList(up_blocks)
        self.dec_stages = nn.ModuleList(dec_stages)

        # Output.
        self.out_0 = MedNeXtOutBlock(
            in_channels=init_filters,
            n_classes=out_channels,
            dim=spatial_dims_str
        )

        # Deep supervision output blocks.
        if use_deep_supervision:
            out_blocks = [
                MedNeXtOutBlock(
                    in_channels=init_filters * (filters_multiplier ** i),
                    n_classes=out_channels,
                    dim=spatial_dims_str
                )
                for i in range(1, len(blocks_up) + 1)
            ]
            out_blocks.reverse()
            self.out_blocks = nn.ModuleList(out_blocks)

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder weights: stem, encoder stages, down blocks, bottleneck."""
        encoder_prefixes = (
            "stem.",
            "enc_stages.",
            "down_blocks.",
            "bottleneck.",
        )
        return OrderedDict(
            {k: v for k, v in self.state_dict().items()
             if k.startswith(encoder_prefixes)}
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict]:
        """Forward pass of the MedNeXt model.

        Args:
            x: Input tensor of shape (B, C, D, H, W) or (B, C, H, W).

        Returns:
            Output tensor or dictionary with predictions and deep supervision
                outputs if in training mode.
        """
        # Apply stem convolution and start encoding.
        x = self.stem(x)
        enc_outputs = []

        for enc_stage, down_block in zip(self.enc_stages, self.down_blocks):
            x = enc_stage(x)
            enc_outputs.append(x)
            x = down_block(x)

        # Apply bottleneck.
        x = self.bottleneck(x)

        # Start decoding and deep supervision.
        if self.use_deep_supervision:
            ds_outputs = []

        for i, (up_block, dec_stage) in enumerate(
            zip(self.up_blocks, self.dec_stages)
        ):
            if self.use_deep_supervision and i < len(self.out_blocks):
                ds_outputs.append(self.out_blocks[i](x)) # pylint: disable=used-before-assignment

            x = up_block(x)
            x = x + enc_outputs[-(i + 1)]
            x = dec_stage(x)

        # Final output.
        x = self.out_0(x)

        # MIST-compatible output.
        if self.training:
            output = {}
            output["prediction"] = x

            if self.use_deep_supervision:
                output["deep_supervision"] = []

                # Reverse the order of deep supervision outputs to go in order
                # from higher resolution to lower resolution.
                ds_outputs = ds_outputs[::-1]

                # Resize deep supervision outputs to match the input shape.
                for ds_output in ds_outputs:
                    output["deep_supervision"].append(
                        nn.functional.interpolate(ds_output, x.shape[2:])
                    )
            else:
                output["deep_supervision"] = None
        else:
            output = x

        return output
