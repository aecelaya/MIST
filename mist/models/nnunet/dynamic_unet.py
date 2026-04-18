"""Modified version of MONAI's Dynamic UNet implementation for MIST.

This module contains a modified version of MONAI's implementation of the
Dynamic UNet architecture. This module is designed to be used with the MIST
and is (in our opinion) more readable than the original MONAI implementation.
This implementation is based on the original MONAI implementation, but has been
modified to be compatible with the MIST.
"""
from collections.abc import Sequence
from typing import Union, Tuple, Optional, Dict
import torch
from torch import nn
import torch.nn.functional as F

# MONAI imports.
from monai.networks.blocks import dynunet_block as dynamic_unet_blocks
from mist.models.nnunet.nnunet_constants import NNUnetConstants as constants


class DynamicUNet(nn.Module):
    """Dynamic UNet architecture.

    This class implements the Dynamic UNet architecture for image segmentation.
    This architecture is based on the UNet architecture with the addition of
    dynamically selecting the parameters of the convolutional blocks based on
    the input image size and spacing. This class also supports deep supervision,
    where intermediate outputs are generated at different depths of the network.
    Deep supervision is useful for training deep networks by providing
    additional supervision signals to the intermediate layers of the network.

    If training, the forward method returns a dictionary containing the
    prediction and deep supervision outputs. If not training, the forward method
    returns the prediction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Sequence[int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        dropout: Optional[Union[Tuple, str, float]] = None,
        use_deep_supervision: bool = False,
        num_deep_supervision_heads: int = 2,
        use_residual_blocks: bool = False,
        trans_bias: bool = False,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.conv_block = (
            dynamic_unet_blocks.UnetResBlock if use_residual_blocks
            else dynamic_unet_blocks.UnetBasicBlock
        )
        self.trans_bias = trans_bias
        self.filters = filters
        self.check_filters()
        self.check_kernel_stride()
        self.input_block = self.get_input_block()
        self.encoder_layers = self.get_encoder_layers()
        self.bottleneck = self.get_bottleneck()
        self.decoder_layers = self.get_decoder_layers()
        self.output_block = self.get_output_block(0)

        # Set up deep supervision. This is only used during training.
        # Deep supervision is used to provide additional supervision signals to
        # the intermediate layers of the network.
        self.use_deep_supervision = use_deep_supervision
        self.num_deep_supervision_heads = num_deep_supervision_heads
        if self.use_deep_supervision:
            # Check that the deep supervision parameters are valid.
            self.check_deep_supervision_parameters()

            # Get the head ids for deep supervision. This helps us to identify
            # the intermediate outputs that we want to use for deep supervision.
            num_upsample_layers = len(self.strides) - 1
            deep_supervision_head_id_start = max(
                0, num_upsample_layers - self.num_deep_supervision_heads - 1
            )
            self.deep_supervision_head_ids = [
                id for id in range(
                    deep_supervision_head_id_start, num_upsample_layers - 1
                )
            ]
            self.deep_supervision_heads = self.get_deep_supervision_heads()

        # Initialize the weights of the network with Kaiming normal
        # initialization.
        self.apply(self.initialize_weights)

    def check_kernel_stride(self):
        """Check that the kernel size and stride are valid."""
        error_message = (
            "Length of kernel_size and strides should be the same, and no less "
            "than 3."
        )
        if (
            len(self.kernel_size) != len(self.strides) or
            len(self.kernel_size) < 3
        ):
            raise ValueError(error_message)

        for idx, k_i in enumerate(self.kernel_size):
            kernel, stride = k_i, self.strides[idx]
            if not isinstance(kernel, int):
                if len(kernel) != 3:
                    raise ValueError(
                        f"Length of kernel_size in block {idx} should be 3 "
                        "(one per spatial dimension)."
                    )
            if not isinstance(stride, int):
                if len(stride) != 3:
                    raise ValueError(
                        f"Length of stride in block {idx} should be 3 "
                        "(one per spatial dimension)."
                    )

    def check_deep_supervision_parameters(self):
        """Check that the deep supervision parameters are valid."""
        num_up_layers = len(self.strides) - 1
        if self.num_deep_supervision_heads >= num_up_layers:
            raise ValueError(
                "num_deep_supervision_heads should be less than the number of "
                "upsampling layers."
            )
        if self.num_deep_supervision_heads < 1:
            raise ValueError(
                "num_deep_supervision_heads should be larger than 0."
            )

    def check_filters(self):
        """Check that the number of filters is valid."""
        if len(self.filters) < len(self.strides):
            raise ValueError(
                "The length of filters should be no less than the length of "
                "strides."
            )
        else:
            self.filters = self.filters[: len(self.strides)]

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict]:
        """Forward pass for the Dynamic UNet architecture."""
        skips = []

        # Input block. This is the first convolutional block in the network.
        # This block is used to set the number of channels to a standard initial
        # value.
        x = self.input_block(x)

        # Append the output input block to the skip connections.
        skips.append(x)

        # Start the encoder. The encoder is a series of convolutional blocks
        # that downsample the input image.
        for encoder_block in self.encoder_layers:
            x = encoder_block(x)
            skips.append(x)

        # The bottleneck layer. This layer is the bottom of the network and
        # contains the coarsest representation of the input image.
        x = self.bottleneck(x)

        # Start the decoder. The decoder is a series of convolutional blocks
        # that upsample the input image. The skip connections are concatenated
        # to the input of each decoder block. We reverse the skip connections
        # list to match the order of the decoder blocks.
        skips.reverse()

        # Initialize the list inputs for the deep supervision heads.
        if self.use_deep_supervision and self.training:
            deep_supervision_head_inputs = []
            final_deep_supervision_output = []

        for i, (skip, decoder_block) in enumerate(
            zip(skips, self.decoder_layers)
        ):
            x = decoder_block(x, skip)

            if (
                self.use_deep_supervision and
                self.training and
                i in self.deep_supervision_head_ids
            ):
                deep_supervision_head_inputs.append(x)  # pylint: disable=possibly-used-before-assignment

        # Reverse the deep supervision head inputs to match the order of the
        # deep supervision heads. Apply the deep supervision heads to their
        # respective inputs. Interpolate the outputs to match the size of the
        # input image.
        if self.use_deep_supervision and self.training:
            deep_supervision_head_inputs.reverse()
            for i, deep_supervision_head_input in enumerate(
                deep_supervision_head_inputs
            ):
                deep_supervision_head_output = (
                    self.deep_supervision_heads[i](deep_supervision_head_input)
                )
                final_deep_supervision_output.append(
                    F.interpolate(deep_supervision_head_output, x.shape[2:])
                )

        if self.training:
            return {
                "prediction": self.output_block(x),
                "deep_supervision": (
                    final_deep_supervision_output if self.use_deep_supervision
                    else None
                ),
            }

        return self.output_block(x)

    def get_input_block(self):
        """Return the input block for the UNet."""
        return self.conv_block(
            3,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_bottleneck(self):
        """Return the bottleneck layer for the UNet."""
        return self.conv_block(
            3,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_output_block(self, level: int) -> dynamic_unet_blocks.UnetOutBlock:
        """Return the output block for the UNet for a given index.

        Args:
            level: The level of the output block. This is used to determine the
                number of input channels for the output block.

        Returns:
            The output block for the UNet.
        """
        return dynamic_unet_blocks.UnetOutBlock(
            3,
            self.filters[level],
            self.out_channels,
            dropout=self.dropout
        )

    def get_encoder_layers(self):
        """Get list of encoder layers for the UNet."""
        in_filters = self.filters[:-2]
        out_filters = self.filters[1:-1]
        strides = self.strides[1:-1]
        kernel_size = self.kernel_size[1:-1]
        return self.get_module_list(
            in_filters,
            out_filters,
            kernel_size,
            strides,
            self.conv_block,  # type: ignore
        )

    def get_decoder_layers(self):
        """Get list of decoder layers for the UNet."""
        in_filters = self.filters[1:][::-1]
        out_filters = self.filters[:-1][::-1]
        strides = self.strides[1:][::-1]
        kernel_size = self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            in_filters,
            out_filters,
            kernel_size,
            strides,
            dynamic_unet_blocks.UnetUpBlock,  # type: ignore
            upsample_kernel_size,
            trans_bias=self.trans_bias,
        )

    def get_module_list(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        trans_bias: bool = False,
    ) -> nn.ModuleList:
        """Get a list of convolutional blocks for the UNet.

        Args:
            in_channels: Number of input channels for each block.
            out_channels: Number of output channels for each block.
            kernel_size: Kernel size for each block.
            strides: Stride for each block.
            conv_block: Convolutional block to use.
            upsample_kernel_size: Kernel size for transposed convolution.

        Returns:
            List of convolutional blocks.
        """
        list_of_layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                    in_channels,
                    out_channels,
                    kernel_size,
                    strides,
                    upsample_kernel_size
            ):
                params = {
                    "spatial_dims": 3,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                list_of_layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(
                in_channels,
                out_channels,
                kernel_size,
                strides
            ):
                params = {
                    "spatial_dims": 3,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                list_of_layers.append(layer)
        return nn.ModuleList(list_of_layers)

    def get_deep_supervision_heads(self):
        """Get the deep supervision heads for the UNet.

        Each head is a 1x1 output conv from the feature maps at a decoder
        level to out_channels. The heads are stored in order from shallowest
        to deepest (i.e. head 0 is closest to the output). They are applied
        in reverse during the forward pass (after
        deep_supervision_head_inputs.reverse()), so head i is paired with
        decoder output at deep_supervision_head_ids[-(i+1)].

        The filter count for head i is filters[i+1], which matches the
        decoder output channels at that depth because the decoder out_filters
        list is filters[:-1][::-1] — meaning the deepest selected decoder
        outputs correspond to filters[1], filters[2], etc.
        """
        return nn.ModuleList(
            [
                self.get_output_block(i + 1) for
                i in range(self.num_deep_supervision_heads)
            ]
        )

    @staticmethod
    def initialize_weights(module):
        """Initialize the weights of the UNet."""
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            module.weight = nn.init.kaiming_normal_(
                module.weight, a=constants.NEGATIVE_SLOPE
            )
            if module.bias is not None:
                module.bias = nn.init.constant_(
                    module.bias, constants.INITIAL_BIAS_VALUE
                )
