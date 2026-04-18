"""Implementation of MedNeXt blocks for the MedNeXt model."""
import torch
from torch import nn


class MedNeXtBlock(nn.Module):
    """MedNeXtBlock class for the MedNeXt model.

    Attributes:
        do_res: Whether to use residual connection.
        conv1: First convolution layer with DepthWise Convolutions.
        norm: Normalization layer.
        conv2: Second convolution (expansion) layer with Conv3D 1x1x1.
        act: Activation function (GeLU).
        conv3: Third convolution (compression) layer with Conv3D 1x1x1.
        global_resp_norm: Whether to use global response normalization.
        global_resp_beta: Learnable parameter for global response normalization.
        global_resp_gamma: Learnable parameter for global response normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: bool = True,
        norm_type: str = "group",
        global_resp_norm: bool = False,
    ):
        """Initialize the MedNeXtBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expansion_ratio: Expansion ratio for the block.
            kernel_size: Kernel size for convolutions.
            use_residual_connection: Whether to use residual connection.
            norm_type: Type of normalization to use.
            global_resp_norm: Whether to use global response normalization.
        """
        super().__init__()

        self.do_res = use_residual_connection

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        if norm_type == "group":
            self.norm = nn.GroupNorm(
                num_groups=in_channels, num_channels=in_channels
            )
        elif norm_type == "layer":
            self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        else:
            raise ValueError(
                f"norm_type must be 'group' or 'layer', got '{norm_type}'."
            )

        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=expansion_ratio * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.act = nn.GELU()

        self.conv3 = nn.Conv3d(
            in_channels=expansion_ratio * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.global_resp_norm = global_resp_norm
        if self.global_resp_norm:
            grn_shape = (1, expansion_ratio * in_channels, 1, 1, 1)
            self.global_resp_beta = nn.Parameter(
                torch.zeros(grn_shape), requires_grad=True
            )
            self.global_resp_gamma = nn.Parameter(
                torch.zeros(grn_shape), requires_grad=True
            )

    def forward(self, x):
        """Forward pass of the MedNeXtBlock.

        Args:
            x: Input tensor of shape (N, C, D, H, W).

        Returns:
            x1: Output tensor after applying the block operations.
        """
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        if self.global_resp_norm:
            gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.global_resp_gamma * (x1 * nx) + self.global_resp_beta + x1

        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    """MedNeXtDownBlock class for downsampling in the MedNeXt model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: bool = False,
        norm_type: str = "group",
        global_resp_norm: bool = False,
    ):
        """Initialize the MedNeXtDownBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expansion_ratio: Expansion ratio for the block.
            kernel_size: Kernel size for convolutions.
            use_residual_connection: Whether to use residual connection.
            norm_type: Type of normalization to use.
            global_resp_norm: Whether to use global response normalization.
        """
        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            use_residual_connection=False,
            norm_type=norm_type,
            global_resp_norm=global_resp_norm,
        )

        self.resample_do_res = use_residual_connection
        if use_residual_connection:
            self.res_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
            )

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):
        """Forward pass of the MedNeXtDownBlock."""
        x1 = super().forward(x)
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res
        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    """MedNeXtUpBlock class for upsampling in the MedNeXt model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: int = 7,
        use_residual_connection: bool = False,
        norm_type: str = "group",
        global_resp_norm: bool = False,
    ):
        """Initialize the MedNeXtUpBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expansion_ratio: Expansion ratio for the block.
            kernel_size: Kernel size for convolutions.
            use_residual_connection: Whether to use residual connection.
            norm_type: Type of normalization to use.
            global_resp_norm: Whether to use global response normalization.
        """
        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            use_residual_connection=False,
            norm_type=norm_type,
            global_resp_norm=global_resp_norm,
        )

        self.resample_do_res = use_residual_connection
        if use_residual_connection:
            self.res_conv = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                output_padding=1,
            )

        self.conv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
            groups=in_channels,
        )

    def forward(self, x):
        """Forward pass of the MedNeXtUpBlock."""
        x1 = super().forward(x)
        if self.resample_do_res:
            x1 = x1 + self.res_conv(x)
        return x1


class MedNeXtOutBlock(nn.Module):
    """MedNeXtOutBlock class for the output block in the MedNeXt model."""

    def __init__(self, in_channels: int, n_classes: int):
        """Initialize the MedNeXtOutBlock.

        Args:
            in_channels: Number of input channels.
            n_classes: Number of output classes.
        """
        super().__init__()
        self.conv_out = nn.Conv3d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass of the MedNeXtOutBlock."""
        return self.conv_out(x)
