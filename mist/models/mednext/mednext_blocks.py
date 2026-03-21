"""Implementation of MedNeXt blocks for the MedNeXt model."""
import torch
from torch import nn


def get_conv_layer(spatial_dim: int=3, transpose: bool=False):
    """Get the appropriate convolution layer based on spatial dimension."""
    if spatial_dim == 2:
        return nn.ConvTranspose2d if transpose else nn.Conv2d
    if spatial_dim == 3:
        return nn.ConvTranspose3d if transpose else nn.Conv3d
    raise ValueError(
        f"Invalid spatial dimension: {spatial_dim}. Must be 2 or 3."
    )


class MedNeXtBlock(nn.Module):
    """MedNeXtBlock class for the MedNeXt model.

    Attributes:
        do_res: Whether to use residual connection.
        dim: Dimension of the input. Can be "2d" or "3d".
        conv1: First convolution layer with DepthWise Convolutions.
        norm: Normalization layer.
        conv2: Second convolution (expansion) layer with Conv3D 1x1x1.
        act: Activation function (GeLU).
        conv3: Third convolution (compression) layer with Conv3D 1x1x1.
        global_resp_norm: Whether to use global response normalization.
        global_resp_beta: Learnable parameter for global response normalization.
        global_resp_gamma: Learnable parameter for global response
            normalization.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int=4,
        kernel_size: int=7,
        use_residual_connection: bool=True,
        norm_type: str="group",
        dim: str="3d",
        global_resp_norm: bool=False,
    ):
        """Initialize the MedNeXtBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expansion_ratio: Expansion ratio for the block.
            kernel_size: Kernel size for convolutions.
            use_residual_connection: Whether to use residual connection.
            norm_type: Type of normalization to use.
            dim: Dimension of the input. Can be "2d" or "3d".
            global_resp_norm: Whether to use global response normalization.
        """
        super().__init__()

        # Set convolution and GRN parameters.
        self.do_res = use_residual_connection
        self.dim = dim
        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3)
        global_resp_norm_param_shape = (1,) * (2 if dim == "2d" else 3)

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        # Normalization Layer.
        if norm_type == "group":
            self.norm = nn.GroupNorm(
                num_groups=in_channels, num_channels=in_channels
            )
        elif norm_type == "layer":
            # GroupNorm(1, C) is equivalent to LayerNorm over the channel
            # dimension for conv feature maps and works for any spatial size,
            # unlike nn.LayerNorm which requires a fixed spatial shape.
            self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        # Second convolution (expansion) layer with Conv3D 1x1x1.
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=expansion_ratio * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=expansion_ratio * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.global_resp_norm = global_resp_norm
        if self.global_resp_norm:
            global_resp_norm_param_shape = (
                (1, expansion_ratio * in_channels) +
                global_resp_norm_param_shape
            )
            self.global_resp_beta = nn.Parameter(
                torch.zeros(global_resp_norm_param_shape), requires_grad=True
            )
            self.global_resp_gamma = nn.Parameter(
                torch.zeros(global_resp_norm_param_shape), requires_grad=True
            )

    def forward(self, x):
        """Forward pass of the MedNeXtBlock.

        Args:
            x: Input tensor of shape (N, C, D, H, W) or (N, C, H, W) for 3D or
                2D data, respectively.

        Returns:
            x1: Output tensor after applying the block operations.
        """
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))

        if self.global_resp_norm:
            if self.dim == "2d":
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            else:
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.global_resp_gamma * (x1 * nx) + self.global_resp_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    """MedNeXtDownBlock class for downsampling in the MedNeXt model.

    Attributes:
        resample_do_res: Whether to use residual connection for downsampling.
        res_conv: Convolution layer for residual connection.
        conv1: Convolution layer for downsampling.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int=4,
        kernel_size: int=7,
        use_residual_connection: bool=False,
        norm_type: str="group",
        dim: str="3d",
        global_resp_norm: bool=False,
    ):
        """Initialize the MedNeXtDownBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expansion_ratio: Expansion ratio for the block.
            kernel_size: Kernel size for convolutions.
            use_residual_connection: Whether to use residual connection.
            norm_type: Type of normalization to use.
            dim: Dimension of the input. Can be "2d" or "3d".
            global_resp_norm: Whether to use global response normalization.
        """
        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            use_residual_connection=False,
            norm_type=norm_type,
            dim=dim,
            global_resp_norm=global_resp_norm,
        )
        # Define convolution layer based on spatial dimension.
        conv = get_conv_layer(spatial_dim=2 if dim == "2d" else 3)

        # Define residual connection if specified.
        self.resample_do_res = use_residual_connection
        if use_residual_connection:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        # Set the convolution layer for downsampling.
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x):
        """Forward pass of the MedNeXtDownBlock.

        Args:
            x: Input tensor of shape (N, C, D, H, W) or (N, C, H, W) for 3D or
                2D data, respectively.

        Returns:
            x1: Output tensor after applying the block operations. This is a 
                downsampled version of the input tensor. of shape
                (N, C, D/2, H/2, W/2) or (N, C, H/2, W/2) for 3D or 2D data,
        """
        x1 = super().forward(x)
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res
        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    """MedNeXtUpBlock class for upsampling in the MedNeXt model.

    Attributes:
        resample_do_res: Whether to use residual connection for upsampling.
        res_conv: Convolution layer for residual connection.
        conv1: Convolution layer for upsampling.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int=4,
        kernel_size: int=7,
        use_residual_connection: bool=False,
        norm_type: str="group",
        dim: str="3d",
        global_resp_norm: bool=False,
    ):
        """Initialize the MedNeXtUpBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expansion_ratio: Expansion ratio for the block.
            kernel_size: Kernel size for convolutions.
            use_residual_connection: Whether to use residual connection.
            norm_type: Type of normalization to use.
            dim: Dimension of the input. Can be "2d" or "3d".
            global_resp_norm: Whether to use global response normalization.
        """
        super().__init__(
            in_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            use_residual_connection=False,
            norm_type=norm_type,
            dim=dim,
            global_resp_norm=global_resp_norm,
        )

        # Define whether to use residual connection for upsampling.
        self.resample_do_res = use_residual_connection

        self.dim = dim
        conv = get_conv_layer(
            spatial_dim=2 if dim == "2d" else 3, transpose=True
        )
        if use_residual_connection:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                output_padding=1,
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
            groups=in_channels,
        )

    def forward(self, x):
        """Forward pass of the MedNeXtUpBlock.

        Args:
            x: Input tensor of shape (N, C, D, H, W) or (N, C, H, W) for 3D or
                2D data, respectively.

        Returns:
            x1: Output tensor after applying the block operations. This is an
                upsampled version of the input tensor of shape
                (N, C, D*2, H*2, W*2) or (N, C, H*2, W*2) for 3D or 2D data,
        """
        x1 = super().forward(x)

        if self.resample_do_res:
            x1 = x1 + self.res_conv(x)

        return x1


class MedNeXtOutBlock(nn.Module):
    """MedNeXtOutBlock class for the output block in the MedNeXt model.

    Attributes:
        conv_out: Convolution layer for the output block.
    """
    def __init__(self, in_channels: int, n_classes: int, dim: str="3d"):
        """Initialize the MedNeXtOutBlock.

        Args:
            in_channels: Number of input channels.
            n_classes: Number of output classes.
            dim: Dimension of the input. Can be "2d" or "3d".
        """
        super().__init__()

        # Define the convolution layer for the output block this will be a
        # pointwise convolution layer that maps the current number of
        # channels to the number of classes.
        conv = get_conv_layer(
            spatial_dim=2 if dim == "2d" else 3, transpose=False
        )
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass of the MedNeXtOutBlock.

        Args:
            x: Input tensor of shape (N, C, D, H, W) or (N, C, H, W) for 3D or
                2D data, respectively.

        Returns:
            Tensor of shape (N, n_classes, D, H, W) or (N, n_classes, H, W)
                for 3D or 2D data, respectively.
        """
        return self.conv_out(x)
