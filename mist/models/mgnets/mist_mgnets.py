"""Implement adaptive Multi-Grid Network (MGNet) architectures."""

from collections import OrderedDict
from typing import List, Sequence, Dict, Optional, Union, Any

import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock

from mist.models.base_model import MISTModel
from mist.models.nnunet import nnunet_utils
from mist.models.nnunet.nnunet_constants import NNUnetConstants as constants
from mist.models.mgnets.mgnets_constants import MGNetConstants as mgnet_constants


class MGNet(MISTModel):
    """
    Adaptive Multi-Grid Network (FMG-Net / W-Net).

    This architecture implements a Generalized U-Net that supports complex
    topological  variations like W-Net or FMG-Net (Full Multigrid Network).
    Instead of a fixed encoder-decoder shape, it treats the network as a grid of
    nodes, where connections are determined by a set of logical wiring rules.

    Wiring Logic (Grid-Based State Machine):
    ----------------------------------------
    1. **Vertical Logic:** Every node receives input from the node directly
        below it in the previous column (standard U-Net upsampling).
    2. **Horizontal Logic (Nearest Encoder):** Every rising node (decoder)
        receives a skip connection from the *nearest* available encoder node at
        the same resolution level to its left.
    3. **Diagonal Logic (Neighbor Peak):** A decoder node receives an additional
       input from its immediate left neighbor *only if* that neighbor was a
       local maxima (a "Peak" of a spike).

    Attributes:
        num_classes: The number of output channels/classes.
        use_deep_supervision: Whether deep supervision is enabled.
        kernels: The kernel sizes for convolutions at each depth.
        strides: The strides for convolutions at each depth.
        num_layers: Total depth of the network (number of resolution levels).
        bottleneck_layer_idx: The index of the deepest layer (num_layers - 1).
        num_aux_heads: The number of deep supervision heads (if enabled).
        filters_per_layer: The number of feature channels at each depth.
        spike_height_schedule: The sequence defining the height of intermediate
            spikes (V-cycles) in the grid.
        main_encoder: The initial contracting path module list.
        spikes: The intermediate up/down V-cycles module list.
        main_decoder_blocks: The final expanding path blocks.
        deep_supervision_heads: The auxiliary output heads.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Sequence[int],
        target_spacing: Sequence[float],
        mg_net: str = "fmgnet",
        use_residual_blocks: bool = False,
        use_deep_supervision: bool = True,
        **kwargs: Any
    ):
        """
        Initializes the MGNet architecture by simulating the grid traversal to
        dynamically calculate channel dependencies.

        MGNet always uses the pocket paradigm: filters are kept constant at
        base_filters across all depths. Representational capacity comes from
        the multigrid topology (number and pattern of V-cycles), not from
        channel widening.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output classes.
            patch_size: The spatial size of the input patch (e.g., 128x128x128).
            target_spacing: The voxel spacing (e.g., [1.0, 1.0, 1.0]).
            mg_net: The topology type. Options: 'wnet', 'fmgnet', or 'unet'.
                Defaults to 'fmgnet'.
            use_residual_blocks: If True, uses ResBlocks; otherwise uses
                BasicBlocks. Defaults to False.
            use_deep_supervision: If True, enables 2 auxiliary loss heads at
                lower resolutions, matching nnUNet's default. Defaults to True.
            **kwargs: Additional keyword arguments (ignored).
        """
        super().__init__()

        if len(patch_size) != 3:
            raise ValueError(
                f"MGNet requires a 3D patch_size, but got {len(patch_size)} "
                "dimensions. For anisotropic data use a thin Z slice "
                "(e.g., 256×256×5) rather than a 2D patch."
            )

        self.num_classes = out_channels
        self.use_deep_supervision = use_deep_supervision

        # --- 1. ADAPTIVE TOPOLOGY CONFIGURATION ---
        self.kernels, self.strides, _ = nnunet_utils.get_unet_params(
            patch_size, target_spacing
        )
        self.num_layers = len(self.strides)
        self.bottleneck_layer_idx = self.num_layers - 1

        # --- 2. DEEP SUPERVISION CONFIGURATION ---
        self.num_aux_heads = 2 if use_deep_supervision else 0

        # --- 3. FILTER & BLOCK CONFIGURATION ---
        # MGNet always uses the pocket paradigm: constant filters at all depths.
        # Representational capacity comes from the multigrid topology, not width.
        base_filters = constants.INITIAL_FILTERS
        self.filters_per_layer = [
            base_filters for _ in range(self.num_layers)
        ]

        self.block_class = (
            UnetResBlock if use_residual_blocks else UnetBasicBlock
        )

        # --- 4. SPIKE SCHEDULE GENERATION ---
        max_spike_height = self.bottleneck_layer_idx

        if mg_net.lower() == "fmgnet":
            # Progressive Refinement: [1, 2, 3, ... Max],
            self.spike_height_schedule = list(range(1, max_spike_height + 1))
        elif mg_net.lower() == "wnet":
            # Sparse W-Pattern: [1, 2, 1, 3, 1, 4 ...].
            self.spike_height_schedule = self._generate_sparse_w_sequence(
                max_spike_height)
        else:
            # Standard U-Net behavior (Single spike at max height, effectively).
            self.spike_height_schedule = [max_spike_height]

        # ====================================================================
        # 5. BUILD NETWORK GRAPH (Simulation Strategy)
        # ====================================================================

        # --- A. MAIN ENCODER (The Left Wall) ---
        self.main_encoder = nn.ModuleList()
        current_input_channels = in_channels

        for depth_idx in range(self.num_layers):
            block = self._make_block(
                in_channels=current_input_channels,
                out_channels=self.filters_per_layer[depth_idx],
                kernel_size=self.kernels[depth_idx],
                stride=self.strides[depth_idx]
            )
            self.main_encoder.append(block)
            current_input_channels = self.filters_per_layer[depth_idx]

        # --- SIMULATION STATE REGISTRY ---
        simulation_peak_registry = {d: False for d in range(self.num_layers)}

        # --- B. INTERMEDIATE SPIKES ---
        self.spikes = nn.ModuleList()

        for spike_height in self.spike_height_schedule:
            spike_module = nn.ModuleDict()
            peak_depth_idx = self.bottleneck_layer_idx - spike_height

            # 1. UPWARD PATH
            up_blocks = nn.ModuleList()
            up_samples = nn.ModuleList()

            for depth_idx in range(
                self.bottleneck_layer_idx - 1, peak_depth_idx - 1, -1
            ):
                expected_in_channels = self.filters_per_layer[depth_idx + 1]
                expected_in_channels += self.filters_per_layer[depth_idx]

                if simulation_peak_registry[depth_idx]:
                    expected_in_channels += self.filters_per_layer[depth_idx]
                    simulation_peak_registry[depth_idx] = False

                up_blocks.append(self._make_block(
                    in_channels=expected_in_channels,
                    out_channels=self.filters_per_layer[depth_idx],
                    kernel_size=self.kernels[depth_idx + 1],
                    stride=[1, 1, 1],
                ))
                up_samples.append(self._make_upsample(
                    in_channels=self.filters_per_layer[depth_idx + 1],
                    scale_factor=self.strides[depth_idx + 1]
                ))

            spike_module["up_blocks"] = up_blocks
            spike_module["up_samples"] = up_samples

            simulation_peak_registry[peak_depth_idx] = True

            # 2. DOWNWARD PATH
            down_blocks = nn.ModuleList()
            previous_channels = self.filters_per_layer[peak_depth_idx]

            for depth_idx in range(
                peak_depth_idx + 1, self.bottleneck_layer_idx + 1
            ):
                down_blocks.append(self._make_block(
                    in_channels=previous_channels,
                    out_channels=self.filters_per_layer[depth_idx],
                    kernel_size=self.kernels[depth_idx],
                    stride=self.strides[depth_idx]
                ))
                previous_channels = self.filters_per_layer[depth_idx]
                simulation_peak_registry[depth_idx] = False

            spike_module["down_blocks"] = down_blocks
            self.spikes.append(spike_module)

        # --- C. MAIN DECODER (Final Ascent) ---
        self.main_decoder_blocks = nn.ModuleList()
        self.main_decoder_upsamples = nn.ModuleList()

        for depth_idx in range(self.bottleneck_layer_idx - 1, -1, -1):
            expected_in_channels = self.filters_per_layer[depth_idx + 1]
            expected_in_channels += self.filters_per_layer[depth_idx]
            if simulation_peak_registry[depth_idx]:
                expected_in_channels += self.filters_per_layer[depth_idx]
                simulation_peak_registry[depth_idx] = False

            self.main_decoder_blocks.append(self._make_block(
                in_channels=expected_in_channels,
                out_channels=self.filters_per_layer[depth_idx],
                kernel_size=self.kernels[depth_idx + 1],
                stride=[1, 1, 1],
            ))
            self.main_decoder_upsamples.append(self._make_upsample(
                in_channels=self.filters_per_layer[depth_idx + 1],
                scale_factor=self.strides[depth_idx + 1]
            ))

        # --- D. OUTPUT HEADS ---
        self.final_output_conv = nn.Conv3d(
            self.filters_per_layer[0], self.num_classes, kernel_size=1
        )

        self.deep_supervision_heads = nn.ModuleList()
        if self.use_deep_supervision:
            for head_idx in range(self.num_aux_heads):
                level_idx = head_idx + 1
                self.deep_supervision_heads.append(
                    nn.Conv3d(
                        self.filters_per_layer[level_idx],
                        self.num_classes,
                        kernel_size=1,
                    )
                )

        self.apply(self._init_weights)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder weights: main encoder path only.

        Spike V-cycles are excluded — they are topology-specific to the
        multigrid structure and do not transfer cleanly across architectures.
        """
        return OrderedDict(
            {k: v for k, v in self.state_dict().items()
             if k.startswith("main_encoder.")}
        )

    def _generate_sparse_w_sequence(self, max_height: int) -> List[int]:
        """
        Generates the recursive V-cycle pattern for W-Net topology.

        The sequence is constructed by creating a full pyramid up to max_height
        and interleaving it with height-1 spikes to maintain high-frequency
        gradients (e.g., [1, 2, 1, 3, 1...]).

        Args:
            max_height: The maximum height (depth from bottleneck) the W-Net
                should reach.

        Returns:
            A sequence of integers representing the height of each intermediate
            spike.
        """
        if max_height <= 1:
            return [1]

        core_sequence = (
            list(range(2, max_height + 1)) + list(range(max_height - 1, 1, -1))
        )
        full_sequence = [1]
        for val in core_sequence:
            full_sequence.extend([val, 1])
        return full_sequence

    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: Sequence[int]
    ) -> nn.Module:
        """
        Creates a computation block (Basic or Residual).

        Handles feature projection for memory efficiency if input channels
        exceed 512. Enforces Instance Normalization.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: 3D kernel size tuple.
            stride: 3D stride tuple.

        Returns:
            The constructed convolutional block (or Sequence containing
            projection).
        """
        if in_channels > mgnet_constants.REDUCTION_THRESHOLD:
            projection = nn.Conv3d(
                in_channels,
                mgnet_constants.REDUCTION_THRESHOLD,
                kernel_size=1,
                bias=False,
            )
            block = self.block_class(
                spatial_dims=3,
                in_channels=mgnet_constants.REDUCTION_THRESHOLD,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=constants.NORMALIZATION,
                act_name=constants.ACTIVATION,
            )
            return nn.Sequential(projection, block)

        return self.block_class(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=constants.NORMALIZATION,
            act_name=constants.ACTIVATION,
        )

    def _make_upsample(
        self, in_channels: int, scale_factor: Sequence[int]
    ) -> nn.Module:
        """
        Creates a learnable transposed convolution upsampling layer.

        Uses kernel_size == stride, which is the standard approach for
        artifact-free upsampling and handles anisotropic strides correctly.

        Args:
            in_channels: Number of input channels.
            scale_factor: The stride (and kernel size) for each spatial
                dimension, derived from the encoder stride at that depth.

        Returns:
            The transposed convolution upsampling layer.
        """
        return nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=tuple(scale_factor),
            stride=tuple(scale_factor),
        )

    def _init_weights(self, module: nn.Module):
        """
        Applies Kaiming He initialization to convolutional weights and handles
        InstanceNorm initialization.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=constants.NEGATIVE_SLOPE)
            if module.bias is not None:
                nn.init.constant_(module.bias, constants.INITIAL_BIAS_VALUE)

        elif isinstance(module, (nn.InstanceNorm3d, nn.BatchNorm3d)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, Any]]:
        """Executes the grid traversal.

        Flow:
        1. Main Encoder (Left Wall) -> Populates initial feature registry.
        2. Spikes (Intermediate V-Cycles) -> Consumes and updates registry.
        3. Main Decoder (Final Ascent) -> Consumes features to produce output.

        Args:
            x: Input tensor of shape (Batch, Channels, H, W, D).

        Returns:
            If inference: Returns the final segmentation map (Tensor).
            If training (with deep supervision): Returns a dict containing:
                - 'prediction': Final output tensor.
                - 'deep_supervision': List of aux outputs from coarse to fine.
        """
        # 1. ENCODER FEATURE REGISTRY
        # Stores the history of features at every depth.
        encoder_feature_registry: Dict[int, List[torch.Tensor]] = {
            d: [] for d in range(self.num_layers)
        }

        # 2. NEIGHBOR PEAK REGISTRY
        # Transient buffer. Stores a feature ONLY if the node immediately to the
        # left was a local maxima (Peak).
        neighbor_peak_registry: Dict[int, Optional[torch.Tensor]] = {
            d: None for d in range(self.num_layers)
        }

        # --- PHASE 1: MAIN ENCODER ---
        current_features = x
        for depth_idx, block in enumerate(self.main_encoder):
            current_features = block(current_features)
            encoder_feature_registry[depth_idx].append(current_features)
            neighbor_peak_registry[depth_idx] = None

        # --- PHASE 2: INTERMEDIATE SPIKES ---
        for spike_module in self.spikes:

            # A. UPWARD PATH.
            for i, (block, upsample) in enumerate(
                zip(spike_module["up_blocks"], spike_module["up_samples"])
            ):
                target_depth_idx = self.bottleneck_layer_idx - 1 - i

                vertical_features = upsample(current_features)
                nearest_encoder_features = (
                    encoder_feature_registry[target_depth_idx][-1]
                )

                inputs_to_concat = [
                    vertical_features, nearest_encoder_features]

                if neighbor_peak_registry[target_depth_idx] is not None:
                    inputs_to_concat.append(
                        neighbor_peak_registry[target_depth_idx])
                    neighbor_peak_registry[target_depth_idx] = None

                current_features = block(torch.cat(inputs_to_concat, dim=1))

            # Peak reached.
            peak_depth_idx = (
                self.bottleneck_layer_idx - len(spike_module["up_blocks"])
            )
            neighbor_peak_registry[peak_depth_idx] = current_features
            encoder_feature_registry[peak_depth_idx].append(current_features)

            # B. DOWNWARD PATH.
            for i, block in enumerate(spike_module["down_blocks"]):
                target_depth_idx = peak_depth_idx + 1 + i
                current_features = block(current_features)
                encoder_feature_registry[target_depth_idx].append(
                    current_features)
                neighbor_peak_registry[target_depth_idx] = None

        # --- PHASE 3: MAIN DECODER ---
        decoder_features_for_deep_supervision = []

        for i, (block, upsample) in enumerate(
            zip(self.main_decoder_blocks, self.main_decoder_upsamples)
        ):
            target_depth_idx = self.bottleneck_layer_idx - 1 - i

            vertical_features = upsample(current_features)
            nearest_encoder_features = (
                encoder_feature_registry[target_depth_idx][-1]
            )
            inputs_to_concat = [vertical_features, nearest_encoder_features]

            if neighbor_peak_registry[target_depth_idx] is not None:
                inputs_to_concat.append(
                    neighbor_peak_registry[target_depth_idx])
                neighbor_peak_registry[target_depth_idx] = None

            current_features = block(torch.cat(inputs_to_concat, dim=1))

            if (
                self.use_deep_supervision
                and 0 < target_depth_idx <= self.num_aux_heads
            ):
                decoder_features_for_deep_supervision.append(current_features)

        # --- OUTPUT ---
        prediction = self.final_output_conv(current_features)

        if self.training:
            deep_supervision_outputs = None
            if self.use_deep_supervision:
                decoder_features_for_deep_supervision.reverse()
                deep_supervision_outputs = []
                for head_idx, features in enumerate(
                    decoder_features_for_deep_supervision
                ):
                    aux_output = self.deep_supervision_heads[head_idx](features)
                    deep_supervision_outputs.append(
                        F.interpolate(aux_output, x.shape[2:])
                    )
            return {
                "prediction": prediction,
                "deep_supervision": deep_supervision_outputs,
            }

        return prediction
