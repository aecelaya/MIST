# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataclass containing constants for MIST loss functions."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class LossConstants:
    """Dataclass containing constants for MIST loss functions."""
    # Small constant to prevent division by zero in loss computations.
    AVOID_DIVISION_BY_ZERO_CONSTANT: float = 1e-6

    # Spatial dimensions for 3D data (H, W, D).
    SPATIAL_DIMS_3D: tuple[int, ...] = (2, 3, 4)
