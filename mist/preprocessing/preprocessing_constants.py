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
"""Constants for MIST data preprocessing."""
import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class PreprocessingConstants:
    """Constants for MIST data preprocessing."""
    # Normalization constants.
    WINDOW_PERCENTILE_LOW = 0.5
    WINDOW_PERCENTILE_HIGH = 99.5

    # RAI orientation constants.
    RAI_ANTS_DIRECTION = np.eye(3)
