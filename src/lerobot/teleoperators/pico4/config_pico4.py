#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field  # noqa: F401

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pico4")
@dataclass
class Pico4Config(TeleoperatorConfig):
    """Configuration for Pico4 VR teleoperator.

    Attributes:
        trigger_threshold: Threshold value (0-1) for trigger to be considered pressed.
        grip_threshold: Threshold value (0-1) for grip to be considered pressed.
        use_left_controller: Whether to use the left controller for teleoperation.
        use_right_controller: Whether to use the right controller for teleoperation.
    """

    trigger_threshold: float = 0.5
    grip_threshold: float = 0.5
    use_left_controller: bool = True
    use_right_controller: bool = True
