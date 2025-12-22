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

    This teleoperator provides 6-DoF absolute pose control using VR controllers,
    suitable for end-effector teleoperation. The output is an accumulated target_pose_6d
    that can be directly sent to a Cartesian controller.

    Attributes:
        use_left_controller: Whether to use the left controller for teleoperation.
        use_right_controller: Whether to use the right controller for teleoperation.
        pos_sensitivity: Sensitivity multiplier for position control (scales delta position).
        ori_sensitivity: Sensitivity multiplier for orientation control (scales delta orientation).
        filter_window_size: Moving average filter window size for smoothing pose changes.
        gripper_width: Maximum gripper position in meters (for clamping).
        grip_threshold: Threshold value (0-1) for grip to be considered pressed (enable control).
        orientation_offset_warning_deg: Warning threshold in degrees for orientation offset at enable.
                                        If controller-robot orientation difference exceeds this,
                                        a warning is logged and orientation control is disabled.
    """

    use_left_controller: bool = False
    use_right_controller: bool = True
    pos_sensitivity: float = 1.0  # Scale factor for position delta
    ori_sensitivity: float = 1.0  # Scale factor for orientation delta
    filter_window_size: int = 5  # Moving average filter window size
    gripper_width: float = 0.1  # Maximum gripper position in meters
    grip_threshold: float = 0.5  # Threshold for grip to enable control
    orientation_offset_warning_deg: float = 10.0  # Warning threshold for orientation offset (degrees)
