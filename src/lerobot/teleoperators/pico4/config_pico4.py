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
        gripper_speed: Speed of gripper open/close (m/s).
        deadzone: Deadzone threshold [0-1] for position/orientation changes. Values below this are treated as zero.
        filter_window_size: Moving average filter window size for smoothing pose changes.
        control_dt: Control loop period in seconds. Should match external loop (e.g., 1/fps).
        gripper_width: Maximum gripper position in meters (for clamping).
        trigger_threshold: Threshold value (0-1) for trigger to be considered pressed.
        grip_threshold: Threshold value (0-1) for grip to be considered pressed.
    """

    use_left_controller: bool = True
    use_right_controller: bool = True
    pos_sensitivity: float = 1.0  # Scale factor for position delta
    ori_sensitivity: float = 1.0  # Scale factor for orientation delta
    gripper_speed: float = 0.01  # m/s for gripper open/close
    deadzone: float = 0.001  # [0-1] threshold for position/orientation changes
    filter_window_size: int = 5  # Moving average filter window size
    control_dt: float = 0.01  # Control loop period in seconds
    gripper_width: float = 0.1  # Maximum gripper position in meters
    trigger_threshold: float = 0.5
    grip_threshold: float = 0.5
