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
from typing import Tuple

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseConfig(TeleoperatorConfig):
    """Configuration for 3D Spacemouse teleoperator.

    This teleoperator provides 6-DoF absolute pose control (translation + rotation)
    suitable for end-effector teleoperation. The output is an accumulated target_pose_6d
    that can be directly sent to a Cartesian controller.

    Attributes:
        pos_sensitivity: Sensitivity multiplier for position control (m/s at max deflection).
        ori_sensitivity: Sensitivity multiplier for orientation control (rad/s at max deflection).
        gripper_speed: Speed of gripper open/close (rad/s).
        deadzone: Deadzone threshold [0-1] for each axis. Values below this are treated as zero.
        max_value: Maximum raw value from spacemouse (300 for wired, 500 for wireless).
        frequency: Polling frequency in Hz for spacemouse backend.
        filter_window_size: Moving average filter window size for smoothing.
        control_dt: Control loop period in seconds. Should match external loop (e.g., 1/fps).
            This ensures consistent velocity scaling regardless of actual call timing.
        invert_axes: Tuple of 6 bools to invert each axis (tx, ty, tz, rx, ry, rz).
        swap_gripper_buttons: If True, swap left/right button for gripper open/close.
        gripper_width: Maximum gripper position in ratio of gripper_max_pos (for clamping).
    """

    pos_sensitivity: float = 0.8  # default 0.8 m/s at max deflection
    ori_sensitivity: float = 1.5  # default 1.5 rad/s at max deflection
    gripper_speed: float = 0.6  # ratio of gripper_max_pos / s for gripper open/close
    deadzone: float = 0.1  # [0-1] threshold
    max_value: int = 500  # 300 for wired, 500 for wireless
    frequency: int = 200  # Hz for spacemouse states polling
    filter_window_size: int = 3  # Moving average filter window size
    control_dt: float = 0.01  # Control loop period in seconds (should match external loop)
    invert_axes: Tuple[bool, bool, bool, bool, bool, bool] = (
        True,  # x-reverse
        True,  # y-reverse
        False,
        True,  # roll-reverse
        True,  # pitch-reverse
        False,
    )
    swap_gripper_buttons: bool = False  # default left button to close, right button to open
    gripper_width: float = 1.0  # Maximum gripper position (ratio of gripper_max_pos)
