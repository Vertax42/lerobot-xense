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

    This teleoperator provides 6-DoF delta control (translation + rotation)
    suitable for end-effector teleoperation.

    Attributes:
        pos_sensitivity: Sensitivity multiplier for position control (m/s at max deflection).
        ori_sensitivity: Sensitivity multiplier for orientation control (rad/s at max deflection).
        gripper_speed: Speed of gripper open/close (units/s).
        deadzone: Deadzone threshold [0-1] for each axis. Values below this are treated as zero.
        max_value: Maximum raw value from spacemouse (300 for wired, 500 for wireless).
        frequency: Polling frequency in Hz.
        invert_axes: Tuple of 6 bools to invert each axis (tx, ty, tz, rx, ry, rz).
        swap_gripper_buttons: If True, swap left/right button for gripper open/close.
    """

    pos_sensitivity: float = 0.8  # m/s at max deflection
    ori_sensitivity: float = 1.5  # rad/s at max deflection
    deadzone: float = 0.1  # [0-1] threshold
    max_value: int = 500  # 300 for wired, 500 for wireless
    frequency: int = 200  # Hz
    filter_window_size: int = 3  # Moving average filter window size
    use_gripper: bool = True  # Whether to include gripper in action space
    use_delta_rot: bool = False  # False: absolute rotation (rx, ry, rz); True: delta rotation
    invert_axes: Tuple[bool, bool, bool, bool, bool, bool] = (
        False,
        False,
        False,
        False,
        False,
        False,
    )
    swap_gripper_buttons: bool = False
