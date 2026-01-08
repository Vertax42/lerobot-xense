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

"""Configuration for Flexiv Rizon4 robot."""

from dataclasses import dataclass, field
from enum import Enum


class SensorOutputType(Enum):
    """Output type for tactile sensors."""

    RECTIFY = "rectify"
    DIFFERENCE = "difference"


@dataclass
class FlareGripperConfig:
    """Configuration for FlareGripper.

    The FlareGripper is UMI-Like gripper device.

    Attributes:
        mac_addr: Serial number of the robot (e.g., "e2b26adbb104")
        cameras: Dictionary of camera configurations
        sensor_keys: Mapping from sensor SN to feature key name

        gripper_v_max: Maximum velocity mm/s
        gripper_f_max: Maximum force N
        init_open: bool, whether to open the gripper on connect
        gripper_max_pos: float, the maximum position of the gripper
        gripper_v_max: Maximum velocity mm/s
        gripper_f_max: Maximum force N
        init_open: bool, whether to open the gripper on connect
    """

    # Gripper identification
    mac_addr: str = "e2b26adbb104"  # Gripper serial number
    # Camera settings
    cam_size: tuple[int, int] = (640, 480)
    # Sensor settings
    rectify_size: tuple[int, int] = (96, 160)
    sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    sensor_keys: dict[str, str] = field(default_factory=dict)

    # Gripper normalization: raw_pos / gripper_max_pos -> [0, 1]
    # Set to the maximum readout value from your gripper
    gripper_max_pos: float = 85.0

    # Gripper control parameters for set_position()
    gripper_v_max: float = 80.0  # Maximum velocity mm/s
    gripper_f_max: float = 20.0  # Maximum force N

    # Initialize gripper to fully open on connect
    init_open: bool = True

    def __post_init__(self):
        if not self.mac_addr:
            raise ValueError("mac_addr is required for XenseFlare")

        # Set default sensor_keys if not provided
        if not self.sensor_keys:
            self.sensor_keys = {
                "left_tactile": "OG000447",
                "right_tactile": "OG000454",
            }
