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

"""
Configuration for Xense Flare - Multi-Modal Data Collection Gripper.

Xense Flare is a data collection gripper with multiple sensor modalities:
- Vive Tracker: Provides 6DoF trajectory data (optional, for standalone use)
- Wrist Camera: Provides visual information
- Tactile Sensors: Provides tactile perception
- Gripper Encoder: Provides gripper position readout (passive, no motor control)

When mounted on a robot arm (e.g., Flexiv Rizon4), Vive Tracker can be disabled
since the robot arm provides pose information.
"""

from dataclasses import dataclass, field
from enum import Enum

from ..config import RobotConfig


class SensorOutputType(Enum):
    """Output type for tactile sensors."""

    RECTIFY = "rectify"
    DIFFERENCE = "difference"


@RobotConfig.register_subclass("xense_flare")
@dataclass
class XenseFlareConfig(RobotConfig):
    """Configuration for Xense Flare Gripper.

    Attributes:
        mac_addr: MAC address of the FlareGrip device (required)
        cam_size: Camera frame size (width, height)
        rectify_size: Sensor rectify output size (width, height)
        enable_gripper: Whether to enable gripper encoder readout
        enable_sensor: Whether to enable tactile sensors
        enable_camera: Whether to enable wrist camera
        gripper_max_pos: Maximum gripper position for normalization
        sensor_keys: Mapping from sensor SN to feature key name
        vive_config_path: Vive Tracker config file path
        vive_lh_config: Vive Tracker lighthouse config
        vive_to_ee_pos: Position offset from Vive Tracker to end-effector [x, y, z] in meters
        vive_to_ee_quat: Rotation offset from Vive Tracker to end-effector [qw, qx, qy, qz]

    Example:
        config = XenseFlareConfig(
            mac_addr="6ebbc5f53240",
            sensor_keys={
                "OG000344": "tactile_left",
                "OG000337": "tactile_right",
            },
        )
        # This will create observation features:
        # - "tactile_left": (H, W, 3)
        # - "tactile_right": (H, W, 3)
    """

    # Device MAC address (required)
    mac_addr: str = ""

    # Camera settings
    cam_size: tuple[int, int] = (640, 480)

    # Sensor settings (actual tactile image resolution: 96x160x3, format: width, height)
    rectify_size: tuple[int, int] = (200, 350)
    sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY

    # Component enable flags
    enable_gripper: bool = True
    enable_sensor: bool = True
    enable_camera: bool = True
    enable_vive: bool = True  # Set to False when mounted on robot arm (pose from arm)

    # Gripper SDK max position (used for motor control, not for normalization)
    gripper_max_pos: float = 85.0

    # HACK: Need to set the maximum readout after calibration, so we can normalize the gripper position
    gripper_max_readout: float = 82.0
    # Sensor SN to feature key mapping
    # If a sensor SN is not in this dict, it will use "sensor_{sn}" as key
    # Example: {"OG000344": "tactile_thumb", "OG000337": "tactile_finger"}
    sensor_keys: dict[str, str] = field(default_factory=dict)

    # Vive Tracker settings (only used when enable_vive=True)
    vive_config_path: str | None = None
    vive_lh_config: str | None = None

    # Vive Tracker to end-effector transformation
    vive_to_ee_pos: list = field(
        default_factory=lambda: [0.0, 0.0, 0.16]  # [x, y, z] in meters
    )

    vive_to_ee_quat: list = field(
        default_factory=lambda: [0.676, -0.207, -0.207, -0.676]  # [qw, qx, qy, qz]
    )

    # Initial TCP pose for Vive pose tracking [x, y, z, qw, qx, qy, qz]
    # Default values from Flexiv Rizon4 home position
    # This determines the coordinate frame origin for the output pose
    init_tcp_pose: list = field(
        default_factory=lambda: [0.693307, -0.114902, 0.14589, 0.004567, 0.003238, 0.999984, 0.001246]
    )

    def __post_init__(self):
        if not self.mac_addr:
            raise ValueError("mac_addr is required for XenseFlare")

        # Set default sensor_keys if not provided
        # NOTE: Update these SNs to match your device's sensors
        if not self.sensor_keys:
            self.sensor_keys = {
                "OG000447": "right_tactile",
                "OG000454": "left_tactile",
            }

    def get_sensor_key(self, sensor_sn: str) -> str:
        """Get the feature key for a sensor SN.

        Args:
            sensor_sn: The sensor serial number

        Returns:
            The feature key name (from sensor_keys if defined, otherwise "sensor_{sn}")
        """
        return self.sensor_keys.get(sensor_sn, f"sensor_{sensor_sn}")
