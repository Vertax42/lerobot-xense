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
- Gripper Motor: Provides gripper motor state

When mounted on a robot arm (e.g., Flexiv Rizon4), Vive Tracker can be disabled
since the robot arm provides pose information.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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
        enable_gripper: Whether to enable gripper motor
        enable_sensor: Whether to enable tactile sensors
        enable_camera: Whether to enable wrist camera
        sensor_keys: Mapping from sensor SN to feature key name
        vive_config_path: Vive Tracker config file path
        vive_lh_config: Vive Tracker lighthouse config
        vive_to_ee_pos: Position offset from Vive Tracker to end-effector [x, y, z] in meters
        vive_to_ee_quat: Rotation offset from Vive Tracker to end-effector [qw, qx, qy, qz]
        init_open: bool, whether to open the gripper on connect

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

    # Sensor settings
    rectify_size: tuple[int, int] = (400, 700)
    sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY

    # Component enable flags
    enable_gripper: bool = True
    enable_sensor: bool = True
    enable_camera: bool = True
    enable_vive: bool = True  # Set to False when mounted on robot arm (pose from arm)

    # Gripper normalization: raw_pos / gripper_max_pos -> [0, 1]
    # Set to the maximum readout value from your gripper
    gripper_max_pos: float = 85.0
    
    # Gripper control parameters for set_position()
    gripper_v_max: float = 80.0  # Maximum velocity mm/s
    gripper_f_max: float = 20.0  # Maximum force N
    
    # Initialize gripper to fully open on connect
    init_open: bool = True

    # Sensor SN to feature key mapping
    # If a sensor SN is not in this dict, it will use "sensor_{sn}" as key
    # Example: {"OG000344": "tactile_thumb", "OG000337": "tactile_finger"}
    sensor_keys: dict[str, str] = field(default_factory=dict)

    # Vive Tracker settings (only used when enable_vive=True)
    vive_config_path: Optional[str] = None
    vive_lh_config: Optional[str] = None

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
        if not self.sensor_keys:
            self.sensor_keys = {
                "OG000454": "right_tactile",
                "OG000447": "left_tactile",
            }
        
    def get_sensor_key(self, sensor_sn: str) -> str:
        """Get the feature key for a sensor SN.

        Args:
            sensor_sn: The sensor serial number

        Returns:
            The feature key name (from sensor_keys if defined, otherwise "sensor_{sn}")
        """
        return self.sensor_keys.get(sensor_sn, f"sensor_{sensor_sn}")
