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
Configuration for Xense Flare Teleoperator.

Xense Flare can be used as a teleoperator to control robot arms (e.g., Flexiv Rizon4).
It provides:
- 6-DoF TCP pose from Vive Tracker
- Gripper position from encoder
"""

from dataclasses import dataclass, field
from typing import Optional

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xense_flare")
@dataclass
class XenseFlareTeleopConfig(TeleoperatorConfig):
    """Configuration for Xense Flare Teleoperator.

    This teleoperator uses Xense Flare gripper with:
    - Vive Tracker for 6-DoF pose tracking
    - Gripper encoder for gripper position

    Attributes:
        mac_addr: MAC address of the FlareGrip device (required)
        enable_gripper: Whether to enable gripper encoder readout
        gripper_max_pos: Maximum gripper position for normalization
        gripper_max_readout: Maximum readout after calibration for normalization
        vive_config_path: Vive Tracker config file path
        vive_lh_config: Vive Tracker lighthouse config
        vive_to_ee_pos: Position offset from Vive Tracker to end-effector [x, y, z] in meters
        vive_to_ee_quat: Rotation offset from Vive Tracker to end-effector [qw, qx, qy, qz]
        device_wait_timeout: Timeout for Vive Tracker device detection
        required_trackers: Number of trackers required
        filter_window_size: Moving average filter window size for smoothing
        position_jump_threshold: Max allowed position change per frame (meters)
        enable_position_jump_filter: Whether to enable position jump filtering
    """

    id: str = "xense_flare"

    # Device MAC address (required)
    mac_addr: str = ""

    # Gripper settings
    enable_gripper: bool = True
    gripper_max_pos: float = 85.0  # SDK max position
    gripper_max_readout: float = 83.5  # Max readout after calibration

    # Vive Tracker settings
    vive_config_path: Optional[str] = None
    vive_lh_config: Optional[str] = None
    device_wait_timeout: float = 10.0  # Timeout for device detection
    required_trackers: int = 1  # Number of trackers required

    # Vive Tracker to end-effector transformation
    vive_to_ee_pos: list = field(
        default_factory=lambda: [0.0, 0.0, 0.16]  # [x, y, z] in meters
    )
    vive_to_ee_quat: list = field(
        default_factory=lambda: [0.676, -0.207, -0.207, -0.676]  # [qw, qx, qy, qz]
    )

    # Filter settings
    filter_window_size: int = 3  # Moving average filter window size
    position_jump_threshold: float = 0.1  # Max position change per frame (meters)
    enable_position_jump_filter: bool = False  # Enable position jump filtering

    def __post_init__(self):
        if not self.mac_addr:
            raise ValueError("mac_addr is required for XenseFlareTeleop")

