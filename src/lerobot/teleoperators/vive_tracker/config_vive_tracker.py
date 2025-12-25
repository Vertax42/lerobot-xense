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

from dataclasses import dataclass
from typing import Optional

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("vive_tracker")
@dataclass
class ViveTrackerConfig(TeleoperatorConfig):
    """Configuration for Vive Tracker teleoperator.

    This teleoperator provides 6-DoF absolute pose tracking using HTC Vive Tracker,
    suitable for end-effector teleoperation. Uses pysurvive library for tracking.

    Unlike relative motion teleoperators (e.g., Pico4 VR controller), this provides
    direct 1:1 absolute pose mapping - the tracker pose IS the target end-effector pose.

    Attributes:
        tracker_name: Name of the tracker device to use (e.g., "T20", "WM0").
                      If None, uses the first detected tracker.
        config_path: Path to pysurvive configuration file.
        lh_config: Lighthouse configuration string.
        device_wait_timeout: Timeout in seconds for waiting for devices.
        required_trackers: Number of trackers required before starting.
        filter_window_size: Moving average filter window size for smoothing.
        position_jump_threshold: Max allowed position change per frame (meters).
        enable_position_jump_filter: Whether to enable position jump filtering.
    """

    id: str = "vive_tracker"

    # Device settings
    tracker_name: Optional[str] = None  # Use first detected tracker if None
    config_path: Optional[str] = None  # pysurvive config file path
    lh_config: Optional[str] = None  # Lighthouse configuration
    device_wait_timeout: float = 10.0  # Timeout for device detection
    required_trackers: int = 1  # Number of trackers required

    # Filter settings
    filter_window_size: int = 1  # Moving average filter window size
    position_jump_threshold: float = 0.05  # Max position change per frame (meters)
    enable_position_jump_filter: bool = False  # Enable position jump filtering
