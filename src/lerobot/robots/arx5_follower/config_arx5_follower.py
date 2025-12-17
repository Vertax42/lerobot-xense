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

from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


class ARX5ControlMode(Enum):
    """Control modes for ARX5 robot arm.

    Attributes:
        JOINT_CONTROL: Joint space position control mode.
            Robot tracks target joint positions directly.
        CARTESIAN_CONTROL: Cartesian/EEF space control mode.
            Robot tracks end-effector pose in 6D space (x, y, z, roll, pitch, yaw).
        TEACH_MODE: Teaching mode with gravity compensation.
            Robot maintains zero torque while compensating for gravity,
            allowing free movement by hand for demonstration recording.
    """

    JOINT_CONTROL = "joint_control"
    CARTESIAN_CONTROL = "cartesian_control"
    TEACH_MODE = "teach_mode"  # Teaching mode with gravity compensation, control mode is joint control


@RobotConfig.register_subclass("arx5_follower")
@dataclass
class ARX5FollowerConfig(RobotConfig):
    """Configuration for single ARX5 arm follower robot."""

    # Arm configuration
    arm_model: str = "X5"
    arm_port: str = "can3"

    # Logging and threading
    log_level: str = "DEBUG"
    use_multithreading: bool = True

    # Control parameters
    controller_dt: float = 0.005  # 200Hz low-level control frequency
    interpolation_controller_dt: float = 0.02  # 50Hz high-level interpolation control frequency

    # default control mode is teach mode
    control_mode: ARX5ControlMode = ARX5ControlMode.CARTESIAN_CONTROL
    # default inference mode is false
    inference_mode: bool = False

    # Preview time in seconds for control interpolation
    # Higher values (0.03-0.05) provide smoother motion but more delay
    # Lower values (0.01-0.02) are more responsive but may cause jittering

    # For Cartesian mode: use default preview time 0.1s in low-level SDK
    preview_time: float = 0.03  # Default 30ms for Joint control

    # Gripper calibration (calibrated value from calibrate.py)
    gripper_open_readout: float = -3.4
    enable_tactile_sensors: bool = False

    # Position settings (Joint space: 6 joints + gripper)
    home_position: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # default start position is [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]
    # modified cartesian start position is [0.0, 0.967, 1.290, -0.970, 0.0, 0.0, 0.0]
    if control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
        start_position = [0.0, 0.967, 1.290, -0.970, 0.0, 0.0, 0.0]
    else:
        start_position = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]

    # Camera configuration
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    def __post_init__(self):
        # Default camera configuration if not provided
        # if not self.cameras:
        #     self.cameras = {
        #         "head": RealSenseCameraConfig(
        #             serial_number_or_name="230322271365",
        #             fps=60,
        #             width=640,
        #             height=480,
        #         ),
        #         "wrist": RealSenseCameraConfig(
        #             serial_number_or_name="230422271416",
        #             fps=60,
        #             width=640,
        #             height=480,
        #         ),
        #     }
        pass
