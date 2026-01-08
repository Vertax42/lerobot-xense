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
from lerobot.cameras.xense import XenseCameraConfig, XenseOutputType
from lerobot.robots.config import RobotConfig


class BiARX5ControlMode(Enum):
    """Control modes for BiARX5 robot arms.

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
    TEACH_MODE = "teach_mode"  # Teaching mode with gravity compensation


@RobotConfig.register_subclass("bi_arx5")
@dataclass
class BiARX5Config(RobotConfig):
    """Configuration for BiARX5 dual-arm robot."""

    # Arm configuration
    left_arm_model: str = "X5"
    left_arm_port: str = "can1"
    right_arm_model: str = "X5"
    right_arm_port: str = "can3"

    # Logging and threading
    log_level: str = "DEBUG"
    use_multithreading: bool = True  # For SDK background_send_recv

    # Control parameters
    controller_dt: float = 0.005  # 200Hz low-level control frequency
    interpolation_controller_dt: float = 0.02  # 50Hz high-level interpolation control frequency

    # Control mode (default: joint control for teleoperation)
    control_mode: BiARX5ControlMode = BiARX5ControlMode.TEACH_MODE

    # Inference mode
    inference_mode: bool = False

    # Preview time in seconds for control interpolation
    # Higher values (0.03-0.05) provide smoother motion but more delay
    # Lower values (0.01-0.02) are more responsive but may cause jittering
    # For Cartesian mode: use default preview time 0.1s in low-level SDK
    preview_time: float = 0.03  # Default 30ms for Joint control

    # Gripper calibration (calibrated values from calibrate.py for left and right arms)
    gripper_open_readout: list[float] = field(default_factory=lambda: [-3.4, -3.4])
    enable_tactile_sensors: bool = False

    # Position settings (Joint space: 6 joints + gripper)
    home_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    start_position: list[float] = field(default_factory=lambda: [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0])
    # Camera configuration
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    def __post_init__(self):
        # Camera configuration based on tactile sensors setting
        if self.enable_tactile_sensors:
            self.cameras = {
                "head": RealSenseCameraConfig(
                    serial_number_or_name="230322271365", fps=60, width=640, height=480
                ),
                "left_wrist": RealSenseCameraConfig(
                    serial_number_or_name="230422271416", fps=60, width=640, height=480
                ),
                "right_wrist": RealSenseCameraConfig(
                    serial_number_or_name="230322274234", fps=60, width=640, height=480
                ),
                "right_tactile_0": XenseCameraConfig(
                    serial_number="OG000344",
                    fps=60,
                    output_types=[XenseOutputType.DIFFERENCE],
                    warmup_s=1.0,
                ),
                "left_tactile_0": XenseCameraConfig(
                    serial_number="OG000337",
                    fps=60,
                    output_types=[XenseOutputType.DIFFERENCE],
                    warmup_s=1.0,
                ),
            }
        else:
            self.cameras = {
                "head": RealSenseCameraConfig(
                    serial_number_or_name="230322271365", fps=60, width=640, height=480
                ),
                "left_wrist": RealSenseCameraConfig(
                    serial_number_or_name="230422271416", fps=60, width=640, height=480
                ),
                "right_wrist": RealSenseCameraConfig(
                    serial_number_or_name="230322274234", fps=60, width=640, height=480
                ),
            }
        pass
