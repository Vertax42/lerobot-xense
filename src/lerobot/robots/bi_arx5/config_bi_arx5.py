#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.xense import XenseCameraConfig, XenseOutputType

# from lerobot.cameras.realsense import RealSenseCameraConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("bi_arx5")
@dataclass
class BiARX5Config(RobotConfig):
    left_arm_model: str = "X5"
    left_arm_port: str = "can1"
    right_arm_model: str = "X5"
    right_arm_port: str = "can3"
    log_level: str = "DEBUG"
    use_multithreading: bool = True
    rpc_timeout: float = 10.0
    controller_dt: float = 0.005  # 100Hz / 200Hz
    interpolation_controller_dt: float = 0.01
    inference_mode: bool = False
    enable_tactile_sensors: bool = False  # whether to enable tactile sensors
    # Preview time in seconds for action interpolation during inference
    # Higher values (0.03-0.05) provide smoother motion but more delay
    # Lower values (0.01-0.02) are more responsive but may cause jittering
    preview_time: float = 0.0  # Default 30ms for smooth inference
    gripper_open_readout: list[float] = field(default_factory=lambda: [-3.5, -3.4])
    home_position: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    start_position: list[float] = field(
        default_factory=lambda: [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    def __post_init__(self):

        # if enable tactile sensors, add Xense configuration
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
                    fps=60,  # Reduced from 60 to reduce loop overhead
                    output_types=[XenseOutputType.DIFFERENCE],
                    warmup_s=1.0,  # Increased warmup time for stable initialization
                    # width=700,
                    # height=400,
                ),
                "left_tactile_0": XenseCameraConfig(
                    serial_number="OG000337",
                    fps=60,  # Reduced from 60 to reduce loop overhead
                    output_types=[XenseOutputType.DIFFERENCE],
                    warmup_s=1.0,  # Increased warmup time for stable initialization
                    # width=700,
                    # height=400,
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
