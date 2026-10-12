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

"""Configuration for the right-arm-only Dobot Nova5 with an integrated DH gripper."""

from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.xense.configuration_xense import XenseOutputType, XenseTactileCameraConfig
from lerobot.robots.config import RobotConfig

from .config_dh_gripper_integrated import DHGripperIntegratedConfig


class ControlMode(str, Enum):
    """Control mode for the single Dobot Nova5 arm.

    JOINT_MOTION uses six joint positions plus one gripper position (7D).
    CARTESIAN_MOTION uses a 9D TCP pose plus one gripper position (10D).
    """

    JOINT_MOTION = "joint_motion_control"
    CARTESIAN_MOTION = "cartesian_motion_control"


class ResetTarget(str, Enum):
    HOME = "home"
    START = "start"


@RobotConfig.register_subclass("dobot_nova5_dh")
@dataclass
class DobotNova5DHConfig(RobotConfig):
    """Right-arm configuration extracted from ``BiDobotNova5DHConfig``.

    The field names and action/observation keys are intentionally unprefixed because this
    device exposes one arm. Defaults correspond to the right arm in the bimanual setup.
    """

    # Robot identification (right arm from bi_dobot_nova5_dh).
    robot_ip: str = "192.168.111.102"
    dashboardPort: int = 29999
    feedPortFour: int = 30004

    control_mode: ControlMode = ControlMode.CARTESIAN_MOTION
    control_frequency: float = 100.0

    go_to_start: bool = False
    reset_target: ResetTarget = ResetTarget.HOME
    go_to_home_on_disconnect: bool = True

    aheadtime: float = 50.0
    gain: float = 500.0

    # Cartesian ServoP safety and IK fallback.
    enable_cartesian_ik_guard: bool = True
    cartesian_ik_servoj: bool = True
    cartesian_ik_backoff_steps: int = 6
    cartesian_ik_servoj_hold_s: float = 0.0
    max_cartesian_step_m: float = 0.15

    # Keep record/teleop loops at camera FPS while Dobot commands run in a worker.
    async_action_worker: bool = True
    async_action_worker_frequency: float = 30.0

    # Tool 0 is the flange. Tool 1 is the configured right-arm end effector.
    use_tool_coordinate: bool = True
    tool_coordinate_index: int = 1

    # Right-arm Cartesian safety workspace, in metres.
    enable_clip: bool = True
    workspace_min_xyz_m: list[float] = field(default_factory=lambda: [-0.60, -0.25, -0.167])
    workspace_max_xyz_m: list[float] = field(default_factory=lambda: [0.75, 0.70, 0.69])

    home_point_list: list[float] = field(
        # default_factory=lambda: [220, 0, 135, -80, -88, 0]
        default_factory=lambda: [190, -12, 110, -25, -96, 10]
    )
    start_position_degree: list[float] = field(
        # default_factory=lambda: [220, 0, 135, -80, -88, 0]
        # default_factory=lambda: [220, 0, 135, -80, -88, 0]
        default_factory=lambda: [190, -12, 110, -25, -96, 10]
    )
    start_vel_scale: int = 60

    # DH Robotics AG-95 through the arm's built-in RS485 end-effector port.
    use_gripper: bool = True
    dedicated_gripper_dashboard: bool = True
    binary_gripper_actions: bool = True
    gripper_open_threshold: float = 0.5

    master_ip: str = "192.168.201.1"
    master_port: int = 60000
    tool_identify: int = 1
    dh_gripper_slave_id: int = 1
    dh_gripper_baudrate: int = 115200
    dh_gripper_force: int = 33
    dh_gripper_init_open: bool = True
    dh_gripper_worker_frequency: float = 50.0
    dh_gripper_position_poll_frequency: float = 0.0
    dh_gripper_command_epsilon: float = 0.0

    dh_gripper: DHGripperIntegratedConfig | None = field(default=None, init=False)

    enable_tactile_sensors: bool = False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Preserve explicitly supplied cameras; otherwise use the head and right-wrist cameras
        # from the bimanual configuration.
        if not self.cameras:
            from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

            self.cameras = {
                "head": RealSenseCameraConfig(
                    serial_number_or_name="254622078230",
                    fps=30,
                    width=1280,
                    height=720,
                    warmup_s=1.0,
                ),
                "wrist": RealSenseCameraConfig(
                    serial_number_or_name="352122272611",
                    fps=30,
                    width=640,
                    height=480,
                    warmup_s=1.0,
                ),
            }

        if self.enable_tactile_sensors:
            self.cameras.update(
                {
                    "tactile_0": XenseTactileCameraConfig(
                        serial_number="OG000339",
                        fps=30,
                        output_types=[XenseOutputType.RECTIFY],
                        warmup_s=0.05,
                    ),
                    "tactile_1": XenseTactileCameraConfig(
                        serial_number="OG000450",
                        fps=30,
                        output_types=[XenseOutputType.RECTIFY],
                        warmup_s=0.05,
                    ),
                }
            )

        super().__post_init__()
        self._validate()

        if self.use_gripper:
            self.dh_gripper = DHGripperIntegratedConfig(
                slave_id=self.dh_gripper_slave_id,
                baudrate=self.dh_gripper_baudrate,
                gripper_force=self.dh_gripper_force,
                init_open=self.dh_gripper_init_open,
                worker_frequency=self.dh_gripper_worker_frequency,
                position_poll_frequency=self.dh_gripper_position_poll_frequency,
                command_epsilon=self.dh_gripper_command_epsilon,
            )
        else:
            self.dh_gripper = None

    def _validate(self) -> None:
        if not 1 <= self.control_frequency <= 100:
            raise ValueError(
                f"control_frequency must be between 1 and 100 Hz, got {self.control_frequency}"
            )
        if self.cartesian_ik_backoff_steps < 0:
            raise ValueError(
                f"cartesian_ik_backoff_steps must be >= 0, got {self.cartesian_ik_backoff_steps}"
            )
        if self.cartesian_ik_servoj_hold_s < 0:
            raise ValueError(
                f"cartesian_ik_servoj_hold_s must be >= 0, got {self.cartesian_ik_servoj_hold_s}"
            )
        if self.max_cartesian_step_m < 0:
            raise ValueError(
                f"max_cartesian_step_m must be >= 0, got {self.max_cartesian_step_m}"
            )
        if self.async_action_worker_frequency <= 0:
            raise ValueError(
                "async_action_worker_frequency must be positive, "
                f"got {self.async_action_worker_frequency}"
            )
        if not 0.0 <= self.gripper_open_threshold <= 1.0:
            raise ValueError(
                f"gripper_open_threshold must be in [0, 1], got {self.gripper_open_threshold}"
            )
        if not 0 <= int(self.tool_coordinate_index) <= 9:
            raise ValueError(
                f"tool_coordinate_index must be between 0 and 9, got {self.tool_coordinate_index}"
            )
        if self.tool_identify not in (1, 2):
            raise ValueError(f"tool_identify must be 1 or 2, got {self.tool_identify}")
        self._validate_vector(self.workspace_min_xyz_m, 3, "workspace_min_xyz_m")
        self._validate_vector(self.workspace_max_xyz_m, 3, "workspace_max_xyz_m")
        for axis, lower, upper in zip(
            ("x", "y", "z"), self.workspace_min_xyz_m, self.workspace_max_xyz_m, strict=True
        ):
            if float(lower) >= float(upper):
                raise ValueError(
                    f"workspace has invalid {axis} bounds: min {lower} >= max {upper}"
                )
        self._validate_vector(self.start_position_degree, 6, "start_position_degree")
        self._validate_vector(self.home_point_list, 6, "home_point_list")
        if not 1 <= self.start_vel_scale <= 100:
            raise ValueError(
                f"start_vel_scale must be between 1 and 100, got {self.start_vel_scale}"
            )

    @staticmethod
    def _validate_vector(values: list[float], expected: int, name: str) -> None:
        if len(values) != expected:
            raise ValueError(f"{name} must have {expected} elements, got {len(values)}")
