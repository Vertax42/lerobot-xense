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

import flexivrdk

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig

from .gripper import GripperConfig, GripperType


class ControlMode(str, Enum):
    """Control mode for Flexiv Rizon4.

    JOINT_POSITION:
        Joint position control (maps to NRT_JOINT_POSITION).
        - Action: joint positions (7D) + gripper (1D) = 8D
        - Observation: joint positions (7D) + gripper (1D) = 8D

    CARTESIAN:
        Cartesian motion control (maps to NRT_CARTESIAN_MOTION_FORCE).
        When use_force=False: pure motion control
        When use_force=True: motion + force control
        - Action: TCP pose (7D) + gripper (1D) = 8D, or pose + wrench (13D) + gripper (1D) = 14D
        - Observation: TCP pose (7D) + gripper (1D) = 8D, or pose + wrench (13D) + gripper (1D) = 14D
    """

    JOINT_POSITION = "joint_position"
    CARTESIAN = "cartesian"


@RobotConfig.register_subclass("flexiv_rizon4")
@dataclass
class FlexivRizon4Config(RobotConfig):
    """Configuration for Flexiv Rizon4 robot.

    The Flexiv Rizon4 is a 7-DOF collaborative robot with force sensing capabilities.

    Attributes:
        robot_sn: Serial number of the robot (e.g., "Rizon4-123456")
        control_mode: Control mode to use
        control_frequency: Control loop frequency in Hz (1-100 Hz for NRT modes)
        cameras: Dictionary of camera configurations
        inference_mode: Whether to use inference mode (vs teleoperation)

        # Joint motion constraints (for JOINT_POSITION and JOINT_IMPEDANCE modes)
        joint_max_vel: Maximum joint velocity [rad/s] for each joint
        joint_max_acc: Maximum joint acceleration [rad/s^2] for each joint

        # Cartesian motion parameters
        cartesian_max_linear_vel: Maximum Cartesian linear velocity [m/s]

        # Force control parameters (for CARTESIAN_MOTION_FORCE mode)
        force_control_frame: Reference frame for force control (flexivrdk.CoordType.WORLD or TCP)
        force_control_axis: Which axes to enable force control [x, y, z, rx, ry, rz]
        max_contact_wrench: Maximum contact wrench [fx, fy, fz, mx, my, mz] in N and Nm
        target_wrench: Target wrench for force control [fx, fy, fz, mx, my, mz]

        # Collision detection thresholds
        ext_force_threshold: External TCP force threshold for collision detection [N]
        ext_torque_threshold: External joint torque threshold for collision detection [Nm]
    """

    # Robot identification
    robot_sn: str = "Rizon4-063423"  # Robot serial number

    # Control settings
    # control_mode: JOINT_POSITION or CARTESIAN
    #   - JOINT_POSITION: maps to NRT_JOINT_POSITION mode
    #   - CARTESIAN: maps to NRT_CARTESIAN_MOTION_FORCE mode
    control_mode: ControlMode = ControlMode.CARTESIAN

    # use_force: Enable force control (only applies to CARTESIAN mode)
    #   - False: pure motion control, action/observation = TCP pose (7D)
    #   - True: motion + force control, action/observation = pose + wrench (13D)
    use_force: bool = False

    # NRT mode supports 1-100 Hz
    control_frequency: float = 100.0  # Hz

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Joint motion constraints (from examples: MAX_VEL = [2.0] * DoF, MAX_ACC = [3.0] * DoF)
    joint_max_vel: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # rad/s
    )
    joint_max_acc: list[float] = field(
        default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]  # rad/s^2
    )

    # Cartesian motion parameters (from example: SEARCH_VELOCITY = 0.02 m/s)
    cartesian_max_linear_vel: float = 0.1  # m/s, conservative default

    # Force control settings (for CARTESIAN_MOTION_FORCE mode)
    # Reference frame for force control (flexivrdk.CoordType.WORLD or flexivrdk.CoordType.TCP)
    force_control_frame: flexivrdk.CoordType = flexivrdk.CoordType.WORLD

    # Which Cartesian axes to enable force control [x, y, z, rx, ry, rz]
    # True = force control, False = motion control
    # Example: [False, False, True, False, False, False] enables force control only on Z axis
    force_control_axis: list[bool] = field(
        default_factory=lambda: [
            False,
            False,
            False,
            False,
            False,
            False,
        ]  # All motion control by default
    )

    # Maximum contact wrench [fx, fy, fz, mx, my, mz] in N and Nm
    # From example: max_wrench = [10.0, 10.0, 10.0, 2.0, 2.0, 2.0]
    # Use inf to disable wrench regulation
    max_contact_wrench: list[float] = field(
        default_factory=lambda: [10.0, 10.0, 10.0, 2.0, 2.0, 2.0]
    )

    # Target wrench for force control [fx, fy, fz, mx, my, mz] in N and Nm
    # Zero means pure motion control (no force applied)
    # From example: PRESSING_FORCE = 5.0 N
    target_wrench: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # Collision detection thresholds (from example)
    # EXT_FORCE_THRESHOLD = 10.0 N, EXT_TORQUE_THRESHOLD = 5.0 Nm
    ext_force_threshold: float = 10.0  # N
    ext_torque_threshold: float = 5.0  # Nm

    # Start position parameters (for MoveJ primitive)
    # Joint positions in degrees (factory-defined home position)
    start_position_degree: list[float] = field(
        default_factory=lambda: [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]
    )
    # Joint velocity scale for moving to start position (1-100, default 20)
    start_vel_scale: int = 20

    # Whether to zero force/torque sensors on connect
    # IMPORTANT: robot must not contact anything during zeroing
    zero_ft_sensor_on_connect: bool = True

    # Log level for the robot SDK
    log_level: str = "WARNING"  # TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Gripper configuration
    # Set gripper_type to GripperType.NONE to disable gripper
    # Available types: NONE, FLEXIV_GRAV
    gripper: GripperConfig = field(default_factory=GripperConfig)

    def __post_init__(self):
        super().__post_init__()

        # Validate control frequency (NRT mode: 1-100 Hz)
        if not 1 <= self.control_frequency <= 100:
            raise ValueError(
                f"control_frequency must be between 1 and 100 Hz for NRT mode, got {self.control_frequency}"
            )

        # Validate joint parameters have correct length (7-DOF robot)
        if len(self.joint_max_vel) != 7:
            raise ValueError(
                f"joint_max_vel must have 7 elements, got {len(self.joint_max_vel)}"
            )
        if len(self.joint_max_acc) != 7:
            raise ValueError(
                f"joint_max_acc must have 7 elements, got {len(self.joint_max_acc)}"
            )

        # Validate Cartesian/force parameters have correct length (6-DOF)
        if len(self.force_control_axis) != 6:
            raise ValueError(
                f"force_control_axis must have 6 elements, got {len(self.force_control_axis)}"
            )
        if len(self.max_contact_wrench) != 6:
            raise ValueError(
                f"max_contact_wrench must have 6 elements, got {len(self.max_contact_wrench)}"
            )
        if len(self.target_wrench) != 6:
            raise ValueError(
                f"target_wrench must have 6 elements, got {len(self.target_wrench)}"
            )

        # Validate start position parameters
        if len(self.start_position_degree) != 7:
            raise ValueError(
                f"start_position_degree must have 7 elements, got {len(self.start_position_degree)}"
            )
        if not 1 <= self.start_vel_scale <= 100:
            raise ValueError(
                f"start_vel_scale must be between 1 and 100, got {self.start_vel_scale}"
            )
