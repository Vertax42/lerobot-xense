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
from lerobot.robots.flexiv_rizon4.config_flare_gripper import FlareGripperConfig, SensorOutputType


class ControlMode(str, Enum):
    """Control mode for Flexiv Rizon4.

    JOINT_IMPEDANCE:
        Joint impedance control (maps to NRT_JOINT_IMPEDANCE).
        Uses impedance control with configurable stiffness via stiffness_ratio.
        - Action: joint positions (7D) + gripper (1D) = 8D
        - Observation: joint positions (7D) + velocities (7D) + efforts (7D) + gripper (1D) = 22D

    CARTESIAN_MOTION_FORCE:
        Cartesian motion control (maps to NRT_CARTESIAN_MOTION_FORCE).
        When use_force=False: pure motion control
        When use_force=True: motion + force control
        - Action: TCP pose (7D) + gripper (1D) = 8D, or pose + wrench (13D) + gripper (1D) = 14D
        - Observation: TCP pose (7D) + gripper (1D) = 8D, or pose + wrench (13D) + gripper (1D) = 14D
    """

    JOINT_IMPEDANCE = "joint_impedance_control"
    CARTESIAN_MOTION_FORCE = "cartesian_motion_force_control"


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

        # Joint motion constraints (for joint impedance control mode)
        joint_max_vel: Maximum joint velocity [rad/s] for each joint
        joint_max_acc: Maximum joint acceleration [rad/s^2] for each joint

        # Cartesian motion parameters
        cartesian_max_linear_vel: Maximum Cartesian linear velocity [m/s]

        # Force control parameters (for CARTESIAN_MOTION_FORCE mode when use_force=True)
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
    # control_mode: JOINT_IMPEDANCE or CARTESIAN_MOTION_FORCE
    #   - JOINT_IMPEDANCE: maps to NRT_JOINT_IMPEDANCE mode (joint impedance control)
    #   - CARTESIAN_MOTION_FORCE: maps to NRT_CARTESIAN_MOTION_FORCE mode
    control_mode: ControlMode = ControlMode.CARTESIAN_MOTION_FORCE

    # use_force: Enable force control (only applies to CARTESIAN_MOTION_FORCE mode)
    #   - False: pure motion control, action/observation = TCP pose (7D)
    #   - True: motion + force control, action/observation = pose + wrench (13D)
    use_force: bool = False

    # NRT mode supports 1-100 Hz
    control_frequency: float = 100.0  # Hz

    # Connection behavior
    go_to_start: bool = (
        True  # If True, move robot to start position after connecting. If False, stay at current position.
    )

    # Camera configurations (external cameras, e.g., scene cameras)
    # Note: When using xense_flare, wrist camera comes from XenseFlareConfig
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Joint motion constraints (from examples: MAX_VEL = [2.0] * DoF, MAX_ACC = [3.0] * DoF)
    joint_max_vel: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # rad/s
    )
    joint_max_acc: list[float] = field(
        default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]  # rad/s^2
    )

    # Joint impedance control settings (for JOINT_IMPEDANCE mode, which uses NRT_JOINT_IMPEDANCE)
    # Stiffness ratio: multiplies nominal joint stiffness K_q_nom
    # Default 1.0 means no adjustment (100 percent nominal stiffness)
    # Example: 0.5 means 50 percent of nominal stiffness (more compliant)
    stiffness_ratio: float = 1.0

    # Cartesian motion parameters (from example: SEARCH_VELOCITY = 0.02 m/s)
    cartesian_max_linear_vel: float = 0.2  # m/s

    # Force control settings (for CARTESIAN_MOTION_FORCE mode when use_force=True)
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
    # Safety limit for contact forces during motion control (not force control mode).
    # When exceeded, robot will stop or reduce speed to prevent damage.
    #
    # Default values are conservative for general manipulation tasks:
    # - Forces: 30 N (suitable for grasping, pushing, insertion)
    # - Torques: 5 Nm (suitable for manipulation with moderate torques)
    #
    # For fine manipulation: use [10.0, 10.0, 10.0, 2.0, 2.0, 2.0]
    # For heavy manipulation: use [50.0, 50.0, 50.0, 10.0, 10.0, 10.0] or higher
    # For force control mode: set to [inf, inf, inf, inf, inf, inf] to disable (handled automatically)
    # Use inf to disable wrench regulation entirely
    max_contact_wrench: list[float] = field(default_factory=lambda: [30.0, 30.0, 30.0, 5.0, 5.0, 5.0])

    # Target wrench for force control [fx, fy, fz, mx, my, mz] in N and Nm
    # Zero means pure motion control (no force applied)
    # From example: PRESSING_FORCE = 5.0 N
    target_wrench: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Collision detection thresholds (from example)
    # EXT_FORCE_THRESHOLD = 10.0 N, EXT_TORQUE_THRESHOLD = 5.0 Nm
    ext_force_threshold: float = 10.0  # N
    ext_torque_threshold: float = 5.0  # Nm

    # Start position parameters (for MoveJ primitive)
    # Joint positions in degrees (factory-defined home position)
    start_position_degree: list[float] = field(
        default_factory=lambda: [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]
    )
    # Joint velocity scale for moving to start position (1-100, default 30)
    start_vel_scale: int = 30

    # Whether to zero force/torque sensors on connect
    # IMPORTANT: robot must not contact anything during zeroing
    zero_ft_sensor_on_connect: bool = True

    # Log level for the robot SDK
    log_level: str = "INFO"  # TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL

    # ========== Flare Gripper (end-effector) settings ==========
    # Whether to use the Flare Gripper end-effector
    # If False, flare_gripper will be None and no gripper functionality is available
    use_gripper: bool = True

    # Gripper identification (MAC address / serial number)
    flare_gripper_mac_addr: str = "e2b26adbb104"

    # Camera settings (wrist camera resolution)
    flare_gripper_cam_size: tuple[int, int] = (640, 480)

    # Tactile sensor settings
    flare_gripper_rectify_size: tuple[int, int] = (400, 700)
    flare_gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    flare_gripper_sensor_keys: dict[str, str] = field(
        default_factory=lambda: {
            "OG000657": "right_tactile",
            "OG000450": "left_tactile",
        }
    )

    # Gripper normalization: raw_pos / gripper_max_pos -> [0, 1]
    flare_gripper_max_pos: float = 85.0

    # Gripper control parameters for set_position()
    flare_gripper_v_max: float = 80.0  # Maximum velocity mm/s
    flare_gripper_f_max: float = 20.0  # Maximum force N

    # Initialize gripper to fully open on connect
    flare_gripper_init_open: bool = True

    # Auto-created in __post_init__ from flare_gripper_* parameters (do not set directly)
    flare_gripper: FlareGripperConfig | None = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        # Validate control frequency (NRT mode: 1-100 Hz)
        if not 1 <= self.control_frequency <= 100:
            raise ValueError(
                f"control_frequency must be between 1 and 100 Hz for NRT mode, got {self.control_frequency}"
            )

        # Validate joint parameters have correct length (7-DOF robot)
        if len(self.joint_max_vel) != 7:
            raise ValueError(f"joint_max_vel must have 7 elements, got {len(self.joint_max_vel)}")
        if len(self.joint_max_acc) != 7:
            raise ValueError(f"joint_max_acc must have 7 elements, got {len(self.joint_max_acc)}")

        # Validate Cartesian/force parameters have correct length (6-DOF)
        if len(self.force_control_axis) != 6:
            raise ValueError(f"force_control_axis must have 6 elements, got {len(self.force_control_axis)}")
        if len(self.max_contact_wrench) != 6:
            raise ValueError(f"max_contact_wrench must have 6 elements, got {len(self.max_contact_wrench)}")
        if len(self.target_wrench) != 6:
            raise ValueError(f"target_wrench must have 6 elements, got {len(self.target_wrench)}")

        # Validate start position parameters
        if len(self.start_position_degree) != 7:
            raise ValueError(
                f"start_position_degree must have 7 elements, got {len(self.start_position_degree)}"
            )
        if not 1 <= self.start_vel_scale <= 100:
            raise ValueError(f"start_vel_scale must be between 1 and 100, got {self.start_vel_scale}")

        # Create FlareGripperConfig from exposed parameters (only if use_gripper=True)
        if self.use_gripper:
            self.flare_gripper = FlareGripperConfig(
                mac_addr=self.flare_gripper_mac_addr,
                cam_size=self.flare_gripper_cam_size,
                rectify_size=self.flare_gripper_rectify_size,
                sensor_output_type=self.flare_gripper_sensor_output_type,
                sensor_keys=self.flare_gripper_sensor_keys,
                gripper_max_pos=self.flare_gripper_max_pos,
                gripper_v_max=self.flare_gripper_v_max,
                gripper_f_max=self.flare_gripper_f_max,
                init_open=self.flare_gripper_init_open,
            )
        else:
            self.flare_gripper = None
