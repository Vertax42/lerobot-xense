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

"""Flexiv Rizon4 robot implementation for LeRobot.

This module provides integration with Flexiv Rizon4 7-DOF collaborative robot,
supporting two control modes (NRT = Non-Real-Time for Python API):

1. JOINT_IMPEDANCE (maps to NRT_JOINT_IMPEDANCE):
   - Action: joint positions (7D) + gripper (1D) = 8D
   - Observation: joint positions (7D) + velocities (7D) + efforts (7D) + gripper (1D) = 22D
   - Uses impedance control with configurable stiffness via stiffness_ratio

2. CARTESIAN_MOTION_FORCE (maps to NRT_CARTESIAN_MOTION_FORCE):
   - When use_force=False: Pure motion control
     - Action: TCP pose (7D)
     - Observation: TCP pose (7D)
   - When use_force=True: Motion + force control
     - Action: TCP pose (7D) + target wrench (6D) = 13D
     - Observation: TCP pose (7D) + external wrench (6D) = 13D

Note: Python API can only use NRT modes due to language timing limitations.

Reference: https://rdk.flexiv.com/api/
"""

import math
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.utils.robot_utils import get_logger

try:
    import flexivrdk
except ImportError as e:
    raise ImportError(
        "flexivrdk is required for Flexiv Rizon4 robot. "
        "Please install it following the instructions at https://rdk.flexiv.com/"
    ) from e

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .flare_gripper import FlareGripper
from .config_flexiv_rizon4 import ControlMode, FlexivRizon4Config

# Alias for flexivrdk.Mode for convenience
Mode = flexivrdk.Mode

# Constants from flexivrdk
CART_DOF = 6  # Cartesian degrees of freedom
JOINT_DOF = 7  # Flexiv Rizon4 robot joint DOF
POSE_SIZE = 7  # Pose size (position + quaternion)


class FlexivRizon4(Robot):
    """Flexiv Rizon4 7-DOF collaborative robot.

    This class implements the LeRobot Robot interface for the Flexiv Rizon4 robot,
    supporting two control modes for joint space and Cartesian space control.

    Control Modes (NRT only for Python API):
        - JOINT_IMPEDANCE: Joint impedance control (maps to NRT_JOINT_IMPEDANCE)
          Uses impedance control with configurable stiffness via stiffness_ratio parameter.
        - CARTESIAN_MOTION_FORCE: Cartesian motion control with optional force (maps to NRT_CARTESIAN_MOTION_FORCE)
          Set use_force=True to enable force control in CARTESIAN_MOTION_FORCE mode.

    Example:
        >>> from lerobot.robots.flexiv_rizon4 import FlexivRizon4, FlexivRizon4Config, ControlMode
        >>> # Joint impedance control
        >>> config = FlexivRizon4Config(
        ...     robot_sn="Rizon4-123456",
        ...     control_mode=ControlMode.JOINT_IMPEDANCE,
        ... )
        >>> # Cartesian control with force sensing
        >>> config = FlexivRizon4Config(
        ...     robot_sn="Rizon4-123456",
        ...     control_mode=ControlMode.CARTESIAN_MOTION_FORCE,
        ...     use_force=True,  # Enable force control
        ... )
        >>> robot = FlexivRizon4(config)
        >>> robot.connect()
        >>> obs = robot.get_observation()
        >>> robot.send_action({"joint_1.pos": 0.0, ...})
        >>> robot.disconnect()
    """

    config_class = FlexivRizon4Config
    name = "flexiv_rizon4"

    def __init__(self, config: FlexivRizon4Config):
        super().__init__(config)
        self.config = config

        # Logger
        self.logger = get_logger("FlexivRizon4")

        # Robot interface (initialized on connect)
        # Note: Python API only supports NRT (Non-Real-Time) modes, no Scheduler needed
        self._robot: flexivrdk.Robot | None = None
        self._is_connected = False

        if config.use_gripper:
            self._flare_gripper = FlareGripper(config.flare_gripper)
            
        # Control state - stores the current flexivrdk.Mode
        self._current_mode: flexivrdk.Mode | None = None

        # Home TCP pose - stored after moving to home position
        # Format: [x, y, z, qw, qx, qy, qz] (7D)
        self._home_tcp_pose: np.ndarray | None = None

        # Gripper key (1D) - always used
        self._gripper_key = "gripper.pos"

        # Initialize keys and buffers based on control mode
        if config.control_mode == ControlMode.JOINT_IMPEDANCE:
            self._init_joint_mode()
        elif config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            self._init_cartesian_mode()
        else:
            raise ValueError(f"Unsupported control_mode: {config.control_mode}")

        np.set_printoptions(precision=6, suppress=True)

    def _init_joint_mode(self) -> None:
        """Initialize keys and buffers for joint impedance control mode."""
        # Joint state observation keys: joint_{1-7}.{pos, vel, effort}
        self._joint_pos_keys = tuple(
            f"joint_{i}.pos" for i in range(1, JOINT_DOF + 1)
        )
        self._joint_vel_keys = tuple(
            f"joint_{i}.vel" for i in range(1, JOINT_DOF + 1)
        )
        self._joint_effort_keys = tuple(
            f"joint_{i}.effort" for i in range(1, JOINT_DOF + 1)
        )

        # Joint action keys: joint_{1-7}.pos
        self._action_joint_keys = tuple(
            f"joint_{i}.pos" for i in range(1, JOINT_DOF + 1)
        )

        # Pre-cache config values as lists (for API calls)
        self._max_vel = self.config.joint_max_vel  # Already a list
        self._max_acc = self.config.joint_max_acc  # Already a list
        
        # Zero velocity array for SendJointPosition API
        self._zero_vel = [0.0] * JOINT_DOF

    def _init_cartesian_mode(self) -> None:
        """Initialize keys and buffers for CARTESIAN_MOTION_FORCE control mode."""
        # TCP pose observation keys: tcp.{x, y, z, qw, qx, qy, qz}
        self._tcp_pose_keys = (
            "tcp.x",
            "tcp.y",
            "tcp.z",
            "tcp.qw",
            "tcp.qx",
            "tcp.qy",
            "tcp.qz",
        )

        # TCP velocity observation keys: tcp.{vx, vy, vz, wx, wy, wz}
        self._tcp_vel_keys = (
            "tcp.vx",
            "tcp.vy",
            "tcp.vz",
            "tcp.wx",
            "tcp.wy",
            "tcp.wz",
        )

        # TCP pose action keys: tcp.{x, y, z, qw, qx, qy, qz}
        self._action_tcp_pose_keys = (
            "tcp.x",
            "tcp.y",
            "tcp.z",
            "tcp.qw",
            "tcp.qx",
            "tcp.qy",
            "tcp.qz",
        )

        # Pre-cache max contact wrench (always needed in Cartesian mode for safety)
        self._max_contact_wrench = self.config.max_contact_wrench

        # Initialize force-related keys if use_force is enabled
        if self.config.use_force:
            # Wrench keys: tcp.{fx, fy, fz, mx, my, mz}
            # Used for both observation (external wrench) and action (target wrench)
            self._wrench_keys = (
                "tcp.fx",
                "tcp.fy",
                "tcp.fz",
                "tcp.mx",
                "tcp.my",
                "tcp.mz",
            )
            # Action wrench keys are the same as observation wrench keys
            self._action_wrench_keys = self._wrench_keys

            # Pre-cache force control axis
            self._force_control_axis = tuple(self.config.force_control_axis)

    @property
    def _action_ft(self) -> dict[str, type]:
        """Return action features based on control_mode and use_force.

        Action space (all include gripper):
        - JOINT_IMPEDANCE: joint positions (7D) + gripper (1D) = 8D
        - CARTESIAN_MOTION_FORCE + use_force=False: TCP pose (7D) + gripper (1D) = 8D
        - CARTESIAN_MOTION_FORCE + use_force=True: TCP pose (7D) + wrench (6D) + gripper (1D) = 14D
        """
        features = {}

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            # Joint positions (7D)
            features.update({key: float for key in self._action_joint_keys})

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            # TCP pose (7D)
            features.update({key: float for key in self._action_tcp_pose_keys})
            if self.config.use_force:
                # + target wrench (6D)
                features.update({key: float for key in self._action_wrench_keys})

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Always include gripper (1D)
        features[self._gripper_key] = float
        return features

    @property
    def _proprioception_ft(self) -> dict[str, type]:
        """Return observation features based on control_mode and use_force.

        Observation space (all include gripper):
        - JOINT_IMPEDANCE: joint pos (7D) + vel (7D) + effort (7D) + gripper pos (1D) = 22D
        - CARTESIAN_MOTION_FORCE + use_force=False: TCP pose (7D) + gripper (1D) = 8D
        - CARTESIAN_MOTION_FORCE + use_force=True: TCP pose (7D) + wrench (6D) + gripper (1D) = 14D
        """
        features = {}

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            # Joint positions (7D)
            features.update({key: float for key in self._joint_pos_keys})
            # Joint velocities (7D)
            features.update({key: float for key in self._joint_vel_keys})
            # Joint efforts/torques (7D)
            features.update({key: float for key in self._joint_effort_keys})

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            # TCP pose (7D)
            features.update({key: float for key in self._tcp_pose_keys})
            if self.config.use_force:
                # + external wrench (6D)
                features.update({key: float for key in self._wrench_keys})

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Always include gripper position (1D)
        features[self._gripper_key] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Return camera/image features from Flare GriNonepper and external cameras."""
        features = {}
        
        if self._flare_gripper and self.config.use_gripper:
            features["wrist_cam"] = (self._flare_gripper.config.cam_size[1], self._flare_gripper.config.cam_size[0], 3)
            features["left_tactile"] = (self._flare_gripper.config.rectify_size[1], self._flare_gripper.config.rectify_size[0], 3)
            features["right_tactile"] = (self._flare_gripper.config.rectify_size[1], self._flare_gripper.config.rectify_size[0], 3)
        
        # External cameras (e.g., scene cameras)
        for cam in self.cameras:
            features[cam] = (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
        
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Return observation features (robot states + cameras)."""
        return {**self._proprioception_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Return action features based on control mode."""
        return self._action_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._robot is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        """Flexiv robots are factory calibrated."""
        return self.is_connected

    def calibrate(self) -> None:
        """Flexiv robots are factory calibrated, no runtime calibration needed."""
        self.logger.info(
            "Flexiv Rizon4 is factory calibrated, no runtime calibration needed."
        )

    def zero_ft_sensor(self) -> None:
        """Zero force-torque sensor offset.

        IMPORTANT: Robot must not contact anything during zeroing.
        This method should be called before using force control.
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.warn(
            "Zeroing force-torque sensors, make sure nothing is in contact with the robot"
        )

        # Switch to primitive execution mode
        self._robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)

        # Execute ZeroFTSensor primitive
        self._robot.ExecutePrimitive("ZeroFTSensor", dict())

        # Wait for primitive to finish
        while not self._robot.primitive_states()["terminated"]:
            time.sleep(0.1)

        self.logger.info("✅ Force-torque sensor zeroed")

    def configure(self) -> None:
        """Configure the robot based on control mode.

        Note: Uses robot.info() to get nominal impedance values (K_q_nom for joint,
        K_x_nom for Cartesian) as recommended in Flexiv RDK examples.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info(f"Configuring robot for {self.config.control_mode.value} mode...")

        # Get robot info for nominal impedance values
        robot_info = self._robot.info()

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            # Joint impedance mode - set joint stiffness based on stiffness_ratio
            if self.config.stiffness_ratio != 1.0:
                K_q_nom = robot_info.K_q_nom
                new_Kq = np.multiply(K_q_nom, self.config.stiffness_ratio)
                self._robot.SetJointImpedance(new_Kq)
                self.logger.info(f"Joint impedance mode - set stiffness (ratio={self.config.stiffness_ratio}): {new_Kq}")
            else:
                self.logger.info(f"Joint impedance mode - using nominal stiffness: {robot_info.K_q_nom}")
                return

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            # Cartesian mode configuration
            K_x_nom = robot_info.K_x_nom
            self.logger.info(f"Cartesian mode - nominal Cartesian stiffness K_x_nom: {K_x_nom}")

            # Configure force control if use_force is enabled
            if self.config.use_force:
                # Zero force-torque sensor before force control
                # IMPORTANT: Robot must not contact anything during zeroing
                self.zero_ft_sensor()

                # Set force control frame (flexivrdk.CoordType.WORLD or TCP)
                self._robot.SetForceControlFrame(self.config.force_control_frame)
                self.logger.info(f"Set force control frame: {self.config.force_control_frame}")

                # Set which axes use force control (use pre-cached tuple)
                self._robot.SetForceControlAxis(list(self._force_control_axis))
                self.logger.info(f"Set force control axis: {self._force_control_axis}")

                # Disable max contact wrench regulation after force control is activated
                # This allows explicit force control on force-controlled axes without interference
                # from the max contact wrench limit. Force-controlled axes will be explicitly
                # regulated, preventing force spikes after disabling the limit.
                self._robot.SetMaxContactWrench([float("inf")] * 6)
                self.logger.info("Max contact wrench regulation disabled (force control active)")
            else:
                # Pure motion control: set max contact wrench for safety
                self._robot.SetMaxContactWrench(self._max_contact_wrench)
                self.logger.info(f"Set max contact wrench: {self._max_contact_wrench}")

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        """Connect to the Flexiv robot.

        Args:
            calibrate: Ignored (Flexiv robots are factory calibrated)
            go_to_start: If provided, overrides config.go_to_start. If None, uses config.go_to_start.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            self.logger.info(f"Connecting to Flexiv robot: {self.config.robot_sn}")

            # Create robot interface
            self._robot = flexivrdk.Robot(self.config.robot_sn)

            # Clear any existing fault
            if self._robot.fault():
                self.logger.warn(
                    "Fault occurred on the connected robot, trying to clear ..."
                )
                # Try to clear the fault
                if not self._robot.ClearFault():
                    raise RuntimeError(
                        "Failed to clear robot fault. Check the robot status."
                    )
                self.logger.info("Fault on the connected robot is cleared")

            # Enable the robot
            self.logger.info("Enabling robot...")
            self._robot.Enable()

            # Wait for robot to become operational
            timeout = 30  # seconds
            start_time = time.time()
            while not self._robot.operational():
                if time.time() - start_time > timeout:
                    raise RuntimeError(
                        f"Robot did not become operational within {timeout} seconds"
                    )
                time.sleep(0.1)

            self.logger.info("Robot is now operational.")

            # Connect Flare Gripper end-effector (provides gripper + wrist_cam + tactile)
            if self._flare_gripper and self.config.use_gripper:
                self.logger.info("Connecting Flare Gripper...")
                self._flare_gripper.connect()

            # Connect external cameras (e.g., scene cameras)
            for cam in self.cameras.values():
                cam.connect()

            # Set _is_connected to True before calling methods that check is_connected
            self._is_connected = True

            # Move to start position if requested (use parameter if provided, otherwise use config)
            self.config.go_to_start = go_to_start if go_to_start is not None else self.config.go_to_start
            if self.config.go_to_start:
                self._go_to_start()

            # Switch to the configured control mode
            self._switch_to_control_mode()

            # Configure control parameters
            self.configure()

            mode_desc = self.config.control_mode.value
            if self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
                mode_desc += " (force enabled)" if self.config.use_force else " (motion only)"
            
            if self._flare_gripper and self.config.use_gripper:
                gripper_status = "with Flare Gripper (gripper + wrist_cam + tactile)"
            self.logger.info(f"✅ Flexiv Rizon4 connected and ready in {mode_desc} mode ({gripper_status}).")

        except Exception as e:
            self.logger.error(f"Failed to connect to Flexiv robot: {e}")
            self._robot = None
            self._is_connected = False
            raise

    def _go_to_home(self) -> None:
        """Move robot to home position using MoveJ primitive.

        Uses ExecutePrimitive("MoveJ") to move to factory-defined home pose:
        - target: [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0] degrees
        - jntVelScale: 20
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Moving to home position...")

        # Switch to primitive execution mode
        self._robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)

        # Factory-defined home position in degrees
        home_position_deg = [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]
        home_jpos = flexivrdk.JPos(home_position_deg)

        # Execute MoveJ primitive to move to home position
        self._robot.ExecutePrimitive(
            "MoveJ",
            {
                "target": home_jpos,
                "jntVelScale": 30,  # Joint velocity scale [1-100]
            },
        )
        if self.config.flare_gripper_init_open:
            self._flare_gripper._gripper.set_position_sync(self.config.flare_gripper_max_pos, vmax=self.config.flare_gripper_v_max / 2, fmax=self.config.flare_gripper_f_max / 2) # fully open
        else:
            self._flare_gripper._gripper.set_position_sync(0.0, vmax=self.config.flare_gripper_v_max / 2, fmax=self.config.flare_gripper_f_max / 2) # fully closed
        # Wait for target reached
        while not self._robot.primitive_states()["reachedTarget"]:
            time.sleep(0.1)
        self._home_tcp_pose = np.array(self._robot.states().tcp_pose)
        self.logger.info(f"Home TCP pose: {self._home_tcp_pose}")
        self.logger.info("✅ Robot at home position.")
        self.logger.info(f"Gripper position: {self._flare_gripper.get_gripper_position()}")

    def _go_to_start(self) -> None:
        """Move robot to start position using MoveJ primitive.

        Uses ExecutePrimitive("MoveJ") with configurable parameters:
        - jntVelScale: Joint velocity scale 1-100 (from config.start_vel_scale)
        - target: Start joint position in degrees (from config.start_position_degree)
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Moving to start position...")

        # Switch to primitive execution mode
        self._robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)

        # Create target joint position from config (in degrees)
        start_jpos = flexivrdk.JPos(self.config.start_position_degree)

        # Execute MoveJ primitive with custom start position
        self._robot.ExecutePrimitive(
            "MoveJ",
            {
                "target": start_jpos,
                "jntVelScale": self.config.start_vel_scale,
            },
        )
        if self.config.flare_gripper_init_open:
            self._flare_gripper._gripper.set_position_sync(self.config.flare_gripper_max_pos, vmax=self.config.flare_gripper_v_max / 2, fmax=self.config.flare_gripper_f_max / 2) # fully open
        else:
            self._flare_gripper._gripper.set_position_sync(0.0, vmax=self.config.flare_gripper_v_max / 2, fmax=self.config.flare_gripper_f_max / 2) # fully closed
        # Wait for target reached
        while not self._robot.primitive_states()["reachedTarget"]:
            time.sleep(0.1)

        self.logger.info("✅ Robot at start position.")
        self.logger.info(f"Gripper position: {self._flare_gripper.get_gripper_position()}")

    def reset_to_initial_position(self) -> None:
        """Reset robot to initial position based on config.go_to_start.

        If config.go_to_start=True, calls _go_to_start().
        Otherwise, calls _go_to_home().
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.go_to_start:
            self.logger.info("Resetting to start position (config.go_to_start=True)")
            self._go_to_start()
        else:
            self.logger.info("Resetting to home position (config.go_to_start=False)")
            self._go_to_home()

        # Switch back to control mode after reset
        self._switch_to_control_mode()

    def _switch_to_control_mode(self) -> None:
        """Switch to the control mode specified in config.

        Maps ControlMode to flexivrdk.Mode:
        - JOINT_IMPEDANCE -> NRT_JOINT_IMPEDANCE (joint impedance control)
        - CARTESIAN_MOTION_FORCE -> NRT_CARTESIAN_MOTION_FORCE

        Note: Python API uses NRT (Non-Real-Time) modes only.
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Map ControlMode to flexivrdk.Mode
        mode_map = {
            ControlMode.JOINT_IMPEDANCE: flexivrdk.Mode.NRT_JOINT_IMPEDANCE,
            ControlMode.CARTESIAN_MOTION_FORCE: flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE,
        }

        flexiv_mode = mode_map.get(self.config.control_mode)
        if flexiv_mode is None:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        current_mode = self._robot.mode()
        if current_mode == flexiv_mode:
            self.logger.info(f"Already in {self.config.control_mode.value} mode.")
            return

        self.logger.info(f"Switching from {current_mode} to {self.config.control_mode.value}...")

        # Check for fault before switching
        if self._robot.fault():
            self.logger.warn("Robot has fault, attempting to clear...")
            self._robot.ClearFault()
            time.sleep(0.5)
            if self._robot.fault():
                raise RuntimeError("Failed to clear robot fault before mode switch")

        self._robot.SwitchMode(flexiv_mode)
        
        # Wait for mode switch to complete and verify
        max_wait = 2.0  # seconds
        wait_start = time.perf_counter()
        while time.perf_counter() - wait_start < max_wait:
            actual_mode = self._robot.mode()
            if actual_mode == flexiv_mode:
                self._current_mode = flexiv_mode
                self.logger.info(f"✅ Now in {self.config.control_mode.value} mode.")
                return
            time.sleep(0.05)
        
        # Mode switch failed
        actual_mode = self._robot.mode()
        raise RuntimeError(
            f"Mode switch failed: expected {flexiv_mode}, got {actual_mode}. "
            f"Robot fault: {self._robot.fault()}"
        )

    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation based on control_mode and use_force.

        Returns a dictionary with observation data. The content depends on control_mode:
        - JOINT_IMPEDANCE: joint_1-7.{pos,vel,effort} (21D) + gripper.pos (1D) = 22D
        - CARTESIAN_MOTION_FORCE + use_force=False: tcp.{x,y,z,qw,qx,qy,qz} (7D) + gripper (1D) = 8D
        - CARTESIAN_MOTION_FORCE + use_force=True: tcp pose + external wrench (13D) + gripper (1D) = 14D

        Also includes camera images if configured.
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get robot states
        states = self._robot.states()
        obs_dict = {}

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            # Joint positions (7D)
            for i, key in enumerate(self._joint_pos_keys):
                obs_dict[key] = float(states.q[i])

            # Joint velocities (7D)
            for i, key in enumerate(self._joint_vel_keys):
                obs_dict[key] = float(states.dq[i])
            
            # Joint efforts/torques (7D)
            for i, key in enumerate(self._joint_effort_keys):
                obs_dict[key] = float(states.tau[i])

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            # TCP pose (7D)
            tcp_pose = states.tcp_pose
            for i, key in enumerate(self._tcp_pose_keys):
                obs_dict[key] = float(tcp_pose[i])

            if self.config.use_force:
                # + external wrench (6D)
                ext_wrench = states.ext_wrench_in_tcp
                for i, key in enumerate(self._wrench_keys):
                    obs_dict[key] = float(ext_wrench[i])

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Get data from Flare Gripper (gripper + wrist_cam + tactile sensors)
        if self._flare_gripper is not None and self.config.use_gripper:
            # Read sensors (keys are mapped from SN to sensor_keys names)
            sensor_data = self._flare_gripper.get_sensor_data()
            for key, data in sensor_data.items():
                obs_dict[key] = data

            # Read wrist camera
            camera_frame = self._flare_gripper.get_camera_frame()
            if camera_frame is not None:
                obs_dict["wrist_cam"] = camera_frame

            # Read gripper position
            obs_dict[self._gripper_key] = self._flare_gripper.get_gripper_position()

        # External camera observations (scene cameras, etc.)
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def get_current_tcp_pose_euler(self) -> np.ndarray:
        """Get current TCP pose in Euler angles format [x, y, z, roll, pitch, yaw, gripper_pos].
        
        This method can be used for getting the current TCP pose in Euler angles format for initializing teleoperators (e.g., spacemouse) with the robot's
        current TCP pose. Only available in CARTESIAN_MOTION_FORCE mode.
        
        Returns:
            numpy array of shape (7,) with [x, y, z, roll, pitch, yaw, gripper_pos]
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.config.control_mode != ControlMode.CARTESIAN_MOTION_FORCE:
            raise ValueError("get_current_tcp_pose_euler requires CARTESIAN_MOTION_FORCE mode")
        
        # Get current TCP pose (quaternion format)
        states = self._robot.states()
        tcp_pose = states.tcp_pose  # [x, y, z, qw, qx, qy, qz]
        
        # Convert quaternion to Euler angles
        qw, qx, qy, qz = tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6]
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        sinp = 2 * (qw * qy - qz * qx)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        
        # Get gripper position directly from XenseFlare gripper
        gripper_pos = 0.0
        if self._flare_gripper and self.config.use_gripper:
            gripper_pos = self._flare_gripper.get_gripper_position()
        
        # Return [x, y, z, roll, pitch, yaw, gripper_pos]
        return np.array([
            float(tcp_pose[0]),  # x
            float(tcp_pose[1]),  # y
            float(tcp_pose[2]),  # z
            float(roll),
            float(pitch),
            float(yaw),
            gripper_pos,
        ], dtype=np.float32)

    def get_current_tcp_pose_quat(self) -> np.ndarray:
        """Get current TCP pose in quaternion format [x, y, z, qw, qx, qy, qz, gripper_pos].
        
        This method can be used for getting the current TCP pose in quaternion format for initializing teleoperators (e.g., pico4) with the robot's
        current TCP pose. Only available in CARTESIAN_MOTION_FORCE mode.
        
        Returns:
            numpy array of shape (8,) with [x, y, z, qw, qx, qy, qz, gripper_pos]
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.config.control_mode != ControlMode.CARTESIAN_MOTION_FORCE:
            raise ValueError("get_current_tcp_pose_quat requires CARTESIAN_MOTION_FORCE mode")
        
        # Get current TCP pose (quaternion format)
        states = self._robot.states()
        tcp_pose = states.tcp_pose  # [x, y, z, qw, qx, qy, qz]
        
        # Get gripper position directly from XenseFlare gripper
        gripper_pos = 0.0
        if self._flare_gripper and self.config.use_gripper:
            gripper_pos = self._flare_gripper.get_gripper_position()
        
        # Return [x, y, z, qw, qx, qy, qz, gripper_pos]
        return np.array([
            float(tcp_pose[0]),  # x
            float(tcp_pose[1]),  # y
            float(tcp_pose[2]),  # z
            float(tcp_pose[3]),  # qw
            float(tcp_pose[4]),  # qx
            float(tcp_pose[5]),  # qy
            float(tcp_pose[6]),  # qz
            gripper_pos,
        ], dtype=np.float32)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action command to the robot.

        The action format depends on the control_mode and use_force:
        - JOINT_IMPEDANCE: {joint_i.pos: float} for i in 1..7, + gripper.pos
        - CARTESIAN_MOTION_FORCE + use_force=False: {tcp.x, ..., tcp.qz: float} + gripper.pos
        - CARTESIAN_MOTION_FORCE + use_force=True: pose + wrench + gripper.pos

        Args:
            action: Dictionary of action values

        Returns:
            The action that was actually sent (may be clipped for safety)
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Check for fault
        if self._robot.fault():
            raise RuntimeError(
                "Robot fault detected. Call robot.clear_fault() or reconnect."
            )

        # Send robot arm action
        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            result = self._send_joint_position_action(action)

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            if self.config.use_force:
                result = self._send_cartesian_motion_force_action(action)
            else:
                result = self._send_cartesian_pure_motion_action(action)

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Send gripper action
        self._send_gripper_action(action)

        return result

    def _send_joint_position_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send joint impedance command (NRT mode).

        Uses SendJointPosition with motion constraints (max_vel, max_acc) from config.
        API signature: SendJointPosition(target_pos, target_vel, max_vel, max_acc)

        Note: target_vel is always [0.0] * DoF for impedance control.

        Action keys: action.joint_{1-7}.pos
        """
        # Extract target positions directly from action
        target_pos = [action[key] for key in self._action_joint_keys]

        # Send command using NRT API
        self._robot.SendJointPosition(
            target_pos,
            self._zero_vel,
            self._max_vel,
            self._max_acc,
        )

        return action

    def _send_cartesian_pure_motion_action(
        self, action: dict[str, Any]
    ) -> dict[str, Any]:
        """Send Cartesian pure motion command (NRT mode, no force control).

        Action keys: action.tcp.{x,y,z,qw,qx,qy,qz}
        
        Note: Calling SendCartesianMotionForce with only target_pose results in pure motion control.
        """
        # Extract target TCP pose directly from action
        target_pose = [action[key] for key in self._action_tcp_pose_keys]

        # Send command using NRT API (pure motion - no wrench parameter needed)
        self._robot.SendCartesianMotionForce(target_pose)

        return action

    def _send_cartesian_motion_force_action(
        self, action: dict[str, Any]
    ) -> dict[str, Any]:
        """Send Cartesian motion-force command (NRT mode).

        Action keys: action.tcp.{x,y,z,qw,qx,qy,qz} + action.tcp.{fx,fy,fz,mx,my,mz}
        """
        # Extract target TCP pose and wrench directly from action
        target_pose = [action[key] for key in self._action_tcp_pose_keys]
        target_wrench = [action[key] for key in self._action_wrench_keys]

        # Send command using NRT API
        self._robot.SendCartesianMotionForce(
            target_pose,
            target_wrench,
        )

        return action

    def _send_gripper_action(self, action: dict[str, Any]) -> None:
        """Send gripper command using Flare Gripper.

        Action key: gripper.pos (normalized 0-1)
        """
        if not self._flare_gripper or not self.config.use_gripper:
            return

        if self._gripper_key not in action:
            return

        # Set gripper position
        self._flare_gripper.set_gripper_position(float(action[self._gripper_key])) # normalized [0, 1]

    def clear_fault(self) -> bool:
        """Attempt to clear robot fault.

        Returns:
            True if fault was cleared, False otherwise
        """
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._robot.fault():
            self.logger.info("No fault to clear.")
            return True

        self.logger.info("Attempting to clear fault...")
        result = self._robot.ClearFault()
        if result:
            self.logger.info("✅ Fault cleared successfully.")
        else:
            self.logger.error("Failed to clear fault.")
        return result

    def disconnect(self) -> None:
        """Disconnect from the robot safely.
        
        This method ensures safe disconnection even if errors occur during the process.
        It will attempt to:
        1. Move robot to home position (if possible)
        2. Stop any ongoing motion
        3. Disconnect gripper and cameras
        4. Clean up all resources
        
        All errors are caught and logged, but the disconnect process continues to ensure cleanup.
        """
        if not self._is_connected:
            self.logger.warn(f"{self} is not connected, skipping disconnect.")
            return

        try:
            self.logger.info("Disconnecting from Flexiv robot...")

            # Move to home position before disconnecting
            try:
                self._go_to_home()
            except Exception as e:
                self.logger.warn(f"Failed to move to home before disconnect: {e}")

            # Stop any ongoing motion
            if self._robot is not None:
                self._robot.Stop()

            # Disconnect XenseFlare (provides gripper + camera + sensors)
            if self._flare_gripper and self.config.use_gripper:
                self._flare_gripper.disconnect()

            # Disconnect external cameras
            for cam in self.cameras.values():
                cam.disconnect()

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        finally:
            self._robot = None
            self._flare_gripper = None
            self._is_connected = False
            self._current_mode = None
            self.logger.info("✅ Flexiv Rizon4 disconnected.")
