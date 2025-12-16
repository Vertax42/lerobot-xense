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

import logging
import math
import time
from functools import cached_property
from typing import Any, Sequence

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_arx5_follower import ARX5FollowerConfig, ARX5ControlMode

# Import ARX5 interface
import os
import sys

# Add ARX5 SDK path to Python path (reuse from bi_arx5)
current_dir = os.path.dirname(os.path.abspath(__file__))
arx5_sdk_path = os.path.join(current_dir, "..", "bi_arx5", "ARX5_SDK", "python")
if arx5_sdk_path not in sys.path:
    sys.path.insert(0, arx5_sdk_path)

try:
    import arx5_interface as arx5
except ImportError as e:
    if "LogLevel" in str(e) and "already registered" in str(e):
        # LogLevel already registered, try to get the existing module
        if "arx5_interface" in sys.modules:
            arx5 = sys.modules["arx5_interface"]
        else:
            raise e
    else:
        raise e

logger = logging.getLogger(__name__)


class ARX5Follower(Robot):
    """
    [Single ARX5 Arm Follower Robot]

    A simplified version of BiARX5 for single-arm operation.
    Suitable for teleoperation with one follower arm.
    """

    config_class = ARX5FollowerConfig
    name = "arx5_follower"

    def __init__(self, config: ARX5FollowerConfig):
        super().__init__(config)
        self.config = config

        # Init arm when connect
        self.arm = None
        self._is_connected = False

        # Control mode state variables
        self._is_gravity_compensation_mode = False
        self._is_cartesian_control_mode = False
        self._is_joint_control_mode = False

        # Use configurable preview time for inference mode
        self.default_preview_time = (
            self.config.preview_time if self.config.inference_mode else 0.0
        )

        # RPC timeout
        self.rpc_timeout: float = getattr(config, "rpc_timeout", 5.0)

        # Pre-compute action keys for faster lookup (performance optimization)
        if config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            self._action_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
            self._gripper_key = "gripper_pos"
        else:
            self._action_keys = [f"joint_{i+1}.pos" for i in range(6)]
            self._gripper_key = "gripper.pos"

        # Pre-allocate command buffers (initialized in connect based on control mode)
        self._cmd_buffer = None  # JointState buffer for joint control
        self._eef_cmd_buffer = None  # EEFState buffer for cartesian control

        # Define home position (all joints at 0, gripper closed)
        self._home_position = self.config.home_position
        self._start_position = self.config.start_position

        # Robot config
        self.robot_config = arx5.RobotConfigFactory.get_instance().get_config(
            config.arm_model
        )

        # Create solver for FK/IK calculations
        urdf_path = os.path.join(
            current_dir, "..", "bi_arx5", "ARX5_SDK", "models", f"{config.arm_model}.urdf"
        )
        self._solver = arx5.Arx5Solver(
            urdf_path,
            self.robot_config.joint_dof,
            self.robot_config.joint_pos_min,
            self.robot_config.joint_pos_max,
        )

        if config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            # Convert joint positions to EEF positions using FK
            home_joint_pos = np.array(self._home_position[:6], dtype=np.float64)
            start_joint_pos = np.array(self._start_position[:6], dtype=np.float64)
            # Replace with EEF positions (x, y, z, roll, pitch, yaw, gripper)
            self._home_position = np.concatenate([
                self._solver.forward_kinematics(home_joint_pos),
                [self._home_position[6]]  # gripper
            ])
            self._start_position = np.concatenate([
                self._solver.forward_kinematics(start_joint_pos),
                [self._start_position[6]]  # gripper
            ])
            logger.info(f"EEF home position (FK): {self._home_position}")
            logger.info(f"EEF start position (FK): {self._start_position}")

        # Set gripper_open_readout
        self.robot_config.gripper_open_readout = config.gripper_open_readout

        # Controller config - select based on control mode
        if config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            controller_type = "cartesian_controller"
            # Cartesian controller requires background_send_recv = True
            use_background = True
        else:
            controller_type = "joint_controller"
            use_background = config.use_multithreading

        self.controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            controller_type, self.robot_config.joint_dof
        )
        logger.info(f"Using {controller_type} for control mode: {config.control_mode.value}")

        # Set controller_dt and default_preview_time
        self.controller_config.controller_dt = config.controller_dt
        self.controller_config.default_preview_time = self.default_preview_time

        # Background send/recv setting
        self.controller_config.background_send_recv = use_background

        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=3, suppress=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Return motor features based on control mode."""
        if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: EEF pose (x, y, z, roll, pitch, yaw) + gripper
            return {
                "x": float,
                "y": float,
                "z": float,
                "roll": float,
                "pitch": float,
                "yaw": float,
                "gripper_pos": float,
            }
        else:
            # Joint mode (including teach mode): 6 joints + gripper
            joint_names = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]
            return {f"{joint}.pos": float for joint in joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        print(f"camera_features: {self._cameras_ft}")
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self.arm is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def is_gravity_compensation_mode(self) -> bool:
        """Check if robot is currently in gravity compensation mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_gravity_compensation_mode

    def is_position_control_mode(self) -> bool:
        """Check if robot is currently in position control mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_joint_control_mode

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            logger.info(f"Creating arm controller (mode: {self.config.control_mode.value})...")
            if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
                self.arm = arx5.Arx5CartesianController(
                    self.robot_config,
                    self.controller_config,
                    self.config.arm_port,
                )
            else:
                # Joint control or Gravity comp mode
                self.arm = arx5.Arx5JointController(
                    self.robot_config,
                    self.controller_config,
                    self.config.arm_port,
                )
            time.sleep(0.5)
            logger.info(f"✓ Arm controller created successfully ({type(self.arm).__name__})")
            logger.info(
                f"preview_time: {self.controller_config.default_preview_time}"
            )
        except Exception as e:
            logger.error(f"Failed to create robot controller: {e}")
            self.arm = None
            raise e

        # Set log level
        self.set_log_level(self.config.log_level)

        # Reset to home using SDK method
        self.reset_to_home()

        # Set gravity compensation gain
        zero_grav_gain = self.arm.get_gain()
        zero_grav_gain.kp()[:] = 0.0
        zero_grav_gain.kd()[:] = self.controller_config.default_kd * 0.15
        zero_grav_gain.gripper_kp = 0.0
        zero_grav_gain.gripper_kd = self.controller_config.default_gripper_kd * 0.25

        self.arm.set_gain(zero_grav_gain)

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        # Initialize command buffer for optimized send_action
        if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            self._eef_cmd_buffer = arx5.EEFState()
            self._cmd_buffer = None  # Not used in cartesian mode
        else:
            self._cmd_buffer = arx5.JointState(self.robot_config.joint_dof)
            self._eef_cmd_buffer = None  # Not used in joint mode

        self._is_connected = True

        # Set control mode state based on config
        if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            self._is_cartesian_control_mode = True
            self._is_joint_control_mode = False
            self._is_gravity_compensation_mode = False
        else:
            # Joint control - start in gravity compensation mode
            self._is_cartesian_control_mode = False
            self._is_joint_control_mode = False
            self._is_gravity_compensation_mode = True

        # Go to start position, ready for data collection or inference
        logger.info("ARX5 Follower Robot connected.")
        if go_to_start:
            self.smooth_go_start(duration=2.0)
            logger.info(
                "✓ Robot go to start position, arm is now in gravity compensation mode"
            )
        else:
            logger.info(
                "Robot go to home position, arm is now in gravity compensation mode"
            )

        gain = self.arm.get_gain()
        logger.info(
            f"Current arm gain: {gain.kp()}, {gain.kd()}, {gain.gripper_kp}, {gain.gripper_kd}"
        )

        if self.config.inference_mode:
            if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
                # Cartesian mode is already in position control
                logger.info("✓ Robot is in cartesian control mode for inference")
            else:
                self.set_to_normal_position_control()
                logger.info("✓ Robot is now in joint position control mode for inference")

    @property
    def is_calibrated(self) -> bool:
        """
        ARX5 does not need to calibrate in runtime
        """
        return self.is_connected

    def calibrate(self) -> None:
        """ARX5 does not need to calibrate in runtime"""
        logger.info("ARX5 does not need to calibrate in runtime, skip...")
        return

    def configure(self) -> None:
        """
        Configure the robot
        """
        pass

    def setup_motors(self) -> None:
        """ARX5 motors are pre-configured, no runtime setup needed"""
        logger.info(
            f"{self} ARX5 motors are pre-configured, no runtime setup needed"
        )
        logger.info("Motor IDs are defined in the robot configuration:")
        logger.info("  - Joint motors: [1, 2, 4, 5, 6, 7]")
        logger.info("  - Gripper motor: 8")
        logger.info("Make sure your hardware matches these ID configurations")
        return

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: get EEF state
            eef_state = self.arm.get_eef_state()
            pose_6d = eef_state.pose_6d().copy()
            for i, key in enumerate(self._action_keys):
                obs_dict[key] = float(pose_6d[i])
            obs_dict["gripper_pos"] = float(eef_state.gripper_pos)
        else:
            # Joint mode (including teach mode): get joint state
            joint_state = self.arm.get_joint_state()
            pos = joint_state.pos().copy()
            for i in range(6):  # 6 joints
                obs_dict[f"joint_{i+1}.pos"] = float(pos[i])
            obs_dict["gripper.pos"] = float(joint_state.gripper_pos)

        # Add camera observations
        camera_times = {}

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            image = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            obs_dict[cam_key] = image
            camera_times[cam_key] = dt_ms

        # Store camera timing info for debugging
        self.last_camera_times = camera_times

        # Print camera read times
        # camera_summary = ", ".join(
        #     [f"{k}:{v:.1f}ms" for k, v in sorted(camera_times.items())]
        # )
        # logger.info(f"📷 Cameras [{parallel_total:.1f}ms total]: {camera_summary}")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: use EEF command
            cmd = self._eef_cmd_buffer
            pose_6d = cmd.pose_6d()
            for i, key in enumerate(self._action_keys):
                pose_6d[i] = action.get(key, pose_6d[i])
            cmd.gripper_pos = action.get(self._gripper_key, cmd.gripper_pos)

            # Set timestamp for trajectory interpolation
            cmd.timestamp = self.arm.get_timestamp() + self.default_preview_time
            # Use set_eef_traj for smoother motion (auto velocity calculation)
            # Debug: Print commands before sending
            # print(
            #     f"Arm command - pose_6d: {cmd.pose_6d()}, gripper: {cmd.gripper_pos}"
            # )
            self.arm.set_eef_traj([cmd])
        else:
            # Joint mode (including teach mode): use joint command
            cmd = self._cmd_buffer
            pos = cmd.pos()
            for i, key in enumerate(self._action_keys):
                pos[i] = action.get(key, pos[i])
            cmd.gripper_pos = action.get(self._gripper_key, cmd.gripper_pos)
            # Debug: Print commands before sending
            # print(
            #     f"Arm command - pos: {cmd.pos()}, gripper: {cmd.gripper_pos}"
            # )
            self.arm.set_joint_cmd(cmd)
        # Return the input action
        return action

    @staticmethod
    def _ease_in_out_quad(t: float) -> float:
        """Smooth easing function used for joint interpolation."""
        tt = t * 2.0
        if tt < 1.0:
            return (tt * tt) / 2.0
        tt -= 1.0
        return -(tt * (tt - 2.0) - 1.0) / 2.0

    def move_joint_trajectory(
        self,
        target_joint_poses: Sequence[float] | Sequence[Sequence[float]],
        durations: float | Sequence[float],
        *,
        easing: str = "ease_in_out_quad",
        steps_per_segment: int | None = None,
    ) -> None:
        """Move the arm smoothly towards the provided joint targets.

        Uses SDK's set_joint_traj for smooth trajectory execution with automatic
        velocity calculation. Gripper is interpolated linearly.

        Args:
            target_joint_poses: A sequence of 6 or 7 joint values (including gripper)
                or a sequence of such sequences to execute multiple segments.
            durations: Duration in seconds for the corresponding target poses.
            easing: Easing profile to apply ("ease_in_out_quad" or "linear").
            steps_per_segment: Optional fixed number of interpolation steps per
                segment. When omitted the controller's ``controller_dt`` is used
                to compute the number of steps from the duration.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
            ValueError: If inputs are malformed.
        """

        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Normalize input to list of targets
        if isinstance(target_joint_poses[0], (int, float)):
            trajectory = [target_joint_poses]
        else:
            trajectory = list(target_joint_poses)

        if isinstance(durations, (int, float)):
            segment_durations = [float(durations)]
        else:
            segment_durations = [float(d) for d in durations]

        if len(trajectory) != len(segment_durations):
            raise ValueError(
                "target_joint_poses and durations must have the same length"
            )

        # Determine controller timestep (fallback to 10 ms if unavailable)
        controller_dt = getattr(self.config, "interpolation_controller_dt", 0.01)

        def _get_current_state() -> tuple[np.ndarray, float]:
            state = self.arm.get_joint_state()
            return state.pos().copy(), state.gripper_pos

        def _parse_target(
            values: Sequence[float], default_gripper: float
        ) -> tuple[np.ndarray, float]:
            arr = np.asarray(values, dtype=float)
            if arr.shape[0] not in (6, 7):
                raise ValueError(
                    "Target must provide 6 joint values (+ optional gripper)"
                )
            if arr.shape[0] == 6:
                return arr, default_gripper
            return arr[:6], float(arr[6])

        def _apply_easing(alpha: float) -> float:
            alpha = max(0.0, min(1.0, alpha))
            if easing == "ease_in_out_quad":
                return self._ease_in_out_quad(alpha)
            if easing == "linear":
                return alpha
            raise ValueError(f"Unsupported easing profile: {easing}")

        current_pos, current_gripper = _get_current_state()

        try:
            for segment, duration in zip(trajectory, segment_durations, strict=True):
                target_pos, target_gripper = _parse_target(segment, current_gripper)

                if duration <= 0:
                    # Instant move
                    joint_cmd = arx5.JointState(self.robot_config.joint_dof)
                    joint_cmd.pos()[:] = target_pos
                    joint_cmd.gripper_pos = target_gripper
                    self.arm.set_joint_cmd(joint_cmd)
                    current_pos = target_pos
                    current_gripper = target_gripper
                    continue

                steps = (
                    steps_per_segment
                    if steps_per_segment is not None
                    else max(1, int(math.ceil(duration / controller_dt)))
                )

                # Build joint trajectory with easing applied to joint positions,
                # linear interpolation for gripper
                joint_traj = []
                current_timestamp = self.arm.get_joint_state().timestamp

                for step in range(1, steps + 1):
                    progress = step / steps
                    ratio = _apply_easing(progress)

                    # Interpolate joint positions with easing
                    interp_pos = current_pos + (target_pos - current_pos) * ratio
                    # Linearly interpolate gripper (no easing needed for gripper)
                    interp_gripper = current_gripper + (
                        target_gripper - current_gripper
                    ) * progress

                    timestamp = current_timestamp + (duration / steps) * step
                    joint_cmd = arx5.JointState(self.robot_config.joint_dof)
                    joint_cmd.pos()[:] = interp_pos
                    joint_cmd.gripper_pos = interp_gripper
                    joint_cmd.timestamp = timestamp
                    joint_traj.append(joint_cmd)

                # Send trajectory to SDK (SDK handles smooth interpolation internally)
                self.arm.set_joint_traj(joint_traj)

                # Wait for trajectory to complete
                time.sleep(duration)

                current_pos = target_pos
                current_gripper = target_gripper

        except KeyboardInterrupt:
            logger.warning(
                "Joint trajectory interrupted by user. Holding current pose."
            )

    def move_eef_trajectory(
        self,
        target_eef_poses: Sequence[float] | Sequence[Sequence[float]],
        durations: float | Sequence[float],
        *,
        easing: str = "ease_in_out_quad",
        steps_per_segment: int | None = None,
    ) -> None:
        """Move the arm smoothly towards the provided EEF targets (Cartesian mode).

        Uses SDK's set_eef_traj for smooth trajectory execution with automatic
        velocity calculation. Gripper is interpolated separately.

        Args:
            target_eef_poses: A sequence of 6 or 7 values (x,y,z,roll,pitch,yaw + optional gripper)
                or a sequence of such sequences to execute multiple segments.
            durations: Duration in seconds for the corresponding target poses.
            easing: Easing profile to apply ("ease_in_out_quad" or "linear").
            steps_per_segment: Optional fixed number of interpolation steps per segment.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
            ValueError: If inputs are malformed or not in Cartesian mode.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode != ARX5ControlMode.CARTESIAN_CONTROL:
            raise ValueError("move_eef_trajectory requires CARTESIAN_CONTROL mode")

        # Normalize input to list of targets
        if isinstance(target_eef_poses[0], (int, float)):
            trajectory = [target_eef_poses]
        else:
            trajectory = list(target_eef_poses)

        if isinstance(durations, (int, float)):
            segment_durations = [float(durations)]
        else:
            segment_durations = [float(d) for d in durations]

        if len(trajectory) != len(segment_durations):
            raise ValueError(
                "target_eef_poses and durations must have the same length"
            )

        controller_dt = getattr(self.config, "interpolation_controller_dt", 0.01)

        def _get_current_state() -> tuple[np.ndarray, float]:
            state = self.arm.get_eef_state()
            return state.pose_6d().copy(), state.gripper_pos

        def _parse_target(
            values: Sequence[float], default_gripper: float
        ) -> tuple[np.ndarray, float]:
            arr = np.asarray(values, dtype=float)
            if arr.shape[0] not in (6, 7):
                raise ValueError(
                    "Target must provide 6 EEF values (+ optional gripper)"
                )
            if arr.shape[0] == 6:
                return arr, default_gripper
            return arr[:6], float(arr[6])

        def _apply_easing(alpha: float) -> float:
            alpha = max(0.0, min(1.0, alpha))
            if easing == "ease_in_out_quad":
                return self._ease_in_out_quad(alpha)
            if easing == "linear":
                return alpha
            raise ValueError(f"Unsupported easing profile: {easing}")

        current_pose, current_gripper = _get_current_state()

        try:
            for segment, duration in zip(trajectory, segment_durations, strict=True):
                target_pose, target_gripper = _parse_target(segment, current_gripper)

                if duration <= 0:
                    # Instant move
                    eef_cmd = arx5.EEFState(target_pose, target_gripper)
                    self.arm.set_eef_cmd(eef_cmd)
                    current_pose = target_pose
                    current_gripper = target_gripper
                    continue

                steps = (
                    steps_per_segment
                    if steps_per_segment is not None
                    else max(1, int(math.ceil(duration / controller_dt)))
                )

                # Build EEF trajectory with easing applied to pose,
                # linear interpolation for gripper
                eef_traj = []
                current_timestamp = self.arm.get_eef_state().timestamp

                for step in range(1, steps + 1):
                    progress = step / steps
                    ratio = _apply_easing(progress)

                    # Interpolate pose with easing
                    interp_pose = current_pose + (target_pose - current_pose) * ratio
                    # Linearly interpolate gripper (no easing needed for gripper)
                    interp_gripper = current_gripper + (
                        target_gripper - current_gripper
                    ) * progress

                    timestamp = current_timestamp + (duration / steps) * step
                    eef_cmd = arx5.EEFState(interp_pose, interp_gripper)
                    eef_cmd.timestamp = timestamp
                    eef_traj.append(eef_cmd)

                # Send trajectory to SDK (SDK handles smooth interpolation internally)
                self.arm.set_eef_traj(eef_traj)

                # Wait for trajectory to complete
                time.sleep(duration)

                current_pose = target_pose
                current_gripper = target_gripper

        except KeyboardInterrupt:
            logger.warning(
                "EEF trajectory interrupted by user. Holding current pose."
            )

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Reset to home and set to damping mode for safety
        try:
            logger.info("Disconnecting arm...")
            self.arm.reset_to_home()
            self.arm.set_to_damping()
            logger.info("✓ Arm disconnected successfully")
        except Exception as e:
            logger.warning(f"Arm disconnect failed: {e}")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Destroy arm object - this triggers SDK cleanup
        self.arm = None

        self._is_connected = False

        logger.info(f"{self} disconnected.")

    def set_log_level(self, level: str):
        """Set robot log level

        Args:
            level: Log level string, supports: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF
        """
        # Convert string to LogLevel enum
        log_level_map = {
            "TRACE": arx5.LogLevel.TRACE,
            "DEBUG": arx5.LogLevel.DEBUG,
            "INFO": arx5.LogLevel.INFO,
            "WARNING": arx5.LogLevel.WARNING,
            "ERROR": arx5.LogLevel.ERROR,
            "CRITICAL": arx5.LogLevel.CRITICAL,
            "OFF": arx5.LogLevel.OFF,
        }

        if level.upper() not in log_level_map:
            raise ValueError(
                f"Invalid log level: {level}. Supported levels: {list(log_level_map.keys())}"
            )

        log_level = log_level_map[level.upper()]

        # Set log level for arm if connected
        if self.arm is not None:
            self.arm.set_log_level(log_level)

    def reset_to_home(self):
        """Reset arm to home position"""
        if self.arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.arm.reset_to_home()
        logger.info("Arm reset to home position.")

    def set_to_gravity_compensation_mode(self):
        """Switch from normal position control to gravity compensation mode"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to gravity compensation mode...")

        # Reset to zero PD with 0.15 * default kd
        zero_grav_gain = arx5.Gain(self.robot_config.joint_dof)
        zero_grav_gain.kp()[:] = 0.0
        zero_grav_gain.kd()[:] = self.controller_config.default_kd * 0.15
        zero_grav_gain.gripper_kp = 0.0
        zero_grav_gain.gripper_kd = self.controller_config.default_gripper_kd * 0.25

        self.arm.set_gain(zero_grav_gain)

        # Update control mode state
        self._is_gravity_compensation_mode = True
        self._is_joint_control_mode = False

        logger.info("✓ Arm is now in gravity compensation mode")

    def set_to_normal_position_control(self):
        """Switch from gravity compensation to normal position control mode"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Switching to normal position control mode...")

        # Reset to default gain
        default_gain = self.arm.get_gain()
        default_gain.kp()[:] = self.controller_config.default_kp * 0.5
        default_gain.kd()[:] = self.controller_config.default_kd * 1.5
        default_gain.gripper_kp = self.controller_config.default_gripper_kp
        default_gain.gripper_kd = self.controller_config.default_gripper_kd

        self.arm.set_gain(default_gain)

        # Update control mode state
        self._is_gravity_compensation_mode = False
        self._is_joint_control_mode = True

        logger.info("✓ Arm is now in normal position control mode")

    def smooth_go_start(
        self, duration: float = 2.0, easing: str = "ease_in_out_quad"
    ) -> None:
        """
        Smoothly move the arm to the start position using trajectory interpolation.

        For Joint mode:
        1. Switches to normal position control mode
        2. Moves the arm to start position over the specified duration
        3. Switches back to gravity compensation mode

        For Cartesian mode:
        1. Moves the arm to start EEF position over the specified duration
        (No mode switching needed - already in position control)

        Args:
            duration: Duration in seconds for the movement (default: 2.0)
            easing: Easing profile to apply ("ease_in_out_quad" or "linear")

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info(f"Smoothly going to start position over {duration:.1f} seconds...")

        if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: use EEF trajectory (already in position control)
            self.move_eef_trajectory(
                target_eef_poses=self._start_position.copy(),
                durations=duration,
                easing=easing,
            )
            logger.info("✓ Successfully going to start position (cartesian mode)")
        else:
            # Joint mode: need to switch modes
            # First, set current position as target to avoid large position error
            state = self.arm.get_joint_state()

            # Set current position as command to avoid SDK protection
            current_cmd = arx5.JointState(self.robot_config.joint_dof)
            current_cmd.pos()[:] = state.pos()
            current_cmd.gripper_pos = state.gripper_pos

            self.arm.set_joint_cmd(current_cmd)

            # Now safe to switch to normal position control
            self.set_to_normal_position_control()

            # Execute smooth trajectory to start position
            self.move_joint_trajectory(
                target_joint_poses=self._start_position.copy(),
                durations=duration,
                easing=easing,
            )

            # Switch back to gravity compensation mode (only for joint mode)
            self.set_to_gravity_compensation_mode()
            logger.info(
                "✓ Successfully going to start position and switched to gravity compensation mode"
            )

    def smooth_go_home(
        self, duration: float = 2.0, easing: str = "ease_in_out_quad"
    ) -> None:
        """
        Smoothly move the arm to the home position using trajectory interpolation.

        For Joint mode:
        1. Switches to normal position control mode
        2. Moves the arm to home position over the specified duration
        3. Switches back to gravity compensation mode

        For Cartesian mode:
        1. Moves the arm to home EEF position over the specified duration
        (No mode switching needed - already in position control)

        Args:
            duration: Duration in seconds for the movement (default: 2.0)
            easing: Easing profile to apply ("ease_in_out_quad" or "linear")

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info(
            f"Smoothly returning to home position over {duration:.1f} seconds..."
        )

        if self.config.control_mode == ARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: use EEF trajectory (already in position control)
            self.move_eef_trajectory(
                target_eef_poses=self._home_position.copy(),
                durations=duration,
                easing=easing,
            )
            logger.info("✓ Successfully returned to home position (cartesian mode)")
        else:
            # Joint mode: need to switch modes
            # First, set current position as target to avoid large position error
            state = self.arm.get_joint_state()

            # Set current position as command to avoid SDK protection
            current_cmd = arx5.JointState(self.robot_config.joint_dof)
            current_cmd.pos()[:] = state.pos()
            current_cmd.gripper_pos = state.gripper_pos

            self.arm.set_joint_cmd(current_cmd)

            # Now safe to switch to normal position control
            self.set_to_normal_position_control()

            # Execute smooth trajectory to home position
            self.move_joint_trajectory(
                target_joint_poses=self._home_position.copy(),
                durations=duration,
                easing=easing,
            )

            # Switch back to gravity compensation mode (only for joint mode)
            self.set_to_gravity_compensation_mode()
            logger.info(
                "✓ Successfully returned to home position and switched to gravity compensation mode"
            )
