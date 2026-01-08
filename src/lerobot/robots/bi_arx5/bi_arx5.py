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

import math
import os
import sys
import time
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config, BiARX5ControlMode
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

# Add ARX5 SDK path to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
arx5_sdk_path = os.path.join(current_dir, "ARX5_SDK", "python")
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


class BiARX5(Robot):
    """
    [Bimanual ARX5 Arms]

    Dual ARX5 Arms Robot with support for Joint and Cartesian control modes.
    """

    config_class = BiARX5Config
    name = "bi_arx5"

    def __init__(self, config: BiARX5Config):
        super().__init__(config)
        self.config = config

        # Logger
        self.logger = get_logger("BiARX5")

        # Init left and right arm when connect
        self.left_arm = None
        self.right_arm = None
        self._is_connected = False

        # Control mode state variables
        self._is_joint_control_mode = False
        self._is_cartesian_control_mode = False
        self._is_gravity_compensation_mode = True

        # Use configurable preview time for inference mode
        # For CARTESIAN_CONTROL, we don't override SDK default (0.1s)
        if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            # Let SDK use its default preview_time (0.1s for cartesian_controller)
            self.default_preview_time = None
            self.logger.info("Cartesian control mode: using SDK default preview_time (0.1s)")
        elif self.config.inference_mode:
            self.default_preview_time = self.config.preview_time
            self.logger.info(
                f"Joint control mode (inference): using preview_time {self.default_preview_time}s"
            )
        else:
            self.default_preview_time = 0.0
            self.logger.info(f"Joint control mode (teleop): using preview_time {self.default_preview_time}s")

        # Pre-compute action keys for faster lookup (performance optimization)
        if config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            self._left_action_keys = [
                "left_x",
                "left_y",
                "left_z",
                "left_roll",
                "left_pitch",
                "left_yaw",
            ]
            self._right_action_keys = [
                "right_x",
                "right_y",
                "right_z",
                "right_roll",
                "right_pitch",
                "right_yaw",
            ]
            self._left_gripper_key = "left_gripper_pos"
            self._right_gripper_key = "right_gripper_pos"
        else:
            self._left_action_keys = [f"left_joint_{i + 1}.pos" for i in range(6)]
            self._right_action_keys = [f"right_joint_{i + 1}.pos" for i in range(6)]
            self._left_gripper_key = "left_gripper.pos"
            self._right_gripper_key = "right_gripper.pos"

        # Pre-allocate command buffers (initialized in connect based on control mode)
        self._left_cmd_buffer = None  # JointState buffer for joint control
        self._right_cmd_buffer = None  # JointState buffer for joint control
        self._left_eef_cmd_buffer = None  # EEFState buffer for cartesian control
        self._right_eef_cmd_buffer = None  # EEFState buffer for cartesian control

        if config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            self.config.start_position = [0.0, 0.967, 1.290, -0.970, 0.0, 0.0, 0.0]
        else:
            self.config.start_position = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]

        # Define home position (all joints at 0, gripper closed)
        self._home_position = self.config.home_position
        self._start_position = self.config.start_position

        # Robot configs
        self.robot_configs = {
            "left_config": arx5.RobotConfigFactory.get_instance().get_config(config.left_arm_model),
            "right_config": arx5.RobotConfigFactory.get_instance().get_config(config.right_arm_model),
        }

        # Create solver for FK/IK calculations (both arms use same model)
        urdf_path = os.path.join(current_dir, "ARX5_SDK", "models", f"{config.left_arm_model}.urdf")
        self._solver = arx5.Arx5Solver(
            urdf_path,
            self.robot_configs["left_config"].joint_dof,
            self.robot_configs["left_config"].joint_pos_min,
            self.robot_configs["left_config"].joint_pos_max,
        )

        # For Cartesian mode, convert joint positions to EEF positions
        if config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            home_joint_pos = np.array(self._home_position[:6], dtype=np.float64)
            start_joint_pos = np.array(self._start_position[:6], dtype=np.float64)

            # Convert home and start positions to EEF space using FK
            self._home_position_eef = np.concatenate(
                [
                    self._solver.forward_kinematics(home_joint_pos),
                    [self._home_position[6]],  # gripper
                ]
            )
            self._start_position_eef = np.concatenate(
                [
                    self._solver.forward_kinematics(start_joint_pos),
                    [self._start_position[6]],  # gripper
                ]
            )
            self.logger.info(f"EEF home position (FK): {self._home_position_eef}")
            self.logger.info(f"EEF start position (FK): {self._start_position_eef}")

        # Set gripper_open_readout for left and right arm
        self.robot_configs["left_config"].gripper_open_readout = config.gripper_open_readout[0]
        self.robot_configs["right_config"].gripper_open_readout = config.gripper_open_readout[1]
        self.logger.info(
            f"Set left gripper_open_readout to: {self.robot_configs['left_config'].gripper_open_readout}"
        )
        self.logger.info(
            f"Set right gripper_open_readout to: {self.robot_configs['right_config'].gripper_open_readout}"
        )

        # Controller config - select based on control mode
        if config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            controller_type = "cartesian_controller"
            # Cartesian controller requires background_send_recv = True
            use_background = True
        else:
            controller_type = "joint_controller"
            use_background = config.use_multithreading

        self.controller_configs = {
            "left_config": arx5.ControllerConfigFactory.get_instance().get_config(
                controller_type, self.robot_configs["left_config"].joint_dof
            ),
            "right_config": arx5.ControllerConfigFactory.get_instance().get_config(
                controller_type, self.robot_configs["right_config"].joint_dof
            ),
        }
        self.logger.info(f"Using {controller_type} for control mode: {config.control_mode.value}")

        # Set controller_dt and default_preview_time
        self.controller_configs["left_config"].controller_dt = config.controller_dt
        self.controller_configs["right_config"].controller_dt = config.controller_dt
        # Only override default_preview_time if not CARTESIAN_CONTROL (preserve SDK default 0.1s)
        if self.default_preview_time is not None:
            self.controller_configs["left_config"].default_preview_time = self.default_preview_time
            self.controller_configs["right_config"].default_preview_time = self.default_preview_time

        # Background send/recv setting
        self.controller_configs["left_config"].background_send_recv = use_background
        self.controller_configs["right_config"].background_send_recv = use_background

        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=3, suppress=True)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Return motor features based on control mode."""
        if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: EEF pose (x, y, z, roll, pitch, yaw) + gripper for each arm
            return {
                "left_x": float,
                "left_y": float,
                "left_z": float,
                "left_roll": float,
                "left_pitch": float,
                "left_yaw": float,
                "left_gripper_pos": float,
                "right_x": float,
                "right_y": float,
                "right_z": float,
                "right_roll": float,
                "right_pitch": float,
                "right_yaw": float,
                "right_gripper_pos": float,
            }
        else:
            # Joint mode (including teach mode): 6 joints + gripper per arm
            joint_names = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]
            return {f"left_{joint}.pos": float for joint in joint_names} | {
                f"right_{joint}.pos": float for joint in joint_names
            }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
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
            and self.left_arm is not None
            and self.right_arm is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def is_gravity_compensation_mode(self) -> bool:
        """Check if robot is currently in gravity compensation mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_gravity_compensation_mode

    def is_joint_control_mode(self) -> bool:
        """Check if robot is currently in joint control mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_joint_control_mode

    def is_cartesian_control_mode(self) -> bool:
        """Check if robot is currently in cartesian control mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._is_cartesian_control_mode

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            self.logger.info(f"Creating left arm controller (mode: {self.config.control_mode.value})...")
            if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
                self.left_arm = arx5.Arx5CartesianController(
                    self.robot_configs["left_config"],
                    self.controller_configs["left_config"],
                    self.config.left_arm_port,
                )
            else:
                self.left_arm = arx5.Arx5JointController(
                    self.robot_configs["left_config"],
                    self.controller_configs["left_config"],
                    self.config.left_arm_port,
                )
            time.sleep(0.5)
            self.logger.info(f"✅ Left arm controller created successfully ({type(self.left_arm).__name__})")
            self.logger.info(
                f"Left arm preview_time: {self.controller_configs['left_config'].default_preview_time}"
            )

            self.logger.info(f"Creating right arm controller (mode: {self.config.control_mode.value})...")
            if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
                self.right_arm = arx5.Arx5CartesianController(
                    self.robot_configs["right_config"],
                    self.controller_configs["right_config"],
                    self.config.right_arm_port,
                )
            else:
                self.right_arm = arx5.Arx5JointController(
                    self.robot_configs["right_config"],
                    self.controller_configs["right_config"],
                    self.config.right_arm_port,
                )
            time.sleep(0.5)
            self.logger.info(
                f"✅ Right arm controller created successfully ({type(self.right_arm).__name__})"
            )
            self.logger.info(
                f"Right arm preview_time: {self.controller_configs['right_config'].default_preview_time}"
            )

            # Verify SDK is using the correct gripper_open_readout
            left_robot_config = self.left_arm.get_robot_config()
            right_robot_config = self.right_arm.get_robot_config()
            self.logger.info(f"SDK left gripper_open_readout: {left_robot_config.gripper_open_readout}")
            self.logger.info(f"SDK right gripper_open_readout: {right_robot_config.gripper_open_readout}")
        except Exception as e:
            self.logger.error(f"Failed to create robot controller: {e}")
            self.left_arm = None
            self.right_arm = None
            raise e

        self._is_connected = True
        # Set log level
        self.set_log_level(self.config.log_level)

        # Reset to home using SDK method
        self.reset_to_home()

        # Set gravity compensation gain
        self.set_to_gravity_compensation_mode()

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        # Initialize command buffers for optimized send_action
        if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            self._left_eef_cmd_buffer = arx5.EEFState()
            self._right_eef_cmd_buffer = arx5.EEFState()
            self._left_cmd_buffer = None
            self._right_cmd_buffer = None
        else:
            self._left_cmd_buffer = arx5.JointState(self.robot_configs["left_config"].joint_dof)
            self._right_cmd_buffer = arx5.JointState(self.robot_configs["right_config"].joint_dof)
            self._left_eef_cmd_buffer = None
            self._right_eef_cmd_buffer = None

        self.logger.info("Dual-ARX5 Robot connected.")
        if go_to_start:
            self.smooth_go_start(duration=2.0)
            self.logger.info("✅ Robot go to start position, both arms are now in gravity compensation mode")
        else:
            self.logger.info("Robot go to home position, both arms are now in gravity compensation mode")

        # Log current gain
        gain = self.left_arm.get_gain()
        self.logger.info(
            f"Current left arm gain: {gain.kp()}, {gain.kd()}, {gain.gripper_kp}, {gain.gripper_kd}"
        )
        gain = self.right_arm.get_gain()
        self.logger.info(
            f"Current right arm gain: {gain.kp()}, {gain.kd()}, {gain.gripper_kp}, {gain.gripper_kd}"
        )

        if self.config.inference_mode:
            if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
                self.set_to_normal_cartesian_control()
                self.logger.info("✅ Robot is now in cartesian control mode for inference")
            elif self.config.control_mode == BiARX5ControlMode.JOINT_CONTROL:
                self.set_to_normal_position_control()
                self.logger.info("✅ Robot is now in joint position control mode for inference")
            else:
                self.logger.error(f"Invalid inference time control mode: {self.config.control_mode.value}")
                raise ValueError(f"Invalid inference time control mode: {self.config.control_mode.value}")
            self.logger.info(
                f"✅ Robot is now connected and ready for inference in {self.config.control_mode.value} mode."
            )
        else:  # Teleoperation mode
            if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
                self.set_to_normal_cartesian_control()
                self.logger.info("✅ Robot is now in cartesian control mode for teleoperation")
            elif self.config.control_mode == BiARX5ControlMode.JOINT_CONTROL:
                self.set_to_normal_position_control()
                self.logger.info("✅ Robot is now in position control mode for teleoperation")
            elif self.config.control_mode == BiARX5ControlMode.TEACH_MODE:
                self.set_to_gravity_compensation_mode()
                self.logger.info("✅ Robot is now in gravity compensation mode for teleoperation")
            else:
                self.logger.error(f"Invalid teleoperation control mode: {self.config.control_mode.value}")
                raise ValueError(f"Invalid teleoperation control mode: {self.config.control_mode.value}")
            self.logger.info(
                f"✅ Robot is now connected and ready for teleoperation in {self.config.control_mode.value} mode."
            )

    @property
    def is_calibrated(self) -> bool:
        """ARX5 does not need to calibrate in runtime"""
        self.logger.info("ARX5 does not need to calibrate in runtime, skip...")
        return True

    def calibrate(self) -> None:
        """ARX5 does not need to calibrate in runtime"""
        self.logger.info("ARX5 does not need to calibrate in runtime, skip...")
        return

    def configure(self) -> None:
        """Configure the robot"""
        self.logger.info("ARX5 does not need to configure in runtime, skip...")
        pass

    def setup_motors(self) -> None:
        """ARX5 motors are pre-configured, no runtime setup needed"""
        self.logger.info(f"{self} ARX5 motors are pre-configured, no runtime setup needed")
        self.logger.info("Motor IDs are defined in the robot configuration:")
        self.logger.info("  - Joint motors: [1, 2, 4, 5, 6, 7]")
        self.logger.info("  - Gripper motor: 8")
        self.logger.info("Make sure your hardware matches these ID configurations")
        return

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: get EEF states
            left_eef_state = self.left_arm.get_eef_state()
            left_pose_6d = left_eef_state.pose_6d().copy()
            for i, key in enumerate(["left_x", "left_y", "left_z", "left_roll", "left_pitch", "left_yaw"]):
                obs_dict[key] = float(left_pose_6d[i])
            obs_dict["left_gripper_pos"] = float(left_eef_state.gripper_pos)

            right_eef_state = self.right_arm.get_eef_state()
            right_pose_6d = right_eef_state.pose_6d().copy()
            for i, key in enumerate(
                [
                    "right_x",
                    "right_y",
                    "right_z",
                    "right_roll",
                    "right_pitch",
                    "right_yaw",
                ]
            ):
                obs_dict[key] = float(right_pose_6d[i])
            obs_dict["right_gripper_pos"] = float(right_eef_state.gripper_pos)
        else:
            # Joint mode (including teach mode): get joint states
            left_joint_state = self.left_arm.get_joint_state()
            left_pos = left_joint_state.pos().copy()
            for i in range(6):
                obs_dict[f"left_joint_{i + 1}.pos"] = float(left_pos[i])
            obs_dict["left_gripper.pos"] = float(left_joint_state.gripper_pos)

            right_joint_state = self.right_arm.get_joint_state()
            right_pos = right_joint_state.pos().copy()
            for i in range(6):
                obs_dict[f"right_joint_{i + 1}.pos"] = float(right_pos[i])
            obs_dict["right_gripper.pos"] = float(right_joint_state.gripper_pos)

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

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: use EEF commands
            # Note: timestamp is not set here, SDK will use controller_config.default_preview_time
            left_cmd = self._left_eef_cmd_buffer
            left_pose_6d = left_cmd.pose_6d()
            for i, key in enumerate(self._left_action_keys):
                left_pose_6d[i] = action.get(key, left_pose_6d[i])
            left_cmd.gripper_pos = action.get(self._left_gripper_key, left_cmd.gripper_pos)
            # Debug: Print commands before sending
            # print(
            #     f"Left arm command - pose_6d: {left_cmd.pose_6d()}, gripper: {left_cmd.gripper_pos}"
            # )
            self.left_arm.set_eef_cmd(left_cmd)

            right_cmd = self._right_eef_cmd_buffer
            right_pose_6d = right_cmd.pose_6d()
            for i, key in enumerate(self._right_action_keys):
                right_pose_6d[i] = action.get(key, right_pose_6d[i])
            right_cmd.gripper_pos = action.get(self._right_gripper_key, right_cmd.gripper_pos)
            # Debug: Print commands before sending
            # print(
            #     f"Right arm command - pose_6d: {right_cmd.pose_6d()}, gripper: {right_cmd.gripper_pos}"
            # )
            self.right_arm.set_eef_cmd(right_cmd)
        else:
            # Joint mode (including teach mode): use joint commands
            left_cmd = self._left_cmd_buffer
            left_pos = left_cmd.pos()
            for i, key in enumerate(self._left_action_keys):
                left_pos[i] = action.get(key, left_pos[i])
            left_cmd.gripper_pos = action.get(self._left_gripper_key, left_cmd.gripper_pos)
            self.left_arm.set_joint_cmd(left_cmd)

            right_cmd = self._right_cmd_buffer
            right_pos = right_cmd.pos()
            for i, key in enumerate(self._right_action_keys):
                right_pos[i] = action.get(key, right_pos[i])
            right_cmd.gripper_pos = action.get(self._right_gripper_key, right_cmd.gripper_pos)
            self.right_arm.set_joint_cmd(right_cmd)

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
        target_joint_poses: (dict[str, Sequence[float]] | Sequence[dict[str, Sequence[float]]]),
        durations: float | Sequence[float],
        *,
        easing: str = "ease_in_out_quad",
        steps_per_segment: int | None = None,
    ) -> None:
        """Move both arms smoothly towards the provided joint targets.

        Uses send_action to send interpolated commands step by step, ensuring
        both arms move synchronously in the same loop iteration.

        Args:
            target_joint_poses: A dictionary with "left" and "right" keys (each a
                sequence of 6 or 7 joint values including the gripper) or a
                sequence of such dictionaries to execute multiple segments.
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

        if isinstance(target_joint_poses, dict):
            trajectory = [target_joint_poses]
        else:
            trajectory = list(target_joint_poses)

        if isinstance(durations, (int, float)):
            segment_durations = [float(durations)]
        else:
            segment_durations = [float(d) for d in durations]

        if len(trajectory) != len(segment_durations):
            raise ValueError("target_joint_poses and durations must have the same length")

        # Determine controller timestep (fallback to 10 ms if unavailable)
        controller_dt = getattr(self.config, "interpolation_controller_dt", 0.01)

        # Fetch the current joint positions as starting state
        def _get_current_state() -> tuple[np.ndarray, np.ndarray]:
            left_state = self.left_arm.get_joint_state()
            right_state = self.right_arm.get_joint_state()
            left = np.concatenate([left_state.pos().copy(), [left_state.gripper_pos]])
            right = np.concatenate([right_state.pos().copy(), [right_state.gripper_pos]])
            return left, right

        current_left, current_right = _get_current_state()

        def _parse_target(
            segment: dict[str, Sequence[float]],
            default_left: np.ndarray,
            default_right: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not {"left", "right"}.issubset(segment):
                raise ValueError("Each segment must contain both 'left' and 'right' targets")

            def _to_array(values: Sequence[float], default: np.ndarray) -> np.ndarray:
                arr = np.asarray(values, dtype=np.float64)
                if arr.shape[0] not in (6, 7):
                    raise ValueError("Each arm target must provide 6 joint values (+ optional gripper)")
                if arr.shape[0] == 6:
                    arr = np.concatenate([arr, [default[-1]]])
                return arr

            left_target = _to_array(segment["left"], default_left)
            right_target = _to_array(segment["right"], default_right)
            return left_target, right_target

        def _apply_easing(alpha: float) -> float:
            alpha = np.clip(alpha, 0.0, 1.0)
            if easing == "ease_in_out_quad":
                return self._ease_in_out_quad(alpha)
            if easing == "linear":
                return alpha
            raise ValueError(f"Unsupported easing profile: {easing}")

        # Pre-store action keys for joint mode (avoid string formatting in loop)
        left_joint_keys = [f"left_joint_{i + 1}.pos" for i in range(6)]
        right_joint_keys = [f"right_joint_{i + 1}.pos" for i in range(6)]

        try:
            for segment, duration in zip(trajectory, segment_durations, strict=True):
                target_left, target_right = _parse_target(segment, current_left, current_right)

                if duration <= 0:
                    action = dict(zip(left_joint_keys, target_left[:6].tolist(), strict=True))
                    action.update(dict(zip(right_joint_keys, target_right[:6].tolist(), strict=True)))
                    action["left_gripper.pos"] = float(target_left[6])
                    action["right_gripper.pos"] = float(target_right[6])
                    self.send_action(action)
                    current_left, current_right = target_left, target_right
                    continue

                steps = (
                    steps_per_segment
                    if steps_per_segment is not None
                    else max(1, int(math.ceil(duration / controller_dt)))
                )

                for step in range(1, steps + 1):
                    progress = step / steps
                    ratio = _apply_easing(progress)
                    interp_left = current_left + (target_left - current_left) * ratio
                    interp_right = current_right + (target_right - current_right) * ratio

                    action = dict(zip(left_joint_keys, interp_left[:6].tolist(), strict=True))
                    action.update(dict(zip(right_joint_keys, interp_right[:6].tolist(), strict=True)))
                    action["left_gripper.pos"] = float(interp_left[6])
                    action["right_gripper.pos"] = float(interp_right[6])

                    self.send_action(action)
                    time.sleep(duration / steps if steps_per_segment else controller_dt)

                current_left, current_right = target_left, target_right
        except KeyboardInterrupt:
            self.logger.warn("Joint trajectory interrupted by user. Holding current pose.")

    def move_eef_trajectory(
        self,
        target_eef_poses: (dict[str, Sequence[float]] | Sequence[dict[str, Sequence[float]]]),
        durations: float | Sequence[float],
        *,
        easing: str = "linear",
        steps_per_segment: int | None = None,
    ) -> None:
        """Move both arms smoothly towards the provided EEF targets (Cartesian mode).

        Uses send_action to send interpolated commands step by step, ensuring
        both arms move synchronously in the same loop iteration.

        Args:
            target_eef_poses: A dictionary with "left" and "right" keys (each a
                sequence of 6 or 7 values for EEF pose + optional gripper) or a
                sequence of such dictionaries to execute multiple segments.
            durations: Duration in seconds for the corresponding target poses.
            easing: Easing profile to apply ("ease_in_out_quad" or "linear").
            steps_per_segment: Optional fixed number of interpolation steps per
                segment. When omitted the controller's ``controller_dt`` is used
                to compute the number of steps from the duration.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
            ValueError: If inputs are malformed or not in Cartesian mode.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode != BiARX5ControlMode.CARTESIAN_CONTROL:
            raise ValueError("move_eef_trajectory requires CARTESIAN_CONTROL mode")

        trajectory = [target_eef_poses] if isinstance(target_eef_poses, dict) else list(target_eef_poses)

        if isinstance(durations, (int, float)):
            segment_durations = [float(durations)]
        else:
            segment_durations = [float(d) for d in durations]

        if len(trajectory) != len(segment_durations):
            raise ValueError("target_eef_poses and durations must have the same length")

        # Determine controller timestep (fallback to 10 ms if unavailable)
        controller_dt = getattr(self.config, "interpolation_controller_dt", 0.01)

        # Fetch the current EEF positions as starting state
        def _get_current_state() -> tuple[np.ndarray, np.ndarray]:
            left_state = self.left_arm.get_eef_state()
            right_state = self.right_arm.get_eef_state()
            left = np.concatenate([left_state.pose_6d().copy(), [left_state.gripper_pos]])
            right = np.concatenate([right_state.pose_6d().copy(), [right_state.gripper_pos]])
            return left, right

        current_left, current_right = _get_current_state()

        def _parse_target(
            segment: dict[str, Sequence[float]],
            default_left: np.ndarray,
            default_right: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not {"left", "right"}.issubset(segment):
                raise ValueError("Each segment must contain both 'left' and 'right' targets")

            def _to_array(values: Sequence[float], default: np.ndarray) -> np.ndarray:
                arr = np.asarray(values, dtype=np.float64)
                if arr.shape[0] not in (6, 7):
                    raise ValueError("Each arm target must provide 6 EEF values (+ optional gripper)")
                if arr.shape[0] == 6:
                    arr = np.concatenate([arr, [default[-1]]])
                return arr

            left_target = _to_array(segment["left"], default_left)
            right_target = _to_array(segment["right"], default_right)
            return left_target, right_target

        def _apply_easing(alpha: float) -> float:
            alpha = np.clip(alpha, 0.0, 1.0)
            if easing == "ease_in_out_quad":
                return self._ease_in_out_quad(alpha)
            if easing == "linear":
                return alpha
            raise ValueError(f"Unsupported easing profile: {easing}")

        try:
            for segment, duration in zip(trajectory, segment_durations, strict=True):
                target_left, target_right = _parse_target(segment, current_left, current_right)

                if duration <= 0:
                    action = dict(zip(self._left_action_keys, target_left[:6].tolist(), strict=True))
                    action[self._left_gripper_key] = float(target_left[6])
                    action.update(dict(zip(self._right_action_keys, target_right[:6].tolist(), strict=True)))
                    action[self._right_gripper_key] = float(target_right[6])
                    self.send_action(action)
                    current_left, current_right = target_left, target_right
                    continue

                steps = (
                    steps_per_segment
                    if steps_per_segment is not None
                    else max(1, int(math.ceil(duration / controller_dt)))
                )

                for step in range(1, steps + 1):
                    progress = step / steps
                    ratio = _apply_easing(progress)
                    interp_left = current_left + (target_left - current_left) * ratio
                    interp_right = current_right + (target_right - current_right) * ratio

                    action = dict(zip(self._left_action_keys, interp_left[:6].tolist(), strict=True))
                    action[self._left_gripper_key] = float(interp_left[6])
                    action.update(dict(zip(self._right_action_keys, interp_right[:6].tolist(), strict=True)))
                    action[self._right_gripper_key] = float(interp_right[6])

                    self.send_action(action)
                    time.sleep(duration / steps if steps_per_segment else controller_dt)

                current_left, current_right = target_left, target_right
        except KeyboardInterrupt:
            self.logger.warn("EEF trajectory interrupted by user. Holding current pose.")

    def disconnect(self):
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            self.logger.info("Disconnecting arms...")
            self.smooth_go_home(easing="ease_in_out_quad")  # Auto-calculated duration
            # Set both arms to damping
            self.left_arm.set_to_damping()
            self.right_arm.set_to_damping()
            self.logger.info("✅ Both arms disconnected successfully")
        except KeyboardInterrupt:
            self.logger.warn("Disconnect interrupted. Forcing damping mode on both arms...")
            self.left_arm.set_to_damping()
            self.right_arm.set_to_damping()
            self.logger.info("✅ Both arms set to damping mode for safety")
        except Exception as e:
            self.logger.warn(f"Failed to disconnect arms: {e}")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Destroy arm objects
        self.left_arm = None
        self.right_arm = None

        self._is_connected = False
        self.logger.info(f"{self} disconnected.")

    def set_log_level(self, level: str):
        """Set robot log level

        Args:
            level: Log level string, supports: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF
        """
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
            raise ValueError(f"Invalid log level: {level}. Supported levels: {list(log_level_map.keys())}")

        log_level = log_level_map[level.upper()]

        if self.left_arm is not None:
            self.left_arm.set_log_level(log_level)
        if self.right_arm is not None:
            self.right_arm.set_log_level(log_level)

    def reset_to_home(self):
        """Reset both arms to home position"""
        if self.left_arm is None or self.right_arm is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_arm.reset_to_home()
        self.right_arm.reset_to_home()
        self.logger.info("Both arms reset to home position.")

    def set_to_gravity_compensation_mode(self):
        """Switch from normal position control to gravity compensation mode.

        Uses SDK's set_to_gravity_compensation() which:
        1. Sets kp=0, kd=default (damping only, no position control)
        2. Resets interpolator to current position (important for Cartesian mode)
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._is_gravity_compensation_mode:
            self.logger.info("Both arms are already in gravity compensation mode")
            return

        self.logger.info("Switching to gravity compensation mode...")

        # Use SDK's set_to_gravity_compensation() which properly resets the interpolator
        if self._is_joint_control_mode:
            self.logger.info("Switching to gravity compensation mode from joint control mode...")
        elif self._is_cartesian_control_mode:
            self.logger.info("Switching to gravity compensation mode from cartesian control mode...")

        self.left_arm.set_to_gravity_compensation()
        self.right_arm.set_to_gravity_compensation()

        # Update control mode state
        self._is_gravity_compensation_mode = True
        self._is_joint_control_mode = False
        self._is_cartesian_control_mode = False

        self.logger.info("✅ Both arms are now in gravity compensation mode")

    def set_to_normal_position_control(self):
        """Switch from gravity compensation to normal position control mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Switching to normal position control mode...")

        is_joint_mode = (
            self.config.control_mode == BiARX5ControlMode.JOINT_CONTROL
            or self.config.control_mode == BiARX5ControlMode.TEACH_MODE
        )

        if self._is_gravity_compensation_mode and is_joint_mode:
            # Reset to default gain
            left_cfg = self.controller_configs["left_config"]
            default_gain = self.left_arm.get_gain()
            default_gain.kp()[:] = left_cfg.default_kp * 0.5
            default_gain.kd()[:] = left_cfg.default_kd * 1.5
            default_gain.gripper_kp = left_cfg.default_gripper_kp
            default_gain.gripper_kd = left_cfg.default_gripper_kd

            self.left_arm.set_gain(default_gain)
            self.right_arm.set_gain(default_gain)

            # Update control mode state
            self._is_joint_control_mode = True
            self._is_cartesian_control_mode = False
            self._is_gravity_compensation_mode = False

            self.logger.info("✅ Both arms are now in normal position control mode")
        elif not self._is_gravity_compensation_mode and is_joint_mode:
            self.logger.info("Both arms are already in normal position control mode")
            return
        else:
            self.logger.warn(f"Can't switch to position control from mode: {self.config.control_mode}")
            return

    def set_to_normal_cartesian_control(self):
        """Switch from gravity compensation to normal cartesian control mode"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Switching to normal cartesian control mode...")

        is_cartesian_mode = self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL

        if self._is_gravity_compensation_mode and is_cartesian_mode:
            # Reset to default gain
            left_cfg = self.controller_configs["left_config"]
            default_gain = self.left_arm.get_gain()
            default_gain.kp()[:] = left_cfg.default_kp
            default_gain.kd()[:] = left_cfg.default_kd
            default_gain.gripper_kp = left_cfg.default_gripper_kp
            default_gain.gripper_kd = left_cfg.default_gripper_kd

            self.left_arm.set_gain(default_gain)
            self.right_arm.set_gain(default_gain)

            # Update control mode state
            self._is_joint_control_mode = False
            self._is_cartesian_control_mode = True
            self._is_gravity_compensation_mode = False

            self.logger.info(
                "✅ Both arms switched from gravity compensation to normal cartesian control mode"
            )
        elif not self._is_gravity_compensation_mode and is_cartesian_mode:
            self.logger.info("Both arms are already in normal cartesian control mode")
            return
        else:
            self.logger.warn(f"Can't switch to cartesian control from mode: {self.config.control_mode}")
            return

    def _calculate_motion_duration(
        self,
        target_left: np.ndarray,
        target_right: np.ndarray,
        min_duration: float = 1.0,
        speed_factor: float = 2.0,
    ) -> float:
        """
        Calculate motion duration based on maximum joint/EEF position error.

        This follows the SDK's reset_to_home logic:
        duration = max(max_pos_error, min_duration)

        Args:
            target_left: Target position for left arm (7 elements: 6 joints/pose + gripper)
            target_right: Target position for right arm (7 elements: 6 joints/pose + gripper)
            min_duration: Minimum duration in seconds (default: 1.0)
            speed_factor: Multiplier for speed adjustment (default: 2.0)

        Returns:
            Calculated duration in seconds
        """
        # Always use Joint space for duration calculation (consistent units in radians)
        # This follows SDK's reset_to_home logic which uses joint position error
        left_state = self.left_arm.get_joint_state()
        right_state = self.right_arm.get_joint_state()
        current_left = np.concatenate([left_state.pos(), [left_state.gripper_pos]])
        current_right = np.concatenate([right_state.pos(), [right_state.gripper_pos]])

        # Calculate maximum position error across both arms (excluding gripper)
        left_error = np.abs(current_left[:6] - target_left[:6]).max()
        right_error = np.abs(current_right[:6] - target_right[:6]).max()
        max_error = max(left_error, right_error)

        # Duration = max(max_error, min_duration) * speed_factor
        duration = max(max_error, min_duration) * speed_factor
        self.logger.info(f"Calculated motion duration: {duration:.1f} seconds")
        return duration

    def smooth_go_start(self, duration: float | None = None, easing: str = "ease_in_out_quad") -> None:
        """
        Smoothly move both arms to the start position using trajectory interpolation.

        For Joint mode:
        1. Switches to normal position control mode
        2. Moves both arms to start position over the specified duration
        3. Switches back to gravity compensation mode

        For Cartesian mode:
        1. Moves both arms to start EEF position over the specified duration
        (No mode switching needed - already in position control)

        Args:
            duration: Duration in seconds for the movement. If None, automatically
                calculated based on distance to target (like SDK's reset_to_home).
            easing: Easing profile to apply ("ease_in_out_quad" or "linear")

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Calculate duration if not provided
        # Always use Joint space for duration calculation (consistent units in radians)
        if duration is None:
            target = np.array(self._start_position)  # Joint space target
            duration = self._calculate_motion_duration(target, target)

        self.logger.info(f"Smoothly going to start position over {duration:.1f} seconds...")

        if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: use EEF trajectory
            self.logger.info("Cartesian mode: use EEF trajectory interpolation.")

            # Set current position as command first (required for interpolator)
            left_state = self.left_arm.get_eef_state()
            right_state = self.right_arm.get_eef_state()

            left_cmd = arx5.EEFState(left_state.pose_6d(), left_state.gripper_pos)
            left_cmd.timestamp = self.left_arm.get_timestamp() + 0.01
            self.left_arm.set_eef_cmd(left_cmd)

            right_cmd = arx5.EEFState(right_state.pose_6d(), right_state.gripper_pos)
            right_cmd.timestamp = self.right_arm.get_timestamp() + 0.01
            self.right_arm.set_eef_cmd(right_cmd)

            # Switch to normal cartesian control
            self.set_to_normal_cartesian_control()

            # Prepare start poses for both arms
            start_poses = {
                "left": self._start_position_eef.copy(),
                "right": self._start_position_eef.copy(),
            }

            self.move_eef_trajectory(
                target_eef_poses=start_poses,
                durations=duration,
                easing=easing,
            )
            self.logger.info(
                f"✅ Successfully going to start position in {self.config.control_mode.value} mode"
            )
        else:
            # Joint mode: use joint trajectory interpolation
            self.logger.info("Joint mode: use joint trajectory interpolation.")

            # First, set current position as target to avoid large position error
            left_state = self.left_arm.get_joint_state()
            right_state = self.right_arm.get_joint_state()

            current_left_cmd = arx5.JointState(self.robot_configs["left_config"].joint_dof)
            current_left_cmd.pos()[:] = left_state.pos()
            current_left_cmd.gripper_pos = left_state.gripper_pos

            current_right_cmd = arx5.JointState(self.robot_configs["right_config"].joint_dof)
            current_right_cmd.pos()[:] = right_state.pos()
            current_right_cmd.gripper_pos = right_state.gripper_pos

            self.left_arm.set_joint_cmd(current_left_cmd)
            self.right_arm.set_joint_cmd(current_right_cmd)

            # Now safe to switch to normal position control
            self.set_to_normal_position_control()

            # Prepare start poses for both arms
            start_poses = {
                "left": self._start_position.copy(),
                "right": self._start_position.copy(),
            }

            # Execute smooth trajectory to start position
            self.move_joint_trajectory(target_joint_poses=start_poses, durations=duration, easing=easing)
            self.logger.info(
                f"✅ Successfully going to start position in {self.config.control_mode.value} mode"
            )

    def smooth_go_home(self, duration: float | None = None, easing: str = "ease_in_out_quad") -> None:
        """
        Smoothly move both arms to the home position using trajectory interpolation.

        For Joint mode:
        1. Switches to normal position control mode
        2. Moves both arms to home position over the specified duration
        3. Switches back to gravity compensation mode

        For Cartesian mode:
        1. Moves both arms to home EEF position over the specified duration

        Args:
            duration: Duration in seconds for the movement. If None, automatically
                calculated based on distance to target (like SDK's reset_to_home).
            easing: Easing profile to apply ("ease_in_out_quad" or "linear")

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Calculate duration if not provided
        # Always use Joint space for duration calculation (consistent units in radians)
        if duration is None:
            target = np.array(self._home_position)  # Joint space target
            duration = self._calculate_motion_duration(target, target)

        self.logger.info(f"Smoothly returning to home position over {duration:.1f} seconds...")

        if self.config.control_mode == BiARX5ControlMode.CARTESIAN_CONTROL:
            # Cartesian mode: use EEF trajectory
            self.logger.info("Cartesian mode: use EEF trajectory interpolation.")

            # Set current position as command first (required for interpolator)
            left_state = self.left_arm.get_eef_state()
            right_state = self.right_arm.get_eef_state()

            left_cmd = arx5.EEFState(left_state.pose_6d(), left_state.gripper_pos)
            left_cmd.timestamp = self.left_arm.get_timestamp() + 0.01
            self.left_arm.set_eef_cmd(left_cmd)

            right_cmd = arx5.EEFState(right_state.pose_6d(), right_state.gripper_pos)
            right_cmd.timestamp = self.right_arm.get_timestamp() + 0.01
            self.right_arm.set_eef_cmd(right_cmd)

            # Switch to normal cartesian control (if in gravity compensation mode)
            self.set_to_normal_cartesian_control()

            # Prepare home poses for both arms
            home_poses = {
                "left": self._home_position_eef.copy(),
                "right": self._home_position_eef.copy(),
            }

            self.move_eef_trajectory(
                target_eef_poses=home_poses,
                durations=duration,
                easing=easing,
            )
            self.logger.info(
                f"✅ Successfully returned to home position in {self.config.control_mode.value} mode"
            )
        else:
            # Joint mode: use joint trajectory
            # First, set current position as target to avoid large position error
            left_state = self.left_arm.get_joint_state()
            right_state = self.right_arm.get_joint_state()

            current_left_cmd = arx5.JointState(self.robot_configs["left_config"].joint_dof)
            current_left_cmd.pos()[:] = left_state.pos()
            current_left_cmd.gripper_pos = left_state.gripper_pos

            current_right_cmd = arx5.JointState(self.robot_configs["right_config"].joint_dof)
            current_right_cmd.pos()[:] = right_state.pos()
            current_right_cmd.gripper_pos = right_state.gripper_pos

            self.left_arm.set_joint_cmd(current_left_cmd)
            self.right_arm.set_joint_cmd(current_right_cmd)

            # Now safe to switch to normal position control
            self.set_to_normal_position_control()

            # Prepare home poses for both arms
            home_poses = {
                "left": self._home_position.copy(),
                "right": self._home_position.copy(),
            }

            # Execute smooth trajectory to home position
            self.move_joint_trajectory(target_joint_poses=home_poses, durations=duration, easing=easing)

            # Switch back to gravity compensation mode
            self.set_to_gravity_compensation_mode()

            self.logger.info(
                "✅ Successfully returned to home position and switched to gravity compensation mode"
            )
