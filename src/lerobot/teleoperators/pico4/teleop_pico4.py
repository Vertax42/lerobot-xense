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

"""
Pico4 VR teleoperator for end-effector control.

This teleoperator provides 6-DoF absolute pose control using VR controllers,
similar to Spacemouse. It outputs accumulated target_pose_6d that can be
directly sent to a Cartesian controller.

The output format matches ARX5 SDK's spacemouse_teleop.py example:
- target_pose_6d: [x, y, z, roll, pitch, yaw] - absolute EEF pose
- gripper_pos: absolute gripper position in meters
"""

import logging
import time
from queue import Queue
from typing import Any

import numpy as np
import spdlog

from lerobot.teleoperators.pico4.config_pico4 import Pico4Config
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import euler_to_quaternion, normalize_quaternion

logger = logging.getLogger(__name__)


class Pico4(Teleoperator):
    """
    Pico4 VR teleoperator for end-effector control.

    This teleoperator reads pose data from Pico4 VR controllers and maintains
    an accumulated target_pose_6d suitable for Cartesian control of robotic arms.

    The VR controller provides:
    - Absolute pose: (x, y, z, qx, qy, qz, qw) - position and quaternion
    - Trigger values (0-1) for both controllers
    - Grip values (0-1) for both controllers
    - Buttons: A, B, X, Y, menu, axis click

    Output action format (matches ARX5 SDK spacemouse_teleop.py):
    - x, y, z, roll, pitch, yaw: absolute EEF target pose (accumulated)
    - gripper_pos: absolute gripper position in meters

    Usage:
    1. Call reset_to_pose() to initialize with robot's current EEF pose
    2. Call get_action() to get updated target_pose_6d based on VR controller input
    """

    config_class = Pico4Config
    name = "pico4"

    def __init__(self, config: Pico4Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._xrt = None
        self.logger = spdlog.ConsoleLogger("Pico4Teleop")

        # Target pose tracking (position in xyz, orientation in quaternion)
        self._target_pos: np.ndarray = np.zeros(3, dtype=np.float32)  # [x, y, z]
        self._target_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # [qx, qy, qz, qw]
        self._target_gripper_pos: float = 0.0
        self._start_pos: np.ndarray = np.zeros(3, dtype=np.float32)  # [x, y, z]
        self._start_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # [qx, qy, qz, qw]
        self._start_gripper_pos: float = 0.0

        # Reference pose for relative control (calibration pose)
        self._ref_controller_pose: np.ndarray | None = None  # [x, y, z, qx, qy, qz, qw]
        self._ref_controller_quat: np.ndarray | None = None  # [qx, qy, qz, qw] for quaternion math

        # Smoothing filter queues (moving average) for absolute position and quaternion
        self._pos_queue: Queue = Queue(self.config.filter_window_size)
        self._quat_queue: Queue = Queue(self.config.filter_window_size)

        # State tracking
        self._enabled: bool = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Pico4 doesn't require calibration."""
        return True

    @property
    def action_features(self) -> dict:
        """
        Return action features matching ARX5 SDK's target_pose_6d format.

        Returns a dictionary with dtype, shape, and names for the action space:
        - x, y, z: absolute EEF position (meters)
        - roll, pitch, yaw: absolute EEF orientation (radians)
        - gripper_pos: absolute gripper position (meters)
        """
        return {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "x": 0,
                "y": 1,
                "z": 2,
                "roll": 3,
                "pitch": 4,
                "yaw": 5,
                "gripper_pos": 6,
            },
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        """Pico4 doesn't support feedback."""
        return {}

    def connect(self, calibrate: bool = True, current_tcp_pose_euler: np.ndarray = np.zeros(7, dtype=np.float32)) -> None:
        """Connect to the Pico4 VR headset via XenseVR SDK."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.logger.info("Connecting to Pico4 VR headset...")
        try:
            import xensevr_pc_service_sdk as xrt
        except ImportError as e:
            raise ImportError(
                "xensevr_pc_service_sdk is required for Pico4 teleoperator. "
                "Please install it according to your Pico4 SDK documentation."
            ) from e

        try:
            xrt.init()
            self._xrt = xrt
            self.logger.info("XenseVR SDK initialized successfully.")
            time.sleep(1.0)  # Wait for SDK to stabilize

            # Set target pose on connect and save initial pose for reset
            # Convert Euler angles to quaternion for internal storage
            roll, pitch, yaw = current_tcp_pose_euler[3:6]
            qw, qx, qy, qz = euler_to_quaternion(roll, pitch, yaw)
            qw, qx, qy, qz = normalize_quaternion(qw, qx, qy, qz)
            
            self._target_pos = current_tcp_pose_euler[:3].copy()
            self._target_quat = np.array([qx, qy, qz, qw], dtype=np.float32)
            self._target_gripper_pos = current_tcp_pose_euler[6]
            
            # Save initial pose for reset functionality
            self._start_pos = current_tcp_pose_euler[:3].copy()
            self._start_quat = np.array([qx, qy, qz, qw], dtype=np.float32)
            self._start_gripper_pos = current_tcp_pose_euler[6]

            # Initialize reference pose tracking (will be set on first frame)
            self._ref_controller_pose = None
            self._ref_controller_quat = None

            self._is_connected = True
            self.logger.info(f"{self} connected successfully.")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize XenseVR SDK: {e}") from e

    def calibrate(self) -> None:
        """No calibration needed for Pico4."""
        pass

    def configure(self) -> None:
        """No additional configuration needed."""
        pass

    def reset_to_pose(self, pose_6d: np.ndarray, gripper_pos: float = 0.0) -> None:
        """
        Reset target pose to a specific pose (e.g., home pose).

        Args:
            pose_6d: 6D EEF pose [x, y, z, roll, pitch, yaw]
            gripper_pos: Gripper position in meters
        """
        # Convert Euler angles to quaternion
        roll, pitch, yaw = pose_6d[3:6]
        qw, qx, qy, qz = euler_to_quaternion(roll, pitch, yaw)
        qw, qx, qy, qz = normalize_quaternion(qw, qx, qy, qz)
        
        self._target_pos = np.array(pose_6d[:3], dtype=np.float32).copy()
        self._target_quat = np.array([qx, qy, qz, qw], dtype=np.float32)
        self._target_gripper_pos = float(gripper_pos)
        
        # Reset reference pose to current controller pose (if available)
        if self._is_connected and self._xrt is not None:
            try:
                if self.config.use_right_controller:
                    controller_pose_raw = self._xrt.get_right_controller_pose()
                elif self.config.use_left_controller:
                    controller_pose_raw = self._xrt.get_left_controller_pose()
                else:
                    controller_pose_raw = None
                
                if controller_pose_raw is not None:
                    self._ref_controller_pose = np.array(controller_pose_raw, dtype=np.float32)
                    self._ref_controller_quat = np.array([
                        controller_pose_raw[3], controller_pose_raw[4], 
                        controller_pose_raw[5], controller_pose_raw[6]
                    ])  # [qx, qy, qz, qw]
                    self._ref_controller_quat = self._normalize_quaternion(self._ref_controller_quat)
            except Exception:
                pass  # If controller not available, keep None
        self.logger.info(f"Reset target pose to: {pose_6d}, gripper: {gripper_pos}")

    def _quaternion_to_euler(self, qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
        """Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw).
        
        Based on spacemouse_teleop.py implementation.
        """
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return (roll, pitch, yaw)

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions q1 * q2. Both in [qx, qy, qz, qw] format."""
        qx1, qy1, qz1, qw1 = q1
        qx2, qy2, qz2, qw2 = q2
        
        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
        
        return np.array([qx, qy, qz, qw])

    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse. Input in [qx, qy, qz, qw] format."""
        qx, qy, qz, qw = q
        norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
        if norm_sq < 1e-10:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([-qx, -qy, -qz, qw]) / norm_sq

    def _quaternion_delta(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Compute orientation delta between two quaternions.
        
        Args:
            q1: Previous quaternion [qx, qy, qz, qw]
            q2: Current quaternion [qx, qy, qz, qw]
        
        Returns:
            Delta as Euler angles (roll, pitch, yaw) in radians.
        """
        # Compute relative rotation: q_delta = q2 * q1^-1
        q1_inv = self._quaternion_inverse(q1)
        q_delta = self._quaternion_multiply(q2, q1_inv)
        
        # Convert to Euler angles
        roll, pitch, yaw = self._quaternion_to_euler(q_delta[3], q_delta[0], q_delta[1], q_delta[2])
        return np.array([roll, pitch, yaw])

    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 0.0, 1.0])  # Return identity quaternion
        return q / norm

    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two quaternions.
        
        Args:
            q1: First quaternion [qx, qy, qz, qw]
            q2: Second quaternion [qx, qy, qz, qw]
            t: Interpolation factor [0, 1]
        
        Returns:
            Interpolated quaternion [qx, qy, qz, qw]
        """
        # Normalize inputs
        q1 = self._normalize_quaternion(q1)
        q2 = self._normalize_quaternion(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If dot product is negative, negate one quaternion for shortest path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # Clamp dot product to valid range
        dot = np.clip(dot, -1.0, 1.0)
        
        # If quaternions are very close, use linear interpolation
        if abs(dot) > 0.9995:
            result = q1 + t * (q2 - q1)
            return self._normalize_quaternion(result)
        
        # Compute angle
        theta = np.arccos(abs(dot))
        sin_theta = np.sin(theta)
        
        # Spherical interpolation
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        result = w1 * q1 + w2 * q2
        
        return self._normalize_quaternion(result)

    def _get_filtered_pose(self, controller_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get filtered absolute position and quaternion with moving average.
        
        Args:
            controller_pose: Current controller pose [x, y, z, qx, qy, qz, qw]
        
        Returns:
            Tuple of (filtered_pos, filtered_quat) where:
            - filtered_pos: [x, y, z] absolute position
            - filtered_quat: [qx, qy, qz, qw] absolute quaternion
        """
        # Extract position and quaternion
        pos = controller_pose[:3].copy()
        quat = np.array([controller_pose[3], controller_pose[4], controller_pose[5], controller_pose[6]])  # [qx, qy, qz, qw]
        quat = self._normalize_quaternion(quat)

        # Moving average filter for position
        if self._pos_queue.full():
            self._pos_queue.get()
        self._pos_queue.put(pos)
        filtered_pos = np.mean(np.array(list(self._pos_queue.queue)), axis=0)

        # For quaternion, use simple averaging (normalize at the end)
        # Note: Proper quaternion averaging would use SLERP, but for small changes this works
        if self._quat_queue.full():
            self._quat_queue.get()
        self._quat_queue.put(quat)
        
        # Average quaternions (simple approach - normalize at the end)
        quat_list = list(self._quat_queue.queue)
        filtered_quat = np.mean(np.array(quat_list), axis=0)
        filtered_quat = self._normalize_quaternion(filtered_quat)

        return filtered_pos, filtered_quat

    def get_action(self) -> dict[str, Any]:
        """
        Get the current target pose from the Pico4 VR controller.

        Returns a dictionary with absolute EEF pose (matching ARX5 SDK format):
        - x, y, z: absolute EEF position (meters)
        - roll, pitch, yaw: absolute EEF orientation (radians)
        - gripper_pos: absolute gripper position (meters)
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get controller pose (use right controller by default, or left if right not available)
        if self.config.use_right_controller:
            controller_pose_raw = self._xrt.get_right_controller_pose()
        elif self.config.use_left_controller:
            controller_pose_raw = self._xrt.get_left_controller_pose()
        else:
            raise ValueError("At least one controller must be enabled (use_left_controller or use_right_controller)")

        controller_pose = np.array(controller_pose_raw, dtype=np.float32)  # [x, y, z, qx, qy, qz, qw]

        # Get filtered absolute pose
        filtered_pos, filtered_quat = self._get_filtered_pose(controller_pose)

        # Set reference pose on first frame (if not set)
        if self._ref_controller_pose is None:
            self._ref_controller_pose = controller_pose.copy()
            self._ref_controller_quat = filtered_quat.copy()
            # Initialize target pose to current robot pose (already set in connect)
            # Convert quaternion to Euler for output
            roll, pitch, yaw = self._quaternion_to_euler(
                self._target_quat[3], self._target_quat[0], self._target_quat[1], self._target_quat[2]
            )
            return {
                "x": float(self._target_pos[0]),
                "y": float(self._target_pos[1]),
                "z": float(self._target_pos[2]),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "gripper_pos": float(self._target_gripper_pos),
            }

        # Compute relative position (delta) from reference
        rel_pos = filtered_pos - self._ref_controller_pose[:3]
        
        # Apply sensitivity scaling to position delta
        scaled_rel_pos = rel_pos * self.config.pos_sensitivity

        # Apply deadzone to position
        pos_norm = np.linalg.norm(scaled_rel_pos)
        if pos_norm < self.config.deadzone:
            scaled_rel_pos = np.zeros(3, dtype=np.float32)
        else:
            # Scale to remove deadzone
            scaled_rel_pos = scaled_rel_pos * (pos_norm - self.config.deadzone) / pos_norm

        # Update target position: start_pos + relative movement (accumulate delta)
        self._target_pos = self._start_pos + scaled_rel_pos

        # For orientation, use quaternion directly: target_quat = filtered_quat * ref_quat^-1 * start_quat
        # This applies the relative rotation from reference to the start orientation
        ref_quat_inv = self._quaternion_inverse(self._ref_controller_quat)
        rel_quat = self._quaternion_multiply(filtered_quat, ref_quat_inv)
        self._target_quat = self._quaternion_multiply(rel_quat, self._start_quat)
        self._target_quat = self._normalize_quaternion(self._target_quat)

        # Use fixed control_dt for consistent velocity scaling
        dt = self.config.control_dt

        # Get button states for gripper control
        if self.config.use_right_controller:
            button_open = self._xrt.get_A_button()  # A button on right controller
            button_close = self._xrt.get_B_button()  # B button on right controller
        else:
            button_open = self._xrt.get_X_button()  # X button on left controller
            button_close = self._xrt.get_Y_button()  # Y button on left controller

        if self.config.swap_gripper_buttons:
            button_open, button_close = button_close, button_open

        # Compute gripper command
        if button_open and not button_close:
            gripper_cmd = 1  # Open
        elif button_close and not button_open:
            gripper_cmd = -1  # Close
        else:
            gripper_cmd = 0  # Stay

        # Update gripper position with clamping
        self._target_gripper_pos += gripper_cmd * self.config.gripper_speed * dt
        if self._target_gripper_pos >= self.config.gripper_width:
            self._target_gripper_pos = self.config.gripper_width
        elif self._target_gripper_pos <= 0:
            self._target_gripper_pos = 0

        # Check if any input is active
        # Compute orientation change magnitude for motion detection
        rel_quat_magnitude = np.linalg.norm(rel_quat[:3])  # Use vector part magnitude as approximation
        motion_active = np.any(np.abs(scaled_rel_pos) > 0.001) or rel_quat_magnitude > 0.001
        self._enabled = motion_active or button_open or button_close

        # Convert target quaternion to Euler angles for output
        roll, pitch, yaw = self._quaternion_to_euler(
            self._target_quat[3], self._target_quat[0], self._target_quat[1], self._target_quat[2]
        )

        # Return absolute pose dict
        return {
            "x": float(self._target_pos[0]),
            "y": float(self._target_pos[1]),
            "z": float(self._target_pos[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "gripper_pos": float(self._target_gripper_pos),
        }

    def get_target_pose_array(self) -> tuple[np.ndarray, float]:
        """
        Get the current target pose as numpy array (for direct use with ARX5 SDK).

        Returns:
            Tuple of (pose_6d, gripper_pos) where pose_6d is [x, y, z, roll, pitch, yaw]
        """
        roll, pitch, yaw = self._quaternion_to_euler(
            self._target_quat[3], self._target_quat[0], self._target_quat[1], self._target_quat[2]
        )
        pose_6d = np.array([
            self._target_pos[0],
            self._target_pos[1],
            self._target_pos[2],
            roll,
            pitch,
            yaw,
        ], dtype=np.float32)
        return pose_6d, self._target_gripper_pos

    def convert_to_flexiv_action(self, pico4_action: dict[str, Any]) -> dict[str, Any]:
        """Convert Pico4 action (Euler angles) to Flexiv Rizon4 action (quaternion).
        
        This matches the behavior of spacemouse_teleop.py example:
        - Pico4 maintains absolute pose in Euler angles [x, y, z, roll, pitch, yaw]
        - Convert to quaternion format [x, y, z, qw, qx, qy, qz] for Flexiv SDK
        
        Args:
            pico4_action: Dictionary with keys {x, y, z, roll, pitch, yaw, gripper_pos}
        
        Returns:
            Dictionary with keys {tcp.x, tcp.y, tcp.z, tcp.qw, tcp.qx, tcp.qy, tcp.qz, gripper.pos}
        """
        # Convert Euler angles to quaternion
        qw, qx, qy, qz = euler_to_quaternion(
            pico4_action["roll"],
            pico4_action["pitch"],
            pico4_action["yaw"],
        )
        
        # Normalize quaternion
        qw, qx, qy, qz = normalize_quaternion(qw, qx, qy, qz)
        
        # Map to Flexiv action format
        return {
            "tcp.x": pico4_action["x"],
            "tcp.y": pico4_action["y"],
            "tcp.z": pico4_action["z"],
            "tcp.qw": qw,
            "tcp.qx": qx,
            "tcp.qy": qy,
            "tcp.qz": qz,
            "gripper.pos": pico4_action["gripper_pos"],
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Pico4 doesn't support feedback."""
        raise NotImplementedError("Feedback is not implemented for Pico4 teleoperator.")

    def disconnect(self) -> None:
        """Disconnect from the Pico4 VR headset."""
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Closing XenseVR SDK...")
        try:
            self._xrt.close()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to close XenseVR SDK: {e}") from e
        finally:
            self._is_connected = False
            self._xrt = None
            self.logger.info(f"{self} disconnected.")

    def __del__(self):
        """Cleanup on deletion."""
        if self._is_connected:
            try:
                self.disconnect()
            except Exception:
                pass
