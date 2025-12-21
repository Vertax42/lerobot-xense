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

    Data flow:
    1. Get raw data from Pico4: controller_pose [x,y,z,qx,qy,qz,qw], grip, trigger
    2. Apply window filter to raw pose data (using SLERP for quaternions)
    3. Transform from Pico4 coordinate system to Flexiv coordinate system
    4. Compute relative changes and update target pose
    5. Output in Flexiv Rizon4 action format

    Output action format (Flexiv Rizon4):
    - tcp.x, tcp.y, tcp.z: absolute EEF target position (meters, accumulated)
    - tcp.qw, tcp.qx, tcp.qy, tcp.qz: absolute EEF target orientation (quaternion, accumulated)
    - gripper.pos: absolute gripper position (meters)
    """

    config_class = Pico4Config
    name = "pico4"

    def __init__(self, config: Pico4Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._xrt = None
        self.logger = spdlog.ConsoleLogger("Pico4Teleop")

        # Target pose tracking (in Flexiv coordinate system)
        self._target_pos: np.ndarray = np.zeros(3, dtype=np.float32)  # [x, y, z]
        self._target_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [qw, qx, qy, qz]
        self._target_gripper_pos: float = 0.0
        self._start_pos: np.ndarray = np.zeros(3, dtype=np.float32)  # [x, y, z]
        self._start_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [qw, qx, qy, qz]
        self._start_gripper_pos: float = 0.0

        # Reference pose for relative control (in Flexiv coordinate system after transformation)
        self._ref_pos: np.ndarray | None = None  # [x, y, z] in Flexiv frame
        self._ref_quat: np.ndarray | None = None  # [qw, qx, qy, qz] in Flexiv frame

        # Window filter queues for raw Pico4 data (before coordinate transformation)
        # Filter raw pose data from Pico4 SDK
        self._raw_pos_queue: Queue = Queue(self.config.filter_window_size)  # Raw position [x, y, z] in Pico4 frame
        self._raw_quat_queue: Queue = Queue(self.config.filter_window_size)  # Raw quaternion [qx, qy, qz, qw] in Pico4 frame

        # State tracking
        self._enabled: bool = False

    @property
    def is_connected(self) -> bool:
        """Check if the Pico4 VR headset is connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Pico4 doesn't require calibration."""
        return self._is_connected

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Return action features matching Flexiv Rizon4 format.

        Returns a dictionary with dtype, shape, and names for the action space:
        - tcp.x, tcp.y, tcp.z: absolute TCP position (meters) in Flexiv frame
        - tcp.qw, tcp.qx, tcp.qy, tcp.qz: absolute TCP orientation (quaternion) in Flexiv frame
        - gripper.pos: absolute gripper position (meters)
        """
        return {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "tcp.x": 0,
                "tcp.y": 1,
                "tcp.z": 2,
                "tcp.qw": 3,
                "tcp.qx": 4,
                "tcp.qy": 5,
                "tcp.qz": 6,
                "gripper.pos": 7,
            },
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        """Pico4 doesn't support feedback."""
        return {}

    def connect(self, calibrate: bool = True, current_tcp_pose_quat: np.ndarray = np.zeros(8, dtype=np.float32)) -> None:
        """Connect to the Pico4 VR headset via xrt SDK.
        
        Args:
            calibrate: Unused, kept for compatibility with Teleoperator interface
            current_tcp_pose_quat: Current TCP pose in quaternion format [x, y, z, qw, qx, qy, qz, gripper_pos]
        """
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
            # Input format: [x, y, z, qw, qx, qy, qz, gripper_pos]
            self._target_pos = current_tcp_pose_quat[:3].copy()  # [x, y, z]
            self._target_quat = normalize_quaternion(current_tcp_pose_quat[3:7])
            self._target_gripper_pos = current_tcp_pose_quat[7]
            
            # Save initial pose for reset functionality
            self._start_pos = current_tcp_pose_quat[:3].copy()
            self._start_quat = normalize_quaternion(current_tcp_pose_quat[3:7])
            self._start_gripper_pos = current_tcp_pose_quat[7]

            # Initialize reference pose tracking (will be set on first frame)
            self._ref_pos = None
            self._ref_quat = None

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
            pose_6d: 6D EEF pose [x, y, z, roll, pitch, yaw] in Flexiv frame
            gripper_pos: Gripper position in meters
        """
        # Convert Euler angles to quaternion
        roll, pitch, yaw = pose_6d[3:6]
        qw, qx, qy, qz = euler_to_quaternion(roll, pitch, yaw)
        qw, qx, qy, qz = normalize_quaternion(qw, qx, qy, qz)
        
        self._target_pos = np.array(pose_6d[:3], dtype=np.float32).copy()
        self._target_quat = np.array([qw, qx, qy, qz], dtype=np.float32)  # [qw, qx, qy, qz]
        self._target_gripper_pos = float(gripper_pos)
        
        # Reset reference pose to current controller pose (if available)
        if self._is_connected and self._xrt is not None:
            try:
                # Get raw controller data
                if self.config.use_right_controller:
                    controller_pose_raw = self._xrt.get_right_controller_pose()
                elif self.config.use_left_controller:
                    controller_pose_raw = self._xrt.get_left_controller_pose()
                else:
                    controller_pose_raw = None
                
                if controller_pose_raw is not None:
                    # Step 1: Apply window filter to raw data
                    filtered_pos_pico, filtered_quat_pico = self._filter_raw_pose(controller_pose_raw)
                    
                    # Step 2: Transform to Flexiv coordinate system
                    pos_flexiv, quat_flexiv = self._transform_pico_to_flexiv_coordinate(
                        filtered_pos_pico, filtered_quat_pico
                    )
                    
                    # Store as reference
                    self._ref_pos = pos_flexiv
                    self._ref_quat = quat_flexiv
            except Exception:
                pass  # If controller not available, keep None
        self.logger.info(f"Reset target pose to: {pose_6d}, gripper: {gripper_pos}")

    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length. Input and output in [qw, qx, qy, qz] format."""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Return identity quaternion [qw, qx, qy, qz]
        return (q / norm).astype(np.float32)

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions q1 * q2. Both in [qw, qx, qy, qz] format."""
        qw1, qx1, qy1, qz1 = q1
        qw2, qx2, qy2, qz2 = q2
        
        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
        
        return np.array([qw, qx, qy, qz], dtype=np.float32)

    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse. Input and output in [qw, qx, qy, qz] format."""
        qw, qx, qy, qz = q
        norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
        if norm_sq < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity [qw, qx, qy, qz]
        return np.array([qw, -qx, -qy, -qz], dtype=np.float32) / norm_sq

    def _quaternion_to_euler(self, qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
        """Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw)."""
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return (roll, pitch, yaw)

    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical Linear Interpolation (SLERP) between two quaternions.
        
        SLERP is the proper way to interpolate between quaternions on the unit sphere,
        ensuring constant angular velocity and shortest path rotation.
        
        Formula: q(t) = (sin((1-t)*θ)/sin(θ)) * q1 + (sin(t*θ)/sin(θ)) * q2
        where θ = arccos(dot(q1, q2))
        
        Args:
            q1: First quaternion [qx, qy, qz, qw]
            q2: Second quaternion [qx, qy, qz, qw]
            t: Interpolation factor [0, 1], where 0 returns q1 and 1 returns q2
        
        Returns:
            Interpolated quaternion [qx, qy, qz, qw]
        """
        # Normalize inputs
        q1 = self._normalize_quaternion(q1)
        q2 = self._normalize_quaternion(q2)
        
        # Compute dot product (cosine of angle between quaternions)
        dot = np.dot(q1, q2)
        
        # If dot product is negative, negate one quaternion for shortest path
        # This ensures we take the shorter rotation path (q and -q represent same rotation)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # Clamp dot product to valid range [-1, 1] for arccos
        dot = np.clip(dot, -1.0, 1.0)
        
        # If quaternions are very close (dot > 0.9995), use linear interpolation
        # This avoids numerical instability when sin(θ) is very small
        if abs(dot) > 0.9995:
            result = q1 + t * (q2 - q1)
            return self._normalize_quaternion(result)
        
        # Compute angle between quaternions
        theta = np.arccos(abs(dot))
        sin_theta = np.sin(theta)
        
        # SLERP weights
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        # Spherical interpolation
        result = w1 * q1 + w2 * q2
        
        return self._normalize_quaternion(result)

    def _filter_raw_pose(self, controller_pose_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply window filter to raw Pico4 controller pose data.
        
        This is Step 1: Filter raw data from Pico4 SDK before coordinate transformation.
        
        For position: Simple moving average (arithmetic mean)
        For quaternion: Sequential SLERP interpolation through the window
        
        Args:
            controller_pose_raw: Raw controller pose from SDK [x, y, z, qx, qy, qz, qw] in Pico4 frame
        
        Returns:
            Tuple of (filtered_pos, filtered_quat) in Pico4 frame
        """
        # Extract position and quaternion from raw data
        pos = controller_pose_raw[:3].copy()  # [x, y, z] in Pico4 frame
        # Pico4 provides [x, y, z, qx, qy, qz, qw], convert to [qw, qx, qy, qz] for internal use
        quat = np.array([
            controller_pose_raw[6],  # qw
            controller_pose_raw[3],  # qx
            controller_pose_raw[4],  # qy
            controller_pose_raw[5],  # qz
        ], dtype=np.float32)  # [qw, qx, qy, qz] in Pico4 frame
        quat = self._normalize_quaternion(quat)

        # Moving average filter for position (window filter)
        if self._raw_pos_queue.full():
            self._raw_pos_queue.get()
        self._raw_pos_queue.put(pos)
        filtered_pos = np.mean(np.array(list(self._raw_pos_queue.queue)), axis=0)

        # SLERP-based filter for quaternion
        # Method: Use SLERP between first and last quaternion in the window (midpoint)
        # This provides smooth interpolation across the entire window
        if self._raw_quat_queue.full():
            self._raw_quat_queue.get()
        self._raw_quat_queue.put(quat)
        
        quat_list = list(self._raw_quat_queue.queue)
        n = len(quat_list)
        
        if n == 1:
            filtered_quat = quat_list[0]
        elif n == 2:
            # For 2 quaternions, SLERP at midpoint (t=0.5)
            filtered_quat = self._slerp_quaternion(quat_list[0], quat_list[1], 0.5)
        else:
            # For multiple quaternions, use recursive SLERP:
            # 1. Split window into two halves
            # 2. SLERP each half to get midpoint
            # 3. SLERP the two midpoints to get final result
            mid = n // 2
            left_half = quat_list[:mid+1]
            right_half = quat_list[mid:]
            
            # SLERP first half: from first to middle
            if len(left_half) == 1:
                left_mid = left_half[0]
            else:
                left_mid = self._slerp_quaternion(left_half[0], left_half[-1], 0.5)
            
            # SLERP second half: from middle to last
            if len(right_half) == 1:
                right_mid = right_half[0]
            else:
                right_mid = self._slerp_quaternion(right_half[0], right_half[-1], 0.5)
            
            # Final SLERP between the two midpoints
            filtered_quat = self._slerp_quaternion(left_mid, right_mid, 0.5)
        
        filtered_quat = self._normalize_quaternion(filtered_quat)

        return filtered_pos, filtered_quat

    def _transform_pico_to_flexiv_coordinate(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform pose from Pico4 coordinate system to Flexiv coordinate system.
        
        This is Step 2: Coordinate transformation after filtering.
        
        Pico4: Right-handed, X right, Y up, Z in (toward user)
        Flexiv: Right-handed, X forward (away from base), Y left, Z up
        
        If user stands in front of robot, "in" (toward user) and "forward" (away from base) are opposite.
        
        Transformation:
        - Pico4 X (right) -> Flexiv Y (left, so negate)
        - Pico4 Y (up) -> Flexiv Z (up, same)
        - Pico4 Z (in, toward user) -> Flexiv X (forward, away from base, so negate if opposite)
        
        Args:
            pos: Position in Pico4 frame [x, y, z]
            quat: Quaternion in Pico4 frame [qw, qx, qy, qz] (from _filter_raw_pose)
        
        Returns:
            Tuple of (transformed_pos, transformed_quat) in Flexiv frame [qw, qx, qy, qz]
        """
        # Position transformation: [Pico4_x, Pico4_y, Pico4_z] -> [Flexiv_x, Flexiv_y, Flexiv_z]
        # Pico4 X (right) -> Flexiv Y (left): negate
        # Pico4 Y (up) -> Flexiv Z (up): same
        # Pico4 Z (in, toward user) -> Flexiv X (forward, away from base): negate (opposite direction)
        transformed_pos = np.array([
            -pos[2],  # Pico4 Z (in) -> Flexiv X (forward, negated because opposite direction)
            -pos[0],  # Pico4 X (right) -> Flexiv Y (left, negated)
            pos[1],   # Pico4 Y (up) -> Flexiv Z (up, same)
        ], dtype=np.float32)
        
        # Quaternion transformation: need to rotate the coordinate frame
        # Frame transformation quaternion: 90° rotation around Y, then -90° around Z
        # Input quat is already in [qw, qx, qy, qz] format from _filter_raw_pose
        quat_internal = quat  # Already in [qw, qx, qy, qz] format
        
        sqrt_half = np.sqrt(0.5)
        q_y_90 = np.array([sqrt_half, 0.0, sqrt_half, 0.0], dtype=np.float32)  # [qw, qx, qy, qz] = [√2/2, 0, √2/2, 0]
        q_z_neg90 = np.array([sqrt_half, 0.0, 0.0, -sqrt_half], dtype=np.float32)  # [qw, qx, qy, qz] = [√2/2, 0, 0, -√2/2]
        q_frame_transform = self._quaternion_multiply(q_y_90, q_z_neg90)
        q_frame_transform = self._normalize_quaternion(q_frame_transform)
        
        # Transform quaternion: q_flexiv = q_transform * q_pico * q_transform^-1
        q_transform_inv = self._quaternion_inverse(q_frame_transform)
        q_temp = self._quaternion_multiply(q_frame_transform, quat_internal)
        transformed_quat = self._quaternion_multiply(q_temp, q_transform_inv)
        transformed_quat = self._normalize_quaternion(transformed_quat)
        
        return transformed_pos, transformed_quat

    def get_action(self) -> dict[str, Any]:
        """
        Get the current target pose from the Pico4 VR controller.

        Data processing pipeline:
        1. Get raw data: controller_pose, grip, trigger from Pico4 SDK
        2. Apply window filter to raw pose data
        3. Transform from Pico4 to Flexiv coordinate system
        4. Compute relative movement and update target pose
        5. Convert to output format [x, y, z, roll, pitch, yaw, gripper_pos]

        Returns a dictionary with absolute EEF pose (matching Flexiv Rizon4 format):
        - tcp.x, tcp.y, tcp.z: absolute TCP position (meters) in Flexiv frame
        - tcp.qw, tcp.qx, tcp.qy, tcp.qz: absolute TCP orientation (quaternion) in Flexiv frame
        - gripper.pos: absolute gripper position (meters)
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Step 1: Get raw data from Pico4 SDK
        if self.config.use_right_controller:
            controller_pose_raw = self._xrt.get_right_controller_pose()  # [x, y, z, qx, qy, qz, qw] in Pico4 frame
            controller_grip = self._xrt.get_right_grip()
            controller_trigger = self._xrt.get_right_trigger()
        elif self.config.use_left_controller:
            controller_pose_raw = self._xrt.get_left_controller_pose()  # [x, y, z, qx, qy, qz, qw] in Pico4 frame
            controller_grip = self._xrt.get_left_grip()
            controller_trigger = self._xrt.get_left_trigger()
        else:
            raise ValueError("At least one controller must be enabled (use_left_controller or use_right_controller)")

        # Step 2: Apply window filter to raw pose data (in Pico4 frame)
        filtered_pos_pico, filtered_quat_pico = self._filter_raw_pose(controller_pose_raw)

        # Step 3: Transform from Pico4 coordinate system to Flexiv coordinate system
        filtered_pos_flexiv, filtered_quat_flexiv = self._transform_pico_to_flexiv_coordinate(
            filtered_pos_pico, filtered_quat_pico
        )

        # Set reference pose on first frame (if not set)
        if self._ref_pos is None:
            # Store reference in Flexiv coordinate system
            self._ref_pos = filtered_pos_flexiv.copy()
            self._ref_quat = filtered_quat_flexiv.copy()
            # Initialize target pose to current robot pose (already set in connect)
            # Return in Flexiv format: [tcp.x, tcp.y, tcp.z, tcp.qw, tcp.qx, tcp.qy, tcp.qz, gripper.pos]
            # _target_quat is in [qw, qx, qy, qz] format
            return {
                "tcp.x": float(self._target_pos[0]),
                "tcp.y": float(self._target_pos[1]),
                "tcp.z": float(self._target_pos[2]),
                "tcp.qw": float(self._target_quat[0]),
                "tcp.qx": float(self._target_quat[1]),
                "tcp.qy": float(self._target_quat[2]),
                "tcp.qz": float(self._target_quat[3]),
                "gripper.pos": float(self._target_gripper_pos),
            }

        # Step 4: Compute relative position (delta) from reference (both in Flexiv frame)
        rel_pos = filtered_pos_flexiv - self._ref_pos
        
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

        # For orientation, use quaternion directly: target_quat = filtered_quat_flexiv * ref_quat^-1 * start_quat
        # This applies the relative rotation from reference to the start orientation
        ref_quat_inv = self._quaternion_inverse(self._ref_quat)
        rel_quat = self._quaternion_multiply(filtered_quat_flexiv, ref_quat_inv)
        self._target_quat = self._quaternion_multiply(rel_quat, self._start_quat)
        self._target_quat = self._normalize_quaternion(self._target_quat)

        # Get button states for gripper control
        if self.config.use_right_controller:
            button_open = self._xrt.get_A_button()  # A button on right controller
            button_close = self._xrt.get_B_button()  # B button on right controller
        else:
            button_open = self._xrt.get_X_button()  # X button on left controller
            button_close = self._xrt.get_Y_button()  # Y button on left controller

        # Compute gripper command
        if button_open and not button_close:
            gripper_cmd = 1  # Open
        elif button_close and not button_open:
            gripper_cmd = -1  # Close
        else:
            gripper_cmd = 0  # Stay

        # Update gripper position with clamping
        dt = self.config.control_dt
        self._target_gripper_pos += gripper_cmd * self.config.gripper_speed * dt
        if self._target_gripper_pos >= self.config.gripper_width:
            self._target_gripper_pos = self.config.gripper_width
        elif self._target_gripper_pos <= 0:
            self._target_gripper_pos = 0

        # Check if any input is active
        rel_quat_magnitude = np.linalg.norm(rel_quat[:3])  # Use vector part magnitude as approximation
        motion_active = np.any(np.abs(scaled_rel_pos) > 0.001) or rel_quat_magnitude > 0.001
        self._enabled = motion_active or button_open or button_close

        # Step 5: Return in Flexiv Rizon4 action format
        # Format: {tcp.x, tcp.y, tcp.z, tcp.qw, tcp.qx, tcp.qy, tcp.qz, gripper.pos}
        # _target_quat is in [qw, qx, qy, qz] format
        return {
            "tcp.x": float(self._target_pos[0]),
            "tcp.y": float(self._target_pos[1]),
            "tcp.z": float(self._target_pos[2]),
            "tcp.qw": float(self._target_quat[0]),
            "tcp.qx": float(self._target_quat[1]),
            "tcp.qy": float(self._target_quat[2]),
            "tcp.qz": float(self._target_quat[3]),
            "gripper.pos": float(self._target_gripper_pos),
        }

    def get_target_pose_array(self) -> tuple[np.ndarray, float]:
        """
        Get the current target pose as numpy array (for direct use with Flexiv SDK).

        Returns:
            Tuple of (tcp_pose, gripper_pos) where tcp_pose is [x, y, z, qw, qx, qy, qz] in Flexiv frame
        """
        # _target_quat is in [qw, qx, qy, qz] format
        tcp_pose = np.array([
            self._target_pos[0],
            self._target_pos[1],
            self._target_pos[2],
            self._target_quat[0],  # qw
            self._target_quat[1],  # qx
            self._target_quat[2],  # qy
            self._target_quat[3],  # qz
        ], dtype=np.float32)
        return tcp_pose, self._target_gripper_pos

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
