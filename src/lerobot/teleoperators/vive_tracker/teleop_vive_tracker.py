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
Vive Tracker Teleoperator for LeRobot.

This teleoperator provides 6-DoF absolute pose tracking using HTC Vive Tracker,
suitable for end-effector teleoperation. Uses pysurvive library for tracking.

Coordinate transformation:
- At connect time: records vive_init_pose and ee_init_pose (from robot)
- Computes transform_matrix = ee_init_pose * vive_init_pose.inverse()
- At get_action: action_pose = transform_matrix * vive_current_pose
"""

import queue
import sys
import threading
import time
from queue import Queue
from typing import Any

import numpy as np
import spdlog

from ..teleoperator import Teleoperator
from .config_vive_tracker import ViveTrackerConfig


class PoseData:
    """Pose data structure for storing and formatting pose information."""

    def __init__(self, device_name: str, timestamp: float, 
                 position: list, rotation: list):
        self.device_name = device_name
        self.timestamp = timestamp
        self.position = position  # [x, y, z]
        self.rotation = rotation  # [qw, qx, qy, qz] quaternion (wxyz format)

    def __str__(self):
        """Format output pose information."""
        return (
            f"{self.device_name}: T: {self.timestamp:.6f} "
            f"P: {self.position[0]:9.6f}, {self.position[1]:9.6f}, {self.position[2]:9.6f} "
            f"R: {self.rotation[0]:9.6f}, {self.rotation[1]:9.6f}, "
            f"{self.rotation[2]:9.6f}, {self.rotation[3]:9.6f}"
        )


class ViveTrackerTeleop(Teleoperator):
    """
    Vive Tracker Teleoperator.
    
    Provides 6-DoF pose tracking using HTC Vive Tracker for robot teleoperation.
    Uses pysurvive library for SteamVR Lighthouse tracking.
    
    Coordinate transformation:
    - transform_matrix = ee_init_pose * vive_init_pose.inverse()
    - action_pose = transform_matrix * vive_current_pose
    
    Output format: [x, y, z, qw, qx, qy, qz] (7D pose in robot frame)
    """

    config_class = ViveTrackerConfig
    name = "vive_tracker"

    def __init__(self, config: ViveTrackerConfig):
        super().__init__(config)
        self.config = config

        # Logger
        self.logger = spdlog.ConsoleLogger("ViveTrackerTeleop")

        # Import pysurvive
        try:
            import pysurvive
            self._pysurvive = pysurvive
        except ImportError:
            raise ImportError(
                "pysurvive library not found. Please install it: "
                "pip install pysurvive or build from source"
            )

        # Build pysurvive parameters
        survive_args = sys.argv[:1]  # Keep program name
        if self.config.config_path:
            survive_args.extend(["--config", self.config.config_path])
        if self.config.lh_config:
            survive_args.extend(["--lh", self.config.lh_config])

        # Initialize pysurvive context
        self.logger.info("Initializing pysurvive context...")
        self._context = pysurvive.SimpleContext(survive_args)
        if not self._context:
            raise RuntimeError("Cannot initialize pysurvive context")
        self.logger.info("✅ Pysurvive context initialized successfully")

        # Connection state
        self._is_connected = False

        # Threading for pysurvive data collection
        self._running = False
        self._pose_queue = queue.Queue(maxsize=100)
        self._data_lock = threading.Lock()
        self._latest_poses: dict[str, PoseData] = {}
        self._devices_info: dict[str, dict] = {}

        # Threads
        self._collector_thread = None
        self._processor_thread = None

        # Active tracker
        self._active_tracker: str | None = None

        # Coordinate transformation matrix (4x4)
        # transform_matrix = ee_init_pose * vive_init_pose.inverse()
        self._transform_matrix: np.ndarray | None = None

        # Window filter queues for raw pose data
        self._raw_pos_queue: Queue = Queue(self.config.filter_window_size)
        self._raw_quat_queue: Queue = Queue(self.config.filter_window_size)

        # Position jump filtering
        self._last_raw_pose: np.ndarray | None = None
        self._jump_filter_count: int = 0

    @property
    def action_features(self) -> dict:
        """Action features: 7D pose [x, y, z, qw, qx, qy, qz]."""
        return {
            "tcp.x": float,
            "tcp.y": float,
            "tcp.z": float,
            "tcp.qw": float,
            "tcp.qx": float,
            "tcp.qy": float,
            "tcp.qz": float,
        }

    @property
    def feedback_features(self) -> dict:
        """No feedback features for this teleoperator."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Vive Tracker uses lighthouse calibration, always True if connected."""
        return self.is_connected

    # ==================== Matrix/Pose Conversion Utilities ====================

    def _pose7d_to_matrix(self, pose: np.ndarray) -> np.ndarray:
        """Convert 7D pose [x, y, z, qw, qx, qy, qz] to 4x4 transformation matrix.
        
        Args:
            pose: 7D pose array [x, y, z, qw, qx, qy, qz]
            
        Returns:
            4x4 transformation matrix
        """
        x, y, z = pose[0], pose[1], pose[2]
        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
        
        # Normalize quaternion
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if norm < 1e-10:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        else:
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        # Build rotation matrix from quaternion
        # R = [[1-2(qy²+qz²), 2(qxqy-qwqz), 2(qxqz+qwqy)],
        #      [2(qxqy+qwqz), 1-2(qx²+qz²), 2(qyqz-qwqx)],
        #      [2(qxqz-qwqy), 2(qyqz+qwqx), 1-2(qx²+qy²)]]
        r00 = 1 - 2*(qy*qy + qz*qz)
        r01 = 2*(qx*qy - qw*qz)
        r02 = 2*(qx*qz + qw*qy)
        r10 = 2*(qx*qy + qw*qz)
        r11 = 1 - 2*(qx*qx + qz*qz)
        r12 = 2*(qy*qz - qw*qx)
        r20 = 2*(qx*qz - qw*qy)
        r21 = 2*(qy*qz + qw*qx)
        r22 = 1 - 2*(qx*qx + qy*qy)
        
        matrix = np.array([
            [r00, r01, r02, x],
            [r10, r11, r12, y],
            [r20, r21, r22, z],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        return matrix

    def _matrix_to_pose7d(self, matrix: np.ndarray) -> np.ndarray:
        """Convert 4x4 transformation matrix to 7D pose [x, y, z, qw, qx, qy, qz].
        
        Args:
            matrix: 4x4 transformation matrix
            
        Returns:
            7D pose array [x, y, z, qw, qx, qy, qz]
        """
        # Extract translation
        x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
        
        # Extract rotation matrix
        r00, r01, r02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        r10, r11, r12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        r20, r21, r22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]
        
        # Convert rotation matrix to quaternion
        # Using Shepperd's method for numerical stability
        trace = r00 + r11 + r22
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (r21 - r12) * s
            qy = (r02 - r20) * s
            qz = (r10 - r01) * s
        elif r00 > r11 and r00 > r22:
            s = 2.0 * np.sqrt(1.0 + r00 - r11 - r22)
            qw = (r21 - r12) / s
            qx = 0.25 * s
            qy = (r01 + r10) / s
            qz = (r02 + r20) / s
        elif r11 > r22:
            s = 2.0 * np.sqrt(1.0 + r11 - r00 - r22)
            qw = (r02 - r20) / s
            qx = (r01 + r10) / s
            qy = 0.25 * s
            qz = (r12 + r21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r22 - r00 - r11)
            qw = (r10 - r01) / s
            qx = (r02 + r20) / s
            qy = (r12 + r21) / s
            qz = 0.25 * s
        
        # Normalize quaternion
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if norm > 1e-10:
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        return np.array([x, y, z, qw, qx, qy, qz], dtype=np.float32)

    def _matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Compute inverse of a 4x4 transformation matrix.
        
        For a transformation matrix [R|t], the inverse is [R^T|-R^T*t].
        
        Args:
            matrix: 4x4 transformation matrix
            
        Returns:
            4x4 inverse transformation matrix
        """
        R = matrix[:3, :3]
        t = matrix[:3, 3]
        
        R_inv = R.T
        t_inv = -R_inv @ t
        
        inv_matrix = np.eye(4, dtype=np.float64)
        inv_matrix[:3, :3] = R_inv
        inv_matrix[:3, 3] = t_inv
        
        return inv_matrix

    # ==================== Quaternion Utilities ====================

    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length. Input/output in [qw, qx, qy, qz] format."""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return (q / norm).astype(np.float32)

    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical Linear Interpolation (SLERP) between two quaternions.

        Args:
            q1: First quaternion [qw, qx, qy, qz]
            q2: Second quaternion [qw, qx, qy, qz]
            t: Interpolation factor [0, 1]

        Returns:
            Interpolated quaternion [qw, qx, qy, qz]
        """
        q1 = self._normalize_quaternion(q1)
        q2 = self._normalize_quaternion(q2)

        dot = np.dot(q1, q2)

        if dot < 0.0:
            q2 = -q2
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        if abs(dot) > 0.9995:
            result = q1 + t * (q2 - q1)
            return self._normalize_quaternion(result)

        theta = np.arccos(abs(dot))
        sin_theta = np.sin(theta)

        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        result = w1 * q1 + w2 * q2

        return self._normalize_quaternion(result)

    # ==================== Filter ====================

    def _filter_raw_pose(self, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply window filter to raw pose data.

        Args:
            pos: Position [x, y, z]
            quat: Quaternion [qw, qx, qy, qz]

        Returns:
            Tuple of (filtered_pos, filtered_quat)
        """
        if self.config.filter_window_size <= 1:
            return pos, quat

        # Moving average filter for position
        if self._raw_pos_queue.full():
            self._raw_pos_queue.get()
        self._raw_pos_queue.put(pos.copy())
        filtered_pos = np.mean(np.array(list(self._raw_pos_queue.queue)), axis=0)

        # SLERP-based filter for quaternion
        if self._raw_quat_queue.full():
            self._raw_quat_queue.get()
        self._raw_quat_queue.put(quat.copy())

        quat_list = list(self._raw_quat_queue.queue)
        n = len(quat_list)

        if n == 1:
            filtered_quat = quat_list[0]
        elif n == 2:
            filtered_quat = self._slerp_quaternion(quat_list[0], quat_list[1], 0.5)
        else:
            mid = n // 2
            left_half = quat_list[:mid+1]
            right_half = quat_list[mid:]

            if len(left_half) == 1:
                left_mid = left_half[0]
            else:
                left_mid = self._slerp_quaternion(left_half[0], left_half[-1], 0.5)

            if len(right_half) == 1:
                right_mid = right_half[0]
            else:
                right_mid = self._slerp_quaternion(right_half[0], right_half[-1], 0.5)

            filtered_quat = self._slerp_quaternion(left_mid, right_mid, 0.5)

        filtered_quat = self._normalize_quaternion(filtered_quat)

        return filtered_pos.astype(np.float32), filtered_quat

    # ==================== Main Methods ====================

    def connect(self, calibrate: bool = True, current_tcp_pose_quat: np.ndarray | None = None) -> None:
        """Start Vive Tracker pose tracking and compute coordinate transformation.
        
        Args:
            calibrate: Unused, kept for API compatibility
            current_tcp_pose_quat: Current robot TCP pose [x, y, z, qw, qx, qy, qz].
                                   Required for computing the vive-to-robot transformation.
        """
        if self._is_connected:
            self.logger.warn("Vive Tracker is already connected")
            return

        if current_tcp_pose_quat is None:
            raise ValueError(
                "current_tcp_pose_quat is required for Vive Tracker. "
                "Please provide the current robot TCP pose [x, y, z, qw, qx, qy, qz]."
            )

        try:
            # Mark as running
            self._running = True
            self._is_connected = True

            # Start threads
            self._collector_thread = threading.Thread(
                target=self._pose_collector, daemon=True
            )
            self._collector_thread.start()

            self._processor_thread = threading.Thread(
                target=self._pose_processor, daemon=True
            )
            self._processor_thread.start()

            self.logger.info("Vive Tracker pose tracking started")

            # Wait for devices
            time.sleep(0.5)
            devices = self._wait_for_devices(
                timeout=self.config.device_wait_timeout,
                required_trackers=self.config.required_trackers,
            )

            trackers = devices["trackers"]
            if not trackers:
                raise RuntimeError("No trackers detected!")

            # Select active tracker
            if self.config.tracker_name and self.config.tracker_name in trackers:
                self._active_tracker = self.config.tracker_name
            else:
                self._active_tracker = trackers[0]
                if self.config.tracker_name:
                    self.logger.warn(
                        f"Requested tracker '{self.config.tracker_name}' not found, "
                        f"using '{self._active_tracker}'"
                    )

            self.logger.info(f"✅ Using tracker: {self._active_tracker}")

            # Wait for initial vive pose
            self.logger.info("Waiting for initial Vive pose...")
            vive_init_pose = self._wait_for_initial_pose(timeout=5.0)
            if vive_init_pose is None:
                raise RuntimeError("Failed to get initial Vive pose")

            # Compute transformation matrix
            # transform_matrix = ee_init_pose * vive_init_pose.inverse()
            ee_init_pose = np.array(current_tcp_pose_quat, dtype=np.float64)
            ee_init_matrix = self._pose7d_to_matrix(ee_init_pose)
            vive_init_matrix = self._pose7d_to_matrix(vive_init_pose)
            vive_init_matrix_inv = self._matrix_inverse(vive_init_matrix)
            
            self._transform_matrix = ee_init_matrix @ vive_init_matrix_inv

            self.logger.info("=" * 50)
            self.logger.info("Coordinate Transformation Computed")
            self.logger.info("=" * 50)
            self.logger.info(f"  EE init pose: {ee_init_pose}")
            self.logger.info(f"  Vive init pose: {vive_init_pose}")
            self.logger.info("  Formula: action_pose = ee_init * vive_init^-1 * vive_current")
            self.logger.info("=" * 50)

            self._log_reference_frame_info()

        except Exception as e:
            self.logger.error(f"Cannot connect to Vive Tracker: {e}")
            self._running = False
            self._is_connected = False
            raise

    def _wait_for_initial_pose(self, timeout: float = 5.0) -> np.ndarray | None:
        """Wait for initial pose from the active tracker.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Initial pose as [x, y, z, qw, qx, qy, qz] or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._data_lock:
                pose = self._latest_poses.get(self._active_tracker)
            
            if pose is not None:
                # Convert PoseData to 7D array [x, y, z, qw, qx, qy, qz]
                return np.array([
                    pose.position[0], pose.position[1], pose.position[2],
                    pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.rotation[3]
                ], dtype=np.float64)
            
            time.sleep(0.05)
        
        return None

    def calibrate(self) -> None:
        """Calibration is handled by pysurvive/lighthouse system."""
        self.logger.info("Vive Tracker uses lighthouse calibration, no runtime calibration needed")

    def configure(self) -> None:
        """No additional configuration needed."""
        pass

    def get_action(self) -> dict[str, Any]:
        """Get current tracker pose as action (transformed to robot frame).
        
        Data processing pipeline:
        1. Get latest pose from pysurvive
        2. Apply position jump filter (if enabled)
        3. Apply window filter
        4. Apply coordinate transformation: action = transform_matrix * vive_pose
        5. Return pose in robot frame
        """
        if not self.is_connected:
            raise RuntimeError("Vive Tracker is not connected")

        if self._transform_matrix is None:
            raise RuntimeError("Transformation matrix not computed. Call connect() first.")

        # Get pose for active tracker
        with self._data_lock:
            pose = self._latest_poses.get(self._active_tracker)

        if pose is None:
            # Return identity pose if no data available
            return {
                "tcp.x": 0.0,
                "tcp.y": 0.0,
                "tcp.z": 0.0,
                "tcp.qw": 1.0,
                "tcp.qx": 0.0,
                "tcp.qy": 0.0,
                "tcp.qz": 0.0,
            }

        # Extract position and quaternion from PoseData
        pos = np.array(pose.position, dtype=np.float32)
        quat = np.array(pose.rotation, dtype=np.float32)  # [qw, qx, qy, qz]

        # Apply position jump filter
        if self.config.enable_position_jump_filter and self._last_raw_pose is not None:
            pos_delta = np.linalg.norm(pos - self._last_raw_pose[:3])
            if pos_delta > self.config.position_jump_threshold:
                self._jump_filter_count += 1
                self.logger.warn(
                    f"Position jump detected: {pos_delta:.4f}m > "
                    f"{self.config.position_jump_threshold:.4f}m, filtering (count: {self._jump_filter_count})"
                )
                pos = self._last_raw_pose[:3].copy()

        self._last_raw_pose = np.concatenate([pos, quat])

        # Apply window filter
        filtered_pos, filtered_quat = self._filter_raw_pose(pos, quat)

        # Build current vive pose as 7D
        vive_current_pose = np.array([
            filtered_pos[0], filtered_pos[1], filtered_pos[2],
            filtered_quat[0], filtered_quat[1], filtered_quat[2], filtered_quat[3]
        ], dtype=np.float64)

        # Apply coordinate transformation
        # action_pose = transform_matrix * vive_current_pose
        vive_current_matrix = self._pose7d_to_matrix(vive_current_pose)
        action_matrix = self._transform_matrix @ vive_current_matrix
        action_pose = self._matrix_to_pose7d(action_matrix)

        return {
            "tcp.x": float(action_pose[0]),
            "tcp.y": float(action_pose[1]),
            "tcp.z": float(action_pose[2]),
            "tcp.qw": float(action_pose[3]),
            "tcp.qx": float(action_pose[4]),
            "tcp.qy": float(action_pose[5]),
            "tcp.qz": float(action_pose[6]),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """No haptic feedback for Vive Tracker."""
        pass

    def disconnect(self) -> None:
        """Disconnect and clean up all resources."""
        if not self._running:
            return

        self.logger.info("Stopping Vive Tracker pose tracking...")
        self._running = False

        # Wait for threads to finish
        if self._collector_thread:
            self._collector_thread.join(timeout=2.0)
        if self._processor_thread:
            self._processor_thread.join(timeout=2.0)

        # Print statistics before cleanup
        self.logger.info("Statistics:")
        for device_name, info in self._devices_info.items():
            self.logger.info(f"  - {device_name}: {info['updates']} updates")
        if self._jump_filter_count > 0:
            self.logger.info(f"  - Position jumps filtered: {self._jump_filter_count}")

        # Clean up all resources
        self._context = None
        self._pose_queue = queue.Queue(maxsize=100)
        self._is_connected = False
        self._transform_matrix = None
        self._collector_thread = None
        self._processor_thread = None
        self._latest_poses.clear()
        self._devices_info.clear()

        self.logger.info("✅ Vive Tracker disconnected")

    # ==================== Helper Methods ====================

    def _wait_for_devices(self, timeout: float, required_trackers: int) -> dict:
        """Wait for devices to be detected."""
        start_time = time.time()

        self.logger.info(
            f"Waiting for devices (timeout={timeout}s, required_trackers={required_trackers})..."
        )

        while time.time() - start_time < timeout:
            with self._data_lock:
                all_devices = list(self._devices_info.keys())

            lighthouses = [n for n in all_devices if n.startswith("LH")]
            trackers = [
                n for n in all_devices 
                if n.startswith("WM") or n.startswith("T2") or n.startswith("HMD")
            ]

            if len(trackers) >= required_trackers and len(lighthouses) >= 1:
                self.logger.info(
                    f"Required devices found: {len(lighthouses)} lighthouses, "
                    f"{len(trackers)} trackers"
                )
                break

            time.sleep(0.1)

        with self._data_lock:
            all_devices = list(self._devices_info.keys())

        lighthouses = [n for n in all_devices if n.startswith("LH")]
        trackers = [
            n for n in all_devices
            if n.startswith("WM") or n.startswith("T2") or n.startswith("HMD")
        ]
        others = [n for n in all_devices if n not in lighthouses and n not in trackers]

        result = {
            "lighthouses": lighthouses,
            "trackers": trackers,
            "others": others,
            "all": all_devices,
        }

        self.logger.info(
            f"Detection complete: {len(lighthouses)} lighthouses, {len(trackers)} trackers"
        )
        return result

    def _pose_collector(self) -> None:
        """Pose collection thread - continuously reads pose data from pysurvive."""
        self.logger.info("Pose collection thread started")

        # Get initial devices
        devices = list(self._context.Objects())
        if devices:
            self.logger.info(f"Detected {len(devices)} initial devices:")
            for device in devices:
                device_name = str(device.Name(), "utf-8")
                self.logger.info(f"  - {device_name}")
                self._devices_info[device_name] = {"updates": 0, "last_update": 0}

        # Continuously get poses
        while self._running and self._context.Running():
            updated = self._context.NextUpdated()
            if updated:
                device_name = str(updated.Name(), "utf-8")

                # Add new device if detected
                with self._data_lock:
                    if device_name not in self._devices_info:
                        self.logger.info(f"New device detected: {device_name}")
                        self._devices_info[device_name] = {
                            "updates": 0,
                            "last_update": 0,
                        }

                # Get pose data
                pose_obj = updated.Pose()
                pose_data = pose_obj[0]
                timestamp = pose_obj[1]

                # Extract raw pose from pysurvive (in lighthouse coordinate frame)
                # pysurvive quaternion is [w, x, y, z], we store as [qw, qx, qy, qz]
                position = [
                    pose_data.Pos[0],
                    pose_data.Pos[1],
                    pose_data.Pos[2],
                ]
                rotation = [
                    pose_data.Rot[0],  # qw
                    pose_data.Rot[1],  # qx
                    pose_data.Rot[2],  # qy
                    pose_data.Rot[3],  # qz
                ]

                pose = PoseData(device_name, timestamp, position, rotation)

                # Update device info
                with self._data_lock:
                    if device_name in self._devices_info:
                        self._devices_info[device_name]["updates"] += 1
                        self._devices_info[device_name]["last_update"] = time.time()

                # Put pose in queue
                try:
                    self._pose_queue.put_nowait(pose)
                except queue.Full:
                    try:
                        self._pose_queue.get_nowait()
                        self._pose_queue.put_nowait(pose)
                    except Exception:
                        pass

    def _pose_processor(self) -> None:
        """Pose processing thread - updates latest poses from queue."""
        self.logger.info("Pose processing thread started")

        while self._running:
            try:
                pose = self._pose_queue.get(timeout=0.1)
                with self._data_lock:
                    self._latest_poses[pose.device_name] = pose
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Cannot process pose data: {e}")

    def _log_reference_frame_info(self) -> None:
        """Log reference coordinate frame information."""
        lighthouses = [
            n for n in self._devices_info.keys() if n.startswith("LH")
        ]
        trackers = [
            n for n in self._devices_info.keys()
            if n.startswith("WM") or n.startswith("T2") or n.startswith("HMD")
        ]

        reference_lh = "LH0" if "LH0" in lighthouses else (
            lighthouses[0] if lighthouses else None
        )

        self.logger.info("=" * 50)
        self.logger.info("Reference Coordinate System Information")
        self.logger.info("=" * 50)
        self.logger.info(f"  Reference Lighthouse: {reference_lh} (origin)")
        self.logger.info(f"  All Lighthouses: {lighthouses}")
        self.logger.info(f"  All Trackers: {trackers}")
        self.logger.info(f"  Active Tracker: {self._active_tracker}")
        self.logger.info("  Coordinate System: Transformed to robot frame")
        self.logger.info("  Quaternion format: [qw, qx, qy, qz]")
        self.logger.info("=" * 50)

    def get_pose(self, device_name: str | None = None) -> PoseData | dict | None:
        """Get latest raw pose data (in lighthouse frame) for specified device.
        
        Note: This returns the raw pose without coordinate transformation.
        Use get_action() to get the transformed pose in robot frame.
        """
        if not self._running:
            return None if device_name else {}

        with self._data_lock:
            if device_name:
                return self._latest_poses.get(device_name)
            else:
                return self._latest_poses.copy()

    def get_devices(self) -> list:
        """Get list of all detected devices."""
        with self._data_lock:
            return list(self._devices_info.keys())

    def get_tracker_devices(self) -> list:
        """Get list of tracker device names only."""
        with self._data_lock:
            return [
                name for name in self._devices_info.keys()
                if name.startswith("WM") or name.startswith("T2") or name.startswith("HMD")
            ]
