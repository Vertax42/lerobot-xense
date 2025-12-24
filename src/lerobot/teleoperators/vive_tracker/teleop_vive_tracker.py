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
"""

import math
import queue
import sys
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import spdlog

from ..teleoperator import Teleoperator
from .config_vive_tracker import ViveTrackerConfig
from .vive_pose_utils import (
    euler_to_quaternion,
    matrix_to_xyz_quaternion,
    quaternion_to_euler,
    xyz_quaternion_to_matrix,
    xyz_rpy_to_matrix,
)


class PoseData:
    """Pose data structure for storing and formatting pose information."""

    def __init__(self, device_name: str, timestamp: float, 
                 position: list, rotation: list):
        self.device_name = device_name
        self.timestamp = timestamp
        self.position = position  # [x, y, z]
        self.rotation = rotation  # [qx, qy, qz, qw] quaternion

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
    
    Output format: [x, y, z, qw, qx, qy, qz] (7D pose in robot frame)
    """

    config_class = ViveTrackerConfig
    name = "vive_tracker"

    def __init__(self, config: ViveTrackerConfig):
        super().__init__(config)
        self.config = config

        # Logger
        self.logger = spdlog.ConsoleLogger("ViveTrackerTeleop")

        # pysurvive context
        self._context = None
        self._is_connected = False

        # Threading
        self._running = False
        self._pose_queue = queue.Queue(maxsize=100)
        self._data_lock = threading.Lock()
        self._latest_poses: dict[str, PoseData] = {}
        self._devices_info: dict[str, dict] = {}

        # Threads
        self._collector_thread = None
        self._processor_thread = None
        self._device_monitor_thread = None

        # Active tracker
        self._active_tracker: str | None = None

        # Filter state
        self._filter_queue: deque = deque(maxlen=max(1, config.filter_window_size))
        self._last_raw_pose: np.ndarray | None = None

        # Coordinate transform matrices (computed from config offsets)
        self._position_offset = np.array(config.position_offset)
        self._rotation_offset_rad = np.radians(config.rotation_offset_deg)

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
        return self._is_connected and self._context is not None

    @property
    def is_calibrated(self) -> bool:
        """Vive Tracker uses lighthouse calibration, always True if connected."""
        return self.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to Vive Tracker via pysurvive."""
        if self._is_connected:
            self.logger.warn("Vive Tracker is already connected")
            return

        try:
            # Import pysurvive
            try:
                import pysurvive
                self._pysurvive = pysurvive
            except ImportError:
                raise ImportError(
                    "pysurvive library not found. Please install it: "
                    "pip install pysurvive or build from source"
                )

            self.logger.info("Initializing pysurvive...")

            # Build pysurvive parameters
            survive_args = sys.argv[:1]  # Keep program name

            if self.config.config_path:
                survive_args.extend(["--config", self.config.config_path])

            if self.config.lh_config:
                survive_args.extend(["--lh", self.config.lh_config])

            # Initialize pysurvive context
            self._context = pysurvive.SimpleContext(survive_args)
            if not self._context:
                raise RuntimeError("Cannot initialize pysurvive context")

            self.logger.info("✅ Pysurvive context initialized successfully")

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

            self._device_monitor_thread = threading.Thread(
                target=self._device_monitor, daemon=True
            )
            self._device_monitor_thread.start()

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
            self._log_reference_frame_info()

        except Exception as e:
            self.logger.error(f"Cannot connect to Vive Tracker: {e}")
            self._running = False
            self._is_connected = False
            raise

    def calibrate(self) -> None:
        """Calibration is handled by pysurvive/lighthouse system."""
        self.logger.info("Vive Tracker uses lighthouse calibration, no runtime calibration needed")

    def configure(self) -> None:
        """Apply configuration settings."""
        self._position_offset = np.array(self.config.position_offset)
        self._rotation_offset_rad = np.radians(self.config.rotation_offset_deg)
        self._filter_queue.clear()
        self._last_raw_pose = None
        self.logger.info("Configuration applied")

    def get_action(self) -> dict[str, Any]:
        """Get current tracker pose as action."""
        if not self.is_connected:
            raise RuntimeError("Vive Tracker is not connected")

        # Get pose for active tracker
        with self._data_lock:
            pose = self._latest_poses.get(self._active_tracker)

        if pose is None:
            # Return zeros if no pose available
            return {
                "tcp.x": 0.0,
                "tcp.y": 0.0,
                "tcp.z": 0.0,
                "tcp.qw": 1.0,
                "tcp.qx": 0.0,
                "tcp.qy": 0.0,
                "tcp.qz": 0.0,
            }

        # Extract position and quaternion
        x, y, z = pose.position
        qx, qy, qz, qw = pose.rotation

        # Apply coordinate transform if configured
        if np.any(self._position_offset != 0) or np.any(self._rotation_offset_rad != 0):
            x, y, z, qw, qx, qy, qz = self._apply_transform(
                x, y, z, qw, qx, qy, qz
            )

        # Apply sensitivity scaling
        x *= self.config.pos_sensitivity
        y *= self.config.pos_sensitivity
        z *= self.config.pos_sensitivity

        # Create raw pose array for filtering
        raw_pose = np.array([x, y, z, qw, qx, qy, qz])

        # Apply position jump filter
        if self.config.enable_position_jump_filter and self._last_raw_pose is not None:
            pos_delta = np.linalg.norm(raw_pose[:3] - self._last_raw_pose[:3])
            if pos_delta > self.config.position_jump_threshold:
                self.logger.warn(
                    f"Position jump detected: {pos_delta:.4f}m > "
                    f"{self.config.position_jump_threshold:.4f}m, filtering"
                )
                raw_pose = self._last_raw_pose.copy()

        self._last_raw_pose = raw_pose.copy()

        # Apply smoothing filter
        filtered_pose = self._filter_pose(raw_pose)

        return {
            "tcp.x": float(filtered_pose[0]),
            "tcp.y": float(filtered_pose[1]),
            "tcp.z": float(filtered_pose[2]),
            "tcp.qw": float(filtered_pose[3]),
            "tcp.qx": float(filtered_pose[4]),
            "tcp.qy": float(filtered_pose[5]),
            "tcp.qz": float(filtered_pose[6]),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """No haptic feedback for Vive Tracker."""
        pass

    def disconnect(self) -> None:
        """Disconnect from Vive Tracker."""
        if not self._running:
            return

        self.logger.info("Stopping Vive Tracker pose tracking...")
        self._running = False

        # Wait for threads to finish
        if self._collector_thread:
            self._collector_thread.join(timeout=2.0)
        if self._processor_thread:
            self._processor_thread.join(timeout=2.0)
        if self._device_monitor_thread:
            self._device_monitor_thread.join(timeout=2.0)

        # Clean up resources
        self._context = None
        self._pose_queue = queue.Queue(maxsize=100)
        self._is_connected = False

        # Print statistics
        self.logger.info("Statistics:")
        for device_name, info in self._devices_info.items():
            self.logger.info(f"  - {device_name}: {info['updates']} updates")

        self.logger.info("✅ Vive Tracker disconnected")

    def _apply_transform(self, x: float, y: float, z: float,
                          qw: float, qx: float, qy: float, qz: float) -> tuple:
        """Apply coordinate transformation to pose."""
        # Convert to matrix
        pose_mat = xyz_quaternion_to_matrix(x, y, z, qx, qy, qz, qw)

        # Apply rotation offset
        roll, pitch, yaw = self._rotation_offset_rad
        if roll != 0 or pitch != 0 or yaw != 0:
            rotation_mat = xyz_rpy_to_matrix(0, 0, 0, roll, pitch, yaw)
            pose_mat = np.dot(pose_mat, rotation_mat)

        # Apply position offset
        pose_mat[0, 3] += self._position_offset[0]
        pose_mat[1, 3] += self._position_offset[1]
        pose_mat[2, 3] += self._position_offset[2]

        # Convert back to position and quaternion
        x, y, z, qx, qy, qz, qw = matrix_to_xyz_quaternion(pose_mat)
        return x, y, z, qw, qx, qy, qz

    def _filter_pose(self, raw_pose: np.ndarray) -> np.ndarray:
        """Apply moving average filter to pose."""
        if self.config.filter_window_size <= 1:
            return raw_pose

        self._filter_queue.append(raw_pose)

        if len(self._filter_queue) == 1:
            return raw_pose

        # Average position
        positions = np.array([p[:3] for p in self._filter_queue])
        avg_pos = np.mean(positions, axis=0)

        # For orientation, use the latest (SLERP would be better but more complex)
        avg_quat = raw_pose[3:7]

        return np.concatenate([avg_pos, avg_quat])

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

    def _device_monitor(self) -> None:
        """Device monitoring thread."""
        self.logger.info("Device monitoring thread started")

        self._update_device_list()

        while self._running and self._context.Running():
            self._update_device_list()
            time.sleep(1.0)

    def _update_device_list(self) -> None:
        """Update device list from pysurvive."""
        try:
            devices = list(self._context.Objects())
            with self._data_lock:
                for device in devices:
                    device_name = str(device.Name(), "utf-8")
                    if device_name not in self._devices_info:
                        self.logger.info(f"New device detected: {device_name}")
                        self._devices_info[device_name] = {
                            "updates": 0,
                            "last_update": 0,
                        }
        except Exception as e:
            self.logger.error(f"Cannot update device list: {e}")

    def _pose_collector(self) -> None:
        """Pose collection thread."""
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

                # Convert pose (pysurvive quaternion is [w,x,y,z], we use [x,y,z,w])
                origin_mat = xyz_quaternion_to_matrix(
                    pose_data.Pos[0],
                    pose_data.Pos[1],
                    pose_data.Pos[2],
                    pose_data.Rot[1],  # qx
                    pose_data.Rot[2],  # qy
                    pose_data.Rot[3],  # qz
                    pose_data.Rot[0],  # qw
                )

                # Apply gripper-specific transform (from XGripper project)
                # Initial rotation: -20 degrees around X axis
                initial_rotation = xyz_rpy_to_matrix(
                    0, 0, 0, -(20.0 / 180.0 * math.pi), 0, 0
                )
                # Alignment rotation: -90 deg X, -90 deg Y
                alignment_rotation = xyz_rpy_to_matrix(
                    0, 0, 0, -90 / 180 * math.pi, -90 / 180 * math.pi, 0
                )
                rotate_matrix = np.dot(initial_rotation, alignment_rotation)

                # Transform to gripper center
                transform_matrix = xyz_rpy_to_matrix(0.172, 0, -0.076, 0, 0, 0)

                # Calculate final transformation
                result_mat = np.matmul(
                    np.matmul(origin_mat, rotate_matrix), transform_matrix
                )

                # Extract position and quaternion
                x, y, z, qx, qy, qz, qw = matrix_to_xyz_quaternion(result_mat)

                position = [x, y, z]
                rotation = [qx, qy, qz, qw]

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
        """Pose processing thread."""
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
        self.logger.info("  Coordinate System: Right-handed")
        self.logger.info("=" * 50)

    def get_pose(self, device_name: str | None = None) -> PoseData | dict | None:
        """
        Get latest pose data for specified device.
        
        Args:
            device_name: Device name, if None returns all poses
            
        Returns:
            PoseData for specific device, or dict of all poses
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

