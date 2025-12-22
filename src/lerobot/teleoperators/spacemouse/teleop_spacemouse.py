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
3D Spacemouse teleoperator for end-effector control.

This teleoperator provides 6-DoF absolute pose control (translation + rotation)
and gripper control via buttons. It outputs accumulated target_pose_6d that can
be directly sent to a Cartesian controller (e.g., Arx5CartesianController).

The output format matches ARX5 SDK's spacemouse_teleop.py example:
- target_pose_6d: [x, y, z, roll, pitch, yaw] - absolute EEF pose
- gripper_pos: absolute gripper position in meters

Based on the 3Dconnexion SpaceMouse using the spnav library.
"""

import time
from multiprocessing.managers import SharedMemoryManager
from queue import Queue
from typing import Any

import numpy as np
import spdlog

from lerobot.teleoperators.spacemouse.config_spacemouse import SpacemouseConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import euler_to_quaternion, normalize_quaternion


class SpacemouseTeleop(Teleoperator):
    """
    3D Spacemouse teleoperator for end-effector control.

    This teleoperator reads 6-DoF motion data from a 3Dconnexion SpaceMouse
    and maintains an accumulated target_pose_6d suitable for Cartesian control
    of robotic arms.

    The spacemouse provides:
    - 6-DoF motion: (dx, dy, dz, drx, dry, drz) - delta position and rotation
    - 2 buttons: typically used for gripper open/close or control events

    Output action format (matches ARX5 SDK spacemouse_teleop.py):
    - x, y, z, roll, pitch, yaw: absolute EEF target pose (accumulated)
    - gripper_pos: absolute gripper position in meters

    Usage:
    1. Call set_target_pose() to initialize with robot's current EEF pose
    2. Call get_action() to get updated target_pose_6d based on spacemouse input
    """

    config_class = SpacemouseConfig
    name = "spacemouse"

    def __init__(self, config: SpacemouseConfig):
        super().__init__(config)
        self.config = config
        self.logger = spdlog.ConsoleLogger("SpacemouseTeleop")
        self._is_connected = False
        self._shm_manager: SharedMemoryManager | None = None
        self._spacemouse = None
        self._start_pose_6d: np.ndarray = np.zeros(6, dtype=np.float32)
        self._start_gripper_pos: float = 0.0
        # Smoothing filter queue (moving average)
        self._motion_queue: Queue = Queue(self.config.filter_window_size)

        # State tracking
        self._enabled: bool = False

        # Event tracking for teleop events
        self._both_buttons_pressed_time: float | None = None
        self._reset_triggered: bool = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Spacemouse doesn't require calibration."""
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
        """Spacemouse doesn't support feedback."""
        return {}

    def connect(self, calibrate: bool = True, current_tcp_pose_euler: np.ndarray = np.zeros(7, dtype=np.float32)) -> None:
        """Connect to the 3D Spacemouse."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.logger.info("Connecting to 3D Spacemouse...")

        # Lazy import to avoid requiring spnav when module is loaded but not used
        try:
            from lerobot.teleoperators.spacemouse.peripherals import Spacemouse
        except ImportError as e:
            raise ImportError(
                "spnav is required for Spacemouse teleoperator."
                "Install it with: pip install spnav."
                "Also ensure spacenavd is installed and running: sudo apt install spacenavd"
            ) from e

        try:
            self._shm_manager = SharedMemoryManager()
            self._shm_manager.start()

            self._spacemouse = Spacemouse(
                shm_manager=self._shm_manager,
                deadzone=self.config.deadzone,
                max_value=self.config.max_value,
                frequency=self.config.frequency,
            )
            self._spacemouse.start(wait=True)

            self._is_connected = True

            # Set target pose on connect and save initial pose for reset
            self._target_pose_6d = current_tcp_pose_euler[:6].copy()
            self._target_gripper_pos = current_tcp_pose_euler[6]
            # Save initial pose for reset functionality
            self._start_pose_6d = current_tcp_pose_euler[:6].copy()
            self._start_gripper_pos = current_tcp_pose_euler[6]

            self.logger.info(f"{self} connected successfully.")

        except Exception as e:
            if self._shm_manager is not None:
                self._shm_manager.shutdown()
                self._shm_manager = None
            raise RuntimeError(f"Failed to connect to Spacemouse: {e}") from e

    def calibrate(self) -> None:
        """No calibration needed for spacemouse."""
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
        self._target_pose_6d = np.array(pose_6d, dtype=np.float32).copy()
        self._target_gripper_pos = float(gripper_pos)
        self.logger.info(f"Reset target pose to: {pose_6d}, gripper: {gripper_pos}")

    def _get_filtered_state(self) -> np.ndarray:
        """Get filtered spacemouse state with moving average."""
        raw_state = self._spacemouse.get_motion_state_transformed()

        # Apply additional deadzone filtering after transformation
        positive_idx = raw_state >= self.config.deadzone
        negative_idx = raw_state <= -self.config.deadzone
        filtered_state = np.zeros_like(raw_state)
        filtered_state[positive_idx] = (raw_state[positive_idx] - self.config.deadzone) / (1 - self.config.deadzone)
        filtered_state[negative_idx] = (raw_state[negative_idx] + self.config.deadzone) / (1 - self.config.deadzone)

        # Apply axis inversion
        invert = np.array(self.config.invert_axes, dtype=np.float32)
        invert = np.where(invert, -1.0, 1.0)
        filtered_state = filtered_state * invert

        # Moving average filter Use public method of queue to avoid race condition
        if self._motion_queue.full():
            self._motion_queue.get()
        self._motion_queue.put(filtered_state)

        return np.mean(np.array(list(self._motion_queue.queue)), axis=0)

    def get_action(self) -> dict[str, Any]:
        """
        Get the current target pose from the Spacemouse.

        Returns a dictionary with absolute EEF pose (matching ARX5 SDK format):
        - x, y, z: absolute EEF position (meters)
        - roll, pitch, yaw: absolute EEF orientation (radians)
        - gripper_pos: absolute gripper position (meters)
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Use fixed control_dt for consistent velocity scaling
        # This should match the external control loop period (e.g., 1/fps)
        dt = self.config.control_dt

        # Get filtered motion state
        state = self._get_filtered_state()  # (6,) normalized [-1, 1]

        # Get button states
        button_left = self._spacemouse.is_button_pressed(0)
        button_right = self._spacemouse.is_button_pressed(1)

        # Compute gripper command based on buttons
        if self.config.swap_gripper_buttons:
            button_open, button_close = button_left, button_right
        else:
            button_open, button_close = button_right, button_left

        if button_open and not button_close:
            gripper_cmd = 1  # Open
        elif button_close and not button_open:
            gripper_cmd = -1  # Close
        else:
            gripper_cmd = 0  # Stay

        # Update target pose with increments (matching ARX5 SDK spacemouse_teleop.py)
        # Position: target_pose_6d[:3] += state[:3] * pos_speed * dt
        self._target_pose_6d[:3] += state[:3] * self.config.pos_sensitivity * dt
        # Orientation: target_pose_6d[3:] += state[3:] * ori_speed * dt
        self._target_pose_6d[3:] += state[3:] * self.config.ori_sensitivity * dt

        # Update gripper position with clamping
        self._target_gripper_pos += gripper_cmd * self.config.gripper_speed * dt
        if self._target_gripper_pos >= self.config.gripper_width:
            self._target_gripper_pos = self.config.gripper_width
        elif self._target_gripper_pos <= 0:
            self._target_gripper_pos = 0

        # Check if any input is active
        motion_active = np.any(np.abs(state) > 0.01)
        self._enabled = motion_active or button_left or button_right

        # Return absolute pose dict
        return {
            "x": float(self._target_pose_6d[0]),
            "y": float(self._target_pose_6d[1]),
            "z": float(self._target_pose_6d[2]),
            "roll": float(self._target_pose_6d[3]),
            "pitch": float(self._target_pose_6d[4]),
            "yaw": float(self._target_pose_6d[5]),
            "gripper_pos": float(self._target_gripper_pos),
        }

    def get_target_pose_array(self) -> tuple[np.ndarray, float]:
        """
        Get the current target pose as numpy array (for direct use with ARX5 SDK).

        Returns:
            Tuple of (pose_6d, gripper_pos) where pose_6d is [x, y, z, roll, pitch, yaw]
        """
        return self._target_pose_6d.copy(), self._target_gripper_pos

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the spacemouse such as intervention status,
        episode termination, success indicators, etc. Mainly used for HIL-SERL integration.

        Spacemouse button mappings:
        - Any motion or button pressed = intervention active
        - Both buttons pressed together for 1s = reset/rerecord episode
        - Left button only = close gripper (normal operation)
        - Right button only = open gripper (normal operation)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self._is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Get current button states
        button_left = self._spacemouse.is_button_pressed(0)
        button_right = self._spacemouse.is_button_pressed(1)

        # Check if both buttons are pressed (for reset)
        both_pressed = button_left and button_right
        current_time = time.monotonic()

        terminate_episode = False
        rerecord_episode = False

        if both_pressed:
            if self._both_buttons_pressed_time is None:
                self._both_buttons_pressed_time = current_time
            elif (current_time - self._both_buttons_pressed_time) > 1.0 and not self._reset_triggered:
                # Both buttons held for 1 second - trigger reset
                terminate_episode = True
                rerecord_episode = True
                self._reset_triggered = True
                self.logger.info("Both buttons held - triggering episode reset")
        else:
            self._both_buttons_pressed_time = None
            self._reset_triggered = False

        return {
            TeleopEvents.IS_INTERVENTION: self._enabled,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: False,  # No success signal from spacemouse
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Spacemouse doesn't support feedback."""
        raise NotImplementedError("Feedback is not supported for Spacemouse teleoperator.")

    def disconnect(self) -> None:
        """Disconnect from the Spacemouse."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Disconnecting from Spacemouse...")

        if self._spacemouse is not None:
            self._spacemouse.stop(wait=True)
            self._spacemouse = None

        if self._shm_manager is not None:
            self._shm_manager.shutdown()
            self._shm_manager = None

        self._is_connected = False
        self.logger.info(f"{self} disconnected.")

    def convert_to_flexiv_action(self, spacemouse_action: dict[str, Any]) -> dict[str, Any]:
        """Convert spacemouse action (Euler angles) to Flexiv Rizon4 action (quaternion).

        This matches the behavior of spacemouse_teleop.py example:
        - Spacemouse maintains absolute pose in Euler angles [x, y, z, roll, pitch, yaw]
        - Convert to quaternion format [x, y, z, qw, qx, qy, qz] for Flexiv SDK

        Args:
            spacemouse_action: Dictionary with keys {x, y, z, roll, pitch, yaw, gripper_pos}
        Returns:
            Dictionary with keys {tcp.x, tcp.y, tcp.z, tcp.qw, tcp.qx, tcp.qy, tcp.qz, gripper.pos}
        """
        # Convert Euler angles to quaternion (matching spacemouse_teleop.py euler_to_quaternion)
        quat_tuple = euler_to_quaternion(
            spacemouse_action["roll"],
            spacemouse_action["pitch"],
            spacemouse_action["yaw"],
        )  # Returns (qw, qx, qy, qz)

        # Normalize quaternion to ensure unit length
        quat = normalize_quaternion(np.array(quat_tuple), input_format="wxyz")

        # Map to Flexiv action format (matching Flexiv SDK SendCartesianMotionForce signature)
        return {
            "tcp.x": spacemouse_action["x"],
            "tcp.y": spacemouse_action["y"],
            "tcp.z": spacemouse_action["z"],
            "tcp.qw": float(quat[0]),
            "tcp.qx": float(quat[1]),
            "tcp.qy": float(quat[2]),
            "tcp.qz": float(quat[3]),
            "gripper.pos": spacemouse_action["gripper_pos"],
        }

    def __del__(self):
        """Cleanup on deletion."""
        if self._is_connected:
            try:
                self.disconnect()
            except Exception:
                pass
