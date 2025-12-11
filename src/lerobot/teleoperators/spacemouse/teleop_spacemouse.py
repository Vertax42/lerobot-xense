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

This teleoperator provides 6-DoF control (translation + rotation)
and gripper control via buttons. It is designed to work with robots
that accept end-effector velocity or delta position commands.

Based on the 3Dconnexion SpaceMouse using the spnav library.
"""

import logging
import time
from multiprocessing.managers import SharedMemoryManager
from queue import Queue
from typing import Any

import numpy as np

from lerobot.teleoperators.spacemouse.config_spacemouse import SpacemouseConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class SpacemouseTeleop(Teleoperator):
    """
    3D Spacemouse teleoperator for end-effector control.

    This teleoperator reads 6-DoF motion data from a 3Dconnexion SpaceMouse
    and provides delta position and rotation commands suitable for end-effector
    teleoperation of robotic arms.

    The spacemouse provides:
    - 6-DoF motion: (dx, dy, dz, drx, dry, drz) - delta position and rotation
    - 2 buttons: typically used for gripper open/close or control events

    Output action format:
    - delta_x, delta_y, delta_z: delta position (always delta)
    - rx, ry, rz OR delta_rx, delta_ry, delta_rz: rotation (absolute or delta based on config)
    - gripper: gripper command (0=close, 1=stay, 2=open)
    """

    config_class = SpacemouseConfig
    name = "spacemouse"

    def __init__(self, config: SpacemouseConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._shm_manager: SharedMemoryManager | None = None
        self._spacemouse = None

        # Smoothing filter queue (moving average)
        self._motion_queue: Queue = Queue(self.config.filter_window_size)

        # State tracking
        self._last_update_time: float = 0.0
        self._enabled: bool = False

        # Accumulated rotation for absolute rotation mode (use_delta_rot=False)
        self._accumulated_rx: float = 0.0
        self._accumulated_ry: float = 0.0
        self._accumulated_rz: float = 0.0

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
        Return action features in the same format as KeyboardEndEffectorTeleop.

        Returns a dictionary with dtype, shape, and names for the action space.
        - Position is always delta: delta_x, delta_y, delta_z
        - Rotation depends on use_delta_rot config:
          - use_delta_rot=False: rx, ry, rz (absolute, accumulated)
          - use_delta_rot=True: delta_rx, delta_ry, delta_rz (relative)
        """
        if self.config.use_delta_rot:
            # Delta rotation mode
            rot_names = {"delta_rx": 3, "delta_ry": 4, "delta_rz": 5}
        else:
            # Absolute rotation mode
            rot_names = {"rx": 3, "ry": 4, "rz": 5}

        base_names = {
            "delta_x": 0,
            "delta_y": 1,
            "delta_z": 2,
            **rot_names,
        }

        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {**base_names, "gripper": 6},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": base_names,
            }

    @property
    def feedback_features(self) -> dict[str, type]:
        """Spacemouse doesn't support feedback."""
        return {}

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the 3D Spacemouse."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting to 3D Spacemouse...")

        # Lazy import to avoid requiring spnav when module is loaded but not used
        try:
            from lerobot.teleoperators.spacemouse.peripherals import Spacemouse
        except ImportError as e:
            raise ImportError(
                "spnav is required for Spacemouse teleoperator. "
                "Install it with: pip install spnav. "
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
            self._last_update_time = time.monotonic()

            # Reset accumulated rotation on connect
            self._accumulated_rx = 0.0
            self._accumulated_ry = 0.0
            self._accumulated_rz = 0.0

            logger.info(f"{self} connected successfully.")

        except Exception as e:
            if self._shm_manager is not None:
                self._shm_manager.shutdown()
                self._shm_manager = None
            raise RuntimeError(f"Failed to connect to Spacemouse: {e}") from e

    def calibrate(self) -> None:
        """Reset accumulated rotation to zero (re-calibration)."""
        self._accumulated_rx = 0.0
        self._accumulated_ry = 0.0
        self._accumulated_rz = 0.0
        logger.info("Spacemouse rotation reset to zero.")

    def configure(self) -> None:
        """No additional configuration needed."""
        pass

    def reset_rotation(self) -> None:
        """Reset accumulated rotation values to zero."""
        self._accumulated_rx = 0.0
        self._accumulated_ry = 0.0
        self._accumulated_rz = 0.0

    def _get_filtered_state(self) -> np.ndarray:
        """Get filtered spacemouse state with moving average."""
        raw_state = self._spacemouse.get_motion_state_transformed()

        # Apply additional deadzone filtering after transformation
        deadzone = self.config.deadzone
        positive_idx = raw_state >= deadzone
        negative_idx = raw_state <= -deadzone
        filtered_state = np.zeros_like(raw_state)
        filtered_state[positive_idx] = (raw_state[positive_idx] - deadzone) / (1 - deadzone)
        filtered_state[negative_idx] = (raw_state[negative_idx] + deadzone) / (1 - deadzone)

        # Apply axis inversion
        invert = np.array(self.config.invert_axes, dtype=np.float32)
        invert = np.where(invert, -1.0, 1.0)
        filtered_state = filtered_state * invert

        # Moving average filter
        if self._motion_queue.full():
            self._motion_queue.get()
        self._motion_queue.put(filtered_state)

        return np.mean(np.array(list(self._motion_queue.queue)), axis=0)

    def get_action(self) -> dict[str, Any]:
        """
        Get the current action from the Spacemouse.

        Returns a dictionary with:
        - delta_x, delta_y, delta_z: always delta position
        - rx, ry, rz OR delta_rx, delta_ry, delta_rz: based on use_delta_rot config
        - gripper: gripper command (0=close, 1=stay, 2=open)
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Calculate dt for velocity scaling
        current_time = time.monotonic()
        dt = current_time - self._last_update_time
        self._last_update_time = current_time

        # Get filtered motion state
        state = self._get_filtered_state()  # (6,) normalized [-1, 1]

        # Get button states
        button_left = self._spacemouse.is_button_pressed(0)
        button_right = self._spacemouse.is_button_pressed(1)

        # Compute delta position (always delta)
        delta_x = float(state[0] * self.config.pos_sensitivity * dt)
        delta_y = float(state[1] * self.config.pos_sensitivity * dt)
        delta_z = float(state[2] * self.config.pos_sensitivity * dt)

        # Compute rotation delta
        delta_rx = float(state[3] * self.config.ori_sensitivity * dt)
        delta_ry = float(state[4] * self.config.ori_sensitivity * dt)
        delta_rz = float(state[5] * self.config.ori_sensitivity * dt)

        # Compute gripper command (0=close, 1=stay, 2=open)
        if self.config.swap_gripper_buttons:
            button_open, button_close = button_left, button_right
        else:
            button_open, button_close = button_right, button_left

        if button_open and not button_close:
            gripper = 2.0  # Open
        elif button_close and not button_open:
            gripper = 0.0  # Close
        else:
            gripper = 1.0  # Stay

        # Check if any input is active
        motion_active = np.any(np.abs(state) > 0.01)
        self._enabled = motion_active or button_left or button_right

        # Build action dict
        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_delta_rot:
            # Delta rotation mode
            action_dict["delta_rx"] = delta_rx
            action_dict["delta_ry"] = delta_ry
            action_dict["delta_rz"] = delta_rz
        else:
            # Absolute rotation mode - accumulate rotation
            self._accumulated_rx += delta_rx
            self._accumulated_ry += delta_ry
            self._accumulated_rz += delta_rz
            action_dict["rx"] = self._accumulated_rx
            action_dict["ry"] = self._accumulated_ry
            action_dict["rz"] = self._accumulated_rz

        if self.config.use_gripper:
            action_dict["gripper"] = gripper

        return action_dict

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
                # Also reset rotation when both buttons held
                self.reset_rotation()
                logger.info("Both buttons held - triggering episode reset and rotation reset")
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

        logger.info("Disconnecting from Spacemouse...")

        if self._spacemouse is not None:
            self._spacemouse.stop(wait=True)
            self._spacemouse = None

        if self._shm_manager is not None:
            self._shm_manager.shutdown()
            self._shm_manager = None

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def __del__(self):
        """Cleanup on deletion."""
        if self._is_connected:
            try:
                self.disconnect()
            except Exception:
                pass
