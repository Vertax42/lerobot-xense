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
Pico4 VR teleoperator using xensevr_pc_service_sdk.

This teleoperator provides:
- Left/Right controller poses (position + quaternion)
- Headset pose
- Trigger and grip inputs for both controllers
"""

import logging
import time
from typing import Any

import numpy as np

from lerobot.teleoperators.pico4.config_pico4 import Pico4Config
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class Pico4(Teleoperator):
    """
    Pico4 VR teleoperator using XenseVR PC Service SDK.

    This teleoperator reads pose data from Pico4 VR controllers and headset,
    as well as trigger and grip inputs. It can be used to control robots
    in teleoperation scenarios.

    The SDK provides:
    - Left/Right controller poses (x, y, z, qx, qy, qz, qw)
    - Headset pose (x, y, z, qx, qy, qz, qw)
    - Trigger values (0-1) for both controllers
    - Grip values (0-1) for both controllers
    """

    config_class = Pico4Config
    name = "pico4"

    def __init__(self, config: Pico4Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._xrt = None

        # Calibration data
        self._left_calib_pos: np.ndarray | None = None
        self._left_calib_quat_inv: np.ndarray | None = None
        self._right_calib_pos: np.ndarray | None = None
        self._right_calib_quat_inv: np.ndarray | None = None
        self._headset_calib_pos: np.ndarray | None = None
        self._headset_calib_quat_inv: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Check if the teleoperator is calibrated."""
        # TODO: Implement calibration check when calibration is enabled
        return True

    @property
    def action_features(self) -> dict[str, type]:
        """Return the action features provided by this teleoperator."""
        features = {}
        if self.config.use_left_controller:
            features.update(
                {
                    "left.pos": np.ndarray,  # shape (3,) - raw position
                    "left.quat": np.ndarray,  # shape (4,) - raw quaternion (xyzw)
                    "left.trigger": float,
                    "left.grip": float,
                    "left.enabled": bool,  # True if grip > threshold
                }
            )
        if self.config.use_right_controller:
            features.update(
                {
                    "right.pos": np.ndarray,  # shape (3,) - raw position
                    "right.quat": np.ndarray,  # shape (4,) - raw quaternion (xyzw)
                    "right.trigger": float,
                    "right.grip": float,
                    "right.enabled": bool,  # True if grip > threshold
                }
            )
        features.update(
            {
                "headset.pos": np.ndarray,  # shape (3,) - raw position
                "headset.quat": np.ndarray,  # shape (4,) - raw quaternion (xyzw)
            }
        )
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Return feedback features (not implemented for Pico4)."""
        return {}

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the Pico4 VR headset via XenseVR SDK."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting to Pico4 VR headset...")
        try:
            import xensevr_pc_service_sdk as xrt
        except ImportError as e:
            raise ImportError(
                "xensevr_pc_service_sdk is required for Pico4 teleoperator. "
                "Please install it according to your Pico4 SDK documentation."
            ) from e
        try:
            self._xrt = xrt.init()
            logger.info("XenseVR SDK initialized successfully.")
            time.sleep(1.0)  # Wait for SDK to stabilize
            self._is_connected = True
            logger.info(f"{self} connected.")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize XenseVR SDK: {e}") from e

    def calibrate(self) -> None:
        """Calibrate the teleoperator by capturing the current pose as the reference."""
        # TODO: Implement calibration when needed
        pass
        # if not self._is_connected:
        #     raise DeviceNotConnectedError(f"{self} is not connected.")
        #
        # print("\n" + "=" * 60)
        # print("  Pico4 Calibration")
        # print("=" * 60)
        # print("\nHold the controllers in your desired starting position.")
        # print("Press and hold both GRIP buttons to capture the calibration pose...")
        # print("=" * 60 + "\n")
        #
        # # Wait for user to press both grips
        # while True:
        #     left_grip = self._xrt.get_left_grip()
        #     right_grip = self._xrt.get_right_grip()
        #
        #     if left_grip > self.config.grip_threshold and right_grip > self.config.grip_threshold:
        #         break
        #     time.sleep(0.02)
        #
        # # Capture calibration poses
        # if self.config.use_left_controller:
        #     left_pose = self._xrt.get_left_controller_pose()
        #     self._left_calib_pos = np.array(left_pose[:3])
        #     # Store quaternion as xyzw for scipy compatibility
        #     self._left_calib_quat_inv = self._quat_inverse(
        #         np.array([left_pose[3], left_pose[4], left_pose[5], left_pose[6]])
        #     )
        #
        # if self.config.use_right_controller:
        #     right_pose = self._xrt.get_right_controller_pose()
        #     self._right_calib_pos = np.array(right_pose[:3])
        #     self._right_calib_quat_inv = self._quat_inverse(
        #         np.array([right_pose[3], right_pose[4], right_pose[5], right_pose[6]])
        #     )
        #
        # # Calibrate headset
        # headset_pose = self._xrt.get_headset_pose()
        # self._headset_calib_pos = np.array(headset_pose[:3])
        # self._headset_calib_quat_inv = self._quat_inverse(
        #     np.array([headset_pose[3], headset_pose[4], headset_pose[5], headset_pose[6]])
        # )
        #
        # print("\nCalibration complete!\n")

    def configure(self) -> None:
        """Configure the teleoperator (no additional configuration needed)."""
        pass

    def get_action(self) -> dict[str, Any]:
        """Get the current action from the Pico4 controllers."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = {}

        # Left controller
        if self.config.use_left_controller:
            left_pose = self._xrt.get_left_controller_pose()
            left_pos = np.array(left_pose[:3])
            left_quat = np.array([left_pose[3], left_pose[4], left_pose[5], left_pose[6]])  # xyzw

            # TODO: Apply calibration when enabled
            # pos_cal = self._quat_rotate_vector(self._left_calib_quat_inv, left_pos - self._left_calib_pos)
            # quat_cal = self._quat_multiply(self._left_calib_quat_inv, left_quat)

            left_trigger = self._xrt.get_left_trigger()
            left_grip = self._xrt.get_left_grip()

            action["left.pos"] = left_pos
            action["left.quat"] = left_quat
            action["left.trigger"] = float(left_trigger)
            action["left.grip"] = float(left_grip)
            action["left.enabled"] = left_grip > self.config.grip_threshold

        # Right controller
        if self.config.use_right_controller:
            right_pose = self._xrt.get_right_controller_pose()
            right_pos = np.array(right_pose[:3])
            right_quat = np.array([right_pose[3], right_pose[4], right_pose[5], right_pose[6]])  # xyzw

            # TODO: Apply calibration when enabled
            # pos_cal = self._quat_rotate_vector(self._right_calib_quat_inv, right_pos - self._right_calib_pos)
            # quat_cal = self._quat_multiply(self._right_calib_quat_inv, right_quat)

            right_trigger = self._xrt.get_right_trigger()
            right_grip = self._xrt.get_right_grip()

            action["right.pos"] = right_pos
            action["right.quat"] = right_quat
            action["right.trigger"] = float(right_trigger)
            action["right.grip"] = float(right_grip)
            action["right.enabled"] = right_grip > self.config.grip_threshold

        # Headset
        headset_pose = self._xrt.get_headset_pose()
        headset_pos = np.array(headset_pose[:3])
        headset_quat = np.array([headset_pose[3], headset_pose[4], headset_pose[5], headset_pose[6]])  # xyzw

        # TODO: Apply calibration when enabled
        # headset_pos_cal = self._quat_rotate_vector(
        #     self._headset_calib_quat_inv, headset_pos - self._headset_calib_pos
        # )
        # headset_quat_cal = self._quat_multiply(self._headset_calib_quat_inv, headset_quat)

        action["headset.pos"] = headset_pos
        action["headset.quat"] = headset_quat

        return action

    def get_key_value_by_name(self, name: str) -> float:
        """Returns the trigger/grip value by name (float).
        Valid names: "left_trigger", "right_trigger", "left_grip", "right_grip".
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if name == "left_trigger":
            return self._xrt.get_left_trigger()
        elif name == "right_trigger":
            return self._xrt.get_right_trigger()
        elif name == "left_grip":
            return self._xrt.get_left_grip()
        elif name == "right_grip":
            return self._xrt.get_right_grip()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'left_trigger', 'right_trigger', 'left_grip', 'right_grip'."
            )

    def get_button_state_by_name(self, name: str) -> bool:
        """Returns the button state by name (bool).
        Valid names: "A", "B", "X", "Y",
                      "left_menu_button", "right_menu_button",
                      "left_axis_click", "right_axis_click"
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if name == "A":
            return self._xrt.get_A_button()
        elif name == "B":
            return self._xrt.get_B_button()
        elif name == "X":
            return self._xrt.get_X_button()
        elif name == "Y":
            return self._xrt.get_Y_button()
        elif name == "left_menu_button":
            return self._xrt.get_left_menu_button()
        elif name == "right_menu_button":
            return self._xrt.get_right_menu_button()
        elif name == "left_axis_click":
            return self._xrt.get_left_axis_click()
        elif name == "right_axis_click":
            return self._xrt.get_right_axis_click()
        else:
            raise ValueError(
                f"Invalid name: {name}. Valid names are: 'A', 'B', 'X', 'Y', "
                "'left_menu_button', 'right_menu_button', 'left_axis_click', 'right_axis_click'."
            )

    def get_timestamp_ns(self) -> int:
        """Returns the current timestamp in nanoseconds (int)."""
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        return self._xrt.get_time_stamp_ns()

    def get_hand_tracking_state(self, hand: str) -> np.ndarray | None:
        """Returns the hand tracking state for the specified hand.
        Valid hands: "left", "right".
        State is a 27 x 7 numpy array, where each row is [x, y, z, qx, qy, qz, qw] for each joint.
        Returns None if hand tracking is inactive (low quality).
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if hand.lower() == "left":
            if not self._xrt.get_left_hand_is_active():
                logger.warning("Left hand data is not active.")
                return None
            return self._xrt.get_left_hand_tracking_state()
        elif hand.lower() == "right":
            if not self._xrt.get_right_hand_is_active():
                logger.warning("Right hand data is not active.")
                return None
            return self._xrt.get_right_hand_tracking_state()
        else:
            raise ValueError(f"Invalid hand: {hand}. Valid hands are: 'left', 'right'.")

    def get_joystick_state(self, controller: str) -> list[float]:
        """Returns the joystick state for the specified controller.
        Valid controllers: "left", "right".
        State is a list with shape (2) representing [x, y] for each joystick.
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if controller.lower() == "left":
            return self._xrt.get_left_axis()
        elif controller.lower() == "right":
            return self._xrt.get_right_axis()
        else:
            raise ValueError(
                f"Invalid controller: {controller}. Valid controllers are: 'left', 'right'."
            )

    def get_motion_tracker_data(self) -> dict:
        """Returns a dictionary of motion tracker data, where the keys are the tracker serial numbers.
        Each value is a dictionary containing the pose, velocity, and acceleration of the tracker.
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        num_motion_data = self._xrt.num_motion_data_available()
        if num_motion_data == 0:
            logger.warning("Motion tracker data is not available.")
            return {}

        poses = self._xrt.get_motion_tracker_pose()
        velocities = self._xrt.get_motion_tracker_velocity()
        accelerations = self._xrt.get_motion_tracker_acceleration()
        serial_numbers = self._xrt.get_motion_tracker_serial_numbers()

        tracker_data = {}
        for i in range(num_motion_data):
            serial = serial_numbers[i]
            tracker_data[serial] = {
                "pose": poses[i],
                "velocity": velocities[i],
                "acceleration": accelerations[i],
            }

        return tracker_data

    def get_body_tracking_data(self) -> dict | None:
        """Returns complete body tracking data or None if unavailable.

        Returns:
            Dict with keys: 'poses', 'velocities', 'accelerations', 'imu_timestamps', 'body_timestamp'
            - poses: (24, 7) array [x,y,z,qx,qy,qz,qw] for each joint
            - velocities: (24, 6) array [vx,vy,vz,wx,wy,wz] for each joint
            - accelerations: (24, 6) array [ax,ay,az,wax,way,waz] for each joint
        """
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._xrt.is_body_data_available():
            logger.warning("Body tracking data is not available.")
            return None

        return {
            "poses": self._xrt.get_body_joints_pose(),
            "velocities": self._xrt.get_body_joints_velocity(),
            "accelerations": self._xrt.get_body_joints_acceleration(),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to the teleoperator (not implemented for Pico4)."""
        # Haptic feedback could be implemented here if the SDK supports it
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        raise NotImplementedError("Feedback is not implemented for Pico4 teleoperator.")

    def disconnect(self) -> None:
        """Disconnect from the Pico4 VR headset."""
        if not self._is_connected or self._xrt is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Closing XenseVR SDK...")
        try:
            self._xrt.close()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to close XenseVR SDK: {e}") from e
        finally:
            self._is_connected = False
            self._xrt = None
            logger.info(f"{self} disconnected.")
