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

"""Xense FlareGripper implementation for LeRobot."""

from xensegripper import XenseCamera, XenseGripper
from xensesdk import Sensor, call_service

from lerobot.robots.flexiv_rizon4.config_flare_gripper import FlareGripperConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger


class FlareGripper:
    config_class = FlareGripperConfig

    def __init__(
        self,
        config: FlareGripperConfig,
    ):
        self._config = config
        self._mac_addr = config.mac_addr
        self._cam_size = config.cam_size
        self._rectify_size = config.rectify_size
        self._sensor_output_type = config.sensor_output_type
        self._sensor_keys = config.sensor_keys
        self._gripper_max_pos = config.gripper_max_pos
        self._gripper_v_max = config.gripper_v_max
        self._gripper_f_max = config.gripper_f_max
        self._init_open = config.init_open
        self._logger = get_logger(f"FlareGripper-{self._mac_addr[:6]}")

        self._is_connected = False
        self._gripper: XenseGripper = None
        self._camera: XenseCamera = None
        self._sensors: dict[str, Sensor] = {}

        self._available_sensors: dict = {}
        self._logger.info(f"Scanning for sensors on device {self._mac_addr}...")
        try:
            sensor_sns = call_service(f"master_{self._mac_addr}", "scan_sensor_sn")
            if not sensor_sns:
                raise RuntimeError("No sensors found")
            self._logger.info(f"Found {len(sensor_sns)} sensor(s):")
            for sn, info in sensor_sns.items():
                self._logger.info(f"  - {sn}: {info}")
            self._available_sensors = sensor_sns
        except Exception as e:
            raise RuntimeError(f"Error scanning sensors: {e}") from e

    def connect(self) -> None:
        """Connect to the Flare Gripper."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._logger.info(f"Connecting to Flare Gripper: {self._mac_addr}")
        try:
            # connect sensors
            if self._available_sensors:
                for sn in self._available_sensors:
                    self._sensors[sn] = Sensor.create(
                        sn, mac_addr=self._mac_addr, rectify_size=self._rectify_size
                    )
                self._logger.info(f"✅ {len(self._sensors)} tactile sensors successfully connected.")
            else:
                self._logger.warn("No tactile sensors found")
        except Exception as e:
            raise RuntimeError(f"Error connecting to Flare Gripper tactile sensors: {e}") from e

        try:
            # connect camera
            camera_id = call_service(f"master_{self._mac_addr}", "list_camera")
            if camera_id is None:
                self._logger.warn("No camera found")
            else:
                self._camera = XenseCamera(
                    next(iter(camera_id.values())),
                    mac_addr=self._mac_addr,
                    frame_size=self._cam_size,
                )
                self._logger.info("✅ Camera successfully connected.")
        except Exception as e:
            raise RuntimeError(f"Error connecting to Flare Gripper camera: {e}") from e

        try:
            # connect gripper
            self._gripper = XenseGripper.create(self._mac_addr)
            if self._gripper is not None:
                self._logger.info("✅ Gripper successfully connected.")
            else:
                self._logger.warn("No gripper found")
        except Exception as e:
            raise RuntimeError(f"Error connecting to Flare Gripper gripper: {e}") from e

        self._is_connected = True
        self._logger.info("✅ Flare Gripper successfully connected.")

    def register_button_callback(self, event_type: str, callback):
        if self._is_connected and self._gripper is not None:
            self._gripper.register_button_callback(event_type, callback)
        else:
            raise DeviceNotConnectedError("Flare Gripper not connected, cannot register button callback")

    def get_sensor(self, id: int | str) -> Sensor | None:
        if isinstance(id, int):
            if id > len(self._sensors) - 1:
                self._logger.error(f"Sensor id {id} out of range")
                return None
            id = list(self._sensors.keys())[id]

        if id not in self._sensors:
            self._logger.error(f"Sensor {id} not found, available sensors: {list(self._sensors.keys())}")
            return None

        return self._sensors[id]

    def get_gripper_position(self) -> float:
        """
        Get current gripper position.

        Returns:
            Gripper position (0=closed, 1=fully open), or 0.0 if not available
        """
        if not self._is_connected or self._gripper is None:
            return 0.0

        try:
            status = self._gripper.get_gripper_status()
            if status is not None:
                raw_pos = float(status.get("position", 0.0))
                # Normalize to [0, 1] range
                if raw_pos < 0.0 or raw_pos > self._gripper_max_pos:
                    raw_pos = max(0.0, min(raw_pos, self._gripper_max_pos))
                normalized_pos = raw_pos / self._gripper_max_pos
                return max(0.0, min(1.0, normalized_pos))
            else:
                return 0.0
        except Exception:
            return 0.0

    def get_sensor_data(self) -> dict[str, any]:
        """
        Get sensor rectify data from all connected sensors.

        Returns:
            Dictionary mapping sensor_keys names (e.g., "left_tactile", "right_tactile")
            to their rectify data (numpy arrays).
        """
        if not self._is_connected:
            return {}

        sensor_data = {}
        for sn, sensor_obj in self._sensors.items():
            try:
                # Get the human-readable key name from sensor_keys mapping
                # If not found in mapping, use SN as fallback
                key_name = self._sensor_keys.get(sn, sn)

                rectify = sensor_obj.selectSensorInfo(Sensor.OutputType.Rectify)
                if rectify is not None:
                    # Convert BGR to RGB
                    if rectify.ndim == 3 and rectify.shape[2] == 3:
                        rectify = rectify[:, :, ::-1].copy()
                    sensor_data[key_name] = rectify
            except Exception as e:
                self._logger.debug(f"Failed to read sensor {sn} rectify data: {e}")

        return sensor_data

    def get_camera_frame(self):
        """
        Get current frame from wrist camera.

        Returns:
            RGB numpy array of shape (H, W, 3), or None if not available
        """
        if not self._is_connected or self._camera is None:
            return None

        try:
            ret, frame = self._camera.read()
            if ret and frame is not None:
                # Convert BGR to RGB
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = frame[:, :, ::-1].copy()
                return frame
        except Exception as e:
            self._logger.debug(f"Failed to read camera frame: {e}")

        return None

    def set_gripper_position(self, normalized_pos: float) -> None:
        """
        Set gripper position.

        Args:
            normalized_pos: Target position in [0, 1] range (0=closed, 1=fully open)
        """
        if not self._is_connected or self._gripper is None:
            raise DeviceNotConnectedError("Gripper not connected")

        if normalized_pos < 0.0 or normalized_pos > 1.0:
            raise ValueError(f"Gripper position must be between 0 and 1, got {normalized_pos}")

        target_pos = normalized_pos * self._gripper_max_pos
        self._gripper.set_position(target_pos, vmax=self._gripper_v_max, fmax=self._gripper_f_max)

    def disconnect(self) -> None:
        """Disconnect from the Flare Gripper."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Flare Gripper not connected")

        self._logger.info("Disconnecting Flare Gripper...")

        # Disconnect sensors
        for sn, sensor_obj in self._sensors.items():
            try:
                sensor_obj.release()
            except Exception as e:
                self._logger.debug(f"Error releasing sensor {sn}: {e}")
        self._sensors.clear()

        # Disconnect camera
        if self._camera is not None:
            try:
                self._camera = None
            except Exception as e:
                self._logger.debug(f"Error releasing camera: {e}")
            self._camera = None

        # Disconnect gripper
        if self._gripper is not None:
            try:
                self._gripper = None
            except Exception as e:
                self._logger.debug(f"Error releasing gripper: {e}")
            self._gripper = None

        self._is_connected = False
        self._logger.info("✅ Flare Gripper disconnected.")
