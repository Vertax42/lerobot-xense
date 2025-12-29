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
Xense Flare - Multi-Modal Data Collection Gripper for LeRobot.

Xense Flare is a data collection gripper designed for robot learning,
integrating multiple sensor modalities:

Data Sources:
- Vive Tracker: Provides 6DoF trajectory data (position + orientation)
- Wrist Camera: Provides visual information (RGB images)
- Tactile Sensors: Provides tactile perception (tactile images)
- Gripper Motor: Provides gripper motor state (position/velocity)

This implementation uses the Xense SDK for sensor and gripper control,
and Vive Tracker (pysurvive) for precise trajectory tracking.
"""

import time
from functools import cached_property
from typing import Any, Union

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

from ..robot import Robot
from .config_xense_flare import XenseFlareConfig, SensorOutputType

# Import Xense SDK components
try:
    from xensesdk import call_service, Sensor
    from xensegripper import XenseGripper, XenseCamera

    XENSE_SDK_AVAILABLE = True
except ImportError:
    XENSE_SDK_AVAILABLE = False

# Import Vive Tracker (required)
try:
    from lerobot.teleoperators.vive_tracker import ViveTrackerTeleop, ViveTrackerConfig

    VIVE_TRACKER_AVAILABLE = True
except ImportError:
    VIVE_TRACKER_AVAILABLE = False


class XenseFlare(Robot):
    """
    Xense Flare - Multi-Modal Data Collection Gripper.

    This is a pure observation device for data collection (similar to teach mode).
    It records multi-modal sensor data while being manually operated.

    Observation features (4 data sources):

    1. Trajectory Data (Vive Tracker):
       - tcp.x/y/z: End-effector position (m)
       - tcp.qw/qx/qy/qz: End-effector orientation (quaternion)

    2. Visual Data (Wrist Camera):
       - wrist_cam: Wrist RGB image (H, W, 3)

    3. Tactile Data (Tactile Sensors):
       - sensor_<sn>: Tactile sensor rectified image (H, W, 3)

    4. Gripper Data (Gripper Motor):
       - gripper.pos: Gripper position (float)

    Note: This device has no action features - it is operated manually
    and only records observations for data collection.
    """

    config_class = XenseFlareConfig
    name = "xense_flare"

    def __init__(self, config: XenseFlareConfig):
        super().__init__(config)
        self.config = config

        if not XENSE_SDK_AVAILABLE:
            raise ImportError(
                "Xense SDK not found. Please install xensesdk, xensegripper packages."
            )

        if config.enable_vive and not VIVE_TRACKER_AVAILABLE:
            raise ImportError(
                "Vive Tracker module not found. Please ensure lerobot.teleoperators.vive_tracker is available."
            )

        # Logger
        self.logger = get_logger(f"XenseFlare-{config.mac_addr[:6]}")

        # Components (initialized on connect)
        self._camera: "XenseCamera" = None
        self._gripper: "XenseGripper" = None
        self._sensors: dict[str, "Sensor"] = {}
        self._vive_tracker = None  # ViveTrackerTeleop, initialized on connect if enabled

        # Available sensors (discovered on connect)
        self._available_sensors: dict = {}

        # Connection state
        self._is_connected = False

        # Vive Tracker config (only used if enable_vive=True)
        self._vive_config = None
        if config.enable_vive:
            self._vive_config = ViveTrackerConfig(
                config_path=self.config.vive_config_path,
                lh_config=self.config.vive_lh_config,
                vive_to_ee_pos=self.config.vive_to_ee_pos,
                vive_to_ee_quat=self.config.vive_to_ee_quat,
            )
        
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        Observation features provided by this robot.

        Returns dict mapping feature names to their types:
        - gripper.pos: float
        - wrist_cam: (H, W, 3) image tuple
        - <sensor_key>: (H, W, 3) tactile image (key from config.sensor_keys or "sensor_{sn}")
        - tcp.x/y/z/qw/qx/qy/qz: float
        """
        features = {}

        # Gripper state
        if self.config.enable_gripper:
            features["gripper.pos"] = float

        # Wrist camera
        if self.config.enable_camera:
            h, w = self.config.cam_size[1], self.config.cam_size[0]
            features["wrist_cam"] = (h, w, 3)

        # Tactile sensors (dynamic based on discovered sensors)
        # Use config.get_sensor_key() to get custom key name or default "sensor_{sn}"
        if self.config.enable_sensor and self._available_sensors:
            h, w = self.config.rectify_size[1], self.config.rectify_size[0]
            for sn in self._available_sensors.keys():
                key = self.config.get_sensor_key(sn)
                features[key] = (h, w, 3)

        # Vive Tracker pose (always included - required component)
        features["tcp.x"] = float
        features["tcp.y"] = float
        features["tcp.z"] = float
        features["tcp.qw"] = float
        features["tcp.qx"] = float
        features["tcp.qy"] = float
        features["tcp.qz"] = float

        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        Action features accepted by this robot.

        Xense Flare is a pure observation device for data collection,
        it has no action features (manually operated).
        """
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if the robot is connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Xense Flare uses factory calibration, always True if connected."""
        return self.is_connected

    def connect(self) -> None:
        """Connect to the Xense Flare device."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.logger.info(f"Connecting to Xense Flare (MAC: {self.config.mac_addr})...")

        # Connect Vive Tracker (optional - disabled when mounted on robot arm)
        if self.config.enable_vive:
            self.logger.info("Connecting Vive Tracker...")
            self._vive_tracker = ViveTrackerTeleop(self._vive_config)
            try:
                init_pose = np.array(self.config.init_tcp_pose, dtype=np.float64)
                self._vive_tracker.connect(current_tcp_pose_quat=init_pose)
                self.logger.info("  ✅ Vive Tracker connected")
            except Exception as e:
                self.logger.error(f"  ❌ Failed to connect Vive Tracker: {e}")
                raise RuntimeError(f"Vive Tracker failed to connect: {e}")
        else:
            self.logger.info("Vive Tracker disabled (mounted on robot arm)")

        # Scan for available sensors
        self.logger.info("Scanning for available sensors...")
        self._available_sensors = self._scan_sensors()
        if self._available_sensors:
            self.logger.info(f"Found {len(self._available_sensors)} sensor(s):")
            for sn, info in self._available_sensors.items():
                self.logger.info(f"  - SN: {sn}, Info: {info}")
        else:
            self.logger.warn("No sensors detected on this device")
            raise RuntimeError("No sensors detected on this device")

        # Initialize sensors
        if self.config.enable_sensor and self._available_sensors:
            self.logger.info("Initializing sensors...")
            for sn in self._available_sensors.keys():
                try:
                    self._sensors[sn] = Sensor.create(
                        sn,
                        mac_addr=self.config.mac_addr,
                        rectify_size=self.config.rectify_size,
                    )
                    self.logger.info(f"  ✅ Sensor {sn} initialized")
                except Exception as e:
                    self.logger.error(f"  ❌ Failed to initialize sensor {sn}: {e}")
                    raise RuntimeError(f"Failed to initialize sensor {sn}: {e}")

        # Initialize camera
        if self.config.enable_camera:
            self.logger.info("Initializing wrist camera...")
            try:
                camera_id = call_service(
                    f"master_{self.config.mac_addr}", "list_camera"
                )
                if camera_id:
                    self._camera = XenseCamera(
                        next(iter(camera_id.values())),
                        mac_addr=self.config.mac_addr,
                        frame_size=self.config.cam_size,
                    )
                    self.logger.info("  ✅ Wrist camera initialized")
                else:
                    self.logger.warn("  ⚠️ No camera found")
            except Exception as e:
                self.logger.error(f"  ❌ Failed to initialize camera: {e}")

        # Initialize gripper
        if self.config.enable_gripper:
            self.logger.info("Initializing gripper...")
            try:
                self._gripper = XenseGripper.create(self.config.mac_addr)
                if self._gripper is not None:
                    if self.config.init_open:
                        self._gripper.set_position_sync(self.config.gripper_max_pos, vmax=self.config.gripper_v_max, fmax=self.config.gripper_f_max)
                        self.logger.info("  ✅ Gripper initialized (fully open)")
                    else:
                        self._gripper.set_position_sync(0.0, vmax=self.config.gripper_v_max, fmax=self.config.gripper_f_max)
                        self.logger.info("  ✅ Gripper initialized (fully closed)")
                else:
                    self.logger.warn("  ⚠️ Gripper creation returned None")
            except Exception as e:
                self.logger.error(f"  ❌ Failed to initialize gripper: {e}")

        self._is_connected = True
        self.logger.info(f"✅ Xense Flare connected (MAC: {self.config.mac_addr})")

    def disconnect(self) -> None:
        """Disconnect from the Xense Flare device."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self.logger.info("Disconnecting from Xense Flare...")

        # Release sensors
        for sn, sensor in self._sensors.items():
            try:
                sensor.release()
                self.logger.info(f"  Sensor {sn} released")
            except Exception as e:
                self.logger.error(f"  Failed to release sensor {sn}: {e}")
        self._sensors.clear()

        # Release camera
        if self._camera is not None:
            try:
                if hasattr(self._camera, "release"):
                    self._camera.release()
                elif hasattr(self._camera, "stop"):
                    self._camera.stop()
                elif hasattr(self._camera, "close"):
                    self._camera.close()
                self.logger.info("  Camera released")
            except Exception as e:
                self.logger.error(f"  Failed to release camera: {e}")
            self._camera = None

        # Gripper doesn't need explicit release
        self._gripper = None

        # Disconnect Vive Tracker
        if self._vive_tracker is not None:
            try:
                self._vive_tracker.disconnect()
                self.logger.info("  Vive Tracker disconnected")
            except Exception as e:
                self.logger.error(f"  Failed to disconnect Vive Tracker: {e}")
            self._vive_tracker = None

        self._is_connected = False
        self.logger.info("✅ Xense Flare disconnected")

    def calibrate(self) -> None:
        """Calibrate the robot (no-op for Xense Flare - manually operated device)."""
        pass

    def configure(self) -> None:
        """Configure the robot (no-op for Xense Flare)."""
        pass

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from the robot.

        Returns:
            dict containing:
            - gripper.pos: Gripper position (if enabled)
            - wrist_cam: Wrist camera image (if enabled)
            - sensor_<sn>: Tactile sensor rectify images (if enabled)
            - tcp.x/y/z/qw/qx/qy/qz: End-effector pose from Vive Tracker (if enable_vive=True)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        obs = {}

        # Get Vive Tracker pose (only if enabled)
        if self.config.enable_vive and self._vive_tracker is not None:
            try:
                action = self._vive_tracker.get_action()
                obs["tcp.x"] = action.get("tcp.x", 0.0)
                obs["tcp.y"] = action.get("tcp.y", 0.0)
                obs["tcp.z"] = action.get("tcp.z", 0.0)
                obs["tcp.qw"] = action.get("tcp.qw", 1.0)
                obs["tcp.qx"] = action.get("tcp.qx", 0.0)
                obs["tcp.qy"] = action.get("tcp.qy", 0.0)
                obs["tcp.qz"] = action.get("tcp.qz", 0.0)
            except Exception as e:
                self.logger.warn(f"Failed to get Vive Tracker pose: {e}")
                obs["tcp.x"] = 0.0
                obs["tcp.y"] = 0.0
                obs["tcp.z"] = 0.0
                obs["tcp.qw"] = 1.0
                obs["tcp.qx"] = 0.0
                obs["tcp.qy"] = 0.0
                obs["tcp.qz"] = 0.0

        # Get gripper state (normalized to [0, 1])
        if self.config.enable_gripper and self._gripper is not None:
            try:
                gripper_status = self._gripper.get_gripper_status()
                if gripper_status is not None:
                    raw_pos = float(gripper_status.get("position", 0.0))
                    # Normalize to [0, 1] range
                    normalized_pos = raw_pos / self.config.gripper_max_pos
                    # Clamp to [0, 1] in case of out-of-range values
                    obs["gripper.pos"] = max(0.0, min(1.0, normalized_pos))
                else:
                    raise ValueError("Failed to get gripper status")
            except Exception as e:
                raise ValueError(f"Failed to get gripper status: {e}")

        # Get wrist camera image (convert BGR to RGB)
        if self.config.enable_camera and self._camera is not None:
            try:
                ret, frame = self._camera.read()
                if ret and frame is not None:
                    # Convert BGR to RGB (SDK returns BGR format)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        frame = frame[:, :, ::-1].copy()
                    obs["wrist_cam"] = frame
                else:
                    # Return black image on failure
                    raise ValueError("Failed to read camera")
            except Exception as e:
                raise ValueError(f"Failed to read camera: {e}")

        # Get sensor data (rectify or difference based on config)
        # Use config.get_sensor_key() to get custom key name or default "sensor_{sn}"
        if self.config.enable_sensor and self._sensors:
            # Map config output type to SDK output type
            if self.config.sensor_output_type == SensorOutputType.RECTIFY:
                sdk_output_type = Sensor.OutputType.Rectify
            else:
                sdk_output_type = Sensor.OutputType.Difference

            for sn, sensor in self._sensors.items():
                key = self.config.get_sensor_key(sn)
                try:
                    data = sensor.selectSensorInfo(sdk_output_type)
                    if data is not None:
                        # Convert BGR to RGB (SDK returns BGR format)
                        if data.ndim == 3 and data.shape[2] == 3:
                            data = data[:, :, ::-1].copy()
                        obs[key] = data
                    else:
                        raise ValueError(f"Failed to read sensor {sn}")
                except Exception as e:
                    raise ValueError(f"Failed to read sensor {sn}: {e}")

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action to the robot.

        Supports gripper position control via gripper.pos key.
        Position is normalized [0, 1] and will be converted to raw position.

        Args:
            action: Dictionary with action keys. Supported:
                - "gripper.pos": Normalized gripper position [0, 1]
                  0 = fully closed, 1 = fully open

        Returns:
            The action that was sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # Gripper control
        if "gripper.pos" in action and self.config.enable_gripper and self._gripper is not None:
            normalized_pos = float(action["gripper.pos"])
            # Clamp to [0, 1]
            if normalized_pos < 0.0 or normalized_pos > 1.0:
                raise ValueError(f"Gripper position must be between 0 and 1, got {normalized_pos}")
            # Convert to raw position
            target_pos = normalized_pos * self.config.gripper_max_pos
            # Send to gripper with velocity and force limits from config    
            self._gripper.set_position(target_pos, vmax=self.config.gripper_v_max, fmax=self.config.gripper_f_max)

        return action

    def _scan_sensors(self) -> dict:
        """Scan for available sensors on the device."""
        try:
            sensor_sns = call_service(
                f"master_{self.config.mac_addr}", "scan_sensor_sn"
            )
            return sensor_sns if sensor_sns else {}
        except Exception as e:
            self.logger.error(f"Error scanning sensors: {e}")
            return {}

    def get_sensor(self, id: Union[int, str]) -> "Sensor":
        """
        Get a sensor by ID or serial number.

        Args:
            id: Sensor index (int) or serial number (str)

        Returns:
            Sensor object or None if not found
        """
        if isinstance(id, int):
            if id >= len(self._sensors):
                self.logger.error(f"Sensor index {id} out of range")
                return None
            id = list(self._sensors.keys())[id]

        if id not in self._sensors:
            self.logger.error(
                f"Sensor {id} not found, available: {list(self._sensors.keys())}"
            )
            return None

        return self._sensors[id]

    def get_vive_tracker(self) -> "ViveTrackerTeleop":
        """
        Get the Vive Tracker teleoperator instance.

        Returns:
            ViveTrackerTeleop instance
        """
        return self._vive_tracker

    def get_raw_pose(self, device_name: str = None) -> dict:
        """
        Get raw pose data from Vive Tracker.

        Args:
            device_name: Specific tracker name, or None for active tracker

        Returns:
            dict with pose data
        """
        if self._vive_tracker is None:
            return {}
        return self._vive_tracker.get_action()

    def register_button_callback(self, event_type: str, callback) -> None:
        """
        Register a callback for gripper button events.

        Args:
            event_type: Type of button event (PRESS, RELEASE, CLICK, DOUBLE_CLICK, LONG_PRESS)
            callback: Callback function (no arguments)
        """
        if self._gripper is not None:
            self._gripper.register_button_callback(event_type, callback)
        else:
            self.logger.warn("No gripper initialized, cannot register callback")

    def get_gripper_position(self) -> float:
        """
        Get current gripper position.

        Returns:
            Gripper position (0=closed, 1=fully open), or 0.0 if not available
        """
        if not self.is_connected or self._gripper is None:
            return 0.0
        try:
            status = self._gripper.get_gripper_status()
            if status is not None:
                return float(status.get("position", 0.0)) / self.config.gripper_max_pos
        except Exception:
            raise ValueError("Failed to get gripper position")

    def get_system_info(self) -> dict:
        """
        Get system information about the connected device.

        Returns:
            dict containing device info, cameras, sensors, gripper status
        """
        info = {
            "mac_addr": self.config.mac_addr,
            "connected": self._is_connected,
            "sensors": list(self._sensors.keys()),
            "camera": self._camera is not None,
            "gripper": self._gripper is not None,
            "vive_tracker": self._vive_tracker is not None,
        }
        return info
