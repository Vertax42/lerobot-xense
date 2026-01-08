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
Xense Flare Teleoperator for LeRobot.

This teleoperator uses Xense Flare gripper to provide:
- 6-DoF TCP pose from Vive Tracker
- Gripper position from encoder

It can be used to teleoperate robot arms like Flexiv Rizon4.
"""

from typing import Any

import numpy as np

from lerobot.utils.robot_utils import get_logger

from ..teleoperator import Teleoperator
from ..vive_tracker import ViveTrackerConfig, ViveTrackerTeleop
from .config_xense_flare_teleop import XenseFlareTeleopConfig

# Import Xense SDK components
try:
    from xensegripper import XenseGripper
    from xensesdk import call_service

    XENSE_SDK_AVAILABLE = True
except ImportError:
    XENSE_SDK_AVAILABLE = False


class XenseFlareTeleop(Teleoperator):
    """
    Xense Flare Teleoperator.

    Provides 6-DoF TCP pose from Vive Tracker and gripper position from encoder
    for teleoperating robot arms.

    Action features (9D pose + 1D gripper = 10D):
    - tcp.x, tcp.y, tcp.z: TCP position (meters)
    - tcp.r1-r6: TCP orientation (6D rotation representation)
    - gripper.pos: Gripper position (0=closed, 1=fully open)

    6D Rotation:
    - r1-r3: First column of rotation matrix
    - r4-r6: Second column of rotation matrix
    """

    config_class = XenseFlareTeleopConfig
    name = "xense_flare"

    def __init__(self, config: XenseFlareTeleopConfig):
        super().__init__(config)
        self.config = config

        if not XENSE_SDK_AVAILABLE:
            raise ImportError("Xense SDK not found. Please install xensesdk, xensegripper packages.")

        # Logger
        self.logger = get_logger(f"XenseFlareTeleop-{config.mac_addr[:6]}")

        # Components (initialized on connect)
        self._gripper: XenseGripper = None
        self._vive_tracker: ViveTrackerTeleop = None

        # Connection state
        self._is_connected = False

        # Vive Tracker config
        self._vive_config = ViveTrackerConfig(
            config_path=config.vive_config_path,
            lh_config=config.vive_lh_config,
            vive_to_ee_pos=config.vive_to_ee_pos,
            vive_to_ee_quat=config.vive_to_ee_quat,
            device_wait_timeout=config.device_wait_timeout,
            required_trackers=config.required_trackers,
            filter_window_size=config.filter_window_size,
            position_jump_threshold=config.position_jump_threshold,
            enable_position_jump_filter=config.enable_position_jump_filter,
        )

    @property
    def action_features(self) -> dict:
        """Action features: 9D pose (xyz + 6D rotation) + gripper position."""
        return {
            "tcp.x": float,
            "tcp.y": float,
            "tcp.z": float,
            "tcp.r1": float,
            "tcp.r2": float,
            "tcp.r3": float,
            "tcp.r4": float,
            "tcp.r5": float,
            "tcp.r6": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        """No feedback features for this teleoperator."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if the teleoperator is connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Xense Flare uses factory calibration, always True if connected."""
        return self.is_connected

    def connect(self, calibrate: bool = True, current_tcp_pose_quat: np.ndarray | None = None) -> None:
        """Connect to Xense Flare and start pose tracking.

        Args:
            calibrate: Unused, kept for API compatibility
            current_tcp_pose_quat: Current robot TCP pose [x, y, z, qw, qx, qy, qz].
                                   Required for computing the vive-to-robot transformation.
        """
        if self._is_connected:
            self.logger.warn("XenseFlareTeleop is already connected")
            return

        if current_tcp_pose_quat is None:
            raise ValueError(
                "current_tcp_pose_quat is required for XenseFlareTeleop. "
                "Please provide the current robot TCP pose [x, y, z, qw, qx, qy, qz]."
            )

        self.logger.info(f"Connecting to Xense Flare Teleop (MAC: {self.config.mac_addr})...")

        # Connect Vive Tracker
        self.logger.info("Connecting Vive Tracker...")
        self._vive_tracker = ViveTrackerTeleop(self._vive_config)
        try:
            init_pose = np.array(current_tcp_pose_quat, dtype=np.float64)
            self._vive_tracker.connect(current_tcp_pose_quat=init_pose)
            self.logger.info("  ✅ Vive Tracker connected")
        except Exception as e:
            self.logger.error(f"  ❌ Failed to connect Vive Tracker: {e}")
            raise RuntimeError(f"Vive Tracker failed to connect: {e}")

        # Initialize gripper
        if self.config.enable_gripper:
            self.logger.info("Initializing gripper...")
            try:
                self._gripper = XenseGripper.create(self.config.mac_addr)
                self.logger.info("  ✅ Gripper initialized")
            except Exception as e:
                self.logger.error(f"  ❌ Failed to initialize gripper: {e}")
                # Continue without gripper - some use cases may not need it
                self._gripper = None

        self._is_connected = True
        self.logger.info(f"✅ Xense Flare Teleop connected (MAC: {self.config.mac_addr})")

    def calibrate(self) -> None:
        """Calibration is handled by Vive Tracker lighthouse system."""
        self.logger.info("Xense Flare uses lighthouse calibration, no runtime calibration needed")

    def configure(self) -> None:
        """No additional configuration needed."""
        pass

    def get_action(self) -> dict[str, Any]:
        """
        Get the current action from Xense Flare.

        Returns a dictionary with:
        - tcp.x, tcp.y, tcp.z: TCP position (meters)
        - tcp.r1-r6: TCP orientation (6D rotation representation)
        - gripper.pos: Gripper position (0=closed, 1=fully open)
        """
        if not self.is_connected:
            raise RuntimeError("XenseFlareTeleop is not connected")

        # Get TCP pose from Vive Tracker (already in 6D rotation format)
        vive_action = self._vive_tracker.get_action()

        # Get gripper position (from cached value - fast)
        gripper_pos = self.get_gripper_position()

        return {
            "tcp.x": vive_action["tcp.x"],
            "tcp.y": vive_action["tcp.y"],
            "tcp.z": vive_action["tcp.z"],
            "tcp.r1": vive_action["tcp.r1"],
            "tcp.r2": vive_action["tcp.r2"],
            "tcp.r3": vive_action["tcp.r3"],
            "tcp.r4": vive_action["tcp.r4"],
            "tcp.r5": vive_action["tcp.r5"],
            "tcp.r6": vive_action["tcp.r6"],
            "gripper.pos": gripper_pos,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """No haptic feedback for Xense Flare."""
        pass

    def disconnect(self) -> None:
        """Disconnect from Xense Flare."""
        if not self._is_connected:
            self.logger.warn("XenseFlareTeleop is not connected, skipping disconnect")
            return

        self.logger.info("Disconnecting from Xense Flare Teleop...")

        # Disconnect Vive Tracker
        if self._vive_tracker is not None:
            try:
                self._vive_tracker.disconnect()
                self.logger.info("  Vive Tracker disconnected")
            except Exception as e:
                self.logger.error(f"  Failed to disconnect Vive Tracker: {e}")
            self._vive_tracker = None

        # Gripper doesn't need explicit release
        self._gripper = None

        self._is_connected = False
        self.logger.info("✅ Xense Flare Teleop disconnected")

    def get_gripper_position(self) -> float:
        """
        Get current gripper position.

        Returns:
            Gripper position (0=closed, 1=fully open), or 1.0 if not available
        """
        if not self._is_connected or self._gripper is None:
            return 1.0  # Default to fully open

        try:
            status = self._gripper.get_gripper_status()
            if status is not None:
                raw_pos = float(status.get("position", 0.0))
                # HACK: the gripper position is reversed, so we need to invert it
                raw_pos -= self.config.gripper_max_pos
                # Normalize to [0, 1] range
                if raw_pos < 0.0 or raw_pos > self.config.gripper_max_readout:
                    self.logger.warn(
                        f"Gripper pos {raw_pos:.2f} out of range [0, {self.config.gripper_max_readout}], clamping"
                    )
                    raw_pos = max(0.0, min(raw_pos, self.config.gripper_max_readout))
                normalized_pos = raw_pos / self.config.gripper_max_readout
                return max(0.0, min(1.0, normalized_pos))
            else:
                return 1.0
        except Exception:
            return 1.0

    def get_vive_tracker(self) -> ViveTrackerTeleop:
        """
        Get the Vive Tracker teleoperator instance.

        Returns:
            ViveTrackerTeleop instance
        """
        return self._vive_tracker

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
