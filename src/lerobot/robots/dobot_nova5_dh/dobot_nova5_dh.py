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

"""Single right-arm Dobot Nova5 with a DH AG-95 gripper on built-in RS485.

The hardware defaults are the right arm extracted from ``bi_dobot_nova5_dh``.
Because this is a single-arm robot, its public action and observation keys are:

* Cartesian: ``tcp.{x,y,z,r1-r6}`` and ``gripper.pos``.
* Joint: ``joint_{1..6}.pos`` and ``gripper.pos``.
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.dobot_nova5.TCP_IP_Python_V4.dobot_api import (
    DobotApiDashboard,
    DobotApiFeedBack,
)
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    euler_to_quaternion,
    get_logger,
    quaternion_to_euler,
    quaternion_to_rotation_6d,
    rotation_6d_to_quaternion,
)

from .config_dobot_nova5_dh import ControlMode, DobotNova5DHConfig, ResetTarget
from .dh_gripper_integrated import DHGripperIntegrated

JOINT_DOF = 6
MM_PER_METER = 1000.0
_MODBUS_RETRIES = 3


class _DobotModbusRTU:
    """Adapt Dobot's Dashboard Modbus proxy to the integrated gripper protocol."""

    def __init__(
        self,
        robot: DobotApiDashboard,
        master_ip: str,
        master_port: int,
        slave_id: int,
        is_rtu: bool = True,
    ) -> None:
        self._robot = robot
        response = self._robot.ModbusCreate(master_ip, master_port, slave_id, is_rtu)
        error_id, values = self._parse(response)
        if error_id != 0 or not values:
            raise RuntimeError(
                f"ModbusCreate failed (error_id={error_id}): {response.strip()}"
            )
        self._index = int(values[0])

    def read_register(self, reg: int) -> int | None:
        for _ in range(_MODBUS_RETRIES):
            with contextlib.suppress(Exception):
                error_id, values = self._parse(
                    self._robot.GetHoldRegs(self._index, reg, 1)
                )
                if error_id == 0 and values:
                    return int(values[0])
        return None

    def write_register(self, reg: int, value: int) -> bool:
        value_string = "{" + str(value) + "}"
        for _ in range(_MODBUS_RETRIES):
            with contextlib.suppress(Exception):
                error_id, _ = self._parse(
                    self._robot.SetHoldRegs(self._index, reg, 1, value_string)
                )
                if error_id == 0:
                    return True
        return False

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._robot.ModbusClose(self._index)

    @staticmethod
    def _parse(response: str) -> tuple[int, list[str]]:
        match = re.match(r"\s*(-?\d+)\s*,\s*\{([^}]*)\}", response)
        if match is None:
            raise RuntimeError(f"Unexpected Dobot response: {response!r}")
        values = [value.strip() for value in match.group(2).split(",") if value.strip()]
        return int(match.group(1)), values


class _FeedState:
    def __init__(self) -> None:
        self.RobotMode = -1
        self.robotCurrentCommandID = -1
        self.MessageSize = -1
        self.DigitalInputs = -1
        self.DigitalOutputs = -1
        self.User = -1
        self.Tool = -1
        self.tcpPose = [0.0] * 6  # xyz in mm, rpy in degrees
        self.qActual = [0.0] * JOINT_DOF


class DobotNova5DH(Robot):
    """Control one Dobot Nova5 arm and its integrated-RS485 DH gripper."""

    config_class = DobotNova5DHConfig
    name = "dobot_nova5_dh"

    def __init__(self, config: DobotNova5DHConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger("DobotNova5DH")

        self._robot: DobotApiDashboard | None = None
        self._gripper_robot: DobotApiDashboard | None = None
        self._feed: DobotApiFeedBack | None = None
        self._feed_data = _FeedState()
        self._feed_lock = threading.Lock()
        self._feed_thread: threading.Thread | None = None

        self._is_connected = False
        self.rt_moving = False
        self._last_servo_period_s = 1.0 / float(config.control_frequency)
        self._cartesian_ik_servoj_until_s = 0.0
        self._last_cartesian_guard_log_s = 0.0
        self._last_cartesian_command_debug: dict[str, Any] = {}
        self._last_obs_timing: dict[str, float] = {}
        self._last_action_timing: dict[str, float] = {}

        self._action_worker_thread: threading.Thread | None = None
        self._action_worker_running = False
        self._action_worker_paused = False
        self._action_worker_busy = False
        self._action_worker_lock = threading.Lock()
        self._action_worker_cv = threading.Condition(self._action_worker_lock)
        self._pending_async_action: dict[str, Any] | None = None
        self._latest_sent_async_action: dict[str, Any] | None = None
        self._last_async_action_timing: dict[str, float] = {}
        self._async_action_drop_count = 0
        self._async_action_send_count = 0
        self._async_action_error_count = 0

        self._gripper: DHGripperIntegrated | None = None
        if config.use_gripper:
            if config.dh_gripper is None:
                raise ValueError("use_gripper=True requires a DH gripper configuration")
            self._gripper = DHGripperIntegrated(config.dh_gripper, name="right")
        self._gripper_connected = False
        self._gripper_key = "gripper.pos"

        if config.control_mode == ControlMode.JOINT_MOTION:
            self._joint_pos_keys = tuple(
                f"joint_{index}.pos" for index in range(1, JOINT_DOF + 1)
            )
            self._action_joint_keys = self._joint_pos_keys
        elif config.control_mode == ControlMode.CARTESIAN_MOTION:
            self._tcp_pose_keys = (
                "tcp.x",
                "tcp.y",
                "tcp.z",
                "tcp.r1",
                "tcp.r2",
                "tcp.r3",
                "tcp.r4",
                "tcp.r5",
                "tcp.r6",
            )
            self._action_tcp_pose_keys = self._tcp_pose_keys
        else:
            raise ValueError(f"Unsupported control_mode: {config.control_mode}")

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _action_ft(self) -> dict[str, type]:
        if self.config.control_mode == ControlMode.JOINT_MOTION:
            features = dict.fromkeys(self._action_joint_keys, float)
        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            features = dict.fromkeys(self._action_tcp_pose_keys, float)
        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")
        features[self._gripper_key] = float
        return features

    @property
    def _proprioception_ft(self) -> dict[str, type]:
        if self.config.control_mode == ControlMode.JOINT_MOTION:
            features = dict.fromkeys(self._joint_pos_keys, float)
        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            features = dict.fromkeys(self._tcp_pose_keys, float)
        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")
        features[self._gripper_key] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            camera_name: (
                self.config.cameras[camera_name].height,
                self.config.cameras[camera_name].width,
                3,
            )
            for camera_name in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._proprioception_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._action_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._robot is not None
            and all(camera.is_connected for camera in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    def calibrate(self) -> None:
        self.logger.info("Dobot Nova5 is factory calibrated; no runtime calibration is needed.")

    def configure(self) -> None:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        tool_index = int(self.config.tool_coordinate_index) if self.config.use_tool_coordinate else 0
        self._raise_if_dobot_error(
            self._robot, self._robot.Tool(tool_index), f"right Tool({tool_index})"
        )
        self._wait_for_feedback_tool_index(tool_index)
        self.logger.info(f"Right arm global tool coordinate set to Tool({tool_index})")

    def set_servo_period(self, period_s: float) -> None:
        self._last_servo_period_s = float(np.clip(period_s, 0.004, 0.2))

    def _wait_for_feedback_tool_index(
        self, tool_index: int, timeout_s: float = 1.0
    ) -> None:
        start_time = time.time()
        while time.time() - start_time <= timeout_s:
            if int(self._feed_data.Tool) == tool_index:
                return
            time.sleep(0.02)
        self.logger.warn(
            f"Right feedback Tool index did not update to {tool_index} within {timeout_s:.1f}s "
            f"(current={self._feed_data.Tool}); continuing with Dashboard Tool setting."
        )

    @staticmethod
    def _parse_dobot_response(response: str) -> tuple[int, list[str]]:
        if not isinstance(response, str):
            raise RuntimeError(f"Invalid Dobot response type: {type(response).__name__}")
        if "Not Tcp" in response:
            raise RuntimeError("Robot is not in TCP control mode")
        match = re.match(r"\s*(-?\d+)\s*,\s*\{([^}]*)\}", response)
        if match is None:
            raise RuntimeError(f"Could not parse Dobot response: {response!r}")
        values = [value.strip() for value in match.group(2).split(",") if value.strip()]
        return int(match.group(1)), values

    def _dobot_error_detail(self) -> str:
        if self._robot is None:
            return ""
        try:
            return self._robot.GetErrorID().strip()
        except Exception as exception:
            return f"failed to read GetErrorID: {exception}"

    def _dashboard_robot_mode(self) -> int | None:
        if self._robot is None:
            return None
        try:
            error_id, values = self._parse_dobot_response(self._robot.RobotMode())
            return int(float(values[0])) if error_id == 0 and values else None
        except (RuntimeError, ValueError):
            return None

    def _raise_if_dobot_error(
        self,
        robot: DobotApiDashboard | None,
        response: str,
        command_name: str,
    ) -> list[str]:
        error_id, values = self._parse_dobot_response(response)
        if error_id == 0:
            return values
        message = f"{command_name} failed with ErrorID {error_id}: {response.strip()}"
        error_detail = self._dobot_error_detail()
        if error_detail:
            message = f"{message}; GetErrorID: {error_detail}"
        raise RuntimeError(message)

    def _wait_for_first_feedback(self, timeout_s: float = 3.0) -> None:
        start_time = time.time()
        while time.time() - start_time <= timeout_s:
            if self._feed_data.MessageSize != -1 or self._feed_data.RobotMode != -1:
                return
            time.sleep(0.02)

    def _wait_until_not_error_mode(self, timeout_s: float = 10.0) -> None:
        start_time = time.time()
        while int(self._feed_data.RobotMode) == 9:
            if time.time() - start_time > timeout_s:
                raise TimeoutError(
                    "Right arm stayed in error mode after ClearError; "
                    f"GetErrorID: {self._dobot_error_detail()}"
                )
            time.sleep(0.1)

    def _wait_for_joint_target(
        self,
        target_joint_deg: list[float],
        description: str,
        tolerance_deg: float = 0.5,
        timeout_s: float = 60.0,
    ) -> None:
        target = np.asarray(target_joint_deg, dtype=np.float64)
        start_time = time.time()
        last_log_time = 0.0
        while True:
            robot_mode = int(self._feed_data.RobotMode)
            current_joint = np.asarray(self._feed_data.qActual, dtype=np.float64)
            max_abs_error = float(np.max(np.abs(current_joint - target)))
            if robot_mode == 9:
                raise RuntimeError(
                    f"Right {description} failed: robot entered error mode; "
                    f"GetErrorID: {self._dobot_error_detail()}"
                )
            if robot_mode == 5 and max_abs_error <= tolerance_deg:
                return
            now = time.time()
            if now - start_time > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for right {description}: "
                    f"max_joint_error={max_abs_error:.3f} deg, RobotMode={robot_mode}, "
                    f"target={target.tolist()}, current={current_joint.tolist()}"
                )
            if now - last_log_time >= 2.0:
                self.logger.info(
                    f"Waiting for right {description}: RobotMode={robot_mode}, "
                    f"max_joint_error={max_abs_error:.3f} deg"
                )
                last_log_time = now
            time.sleep(0.1)

    def _wait_for_command_id(
        self,
        command_id: int,
        description: str,
        timeout_s: float = 60.0,
    ) -> None:
        start_time = time.time()
        last_log_time = 0.0
        while True:
            robot_mode = int(self._feed_data.RobotMode)
            current_command_id = int(self._feed_data.robotCurrentCommandID)
            if robot_mode == 9:
                raise RuntimeError(
                    f"Right {description} failed while waiting for command {command_id}; "
                    f"GetErrorID: {self._dobot_error_detail()}"
                )
            if robot_mode == 5 and current_command_id == int(command_id):
                return
            now = time.time()
            if now - start_time > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for right {description}: target command ID={command_id}, "
                    f"current command ID={current_command_id}, RobotMode={robot_mode}"
                )
            if now - last_log_time >= 2.0:
                self.logger.info(
                    f"Waiting for right {description}: RobotMode={robot_mode}, "
                    f"CurrentCommandId={current_command_id}, target={command_id}"
                )
                last_log_time = now
            time.sleep(0.1)

    def _move_joint_movj(self, joint_degrees: list[float], description: str) -> None:
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        target = [float(value) for value in joint_degrees]
        self._raise_if_dobot_error(
            self._robot,
            self._robot.SpeedFactor(int(self.config.start_vel_scale)),
            "right SpeedFactor",
        )
        response = self._robot.MovJ(
            *target,
            1,
            v=int(self.config.start_vel_scale),
        )
        self.logger.info(f"Right {description} MovJ(joint) response: {response.strip()}")
        values = self._raise_if_dobot_error(self._robot, response, "right MovJ(joint)")
        if values:
            try:
                self._wait_for_command_id(int(float(values[0])), description)
                return
            except (ValueError, TypeError):
                self.logger.warn(
                    f"Right {description} command ID parse failed ({values}); "
                    "falling back to joint-error completion check."
                )
        self._wait_for_joint_target(target, description)

    def _feed_loop(self) -> None:
        if self._feed is None:
            return
        while self._is_connected or self._robot is not None:
            try:
                feed_info = self._feed.feedBackData()
            except Exception:
                return
            if feed_info is None or hex(feed_info["TestValue"][0]) != "0x123456789abcdef":
                continue
            with self._feed_lock:
                self._feed_data.MessageSize = feed_info["len"][0]
                self._feed_data.RobotMode = feed_info["RobotMode"][0]
                self._feed_data.DigitalInputs = feed_info["DigitalInputs"][0]
                self._feed_data.DigitalOutputs = feed_info["DigitalOutputs"][0]
                self._feed_data.User = int(feed_info["User"][0])
                self._feed_data.Tool = int(feed_info["Tool"][0])
                self._feed_data.robotCurrentCommandID = feed_info["CurrentCommandId"][0]
                self._feed_data.tcpPose = feed_info["ToolVectorActual"][0]
                self._feed_data.qActual = feed_info["QActual"][0]

    def _create_gripper_modbus(self) -> _DobotModbusRTU:
        if self._robot is None or self.config.dh_gripper is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        dedicated_error: Exception | None = None
        if self.config.dedicated_gripper_dashboard:
            try:
                self.logger.info(
                    "Connecting right DH gripper on dedicated Dashboard socket: "
                    f"{self.config.robot_ip}:{self.config.dashboardPort}"
                )
                self._gripper_robot = DobotApiDashboard(
                    self.config.robot_ip, self.config.dashboardPort
                )
                return _DobotModbusRTU(
                    self._gripper_robot,
                    self.config.master_ip,
                    self.config.master_port,
                    self.config.dh_gripper.slave_id,
                )
            except Exception as exception:
                dedicated_error = exception
                if self._gripper_robot is not None:
                    with contextlib.suppress(Exception):
                        self._gripper_robot.close()
                    self._gripper_robot = None
                self.logger.warn(
                    "Right dedicated gripper Dashboard failed; falling back to the shared "
                    f"robot Dashboard: {exception}"
                )
        try:
            return _DobotModbusRTU(
                self._robot,
                self.config.master_ip,
                self.config.master_port,
                self.config.dh_gripper.slave_id,
            )
        except Exception as exception:
            if dedicated_error is not None:
                raise RuntimeError(
                    "Right gripper Modbus failed on dedicated and shared Dashboard "
                    f"connections. dedicated={dedicated_error}; shared={exception}"
                ) from exception
            raise

    def connect(self, calibrate: bool = False, go_to_start: bool | None = None) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} is already connected; do not call robot.connect() twice."
            )
        try:
            self._feed_data = _FeedState()
            self.logger.info(f"Connecting right Dobot Nova5: {self.config.robot_ip}")
            self._robot = DobotApiDashboard(
                self.config.robot_ip, self.config.dashboardPort
            )
            self._feed = DobotApiFeedBack(
                self.config.robot_ip, self.config.feedPortFour
            )
            self._is_connected = True
            self._feed_thread = threading.Thread(
                target=self._feed_loop,
                name="DobotNova5DHFeed",
                daemon=True,
            )
            self._feed_thread.start()
            self._wait_for_first_feedback()

            if int(self._feed_data.RobotMode) == 9:
                self.logger.warn(
                    "Right robot is in error mode before enabling; trying ClearError."
                )
                self._raise_if_dobot_error(
                    self._robot, self._robot.ClearError(), "right ClearError"
                )
                self._wait_until_not_error_mode()

            self.logger.info("Enabling right robot...")
            enable_response = self._robot.EnableRobot()
            enable_error, _ = self._parse_dobot_response(enable_response)
            if enable_error != 0:
                current_mode = self._dashboard_robot_mode()
                if current_mode not in (5, 6, 7, 8):
                    raise RuntimeError(
                        f"Right EnableRobot failed with ErrorID {enable_error}: "
                        f"{enable_response.strip()} (RobotMode={current_mode}); "
                        f"GetErrorID: {self._dobot_error_detail()}"
                    )
                self.logger.warn(
                    f"Right EnableRobot returned {enable_error}, but RobotMode={current_mode}; "
                    "continuing with the existing enabled/control state."
                )

            self._wait_until_operational()
            self._connect_gripper()

            for camera in self.cameras.values():
                camera.connect()

            should_go_to_start = (
                bool(go_to_start) if go_to_start is not None else self.config.go_to_start
            )
            if should_go_to_start:
                self._go_to_start()

            self.configure()
            self._start_action_worker()
            self.logger.info("Dobot Nova5 right arm connected and ready.")
        except Exception:
            self._cleanup_failed_connect()
            raise

    def _wait_until_operational(self, timeout_s: float = 30.0) -> None:
        start_time = time.time()
        last_log_time = 0.0
        while True:
            feed_mode = int(self._feed_data.RobotMode)
            dashboard_mode = self._dashboard_robot_mode()
            current_mode = dashboard_mode if dashboard_mode is not None else feed_mode
            if current_mode == 5:
                return
            now = time.time()
            if now - last_log_time >= 2.0:
                self.logger.info(
                    "Waiting for right robot to become operational: "
                    f"feed/dashboard={feed_mode}/{dashboard_mode}"
                )
                last_log_time = now
            if now - start_time > timeout_s:
                raise RuntimeError(
                    f"Right robot did not become operational within {timeout_s:.0f} seconds: "
                    f"feed/dashboard={feed_mode}/{dashboard_mode}; "
                    f"GetErrorID={self._dobot_error_detail()}"
                )
            time.sleep(0.1)

    def _connect_gripper(self) -> None:
        if not self._gripper or not self.config.use_gripper or self._robot is None:
            return
        self.logger.info("Connecting right DH gripper through the robot RS485 port...")
        try:
            self._raise_if_dobot_error(
                self._robot,
                self._robot.SetToolMode(1, 1, self.config.tool_identify),
                "right SetToolMode",
            )
            self._raise_if_dobot_error(
                self._robot,
                self._robot.SetTool485(
                    self.config.dh_gripper_baudrate,
                    "N",
                    1,
                    self.config.tool_identify,
                ),
                "right SetTool485",
            )
            self._gripper.connect(self._create_gripper_modbus())
            self._gripper_connected = True
        except Exception as exception:
            if self._gripper_robot is not None:
                with contextlib.suppress(Exception):
                    self._gripper_robot.close()
                self._gripper_robot = None
            self._gripper_connected = False
            self.logger.error(
                "Failed to connect the right DH gripper; continuing without gripper "
                f"control: {exception}"
            )

    def _cleanup_failed_connect(self) -> None:
        self._stop_action_worker()
        if self._gripper and self._gripper_connected:
            with contextlib.suppress(Exception):
                self._gripper.disconnect()
        for camera in self.cameras.values():
            if camera.is_connected:
                with contextlib.suppress(Exception):
                    camera.disconnect()
        for client in (self._gripper_robot, self._feed, self._robot):
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()
        self._is_connected = False
        self._robot = None
        self._gripper_robot = None
        self._feed = None
        self._gripper_connected = False
        if self._feed_thread is not None:
            self._feed_thread.join(timeout=1.0)
            self._feed_thread = None

    def _initialize_gripper_position(self) -> None:
        if not (
            self._gripper
            and self.config.use_gripper
            and self._gripper_connected
        ):
            return
        target = 1.0 if self.config.dh_gripper_init_open else 0.0
        try:
            self._gripper.initialize_gripper_position(target)
        except Exception as exception:
            self._gripper_connected = False
            self.logger.error(
                "Right DH gripper initial move failed; disabling gripper commands for "
                f"this session: {exception}"
            )

    def _go_to_start(self) -> None:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.logger.info("Moving right arm to start position...")
        self._move_joint_movj(self.config.start_position_degree, "start position")
        self._initialize_gripper_position()
        self.logger.info("Right arm reached the start position.")

    def _go_to_home(self) -> None:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.logger.info("Moving right arm to home position...")
        self._move_joint_movj(self.config.home_point_list, "home position")
        self._initialize_gripper_position()
        self.logger.info("Right arm reached the home position.")

    def reset_to_initial_position(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._pause_action_worker():
            if self.config.reset_target == ResetTarget.START:
                self._go_to_start()
            elif self.config.reset_target == ResetTarget.HOME:
                self._go_to_home()
            else:
                raise ValueError(f"Unsupported reset_target: {self.config.reset_target}")

    @staticmethod
    def _read_tcp_pose_quat_from_feed(feed: _FeedState) -> np.ndarray:
        tcp_pose = np.asarray(feed.tcpPose, dtype=np.float64)
        position_m = tcp_pose[:3] / MM_PER_METER
        quaternion = euler_to_quaternion(
            np.deg2rad(tcp_pose[3]),
            np.deg2rad(tcp_pose[4]),
            np.deg2rad(tcp_pose[5]),
        )
        return np.asarray([*position_m, *quaternion], dtype=np.float32)

    def _normalize_gripper_command(self, position: float) -> float:
        clipped = max(0.0, min(1.0, float(position)))
        if not self.config.binary_gripper_actions:
            return clipped
        return 1.0 if clipped >= float(self.config.gripper_open_threshold) else 0.0

    def _read_gripper_pos(self) -> float:
        if self._gripper and self.config.use_gripper and self._gripper_connected:
            return self._normalize_gripper_command(
                float(self._gripper.get_gripper_position())
            )
        return 0.0

    def get_current_tcp_pose_quat(self) -> np.ndarray:
        """Return ``[x, y, z, qw, qx, qy, qz, gripper]`` for the right arm."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._feed_lock:
            pose = self._read_tcp_pose_quat_from_feed(self._feed_data)
        return np.asarray([*pose, self._read_gripper_pos()], dtype=np.float32)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        observation_start = time.perf_counter()
        timing: dict[str, float] = {}
        observation: dict[str, Any] = {}

        arm_start = time.perf_counter()
        with self._feed_lock:
            if self.config.control_mode == ControlMode.JOINT_MOTION:
                joints = list(self._feed_data.qActual)
                for index, key in enumerate(self._joint_pos_keys):
                    observation[key] = joints[index]
            elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
                pose = np.asarray(self._feed_data.tcpPose, dtype=np.float64).copy()
                observation["tcp.x"] = pose[0] / MM_PER_METER
                observation["tcp.y"] = pose[1] / MM_PER_METER
                observation["tcp.z"] = pose[2] / MM_PER_METER
                quaternion = euler_to_quaternion(
                    np.deg2rad(pose[3]),
                    np.deg2rad(pose[4]),
                    np.deg2rad(pose[5]),
                )
                rotation_6d = quaternion_to_rotation_6d(*quaternion)
                for index in range(6):
                    observation[f"tcp.r{index + 1}"] = rotation_6d[index]
            else:
                raise ValueError(
                    f"Unsupported control_mode: {self.config.control_mode}"
                )
        timing["arm_ms"] = (time.perf_counter() - arm_start) * 1e3

        gripper_start = time.perf_counter()
        observation[self._gripper_key] = self._read_gripper_pos()
        timing["gripper_ms"] = (time.perf_counter() - gripper_start) * 1e3

        cameras_start = time.perf_counter()
        for camera_key, camera in self.cameras.items():
            camera_start = time.perf_counter()
            observation[camera_key] = camera.async_read()
            timing[f"cam[{camera_key}]_ms"] = (
                time.perf_counter() - camera_start
            ) * 1e3
        timing["cameras_ms"] = (time.perf_counter() - cameras_start) * 1e3
        timing["total_ms"] = (time.perf_counter() - observation_start) * 1e3
        self._last_obs_timing = timing
        return observation

    def _send_joint_action(self, action: dict[str, Any]) -> None:
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        target = [float(action[key]) for key in self._action_joint_keys]
        response = self._robot.ServoJ(*target)
        self._raise_if_dobot_error(self._robot, response, "right ServoJ")

    def _clip_workspace_position(self, position_m: np.ndarray) -> np.ndarray:
        if not self.config.enable_clip:
            return position_m
        return np.clip(
            position_m,
            np.asarray(self.config.workspace_min_xyz_m, dtype=np.float64),
            np.asarray(self.config.workspace_max_xyz_m, dtype=np.float64),
        )

    def _cartesian_feed_state(self) -> tuple[np.ndarray, np.ndarray]:
        with self._feed_lock:
            tcp_pose = np.asarray(self._feed_data.tcpPose, dtype=np.float64).copy()
            joint_position = np.asarray(
                self._feed_data.qActual, dtype=np.float64
            ).copy()
        return tcp_pose, joint_position

    def _parse_ik_joint_values(
        self, response: str, command_name: str
    ) -> np.ndarray | None:
        error_id, values = self._parse_dobot_response(response)
        if error_id != 0:
            return None
        if len(values) < JOINT_DOF:
            self.logger.warn(
                f"{command_name} returned too few joint values: {response.strip()}"
            )
            return None
        try:
            return np.asarray(values[:JOINT_DOF], dtype=np.float64)
        except ValueError:
            self.logger.warn(
                f"{command_name} returned invalid joint values: {response.strip()}"
            )
            return None

    def _inverse_kin_with_joint_near(
        self,
        pose_mm_deg: np.ndarray,
        joint_near_deg: np.ndarray,
    ) -> np.ndarray | None:
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        joint_near = "{" + ",".join(
            f"{float(value):.6f}" for value in joint_near_deg
        ) + "}"
        response = self._robot.InverseKin(
            *[float(value) for value in pose_mm_deg],
            useJointNear=1,
            JointNear=joint_near,
        )
        return self._parse_ik_joint_values(response, "right InverseKin")

    def _resolve_cartesian_servo_target(
        self, target_pose_mm_deg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.config.enable_cartesian_ik_guard:
            return target_pose_mm_deg, None

        current_pose, current_joint = self._cartesian_feed_state()
        target_joint = self._inverse_kin_with_joint_near(
            target_pose_mm_deg, current_joint
        )
        if target_joint is not None:
            return target_pose_mm_deg, target_joint

        backoff_steps = int(self.config.cartesian_ik_backoff_steps)
        for step in range(backoff_steps, 0, -1):
            ratio = step / float(backoff_steps + 1)
            candidate_pose = current_pose + ratio * (target_pose_mm_deg - current_pose)
            target_joint = self._inverse_kin_with_joint_near(
                candidate_pose, current_joint
            )
            if target_joint is not None:
                now = time.perf_counter()
                if now - self._last_cartesian_guard_log_s >= 1.0:
                    self.logger.warn(
                        "Right Cartesian target was rejected by the IK guard; "
                        f"backing off to {ratio:.2f} of the requested step near "
                        f"pose={target_pose_mm_deg.tolist()}"
                    )
                    self._last_cartesian_guard_log_s = now
                return candidate_pose, target_joint

        now = time.perf_counter()
        if now - self._last_cartesian_guard_log_s >= 1.0:
            self.logger.warn(
                "Right Cartesian target was rejected by the IK guard; holding the "
                f"current pose near a singularity for target={target_pose_mm_deg.tolist()}"
            )
            self._last_cartesian_guard_log_s = now
        return current_pose, current_joint

    def _send_cartesian_servoj(
        self, target_joint_deg: np.ndarray, command_name: str
    ) -> None:
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        response = self._robot.ServoJ(*[float(value) for value in target_joint_deg])
        self._raise_if_dobot_error(self._robot, response, command_name)

    @staticmethod
    def _format_array(values: np.ndarray | list[float], precision: int = 3) -> str:
        return np.array2string(
            np.asarray(values, dtype=np.float64),
            precision=precision,
            suppress_small=True,
        )

    def _log_cartesian_command_failure(
        self, response: str | None, command_name: str
    ) -> None:
        debug = self._last_cartesian_command_debug
        if not debug:
            self.logger.error(
                f"Right {command_name} failed, but no Cartesian debug context was recorded."
            )
            return
        target_pose = np.asarray(debug["target_pose_mm_deg"], dtype=np.float64)
        reasons: list[str] = []
        error_detail = self._dobot_error_detail()
        if "肩奇异" in error_detail or "shoulder singularity" in error_detail.lower():
            reasons.append("Dobot reported a shoulder singularity")
        if "逆解算无解" in error_detail or "InverseKin" in error_detail:
            reasons.append("Dobot reported no inverse-kinematics solution")
        if abs(target_pose[0]) < 120.0 and target_pose[2] > 450.0:
            reasons.append("the high TCP target is close to the shoulder singular axis")
        if debug["workspace_clipped"]:
            reasons.append("workspace clipping moved the target onto a reachability boundary")
        if debug["step_limited"]:
            reasons.append("max_cartesian_step_m limited this frame to an intermediate point")
        if not reasons:
            reasons.append("Dobot returned an error not classified by the local heuristics")

        self.logger.error(
            "Right Cartesian command diagnostics:\n"
            f"  command={command_name}\n"
            f"  likely_reason={'; '.join(reasons)}\n"
            f"  response={response.strip() if isinstance(response, str) else response}\n"
            f"  GetErrorID={error_detail}\n"
            f"  requested_pos_m={self._format_array(debug['requested_pos_m'])}\n"
            f"  clipped_pos_m={self._format_array(debug['clipped_pos_m'])} "
            f"workspace_clipped={debug['workspace_clipped']}\n"
            f"  current_pose_mm_deg={self._format_array(debug['current_pose_mm_deg'])}\n"
            f"  target_pose_mm_deg={self._format_array(target_pose)}\n"
            f"  current_joint_deg={self._format_array(debug['current_joint_deg'])}\n"
            f"  step_limited={debug['step_limited']} "
            f"max_step_mm={debug['max_step_mm']:.1f}"
        )

    def _send_cart_action(self, action: dict[str, Any]) -> None:
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if "arm.enabled" in action and not bool(action["arm.enabled"]):
            return

        requested_position = np.asarray(
            [action["tcp.x"], action["tcp.y"], action["tcp.z"]],
            dtype=np.float64,
        )
        clipped_position = self._clip_workspace_position(requested_position)
        current_pose, current_joint = self._cartesian_feed_state()

        target_position_mm = clipped_position * MM_PER_METER
        target_position_before_step = target_position_mm.copy()
        max_step_mm = float(self.config.max_cartesian_step_m) * MM_PER_METER
        step_limited = False
        if max_step_mm > 0.0:
            delta_position = target_position_mm - current_pose[:3]
            delta_norm = float(np.linalg.norm(delta_position))
            if delta_norm > max_step_mm:
                target_position_mm = current_pose[:3] + delta_position * (
                    max_step_mm / delta_norm
                )
                step_limited = True

        rotation_6d = np.asarray(
            [action[f"tcp.r{index}"] for index in range(1, 7)],
            dtype=np.float64,
        )
        quaternion = rotation_6d_to_quaternion(rotation_6d)
        euler = quaternion_to_euler(*quaternion)
        target_pose = np.asarray(
            [*target_position_mm, *np.rad2deg(euler)],
            dtype=np.float64,
        )
        self._last_cartesian_command_debug = {
            "requested_pos_m": requested_position.copy(),
            "clipped_pos_m": clipped_position.copy(),
            "workspace_clipped": bool(
                not np.allclose(
                    requested_position,
                    clipped_position,
                    rtol=0.0,
                    atol=1e-9,
                )
            ),
            "current_pose_mm_deg": current_pose.copy(),
            "current_joint_deg": current_joint.copy(),
            "target_pos_before_step_mm": target_position_before_step,
            "target_pose_mm_deg": target_pose.copy(),
            "step_limited": step_limited,
            "max_step_mm": max_step_mm,
        }

        if (
            self.config.enable_cartesian_ik_guard
            and self.config.cartesian_ik_servoj
            and time.perf_counter() < self._cartesian_ik_servoj_until_s
        ):
            target_pose, target_joint = self._resolve_cartesian_servo_target(target_pose)
            if target_joint is not None:
                self._send_cartesian_servoj(target_joint, "right ServoJ IK guard")
                return

        response = self._robot.ServoP(*[float(value) for value in target_pose])
        response_error_id, _ = self._parse_dobot_response(response)
        if response_error_id == 0:
            return

        if not (
            self.config.enable_cartesian_ik_guard
            and self.config.cartesian_ik_servoj
            and response_error_id in (-2, -1)
        ):
            with contextlib.suppress(Exception):
                self._log_cartesian_command_failure(response, "right ServoP")
            self._raise_if_dobot_error(self._robot, response, "right ServoP")
            return

        original_response = response
        self._cartesian_ik_servoj_until_s = (
            time.perf_counter() + float(self.config.cartesian_ik_servoj_hold_s)
        )
        _, target_joint = self._resolve_cartesian_servo_target(target_pose)
        if target_joint is None:
            return
        try:
            self._send_cartesian_servoj(
                target_joint, "right ServoJ after ServoP IK fallback"
            )
        except Exception as exception:
            with contextlib.suppress(Exception):
                self._log_cartesian_command_failure(
                    original_response, "right ServoJ fallback"
                )
            self.logger.warn(
                f"Right ServoJ fallback failed; skipping this frame: {exception}"
            )

    def _send_gripper_action(self, action: dict[str, Any]) -> None:
        if not (
            self._gripper
            and self.config.use_gripper
            and self._gripper_connected
            and self._gripper_key in action
        ):
            return
        try:
            position = self._normalize_gripper_command(action[self._gripper_key])
            action[self._gripper_key] = position
            self._gripper.set_gripper_position(position)
        except Exception as exception:
            self._gripper_connected = False
            self.logger.error(
                "Right DH gripper command failed; disabling gripper commands for this "
                f"session: {exception}"
            )

    def _action_worker_loop(self) -> None:
        minimum_period = 1.0 / max(
            float(self.config.async_action_worker_frequency), 1e-6
        )
        next_send = time.perf_counter()
        while True:
            with self._action_worker_cv:
                while self._action_worker_running and (
                    self._action_worker_paused
                    or self._pending_async_action is None
                ):
                    self._action_worker_cv.wait(timeout=0.1)
                if not self._action_worker_running:
                    break
                action = self._pending_async_action
                self._pending_async_action = None

            if action is None:
                continue
            now = time.perf_counter()
            if now < next_send:
                time.sleep(next_send - now)
            send_start = time.perf_counter()
            try:
                with self._action_worker_cv:
                    self._action_worker_busy = True
                sent_action = self._send_action_sync(action)
                elapsed_ms = (time.perf_counter() - send_start) * 1e3
                with self._action_worker_lock:
                    self._latest_sent_async_action = sent_action
                    self._async_action_send_count += 1
                    self._last_async_action_timing = {
                        "worker_send_ms": elapsed_ms,
                        "queue_drop_count": float(self._async_action_drop_count),
                        "send_count": float(self._async_action_send_count),
                        "error_count": float(self._async_action_error_count),
                    }
            except Exception as exception:
                with self._action_worker_lock:
                    self._async_action_error_count += 1
                    self._last_async_action_timing = {
                        "worker_error_count": float(self._async_action_error_count),
                        "queue_drop_count": float(self._async_action_drop_count),
                    }
                self.logger.warn(
                    f"Async action worker skipped a failed action: {exception}"
                )
            finally:
                with self._action_worker_cv:
                    self._action_worker_busy = False
                    self._action_worker_cv.notify_all()
            next_send = max(next_send + minimum_period, time.perf_counter())

    def _start_action_worker(self) -> None:
        if not self.config.async_action_worker or self._action_worker_thread is not None:
            return
        self._action_worker_running = True
        self._action_worker_paused = False
        self._action_worker_thread = threading.Thread(
            target=self._action_worker_loop,
            name="DobotNova5DHActionWorker",
            daemon=True,
        )
        self._action_worker_thread.start()
        self.logger.info(
            "Async action worker started at "
            f"{self.config.async_action_worker_frequency:.1f} Hz."
        )

    def _stop_action_worker(self) -> None:
        with self._action_worker_cv:
            self._action_worker_running = False
            self._pending_async_action = None
            self._action_worker_cv.notify_all()
        if self._action_worker_thread is not None:
            self._action_worker_thread.join(timeout=2.0)
            self._action_worker_thread = None

    @contextlib.contextmanager
    def _pause_action_worker(self):
        if not self.config.async_action_worker:
            yield
            return
        with self._action_worker_cv:
            previous_paused = self._action_worker_paused
            self._action_worker_paused = True
            self._pending_async_action = None
            self._action_worker_cv.notify_all()
            while self._action_worker_busy:
                self._action_worker_cv.wait(timeout=0.05)
        try:
            yield
        finally:
            with self._action_worker_cv:
                self._action_worker_paused = previous_paused
                self._action_worker_cv.notify_all()

    def _submit_async_action(self, action: dict[str, Any]) -> dict[str, Any]:
        submit_start = time.perf_counter()
        with self._action_worker_cv:
            if self._pending_async_action is not None:
                self._async_action_drop_count += 1
            self._pending_async_action = dict(action)
            last_sent = self._latest_sent_async_action
            worker_timing = dict(self._last_async_action_timing)
            self._action_worker_cv.notify()
        submit_ms = (time.perf_counter() - submit_start) * 1e3
        self._last_action_timing = {
            "async_submit_ms": submit_ms,
            "total_ms": submit_ms,
            **worker_timing,
        }
        return last_sent if last_sent is not None else action

    def _send_action_sync(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        action_start = time.perf_counter()
        timing: dict[str, float] = {}

        if int(self._feed_data.RobotMode) == 9:
            diagnostics_start = time.perf_counter()
            with contextlib.suppress(Exception):
                self._log_cartesian_command_failure(None, "right RobotMode=9")
            timing["fault_diagnostics_ms"] = (
                time.perf_counter() - diagnostics_start
            ) * 1e3
            self.logger.warn(
                "Right robot fault detected; trying ClearError and skipping this frame. "
                f"GetErrorID: {self._dobot_error_detail()}"
            )
            clear_start = time.perf_counter()
            self.clear_fault()
            timing["clear_fault_ms"] = (
                time.perf_counter() - clear_start
            ) * 1e3
            timing["total_ms"] = (time.perf_counter() - action_start) * 1e3
            self._last_action_timing = timing
            return action

        arm_start = time.perf_counter()
        if self.config.control_mode == ControlMode.JOINT_MOTION:
            self._send_joint_action(action)
        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            self._send_cart_action(action)
        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")
        timing["arm_ms"] = (time.perf_counter() - arm_start) * 1e3

        gripper_start = time.perf_counter()
        self._send_gripper_action(action)
        timing["gripper_ms"] = (time.perf_counter() - gripper_start) * 1e3
        timing["total_ms"] = (time.perf_counter() - action_start) * 1e3
        self._last_action_timing = timing
        return action

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.config.async_action_worker:
            return self._submit_async_action(action)
        return self._send_action_sync(action)

    def clear_fault(self) -> bool:
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if int(self._feed_data.RobotMode) != 9:
            return True
        try:
            self._raise_if_dobot_error(
                self._robot, self._robot.ClearError(), "right ClearError"
            )
            return True
        except Exception as exception:
            self.logger.error(f"Failed to clear the right robot fault: {exception}")
            return False

    def disconnect(self) -> None:
        if not self._is_connected:
            self.logger.warn(f"{self} is not connected; skipping disconnect.")
            return
        try:
            self.logger.info("Disconnecting the Dobot Nova5 right arm...")
            self._stop_action_worker()
            if self.config.go_to_home_on_disconnect:
                try:
                    self._go_to_home()
                except Exception as exception:
                    self.logger.warn(
                        f"Failed to move right arm home before disconnect: {exception}"
                    )

            # The gripper must disconnect before the Dashboard socket because opening the
            # gripper and ModbusClose both travel over that TCP connection.
            if self._gripper and self.config.use_gripper and self._gripper_connected:
                with contextlib.suppress(Exception):
                    self._gripper.disconnect()

            if self._robot is not None:
                with contextlib.suppress(Exception):
                    self._robot.Stop()
                with contextlib.suppress(Exception):
                    self._robot.close()
            if self._gripper_robot is not None:
                with contextlib.suppress(Exception):
                    self._gripper_robot.close()
            if self._feed is not None:
                with contextlib.suppress(Exception):
                    self._feed.close()
            for camera in self.cameras.values():
                if camera.is_connected:
                    with contextlib.suppress(Exception):
                        camera.disconnect()
        except Exception as exception:
            self.logger.error(f"Error while disconnecting the right arm: {exception}")
        finally:
            self._robot = None
            self._gripper_robot = None
            self._feed = None
            self._gripper_connected = False
            self._is_connected = False
            if self._feed_thread is not None:
                self._feed_thread.join(timeout=1.0)
                self._feed_thread = None
            self.logger.info("Dobot Nova5 right arm disconnected.")
