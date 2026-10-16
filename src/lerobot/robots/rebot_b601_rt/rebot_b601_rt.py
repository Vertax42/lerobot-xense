"""LeRobot adapter; all motor IO, scheduling and IK run in rebot_rt C++."""

import time
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import rebot_rt

from lerobot.cameras import make_cameras_from_configs
from lerobot.grippers import make_gripper_from_config
from lerobot.grippers.camera_injection import (
    adopt_taccap_mcu_device,
    attach_wrist_fisheye_calibration,
    inject_serial_gripper_cameras,
    inject_taccap_cameras,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.robot_utils import best_effort, get_logger, quaternion_to_euler
from lerobot.utils.rotation import Rotation

from ..robot import Robot
from .can_diagnostics import can_error_hint
from .config_rebot_b601_rt import RebotB601RTConfig

RobotAction = dict[str, Any]
RobotObservation = dict[str, Any]
TCP_KEYS = ("tcp.x", "tcp.y", "tcp.z", *(f"tcp.r{i}" for i in range(1, 7)))
JOINT_KEYS = tuple(f"joint_{i}.pos" for i in range(1, 7))


def _pose_from_action(action: RobotAction) -> np.ndarray:
    values = np.asarray([action[key] for key in TCP_KEYS], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("TCP action contains non-finite values")
    x, y = values[3:6], values[6:9]
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        raise ValueError("TCP rotation column 1 is zero")
    x = x / x_norm
    y = y - x * np.dot(x, y)
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-8:
        raise ValueError("TCP rotation columns are parallel")
    y = y / y_norm
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    pose[:3, 3] = values[:3]
    return pose


def _action_from_pose(pose: np.ndarray) -> dict[str, float]:
    values = (*pose[:3, 3], *pose[:3, 0], *pose[:3, 1])
    return dict(zip(TCP_KEYS, (float(v) for v in values), strict=True))


class RebotB601RT(Robot):
    config_class = RebotB601RTConfig
    name = "rebot_b601_rt"
    supports_position_hold = True

    def __init__(self, config: RebotB601RTConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._native: rebot_rt.Robot | None = None
        self._ik_failed = False
        self._last_ik_warning = float("-inf")
        self._last_tracking_log = float("-inf")
        self._reset_target: list[float] | None = None
        self._reset_started_at = 0.0
        self._reset_settled_since: float | None = None
        self._gripper = make_gripper_from_config(config.gripper)
        if config.gripper is not None and config.gripper.auto_discover_cameras:
            side = config.gripper.side
            if config.gripper.type == "taccap_follower":
                devices = inject_taccap_cameras(
                    config.cameras,
                    sides=(side,),
                    enable_tactile=config.gripper.enable_tactile,
                    logger=self.logger,
                    undistort_wrist=config.gripper.undistort_wrist,
                    fisheye_balance=config.gripper.fisheye_balance,
                )
                for found_side, device in devices.items():
                    adopt_taccap_mcu_device(self._gripper, found_side, device, self.logger)
            elif config.gripper.type == "serial":
                inject_serial_gripper_cameras(
                    config.cameras,
                    sides=(side,),
                    enable_tactile=config.gripper.enable_tactile,
                    logger=self.logger,
                )
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **dict.fromkeys(TCP_KEYS, float),
            **({"gripper.pos": float} if self.config.has_gripper or self._gripper is not None else {}),
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = dict.fromkeys(
            JOINT_KEYS if self.config.use_joint_observation else TCP_KEYS, float
        )
        if self.config.has_gripper or self._gripper is not None:
            features["gripper.pos"] = float
        for name, cam in self.cameras.items():
            cfg = self.config.cameras[name]
            if getattr(cam, "use_rgb", True):
                features[name] = (cfg.height, cfg.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{name}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @property
    def is_connected(self) -> bool:
        return (
            self._native is not None
            and self._native.connected
            and (self._gripper is None or self._gripper.is_connected)
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        # Zero offsets live in motor firmware and must be established with the vendor tool.
        return True

    def calibrate(self) -> None:
        raise RuntimeError("B601 motor zero offsets must be set with the vendor calibration tool")

    def configure(self) -> None:
        if self._native is None:
            raise RuntimeError("connect() must be called before configure()")

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        assets = Path(rebot_rt.__file__).parent / "assets"
        urdf = self.config.urdf_path or str(assets / self.config.default_urdf)
        end_frame = self.config.end_frame or ("taccap_eef_link" if self.config.uses_taccap_model else "")
        if self.config.urdf_path is None and not Path(urdf).is_file():
            raise FileNotFoundError(f"Bundled URDF was not found at {urdf}")
        family = rebot_rt.Family.DM if self.config.motor_family == "dm" else rebot_rt.Family.RS
        motor_configs = []
        if self.config.motor_configs:
            defaults = rebot_rt.default_motor_configs(family)
            allowed = {
                "send_id",
                "feedback_id",
                "model",
                "pmax",
                "vmax",
                "tmax",
                "kp",
                "kd",
                "vel_kp",
                "vel_ki",
                "pos_kp",
                "pos_ki",
                "kpmax",
                "kdmax",
                "direction",
            }
            for index, changes in enumerate(self.config.motor_configs):
                unknown = set(changes) - allowed
                if unknown:
                    raise ValueError(f"Unknown motor config fields at joint {index + 1}: {sorted(unknown)}")
                native_motor = defaults[index]
                if "model" in changes:
                    native_motor.set_model(family, str(changes["model"]))
                for key, value in changes.items():
                    if key != "model":
                        setattr(native_motor, key, value)
                motor_configs.append(native_motor)
        if not hasattr(rebot_rt.State, "mit_command_velocity"):
            raise RuntimeError("libpyrebot must be rebuilt/updated to support mit_velocity_feedforward")
        native = rebot_rt.Robot(
            family,
            self.config.channel,
            urdf,
            end_frame,
            self.config.has_gripper,
            self.config.max_joint_speed,
            self.config.command_timeout_sec,
            self.config.dm_serial_baud,
            motor_configs,
            self.config.control_mode,
            self.config.mit_velocity_feedforward,
        )
        if self.config.interpolate_cartesian_targets and not hasattr(native, "queue_target_pose"):
            raise RuntimeError("libpyrebot must be rebuilt/updated to support native Cartesian interpolation")
        try:
            native.connect()
            native.start()
            if self._gripper is not None:
                self._gripper.connect()
            attach_wrist_fisheye_calibration(
                self.cameras, {getattr(self.config.gripper, "side", "left"): self._gripper}, self.logger
            )
            for cam in self.cameras.values():
                cam.connect()
        except Exception as exc:
            self._close_devices(native)
            self._log_can_error_hint(str(exc))
            raise
        self._native = native
        self.logger.info(
            f"B601 control_mode={self.config.control_mode}, max_joint_speed={self.config.max_joint_speed} rad/s, "
            f"cartesian_interpolation={self.config.interpolate_cartesian_targets} (100 Hz), "
            f"mit_velocity_feedforward={self.config.mit_velocity_feedforward}"
        )

    @check_if_not_connected
    def reset_to_initial_position(self) -> None:
        """Start a joint-space return to the configured A-button pose."""
        self._validate_reset()
        target = [*self.config.reset_position_rad, self._native.get_target_joints()[6]]
        speed = min(self.config.reset_joint_speed, self.config.max_joint_speed)
        self._native.set_target_joints(target, max_joint_speed=speed)
        self._reset_target = target
        self._reset_started_at = time.monotonic()
        self._reset_settled_since = None
        self.logger.info(f"Returning to reset_position_rad={self.config.reset_position_rad} at {speed} rad/s")

    def _validate_reset(self) -> None:
        if self.config.reset_position_rad is None:
            raise ValueError("Set reset_position_rad to six joint angles before requesting a reset")
        state = self._native.get_state()
        if state.fault:
            self._raise_control_fault(state.fault_reason)

    @property
    def rt_moving(self) -> bool:
        if self._native is None:
            return False
        state = self._native.get_state()
        if state.fault:
            self._raise_control_fault(state.fault_reason)
        if self._reset_target is None:
            return False
        now = time.monotonic()
        if now - self._reset_started_at > self.config.reset_timeout_sec:
            self._cancel_reset()
            raise TimeoutError("B601 reset did not reach the configured pose before reset_timeout_sec")
        target = np.asarray(self._reset_target[:6])
        command_arrived = np.max(np.abs(np.asarray(state.command[:6]) - target)) < 1e-6
        feedback_arrived = np.max(np.abs(np.asarray(state.q[:6]) - target)) <= self.config.reset_tolerance_rad
        if command_arrived and feedback_arrived:
            if self._reset_settled_since is None:
                self._reset_settled_since = now
            elif now - self._reset_settled_since >= 0.1:
                self._reset_target = None
                self.logger.info("B601 reset complete")
                return False
        else:
            self._reset_settled_since = None
        # Reset lasts longer than the teleop watchdog. Keep its goal alive while
        # the shared loop stands aside, until measured feedback settles.
        self._native.set_target_joints(
            self._reset_target, max_joint_speed=min(self.config.reset_joint_speed, self.config.max_joint_speed)
        )
        return True

    def _cancel_reset(self) -> None:
        self._native.hold_position()
        self._reset_target = None
        self._reset_settled_since = None

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._native.get_state()
        if state.fault:
            self._raise_control_fault(state.fault_reason)
        now = time.monotonic()
        if self.config.interpolate_cartesian_targets and state.cartesian_ik_valid:
            if hasattr(state, "cartesian_tracking_limited"):
                # Partial tracking can mean a speed-limited or unreachable pose;
                # it is not an IK rejection. Report it in the tracking trace.
                if state.cartesian_tracking_limited != self._ik_failed and now - self._last_ik_warning >= 1.0:
                    self.logger.info(
                        "B601 Cartesian tracking limited; approaching the target within joint limits"
                        if state.cartesian_tracking_limited
                        else "B601 Cartesian tracking reached the sampled target"
                    )
                    self._last_ik_warning = now
                self._ik_failed = state.cartesian_tracking_limited
            else:
                self._log_ik_status(state.cartesian_ik_success, state.cartesian_ik_error, state.q)
        if (
            self.config.trace_joint_tracking
            and now - self._last_tracking_log >= 1 / self.config.trace_joint_tracking_hz
        ):

            def fmt(values):
                return [round(float(v), 4) for v in values[:6]]

            tracking_error = np.asarray(state.command) - np.asarray(state.q)
            interpolation = (
                f" cartesian_dt_ms={state.cartesian_duration_sec * 1000:.1f} "
                f"ik_solves={state.cartesian_ik_solves} overruns={state.overruns}"
                f" tracking_limited={getattr(state, 'cartesian_tracking_limited', False)}"
                f" tracking_progress={getattr(state, 'cartesian_tracking_progress', False)}"
                if self.config.interpolate_cartesian_targets
                else ""
            )
            self.logger.info(
                f"Joint tracking: goal={fmt(state.target)} command={fmt(state.command)} "
                f"measured={fmt(state.q)} error={fmt(tracking_error)} velocity={fmt(state.command_velocity)}"
                + (f" mit_dq={fmt(state.mit_command_velocity)}" if self.config.control_mode == "mit" else "")
                + interpolation
            )
            self._last_tracking_log = now
        if self.config.use_joint_observation:
            obs = dict(zip(JOINT_KEYS, (float(v) for v in state.q[:6]), strict=True))
        else:
            obs = _action_from_pose(self._native.forward_kinematics(state.q))
        if self._gripper is not None:
            obs["gripper.pos"] = self._gripper.get_gripper_position()
        elif self.config.has_gripper:
            obs["gripper.pos"] = float(state.q[6])
        for name, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                obs[name] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{name}_depth"] = cam.read_latest_depth()
        return obs

    @check_if_not_connected
    def get_current_tcp_pose_quat(self) -> np.ndarray:
        """Measured base-frame TCP as [x, y, z, qw, qx, qy, qz, gripper]."""
        state = self._native.get_state()
        if state.fault:
            self._raise_control_fault(state.fault_reason)
        pose = self._native.forward_kinematics(state.q)
        qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        gripper = 0.0
        if self._gripper is not None:
            gripper = self._gripper.get_gripper_position()
        elif self.config.has_gripper:
            gripper = float(state.q[6])
        return np.array([*pose[:3, 3], qw, qx, qy, qz, gripper], dtype=np.float64)

    @check_if_not_connected
    def get_current_tcp_pose_euler(self) -> np.ndarray:
        """Measured base-frame TCP as [x, y, z, roll, pitch, yaw, gripper]."""
        pose = self.get_current_tcp_pose_quat()
        return np.array([*pose[:3], *quaternion_to_euler(*pose[3:7]), pose[7]], dtype=np.float64)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._reset_target is not None:
            raise RuntimeError("B601 reset is in progress; wait for rt_moving=False before sending teleop actions")
        pose = _pose_from_action(action)
        gripper = float(action["gripper.pos"]) if self.config.has_gripper else 0.0
        external_target = None
        if self._gripper is not None:
            external_target = float(action["gripper.pos"])
            if not 0.0 <= external_target <= 1.0:
                raise ValueError("External gripper target must be in [0, 1]")
        if self.config.interpolate_cartesian_targets:
            self._native.queue_target_pose(pose, gripper)
            # Queuing is not IK success. Report the latest accepted native
            # joint target; asynchronous substep failures are logged from state.
            target_joints = self._native.get_target_joints()
        else:
            result = self._native.set_target_pose(pose, gripper)
            state = self._native.get_state()
            if state.fault:
                self._raise_control_fault(state.fault_reason)
            target_joints = result.target if result.success else self._native.get_target_joints()
            self._log_ik_status(result.success, result.error if not result.success else 0.0, state.q)
        sent = _action_from_pose(self._native.forward_kinematics(target_joints))
        if self._gripper is not None:
            self._gripper.set_gripper_position(external_target)
            sent["gripper.pos"] = external_target
        elif self.config.has_gripper:
            sent["gripper.pos"] = float(target_joints[6])
        return sent

    def _log_can_error_hint(self, error: str) -> None:
        hint = can_error_hint(self.config.channel, error)
        if hint is not None:
            self.logger.error(hint)

    def _raise_control_fault(self, reason: str) -> None:
        self._log_can_error_hint(reason)
        raise RuntimeError(f"B601 C++ control fault: {reason}")

    def _log_ik_status(self, success: bool, error: float, measured) -> None:
        if success:
            if self._ik_failed:
                self.logger.info("B601 IK recovered; accepting Cartesian targets again")
            self._ik_failed = False
        else:
            now = time.monotonic()
            if now - self._last_ik_warning >= 1.0:
                self.logger.warn(
                    f"B601 IK target rejected (error={error:.6g}); holding previous arm target. "
                    f"Measured joints: {[round(float(q), 6) for q in measured[:6]]}"
                )
                self._last_ik_warning = now
            self._ik_failed = True

    @check_if_not_connected
    def disconnect(self) -> None:
        self._close_devices(self._native)
        self._native = None
        self._reset_target = None

    def _close_devices(self, native: rebot_rt.Robot) -> None:
        # Attempt every release even if one device fails during teardown.
        with best_effort(self.logger, "closing the native arm"):
            native.close()
        for cam in self.cameras.values():
            with best_effort(self.logger, "disconnecting a camera"):
                if cam.is_connected:
                    cam.disconnect()
        if self._gripper is not None:
            with best_effort(self.logger, "disconnecting the external gripper"):
                if self._gripper.is_connected:
                    self._gripper.disconnect()
