"""Bimanual LeRobot facade over two native 100 Hz controllers."""

import time
from dataclasses import replace
from functools import cached_property
from typing import Any

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.robot_utils import best_effort, get_logger

from ..rebot_b601_rt import RebotB601RT
from ..robot import Robot
from .config_bi_rebot_b601_rt import BiRebotB601RTConfig

RobotAction = dict[str, Any]
RobotObservation = dict[str, Any]


class BiRebotB601RT(Robot):
    config_class = BiRebotB601RTConfig
    name = "bi_rebot_b601_rt"
    supports_position_hold = True

    def __init__(self, config: BiRebotB601RTConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._top_camera_names = set(config.cameras)
        left_config = replace(
            config.left_arm_config,
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            cameras={**config.left_arm_config.cameras, **config.cameras},
        )
        right_config = replace(
            config.right_arm_config,
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
        )
        self.left_arm = RebotB601RT(left_config)
        self.right_arm = RebotB601RT(right_config)
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        raise RuntimeError("B601 motor zero offsets must be set with the vendor calibration tool")

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **{
                k if k in self._top_camera_names else f"left_{k}": v
                for k, v in self.left_arm.observation_features.items()
            },
            **{f"right_{k}": v for k, v in self.right_arm.observation_features.items()},
        }

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate=calibrate)
        try:
            self.right_arm.connect(calibrate=calibrate)
        except BaseException:
            self.left_arm.disconnect(reset=False)
            raise

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left = {
            k if k in self._top_camera_names else f"left_{k}": v for k, v in self.left_arm.get_observation().items()
        }
        right = {f"right_{k}": v for k, v in self.right_arm.get_observation().items()}
        return {**left, **right}

    @check_if_not_connected
    def get_current_tcp_pose_quat(self):
        """Each arm's measured TCP in its own base frame, for Bi-Pico4 sync."""
        return self.left_arm.get_current_tcp_pose_quat(), self.right_arm.get_current_tcp_pose_quat()

    @check_if_not_connected
    def reset_to_initial_position(self) -> None:
        if any(arm.config.reset_position_rad is None for arm in (self.left_arm, self.right_arm)):
            raise ValueError("Both arms need reset_position_rad before using the reset button")
        # Check both controllers before either arm starts moving.
        self.left_arm._validate_reset()
        self.right_arm._validate_reset()
        self.left_arm.reset_to_initial_position()
        try:
            self.right_arm.reset_to_initial_position()
        except Exception:
            with best_effort(self.left_arm.logger, "cancelling the left arm reset"):
                self.left_arm._cancel_reset()
            raise

    @property
    def rt_moving(self) -> bool:
        # Evaluate both properties: each refreshes its native reset watchdog.
        left = self.left_arm.rt_moving
        right = self.right_arm.rt_moving
        return left or right

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}
        left_sent = self.left_arm.send_action(left)
        right_sent = self.right_arm.send_action(right)
        return {
            **{f"left_{k}": v for k, v in left_sent.items()},
            **{f"right_{k}": v for k, v in right_sent.items()},
        }

    def disconnect(self, *, reset: bool = True) -> None:
        """Return both arms together, then stop both controllers before peripheral cleanup."""
        try:
            if reset and self.config.reset_on_exit and self.is_connected:
                self.logger.info(
                    "B601 exit reset: returning both arms to the configured poses; press Ctrl+C again to abort"
                )
                self.reset_to_initial_position()
                while self.rt_moving:
                    time.sleep(0.01)
                self.logger.info("B601 exit reset complete; disconnecting both arms")
        except KeyboardInterrupt:
            self.logger.warn("B601 exit reset interrupted; disconnecting both arms immediately")
        except Exception as error:
            self.logger.error(f"B601 exit reset failed; disconnecting both arms: {error}")
        finally:
            # Stop both controllers before either arm's slower camera/gripper teardown.
            for arm in (self.left_arm, self.right_arm):
                with best_effort(arm.logger, "stopping the native arm"):
                    if arm._native is not None:
                        arm._native.stop()
            try:
                self.right_arm.disconnect(reset=False)
            finally:
                self.left_arm.disconnect(reset=False)
