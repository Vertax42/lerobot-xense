"""Cartesian C++ RT driver configuration for the DM and RS B601 arms."""

import math
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.configs import CameraConfig
from lerobot.grippers import GripperConfig
from lerobot.robots.config import RobotConfig

DEFAULT_RS_TACCAP_URDF = "RS_Taccap/urdf/rebot_b601_rs_taccap_v2_gripper.urdf"


@RobotConfig.register_subclass("rebot_b601_rt")
@dataclass
class RebotB601RTConfig(RobotConfig):
    motor_family: str = "dm"  # dm or rs
    channel: str = "can0"  # SocketCAN interface, or /dev/ttyACM* for DM serial bridge
    dm_serial_baud: int = 921600
    control_mode: str = "pos_vel"  # position/velocity loop, or mit
    interpolate_cartesian_targets: bool = True  # native timed Cartesian substeps, in both modes
    mit_velocity_feedforward: bool = True  # False sends MIT dq=0 while retaining position interpolation/Kd
    motor_configs: list[dict[str, int | float | str]] = field(default_factory=list)
    urdf_path: str | None = None
    # Empty uses the selected URDF's default tool frame.
    end_frame: str | None = None
    # An external gripper uses its own driver instead of the seventh arm motor.
    # taccap_follower selects RS_Taccap for either motor family; None keeps
    # the original DM/RS model and its optional motor gripper.
    gripper: GripperConfig | None = None
    # Whether the native arm controller drives a seventh CAN motor.
    has_gripper: bool | None = None
    max_joint_speed: float = 0.5  # rad/s, fallback for all joints and native gripper speed
    max_joint_speeds: list[float] | None = None  # optional [J1, ..., J6], replaces arm scalar limit
    coordinate_cartesian_speed: bool = True  # scale TCP translation/rotation together before bounded IK
    command_timeout_sec: float = 0.5
    reset_position_rad: list[float] | None = None  # six arm joints; A-button target
    reset_on_exit: bool = False  # teleoperate: return to reset pose on Ctrl+C/normal loop completion
    reset_joint_speed: float = 0.5  # rad/s, additionally capped by each joint's configured limit
    reset_timeout_sec: float = 30.0
    reset_tolerance_rad: float = 0.03
    trace_joint_tracking: bool = False
    trace_joint_tracking_hz: float = 2.0
    use_joint_observation: bool = False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    @classmethod
    def get_known_choices(cls) -> dict[str, type]:
        # This concrete config also appears inside BiRebotB601RTConfig. Exposing
        # the root RobotConfig registry there recursively offers the dual-arm
        # config inside each arm while Draccus builds CLI arguments.
        return {cls.get_choice_name(cls): cls}

    def __post_init__(self):
        super().__post_init__()
        self.motor_family = self.motor_family.lower()
        if self.motor_family not in ("dm", "rs"):
            raise ValueError("motor_family must be 'dm' or 'rs'")
        self.control_mode = self.control_mode.lower()
        if self.control_mode not in ("mit", "pos_vel"):
            raise ValueError("control_mode must be 'mit' or 'pos_vel'")
        if self.has_gripper is None:
            self.has_gripper = self.gripper is None and not (
                self.uses_taccap_model or self.end_frame == "taccap_eef_link"
            )
        if self.gripper is not None and self.has_gripper:
            raise ValueError("An external gripper requires has_gripper=False for the native arm controller")
        if not self.channel:
            raise ValueError("channel must name a SocketCAN interface or DM serial device")
        if self.motor_family == "rs" and self.channel.startswith("/"):
            raise ValueError("RS requires a SocketCAN interface such as can0")
        if self.dm_serial_baud <= 0:
            raise ValueError("dm_serial_baud must be positive")
        if self.motor_configs and len(self.motor_configs) != 7:
            raise ValueError("motor_configs must contain seven entries")
        if any(not math.isfinite(v) or v <= 0 for v in (self.max_joint_speed, self.command_timeout_sec)):
            raise ValueError("speed and timeout must be positive and finite")
        if self.max_joint_speeds is not None and (
            len(self.max_joint_speeds) != 6 or any(not math.isfinite(v) or v <= 0 for v in self.max_joint_speeds)
        ):
            raise ValueError("max_joint_speeds must contain six finite positive speeds in rad/s")
        if not math.isfinite(self.reset_joint_speed) or self.reset_joint_speed <= 0:
            raise ValueError("reset_joint_speed must be positive and finite")
        if self.reset_position_rad is not None and (
            len(self.reset_position_rad) != 6 or not all(math.isfinite(q) for q in self.reset_position_rad)
        ):
            raise ValueError("reset_position_rad must contain six finite joint angles in radians")
        if self.reset_on_exit and self.reset_position_rad is None:
            raise ValueError("reset_on_exit requires reset_position_rad")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.reset_timeout_sec,
                self.reset_tolerance_rad,
                self.trace_joint_tracking_hz,
            )
        ):
            raise ValueError("reset timeout, reset tolerance and tracking log rate must be positive and finite")
        if self.urdf_path is not None and not Path(self.urdf_path).is_file():
            raise ValueError(f"URDF does not exist: {self.urdf_path}")

    @property
    def default_urdf(self) -> str:
        """Bundled model selected by the mounted gripper and motor family."""
        if self.gripper is not None and self.gripper.type == "taccap_follower":
            return DEFAULT_RS_TACCAP_URDF
        family = self.motor_family.upper()
        return f"{family}/urdf/ReBot_Arm_{family}.urdf"

    @property
    def uses_taccap_model(self) -> bool:
        return Path(self.urdf_path or self.default_urdf).name == Path(DEFAULT_RS_TACCAP_URDF).name
