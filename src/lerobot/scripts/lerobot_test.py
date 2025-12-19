#!/usr/bin/env python
#
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
Dev sandbox script for quickly importing and experimenting with Robot devices.

This is intentionally lightweight and safe-by-default:
- it can import/register built-in robots/teleoperators/cameras
- it does NOT connect to hardware unless you do so manually in the REPL

Examples:

```bash
# Verify imports/registering work
lerobot-test

# Print discovered/registered type names
lerobot-test --list all

# Start a Python REPL with common symbols pre-imported
lerobot-test --repl
```
"""

from __future__ import annotations

import argparse
import code
from collections import deque
import importlib
import logging
from dataclasses import asdict, dataclass, is_dataclass
from pprint import pformat
from typing import Any

import numpy as np
import time

from lerobot.cameras.configs import CameraConfig
from lerobot.configs import parser
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.utils import init_logging
# Delay import of rerun and visualization utils - only needed for camera tests
# import rerun as rr
# from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
# from lerobot.processor import make_default_processors

logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for testing devices.
    
    At least one of robot, camera, or teleop must be specified.
    """
    # Robot to test (optional)
    robot: RobotConfig | None = None
    # Camera to test (optional)  
    camera: CameraConfig | None = None
    # Teleoperator to test (optional)
    teleop: TeleoperatorConfig | None = None
    # Camera read mode for camera tests: "async" or "sync"
    camera_read_mode: str = "async"
    # Camera timing window size (samples)
    camera_timing_window: int = 120
    # Camera timing log period in seconds
    camera_timing_log_period_s: float = 1.0
    # Do not auto-import third-party plugins
    no_plugins: bool = False


def _test_robot(
    robot_config: RobotConfig,
    camera_read_mode: str = "async",
    camera_timing_window: int = 120,
    camera_timing_log_period_s: float = 1.0,
):
    """Test a robot by continuously fetching and printing observations."""
    from lerobot.robots import make_robot_from_config
    
    robot = None
    try:
        logger.info(f"Creating {robot_config.type} robot...")
        robot = make_robot_from_config(robot_config)
        
        # Connect to robot with error handling
        try:
            logger.info("Connecting to robot...")
            robot.connect(go_to_start=True)
            logger.info("✓ Robot connected. Starting observation loop. Press Ctrl+C to stop.")
            print()
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            logger.exception("Connection failed")
            raise
        
        # Send initial TCP pose action (only for CARTESIAN_MOTION_FORCE mode)
        try:
            from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode
            if hasattr(robot.config, 'control_mode') and robot.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
                target_tcp_pose = [0.68783, -0.115326, 0.328386, 0.004519, 0.003284, 0.999984, 0.001275]
                # Format: [x, y, z, qw, qx, qy, qz]
                action = {
                    "tcp.x": target_tcp_pose[0],
                    "tcp.y": target_tcp_pose[1],
                    "tcp.z": target_tcp_pose[2],
                    "tcp.qw": target_tcp_pose[3],
                    "tcp.qx": target_tcp_pose[4],
                    "tcp.qy": target_tcp_pose[5],
                    "tcp.qz": target_tcp_pose[6],
                    "gripper.pos": 0.0,  # Keep gripper at current position or set to 0
                }
                logger.info(f"Sending initial TCP pose action: {target_tcp_pose}")
                time.sleep(0.5)
                robot.send_action(action)
                logger.info("✓ Initial TCP pose action sent.")
                time.sleep(0.5)  # Give robot time to start moving
        except (ImportError, AttributeError):
            # Not a Flexiv robot or control_mode not available, skip
            pass
        except Exception as e:
            logger.warning(f"Failed to send initial TCP pose action: {e}")
            # Continue anyway - this is not critical
        
        # Continuous observation loop
        loop_count = 0
        try:
            while True:
                try:
                    obs = robot.get_observation()
                    
                    # Print observation data (same line update)
                    # print(f"\r[{loop_count:06d}] ", end="", flush=True)
                    # for key, value in obs.items():
                    #     if isinstance(value, (int, float)):
                    #         print(f"{key}={value:8.4f} ", end="", flush=True)
                    #     elif isinstance(value, np.ndarray):
                    #         print(f"{key}=array{value.shape} ", end="", flush=True)
                    #     else:
                    #         print(f"{key}={str(value)[:20]} ", end="", flush=True)
                    
                    loop_count += 1
                    time.sleep(0.1)  # 10 Hz update rate
                except Exception as e:
                    logger.error(f"Error getting observation: {e}")
                    # Continue loop - don't break on single observation error
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print()  # New line after same-line updates
            logger.info("Observation loop interrupted by user")
            
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error during robot testing: {e}")
        logger.exception("Test failed")
    finally:
        # Final safety check - ensure robot is safely disconnected
        if robot is not None:
            try:
                if robot.is_connected:
                    logger.info("Safely disconnecting robot...")
                    robot.disconnect()
                    logger.info("✓ Robot safely disconnected.")
            except Exception as e:
                logger.error(f"Error during robot disconnect: {e}")
                # Force cleanup for Flexiv robots
                try:
                    if hasattr(robot, '_robot') and robot._robot is not None:
                        logger.warning("Attempting emergency stop...")
                        robot._robot.Stop()
                except Exception as stop_error:
                    logger.error(f"Error during emergency stop: {stop_error}")


def _test_camera(
    camera_config: CameraConfig,
    camera_read_mode: str = "async",
    camera_timing_window: int = 120,
    camera_timing_log_period_s: float = 1.0,
):
    """Test a camera by continuously reading and displaying frames."""
    # Lazy import of heavy dependencies - only needed for camera tests
    import rerun as rr
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
    from lerobot.processor import make_default_processors
    from lerobot.cameras import make_cameras_from_configs
    
    init_rerun(session_name=f"{camera_config.type}-test")
    config = {camera_config.type: camera_config}
    
    _, _, robot_observation_processor = make_default_processors()
    
    cameras = make_cameras_from_configs(config)
    logger.info(f"Initializing cameras: {cameras}")
    for cam_key, cam in cameras.items():
        logger.info(f"Connecting to {cam_key}...")
        cam.connect()
        logger.info(f"Connected to {cam_key}.")
    
    timing_history: dict[str, deque[float]] = {
        cam_key: deque(maxlen=max(1, int(camera_timing_window))) for cam_key in cameras.keys()
    }
    total_history: deque[float] = deque(maxlen=max(1, int(camera_timing_window)))
    last_timing_log_t = time.perf_counter()
    
    def get_observation() -> dict[str, Any]:
        nonlocal last_timing_log_t
        t0 = time.perf_counter()
        obs_dict = {}
        camera_times = {}
        for cam_key, cam in cameras.items():
            start = time.perf_counter()
            if camera_read_mode == "sync":
                data = cam.read()
            else:
                data = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            if isinstance(data, dict):
                for out_k, out_v in data.items():
                    obs_dict[f"{cam_key}.{out_k}"] = out_v
            else:
                obs_dict[cam_key] = data
            camera_times[cam_key] = dt_ms
        
        total_dt_ms = (time.perf_counter() - t0) * 1e3
        now = time.perf_counter()
        for cam_key, dt_ms in camera_times.items():
            timing_history[cam_key].append(float(dt_ms))
        total_history.append(float(total_dt_ms))
        
        if (now - last_timing_log_t) >= float(camera_timing_log_period_s):
            last_timing_log_t = now
            mode_label = "read" if camera_read_mode == "sync" else "async_read(cached)"
            cam_lines: list[str] = []
            total_last = 0.0
            total_avg = 0.0
            total_p95 = 0.0
            total_max = 0.0
            if total_history:
                tarr = np.asarray(total_history, dtype=np.float32)
                total_last = float(tarr[-1])
                total_avg = float(tarr.mean())
                total_p95 = float(np.percentile(tarr, 95))
                total_max = float(tarr.max())
            
            cam_keys = sorted(timing_history.keys())
            name_w = max((len(k) for k in cam_keys), default=0)
            name_w = min(max(name_w, 10), 32)
            
            for cam_key in cam_keys:
                hist = timing_history[cam_key]
                if not hist:
                    continue
                arr = np.asarray(hist, dtype=np.float32)
                last = float(arr[-1])
                avg = float(arr.mean())
                p95 = float(np.percentile(arr, 95))
                mx = float(arr.max())
                cam_lines.append(
                    f"  - {cam_key:<{name_w}}  last {last:7.3f} ms  avg {avg:7.3f} ms  "
                    f"p95 {p95:7.3f} ms  max {mx:7.3f} ms"
                )
            
            n = max((len(h) for h in timing_history.values()), default=0)
            header = (
                f"📷 get_observation {mode_label} stats "
                f"(window={int(camera_timing_window)}, n={n}, every={camera_timing_log_period_s:.1f}s): "
                f"total last {total_last:.3f} ms  avg {total_avg:.3f} ms  "
                f"p95 {total_p95:.3f} ms  max {total_max:.3f} ms"
            )
            lines = [header, *cam_lines] if cam_lines else [header]
            logger.info("\n".join(lines))
        return obs_dict
    
    try:
        while True:
            obs = get_observation()
            obs_processed = robot_observation_processor(obs)
            log_rerun_data(observation=obs_processed)
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        for cam_key, cam in cameras.items():
            cam.disconnect()
        rr.rerun_shutdown()
        logger.info("Exiting camera test loop...")


def _test_teleop(teleop_config: TeleoperatorConfig):
    """Test a teleoperator by continuously reading and displaying actions."""
    from lerobot.teleoperators import make_teleoperator_from_config
    
    teleop = make_teleoperator_from_config(teleop_config)
    logger.info(f"Connecting to {teleop_config.type}...")
    teleop.connect()
    logger.info(f"Connected to {teleop_config.type}. Press Ctrl+C to exit.")
    print()
    try:
        while True:
            action = teleop.get_action()
            # Format action values with fixed width for stable display
            x = action.get("x", 0.0)
            y = action.get("y", 0.0)
            z = action.get("z", 0.0)
            roll = action.get("roll", 0.0)
            pitch = action.get("pitch", 0.0)
            yaw = action.get("yaw", 0.0)
            # Print on same line using carriage return
            print(
                f"\rPos: x={x:+7.3f} y={y:+7.3f} z={z:+7.3f} | "
                f"Rot: r={roll:+7.3f} p={pitch:+7.3f} y={yaw:+7.3f}",
                end="", flush=True
            )
            time.sleep(1 / 200)
    except KeyboardInterrupt:
        pass
    finally:
        print()  # New line after same-line updates
        logger.info("Disconnecting from teleoperator...")
        teleop.disconnect()


def _safe_import(module: str) -> tuple[bool, str | None]:
    """Import a module and return (ok, error_message)."""
    try:
        importlib.import_module(module)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _register_builtin_devices(device_types: list[str] | None = None) -> dict[str, dict[str, str | None]]:
    """
    Import config modules so draccus ChoiceRegistry subclasses get registered.

    Args:
        device_types: Optional list of device types to register. If None, registers all.
                     Valid values: "robots", "cameras", "teleoperators"

    Returns a dict describing import errors (if any).
    """
    # NOTE: Importing *config* modules is preferred over importing the full package,
    # because some packages' __init__ import hardware-backed implementations.
    all_modules: dict[str, list[str]] = {
        "cameras": [
            "lerobot.cameras.opencv.configuration_opencv",
            "lerobot.cameras.realsense.configuration_realsense",
            "lerobot.cameras.xense.configuration_xense",
        ],
        "robots": [
            "lerobot.robots.koch_follower.config_koch_follower",
            "lerobot.robots.so100_follower.config_so100_follower",
            "lerobot.robots.so101_follower.config_so101_follower",
            "lerobot.robots.bi_so100_follower.config_bi_so100_follower",
            "lerobot.robots.bi_arx5.config_bi_arx5",
            "lerobot.robots.arx5_follower.config_arx5_follower",
            "lerobot.robots.flexiv_rizon4.config_flexiv_rizon4",
        ],
        "teleoperators": [
            "lerobot.teleoperators.keyboard.configuration_keyboard",
            "lerobot.teleoperators.koch_leader.config_koch_leader",
            "lerobot.teleoperators.so100_leader.config_so100_leader",
            "lerobot.teleoperators.so101_leader.config_so101_leader",
            "lerobot.teleoperators.mock_teleop",
            "lerobot.teleoperators.gamepad.configuration_gamepad",
            "lerobot.teleoperators.homunculus.config_homunculus",
            "lerobot.teleoperators.bi_so100_leader.config_bi_so100_leader",
            "lerobot.teleoperators.pico4.config_pico4",
            "lerobot.teleoperators.spacemouse.config_spacemouse",
        ],
    }

    # Filter modules based on device_types if specified
    if device_types is not None:
        modules = {k: v for k, v in all_modules.items() if k in device_types}
    else:
        modules = all_modules

    errors: dict[str, dict[str, str | None]] = {k: {} for k in modules}
    for group, paths in modules.items():
        for path in paths:
            ok, err = _safe_import(path)
            if not ok:
                errors[group][path] = err
            else:
                errors[group][path] = None
    return errors


def _list_registered_choices() -> dict[str, list[str]]:
    """
    Best-effort listing of draccus ChoiceRegistry-registered type names.
    """
    # These imports are local so `lerobot-test` can still run import-only mode
    # even when some optional deps are missing.
    from lerobot.cameras.configs import CameraConfig
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    def _iter_subclasses(base: type) -> list[type]:
        out: list[type] = []
        stack = list(base.__subclasses__())
        while stack:
            sub = stack.pop()
            out.append(sub)
            stack.extend(sub.__subclasses__())
        return out

    def _choice_name(base: type, sub: type) -> str | None:
        # 1) Try draccus ChoiceRegistry API patterns
        for owner in (base, sub):
            meth = getattr(owner, "get_choice_name", None)
            if callable(meth):
                try:
                    name = meth(sub)  # common signature: get_choice_name(cls)
                except TypeError:
                    try:
                        name = meth()  # fallback: get_choice_name()
                    except Exception:
                        continue
                except Exception:
                    continue
                if name is not None:
                    s = str(name)
                    if s and s != sub.__name__:
                        return s

        # 2) Try common attribute names used by registries/decorators
        for attr in (
            "choice_name",
            "_choice_name",
            "__choice_name__",
            "CHOICE_NAME",
            "_draccus_choice_name",
        ):
            v = getattr(sub, attr, None)
            if isinstance(v, str) and v:
                return v
        return None

    def extract_choice_names(cls: type) -> list[str]:
        candidates: list[str] = []
        # Prefer registry dicts if exposed
        for attr in ("_registry", "registry", "_ChoiceRegistry__registry"):
            reg = getattr(cls, attr, None)
            if isinstance(reg, dict):
                candidates.extend([str(k) for k in reg.keys()])

        # Fallback: walk subclasses and infer choice names
        if not candidates:
            for sub in _iter_subclasses(cls):
                name = _choice_name(cls, sub)
                if name:
                    candidates.append(name)

        # de-duplicate, keep deterministic order
        return sorted(set(candidates))

    return {
        "robots": extract_choice_names(RobotConfig),
        "teleoperators": extract_choice_names(TeleoperatorConfig),
        "cameras": extract_choice_names(CameraConfig),
    }


def _build_sample_configs() -> dict[str, Any]:
    """
    Provide a handful of sample config instances for quick experimentation.
    """
    samples: dict[str, Any] = {}

    # Cameras
    try:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        samples["opencv_camera_cfg"] = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
    except Exception as e:
        samples["opencv_camera_cfg_error"] = f"{type(e).__name__}: {e}"

    try:
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        samples["realsense_camera_cfg"] = RealSenseCameraConfig(
            serial_number_or_name="000000000000", fps=30, width=640, height=480
        )
    except Exception as e:
        samples["realsense_camera_cfg_error"] = f"{type(e).__name__}: {e}"

    try:
        from lerobot.cameras.xense.configuration_xense import XenseCameraConfig, XenseOutputType

        samples["xense_camera_cfg"] = XenseCameraConfig(
            serial_number="OG000456",
            fps=30,
            output_types=[XenseOutputType.DIFFERENCE],
        )
    except Exception as e:
        samples["xense_camera_cfg_error"] = f"{type(e).__name__}: {e}"

    # Teleoperators
    try:
        from lerobot.teleoperators.mock_teleop import MockTeleopConfig

        samples["mock_teleop_cfg"] = MockTeleopConfig()
    except Exception as e:
        samples["mock_teleop_cfg_error"] = f"{type(e).__name__}: {e}"

    try:
        from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

        samples["keyboard_teleop_cfg"] = KeyboardTeleopConfig()
    except Exception as e:
        samples["keyboard_teleop_cfg_error"] = f"{type(e).__name__}: {e}"

    # Robots (configs only; do NOT connect/instantiate hardware)
    try:
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

        # Note: ports/IDs are placeholders for dev testing
        samples["so101_robot_cfg"] = SO101FollowerConfig(port="/dev/ttyUSB0")
    except Exception as e:
        samples["so101_robot_cfg_error"] = f"{type(e).__name__}: {e}"

    try:
        from lerobot.robots.arx5_follower.config_arx5_follower import ARX5FollowerConfig

        samples["arx5_robot_cfg"] = ARX5FollowerConfig(port="/dev/ttyUSB0")
    except Exception as e:
        samples["arx5_robot_cfg_error"] = f"{type(e).__name__}: {e}"

    try:
        from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config

        samples["bi_arx5_robot_cfg"] = BiARX5Config(enable_tactile_sensors=False)
    except Exception as e:
        samples["bi_arx5_robot_cfg_error"] = f"{type(e).__name__}: {e}"

    return samples


def _to_printable(x: Any) -> Any:
    if is_dataclass(x):
        try:
            return asdict(x)
        except Exception:
            return str(x)
    return x


@parser.wrap()
def test_with_config(cfg: TestConfig):
    """Test device using configuration interface.
    
    Examples:
        lerobot-test --robot.type=flexiv_rizon4
        lerobot-test --teleop.type=spacemouse
        lerobot-test --camera.type=opencv
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Determine what device type to test and only register that type for faster startup
    device_types_to_register = []
    if cfg.robot is not None:
        device_types_to_register.append("robots")
    if cfg.camera is not None:
        device_types_to_register.append("cameras")
    if cfg.teleop is not None:
        device_types_to_register.append("teleoperators")
    
    # Only register third-party plugins if needed (they can be slow to import)
    # Skip if no device specified or if explicitly disabled
    if device_types_to_register and not cfg.no_plugins:
        try:
            from lerobot.utils.import_utils import register_third_party_devices
            register_third_party_devices()
        except Exception:
            logger.exception("Failed importing third-party plugins.")
    
    # Only register the device type we need (much faster than registering all)
    if device_types_to_register:
        _register_builtin_devices(device_types_to_register)
    else:
        # If nothing specified, register all for --list functionality
        _register_builtin_devices()
    
    # Determine what to test based on config
    if cfg.robot is not None:
        _test_robot(cfg.robot, cfg.camera_read_mode, cfg.camera_timing_window, cfg.camera_timing_log_period_s)
    elif cfg.camera is not None:
        _test_camera(cfg.camera, cfg.camera_read_mode, cfg.camera_timing_window, cfg.camera_timing_log_period_s)
    elif cfg.teleop is not None:
        _test_teleop(cfg.teleop)
    else:
        # If nothing specified, just show help or list
        logger.warning("No device specified. Use --robot.type, --camera.type, or --teleop.type")
        logger.info("Use --list to see available devices")


def main(argv: list[str] | None = None) -> None:
    import sys
    if argv is None:
        argv = sys.argv[1:]
    
    # Check for --list or --repl (special commands that don't need config)
    if "--list" in argv or "--repl" in argv or len(argv) == 0:
        # Use argparse for these special commands
        parser = argparse.ArgumentParser(
            prog="lerobot-test",
            description="Dev sandbox for importing and experimenting with LeRobot robots/teleoperators/cameras.",
        )
        parser.add_argument(
            "--list",
            choices=["robots", "teleoperators", "cameras", "all"],
            default=None,
            help="Print registered type names (best-effort).",
        )
        parser.add_argument(
            "--repl",
            action="store_true",
            help="Start a Python REPL with common symbols and sample configs preloaded.",
        )
        parser.add_argument(
            "--no-plugins",
            action="store_true",
            help="Do not auto-import third-party plugins (lerobot_robot_*, lerobot_camera_*, lerobot_teleoperator_*).",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable INFO logging.",
        )
        args = parser.parse_args(argv)

        # Default to INFO when running device tests so timing logs show up.
        log_level = logging.INFO if args.verbose else logging.WARNING
        logging.basicConfig(level=log_level, force=True)

        # Determine which device types to register based on --list argument
        if args.list == "all":
            device_types_to_register = None  # Register all
        elif args.list is not None:
            device_types_to_register = [args.list]  # Only register the requested type
        else:
            device_types_to_register = None  # For --repl or no args, register all

        # Only register third-party plugins if needed (they can be slow to import)
        if not args.no_plugins and device_types_to_register is not None:
            try:
                from lerobot.utils.import_utils import register_third_party_devices
                register_third_party_devices()
            except Exception:
                logger.exception("Failed importing third-party plugins.")

        import_errors = _register_builtin_devices(device_types_to_register)

        if args.list is not None:
            choices = _list_registered_choices()
            if args.list == "all":
                for group in ("robots", "teleoperators", "cameras"):
                    print(f"\n[{group}]")
                    for name in choices.get(group, []):
                        print(f"- {name}")
            else:
                print(f"\n[{args.list}]")
                for name in choices.get(args.list, []):
                    print(f"- {name}")

        # Show import errors
        failed = {
            group: {m: err for m, err in errs.items() if err is not None}
            for group, errs in import_errors.items()
        }
        failed = {g: v for g, v in failed.items() if v}
        if failed:
            print("\n[import errors]")
            for group, errs in failed.items():
                print(f"- {group}:")
                for mod, err in errs.items():
                    print(f"  - {mod}: {err}")

        if args.repl:
            _start_repl()
        
        return
    
    # Use new config-based interface for device testing
    test_with_config()


def _start_repl() -> None:
    """Start a Python REPL with common symbols and sample configs preloaded."""
    # Common imports exposed in the interactive namespace.
    from lerobot.cameras import Camera, CameraConfig, make_cameras_from_configs
    from lerobot.robots import Robot, RobotConfig, make_robot_from_config
    from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config

    # Sample configs (best-effort)
    samples = _build_sample_configs()

    banner_lines = [
        "LeRobot dev REPL",
        "",
        "Available symbols:",
        "  - RobotConfig, TeleoperatorConfig, CameraConfig",
        "  - make_robot_from_config(cfg), make_teleoperator_from_config(cfg), make_cameras_from_configs(dict)",
        "  - samples (dict): a few ready-to-use config instances",
        "",
        "Tip: configs do NOT connect to hardware by default. You can instantiate/connect manually.",
    ]
    local_vars = {
        "Robot": Robot,
        "RobotConfig": RobotConfig,
        "Teleoperator": Teleoperator,
        "TeleoperatorConfig": TeleoperatorConfig,
        "Camera": Camera,
        "CameraConfig": CameraConfig,
        "make_robot_from_config": make_robot_from_config,
        "make_teleoperator_from_config": make_teleoperator_from_config,
        "make_cameras_from_configs": make_cameras_from_configs,
        "samples": {k: _to_printable(v) for k, v in samples.items()},
        "_raw_samples": samples,
    }
    code.interact(banner="\n".join(banner_lines), local=local_vars)


if __name__ == "__main__":
    main()
