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
from dataclasses import asdict, is_dataclass
from typing import Any
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
import time
import rerun as rr
import numpy as np

logger = logging.getLogger(__name__)


def test(
    device: str,
    camera_read_mode: str = "async",
    camera_timing_window: int = 120,
    camera_timing_log_period_s: float = 1.0,
):
    if device == "xense-camera":
        from lerobot.cameras.xense import XenseCameraConfig
        from lerobot.cameras.realsense import RealSenseCameraConfig
        init_rerun(session_name="xense-camera-test")
        config = {
            "xense_camera": XenseCameraConfig(
                serial_number="OG000456",
                fps=30,
                output_types=["force_resultant"],
                # image output: [rectify, difference, depth]
                # force output: [force, force_norm, force_resultant]
                # marker output: [marker_2d]
                # mesh output: [mesh_3d, mesh_3d_init, mesh_3d_flow]
            ),
            "realsense_camera": RealSenseCameraConfig(
                serial_number_or_name="230422271416", fps=60, width=640, height=480
            ),
        }

        _, _, robot_observation_processor = make_default_processors()

        from lerobot.cameras import make_cameras_from_configs
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
                    # Blocking call that waits on the camera hardware / SDK.
                    data = cam.read()
                    print(f"DEBUG: {cam_key} read data shape: {data.shape}")
                else:
                    # NOTE: Most camera implementations use a background thread; async_read()
                    # typically returns the *latest cached* frame and will often be near-0ms
                    # once the background reader is running.
                    data = cam.async_read()
                dt_ms = (time.perf_counter() - start) * 1e3
                # Xense camera may return:
                # - np.ndarray when only one output type is requested
                # - dict[str, np.ndarray] when multiple output types are requested
                if isinstance(data, dict):
                    for out_k, out_v in data.items():
                        obs_dict[f"{cam_key}.{out_k}"] = out_v
                else:
                    # Single output: use fixed key (cam_key) so Rerun viewer doesn't need
                    # manual adjustment when switching output_types between runs.
                    obs_dict[cam_key] = data
                camera_times[cam_key] = dt_ms

            total_dt_ms = (time.perf_counter() - t0) * 1e3
            # Update timing history and log aggregated stats at a fixed cadence.
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
                name_w = min(max(name_w, 10), 32)  # keep it sane

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
                # Rerun logging is expensive; throttle the visualization loop.
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            pass
        finally:
            for cam_key, cam in cameras.items():
                cam.disconnect()
            rr.rerun_shutdown()
            logger.info("Exiting xense-camera test loop...")

    elif device == "spacemouse":
        from lerobot.teleoperators.spacemouse import SpacemouseTeleop
        from lerobot.teleoperators.spacemouse.config_spacemouse import SpacemouseConfig
        config = SpacemouseConfig(
            pos_sensitivity=0.8,
            ori_sensitivity=1.5,
            deadzone=0.0,
            max_value=500,
            frequency=200,
            filter_window_size=3,
        )

        teleop = SpacemouseTeleop(config)
        logger.info("Connecting to spacemouse...")
        teleop.connect()
        logger.info("Connected to spacemouse. Press Ctrl+C to exit.")
        print()  # Empty line for status display
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
            logger.info("Disconnecting from spacemouse...")
            teleop.disconnect()
    elif device == "pico4":
        from lerobot.teleoperators.pico4 import Pico4
        teleop = Pico4()
        teleop.connect()
        teleop.disconnect()
    else:
        raise ValueError(f"Invalid device for testing: {device}")


def _safe_import(module: str) -> tuple[bool, str | None]:
    """Import a module and return (ok, error_message)."""
    try:
        importlib.import_module(module)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _register_builtin_devices() -> dict[str, dict[str, str | None]]:
    """
    Import config modules so draccus ChoiceRegistry subclasses get registered.

    Returns a dict describing import errors (if any).
    """
    # NOTE: Importing *config* modules is preferred over importing the full package,
    # because some packages' __init__ import hardware-backed implementations.
    modules: dict[str, list[str]] = {
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


def main(argv: list[str] | None = None) -> None:
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
        "--device",
        choices=["spacemouse", "pico4", "xense-camera", "xense-flare", "arx5_follower", "dobot_nova5"],
        default=None,
        help="Test device.",
    )
    parser.add_argument(
        "--camera-read-mode",
        choices=["async", "sync"],
        default="sync",
        help=(
            "For camera device tests: use async_read (default, often cached/near-0ms after warmup) "
            "or sync read (blocking call that measures actual camera read latency)."
        ),
    )
    parser.add_argument(
        "--camera-timing-window",
        type=int,
        default=120,
        help="Rolling window size (samples) for camera timing stats.",
    )
    parser.add_argument(
        "--camera-timing-log-period",
        type=float,
        default=1.0,
        help="Seconds between aggregated camera timing logs.",
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
    # Use force=True so it still works if another library configured logging already.
    log_level = logging.INFO if (args.verbose or args.device is not None) else logging.WARNING
    logging.basicConfig(level=log_level, force=True)

    if not args.no_plugins:
        try:
            from lerobot.utils.import_utils import register_third_party_devices

            register_third_party_devices()
        except Exception:
            logger.exception("Failed importing third-party plugins.")

    import_errors = _register_builtin_devices()

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

    # Always show import issues (if any) so it's obvious what failed in the current environment.
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
    else:
        test(
            args.device,
            camera_read_mode=args.camera_read_mode,
            camera_timing_window=args.camera_timing_window,
            camera_timing_log_period_s=args.camera_timing_log_period,
        )


if __name__ == "__main__":
    main()
