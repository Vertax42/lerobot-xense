# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

# Import mock_teleop FIRST to register its config with draccus ChoiceRegistry
# This must happen before any other imports that might use TeleoperatorConfig
import time
import traceback
from dataclasses import asdict, dataclass
from pprint import pformat
import math
from typing import Any

import rerun as rr
import spdlog

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,
)  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    arx5_follower,
    bi_so100_follower,
    bi_arx5,
    flexiv_rizon4,  # noqa: F401
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    mock_teleop,
    spacemouse,
    pico4,
)

from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Create global logger for teleoperate script
logger = spdlog.ConsoleLogger("Teleoperate")


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi)
    # its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 100
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    debug_timing: bool = False
    # Dryrun mode: print actions without sending to robot
    dryrun: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    display_data: bool = False,
    duration: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


def arx5_teleop_loop(
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    display_data: bool = False,
    duration: float | None = None,
    debug_timing: bool = False,
):
    """
    Teleop loop for ARX5 robots (both single-arm and bimanual).

    This function continuously reads robot state, processes observations through optional
    pipelines, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Supports:
    - Single arm mode (arx5_follower): robot.arm
    - Bimanual mode (bi_arx5): robot.left_arm, robot.right_arm
    """
    start = time.perf_counter()
    timing_stats = {
        "robot_obs_times": [],
        "camera_obs_times": {},
        "total_obs_times": [],
        "loop_times": [],
    }

    # Detect arm mode: single arm vs bimanual
    is_bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")
    is_single_arm = hasattr(robot, "arm") and not is_bimanual

    if not is_bimanual and not is_single_arm:
        raise ValueError("Robot must have either 'arm' (single) or 'left_arm'/'right_arm' (bimanual)")

    # Identify camera keys
    camera_keys = [
        key for key in robot.observation_features.keys() if not key.endswith(".pos")
    ]
    for cam_key in camera_keys:
        timing_stats["camera_obs_times"][cam_key] = []

    while True:
        loop_start = time.perf_counter()

        # Time the complete observation acquisition
        obs_start = time.perf_counter()

        # Get robot state (joints) timing
        robot_state_start = time.perf_counter()

        if is_bimanual:
            left_joint_state = robot.left_arm.get_joint_state()
            right_joint_state = robot.right_arm.get_joint_state()
        else:  # single arm
            joint_state = robot.arm.get_joint_state()

        robot_obs_time = time.perf_counter() - robot_state_start
        timing_stats["robot_obs_times"].append(robot_obs_time * 1000)  # Convert to ms

        # Get camera observations timing
        camera_obs_start = time.perf_counter()
        camera_observations = {}
        camera_times = {}
        for cam_key, cam in robot.cameras.items():
            cam_start = time.perf_counter()
            camera_observations[cam_key] = cam.async_read()
            cam_time = time.perf_counter() - cam_start
            cam_time_ms = cam_time * 1000
            camera_times[cam_key] = cam_time_ms
            timing_stats["camera_obs_times"][cam_key].append(cam_time_ms)

        total_camera_time = time.perf_counter() - camera_obs_start
        total_camera_time_ms = total_camera_time * 1000

        # Build complete observation dict (similar to robot.get_observation())
        raw_observation = {}

        if is_bimanual:
            # Add left arm joint observations
            left_pos = left_joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"left_joint_{i+1}.pos"] = float(left_pos[i])
            raw_observation["left_gripper.pos"] = float(left_joint_state.gripper_pos)

            # Add right arm joint observations
            right_pos = right_joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"right_joint_{i+1}.pos"] = float(right_pos[i])
            raw_observation["right_gripper.pos"] = float(right_joint_state.gripper_pos)
        else:  # single arm
            # Add single arm joint observations
            pos = joint_state.pos().copy()
            for i in range(6):
                raw_observation[f"joint_{i+1}.pos"] = float(pos[i])
            raw_observation["gripper.pos"] = float(joint_state.gripper_pos)

        # Add camera observations
        raw_observation.update(camera_observations)

        total_obs_time = time.perf_counter() - obs_start
        timing_stats["total_obs_times"].append(total_obs_time * 1000)  # Convert to ms

        # Extract joint positions as action
        raw_action = {}
        for key, value in raw_observation.items():
            if (
                key.endswith(".pos")
                and not key.startswith("head")
                and not key.startswith("left_wrist")
                and not key.startswith("right_wrist")
            ):
                raw_action[key] = value

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(raw_observation)

            log_rerun_data(
                observation=obs_transition,
                action=raw_action,
            )

            # Only show motor data if NOT in debug_timing mode (to avoid conflicts)
            if not debug_timing:
                if is_bimanual:
                    # Separate left and right arm data for two-column display
                    left_motors = {
                        k: v for k, v in raw_action.items() if k.startswith("left_")
                    }
                    right_motors = {
                        k: v for k, v in raw_action.items() if k.startswith("right_")
                    }

                    # Calculate column width
                    col_width = 25

                    # Print header
                    print("\n" + "-" * (col_width * 2 + 3))
                    print(f"{'LEFT ARM':<{col_width}} | {'RIGHT ARM':<{col_width}}")
                    print("-" * (col_width * 2 + 3))

                    # Display motors side by side
                    max_motors = max(len(left_motors), len(right_motors))
                    left_items = list(left_motors.items())
                    right_items = list(right_motors.items())

                    for i in range(max_motors):
                        left_str = ""
                        right_str = ""

                        if i < len(left_items):
                            motor_name = left_items[i][0].replace("left_", "")
                            left_str = f"{motor_name}: {left_items[i][1]:>7.3f}"

                        if i < len(right_items):
                            motor_name = right_items[i][0].replace("right_", "")
                            right_str = f"{motor_name}: {right_items[i][1]:>7.3f}"

                        print(f"{left_str:<{col_width}} | {right_str:<{col_width}}")

                    # Move cursor up: 1 blank line + 1 top line + 1 header + 1 separator + max_motors data lines
                    move_cursor_up(max_motors + 4)
                else:  # single arm
                    # Single column display for single arm
                    col_width = 20

                    # Print header
                    print("\n" + "-" * (col_width + 12))
                    print(f"{'JOINT':<{col_width}} | {'VALUE':>7}")
                    print("-" * (col_width + 12))

                    # Display motors
                    motor_items = list(raw_action.items())
                    for motor, value in motor_items:
                        print(f"{motor:<{col_width}} | {value:>7.3f}")

                    # Move cursor up
                    move_cursor_up(len(motor_items) + 4)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)
        # print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        # if duration is not None and time.perf_counter() - start >= duration:
        #     return
        if debug_timing:
            # Display timing info with cursor movement for smooth refresh
            print()
            print("🔍 TELEOP TIMING DEBUG")
            print("=" * 50)
            print(f"🤖 Robot state:     {robot_obs_time * 1000:.1f}ms")
            print(f"📷 Total cameras:   {total_camera_time_ms:.1f}ms")
            print()

            # Display individual camera timings with stability indicators
            num_cameras = len(camera_times)
            for cam_key, cam_time_ms in camera_times.items():
                if cam_time_ms > 10:  # Slow camera warning
                    print(f"🐌 {cam_key:12}: {cam_time_ms:5.1f}ms ⚠️")
                elif cam_time_ms > 5:  # Medium speed
                    print(f"⚡ {cam_key:12}: {cam_time_ms:5.1f}ms")
                else:  # Fast camera
                    print(f"✅ {cam_key:12}: {cam_time_ms:5.1f}ms")

            print()
            print(f"📊 Total observation: {total_obs_time * 1000:.1f}ms")
            print(f"⏱️  Loop time:        {loop_s * 1000:.1f}ms")
            print(f"🎯 Target period:     {1000/fps:.1f}ms")
            print(f"📈 Loop efficiency:   {(1000/fps)/(loop_s * 1000)*100:.1f}%")

            # Camera stability warning
            extra_warning_lines = 0
            if total_camera_time_ms > 20:
                print()
                print(f"⚠️  SLOW CAMERAS DETECTED! Total: {total_camera_time_ms:.1f}ms")
                extra_warning_lines = 2

            print("=" * 50)

            # Move cursor up to refresh in place
            # Count: 1 blank + 1 title + 1 sep + 2 info + 1 blank + cameras + 1 blank + 4 summary + warning + 1 sep
            total_lines = (
                1 + 1 + 1 + 2 + 1 + num_cameras + 1 + 4 + extra_warning_lines + 1
            )
            move_cursor_up(total_lines)
        else:
            # Simplified output - only show warnings
            if total_camera_time_ms > 20:
                print(f"⚠️  SLOW CAMERAS: {total_camera_time_ms:.1f}ms")
                for cam_key, cam_time_ms in camera_times.items():
                    if cam_time_ms > 10:
                        print(f"  🐌 {cam_key}: {cam_time_ms:.1f}ms")

        if duration is not None and time.perf_counter() - start >= duration:
            # Print final statistics before exiting
            if len(timing_stats["robot_obs_times"]) > 10:
                print("\n=== FINAL TIMING REPORT ===")
                all_robot = timing_stats["robot_obs_times"]
                all_total = timing_stats["total_obs_times"]
                all_loops = timing_stats["loop_times"]

                print(f"Total samples: {len(all_robot)}")
                print(f"Robot obs - avg: {sum(all_robot)/len(all_robot):.2f}ms")
                print(f"Total obs - avg: {sum(all_total)/len(all_total):.2f}ms")
                print(f"Loop time - avg: {sum(all_loops)/len(all_loops):.2f}ms")

                # Final camera analysis
                for cam_key, cam_times in timing_stats["camera_obs_times"].items():
                    if cam_times:
                        avg_cam_time = sum(cam_times) / len(cam_times)
                        print(f"{cam_key} - avg: {avg_cam_time:.2f}ms")
            return




def spacemouse_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
):
    """
    Teleop loop for Spacemouse.
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    
    # Check if this is Flexiv Rizon4 robot in CARTESIAN_MOTION_FORCE mode (needs special conversion)
    from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode
    is_flexiv = (
        robot.name == "flexiv_rizon4"
        and hasattr(robot.config, "control_mode")
        and robot.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE
        and teleop.name == "spacemouse"
    )

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()

        # Check for reset event (both buttons pressed simultaneously - immediate reset like original code)
        if teleop.name == "spacemouse":
            # Get button states directly from spacemouse (matching original code logic)
            button_left = teleop._spacemouse.is_button_pressed(0)
            button_right = teleop._spacemouse.is_button_pressed(1)

            if button_left and button_right:
                # Both buttons pressed: Reset to initial position (immediate, no 1 second wait)
                # For Flexiv robots, use robot's reset method which calls go_to_home or go_to_start
                # based on config.go_to_start
                if is_flexiv and hasattr(robot, 'reset_to_initial_position'):
                    try:
                        # First, reset robot to initial position
                        robot.reset_to_initial_position()
                        # Then, get robot's current actual position and update teleop target pose
                        # This ensures teleop target matches robot's actual position after reset
                        # so teleoperation can continue smoothly from the reset position
                        current_pose_euler = robot.get_current_tcp_pose_euler()
                        teleop.reset_to_pose(current_pose_euler[:6], current_pose_euler[6])
                        # Also update saved start pose for future resets
                        teleop._start_pose_6d = current_pose_euler[:6].copy()
                        teleop._start_gripper_pos = current_pose_euler[6]
                        logger.info("Reset to initial position triggered by both buttons")
                    except Exception as e:
                        logger.error(f"Failed to reset robot position: {e}\n{traceback.format_exc()}")
                else:
                    # For other robots or fallback: use saved initial pose from teleop.connect()
                    if hasattr(teleop, '_start_pose_6d') and hasattr(teleop, '_start_gripper_pos'):
                        teleop.reset_to_pose(teleop._start_pose_6d, teleop._start_gripper_pos)
                        logger.info("Reset to initial position triggered by both buttons")
                # Continue to next iteration (skip sending action this cycle)
                continue

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Convert spacemouse action to Flexiv format if needed
        if is_flexiv:
            # Use teleoperator's conversion method to convert Euler angles to quaternion
            robot_action_to_send = teleop.convert_to_flexiv_action(teleop_action)
        else:
            # Process action for robot through pipeline (for other robots)
            robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.3f}")
            move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        
        # Print time and actions (key-value pairs)
        action_str = ", ".join([f"{k}={v:.4f}" for k, v in robot_action_to_send.items()])
        if dryrun:
            print(f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | [DRYRUN] Actions: {action_str}", end="", flush=True)
        else:
            print(f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | Actions: {action_str}", end="", flush=True)

        if duration is not None and time.perf_counter() - start >= duration:
            return


def pico4_teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    display_data: bool = False,
    duration: float | None = None,
    dryrun: bool = False,
):
    """
    Teleop loop for Pico4 VR controller with Flexiv Rizon4 robot.

    Pico4 outputs actions directly in Flexiv format:
    - tcp.x, tcp.y, tcp.z: absolute TCP position (meters)
    - tcp.qw, tcp.qx, tcp.qy, tcp.qz: absolute TCP orientation (quaternion)
    - gripper.pos: absolute gripper position (meters)

    Control scheme:
    - Grip: Enable control (must be held to move robot)
    - Trigger: Controls gripper position (0=closed, 1=open)
    - A button: Reset to initial position
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation (for visualization)
        obs = robot.get_observation()

        # Get teleop action first (this also caches A button state for get_reset_button)
        raw_action = teleop.get_action()

        # Check for reset button (uses cached A button state from get_action)
        reset_button = teleop.get_reset_button()
        if reset_button:
            try:
                if not dryrun:
                    # Reset robot to initial position
                    if hasattr(robot, 'reset_to_initial_position'):
                        robot.reset_to_initial_position()
                    logger.info("Reset to initial position (A button pressed)")
                else:
                    logger.info("[DRYRUN] Reset to initial position (A button pressed) - robot movement skipped")
                
                # Always reset teleop state (both dryrun and normal mode)
                current_pose_quat = robot.get_current_tcp_pose_quat()
                teleop.reset_to_pose(current_pose_quat[:7], current_pose_quat[7])
            except Exception as e:
                logger.error(f"Failed to reset robot position: {e}\n{traceback.format_exc()}")
            # Skip this loop iteration (don't send action after reset)
            continue

        # Process teleop action through pipeline (usually identity)
        teleop_action = teleop_action_processor((raw_action, obs))

        # For Pico4 + Flexiv, action is already in correct format
        # No conversion needed (unlike Spacemouse which needs Euler->Quaternion)
        robot_action_to_send = teleop_action

        # Send action to robot
        if not dryrun:
            _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.4f}")
            move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        # Print status line with enable state and grip value for debugging
        enable_str = "ENABLED" if teleop._enabled else "DISABLED"
        ori_str = "ORI:ON" if teleop._orientation_control_active else "ORI:OFF"
        grip_str = f"grip={teleop._last_grip:.2f}"
        action_str = ", ".join([f"{k}={v:.4f}" for k, v in robot_action_to_send.items()])
        if dryrun:
            print(f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | [DRYRUN] | {enable_str} | {grip_str} | {ori_str} | {action_str}", end="", flush=True)
        else:
            print(f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {enable_str} | {grip_str} | {ori_str} | {action_str}", end="", flush=True)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    logger.info(pformat(asdict(cfg)))
    if cfg.dryrun:
        logger.warn("⚠️  DRYRUN MODE ENABLED - Actions will be printed but NOT sent to robot")
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    # Check if this is ARX5 robot (single arm or bimanual)
    if cfg.robot.type in ("bi_arx5", "arx5_follower"):
        mode = "bimanual" if cfg.robot.type == "bi_arx5" else "single-arm"
        logger.info(f"Detected ARX5 robot ({mode}), using specialized teleop loop")

        # Create robot instance
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        logger.info(f"Start EEF pose: {robot.get_start_eef_pose()}")
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )
        if cfg.teleop.type == "spacemouse":
            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect(start_eef_pose=robot.get_start_eef_pose())
            logger.info("Connected to Spacemouse")
            try:
                spacemouse_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                )
            except KeyboardInterrupt:
                pass
            finally:
                if cfg.display_data:
                    rr.rerun_shutdown()
                robot.disconnect()
                teleop.disconnect()
        else:
            try:
                arx5_teleop_loop(
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                pass
            finally:
                if cfg.display_data:
                    rr.rerun_shutdown()
                robot.disconnect()
    # Check if this is Flexiv Rizon4 robot with pico4
    elif cfg.robot.type == "flexiv_rizon4" and cfg.teleop.type == "pico4":
        logger.info("Detected Flexiv Rizon4 robot with Pico4, using specialized teleop loop")

        robot = None
        teleop = None

        try:
            # Create robot instance
            robot = make_robot_from_config(cfg.robot)

            # Ensure robot is in CARTESIAN_MOTION_FORCE mode for pico4 teleop
            from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode
            if robot.config.control_mode != ControlMode.CARTESIAN_MOTION_FORCE:
                raise ValueError(
                    f"Pico4 teleoperation requires CARTESIAN_MOTION_FORCE mode, "
                    f"but robot is configured with {robot.config.control_mode}"
                )

            # Connect to robot with error handling
            try:
                robot.connect(go_to_start=False)
                logger.info(f"Start EEF pose: {robot.get_current_tcp_pose_quat()}")
            except Exception as e:
                logger.error(f"Failed to connect to robot: {e}\n{traceback.format_exc()}")
                raise

            teleop_action_processor, robot_action_processor, robot_observation_processor = (
                make_default_processors()
            )

            # Connect to teleoperator with error handling
            try:
                teleop = make_teleoperator_from_config(cfg.teleop)
                teleop.connect(current_tcp_pose_quat=robot.get_current_tcp_pose_quat())
                logger.info("Connected to Pico4")
            except Exception as e:
                logger.error(f"Failed to connect to Pico4: {e}\n{traceback.format_exc()}")
                raise

            # Run teleoperation loop
            try:
                pico4_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")
            except Exception as e:
                logger.error(f"Error during teleoperation loop: {e}\n{traceback.format_exc()}")
                raise

        except Exception as e:
            logger.error(f"Error in teleoperation setup or execution: {e}\n{traceback.format_exc()}")
            logger.error(f"Teleoperation failed\n{traceback.format_exc()}")
        finally:
            # Safe disconnect - ensure both robot and teleop are disconnected
            if cfg.display_data:
                try:
                    rr.rerun_shutdown()
                except Exception as e:
                    logger.warn(f"Error shutting down rerun: {e}")

            if teleop is not None:
                try:
                    if teleop.is_connected:
                        teleop.disconnect()
                        logger.info("Pico4 disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting Pico4: {e}\n{traceback.format_exc()}")

            if robot is not None:
                try:
                    if robot.is_connected:
                        robot.disconnect()
                        logger.info("Robot safely disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting robot: {e}\n{traceback.format_exc()}")
                    # Force cleanup even if disconnect fails
                    try:
                        if hasattr(robot, '_robot') and robot._robot is not None:
                            robot._robot.Stop()
                    except Exception:
                        pass
    # Check if this is Flexiv Rizon4 robot with spacemouse
    elif cfg.robot.type == "flexiv_rizon4" and cfg.teleop.type == "spacemouse":
        logger.info("Detected Flexiv Rizon4 robot with Spacemouse, using specialized teleop loop")

        robot = None
        teleop = None

        try:
            # Create robot instance
            robot = make_robot_from_config(cfg.robot)

            # Ensure robot is in CARTESIAN_MOTION_FORCE mode for spacemouse teleop
            from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode
            if robot.config.control_mode != ControlMode.CARTESIAN_MOTION_FORCE:
                raise ValueError(
                    f"Spacemouse teleoperation requires CARTESIAN_MOTION_FORCE mode, "
                    f"but robot is configured with {robot.config.control_mode}"
                )
            
            # Connect to robot with error handling
            try:
                robot.connect(go_to_start=True)
                logger.info(f"Start EEF pose: {robot.get_current_tcp_pose_euler()}")
                logger.info(f"Start TCP pose: {robot.get_current_tcp_pose_quat()}")
            except Exception as e:
                logger.error(f"Failed to connect to robot: {e}\n{traceback.format_exc()}")
                raise
            
            teleop_action_processor, robot_action_processor, robot_observation_processor = (
                make_default_processors()
            )
            
            # Connect to teleoperator with error handling
            try:
                teleop = make_teleoperator_from_config(cfg.teleop)
                teleop.connect(current_tcp_pose_euler=robot.get_current_tcp_pose_euler())
                logger.info("Connected to Spacemouse")
            except Exception as e:
                logger.error(f"Failed to connect to Spacemouse: {e}\n{traceback.format_exc()}")
                raise
            
            # Run teleoperation loop
            try:
                spacemouse_teleop_loop(
                    teleop=teleop,
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dryrun=cfg.dryrun,
                )
            except KeyboardInterrupt:
                logger.info("Teleoperation interrupted by user")
            except Exception as e:
                logger.error(f"Error during teleoperation loop: {e}\n{traceback.format_exc()}")
                raise
                
        except Exception as e:
            logger.error(f"Error in teleoperation setup or execution: {e}\n{traceback.format_exc()}")
            logger.error(f"Teleoperation failed\n{traceback.format_exc()}")
        finally:
            # Safe disconnect - ensure both robot and teleop are disconnected
            if cfg.display_data:
                try:
                    rr.rerun_shutdown()
                except Exception as e:
                    logger.warn(f"Error shutting down rerun: {e}")
            
            if teleop is not None:
                try:
                    if teleop.is_connected:
                        teleop.disconnect()
                        logger.info("Spacemouse disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting Spacemouse: {e}\n{traceback.format_exc()}")
            
            if robot is not None:
                try:
                    if robot.is_connected:
                        robot.disconnect()
                        logger.info("Robot safely disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting robot: {e}\n{traceback.format_exc()}")
                    # Force cleanup even if disconnect fails
                    try:
                        if hasattr(robot, '_robot') and robot._robot is not None:
                            robot._robot.Stop()
                    except Exception:
                        pass
    else:
        teleop = make_teleoperator_from_config(cfg.teleop)
        robot = make_robot_from_config(cfg.robot)
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        teleop.connect()
        robot.connect()

        try:
            teleop_loop(
                teleop=teleop,
                robot=robot,
                fps=cfg.fps,
                display_data=cfg.display_data,
                duration=cfg.teleop_time_s,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )
        except KeyboardInterrupt:
            pass
        finally:
            if cfg.display_data:
                rr.rerun_shutdown()
            teleop.disconnect()
            robot.disconnect()


def main():
    # Mock teleop is now available as a regular teleoperator
    register_third_party_devices()
    teleoperate()


if __name__ == "__main__":
    main()
