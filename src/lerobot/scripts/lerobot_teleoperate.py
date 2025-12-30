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
    xense_flare,  # noqa: F401
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
    vive_tracker,
)

from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait, get_logger
from lerobot.utils.utils import move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Create global logger for teleoperate script
logger = get_logger("Teleoperate")


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
    debug_timing: bool = False,
):
    """
    Teleop loop for Spacemouse.
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    timing_stats = {"obs_times": [], "loop_times": []}
    
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

        # Get robot observation with timing
        obs_start = time.perf_counter()
        obs = robot.get_observation()
        obs_time = time.perf_counter() - obs_start
        timing_stats["obs_times"].append(obs_time * 1000)

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
            
            # Log raw observation directly (including images from XenseFlare)
            log_rerun_data(
                observation=obs,  # Use raw obs to ensure images are included
                action=teleop_action,
            )

            # Only show terminal display if not in debug_timing mode
            if not debug_timing:
                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                # Display the final robot action that was sent
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.3f}")
                move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)

        if debug_timing:
            # Display detailed timing info (clean single-line output)
            print(f"\r\033[K🔍 obs: {obs_time*1000:5.1f}ms | loop: {loop_s*1000:5.1f}ms | target: {1000/fps:.1f}ms | eff: {(1/fps)/loop_s*100:5.1f}%", end="", flush=True)
        elif not display_data:
            # Print time and actions (key-value pairs) only when not display_data
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


def vive_tracker_teleop_loop(
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
    Teleop loop for Vive Tracker with Flexiv Rizon4 robot.

    Vive Tracker outputs actions directly in Flexiv format:
    - tcp.x, tcp.y, tcp.z: absolute TCP position (meters)
    - tcp.qw, tcp.qx, tcp.qy, tcp.qz: absolute TCP orientation (quaternion)

    Control scheme:
    - Vive Tracker provides absolute 6-DoF pose tracking
    - No enable/disable control (always active after connect)
    - Coordinate transformation: action = ee_init @ inv(vive_init @ vive2ee) @ (vive_current @ vive2ee)
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation (for visualization)
        obs = robot.get_observation()

        # Get teleop action from Vive Tracker
        try:
            raw_action = teleop.get_action()
        except Exception as e:
            logger.error(f"Error getting Vive Tracker action: {e}")
            # On error, skip this iteration
            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / fps - dt_s)
            continue

        # Process teleop action through pipeline (usually identity)
        teleop_action = teleop_action_processor((raw_action, obs))

        # For Vive Tracker + Flexiv, action is already in correct format
        # No conversion needed (same as Pico4)
        robot_action_to_send = teleop_action

        # Send action to robot
        if not dryrun:
            try:
                _ = robot.send_action(robot_action_to_send)
            except Exception as e:
                logger.error(f"Error sending action to robot: {e}")

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

        # Print status line
        action_str = ", ".join([f"{k}={v:.4f}" for k, v in robot_action_to_send.items()])
        if dryrun:
            print(f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | [DRYRUN] | {action_str}", end="", flush=True)
        else:
            print(f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {action_str}", end="", flush=True)

        if duration is not None and time.perf_counter() - start >= duration:
            return


def xense_flare_teleop_loop(
    robot: Robot,
    fps: int,
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    display_data: bool = False,
    duration: float | None = None,
    debug_timing: bool = False,
):
    """
    Data collection loop for Xense Flare gripper.

    Xense Flare is a pure observation device (similar to teach mode).
    This loop continuously reads multi-modal sensor data:
    - Vive Tracker: 6DoF pose (tcp.x/y/z/qw/qx/qy/qz)
    - Wrist Camera: RGB image
    - Tactile Sensors: Tactile images
    - Gripper: Position

    No actions are sent to the robot - it is manually operated.
    """
    import numpy as np
    import warnings
    
    # Suppress Rerun's numpy compatibility warnings (doesn't affect functionality)
    warnings.filterwarnings("ignore", message=".*RotationQuatBatch.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="rerun")
    
    start = time.perf_counter()
    timing_stats = {
        "vive_times": [],
        "gripper_times": [],
        "camera_times": [],
        "sensor_times": [],
        "total_obs_times": [],
        "loop_times": [],
    }
    
    # Trajectory visualization settings
    trajectory_points: list[np.ndarray] = []
    max_trajectory_points = 500
    trajectory_line_radius = 0.005

    while True:
        loop_start = time.perf_counter()

        # Time the complete observation acquisition
        obs_start = time.perf_counter()

        try:
            # Get all observations from the robot
            obs = robot.get_observation()
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / fps - dt_s)
            continue

        total_obs_time = time.perf_counter() - obs_start
        timing_stats["total_obs_times"].append(total_obs_time * 1000)

        # Extract action from observation (for logging purposes)
        # Xense Flare's "action" is just the gripper position
        raw_action = {}
        if "gripper.pos" in obs:
            raw_action["gripper.pos"] = obs["gripper.pos"]

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=raw_action,
            )
            
            # Log tracker pose and trajectory visualization
            if "tcp.x" in obs and "tcp.y" in obs and "tcp.z" in obs:
                pos = np.array([obs["tcp.x"], obs["tcp.y"], obs["tcp.z"]])
                rot_xyzw = [
                    obs.get("tcp.qx", 0.0),
                    obs.get("tcp.qy", 0.0),
                    obs.get("tcp.qz", 0.0),
                    obs.get("tcp.qw", 1.0),
                ]
                
                # Log 3D transform
                rr.log(
                    "tracker/pose",
                    rr.Transform3D(
                        translation=pos.tolist(),
                        rotation=rr.Quaternion(xyzw=rot_xyzw),
                    ),
                )
                
                # Log current position as a red point
                rr.log("tracker/point", rr.Points3D([pos], radii=[0.015], colors=[[255, 50, 50]]))
                
                # Log coordinate axes (XYZ arrows)
                qx, qy, qz, qw = rot_xyzw[0], rot_xyzw[1], rot_xyzw[2], rot_xyzw[3]
                # Quaternion to rotation matrix
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                ])
                axis_length = 0.08
                x_axis = R @ np.array([axis_length, 0, 0])
                y_axis = R @ np.array([0, axis_length, 0])
                z_axis = R @ np.array([0, 0, axis_length])
                
                rr.log(
                    "tracker/axes",
                    rr.Arrows3D(
                        origins=np.array([pos, pos, pos]),
                        vectors=np.array([x_axis, y_axis, z_axis]),
                        colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),  # RGB for XYZ
                        radii=0.003,
                    ),
                )
                
                # Trajectory visualization
                trajectory_points.append(pos.copy())
                
                # Keep only the last max_trajectory_points
                if len(trajectory_points) > max_trajectory_points:
                    trajectory_points.pop(0)
                
                # Log trajectory as line strip
                if len(trajectory_points) >= 2:
                    points_array = np.array(trajectory_points)
                    rr.log("tracker/trajectory", rr.LineStrips3D([points_array], radii=[trajectory_line_radius]))
            
            # Log lighthouse poses (from vive_tracker directly)
            vive_tracker = robot.get_vive_tracker()
            if vive_tracker is not None:
                try:
                    all_poses = vive_tracker.get_pose()
                    if all_poses:
                        for device_name, pose_data in all_poses.items():
                            if device_name.startswith("LH") and pose_data is not None:
                                lh_pos = list(pose_data.position)
                                lh_rot = pose_data.rotation  # [qw, qx, qy, qz]
                                # Convert from [qw, qx, qy, qz] to [qx, qy, qz, qw]
                                lh_rot_xyzw = [lh_rot[1], lh_rot[2], lh_rot[3], lh_rot[0]]
                                
                                # Color: LH0=green, LH1=blue, others=orange
                                if device_name == "LH0":
                                    color = [0, 255, 100]
                                elif device_name == "LH1":
                                    color = [100, 180, 255]
                                else:
                                    color = [255, 200, 100]
                                
                                base_path = f"lighthouse/{device_name}"
                                
                                # Log 3D transform
                                rr.log(
                                    f"{base_path}/pose",
                                    rr.Transform3D(
                                        translation=lh_pos,
                                        rotation=rr.Quaternion(xyzw=lh_rot_xyzw),
                                    ),
                                )
                                
                                # Log as large point with label
                                rr.log(
                                    f"{base_path}/point",
                                    rr.Points3D([lh_pos], radii=[0.05], colors=[color], labels=[device_name]),
                                )
                                
                                # Log coordinate axes
                                qx, qy, qz, qw = lh_rot_xyzw[0], lh_rot_xyzw[1], lh_rot_xyzw[2], lh_rot_xyzw[3]
                                R = np.array([
                                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                                    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                                ])
                                lh_axis_length = 0.15
                                lh_x_axis = R @ np.array([lh_axis_length, 0, 0])
                                lh_y_axis = R @ np.array([0, lh_axis_length, 0])
                                lh_z_axis = R @ np.array([0, 0, lh_axis_length])
                                
                                rr.log(
                                    f"{base_path}/axes",
                                    rr.Arrows3D(
                                        origins=np.array([lh_pos, lh_pos, lh_pos]),
                                        vectors=np.array([lh_x_axis, lh_y_axis, lh_z_axis]),
                                        colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
                                        radii=0.005,
                                    ),
                                )
                except Exception:
                    pass  # Lighthouse visualization is optional

            if not debug_timing:
                # Display observation data
                col_width = 25

                # Print header
                print("\n" + "-" * (col_width + 15))
                print(f"{'OBSERVATION':<{col_width}} | {'VALUE':>12}")
                print("-" * (col_width + 15))

                # Display pose data
                pose_keys = ["tcp.x", "tcp.y", "tcp.z", "tcp.qw", "tcp.qx", "tcp.qy", "tcp.qz"]
                for key in pose_keys:
                    if key in obs:
                        print(f"{key:<{col_width}} | {obs[key]:>12.4f}")

                # Display gripper
                if "gripper.pos" in obs:
                    print(f"{'gripper.pos':<{col_width}} | {obs['gripper.pos']:>12.4f}")

                # Count images
                image_count = sum(1 for k, v in obs.items() if hasattr(v, 'shape') and len(v.shape) >= 2)
                print(f"{'[Images]':<{col_width}} | {image_count:>12}", flush=True)

                # Move cursor up
                move_cursor_up(len(pose_keys) + 4)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        timing_stats["loop_times"].append(loop_s * 1000)

        if debug_timing:
            # Display timing info
            print()
            print("🔍 XENSE FLARE TIMING DEBUG")
            print("=" * 50)
            print(f"📊 Total observation: {total_obs_time * 1000:.1f}ms")
            print(f"⏱️  Loop time:        {loop_s * 1000:.1f}ms")
            print(f"🎯 Target period:     {1000/fps:.1f}ms")
            print(f"📈 Loop efficiency:   {(1000/fps)/(loop_s * 1000)*100:.1f}%")
            print("=" * 50, flush=True)

            # Move cursor up to refresh in place
            move_cursor_up(8)
        else:
            # Simple status line
            pose_str = ""
            if "tcp.x" in obs and "tcp.y" in obs and "tcp.z" in obs:
                pose_str = f"pos=[{obs['tcp.x']:.3f}, {obs['tcp.y']:.3f}, {obs['tcp.z']:.3f}]"
            gripper_str = f"grip={obs.get('gripper.pos', 0.0):.2f}"
            print(f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | {pose_str} | {gripper_str}", end="", flush=True)

        if duration is not None and time.perf_counter() - start >= duration:
            # Print final statistics before exiting
            if len(timing_stats["total_obs_times"]) > 10:
                print("\n=== FINAL TIMING REPORT ===")
                all_total = timing_stats["total_obs_times"]
                all_loops = timing_stats["loop_times"]

                print(f"Total samples: {len(all_total)}")
                print(f"Total obs - avg: {sum(all_total)/len(all_total):.2f}ms")
                print(f"Loop time - avg: {sum(all_loops)/len(all_loops):.2f}ms")
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    logger.info(pformat(asdict(cfg)))
    if cfg.dryrun:
        logger.warn("⚠️  DRYRUN MODE ENABLED - Actions will be printed but NOT sent to robot")
    if cfg.display_data:
        # Use robot and teleop names in session name
        teleop_name = cfg.teleop.type if cfg.teleop else "none"
        session_name = f"teleop_{cfg.robot.type}_{teleop_name}"
        init_rerun(session_name=session_name)

    # Check if this is Xense Flare (data collection gripper - no teleoperator needed)
    if cfg.robot.type == "xense_flare":
        logger.info("Detected Xense Flare data collection gripper")

        robot = None

        try:
            # Create robot instance
            robot = make_robot_from_config(cfg.robot)

            # Connect to robot
            try:
                robot.connect()
                logger.info("✅ Xense Flare connected")
                logger.info(f"   MAC: {robot.config.mac_addr}")
                logger.info(f"   Sensors: {list(robot._sensors.keys())}")
                logger.info(f"   Camera: {'Yes' if robot._camera else 'No'}")
                logger.info(f"   Gripper: {'Yes' if robot._gripper else 'No'}")
                logger.info(f"   Vive Tracker: {'Yes' if robot._vive_tracker else 'No'}")
            except Exception as e:
                logger.error(f"Failed to connect to Xense Flare: {e}\n{traceback.format_exc()}")
                raise

            _, _, robot_observation_processor = make_default_processors()

            # Run data collection loop
            try:
                xense_flare_teleop_loop(
                    robot=robot,
                    fps=cfg.fps,
                    display_data=cfg.display_data,
                    duration=cfg.teleop_time_s,
                    robot_observation_processor=robot_observation_processor,
                    debug_timing=cfg.debug_timing,
                )
            except KeyboardInterrupt:
                logger.info("Data collection interrupted by user")
            except Exception as e:
                logger.error(f"Error during data collection: {e}\n{traceback.format_exc()}")
                raise

        except Exception as e:
            logger.error(f"Error in Xense Flare setup: {e}\n{traceback.format_exc()}")
        finally:
            # Safe disconnect
            if cfg.display_data:
                try:
                    rr.rerun_shutdown()
                except Exception as e:
                    logger.warn(f"Error shutting down rerun: {e}")

            if robot is not None:
                try:
                    if robot.is_connected:
                        robot.disconnect()
                        logger.info("✅ Xense Flare disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting Xense Flare: {e}\n{traceback.format_exc()}")

    # Check if this is ARX5 robot (single arm or bimanual)
    elif cfg.robot.type in ("bi_arx5", "arx5_follower"):
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
                    debug_timing=cfg.debug_timing,
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
                    debug_timing=cfg.debug_timing,
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
    # Check if this is Flexiv Rizon4 robot with vive_tracker
    elif cfg.robot.type == "flexiv_rizon4" and cfg.teleop.type == "vive_tracker":
        logger.info("Detected Flexiv Rizon4 robot with Vive Tracker, using specialized teleop loop")

        robot = None
        teleop = None

        try:
            # Create robot instance
            robot = make_robot_from_config(cfg.robot)

            # Ensure robot is in CARTESIAN_MOTION_FORCE mode for vive_tracker teleop
            from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode
            if robot.config.control_mode != ControlMode.CARTESIAN_MOTION_FORCE:
                raise ValueError(
                    f"Vive Tracker teleoperation requires CARTESIAN_MOTION_FORCE mode, "
                    f"but robot is configured with {robot.config.control_mode}"
                )

            # Connect to robot with error handling
            try:
                robot.connect(go_to_start=False)
                logger.info(f"Start TCP pose (quat): {robot.get_current_tcp_pose_quat()}")
            except Exception as e:
                logger.error(f"Failed to connect to robot: {e}\n{traceback.format_exc()}")
                raise

            teleop_action_processor, robot_action_processor, robot_observation_processor = (
                make_default_processors()
            )

            # Connect to teleoperator with error handling
            try:
                teleop = make_teleoperator_from_config(cfg.teleop)
                # Vive Tracker requires current TCP pose for coordinate transformation
                # get_current_tcp_pose_quat() returns 8D [x,y,z,qw,qx,qy,qz,gripper], take first 7
                current_tcp_pose = robot.get_current_tcp_pose_quat()[:7]
                teleop.connect(current_tcp_pose_quat=current_tcp_pose)
                logger.info("Connected to Vive Tracker")
            except Exception as e:
                logger.error(f"Failed to connect to Vive Tracker: {e}\n{traceback.format_exc()}")
                raise

            # Run teleoperation loop
            try:
                vive_tracker_teleop_loop(
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
                        logger.info("Vive Tracker disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting Vive Tracker: {e}\n{traceback.format_exc()}")

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
