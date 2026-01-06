#!/usr/bin/env python

# Copyright 2025 The Xense Robotics Inc. team. All rights reserved.
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
Xense Flare Example Script

Demonstrates Xense Flare data collection gripper functionality with Rerun visualization.
All sensor data (Vive tracker, gripper, camera, tactile sensors) is 
visualized in the Rerun viewer.

Usage:
    python -m lerobot.robots.xense_flare.xense_flare_example --mac_addr 6ebbc5f53240
    python -m lerobot.robots.xense_flare.xense_flare_example --mac_addr 6ebbc5f53240 --no-rerun
    python -m lerobot.robots.xense_flare.xense_flare_example --mac_addr 6ebbc5f53240 --fps 60
"""

import os
import sys
import time
import argparse
import warnings
import logging
from collections import defaultdict
import numpy as np

from lerobot.utils.robot_utils import get_logger
from lerobot.robots import make_robot_from_config

# Check Rerun availability
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False

# Create logger
logger = get_logger("XenseFlareExample")

# Entity path prefixes for Rerun
CAMERA_PREFIX = "camera"
GRIPPER_PREFIX = "gripper"
TRACKER_PREFIX = "vive_tracker"
SENSOR_PREFIX = "sensors"


# =============================================================================
# Rerun Visualization Helper Functions
# =============================================================================

def init_rerun(
    session_name: str = "xense_flare_example",
    spawn: bool = True,
    memory_limit: str | None = None,
) -> bool:
    """
    Initialize the Rerun SDK for visualizing Xense Flare data.

    Args:
        session_name: Name of the Rerun session/recording.
        spawn: Whether to spawn the Rerun viewer automatically.
        memory_limit: Memory limit for Rerun (e.g., "10%", "2GB").

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    if not RERUN_AVAILABLE:
        logger.warn("Rerun is not installed. Install it with: pip install rerun-sdk")
        return False

    try:
        # Configure flush settings for better streaming performance
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        # Suppress warnings and logging to avoid clutter
        warnings.filterwarnings("ignore")
        logging.getLogger("rerun").setLevel(logging.CRITICAL)
        
        # Save original showwarning before Rerun can override it
        original_showwarning = warnings.showwarning
        
        # Initialize Rerun
        rr.init(session_name)
        
        # Restore original showwarning
        warnings.showwarning = original_showwarning

        # Spawn viewer
        if spawn:
            if memory_limit is None:
                memory_limit = os.getenv("XENSE_RERUN_MEMORY_LIMIT", "10%")
            rr.spawn(memory_limit=memory_limit)

        logger.info(f"Rerun initialized with session: {session_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Rerun: {e}")
        return False


def log_camera_image(
    image: np.ndarray,
    camera_name: str = "wrist",
    entity_path: str | None = None,
    color_format: str = "BGR",
) -> None:
    """
    Log a camera image to Rerun.

    Args:
        image: Image array in HWC format (Height, Width, Channels).
        camera_name: Name of the camera (e.g., "wrist", "side").
        entity_path: Custom entity path. If None, uses "{CAMERA_PREFIX}/{camera_name}".
        color_format: Color format of input image. "RGB" or "BGR" (default).
            If "BGR", will be converted to RGB before logging.
    """
    if not RERUN_AVAILABLE or image is None:
        return

    path = entity_path or f"{CAMERA_PREFIX}/{camera_name}"

    # Handle CHW -> HWC conversion if needed
    if (
        image.ndim == 3
        and image.shape[0] in (1, 3, 4)
        and image.shape[-1] not in (1, 3, 4)
    ):
        image = np.transpose(image, (1, 2, 0))

    # Convert BGR to RGB if needed
    if color_format.upper() == "BGR" and image.ndim == 3 and image.shape[2] == 3:
        image = image[:, :, ::-1]

    rr.log(path, rr.Image(image))


def log_gripper_state(
    position: float | None = None,
    entity_path: str = GRIPPER_PREFIX,
) -> None:
    """
    Log gripper state to Rerun.

    Args:
        position: Gripper position (0-85 range, 85 is fully open).
        entity_path: Base entity path for gripper data.
    """
    if not RERUN_AVAILABLE:
        return

    if position is not None:
        rr.log(f"{entity_path}/position", rr.Scalar(float(position)))


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])


def log_coordinate_axes(
    entity_path: str,
    position: np.ndarray,
    rotation_xyzw: np.ndarray,
    axis_length: float = 0.1,
) -> None:
    """
    Log coordinate axes (XYZ) at a given pose.
    
    Args:
        entity_path: Base entity path
        position: Position [x, y, z]
        rotation_xyzw: Quaternion [qx, qy, qz, qw]
        axis_length: Length of each axis arrow
    """
    if not RERUN_AVAILABLE:
        return
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(rotation_xyzw)
    
    # Axis directions in world frame
    x_axis = R @ np.array([axis_length, 0, 0])
    y_axis = R @ np.array([0, axis_length, 0])
    z_axis = R @ np.array([0, 0, axis_length])
    
    # Log arrows for each axis
    origins = np.array([position, position, position])
    vectors = np.array([x_axis, y_axis, z_axis])
    colors = np.array([
        [255, 0, 0],    # X - Red
        [0, 255, 0],    # Y - Green
        [0, 0, 255],    # Z - Blue
    ])
    
    rr.log(
        f"{entity_path}/axes",
        rr.Arrows3D(
            origins=origins,
            vectors=vectors,
            colors=colors,
            radii=0.003,
        ),
    )


def log_vive_pose(
    device_name: str,
    position: list[float] | np.ndarray,
    rotation_xyzw: list[float] | np.ndarray,
    entity_path: str | None = None,
) -> None:
    """
    Log Vive Tracker pose to Rerun.

    Logs:
    - 3D transform (position + rotation as quaternion)
    - Red point at tracker position
    - Coordinate axes (XYZ arrows)
    - Individual position/rotation components as scalars

    Args:
        device_name: Name of the Vive device (e.g., "tracker").
        position: Position [x, y, z] in meters.
        rotation_xyzw: Quaternion [qx, qy, qz, qw] (Rerun format).
        entity_path: Custom base entity path. If None, uses "{TRACKER_PREFIX}/{device_name}".
    """
    if not RERUN_AVAILABLE:
        return

    if position is None or rotation_xyzw is None:
        return

    base_path = entity_path or f"{TRACKER_PREFIX}/{device_name}"
    pos = np.array(position)
    rot = np.array(rotation_xyzw)  # [qx, qy, qz, qw]

    # Log 3D transform
    rr.log(
        f"{base_path}/pose",
        rr.Transform3D(
            translation=pos,
            rotation=rr.Quaternion(xyzw=rot),
        ),
    )

    # Log position as RED 3D point for tracker
    rr.log(
        f"{base_path}/point",
        rr.Points3D(
            [pos],
            radii=[0.015],
            colors=[[255, 50, 50]],  # Red
        ),
    )
    
    # Log coordinate axes
    log_coordinate_axes(base_path, pos, rot, axis_length=0.08)

    # Log individual position components as scalars for plotting
    rr.log(f"{base_path}/position/x", rr.Scalar(float(pos[0])))
    rr.log(f"{base_path}/position/y", rr.Scalar(float(pos[1])))
    rr.log(f"{base_path}/position/z", rr.Scalar(float(pos[2])))

    # Log quaternion components
    rr.log(f"{base_path}/rotation/qx", rr.Scalar(float(rot[0])))
    rr.log(f"{base_path}/rotation/qy", rr.Scalar(float(rot[1])))
    rr.log(f"{base_path}/rotation/qz", rr.Scalar(float(rot[2])))
    rr.log(f"{base_path}/rotation/qw", rr.Scalar(float(rot[3])))


def log_lighthouse(
    device_name: str,
    position: list[float] | np.ndarray,
    rotation_wxyz: list[float] | np.ndarray,
) -> None:
    """
    Log lighthouse as a static reference marker with label and coordinate axes.
    
    Args:
        device_name: Name of the lighthouse device (e.g., "LH0", "LH1")
        position: Position [x, y, z]
        rotation_wxyz: Quaternion [qw, qx, qy, qz] (wxyz format from pysurvive)
    """
    if not RERUN_AVAILABLE:
        return
    
    pos = np.array(position)
    rot = np.array(rotation_wxyz)  # [qw, qx, qy, qz]
    # Convert from [qw, qx, qy, qz] to [qx, qy, qz, qw] for Rerun
    rot_xyzw = np.array([rot[1], rot[2], rot[3], rot[0]])
    base_path = f"{TRACKER_PREFIX}/{device_name}"
    
    # Use different colors for LH0/LH1
    if device_name == "LH0":
        color = [0, 255, 100]  # Green
    elif device_name == "LH1":
        color = [100, 180, 255]  # Blue
    else:
        color = [255, 200, 100]  # Orange
    
    # Log 3D transform
    rr.log(
        f"{base_path}/pose",
        rr.Transform3D(
            translation=pos,
            rotation=rr.Quaternion(xyzw=rot_xyzw),
        ),
    )
    
    # Log as LARGE 3D point with label
    rr.log(
        f"{base_path}/point",
        rr.Points3D(
            [pos],
            radii=[0.05],  # Larger radius for lighthouses
            colors=[color],
            labels=[device_name],
        ),
    )
    
    # Log coordinate axes for lighthouse pose
    log_coordinate_axes(base_path, pos, rot_xyzw, axis_length=0.15)


class TrajectoryVisualizer:
    """
    Helper class to visualize trajectories over time.

    Maintains a history of positions for each device and renders
    them as line strips in 3D space.
    """

    def __init__(self, max_points: int = 500, line_radius: float = 0.005):
        """
        Initialize trajectory visualizer.

        Args:
            max_points: Maximum number of points to keep in trajectory history.
            line_radius: Radius of the trajectory line.
        """
        self.max_points = max_points
        self.line_radius = line_radius
        self.trajectories: dict[str, list[np.ndarray]] = defaultdict(list)

    def add_point(
        self,
        device_name: str,
        position: list[float] | np.ndarray,
        entity_path: str | None = None,
    ) -> None:
        """
        Add a point to the trajectory and log the updated trajectory.

        Args:
            device_name: Name of the device.
            position: Position [x, y, z] to add.
            entity_path: Custom entity path for the trajectory.
        """
        if not RERUN_AVAILABLE:
            return

        pos = np.array(position)
        self.trajectories[device_name].append(pos)

        # Keep only the last max_points
        if len(self.trajectories[device_name]) > self.max_points:
            self.trajectories[device_name] = self.trajectories[device_name][-self.max_points:]

        # Log trajectory as line strip
        path = entity_path or f"{TRACKER_PREFIX}/{device_name}/trajectory"
        points = np.array(self.trajectories[device_name])

        if len(points) >= 2:
            rr.log(path, rr.LineStrips3D([points], radii=[self.line_radius]))

    def clear(self, device_name: str | None = None) -> None:
        """Clear trajectory history."""
        if device_name:
            self.trajectories[device_name] = []
        else:
            self.trajectories.clear()


# Global trajectory visualizer instance
_trajectory_viz = None


def get_trajectory_visualizer() -> TrajectoryVisualizer:
    """Get or create the global trajectory visualizer."""
    global _trajectory_viz
    if _trajectory_viz is None:
        _trajectory_viz = TrajectoryVisualizer()
    return _trajectory_viz


# =============================================================================
# Terminal Display Functions
# =============================================================================

def move_cursor_up(lines: int):
    """Move cursor up N lines"""
    sys.stdout.write(f"\033[{lines}A")
    sys.stdout.flush()


def make_button_callback(event_type: str):
    """Create a callback function for a specific button event type"""
    def callback():
        logger.info(f"[Button Event] {event_type}")
    return callback


def format_live_data(robot, obs: dict, frame_count: int, fps: float, timing: dict = None) -> list[str]:
    """Format live data for terminal display, returns list of lines"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  Xense Flare Live Monitor | Frame: {frame_count:6d} | FPS: {fps:5.1f}")
    lines.append("=" * 70)
    
    # Timing Section (if available)
    if timing:
        lines.append("")
        lines.append("┌─ ⏱️  Latency (ms) ──────────────────────────────────────────────────")
        lines.append(f"│  Observation: {timing.get('obs', 0):.1f}  |  "
                    f"Rerun: {timing.get('rerun', 0):.1f}  |  Total: {timing.get('total', 0):.1f}")
        lines.append("└" + "─" * 69)
    
    # Vive Tracker Section
    lines.append("")
    lines.append("┌─ 🎯 Vive Tracker ─────────────────────────────────────────────────")
    if "tcp.x" in obs:
        pos = [obs.get("tcp.x", 0), obs.get("tcp.y", 0), obs.get("tcp.z", 0)]
        rot = [obs.get("tcp.qw", 1), obs.get("tcp.qx", 0), obs.get("tcp.qy", 0), obs.get("tcp.qz", 0)]
        lines.append(f"│  📡 Tracker Pos: [{pos[0]:+7.3f}, {pos[1]:+7.3f}, {pos[2]:+7.3f}]")
        lines.append(f"│            Rot: [{rot[0]:+6.3f}, {rot[1]:+6.3f}, {rot[2]:+6.3f}, {rot[3]:+6.3f}]")
    else:
        lines.append("│  (No Vive data)")
    lines.append("└" + "─" * 69)
    
    # Gripper Section
    lines.append("")
    lines.append("┌─ 🤖 Gripper ───────────────────────────────────────────────────────")
    if "gripper.pos" in obs:
        pos = obs['gripper.pos']
        # Create visual bar for position (0-85, where 85 is fully open)
        GRIPPER_MAX_POS = 85.0
        bar_width = 20
        filled = int(max(0, min(GRIPPER_MAX_POS, pos)) / GRIPPER_MAX_POS * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"│  Position: [{bar}] {pos:6.2f}")
    else:
        lines.append("│  (No gripper data)")
    lines.append("└" + "─" * 69)
    
    # Sensor Section
    lines.append("")
    lines.append("┌─ 🦖 Tactile Sensors ───────────────────────────────────────────────")
    sensor_keys = [k for k in obs.keys() if "tactile" in k.lower() or k.startswith("sensor_")]
    if sensor_keys:
        for key in sensor_keys:
            img = obs[key]
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                lines.append(f"│  {key}: {w}x{h} ✅")
            else:
                lines.append(f"│  {key}: {type(img)}")
    else:
        lines.append("│  (No sensor data)")
    lines.append("└" + "─" * 69)
    
    # Camera Section
    lines.append("")
    lines.append("┌─ 📷 Camera ────────────────────────────────────────────────────────")
    if "wrist_cam" in obs:
        img = obs["wrist_cam"]
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1
            lines.append(f"│  Image: {w}x{h} ({channels}ch) | dtype: {img.dtype}")
        else:
            lines.append(f"│  Image type: {type(img)}")
    else:
        lines.append("│  (No camera data)")
    lines.append("└" + "─" * 69)
    
    lines.append("")
    lines.append("Press Ctrl+C to quit | View all data in Rerun viewer")
    
    return lines


def log_to_rerun(obs: dict, frame_count: int, robot=None):
    """
    Log observation data to Rerun.
    
    Data sources (from robot.get_observation()):
    - Vive Tracker pose: via vive_tracker.get_action() -> tcp.x/y/z/qw/qx/qy/qz
    - Gripper state: via gripper.get_gripper_status() -> gripper.pos
    - Camera image: via camera.read() -> wrist_cam (BGR format)
    - Tactile sensors: via sensor.selectSensorInfo() -> sensor_<sn> (BGR format)
    
    Visualization:
    - Camera/Tactile images: BGR to RGB conversion
    - Vive Tracker: 3D pose, coordinate axes, trajectory line
    - Lighthouses: Static reference markers with coordinate axes
    - Gripper: Position as scalar plot
    
    Args:
        obs: Observation dictionary from robot.get_observation()
        frame_count: Current frame number
        robot: Optional robot instance to access vive_tracker for lighthouse poses
    """
    # Log Vive Tracker pose with trajectory and coordinate axes
    # Data from: vive_tracker.get_action() -> tcp.x/y/z/qw/qx/qy/qz
    if "tcp.x" in obs:
        pos = [obs.get("tcp.x", 0), obs.get("tcp.y", 0), obs.get("tcp.z", 0)]
        # xyzw format for Rerun
        rot_xyzw = [
            obs.get("tcp.qx", 0),
            obs.get("tcp.qy", 0),
            obs.get("tcp.qz", 0),
            obs.get("tcp.qw", 1),
        ]
        
        # Log pose with coordinate axes
        log_vive_pose(
            device_name="tracker",
            position=pos,
            rotation_xyzw=rot_xyzw,
        )
        
        # Add to trajectory
        trajectory_viz = get_trajectory_visualizer()
        trajectory_viz.add_point("tracker", pos)
    
    # Log lighthouse poses (static reference markers)
    if robot is not None:
        vive_tracker = robot.get_vive_tracker()
        if vive_tracker is not None:
            # Get all device poses (including lighthouses)
            all_poses = vive_tracker.get_pose()  # Returns dict of device -> PoseData
            if all_poses:
                for device_name, pose_data in all_poses.items():
                    # Only log lighthouses (device names starting with "LH")
                    if device_name.startswith("LH") and pose_data is not None:
                        # PoseData has position and rotation attributes
                        # rotation is [qw, qx, qy, qz] format
                        log_lighthouse(
                            device_name=device_name,
                            position=pose_data.position,
                            rotation_wxyz=pose_data.rotation,
                        )
    
    # Log gripper position
    if "gripper.pos" in obs:
        log_gripper_state(position=obs["gripper.pos"])
    
    # Log tactile sensor images (already RGB from get_observation)
    for key in obs.keys():
        if "tactile" in key.lower() or key.startswith("sensor_"):
            img = obs[key]
            if isinstance(img, np.ndarray) and img.ndim >= 2:
                log_camera_image(
                    image=img,
                    camera_name=key,
                    entity_path=f"{SENSOR_PREFIX}/{key}",
                    color_format="RGB",
                )
    
    # Log wrist camera (already RGB from get_observation)
    if "wrist_cam" in obs:
        img = obs["wrist_cam"]
        if isinstance(img, np.ndarray) and img.ndim >= 2:
            log_camera_image(
                image=img,
                camera_name="wrist",
                color_format="RGB",
            )


def run_live_monitor(robot, args, use_rerun: bool = False):
    """Run continuous live monitoring with optional Rerun visualization"""
    logger.info("=" * 70)
    if use_rerun:
        logger.info("Starting Live Monitor with Rerun Visualization")
        logger.info("  📊 All data is being logged to Rerun viewer")
    else:
        logger.info("Starting Live Monitor (Rerun visualization disabled)")
    logger.info("=" * 70)
    
    # Print separator and prepare display area
    print()  # Blank line to separate logger output from live display
    
    frame_count = 0
    fps = 0.0
    fps_update_interval = 10  # Update FPS every N frames
    last_fps_time = time.time()
    
    # Number of lines we print (for cursor movement)
    num_lines = 0
    first_print = True  # Track first print to avoid cursor movement issues
    
    # Timing accumulators
    time_obs = 0.0  # Total get_observation() time
    time_rerun = 0.0  # Rerun logging time
    
    # Last timing info for display
    last_timing_info = None
    
    target_interval = 1.0 / args.fps if args.fps > 0 else 0
    
    try:
        while True:
            loop_start = time.time()
            
            # Get observation
            t0 = time.perf_counter()
            obs = robot.get_observation()
            t1 = time.perf_counter()
            
            # Accumulate observation time
            time_obs += (t1 - t0)
            
            # Log to Rerun (only if visualization is enabled)
            if use_rerun:
                t2 = time.perf_counter()
                log_to_rerun(obs, frame_count, robot=robot)
                time_rerun += time.perf_counter() - t2
            
            frame_count += 1
            
            # Update FPS and timing stats
            if frame_count % fps_update_interval == 0:
                current_time = time.time()
                fps = fps_update_interval / (current_time - last_fps_time)
                last_fps_time = current_time
                
                # Calculate average timing (in ms)
                avg_obs = (time_obs / fps_update_interval) * 1000
                avg_rerun = (time_rerun / fps_update_interval) * 1000
                total = avg_obs + avg_rerun
                
                # Store timing info for display
                last_timing_info = {
                    "obs": avg_obs,
                    "rerun": avg_rerun,
                    "total": total,
                }
                
                # Reset accumulators
                time_obs = 0.0
                time_rerun = 0.0
            
            # Format and print data to terminal
            if not args.no_print and frame_count % args.print_interval == 0:
                lines = format_live_data(robot, obs, frame_count, fps, timing=last_timing_info)
                
                # Move cursor up to overwrite previous output (skip on first print)
                if not first_print and num_lines > 0:
                    move_cursor_up(num_lines)
                first_print = False
                
                # Print all lines (clear each line before printing)
                for line in lines:
                    sys.stdout.write(f"\033[2K{line}\n")  # \033[2K clears entire line
                sys.stdout.flush()
                
                num_lines = len(lines)
            
            # Control loop rate
            elapsed = time.time() - loop_start
            if target_interval > 0:
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Xense Flare Example - Live monitoring with Rerun visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m lerobot.robots.xense_flare.xense_flare_example --mac_addr 6ebbc5f53240
  python -m lerobot.robots.xense_flare.xense_flare_example --mac_addr 6ebbc5f53240 --no-sensor
  python -m lerobot.robots.xense_flare.xense_flare_example --mac_addr 6ebbc5f53240 --fps 30
        """
    )
    parser.add_argument(
        "--mac_addr",
        type=str,
        default="6ebbc5f53240",
        help="MAC address of the Xense Flare device",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target frame rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Disable gripper",
    )
    parser.add_argument(
        "--no-sensor",
        action="store_true",
        help="Disable tactile sensors",
    )
    parser.add_argument(
        "--no-cam",
        action="store_true",
        help="Disable camera",
    )
    parser.add_argument(
        "--sensor-output",
        type=str,
        choices=["rectify", "difference"],
        default="rectify",
        help="Sensor output type (default: rectify)",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=1,
        help="Print to terminal every N frames (default: 1)",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Disable terminal printing for maximum performance",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun visualization",
    )
    args = parser.parse_args()

    # Check Rerun availability
    use_rerun = not args.no_rerun
    if use_rerun and not RERUN_AVAILABLE:
        logger.warn("Rerun is not installed, disabling visualization")
        logger.warn("Install with: pip install rerun-sdk")
        use_rerun = False

    logger.info("=" * 70)
    if use_rerun:
        logger.info("Xense Flare Example - Rerun Visualization")
    else:
        logger.info("Xense Flare Example - Data Collection Mode")
    logger.info("=" * 70)
    logger.info(f"MAC Address: {args.mac_addr}")
    logger.info(f"Components: gripper={'OFF' if args.no_gripper else 'ON'}, "
                f"sensor={'OFF' if args.no_sensor else 'ON'}, "
                f"cam={'OFF' if args.no_cam else 'ON'}")
    logger.info(f"Sensor Output: {args.sensor_output}")
    logger.info(f"Visualization: rerun={'OFF' if not use_rerun else 'ON'}")
    logger.info("=" * 70)

    # Initialize Rerun with proper configuration
    if use_rerun:
        logger.info("Initializing Rerun visualizer...")
        if not init_rerun(session_name="xense_flare_example", spawn=True):
            logger.error("Failed to initialize Rerun, disabling visualization")
            use_rerun = False
        else:
            logger.info("Rerun visualizer initialized! Viewer window should open.")

    # Import and create robot
    from lerobot.robots.xense_flare import XenseFlareConfig, XenseFlare
    from lerobot.robots.xense_flare.config_xense_flare import SensorOutputType
    
    # Create config
    sensor_output = SensorOutputType.RECTIFY if args.sensor_output == "rectify" else SensorOutputType.DIFFERENCE
    
    config = XenseFlareConfig(
        mac_addr=args.mac_addr,
        enable_gripper=not args.no_gripper,
        enable_sensor=not args.no_sensor,
        enable_camera=not args.no_cam,
        sensor_output_type=sensor_output,
    )
    
    logger.info("Initializing Xense Flare...")
    robot = make_robot_from_config(config)
    
    try:
        # Connect
        robot.connect()
        logger.info("Xense Flare connected!")
        
        # Show system info
        info = robot.get_system_info()
        logger.info(f"  MAC: {info['mac_addr']}")
        logger.info(f"  Sensors: {info['sensors']}")
        logger.info(f"  Camera: {'Yes' if info['camera'] else 'No'}")
        logger.info(f"  Gripper: {'Yes' if info['gripper'] else 'No'}")
        logger.info(f"  Vive Tracker: {'Yes' if info['vive_tracker'] else 'No'}")

        # Register button callbacks
        if not args.no_gripper:
            robot.register_button_callback("PRESS", make_button_callback("PRESS"))
            robot.register_button_callback("RELEASE", make_button_callback("RELEASE"))
            robot.register_button_callback("CLICK", make_button_callback("CLICK"))
            robot.register_button_callback("DOUBLE_CLICK", make_button_callback("DOUBLE_CLICK"))
            robot.register_button_callback("LONG_PRESS", make_button_callback("LONG_PRESS"))
            logger.info("Button callbacks registered.")

        # Run live monitor
        run_live_monitor(robot, args, use_rerun=use_rerun)

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Disconnecting Xense Flare...")
        if robot.is_connected:
            robot.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()

