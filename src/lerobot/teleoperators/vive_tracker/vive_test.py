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
Vive Tracker Teleoperator Test - Test lerobot ViveTrackerTeleop with terminal output and Rerun visualization
"""

import argparse
import signal
import time
from collections import deque

import numpy as np
import spdlog

from lerobot.teleoperators.vive_tracker import ViveTrackerConfig, ViveTrackerTeleop

# Get logger for this module
logger = spdlog.ConsoleLogger("ViveTrackerTest")

# Global flag for graceful shutdown
running = True

# Check if rerun is available
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    logger.warn("Rerun not installed. Install with: pip install rerun-sdk")


def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown"""
    global running
    print("\n\nReceived Ctrl+C, stopping...")
    running = False


class RerunVisualizer:
    """Rerun visualizer for Vive Tracker pose data."""

    def __init__(self, session_name: str = "vive_tracker_test", trajectory_length: int = 500):
        """
        Initialize Rerun visualizer.

        Args:
            session_name: Name for the Rerun session
            trajectory_length: Number of points to keep in trajectory
        """
        self.session_name = session_name
        self.trajectory_length = trajectory_length
        self.trajectory: deque = deque(maxlen=trajectory_length)
        self._initialized = False

    def init(self) -> bool:
        """Initialize Rerun session."""
        if not RERUN_AVAILABLE:
            logger.error("Rerun not available")
            return False

        try:
            rr.init(self.session_name)
            rr.spawn(memory_limit="500MB")

            # Set up 3D view
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

            # Log coordinate axes
            self._log_axes()

            # Log ground plane
            self._log_ground_plane()

            self._initialized = True
            logger.info(f"✅ Rerun initialized: {self.session_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Rerun: {e}")
            return False

    def _log_axes(self):
        """Log coordinate axes for reference."""
        axis_length = 0.3
        axis_radius = 0.005

        # X axis (red)
        rr.log(
            "world/axes/x",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[[axis_length, 0, 0]],
                colors=[[255, 0, 0]],
                radii=[axis_radius],
            ),
            static=True,
        )
        # Y axis (green)
        rr.log(
            "world/axes/y",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[[0, axis_length, 0]],
                colors=[[0, 255, 0]],
                radii=[axis_radius],
            ),
            static=True,
        )
        # Z axis (blue)
        rr.log(
            "world/axes/z",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[[0, 0, axis_length]],
                colors=[[0, 0, 255]],
                radii=[axis_radius],
            ),
            static=True,
        )

    def _log_ground_plane(self):
        """Log a ground plane grid for reference."""
        # Create grid lines
        grid_size = 2.0
        grid_step = 0.2
        lines = []

        # X lines
        for y in np.arange(-grid_size, grid_size + grid_step, grid_step):
            lines.append([[-grid_size, y, 0], [grid_size, y, 0]])

        # Y lines
        for x in np.arange(-grid_size, grid_size + grid_step, grid_step):
            lines.append([[x, -grid_size, 0], [x, grid_size, 0]])

        rr.log(
            "world/ground_grid",
            rr.LineStrips3D(lines, colors=[[100, 100, 100, 100]], radii=[0.001]),
            static=True,
        )

    def log_pose(self, action: dict, device_name: str = "tracker"):
        """
        Log pose data to Rerun.

        Args:
            action: Action dictionary with tcp.x/y/z and tcp.qw/qx/qy/qz
            device_name: Name of the device for logging
        """
        if not self._initialized:
            return

        # Extract position
        x = action.get("tcp.x", 0.0)
        y = action.get("tcp.y", 0.0)
        z = action.get("tcp.z", 0.0)
        position = [x, y, z]

        # Extract quaternion (Rerun uses [x, y, z, w] format)
        qw = action.get("tcp.qw", 1.0)
        qx = action.get("tcp.qx", 0.0)
        qy = action.get("tcp.qy", 0.0)
        qz = action.get("tcp.qz", 0.0)
        quaternion = rr.Quaternion(xyzw=[qx, qy, qz, qw])

        # Log transform (position and rotation)
        rr.log(
            f"world/{device_name}",
            rr.Transform3D(
                translation=position,
                rotation=quaternion,
            ),
        )

        # Log tracker as a box
        rr.log(
            f"world/{device_name}/body",
            rr.Boxes3D(
                half_sizes=[[0.05, 0.03, 0.02]],
                colors=[[0, 200, 255]],
            ),
        )

        # Log orientation axes on the tracker
        axis_length = 0.1
        # Forward (X) - red
        rr.log(
            f"world/{device_name}/axes/x",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[[axis_length, 0, 0]],
                colors=[[255, 100, 100]],
                radii=[0.003],
            ),
        )
        # Left (Y) - green
        rr.log(
            f"world/{device_name}/axes/y",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[[0, axis_length, 0]],
                colors=[[100, 255, 100]],
                radii=[0.003],
            ),
        )
        # Up (Z) - blue
        rr.log(
            f"world/{device_name}/axes/z",
            rr.Arrows3D(
                origins=[[0, 0, 0]],
                vectors=[[0, 0, axis_length]],
                colors=[[100, 100, 255]],
                radii=[0.003],
            ),
        )

        # Add to trajectory
        self.trajectory.append(position)

        # Log trajectory
        if len(self.trajectory) > 1:
            trajectory_points = list(self.trajectory)
            rr.log(
                f"world/{device_name}/trajectory",
                rr.LineStrips3D(
                    [trajectory_points],
                    colors=[[255, 165, 0, 200]],  # Orange with transparency
                    radii=[0.002],
                ),
            )

        # Log scalar values for time series
        rr.log(f"pose/{device_name}/x", rr.Scalars(values=[x]))
        rr.log(f"pose/{device_name}/y", rr.Scalars(values=[y]))
        rr.log(f"pose/{device_name}/z", rr.Scalars(values=[z]))
        rr.log(f"pose/{device_name}/qw", rr.Scalars(values=[qw]))
        rr.log(f"pose/{device_name}/qx", rr.Scalars(values=[qx]))
        rr.log(f"pose/{device_name}/qy", rr.Scalars(values=[qy]))
        rr.log(f"pose/{device_name}/qz", rr.Scalars(values=[qz]))

    def log_timing(self, get_action_ms: float, total_ms: float, fps: float):
        """Log timing metrics."""
        if not self._initialized:
            return

        rr.log("timing/get_action_ms", rr.Scalars(values=[get_action_ms]))
        rr.log("timing/total_ms", rr.Scalars(values=[total_ms]))
        rr.log("timing/fps", rr.Scalars(values=[fps]))

    def shutdown(self):
        """Shutdown Rerun."""
        if self._initialized:
            try:
                rr.disconnect()
            except Exception:
                pass
            self._initialized = False


def print_action_data(action: dict, update_count: int, fps: float = 0.0, timing: dict = None, use_rerun: bool = False):
    """
    Print action data in a formatted table.

    Args:
        action: Dictionary of action features from teleop.get_action()
        update_count: Current update count for display
        fps: Current frames per second
        timing: Dictionary with timing info (get_action_ms, total_ms)
        use_rerun: Whether Rerun visualization is enabled
    """
    # Header
    rerun_str = " | Rerun: ON" if use_rerun else ""
    print(f"\n{'='*75}")
    print(f"  ViveTrackerTeleop Test (Update #{update_count}) | FPS: {fps:.1f}{rerun_str}")
    print(f"{'='*75}")

    # Performance section
    if timing:
        rerun_timing = f" | Rerun: {timing.get('rerun', 0):.2f}ms" if timing.get('rerun', 0) > 0 else ""
        print(
            f"\n  ⏱️  Latency: get_action: {timing.get('get_action', 0):.2f}ms{rerun_timing} | "
            f"Total: {timing.get('total', 0):.2f}ms"
        )

    if not action:
        print("  No action data available")
        print(f"{'='*75}\n")
        return

    # Position
    x = action.get("tcp.x", 0.0)
    y = action.get("tcp.y", 0.0)
    z = action.get("tcp.z", 0.0)

    # Quaternion
    qw = action.get("tcp.qw", 1.0)
    qx = action.get("tcp.qx", 0.0)
    qy = action.get("tcp.qy", 0.0)
    qz = action.get("tcp.qz", 0.0)

    print(f"\n  [Action Output]")
    print(f"    Position (m):    X={x:+9.5f}  Y={y:+9.5f}  Z={z:+9.5f}")
    print(f"    Rotation (quat): W={qw:+9.5f}  X={qx:+9.5f}  Y={qy:+9.5f}  Z={qz:+9.5f}")

    print(f"\n{'='*75}")
    print("  Press Ctrl+C to stop")
    print(f"{'='*75}\n")


def run_terminal_test(rate_hz: float = 100.0, config_path: str = None, lh_config: str = None, use_rerun: bool = False):
    """
    Run terminal test for ViveTrackerTeleop.

    Args:
        rate_hz: Update rate in Hz (default 100 Hz)
        config_path: Optional pysurvive config file path
        lh_config: Optional lighthouse configuration
        use_rerun: Enable Rerun visualization
    """
    global running

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 60)
    logger.info("ViveTrackerTeleop Test - LeRobot Teleoperator")
    logger.info("=" * 60)

    # Initialize Rerun if requested
    viz = None
    if use_rerun:
        if not RERUN_AVAILABLE:
            logger.error("Rerun is not installed. Install with: pip install rerun-sdk")
            return False
        logger.info("Initializing Rerun visualizer...")
        viz = RerunVisualizer(session_name="vive_tracker_test")
        if not viz.init():
            logger.error("Failed to initialize Rerun visualizer")
            return False
        logger.info("✅ Rerun visualizer initialized!")

    # Create config
    config = ViveTrackerConfig(
        id="vive_test",
        tracker_name=None,  # Use first detected tracker
        config_path=config_path,
        lh_config=lh_config,
        device_wait_timeout=10.0,
        required_trackers=1,
        filter_window_size=1,
        enable_position_jump_filter=True,
    )

    logger.info("Initializing ViveTrackerTeleop...")

    teleop = ViveTrackerTeleop(config)

    try:
        teleop.connect()
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        if viz:
            viz.shutdown()
        return False

    logger.info("✅ Connected successfully!")
    logger.info(f"Starting terminal test at {rate_hz} Hz...")
    if use_rerun:
        logger.info("Rerun visualization enabled - view in Rerun viewer")
    logger.info("Press Ctrl+C to stop\n")

    time.sleep(1.0)  # Give time for user to read

    update_count = 0
    sleep_time = 1.0 / rate_hz

    # FPS tracking
    fps = 0.0
    fps_update_interval = 10
    last_fps_time = time.time()

    # Timing accumulators
    time_get_action = 0.0
    time_rerun = 0.0
    time_total = 0.0

    # Last timing info for display
    last_timing_info = None

    try:
        while running:
            loop_start = time.perf_counter()

            # Get action from teleoperator
            t0 = time.perf_counter()
            try:
                action = teleop.get_action()
            except Exception as e:
                logger.error(f"Error getting action: {e}")
                action = {}
            t1 = time.perf_counter()
            time_get_action += (t1 - t0)

            # Log to Rerun if enabled
            if viz and action:
                t2 = time.perf_counter()
                viz.log_pose(action, device_name="tracker")
                time_rerun += time.perf_counter() - t2

            # Clear screen and print data
            print("\033[2J\033[H", end="")  # Clear screen and move to top
            print_action_data(action, update_count, fps=fps, timing=last_timing_info, use_rerun=use_rerun)

            update_count += 1

            # Calculate total loop time (before sleep)
            loop_end = time.perf_counter()
            time_total += (loop_end - loop_start)

            # Update FPS and timing stats
            if update_count % fps_update_interval == 0:
                current_time = time.time()
                fps = fps_update_interval / (current_time - last_fps_time)
                last_fps_time = current_time

                # Calculate average timing (in ms)
                avg_get_action = (time_get_action / fps_update_interval) * 1000
                avg_rerun = (time_rerun / fps_update_interval) * 1000
                avg_total = (time_total / fps_update_interval) * 1000

                last_timing_info = {
                    "get_action": avg_get_action,
                    "rerun": avg_rerun,
                    "total": avg_total,
                }

                # Log timing to Rerun
                if viz:
                    viz.log_timing(avg_get_action, avg_total, fps)

                # Reset accumulators
                time_get_action = 0.0
                time_rerun = 0.0
                time_total = 0.0

            # Dynamic sleep: only sleep for remaining time to achieve target rate
            elapsed = time.perf_counter() - loop_start
            remaining_sleep = sleep_time - elapsed
            if remaining_sleep > 0:
                time.sleep(remaining_sleep)

    except Exception as e:
        logger.error(f"Error during test: {e}")
    finally:
        logger.info("Disconnecting ViveTrackerTeleop...")
        teleop.disconnect()
        if viz:
            viz.shutdown()
        logger.info("Done!")

    return True


def run_benchmark(config_path: str = None, lh_config: str = None):
    """
    Run benchmark to test ViveTrackerTeleop throughput.
    Measures get_action() call rate without rate limiting.
    """
    global running

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 70)
    logger.info("ViveTrackerTeleop Benchmark - Testing get_action() throughput")
    logger.info("=" * 70)

    # Create config
    config = ViveTrackerConfig(
        id="vive_benchmark",
        config_path=config_path,
        lh_config=lh_config,
        device_wait_timeout=10.0,
        required_trackers=1,
        filter_window_size=1,
        enable_position_jump_filter=False,  # Disable for raw benchmark
    )

    teleop = ViveTrackerTeleop(config)

    try:
        teleop.connect()
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return False

    logger.info("Starting benchmark... Press Ctrl+C to stop")
    logger.info("=" * 70)

    # Tracking stats
    read_count = 0
    last_action = None

    # Time tracking
    start_time = time.time()
    last_report_time = start_time
    report_interval = 1.0  # Report every 1 second

    # Timing accumulator
    time_get_action = 0.0
    reads_since_report = 0

    # Change detection
    change_count = 0

    try:
        while running:
            t0 = time.perf_counter()

            try:
                action = teleop.get_action()
            except Exception:
                continue

            t1 = time.perf_counter()
            time_get_action += (t1 - t0)

            # Check if action changed
            if last_action is not None and action:
                changed = False
                for key in ["tcp.x", "tcp.y", "tcp.z"]:
                    if abs(action.get(key, 0) - last_action.get(key, 0)) > 1e-6:
                        changed = True
                        break
                if changed:
                    change_count += 1

            last_action = action.copy() if action else None
            read_count += 1
            reads_since_report += 1

            # Report stats every interval
            current_time = time.time()
            if current_time - last_report_time >= report_interval:
                elapsed = current_time - last_report_time

                # Calculate rates
                read_rate = reads_since_report / elapsed
                avg_read_time = (time_get_action / reads_since_report) * 1000 if reads_since_report > 0 else 0
                change_rate = change_count / (current_time - start_time)

                logger.info(
                    f"Read rate: {read_rate:.0f}/s | "
                    f"Latency: {avg_read_time:.3f}ms | "
                    f"Data change rate: {change_rate:.1f}Hz"
                )

                # Reset for next interval
                last_report_time = current_time
                time_get_action = 0.0
                reads_since_report = 0

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info("Summary:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Total reads: {read_count}")
        logger.info(f"  Average rate: {read_count / total_time:.1f} reads/s")
        logger.info(f"  Data changes: {change_count} ({change_count / total_time:.1f} Hz)")
        logger.info("=" * 70)

        logger.info("Disconnecting...")
        teleop.disconnect()
        logger.info("Done!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="ViveTrackerTeleop Test - Test lerobot Vive Tracker teleoperator"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Update rate in Hz (default: 100)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pysurvive configuration file",
    )
    parser.add_argument(
        "--lh",
        type=str,
        default=None,
        help="Lighthouse configuration string",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Enable Rerun visualization for 3D pose display",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode to test maximum throughput (no rate limiting)",
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(config_path=args.config, lh_config=args.lh)
    else:
        run_terminal_test(rate_hz=args.rate, config_path=args.config, lh_config=args.lh, use_rerun=args.rerun)


if __name__ == "__main__":
    main()
