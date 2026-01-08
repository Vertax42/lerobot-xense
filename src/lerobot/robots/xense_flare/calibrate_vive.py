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
Vive Tracker Calibration Script for Xense Flare.

This script forces recalibration of the Vive Tracker system using pysurvive.
Run this when the tracker pose seems incorrect or after moving the lighthouses.

Usage:
    python -m lerobot.robots.xense_flare.calibrate_vive
    python -m lerobot.robots.xense_flare.calibrate_vive --timeout 120
    python -m lerobot.robots.xense_flare.calibrate_vive --origin lh0

Origin Modes:
    tracker: (Default) The coordinate origin is at the Tracker's position during calibration.
             LH0 will be placed on the +Y axis. Good for robot-centric applications.
    lh0:     The coordinate origin is at Lighthouse 0 (LH0) position.
             LH0 looks in the +X direction. Good for fixed room-centric applications.

Instructions:
    1. Make sure all Lighthouse base stations are powered on and visible
    2. Place the tracker(s) in a stable position with clear line of sight to all lighthouses
    3. Keep the tracker(s) stationary during calibration
    4. Wait for calibration to complete

Examples:
    # Default calibration (tracker as origin, 60s timeout)
    python -m lerobot.robots.xense_flare.calibrate_vive

    # Use Lighthouse 0 as origin
    python -m lerobot.robots.xense_flare.calibrate_vive --origin lh0

    # Longer timeout (2 minutes)
    python -m lerobot.robots.xense_flare.calibrate_vive --timeout 120
"""

import argparse
import signal
import sys
import time

import pysurvive

from lerobot.utils.robot_utils import get_logger

logger = get_logger("calibrate_vive")

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown"""
    global running
    logger.info("Received Ctrl+C, stopping calibration...")
    running = False


def is_tracker_device(name: str) -> bool:
    """Check if a device name is a tracker (WM, T2, HMD)"""
    return name.startswith("WM") or name.startswith("T2") or name.startswith("HMD")


def is_lighthouse_device(name: str) -> bool:
    """Check if a device name is a lighthouse"""
    return name.startswith("LH")


def print_banner():
    logger.info("=" * 60)
    logger.info("         Vive Tracker Calibration Tool")
    logger.info("=" * 60)


def calibrate_vive(timeout=60, origin="tracker"):
    """
    Force recalibration of the Vive Tracker system.

    Args:
        timeout: Maximum time to wait for calibration (seconds)
        origin: Origin mode - "tracker" (default) or "lh0"
                - "tracker": Origin at Tracker position, LH0 on +Y axis
                - "lh0": Origin at LH0 position, LH0 looks in +X direction
    """
    global running

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print_banner()

    # Display origin mode
    if origin == "lh0":
        logger.info("Origin Mode: LH0 (Lighthouse 0 at origin, looking +X)")
    else:
        logger.info("Origin Mode: Tracker (Tracker at origin, LH0 on +Y axis)")

    logger.info("Starting forced calibration...")
    logger.info("Please ensure:")
    logger.info("  - All Lighthouse base stations are powered on")
    logger.info("  - Tracker(s) have clear line of sight to lighthouses")
    logger.info("  - Tracker(s) are stationary during calibration")

    # Build arguments with force-calibrate flag
    args = sys.argv[:1]  # Keep program name
    args.extend(["--force-calibrate"])

    # Add center-on-lh0 flag if origin is lh0
    if origin == "lh0":
        args.extend(["--center-on-lh0"])

    # Initialize pysurvive context with calibration flag
    logger.info("Initializing pysurvive with --force-calibrate...")
    try:
        actx = pysurvive.SimpleContext(args)
    except Exception as e:
        logger.error(f"Failed to initialize pysurvive: {e}")
        return False

    logger.info("Waiting for devices (using NextUpdated method)...")

    # Device tracking
    detected_devices = {}  # {name: {"samples": 0, "valid": 0, "last_pos": None}}
    start_time = time.time()
    last_status_time = 0

    # Initial device detection phase (5 seconds)
    logger.info("Device detection phase (5 seconds)...")
    detection_start = time.time()
    while time.time() - detection_start < 5.0 and running:
        updated = actx.NextUpdated()
        if updated:
            name = str(updated.Name(), "utf-8")
            if name not in detected_devices:
                detected_devices[name] = {"samples": 0, "valid": 0, "last_pos": None}
                logger.info(f"Detected: {name}")
        time.sleep(0.01)

    # Categorize devices
    lighthouses = [n for n in detected_devices if is_lighthouse_device(n)]
    trackers = [n for n in detected_devices if is_tracker_device(n)]

    logger.info(f"Detected Lighthouses: {lighthouses}")
    logger.info(f"Detected Trackers: {trackers}")

    if len(lighthouses) < 2:
        logger.warn(
            f"Only {len(lighthouses)} lighthouse(s) detected. "
            "For best results, 2 lighthouses are recommended."
        )

    if not trackers:
        logger.warn("No trackers detected! Will continue waiting during calibration...")

    logger.info("Calibrating... Keep all trackers stationary!")
    logger.info("Press Ctrl+C to stop")

    # Run calibration loop
    try:
        while actx.Running() and running:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > timeout:
                logger.info(f"Calibration timeout ({timeout}s) reached.")
                break

            updated = actx.NextUpdated()
            if updated:
                name = str(updated.Name(), "utf-8")

                # Add new device if not seen before
                if name not in detected_devices:
                    detected_devices[name] = {"samples": 0, "valid": 0, "last_pos": None, "last_rot": None}
                    if is_tracker_device(name):
                        logger.info(f"New tracker detected during calibration: {name}")
                        trackers.append(name)
                    elif is_lighthouse_device(name):
                        lighthouses.append(name)

                pose_obj = updated.Pose()
                pose_data = pose_obj[0]

                # Get position and rotation
                pos = [pose_data.Pos[0], pose_data.Pos[1], pose_data.Pos[2]]
                rot = [pose_data.Rot[0], pose_data.Rot[1], pose_data.Rot[2], pose_data.Rot[3]]  # [w, x, y, z]

                detected_devices[name]["samples"] += 1
                detected_devices[name]["last_pos"] = pos
                detected_devices[name]["last_rot"] = rot

                # Check if position is valid (not NaN and within reasonable range)
                if all(abs(p) < 100 for p in pos) and not any(p != p for p in pos):
                    detected_devices[name]["valid"] += 1

                # Print status every 3 seconds
                current_time = time.time()
            if current_time - last_status_time >= 3.0:
                last_status_time = current_time
                logger.info(f"[{elapsed:.0f}s] Calibration Status:")
                logger.info("-" * 50)

                for name in trackers:
                    if name in detected_devices:
                        info = detected_devices[name]
                        pos = info["last_pos"]
                        rot = info["last_rot"]
                        if pos:
                            pos_str = f"[{pos[0]:+7.3f}, {pos[1]:+7.3f}, {pos[2]:+7.3f}]"
                        else:
                            pos_str = "[no data]"
                        if rot:
                            rot_str = f"[{rot[0]:+6.3f}, {rot[1]:+6.3f}, {rot[2]:+6.3f}, {rot[3]:+6.3f}]"
                        else:
                            rot_str = "[no data]"
                        logger.info(f"  {name}: Samples={info['samples']:5d}, Valid={info['valid']:5d}")
                        logger.info(f"         Pos={pos_str}")
                        logger.info(f"         Rot={rot_str}")

                if not trackers:
                    logger.info("  No trackers detected yet...")
                logger.info("-" * 50)

            time.sleep(0.001)

    except Exception as e:
        logger.error(f"Error during calibration: {e}")

    # Print summary
    logger.info("=" * 60)
    logger.info("         Calibration Summary")
    logger.info("=" * 60)
    logger.info(f"Duration: {time.time() - start_time:.1f} seconds")
    logger.info(
        f"Origin Mode: {'LH0 (Lighthouse at origin)' if origin == 'lh0' else 'Tracker (Tracker at origin)'}"
    )
    logger.info(f"Lighthouses: {lighthouses}")
    logger.info(f"Trackers: {trackers}")

    # Per-tracker summary
    all_success = True
    for name in trackers:
        if name in detected_devices:
            info = detected_devices[name]
            status = "OK" if info["valid"] > 100 else "LOW SAMPLES"
            if info["valid"] <= 100:
                all_success = False
            logger.info(f"  {name}: {info['samples']} samples, {info['valid']} valid - {status}")

    if not trackers:
        logger.info("  No trackers were detected!")
        all_success = False

    if all_success and trackers:
        logger.info("Calibration completed successfully for all trackers!")
        logger.info("Configuration should be saved to ~/.config/libsurvive/")
        return True
    else:
        logger.warn("Some trackers have low sample counts. Try:")
        logger.warn("  - Check if lighthouses are powered on (green LED)")
        logger.warn("  - Ensure trackers have clear line of sight to lighthouses")
        logger.warn("  - Make sure trackers are paired with dongles (green LED on tracker)")
        logger.warn("  - Move trackers closer to lighthouses")
        logger.warn("  - Run calibration for longer with --timeout")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Vive Tracker Calibration Tool - Calibrates all detected trackers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Origin Modes:
    tracker  - Origin at Tracker position during calibration (default)
               LH0 will be placed on the +Y axis
               Good for robot-centric applications

    lh0      - Origin at Lighthouse 0 (LH0) position
               LH0 looks in the +X direction
               Good for fixed room-centric applications

Examples:
    python calibrate_vive.py                        # Default: tracker as origin
    python calibrate_vive.py --origin lh0           # LH0 as origin
    python calibrate_vive.py --origin tracker       # Tracker as origin (explicit)
    python calibrate_vive.py --timeout 120          # 2 minute calibration
    python calibrate_vive.py --origin lh0 --timeout 120
        """,
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Calibration timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--origin",
        type=str,
        choices=["tracker", "lh0"],
        default="tracker",
        help="Coordinate origin mode: 'tracker' (origin at Tracker) or 'lh0' (origin at Lighthouse 0). Default: tracker",
    )

    args = parser.parse_args()
    calibrate_vive(args.timeout, args.origin)


if __name__ == "__main__":
    main()
