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
Vive Tracker Calibration Script
This script forces recalibration of the Vive Tracker system using pysurvive.

Usage:
    python calibrate_vive.py [--timeout 60] [--origin tracker|lh0]

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
"""

import pysurvive
import sys
import time
import argparse
import signal
import os
import re
import threading
import io
from pathlib import Path

from lerobot.utils.robot_utils import get_logger

logger = get_logger("ViveT_Calib")


class StdoutCapture:
    """Capture C library stdout output (like libsurvive) in a background thread."""
    
    def __init__(self):
        self.captured_lines: list[str] = []
        self._old_stdout_fd = None
        self._old_stderr_fd = None
        self._pipe_read = None
        self._pipe_write = None
        self._capture_thread = None
        self._running = False
    
    def start(self):
        """Start capturing stdout."""
        # Save original stdout file descriptor
        self._old_stdout_fd = os.dup(sys.stdout.fileno())
        
        # Create a pipe
        self._pipe_read, self._pipe_write = os.pipe()
        
        # Redirect stdout to the pipe
        os.dup2(self._pipe_write, sys.stdout.fileno())
        
        # Start capture thread
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
    
    def _capture_loop(self):
        """Background thread to read from pipe."""
        try:
            with os.fdopen(self._pipe_read, 'r', buffering=1) as pipe:
                while self._running:
                    line = pipe.readline()
                    if line:
                        self.captured_lines.append(line.rstrip())
                        # Also print to original stdout
                        os.write(self._old_stdout_fd, (line).encode())
        except Exception:
            pass
    
    def stop(self):
        """Stop capturing and restore stdout."""
        self._running = False
        
        # Restore original stdout
        if self._old_stdout_fd is not None:
            os.dup2(self._old_stdout_fd, sys.stdout.fileno())
            os.close(self._old_stdout_fd)
            self._old_stdout_fd = None
        
        # Close pipe write end
        if self._pipe_write is not None:
            try:
                os.close(self._pipe_write)
            except Exception:
                pass
            self._pipe_write = None
        
        # Wait for capture thread
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=0.5)
            self._capture_thread = None
    
    def get_acc_errors(self) -> dict[int, float]:
        """Parse captured output for 'acc err' values.
        
        Returns:
            Dict mapping lighthouse ID to acc_err value
        """
        acc_errors = {}
        # Pattern: "Global solve with ... for X ... (acc err Y.YYYY)"
        pattern = r"Global solve.*for (\d+).*\(acc err ([\d.]+)\)"
        
        for line in self.captured_lines:
            match = re.search(pattern, line)
            if match:
                lh_id = int(match.group(1))
                acc_err = float(match.group(2))
                acc_errors[lh_id] = acc_err
        
        return acc_errors
    
    def has_global_solve(self) -> bool:
        """Check if any 'Global solve' message was captured."""
        return any("Global solve" in line for line in self.captured_lines)

# Global flag for graceful shutdown
running = True

# Calibration state tracking
class CalibrationState:
    """Track calibration progress and key events."""
    
    def __init__(self):
        # Lighthouse states
        self.lh_detected: dict[str, bool] = {}  # LH name -> detected
        self.lh_pose_solved: dict[str, bool] = {}  # LH name -> pose solved
        self.lh_first_valid_pose: dict[str, float] = {}  # LH name -> timestamp of first valid pose
        
        # Tracker states
        self.tracker_detected: dict[str, bool] = {}  # Tracker name -> detected
        self.tracker_first_valid_pose: dict[str, float] = {}  # Tracker name -> timestamp
        self.tracker_pose_stable: dict[str, bool] = {}  # Tracker name -> pose stable
        
        # Overall calibration
        self.calibration_started = False
        self.calibration_converged = False
        self.config_saved = False
        
        # Pose stability tracking (for detecting convergence)
        self.pose_history: dict[str, list] = {}  # name -> list of recent poses
        self.stability_window = 10  # Number of samples to check stability
        self.stability_threshold = 0.005  # Max position variance for "stable" (5mm)
    
    def check_pose_stability(self, name: str, pos: list) -> bool:
        """Check if pose is stable (low variance over recent samples)."""
        if name not in self.pose_history:
            self.pose_history[name] = []
        
        self.pose_history[name].append(pos)
        
        # Keep only recent samples
        if len(self.pose_history[name]) > self.stability_window:
            self.pose_history[name] = self.pose_history[name][-self.stability_window:]
        
        # Need enough samples
        if len(self.pose_history[name]) < self.stability_window:
            return False
        
        # Calculate variance
        import numpy as np
        poses = np.array(self.pose_history[name])
        variance = np.var(poses, axis=0).sum()
        
        return variance < self.stability_threshold
    
    def get_status_summary(self) -> str:
        """Get a formatted status summary."""
        lines = []
        
        # Lighthouse status
        lh_count = len(self.lh_detected)
        lh_solved = sum(1 for v in self.lh_pose_solved.values() if v)
        lines.append(f"  Lighthouses: {lh_count} detected, {lh_solved} pose solved")
        for lh_name in sorted(self.lh_detected.keys()):
            detected = "✅" if self.lh_detected.get(lh_name) else "❌"
            solved = "✅" if self.lh_pose_solved.get(lh_name) else "⏳"
            lines.append(f"    {lh_name}: Detected {detected} | Pose {solved}")
        
        # Tracker status
        tracker_count = len(self.tracker_detected)
        tracker_stable = sum(1 for v in self.tracker_pose_stable.values() if v)
        lines.append(f"  Trackers: {tracker_count} detected, {tracker_stable} stable")
        for tracker_name in sorted(self.tracker_detected.keys()):
            detected = "✅" if self.tracker_detected.get(tracker_name) else "❌"
            has_pose = "✅" if tracker_name in self.tracker_first_valid_pose else "⏳"
            stable = "✅" if self.tracker_pose_stable.get(tracker_name) else "⏳"
            lines.append(f"    {tracker_name}: Detected {detected} | Pose {has_pose} | Stable {stable}")
        
        return "\n".join(lines)


def check_config_file() -> tuple[bool, str]:
    """Check if libsurvive config file exists and when it was last modified."""
    config_path = Path.home() / ".config" / "libsurvive" / "config.json"
    if config_path.exists():
        mtime = os.path.getmtime(config_path)
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        return True, mtime_str
    return False, ""


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

    # Check existing config file
    config_exists, config_mtime = check_config_file()
    if config_exists:
        logger.info(f"📁 Existing config found (last modified: {config_mtime})")
        logger.info("   Will be overwritten by new calibration")
    else:
        logger.info("📁 No existing config found, will create new one")

    # Display origin mode
    if origin == "lh0":
        logger.info("🎯 Origin Mode: LH0 (Lighthouse 0 at origin, looking +X)")
    else:
        logger.info("🎯 Origin Mode: Tracker (Tracker at origin, LH0 on +Y axis)")

    logger.info("")
    logger.info("📋 Pre-calibration checklist:")
    logger.info("   ✓ All Lighthouse base stations powered on (check green LED)")
    logger.info("   ✓ Tracker(s) have clear line of sight to lighthouses")
    logger.info("   ✓ Tracker(s) are stationary during calibration")
    logger.info("")

    # Initialize calibration state tracker
    cal_state = CalibrationState()
    config_mtime_before = os.path.getmtime(Path.home() / ".config" / "libsurvive" / "config.json") if config_exists else 0

    # Build arguments with force-calibrate flag
    args = sys.argv[:1]  # Keep program name
    args.extend(["--force-calibrate"])
    
    # Add center-on-lh0 flag if origin is lh0
    if origin == "lh0":
        args.extend(["--center-on-lh0"])

    # Start stdout capture to catch libsurvive "acc err" messages
    stdout_capture = StdoutCapture()
    stdout_capture.start()

    # Initialize pysurvive context with calibration flag
    logger.info("🚀 Initializing pysurvive with --force-calibrate...")
    try:
        actx = pysurvive.SimpleContext(args)
    except Exception as e:
        stdout_capture.stop()
        logger.error(f"❌ Failed to initialize pysurvive: {e}")
        return False

    logger.info("⏳ Waiting for devices...")

    # Device tracking
    detected_devices = {}  # {name: {"samples": 0, "valid": 0, "last_pos": None}}
    start_time = time.time()
    last_status_time = 0
    
    # Initial device detection phase (5 seconds)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 1: Device Detection (5 seconds)")
    logger.info("=" * 60)
    detection_start = time.time()
    while time.time() - detection_start < 5.0 and running:
        updated = actx.NextUpdated()
        if updated:
            name = str(updated.Name(), "utf-8")
            if name not in detected_devices:
                detected_devices[name] = {"samples": 0, "valid": 0, "last_pos": None}
                
                # Log detection with appropriate icon
                if is_lighthouse_device(name):
                    logger.info(f"🏠 Lighthouse detected: {name}")
                    cal_state.lh_detected[name] = True
                elif is_tracker_device(name):
                    logger.info(f"📡 Tracker detected: {name}")
                    cal_state.tracker_detected[name] = True
                else:
                    logger.info(f"❓ Unknown device detected: {name}")
        time.sleep(0.01)
    
    # Categorize devices
    lighthouses = [n for n in detected_devices if is_lighthouse_device(n)]
    trackers = [n for n in detected_devices if is_tracker_device(n)]
    
    logger.info("")
    logger.info(f"📊 Detection Summary:")
    logger.info(f"   Lighthouses: {len(lighthouses)} - {lighthouses}")
    logger.info(f"   Trackers: {len(trackers)} - {trackers}")

    if len(lighthouses) < 2:
        logger.warn(f"⚠️  Only {len(lighthouses)} lighthouse(s) detected. "
              "2 lighthouses recommended for best results.")

    if not trackers:
        logger.warn("⚠️  No trackers detected! Will continue waiting...")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Calibration (Keep trackers stationary!)")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    # Run calibration loop
    try:
        while actx.Running() and running:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > timeout:
                logger.info(f"⏰ Calibration timeout ({timeout}s) reached.")
                break

            updated = actx.NextUpdated()
            if updated:
                name = str(updated.Name(), "utf-8")
                current_time = time.time()
                
                # Add new device if not seen before
                if name not in detected_devices:
                    detected_devices[name] = {"samples": 0, "valid": 0, "last_pos": None, "last_rot": None}
                    if is_tracker_device(name):
                        logger.info(f"📡 New tracker detected during calibration: {name}")
                        trackers.append(name)
                        cal_state.tracker_detected[name] = True
                    elif is_lighthouse_device(name):
                        logger.info(f"🏠 New lighthouse detected during calibration: {name}")
                        lighthouses.append(name)
                        cal_state.lh_detected[name] = True
                
                pose_obj = updated.Pose()
                pose_data = pose_obj[0]

                # Get position and rotation
                pos = [pose_data.Pos[0], pose_data.Pos[1], pose_data.Pos[2]]
                rot = [pose_data.Rot[0], pose_data.Rot[1], pose_data.Rot[2], pose_data.Rot[3]]  # [w, x, y, z]
                
                detected_devices[name]["samples"] += 1
                detected_devices[name]["last_pos"] = pos
                detected_devices[name]["last_rot"] = rot
                
                # Check if position is valid (not NaN and within reasonable range)
                is_valid = all(abs(p) < 100 for p in pos) and not any(p != p for p in pos)
                
                if is_valid:
                    detected_devices[name]["valid"] += 1
                    
                    # Track first valid pose for each device
                    if is_lighthouse_device(name):
                        if name not in cal_state.lh_first_valid_pose:
                            cal_state.lh_first_valid_pose[name] = current_time
                            cal_state.lh_pose_solved[name] = True
                            logger.info(f"✅ {name}: Lighthouse pose SOLVED! "
                                       f"Pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")
                    
                    elif is_tracker_device(name):
                        if name not in cal_state.tracker_first_valid_pose:
                            cal_state.tracker_first_valid_pose[name] = current_time
                            logger.info(f"✅ {name}: First valid pose received! "
                                       f"Pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")
                        
                        # Check pose stability
                        if not cal_state.tracker_pose_stable.get(name, False):
                            if cal_state.check_pose_stability(name, pos):
                                cal_state.tracker_pose_stable[name] = True
                                logger.info(f"✅ {name}: Pose STABLE (variance < 5mm)")

            # Print status every 3 seconds
            if time.time() - last_status_time >= 3.0:
                last_status_time = time.time()
                
                # Check if config file was updated
                config_path = Path.home() / ".config" / "libsurvive" / "config.json"
                if config_path.exists():
                    config_mtime_now = os.path.getmtime(config_path)
                    if config_mtime_now > config_mtime_before and not cal_state.config_saved:
                        cal_state.config_saved = True
                        logger.info(f"💾 Config file SAVED to {config_path}")
                
                logger.info("")
                logger.info(f"[{elapsed:.0f}s] 📊 Calibration Status:")
                logger.info("-" * 50)
                
                # Show tracker status (Lighthouse 2.0 doesn't report samples via NextUpdated)
                for tracker_name in sorted(trackers):
                    if tracker_name in detected_devices:
                        info = detected_devices[tracker_name]
                        pos = info["last_pos"]
                        has_pose = "✅" if tracker_name in cal_state.tracker_first_valid_pose else "⏳"
                        stable = "✅" if cal_state.tracker_pose_stable.get(tracker_name) else "⏳"
                        
                        if pos:
                            pos_str = f"[{pos[0]:+7.3f}, {pos[1]:+7.3f}, {pos[2]:+7.3f}]"
                        else:
                            pos_str = "[waiting...]"
                        
                        logger.info(f"  📡 {tracker_name}: Pose {has_pose} Stable {stable}")
                        logger.info(f"      Samples={info['samples']:5d}, Valid={info['valid']:5d}")
                        logger.info(f"      Pos={pos_str}")
                
                if not trackers:
                    logger.info("  ⏳ No trackers detected yet...")
                
                # Overall calibration status
                # Note: Lighthouse 2.0 doesn't send pose via NextUpdated, so we infer success from tracker status
                all_trackers_stable = all(cal_state.tracker_pose_stable.get(t, False) for t in trackers) if trackers else False
                
                if all_trackers_stable and cal_state.config_saved:
                    logger.info("")
                    logger.info("🎉 CALIBRATION CONVERGED - All conditions met!")
                    cal_state.calibration_converged = True
                elif trackers and not all_trackers_stable:
                    logger.info("")
                    logger.info("⏳ Waiting for tracker poses to stabilize...")
                
                logger.info("-" * 50)

            time.sleep(0.001)

    except Exception as e:
        logger.error(f"❌ Error during calibration: {e}")

    # Stop stdout capture and get acc_err values
    stdout_capture.stop()
    acc_errors = stdout_capture.get_acc_errors()
    has_global_solve = stdout_capture.has_global_solve()

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("         📊 Calibration Summary")
    logger.info("=" * 60)
    logger.info(f"⏱️  Duration: {time.time() - start_time:.1f} seconds")
    logger.info(f"🎯 Origin Mode: {'LH0 (Lighthouse at origin)' if origin == 'lh0' else 'Tracker (Tracker at origin)'}")
    logger.info("")
    
    # Lighthouse summary with acc_err from libsurvive
    logger.info("🏠 Lighthouses:")
    if lighthouses:
        logger.info(f"   Detected: {len(lighthouses)} - {lighthouses}")
    else:
        logger.info("   ❌ None detected!")
    
    # Show Global Solve status and acc_err
    if has_global_solve:
        logger.info("   Global Solve: ✅ SUCCESS")
        if acc_errors:
            for lh_id, acc_err in sorted(acc_errors.items()):
                lh_name = f"LH{lh_id}"
                # acc_err < 0.01 is excellent, < 0.1 is good
                if acc_err < 0.01:
                    status = "✅ excellent"
                elif acc_err < 0.1:
                    status = "✅ good"
                else:
                    status = "⚠️  high"
                logger.info(f"      {lh_name}: acc_err = {acc_err:.6f} ({status})")
    else:
        logger.info("   Global Solve: ⏳ Not detected (may need more time)")
    logger.info("")
    
    # Tracker summary
    logger.info("📡 Trackers:")
    all_trackers_ok = True
    for tracker_name in sorted(trackers):
        if tracker_name in detected_devices:
            info = detected_devices[tracker_name]
            has_pose = tracker_name in cal_state.tracker_first_valid_pose
            is_stable = cal_state.tracker_pose_stable.get(tracker_name, False)
            
            if has_pose and is_stable:
                status = "✅ OK (stable)"
            elif has_pose:
                status = "⚠️  UNSTABLE (pose received but variance high)"
                all_trackers_ok = False
            elif info["valid"] > 100:
                status = "⚠️  LOW STABILITY"
                all_trackers_ok = False
            else:
                status = "❌ FAILED (no valid pose)"
                all_trackers_ok = False
            
            logger.info(f"   {tracker_name}: {status}")
            logger.info(f"      Samples: {info['samples']}, Valid: {info['valid']}")
            if info.get("last_pos"):
                pos = info["last_pos"]
                logger.info(f"      Last Pos: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")
    
    if not trackers:
        logger.info("   ❌ No trackers detected!")
        all_trackers_ok = False
    
    logger.info("")
    
    # Config file status
    config_path = Path.home() / ".config" / "libsurvive" / "config.json"
    if config_path.exists():
        config_mtime_now = os.path.getmtime(config_path)
        if config_mtime_now > config_mtime_before:
            logger.info(f"💾 Config: ✅ SAVED to {config_path}")
        else:
            logger.info(f"💾 Config: ⚠️  Not updated (using existing)")
    else:
        logger.info(f"💾 Config: ❌ NOT SAVED")
    
    logger.info("")
    
    # Overall result
    # Note: Lighthouse 2.0 doesn't send pose updates via NextUpdated()
    # If trackers have stable valid poses AND global solve succeeded, calibration is successful
    all_success = all_trackers_ok and trackers and (lighthouses or has_global_solve)
    
    if all_success:
        logger.info("=" * 60)
        logger.info("🎉 CALIBRATION SUCCESSFUL!")
        logger.info("=" * 60)
        logger.info("All trackers have stable poses.")
        if acc_errors:
            max_err = max(acc_errors.values())
            logger.info(f"Max acc_err: {max_err:.6f} (lower is better)")
        logger.info(f"Config saved to: {config_path}")
        return True
    else:
        logger.info("=" * 60)
        logger.warn("⚠️  CALIBRATION INCOMPLETE")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Troubleshooting tips:")
        if not lighthouses:
            logger.warn("  🏠 Lighthouse issues:")
            logger.warn("     - Check if lighthouses are powered on (green LED)")
            logger.warn("     - Ensure lighthouses can see each other")
            logger.warn("     - Check for reflective surfaces causing interference")
        if not all_trackers_ok:
            logger.warn("  📡 Tracker issues:")
            logger.warn("     - Ensure trackers have clear line of sight to BOTH lighthouses")
            logger.warn("     - Check tracker is paired (green LED on tracker)")
            logger.warn("     - Keep trackers STATIONARY during calibration")
            logger.warn("     - Move trackers closer to lighthouses")
        logger.warn("  ⏱️  Try running with longer timeout: --timeout 120")
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
        """
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Calibration timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--origin",
        type=str,
        choices=["tracker", "lh0"],
        default="tracker",
        help="Coordinate origin mode: 'tracker' (origin at Tracker) or 'lh0' (origin at Lighthouse 0). Default: tracker"
    )

    args = parser.parse_args()
    calibrate_vive(args.timeout, args.origin)


if __name__ == "__main__":
    main()
