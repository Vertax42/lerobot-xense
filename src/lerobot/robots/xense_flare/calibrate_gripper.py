#!/usr/bin/env python

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
Gripper Calibration Script for Xense Flare.

This script calibrates the gripper encoder to ensure accurate position readings.
Run this when the gripper position reading seems incorrect or after hardware changes.

Usage:
    python -m lerobot.robots.xense_flare.calibrate_gripper --mac_addr=6ebbc5f53240
    
Or with custom max position:
    python -m lerobot.robots.xense_flare.calibrate_gripper --mac_addr=6ebbc5f53240 --gripper_max_pos=90.0
"""

import argparse
import time

from lerobot.robots.xense_flare import XenseFlare
from lerobot.robots.xense_flare.config_xense_flare import XenseFlareConfig
from lerobot.utils.robot_utils import get_logger

logger = get_logger("GripperCalibration")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Xense Flare gripper encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic calibration
    python -m lerobot.robots.xense_flare.calibrate_gripper --mac_addr=6ebbc5f53240
    
    # With custom max position
    python -m lerobot.robots.xense_flare.calibrate_gripper --mac_addr=6ebbc5f53240 --gripper_max_pos=90.0
    
    # Interactive mode (shows real-time position after calibration)
    python -m lerobot.robots.xense_flare.calibrate_gripper --mac_addr=6ebbc5f53240 --interactive
        """
    )
    parser.add_argument(
        "--mac_addr",
        type=str,
        # required=True,
        default="6ebbc5f53240",
        help="MAC address of the Xense Flare device (e.g., 6ebbc5f53240)"
    )
    parser.add_argument(
        "--gripper_max_pos",
        type=float,
        default=85.0,
        help="Maximum gripper position for normalization (default: 85.0)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode after calibration to monitor position"
    )
    
    args = parser.parse_args()
    
    # Create config (only enable gripper, disable other components for faster startup)
    config = XenseFlareConfig(
        mac_addr=args.mac_addr,
        gripper_max_pos=args.gripper_max_pos,
        enable_gripper=True,
        enable_sensor=False,
        enable_camera=False,
        enable_vive=False,
    )
    
    logger.info("=" * 60)
    logger.info("Xense Flare Gripper Calibration")
    logger.info("=" * 60)
    logger.info(f"MAC Address: {args.mac_addr}")
    logger.info(f"Gripper Max Position: {args.gripper_max_pos}")
    logger.info("")
    
    xense_flare = None
    try:
        # Create and connect
        logger.info("Connecting to device...")
        xense_flare = XenseFlare(config)
        xense_flare.connect()
        
        # Show current position before calibration
        try:
            pos_before = xense_flare.get_gripper_position()
            logger.info(f"Position before calibration: {pos_before:.4f} (normalized)")
        except Exception as e:
            logger.warn(f"Could not read position before calibration: {e}")
        
        # Perform calibration
        logger.info("")
        logger.info("Starting calibration...")
        logger.info("Please ensure the gripper is in the correct reference position.")
        input("Press Enter to continue with calibration...")
        
        xense_flare.calibrate_gripper()
        
        # Show position after calibration
        time.sleep(0.5)  # Wait for calibration to settle
        try:
            pos_after = xense_flare.get_gripper_position()
            logger.info(f"Position after calibration: {pos_after:.4f} (normalized)")
        except Exception as e:
            logger.warn(f"Could not read position after calibration: {e}")
        
        logger.info("")
        logger.info("✅ Calibration complete!")
        
        # Interactive step: measure max position
        logger.info("")
        logger.info("=" * 60)
        logger.info("Now let's measure the maximum gripper position.")
        logger.info("Please FULLY OPEN the gripper to its maximum position.")
        logger.info("=" * 60)
        input("Press Enter when the gripper is fully open...")
        
        # Read raw position at max open
        try:
            gripper_status = xense_flare._gripper.get_gripper_status()
            if gripper_status is not None:
                raw_max_pos = float(gripper_status.get("position", 0.0)) - xense_flare.config.gripper_max_readout
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"📏 Maximum gripper readout (raw): {raw_max_pos:.2f}")
                logger.info("=" * 60)
                logger.info("")
                logger.info("Use this value as 'gripper_max_readout' in your config:")
                logger.info(f"  gripper_max_readout={raw_max_pos:.1f}")
                logger.info("")
            else:
                logger.warn("Could not read gripper status")
        except Exception as e:
            logger.warn(f"Could not read max position: {e}")
        
        # Interactive mode
        if args.interactive:
            logger.info("")
            logger.info("Entering interactive mode. Press Ctrl+C to exit.")
            logger.info("Move the gripper to see real-time position updates.")
            logger.info("")
            
            try:
                while True:
                    try:
                        pos = xense_flare.get_gripper_position()
                        raw_pos = pos * args.gripper_max_pos
                        print(f"\rGripper Position: {pos:.4f} (normalized) | {raw_pos:.2f} mm (raw)    ", end="", flush=True)
                    except Exception as e:
                        print(f"\rError reading position: {e}    ", end="", flush=True)
                    time.sleep(0.05)  # 20 Hz update
            except KeyboardInterrupt:
                print()  # New line after Ctrl+C
                logger.info("Exiting interactive mode.")
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise
    finally:
        if xense_flare is not None:
            logger.info("Disconnecting...")
            xense_flare.disconnect()
            logger.info("Done.")


if __name__ == "__main__":
    main()

