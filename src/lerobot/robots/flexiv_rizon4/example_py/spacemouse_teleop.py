#!/usr/bin/env python

"""spacemouse_teleop.py

Spacemouse teleoperation for Flexiv Rizon4 robot using Cartesian motion control.

Controls:
- Spacemouse 6DOF: Control TCP position and orientation
- Both buttons (pressed together): Reset to home position
- Left/Right buttons: Gripper control (only when --gripper flag is enabled)

Note: Gripper control is DISABLED by default. Use --gripper flag to enable.

Dependencies:
    pip install spnav numpy spdlog flexivrdk
    Also requires shared_memory module from lerobot or similar project.
"""

__copyright__ = "Copyright (C) 2016-2025 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import math
import argparse
import spdlog
import numpy as np
import flexivrdk
from lerobot.teleoperators.spacemouse.peripherals import Spacemouse
from multiprocessing.managers import SharedMemoryManager
from queue import Queue

# ==================================================================================================
# Global Constants
# ==================================================================================================

# Control frequency [Hz]
DEFAULT_FREQUENCY = 100

# Spacemouse sensitivity
POS_SPEED = 0.3        # Position speed scale [m/s at max input]
ORI_SPEED = 0.8         # Orientation speed scale [rad/s at max input]
GRIPPER_SPEED = 0.02    # Gripper speed [m/s]

# Spacemouse deadzone threshold [0-1]
DEADZONE_THRESHOLD = 0.1

# Spacemouse axis inversion (x, y, z, roll, pitch, yaw)
INVERT_AXES = (
    True,   # x - reverse
    True,   # y - reverse
    False,  # z
    True,   # roll - reverse
    True,   # pitch - reverse
    False,  # yaw
)

# Moving average window size for spacemouse input smoothing
WINDOW_SIZE = 3

# External force/torque thresholds for collision detection [N] / [Nm]
EXT_FORCE_THRESHOLD = 15.0
EXT_TORQUE_THRESHOLD = 8.0

# Default start position in degrees
START_POSITION_DEG = [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]


# ==================================================================================================
# Utility Functions  
# ==================================================================================================

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion [qw, qx, qy, qz]."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qw, qx, qy, qz])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions q1 * q2. Both in [qw, qx, qy, qz] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_to_euler(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to Euler angles [roll, pitch, yaw] in radians."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.array([roll, pitch, yaw])


class SpacemouseTeleop:
    """Spacemouse teleoperation controller for Flexiv robot."""
    
    def __init__(self, robot: flexivrdk.Robot, logger: spdlog.ConsoleLogger, 
                 frequency: int = DEFAULT_FREQUENCY, enable_collision: bool = False,
                 enable_gripper: bool = False):
        self.robot = robot
        self.logger = logger
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.enable_collision = enable_collision
        self.enable_gripper = enable_gripper
        
        # Gripper interface (optional)
        self.gripper = None
        if enable_gripper:
            try:
                self.gripper = flexivrdk.Gripper(robot)
                self.gripper_pos = 0.0
                self.gripper_max = 0.08  # Default max width, will be updated
                self.logger.info("Gripper initialized")
            except Exception as e:
                self.logger.warn(f"Failed to initialize gripper: {e}")
                self.gripper = None
        
        # Spacemouse state
        self.spacemouse_queue = Queue(WINDOW_SIZE)
        
        # Target pose [x, y, z, qw, qx, qy, qz]
        self.target_pose = None
        self.init_pose = None
        
    def get_filtered_spacemouse_output(self, sm: Spacemouse) -> np.ndarray:
        """Get filtered spacemouse output with moving average."""
        raw_state = sm.get_motion_state_transformed()
        
        # Apply deadzone filtering
        positive_idx = raw_state >= DEADZONE_THRESHOLD
        negative_idx = raw_state <= -DEADZONE_THRESHOLD
        filtered_state = np.zeros_like(raw_state)
        filtered_state[positive_idx] = (raw_state[positive_idx] - DEADZONE_THRESHOLD) / (1 - DEADZONE_THRESHOLD)
        filtered_state[negative_idx] = (raw_state[negative_idx] + DEADZONE_THRESHOLD) / (1 - DEADZONE_THRESHOLD)
        
        # Apply axis inversion (after deadzone)
        invert = np.array(INVERT_AXES, dtype=np.float32)
        invert = np.where(invert, -1.0, 1.0)
        filtered_state = filtered_state * invert
        
        # Moving average filter (use public methods to avoid race condition)
        if self.spacemouse_queue.full():
            self.spacemouse_queue.get()
        self.spacemouse_queue.put(filtered_state)
        
        return np.mean(np.array(list(self.spacemouse_queue.queue)), axis=0)
    
    def update_target_pose(self, spacemouse_state: np.ndarray, dt: float):
        """Update target pose based on spacemouse input.
        
        Args:
            spacemouse_state: [x, y, z, roll, pitch, yaw] from spacemouse, normalized [-1, 1]
            dt: Time delta in seconds
        """
        # Update position (incremental)
        delta_pos = spacemouse_state[:3] * POS_SPEED * dt
        self.target_pose[:3] += delta_pos # [x, y, z, qw, qx, qy, qz] in meters and quaternion
        
        # Update orientation (incremental via quaternion multiplication)
        delta_euler = spacemouse_state[3:] * ORI_SPEED * dt
        delta_quat = euler_to_quaternion(delta_euler[0], delta_euler[1], delta_euler[2])
        
        current_quat = self.target_pose[3:7]
        new_quat = quaternion_multiply(delta_quat, current_quat)
        self.target_pose[3:7] = normalize_quaternion(new_quat)
    
    def update_gripper(self, button_left: bool, button_right: bool, dt: float):
        """Update gripper position based on button state."""
        if self.gripper is None:
            return
            
        if button_left and not button_right:
            # Close gripper
            self.gripper_pos = max(0.0, self.gripper_pos - GRIPPER_SPEED * dt)
        elif button_right and not button_left:
            # Open gripper
            self.gripper_pos = min(self.gripper_max, self.gripper_pos + GRIPPER_SPEED * dt)
        
        # Send gripper command (non-blocking move)
        try:
            self.gripper.Move(self.gripper_pos, 0.1, 20)  # width, velocity, force
        except Exception:
            pass  # Ignore gripper errors during teleop
    
    def check_collision(self) -> bool:
        """Check for collision using external force/torque."""
        if not self.enable_collision:
            return False
            
        states = self.robot.states()
        
        # Check external wrench
        ext_force = np.array(states.ext_wrench_in_world[:3])
        if np.linalg.norm(ext_force) > EXT_FORCE_THRESHOLD:
            return True
        
        # Check external joint torques
        for v in states.tau_ext:
            if abs(v) > EXT_TORQUE_THRESHOLD:
                return True
        
        return False
    
    def reset_to_home(self):
        """Reset robot to home position."""
        mode = flexivrdk.Mode
        
        self.logger.info("Resetting to home position...")
        self.robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        
        start_jpos = flexivrdk.JPos(START_POSITION_DEG)
        self.robot.ExecutePrimitive("MoveJ", {
            "target": start_jpos,
            "jntVelScale": 30,  # Joint velocity scale [1-100]
        })
        
        while not self.robot.primitive_states()["reachedTarget"]:
            time.sleep(0.1)
            
        self.logger.info("Home position reached")
        
        # Re-enter Cartesian control mode
        self.robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SetForceControlAxis([False] * 6)
        
        # Update target pose to current (convert to numpy array)
        self.target_pose = np.array(self.robot.states().tcp_pose)
        self.init_pose = self.target_pose.copy()
    
    def print_status(self, states, loop_counter: int):
        """Print current robot status."""
        if loop_counter % self.frequency == 0:  # Print every second
            pos = self.target_pose[:3]
            euler = quaternion_to_euler(*self.target_pose[3:7])
            
            print(f"\r[Teleop] Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m | "
                  f"Ori: [{np.rad2deg(euler[0]):.1f}, {np.rad2deg(euler[1]):.1f}, {np.rad2deg(euler[2]):.1f}] deg", 
                  end="", flush=True)
    
    def run(self):
        """Main teleoperation loop."""
        mode = flexivrdk.Mode
        
        # Switch to Cartesian motion force mode
        self.robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        self.robot.SetForceControlAxis([False] * 6)  # Pure motion control
        
        # Initialize target pose (convert to numpy array for easier manipulation)
        self.target_pose = np.array(self.robot.states().tcp_pose)
        self.init_pose = self.target_pose.copy()
        
        self.logger.info(f"Starting teleop at {self.frequency} Hz")
        self.logger.info("Controls: Spacemouse 6DOF for TCP control")
        self.logger.info("          Both buttons: Reset to home")
        if self.enable_gripper and self.gripper is not None:
            self.logger.info("          Left button: Close gripper | Right button: Open gripper")
        
        with SharedMemoryManager() as shm_manager:
            with Spacemouse(shm_manager=shm_manager, deadzone=DEADZONE_THRESHOLD, max_value=500) as sm:
                
                self.logger.info("Spacemouse ready. Waiting for input to start tracking...")
                
                # Wait for first spacemouse input
                while True:
                    button_left = sm.is_button_pressed(0)
                    button_right = sm.is_button_pressed(1)
                    state = self.get_filtered_spacemouse_output(sm)
                    
                    if state.any() or button_left or button_right:
                        self.logger.info("Tracking started!")
                        break
                    time.sleep(0.01)
                
                # Main control loop
                start_time = time.monotonic()
                loop_counter = 0
                
                while True:
                    loop_start = time.monotonic()
                    
                    # Get spacemouse state
                    state = self.get_filtered_spacemouse_output(sm)
                    button_left = sm.is_button_pressed(0)
                    button_right = sm.is_button_pressed(1)
                    
                    # Both buttons: Reset to home
                    if button_left and button_right:
                        print()  # New line after status
                        self.reset_to_home()
                        start_time = time.monotonic()
                        loop_counter = 0
                        continue
                    
                    # Update target pose from spacemouse
                    self.update_target_pose(state, self.period)
                    
                    # Update gripper
                    self.update_gripper(button_left, button_right, self.period)
                    
                    # Check for fault
                    if self.robot.fault():
                        raise Exception("Fault occurred on robot")
                    
                    # Check collision
                    if self.check_collision():
                        self.robot.Stop()
                        self.logger.warn("Collision detected! Stopping robot.")
                        break
                    
                    # Send Cartesian command
                    self.robot.SendCartesianMotionForce(self.target_pose)
                    
                    # Print status
                    self.print_status(self.robot.states(), loop_counter)
                    
                    # Control loop timing
                    loop_counter += 1
                    elapsed = time.monotonic() - loop_start
                    if elapsed < self.period:
                        time.sleep(self.period - elapsed)


def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(description="Spacemouse teleoperation for Flexiv robot")
    argparser.add_argument(
        "robot_sn",
        nargs="?",
        default="Rizon4-063423",
        help="Serial number of the robot (default: Rizon4-063423)",
    )
    argparser.add_argument(
        "--frequency", "-f",
        type=int,
        default=DEFAULT_FREQUENCY,
        help=f"Control frequency in Hz, 1-100 (default: {DEFAULT_FREQUENCY})",
    )
    argparser.add_argument(
        "--collision",
        action="store_true",
        help="Enable collision detection",
    )
    argparser.add_argument(
        "--gripper",
        action="store_true",
        help="Enable gripper control (disabled by default)",
    )
    args = argparser.parse_args()
    
    # Validate frequency
    assert 1 <= args.frequency <= 100, "Frequency must be between 1 and 100 Hz"
    
    # Setup logger
    logger = spdlog.ConsoleLogger("SpacemouseTeleop")
    mode = flexivrdk.Mode
    robot = None
    
    try:
        # Initialize robot
        logger.info(f"Connecting to robot {args.robot_sn}...")
        robot = flexivrdk.Robot(args.robot_sn)
        
        # Clear fault if any
        if robot.fault():
            logger.warn("Fault occurred on robot, trying to clear...")
            if not robot.ClearFault():
                logger.error("Fault cannot be cleared, exiting...")
                return 1
            logger.info("Fault cleared")
        
        # Enable robot
        logger.info("Enabling robot...")
        robot.Enable()
        
        while not robot.operational():
            time.sleep(1)
        logger.info("Robot is operational")
        
        # # Move to home position
        # logger.info("Moving to home position...")
        # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        # robot.ExecutePlan("PLAN-Home")
        # while robot.busy():
        #     time.sleep(0.5)
        
        # Move to start position
        logger.info("Moving to start position...")
        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        start_jpos = flexivrdk.JPos(START_POSITION_DEG)
        robot.ExecutePrimitive("MoveJ", {
            "target": start_jpos,
            "jntVelScale": 30,  # Joint velocity scale [1-100]
        })
        while not robot.primitive_states()["reachedTarget"]:
            time.sleep(0.5)
        logger.info("Start position reached")
        
        # Zero force/torque sensors
        logger.info("Zeroing force/torque sensors (ensure robot is not in contact with anything)...")
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not robot.primitive_states()["terminated"]:
            time.sleep(0.5)
        logger.info("Sensor zeroing complete")
        
        # Create and run teleop controller
        teleop = SpacemouseTeleop(
            robot=robot,
            logger=logger,
            frequency=args.frequency,
            enable_collision=args.collision,
            enable_gripper=args.gripper,
        )
        teleop.run()
        
    except KeyboardInterrupt:
        # Handle Ctrl+C: safely move robot back to home
        print()  # New line
        logger.info("Ctrl+C detected, safely moving robot to home position...")
        logger.warn("Please wait for robot to return home. Press Ctrl+C again to force quit (not recommended).")
        if robot is not None and robot.operational():
            try:
                # Switch to primitive execution mode and use MoveJ to go home
                robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
                home_jpos = flexivrdk.JPos(START_POSITION_DEG)
                robot.ExecutePrimitive(
                    "MoveJ",
                    {
                        "target": home_jpos,
                        "jntVelScale": 20,  # Joint velocity scale [1-100]
                    },
                )
                # Wait for MoveJ to finish (with Ctrl+C protection)
                try:
                    while not robot.primitive_states()["reachedTarget"]:
                        time.sleep(0.1)
                    logger.info("Robot safely returned to home position")
                except KeyboardInterrupt:
                    logger.warn("Force quit requested. Robot may not be at home position!")
                    robot.Stop()
            except Exception as home_err:
                logger.error(f"Failed to return home: {home_err}")
        logger.info("Exiting program")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    finally:
        # Cleanup: always executed regardless of how the program exits
        if robot is not None:
            logger.info("Program finished, disconnecting from robot...")
            # Robot object will be garbage collected and connection closed automatically
        logger.info("Program terminated")
    
    return 0


if __name__ == "__main__":
    exit(main())
