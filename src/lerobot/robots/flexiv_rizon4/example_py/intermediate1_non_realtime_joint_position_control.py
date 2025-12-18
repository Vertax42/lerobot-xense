#!/usr/bin/env python

"""intermediate1_non_realtime_joint_position_control.py

This tutorial runs non-real-time joint position control to hold or sine-sweep all robot joints.
"""

__copyright__ = "Copyright (C) 2016-2025 Flexiv Ltd. All Rights Reserved." + "Copyright (C) 2025 The XenseRobotics Inc. team. All rights reserved."
__author__ = "Flexiv" + "The XenseRobotics Inc. team."

import time
import math
import argparse
import spdlog  # pip install spdlog
import flexivrdk  # pip install flexivrdk

def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion to Euler angles [roll, pitch, yaw] in radians."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return [roll, pitch, yaw]


def print_observation(robot, logger):
    """Print robot observation, similar to basics1_display_robot_states.py"""
    states = robot.states()
    fmt = lambda arr: [f"{x:.4f}" for x in arr]
    
    logger.info("Current robot observation:")
    # fmt: off
    print("{")
    print(f"  q [rad]:     {fmt(states.q)}")
    print(f"  dq [rad/s]:  {fmt(states.dq)}")
    print(f"  tau [Nm]:    {fmt(states.tau)}")
    print(f"  tau_ext [Nm]: {fmt(states.tau_ext)}")
    # TCP pose 6D
    pos = states.tcp_pose[0:3]
    euler = quat_to_euler(*states.tcp_pose[3:7])
    print(f"  tcp_pos [m]: {fmt(pos)}")
    print(f"  tcp_euler [rad]: {fmt(euler)}")
    print(f"  ft_sensor [N,Nm]: {fmt(states.ft_sensor_raw)}")
    print("}", flush=True)
    # fmt: on


def main():
    # Program Setup
    # ==============================================================================================
    # Parse arguments
    argparser = argparse.ArgumentParser()
    # Required arguments
    argparser.add_argument(
        "robot_sn",
        nargs="?",
        default="Rizon4-063423",
        help="Serial number of the robot to connect. Remove any space, e.g. Rizon4s-123456",
    )
    argparser.add_argument(
        "frequency", help="Command frequency, 1 to 100 [Hz]", type=int
    )
    # Optional arguments
    argparser.add_argument(
        "--hold",
        action="store_true",
        help="Robot holds current joint positions, otherwise do a sine-sweep",
    )
    args = argparser.parse_args()

    # Check if arguments are valid
    frequency = args.frequency
    assert frequency >= 1 and frequency <= 100, "Invalid <frequency> input"

    # Define alias
    logger = spdlog.ConsoleLogger("Example")
    mode = flexivrdk.Mode

    # Print description
    logger.info(
        ">>> Tutorial description <<<\nThis tutorial runs non-real-time joint position control to "
        "hold or sine-sweep all robot joints.\n"
    )

    # Initialize robot variable for access in exception handlers
    robot = None
    START_POSITION_DEG = [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]
    try:
        # RDK Initialization
        # ==========================================================================================
        # Instantiate robot interface
        robot = flexivrdk.Robot(args.robot_sn)

        # Clear fault on the connected robot if any
        if robot.fault():
            logger.warn("Fault occurred on the connected robot, trying to clear ...")
            # Try to clear the fault
            if not robot.ClearFault():
                logger.error("Fault cannot be cleared, exiting ...")
                return 1
            logger.info("Fault on the connected robot is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        logger.info("Enabling robot ...")
        robot.Enable()

        # Wait for the robot to become operational
        while not robot.operational():
            time.sleep(1)

        logger.info("Robot is now operational")

        # # Move robot to home pose
        # logger.info("Moving to home pose")
        # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        # robot.ExecutePlan("PLAN-Home")
        # # Wait for the plan to finish
        # while robot.busy():
        #     time.sleep(1)

        # Move robot to start pose using primitive execution
        logger.info("Moving to start pose")
        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        time.sleep(0.1)

        logger.info(f"Executing primitive MoveJ to move to start pose {START_POSITION_DEG}")
        start_jpos = flexivrdk.JPos(START_POSITION_DEG)

        robot.ExecutePrimitive(
            "MoveJ",
            {
                "target": start_jpos,
                "jntVelScale": 30,  # Joint velocity scale [1-100]
            },
        )
        # Wait for reached target
        # Note: primitive_states() returns a dictionary of {pt_state_name, [pt_state_values]}
        while not robot.primitive_states()["reachedTarget"]:
            logger.info("Waiting for start pose to be reached...")
            time.sleep(0.1)
        logger.info("Start pose reached!")
        # Non-real-time Joint Position Control
        # ==========================================================================================
        # Switch to non-real-time joint position control mode
        robot.SwitchMode(mode.NRT_JOINT_POSITION)
        time.sleep(0.1)
        period = 1.0 / frequency
        loop_time = 0
        logger.info(
            f"Sending command to robot at {frequency} Hz, or {period} seconds interval"
        )

        # Use current robot joint positions as initial positions
        init_pos = robot.states().q.copy()
        logger.info(f"Initial positions set to: {init_pos}")

        # Robot joint degrees of freedom
        DoF = robot.info().DoF

        # Initialize target vectors
        target_pos = init_pos.copy()
        target_vel = [0.0] * DoF
        target_acc = [0.0] * DoF

        # Joint motion constraints
        MAX_VEL = [2.0] * DoF
        MAX_ACC = [1.0] * DoF

        # Joint sine-sweep amplitude [rad]
        SWING_AMP = 0.15

        # Joint sine-sweep frequency [Hz]
        SWING_FREQ = 0.3

        # Send command periodically at user-specified frequency
        while True:
            # Use sleep to control loop period
            time.sleep(period)

            # Monitor fault on the connected robot
            if robot.fault():
                raise Exception("Fault occurred on the connected robot, exiting ...")

            # Sine-sweep all joints
            if not args.hold:
                for i in range(DoF):
                    target_pos[i] = init_pos[i] + SWING_AMP * math.sin(
                        2 * math.pi * SWING_FREQ * loop_time
                    )
            # Otherwise all joints will hold at initial positions

            # Send command
            robot.SendJointPosition(target_pos, target_vel, MAX_VEL, MAX_ACC)

            # Print observation
            print_observation(robot, logger)

            # Increment loop time
            loop_time += period

    except KeyboardInterrupt:
        # Handle Ctrl+C: safely move robot back to home
        logger.info("Ctrl+C detected, safely moving robot to home position...")
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
                # Wait for MoveJ to finish
                while not robot.primitive_states()["reachedTarget"]:
                    time.sleep(0.1)
                logger.info("Robot safely returned to home position")
            except Exception as home_err:
                logger.error(f"Failed to return home: {home_err}")
        logger.info("Exiting program")

    except Exception as e:
        # Print exception error message
        logger.error(str(e))

    finally:
        # Cleanup: always executed regardless of how the program exits
        if robot is not None:
            logger.info("Program finished, disconnecting from robot...")
            # Robot object will be garbage collected and connection closed automatically
        logger.info("Program terminated")


if __name__ == "__main__":
    main()
