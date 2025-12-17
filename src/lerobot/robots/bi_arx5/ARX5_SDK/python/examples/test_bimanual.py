import time
import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import click


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


@click.command()
@click.argument("model0")  # ARX arm model: X5 or L5
@click.argument("interface0")  # can bus name (can0 etc.)
@click.argument("model1")  # ARX arm model: X5 or L5
@click.argument("interface1")  # can bus name (can0 etc.)
def main(model0: str, interface0: str, model1: str, interface1: str):
    np.set_printoptions(precision=3, suppress=True)
    assert interface0 != interface1
    arx5_0 = arx5.Arx5JointController(model0, interface0)
    arx5_1 = arx5.Arx5JointController(model1, interface1)
    robot_config = arx5_0.get_robot_config()
    controller_config = arx5_0.get_controller_config()

    arx5_0.reset_to_home()
    arx5_1.reset_to_home()

    target_joint_poses = np.array([1.0, 2.0, 2.0, 1.5, 1.5, -1.57])
    step_num = 1500  # 3s

    try:
        for i in range(step_num):
            cmd = arx5.JointState(robot_config.joint_dof)
            # i = 0
            cmd.pos()[0:4] = easeInOutQuad(float(i) / step_num) * target_joint_poses[0:4]
            cmd.gripper_pos = (
                easeInOutQuad((i / (step_num - 1))) * robot_config.gripper_width
            )
            arx5_0.set_joint_cmd(cmd)
            arx5_1.set_joint_cmd(cmd)
            joint_state = arx5_0.get_joint_state()
            joint_state = arx5_1.get_joint_state()
            arm_dof_pos = joint_state.pos().copy()
            arm_dof_vel = joint_state.vel().copy()
            print(arm_dof_pos, arm_dof_vel)
            print(f"gripper: {joint_state.gripper_pos:.05f}")
            time.sleep(controller_config.controller_dt)

        arx5_0.set_to_damping()
        arx5_1.set_to_gravity_compensation()
        default_gain = arx5.Gain(robot_config.joint_dof)
        default_gain.kp()[:] = controller_config.default_kp
        default_gain.kd()[:] = controller_config.default_kd
        default_gain.gripper_kp = controller_config.default_gripper_kp
        default_gain.gripper_kd = controller_config.default_gripper_kd
        arx5_0.set_gain(default_gain)

        for i in range(step_num):
            joint_state = arx5_0.get_joint_state()
            joint_state = arx5_1.get_joint_state()
            arm_dof_pos = joint_state.pos().copy()
            arm_dof_vel = joint_state.vel().copy()
            print(arm_dof_pos, arm_dof_vel)
            print(f"gripper: {joint_state.gripper_pos:.05f}")
            time.sleep(controller_config.controller_dt)
        for i in range(step_num):
            cmd = arx5.JointState(robot_config.joint_dof)
            cmd.pos()[0:4] = (
                easeInOutQuad((1 - float(i) / step_num)) * target_joint_poses[0:4]
            )
            cmd.gripper_pos = (
                easeInOutQuad((1 - i / (step_num - 1))) * robot_config.gripper_width
            )
            arx5_0.set_joint_cmd(cmd)
            arx5_1.set_joint_cmd(cmd)
            time.sleep(controller_config.controller_dt)
    except KeyboardInterrupt as e:
        print(f"\nKeyboard interrupt detected. Resetting arms to home position... {e}")
        raise e
    finally:
        arx5_0.reset_to_home()
        arx5_1.reset_to_home()
        arx5_0.set_to_damping()
        arx5_1.set_to_damping()
        print("Safe to exit...")


if __name__ == "__main__":
    main()
