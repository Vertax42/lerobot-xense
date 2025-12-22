import time
import os
import sys
import threading

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Import after path setup - this is required for the SDK
import arx5_interface as arx5  # noqa: E402


# 移除了 easeInOutQuad 函数，因为不再需要运动控制


def print_state_continuously(controller, stop_event):
    """持续打印机器人状态的函数（包含夹爪数据）"""
    while not stop_event.is_set():
        try:
            joint_state = controller.get_joint_state()
            pos = joint_state.pos()
            torque = joint_state.torque()
            gripper_pos = joint_state.gripper_pos
            gripper_vel = joint_state.gripper_vel
            gripper_torque = joint_state.gripper_torque

            # 检查关节位置是否超出范围
            robot_config = controller.get_robot_config()
            pos_warnings = []
            for i in range(len(pos)):
                min_pos = robot_config.joint_pos_min[i] - 3.14
                max_pos = robot_config.joint_pos_max[i] + 3.14
                if pos[i] < min_pos or pos[i] > max_pos:
                    pos_warnings.append(f"J{i}:{pos[i]:.3f}")

            # 检查夹爪位置
            gripper_warning = ""
            min_gripper = -0.01
            max_gripper = robot_config.gripper_width + 0.01
            if gripper_pos < min_gripper or gripper_pos > max_gripper:
                gripper_warning = f" [夹爪位置异常: {gripper_pos:.3f}]"

            print(
                f"\r夹爪位置: {gripper_pos:.3f} | 夹爪速度: {gripper_vel:.3f} | "
                f"夹爪力矩: {gripper_torque:.3f} | 关节力矩: {torque} "
                f"{pos_warnings} {gripper_warning}",
                end="",
                flush=True,
            )
            time.sleep(0.1)  # 每100ms打印一次
        except Exception as e:
            print(f"\n状态打印错误: {e}")
            break


@click.command()
@click.argument("model")  # ARX arm model: X5 or L5
@click.argument("interface")  # can bus name (can0 etc.)
def main(model: str, interface: str):
    """
    重力补偿状态监控脚本

    功能：
    - 初始化后先重置机器人到home位置
    - 设置关节gain为0，实现纯重力补偿（无位置/速度控制）
    - 保持夹爪正常控制，避免夹爪失控
    - 进入重力补偿模式，机器人保持在当前位置
    - 实时监控并打印关节位置和速度
    - 支持 Ctrl+C 中断监控

    注意：
    - 关节控制增益(kp, kd)设为0，只进行重力补偿
    - 夹爪保持正常控制增益，确保夹爪稳定
    """

    # To initialize robot with different configurations,
    # you can create RobotConfig and ControllerConfig by yourself and modify
    # based on it
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    # 保持夹爪正常工作，不修改gripper_motor_type
    # robot_config.gripper_open_readout = -3.45  # -3.26
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    # Modify the default configuration here
    # controller_config.controller_dt = 0.01 # etc.

    USE_MULTITHREADING = True
    if USE_MULTITHREADING:
        # Will create another thread that communicates with the arm, so each
        # send_recv_once() will take no time for the main thread to execute.
        # Otherwise (without background send/recv), send_recv_once() will block
        # the main thread until the arm responds (usually 2ms).
        controller_config.background_send_recv = True
    else:
        controller_config.background_send_recv = False

    # 夹爪保持正常工作状态
    # 设置夹爪控制参数

    # 设置关节gain为0实现纯重力补偿，但保持夹爪正常控制
    print("设置关节gain为0实现纯重力补偿，保持夹爪正常控制...")
    gain = arx5.Gain(robot_config.joint_dof)
    gain.kp()[:] = 0.0  # 关节位置增益设为0
    gain.kd()[:] = 0.0  # 关节速度增益设为0
    # 保持夹爪的默认增益，确保夹爪稳定
    gain.gripper_kp = 0.0
    gain.gripper_kd = controller_config.default_gripper_kd

    # 打印配置信息
    print(f"机器人模型: {model}")
    print(f"CAN接口: {interface}")
    print(f"夹爪电机类型: {robot_config.gripper_motor_type}")
    print(f"夹爪状态: 正常工作 ({robot_config.gripper_motor_type})")
    print(f"关节自由度: {robot_config.joint_dof}")
    print("=" * 50)

    arx5_joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )

    # Or you can directly use the model and interface name
    # arx5_joint_controller = arx5.Arx5JointController(model, interface)

    np.set_printoptions(precision=3, suppress=True)
    arx5_joint_controller.set_log_level(arx5.LogLevel.DEBUG)
    robot_config = arx5_joint_controller.get_robot_config()
    controller_config = arx5_joint_controller.get_controller_config()

    # 应用gain设置
    arx5_joint_controller.set_gain(gain)
    print("✅ Gain设置完成:")
    print(f"  关节 kp={gain.kp()}, kd={gain.kd()}")
    print(f"  夹爪 kp={gain.gripper_kp}, kd={gain.gripper_kd}")

    # 移除了 step_num 变量，因为不再需要运动控制

    # 启动状态打印线程
    stop_printing = threading.Event()
    state_thread = threading.Thread(
        target=print_state_continuously, args=(arx5_joint_controller, stop_printing)
    )
    state_thread.daemon = True
    state_thread.start()

    print("开始重力补偿状态监控...")
    print("机器人将保持在当前位置，只进行状态监控")
    print("按 Ctrl+C 停止监控")
    print("=" * 50)

    # 先重置到home位置
    print("重置机器人到home位置...")
    # arx5_joint_controller.reset_to_home()
    print("✅ 机器人已重置到home位置")

    # 进入重力补偿模式，不发送运动指令
    print("进入重力补偿模式...")
    # arx5_joint_controller.set_to_damping()
    try:
        # 持续监控状态，不发送控制指令
        print("开始持续状态监控...")
        while True:
            # 只进行通信，不发送控制指令
            if not USE_MULTITHREADING:
                arx5_joint_controller.send_recv_once()
            else:
                time.sleep(controller_config.controller_dt)

    except KeyboardInterrupt:
        print("\n\nUser interrupt, resetting to home")
        # print(f"Teleop recording is terminated. Resetting to home.")
        arx5_joint_controller.reset_to_home()
        arx5_joint_controller.set_to_damping()
    except Exception as e:
        print(f"\n\n监控过程中出现错误: {e}")
    finally:
        # 停止状态打印线程
        stop_printing.set()
        print("\n状态打印已停止")
        print("重力补偿监控结束")


main()
