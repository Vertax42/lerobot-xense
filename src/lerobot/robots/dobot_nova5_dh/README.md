# dobot_nova5_dh

单臂 Dobot Nova5 模块。它从 `bi_dobot_nova5_dh` 提取右臂硬件配置，并通过机械臂末端内置
RS485 接口控制大寰 AG-95 夹爪，不需要 USB-RS485 转换器。

## 与双臂版本的差异

- 只连接原双臂系统的右臂，默认 IP 为 `192.168.111.102`。
- 只保留右臂的 home/start、工作空间、腕部相机及触觉传感器配置。
- 单臂公共字段不带 `right_` 前缀，例如 `robot_ip`、`workspace_min_xyz_m`。
- 动作和观测键使用单臂格式：
  - 关节控制：`joint_1.pos` ... `joint_6.pos`、`gripper.pos`。
  - 笛卡尔控制：`tcp.x/y/z/r1-r6`、`gripper.pos`。
- 删除双臂同步移动、移动顺序和并行发送逻辑。

## 使用

```python
from lerobot.robots.dobot_nova5_dh import (
    ControlMode,
    DobotNova5DH,
    DobotNova5DHConfig,
)

config = DobotNova5DHConfig(
    robot_ip="192.168.111.102",
    control_mode=ControlMode.CARTESIAN_MOTION,
    use_gripper=True,
    tool_identify=1,
    dh_gripper_baudrate=115200,
)
robot = DobotNova5DH(config)
robot.connect()
```

命令行使用的机器人类型为：

```bash
--robot.type=dobot_nova5_dh
```

默认启用 DH 夹爪。夹爪通信路径为：

```text
DobotApiDashboard
  -> SetToolMode / SetTool485
  -> ModbusCreate(isRTU=True)
  -> 机械臂末端 RS485
  -> DH AG-95
```

`dedicated_gripper_dashboard=True` 时，夹爪优先使用独立 Dashboard TCP 连接；连接失败会自动退回
机械臂控制所使用的 Dashboard 连接。
