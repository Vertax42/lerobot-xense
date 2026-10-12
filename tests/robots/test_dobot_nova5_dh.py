"""Verify the vendored SDK and Nova5 adapter without connecting to hardware."""

import socket

import draccus
import numpy as np
import pytest
from TCP_IP_Python_V4.dobot_api import DobotApiDashboard, MyType, alarmAlarmJsonFile

from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.dobot_nova5_dh import ControlMode, DobotNova5DH, DobotNova5DHConfig, ResetTarget
from lerobot.robots.dobot_nova5_dh.dobot_nova5_dh import _DobotModbusRTU


@pytest.fixture(autouse=True)
def forbid_robot_connections(monkeypatch):
    def fail_connect(*args, **kwargs):
        pytest.fail("An offline test attempted to connect to hardware")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)


@pytest.mark.parametrize("mode,dimension", [(ControlMode.JOINT_MOTION, 7), (ControlMode.CARTESIAN_MOTION, 10)])
def test_robot_factory_and_action_contract(monkeypatch, tmp_path, mode, dimension):
    # Camera construction is independent of the arm/SDK contract under test.
    monkeypatch.setattr("lerobot.robots.dobot_nova5_dh.dobot_nova5_dh.make_cameras_from_configs", lambda configs: {})
    config = draccus.decode(
        RobotConfig,
        {"type": "dobot_nova5_dh", "control_mode": mode.value, "calibration_dir": str(tmp_path)},
    )
    robot = make_robot_from_config(config)
    assert isinstance(robot, DobotNova5DH)
    assert not robot.is_connected
    assert len(robot.action_features) == dimension
    assert robot.action_features == robot.observation_features
    assert "gripper.pos" in robot.action_features
    assert config.dh_gripper is not None
    assert config.dashboardPort == 29999
    assert config.feedPortFour == 30004
    # Keep the existing public enum string representation as well as its value.
    assert str(mode) == f"ControlMode.{mode.name}"
    assert str(ResetTarget.HOME) == "ResetTarget.HOME"


def test_alarm_resources_load_outside_sdk_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    controller, servo = alarmAlarmJsonFile()
    assert controller and servo
    assert all("id" in alarm for alarm in controller)
    assert all("id" in alarm for alarm in servo)


def test_modbus_adapter_uses_real_sdk_command_serialization(monkeypatch):
    # Bypass the SDK constructor's socket connection; exercise its real methods.
    dashboard = object.__new__(DobotApiDashboard)
    dashboard.socket_dobot = 0
    commands = []

    def send(command):
        commands.append(command)
        return "0,{2};" if command.startswith("ModbusCreate") else "0,{321};"

    monkeypatch.setattr(dashboard, "sendRecvMsg", send)
    adapter = _DobotModbusRTU(dashboard, "192.168.201.1", 60000, 1)
    assert adapter.read_register(0x0202) == 321
    assert adapter.write_register(0x0103, 1000)
    adapter.close()
    assert commands == [
        "ModbusCreate(192.168.201.1,60000,1,1)",
        "GetHoldRegs(2,514,1)",
        "SetHoldRegs(2,259,1,{1000})",
        "ModbusClose(2)",
    ]


def test_feedback_layout_matches_driver(monkeypatch, tmp_path):
    monkeypatch.setattr("lerobot.robots.dobot_nova5_dh.dobot_nova5_dh.make_cameras_from_configs", lambda configs: {})
    robot = DobotNova5DH(DobotNova5DHConfig(calibration_dir=tmp_path, use_gripper=False))
    data = np.zeros(1, dtype=MyType)
    data["TestValue"] = 0x123456789ABCDEF
    data["len"] = MyType.itemsize
    data["RobotMode"] = 5
    data["Tool"] = 1
    data["CurrentCommandId"] = 42
    data["ToolVectorActual"] = [[100, 200, 300, 0, 0, 90]]
    data["QActual"] = [[1, 2, 3, 4, 5, 6]]

    class Feedback:
        def feedBackData(self):  # noqa: N802 - vendor SDK method name
            robot._robot = None
            return data

    robot._feed = Feedback()
    robot._robot = object()
    robot._feed_loop()
    assert robot._feed_data.RobotMode == 5
    assert robot._feed_data.Tool == 1
    assert robot._feed_data.robotCurrentCommandID == 42
    assert robot._feed_data.MessageSize == MyType.itemsize
    np.testing.assert_array_equal(robot._feed_data.tcpPose, data["ToolVectorActual"][0])
    np.testing.assert_array_equal(robot._feed_data.qActual, [1, 2, 3, 4, 5, 6])
