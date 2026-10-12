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

"""Configuration for DH Robotics AG-95 gripper via the Dobot arm's built-in RS485 port."""

import warnings
from dataclasses import dataclass


@dataclass
class DHGripperIntegratedConfig:
    """Configuration for DHGripperIntegrated.

    Communication is routed through the robot arm controller's built-in RS485
    end-effector port using the Dobot SDK Modbus proxy API
    (ModbusCreate with isRTU=True / GetHoldRegs / SetHoldRegs / ModbusClose).
    No USB-serial adapter or pyserial dependency is required.

    Position convention (DH hardware register):
        0    = fully closed
        1000 = fully open

    Normalized position (lerobot convention):
        0.0  = fully closed
        1.0  = fully open

    Attributes:
        slave_id:      Modbus slave ID of the DH gripper (1–247, default 1).
        baudrate:      RS485 baud rate. Must match the gripper hardware setting
                       (default 115200; also 9600 / 19200 / 38400).
        gripper_speed: Target movement speed. This parameter does not take effect.
        gripper_force: Target gripping force, 20–100 % (default 30; max 160 N).
        init_open:     If True, move gripper to fully open after hardware init.
        init_timeout:  Seconds to wait for hardware self-initialisation
                       (register 0x0100 → 0xA5 sequence, default 10.0 s).
        worker_frequency: Best-effort background Modbus worker loop frequency in Hz.
                          Must be at least 10 Hz. Defaults to 100 Hz to match the
                          Nova5 NRT control-rate ceiling.
        position_poll_frequency: Background current-position read frequency in Hz.
                                 Set 0 to disable hardware position polling and
                                 use the latest commanded position as cached state.
        command_epsilon: Minimum normalized target-position change before the
                         worker sends another position register write.
    """

    slave_id: int = 1
    baudrate: int = 115200
    gripper_speed: int = 0  # does not take effect
    gripper_force: int = 30
    init_open: bool = True
    init_timeout: float = 10.0
    worker_frequency: float = 100.0
    position_poll_frequency: float = 20.0
    command_epsilon: float = 0.0

    def __post_init__(self):
        if not 1 <= self.slave_id <= 247:
            raise ValueError(
                f"DHGripperIntegratedConfig: slave_id {self.slave_id} "
                "out of Modbus range [1, 247]."
            )
        if self.baudrate not in (9600, 19200, 38400, 115200):
            raise ValueError(
                f"DHGripperIntegratedConfig: unsupported baudrate {self.baudrate}. "
                "Choose from 9600, 19200, 38400, 115200."
            )
        if self.gripper_speed != 0:
            warnings.warn(
                "DHGripperIntegratedConfig: gripper_speed does not take effect and will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if not 20 <= self.gripper_force <= 100:
            raise ValueError(
                f"DHGripperIntegratedConfig: gripper_force {self.gripper_force} "
                "out of range [20, 100]."
            )
        if self.init_timeout <= 0:
            raise ValueError(
                "DHGripperIntegratedConfig: init_timeout must be positive, "
                f"got {self.init_timeout}."
            )
        if self.worker_frequency < 10.0:
            raise ValueError(
                "DHGripperIntegratedConfig: worker_frequency must be at least 10 Hz, "
                f"got {self.worker_frequency}."
            )
        if not 0.0 <= self.position_poll_frequency <= self.worker_frequency:
            raise ValueError(
                "DHGripperIntegratedConfig: position_poll_frequency must be in "
                f"[0, worker_frequency], got {self.position_poll_frequency}."
            )
        if not 0.0 <= self.command_epsilon <= 1.0:
            raise ValueError(
                "DHGripperIntegratedConfig: command_epsilon must be in [0, 1], "
                f"got {self.command_epsilon}."
            )
